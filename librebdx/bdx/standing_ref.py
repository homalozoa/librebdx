"""Perpetual (standing) reference generator for the BDX-R character.

Implements f_perp(path_frame, g_perp) of Grandia et al. (Eq. 2, 5): given head
commands (height, yaw, pitch, roll offsets w.r.t. a nominal head pose) and torso
commands (height offset, yaw, pitch, roll), solve for a whole-body pose with the
feet at the nominal stance and the centre of mass above the middle of the
support polygon. Everything is expressed in path-frame coordinates (path frame
origin = middle of the feet on the ground, x forward).

The solve is a batched least-squares IK in torch so that thousands of RL
environments can query references every control step (warm started).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from librebdx.kin.ik import BatchedLSQ
from librebdx.kin.torch_kin import TorchKinematics, euler_zyx_to_mat, so3_log

XML = Path(__file__).parents[1] / "robots" / "bdxr" / "bdxr.xml"

LEG_L = ["Left_Hip_Yaw", "Left_Hip_Roll", "Left_Hip_Pitch", "Left_Knee", "Left_Ankle"]
LEG_R = ["Right_Hip_Yaw", "Right_Hip_Roll", "Right_Hip_Pitch", "Right_Knee", "Right_Ankle"]
NECK = ["Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll"]

# Command g_perp = (dh_head, dyaw_head, dpitch_head, droll_head,
#                   dh_torso, yaw_torso, pitch_torso, roll_torso)
CMD_NAMES = ("dh_head", "yaw_head", "pitch_head", "roll_head", "dh_torso", "yaw_torso", "pitch_torso", "roll_torso")
# Ranges from scripts/sweep_bdx_standing_ranges.py. The BDX-R neck has a single
# pitch link, so the head height range is small (the paper's robot has an
# extra neck-forward joint); the up/down command drives Neck_Pitch.
CMD_LO = (-0.03, -0.80, -0.20, -0.45, -0.05, -0.22, -0.30, -0.15)
CMD_HI = (0.01, 0.80, 0.50, 0.45, 0.03, 0.22, 0.15, 0.15)
NECK_PER_M = 10.0


@dataclass
class Nominal:
  torso_height: float = 0.225
  stance_half_width: float = 0.108
  stance_x: float = 0.0
  neck: tuple = (0.55, -0.55, 0.0, 0.0)


class BdxStandingReference:
  def __init__(self, device="cpu", nominal: Nominal = Nominal()):
    self.kin = k = TorchKinematics(XML, device=device)
    self.device = device
    self.nom = nominal
    self.leg_l, self.leg_r, self.neck = k.jidx(LEG_L), k.jidx(LEG_R), k.jidx(NECK)
    self.nq = k.nq
    lo = torch.cat([torch.full((2,), -0.3, device=device), k.joint_range[:, 0]])
    hi = torch.cat([torch.full((2,), 0.3, device=device), k.joint_range[:, 1]])
    self.cmd_lo = torch.tensor(CMD_LO, device=device)
    self.cmd_hi = torch.tensor(CMD_HI, device=device)
    self._calibrate_nominal()
    self.lsq = BatchedLSQ(self._residual, lo, hi, damping=1e-5)

  # ------------------------------------------------------------------ setup
  def _calibrate_nominal(self):
    """Nominal pose: feet flat at stance, torso level at nominal height, neck as given.

    The sole forward direction in the foot frame and the nominal head pose
    are read off this configuration.
    """
    k, dev = self.kin, self.device
    q = torch.zeros(1, k.nq, device=dev)
    q[0, self.neck] = torch.tensor(self.nom.neck, device=dev)
    I = torch.eye(3, device=dev)[None]
    P, R = k.forward(torch.zeros(1, 3, device=dev), I, q)
    # Foot "forward" axis in the sole frame: world x expressed in the sole frame
    # at q=0 (the soles are flat and point forward at the zero pose).
    self.fwd_local = {}
    for s in ("left_sole", "right_sole"):
      _, Rs = k.site(P, R, s)
      self.fwd_local[s] = (Rs[0].T @ torch.tensor([1.0, 0, 0], device=dev))
    self.feet_target = {
      "left_sole": torch.tensor([self.nom.stance_x, self.nom.stance_half_width, 0.0], device=dev),
      "right_sole": torch.tensor([self.nom.stance_x, -self.nom.stance_half_width, 0.0], device=dev),
    }
    # Solve the nominal legs with level torso and nominal neck (head free).
    self._fix_head = False
    x0 = torch.zeros(1, 2 + k.nq, device=dev)
    x0[0, 2:][self.neck] = q[0, self.neck]
    g0 = torch.zeros(1, 8, device=dev)
    lo = torch.cat([torch.full((2,), -0.3, device=dev), k.joint_range[:, 0]])
    hi = torch.cat([torch.full((2,), 0.3, device=dev), k.joint_range[:, 1]])
    lsq = BatchedLSQ(self._residual, lo, hi, damping=1e-5)
    x, r = lsq.solve(x0, g0, iters=30)
    self.x_nominal = x[0].clone()
    root_pos, root_R, qq = self._unpack(x, g0)
    P, R = k.forward(root_pos, root_R, qq)
    hp, hR = k.site(P, R, "head")
    self.head_height0 = float(hp[0, 2])
    self.head_R0 = hR[0].clone()
    self.head_xy0 = hp[0, :2].clone()
    self._fix_head = True
    self.nominal_residual = float(r.abs().max())

  # --------------------------------------------------------------- residual
  def _unpack(self, x, g):
    root_xy = x[:, :2]
    q = x[:, 2:]
    z = self.nom.torso_height + g[:, 4]
    root_pos = torch.cat([root_xy, z[:, None]], -1)
    root_R = euler_zyx_to_mat(g[:, 5], g[:, 6], g[:, 7])
    return root_pos, root_R, q

  def _head_target(self, g):
    Rc = euler_zyx_to_mat(g[:, 1], g[:, 2], g[:, 3])
    return self.nom_head_height(g), Rc @ self.head_R0

  def nom_head_height(self, g):
    # Head height follows the torso height command (gaze height is relative to
    # the crouch), plus the head's own up/down offset.
    return self.head_height0 + g[:, 0] + g[:, 4]

  def _residual(self, x, g):
    k = self.kin
    root_pos, root_R, q = self._unpack(x, g)
    P, R = k.forward(root_pos, root_R, q)
    res = []
    for s in ("left_sole", "right_sole"):
      p, Rs = k.site(P, R, s)
      res.append(p - self.feet_target[s])
      fwd = (Rs @ self.fwd_local[s].unsqueeze(-1)).squeeze(-1)
      # Forward axis horizontal (sole flat in pitch) and pointing along +x.
      res.append(torch.stack([fwd[:, 1], fwd[:, 2]], -1))
    c = k.com(P, R)
    res.append(5.0 * (c[:, :2] - 0.5 * (self.feet_target["left_sole"][:2] + self.feet_target["right_sole"][:2])))
    if self._fix_head:
      # Head height has a fold w.r.t. the single neck-pitch link, so the
      # up/down command drives Neck_Pitch monotonically (NECK_PER_M rad per
      # metre of head drop) and the three head joints solve the gaze exactly.
      _, hR = k.site(P, R, "head")
      _, R_tgt = self._head_target(g)
      neck_tgt = self.nom.neck[0] - NECK_PER_M * g[:, 0]
      res.append((q[:, self.neck[0]] - neck_tgt)[:, None])
      res.append(0.3 * so3_log(R_tgt.transpose(-1, -2) @ hR))
    else:
      res.append(0.1 * (q[:, self.neck] - torch.tensor(self.nom.neck, device=q.device)))
    return torch.cat(res, -1)

  # ------------------------------------------------------------------ query
  def initial_guess(self, n):
    return self.x_nominal.expand(n, -1).clone()

  def solve(self, g, x_warm=None, iters=3):
    """g (B,8) commands -> dict of path-frame reference quantities."""
    if x_warm is None:
      x_warm = self.initial_guess(g.shape[0])
      iters = max(iters, 15)
    x, r = self.lsq.solve(x_warm, g, iters=iters, tol=1e-10)
    root_pos, root_R, q = self._unpack(x, g)
    return x, {"root_pos": root_pos, "root_R": root_R, "q": q, "residual": r}

  def sample_commands(self, n, generator=None):
    u = torch.rand(n, 8, device=self.device, generator=generator)
    return self.cmd_lo + u * (self.cmd_hi - self.cmd_lo)
