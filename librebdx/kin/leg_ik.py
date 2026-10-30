"""Generic batched whole-body IK for legged characters.

Unknowns: root horizontal position (x, y) and the leg joints. Given per sample:
root height, root rotation, foot target poses, non-leg joint values (neck,
tail, ...) and a CoM (x, y) target. Feet are constrained in position + yaw +
pitch (5 per foot); with ``foot_roll=True`` also in roll (6-DoF legs).
"""

from __future__ import annotations

import torch

from librebdx.kin.ik import BatchedLSQ
from librebdx.kin.torch_kin import TorchKinematics


class LegIK:
  def __init__(self, xml, leg_joints, foot_sites, foot_roll=True, device="cpu", com_weight=5.0, xy_bound=0.5):
    self.kin = k = TorchKinematics(xml, device=device)
    self.device = device
    self.leg = k.jidx(leg_joints)
    self.other = [i for i in range(k.nq) if i not in self.leg]
    self.feet = list(foot_sites)
    self.foot_roll = foot_roll
    self.com_weight = com_weight
    nl = len(self.leg)
    lo = torch.cat([torch.full((2,), -xy_bound, device=device), k.joint_range[self.leg, 0]])
    hi = torch.cat([torch.full((2,), xy_bound, device=device), k.joint_range[self.leg, 1]])
    self.lsq = BatchedLSQ(self._residual, lo, hi, damping=1e-6)
    self.nl = nl
    order = self.leg + self.other
    self._perm = torch.tensor([order.index(i) for i in range(k.nq)], device=device)

  # aux layout: [z(1), R(9), q_other(no), com_xy(2), (foot_p(3), foot_R(9)) * nfeet]
  def pack(self, z, R, q_other, com_xy, foot_p, foot_R):
    parts = [z[:, None], R.reshape(-1, 9), q_other, com_xy]
    for i in range(len(self.feet)):
      parts += [foot_p[:, i], foot_R[:, i].reshape(-1, 9)]
    return torch.cat(parts, -1)

  def _unpack_aux(self, aux):
    no = len(self.other)
    z = aux[:, 0]
    R = aux[:, 1:10].reshape(-1, 3, 3)
    q_other = aux[:, 10 : 10 + no]
    com = aux[:, 10 + no : 12 + no]
    o = 12 + no
    fp, fR = [], []
    for _ in self.feet:
      fp.append(aux[:, o : o + 3])
      fR.append(aux[:, o + 3 : o + 12].reshape(-1, 3, 3))
      o += 12
    return z, R, q_other, com, fp, fR

  def full_q(self, x, q_other):
    # Out-of-place gather (vmap-safe): concat [legs, others] then permute.
    cat = torch.cat([x[:, 2:], q_other], -1)
    return cat[:, self._perm]

  def _residual(self, x, aux):
    k = self.kin
    z, R, q_other, com, fp, fR = self._unpack_aux(aux)
    root = torch.cat([x[:, :2], z[:, None]], -1)
    q = self.full_q(x, q_other)
    P, Rb = k.forward(root, R, q)
    res = []
    for s, p_t, R_t in zip(self.feet, fp, fR):
      p, Rs = k.site(P, Rb, s)
      res.append(p - p_t)
      E = R_t.transpose(-1, -2) @ Rs
      # yaw/pitch errors from the forward axis; roll from the lateral axis.
      res.append(torch.stack([E[:, 1, 0], E[:, 2, 0]], -1))
      if self.foot_roll:
        res.append(E[:, 2, 1:2])
    c = k.com(P, Rb)
    res.append(self.com_weight * (c[:, :2] - com))
    return torch.cat(res, -1)

  def solve(self, aux, x0, iters=20):
    x, r = self.lsq.solve(x0, aux, iters=iters, tol=1e-12)
    return x, r
