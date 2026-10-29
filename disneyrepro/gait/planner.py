"""Procedural gait planner r(t) = g(v, s, Pi, t) of Hopkins et al. (Sec. 4).

The planner state is the gait phase phi (contact events at fixed phases 0, 0.1,
0.5, 0.6 thanks to the phase map) and the path frame. Outputs per update:
pelvis height and rotation, left/right foot poses, upper-body joint targets,
contact flags, and the support schedule over a preview horizon for the CoM MPC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from disneyrepro.gait.params import GaitStyle

PHI_LTD, PHI_RLO, PHI_RTD = 0.1, 0.5, 0.6


def rotz(a):
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def roty(a):
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotx(a):
  c, s = math.cos(a), math.sin(a)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def wrap(a):
  return (a + math.pi) % (2 * math.pi) - math.pi


def compose2d(p, q):
  """Planar pose composition p (+) q, poses (x, y, yaw)."""
  c, s = math.cos(p[2]), math.sin(p[2])
  return np.array([p[0] + c * q[0] - s * q[1], p[1] + s * q[0] + c * q[1], p[2] + q[2]])


def predict_path(p, v, tau):
  """Path pose after tau seconds at constant body velocity v = (vx, vy, wz)."""
  vx, vy, w = v
  if abs(w) < 1e-6:
    dx, dy = vx * tau, vy * tau
  else:
    sw, cw = math.sin(w * tau), math.cos(w * tau)
    dx = (vx * sw + vy * (cw - 1)) / w
    dy = (vy * sw - vx * (cw - 1)) / w
  return compose2d(p, np.array([dx, dy, w * tau]))


@dataclass
class FootGeom:
  heel_x: float  # heel / toe positions along the sole, relative to the sole centre
  toe_x: float
  half_width: float
  line_contact: bool = False


@dataclass
class PlannerOutput:
  pelvis_z: float
  pelvis_R: np.ndarray
  foot_pos: dict
  foot_R: dict
  contact: dict
  contact_mode: dict  # "flat" | "heel" | "toe" | "swing"
  upper: dict
  supports: list = field(default_factory=list)  # (t0, t1, [(side, pose2d), ...])
  phi: float = 0.0
  path: np.ndarray = None


class GaitPlanner:
  def __init__(self, style: GaitStyle, foot: FootGeom, init_path=(0.0, 0.0, 0.0), horizon=2.0):
    self.style = style
    self.foot = foot
    self.horizon = horizon
    self.path = np.array(init_path, float)
    self.v_cmd = np.zeros(3)
    self.v_offset = np.zeros(2)  # velocity adjustment (gait feedback)
    self.step_offset = np.zeros(2)  # step adjustment (world xy)
    self.kappa = 1.0
    self.phi = PHI_RTD + 1e-6  # start in double support
    self.blend = style.blend(self.v_cmd)
    self.latched = {s: compose2d(self.path, self._stance(s)) for s in ("l", "r")}
    self.target = dict(self.latched)
    self.time = 0.0
    self.events = []

  # ---------------------------------------------------------------- helpers
  @property
  def v(self):
    return np.array([self.v_cmd[0] + self.v_offset[0], self.v_cmd[1] + self.v_offset[1], self.v_cmd[2]])

  def _stance(self, side):
    b = self.blend
    sgn = 1.0 if side == "l" else -1.0
    return np.array([sgn * 0.5 * b.attr("stance_x_bias"), sgn * 0.5 * b.attr("stance_width"), sgn * 0.5 * b.attr("stance_splay")])

  def swinging(self, side, phi=None):
    phi = self.phi if phi is None else phi
    return phi < PHI_LTD if side == "l" else PHI_RLO <= phi < PHI_RTD

  def _time_to_phase(self, phi_target, cycles_ahead=0):
    """Seconds until phase phi_target (in the current or a later cycle)."""
    pm = self.blend.phase_map
    return (cycles_ahead * self.blend.T + pm(phi_target) - pm(self.phi)) / max(self.kappa, 1e-3)

  def _foothold_time(self, side, k=0):
    """Seconds until the middle of the k-th upcoming contact interval of `side`."""
    b = self.blend
    t_ltd, t_rlo, t_rtd, T = b.t_events[1], b.t_events[2], b.t_events[3], b.T
    t_now = b.phase_map(self.phi)
    if side == "l":
      mid = 0.5 * (t_ltd + T)
      base = 0 if self.phi < PHI_LTD else T
    else:
      mid = 0.5 * (t_rtd + T + t_rlo)
      base = 0 if self.phi < PHI_RTD else T
    return (base + k * T + mid - t_now) / max(self.kappa, 1e-3)

  def _predicted_foothold(self, side, k=0):
    tau = self._foothold_time(side, k)
    pose = compose2d(predict_path(self.path, self.v, tau), self._stance(side))
    if k == 0:
      pose[:2] += self.step_offset
      pose = self._safety_project(side, pose)
    return pose

  def _safety_project(self, side, pose):
    """Keep the swing foothold out of the support foot (conservative heuristic)."""
    other = "r" if side == "l" else "l"
    sup = self.latched[other]
    c, s = math.cos(sup[2]), math.sin(sup[2])
    d = pose[:2] - sup[:2]
    loc = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    sgn = 1.0 if side == "l" else -1.0
    min_w = 2 * self.foot.half_width + 0.02
    if sgn * loc[1] < min_w:
      loc[1] = sgn * min_w
    pose = pose.copy()
    pose[:2] = sup[:2] + np.array([c * loc[0] - s * loc[1], s * loc[0] + c * loc[1]])
    return pose

  # ---------------------------------------------------------------- update
  def update(self, dt, v_cmd=None):
    if v_cmd is not None:
      self.v_cmd = np.asarray(v_cmd, float)
    self.blend = self.style.blend(self.v)
    sdt = dt * self.kappa
    self.time += dt
    old = self.phi
    # Advance gait time exactly through the phase map (phi <- Omega^-1(Omega(phi) + dt));
    # Euler steps on phi overshoot the short double-support segments.
    pm = self.blend.phase_map
    t_new = pm(self.phi) + sdt
    T = self.blend.T
    self.phi = pm.inverse(t_new) if t_new < T else 1.0 + pm.inverse(t_new - T)
    self.path = predict_path(self.path, self.v, sdt)
    # Contact events at fixed phases.
    for phi_e, name in ((PHI_LTD, "LTD"), (PHI_RLO, "RLO"), (PHI_RTD, "RTD")):
      if old < phi_e <= self.phi:
        self._event(name)
    if self.phi >= 1.0:
      self.phi -= 1.0
      self._event("LLO")
    for side in ("l", "r"):
      if self.swinging(side):
        self.target[side] = self._predicted_foothold(side)

  def _event(self, name):
    self.events.append((self.time, name))
    if name == "LTD":
      self.latched["l"] = self.target["l"].copy()
    elif name == "RTD":
      self.latched["r"] = self.target["r"].copy()
    if name in ("LTD", "RTD"):
      self.step_offset[:] = 0.0

  # ---------------------------------------------------------------- output
  def _heel_toe(self, pitch_heel, pitch_toe):
    """Sequential pivots about the heel, then about the (moved) toe (Fig. 5)."""
    f = self.foot
    h = np.array([f.heel_x, 0, 0.0])
    t = np.array([f.toe_x, 0, 0.0])
    R1 = roty(pitch_heel)
    t1 = h + R1 @ (t - h)
    R2 = roty(pitch_toe)
    R = R2 @ R1
    # p -> t1 + R2 (h + R1 (p - h) - t1); offset of the frame origin (p = 0):
    off = t1 + R2 @ (h + R1 @ (-h) - t1)
    return off, R

  def _stance_pitch(self, side):
    """Heel/toe pitch of a stance foot near lift-off / after touch-down."""
    b = self.blend
    pm = b.phase_map
    t = pm(self.phi)
    T = b.T
    t_lo = 0.0 if side == "l" else b.t_events[2]
    t_td = b.t_events[1] if side == "l" else b.t_events[3]
    pre = "lfoot" if side == "l" else "rfoot"
    to_lo = (t_lo - t) % T
    since_td = (t - t_td) % T
    dlo, dtd = b.attr("dt_psi_lo"), b.attr("dt_psi_td")
    heel, toe = 0.0, 0.0
    if dlo > 1e-4 and to_lo < dlo:
      phi_lo = 0.0 if side == "l" else PHI_RLO
      w = 1 - to_lo / dlo
      toe = w * b.curve(f"{pre}_toe", phi_lo + 1e-6)
      heel = w * b.curve(f"{pre}_heel", phi_lo + 1e-6)
    if dtd > 1e-4 and since_td < dtd:
      phi_td = PHI_LTD if side == "l" else PHI_RTD
      w = 1 - since_td / dtd
      heel += w * b.curve(f"{pre}_heel", phi_td - 1e-6)
      toe += w * b.curve(f"{pre}_toe", phi_td - 1e-6)
    return min(heel, 0.0), max(toe, 0.0)

  def output(self) -> PlannerOutput:
    b = self.blend
    phi = self.phi
    yaw = self.path[2]
    pel_R = rotz(yaw) @ rotz(b.curve("pelvis_yaw", phi)) @ roty(b.curve("pelvis_pitch", phi)) @ rotx(b.curve("pelvis_roll", phi))
    foot_pos, foot_R, contact, mode = {}, {}, {}, {}
    for side in ("l", "r"):
      pre = "lfoot" if side == "l" else "rfoot"
      if self.swinging(side):
        t = b.phase_map(phi)
        t_lo = 0.0 if side == "l" else b.t_events[2]
        t_td = b.t_events[1] if side == "l" else b.t_events[3]
        s = min(max((t - t_lo) / (t_td - t_lo), 0.0), 1.0)
        a = 3 * s * s - 2 * s**3
        p0, p1 = self.latched[side], self.target[side]
        nom = np.array([p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]), p0[2] + a * wrap(p1[2] - p0[2])])
        heel, toe = min(b.curve(f"{pre}_heel", phi), 0.0), max(b.curve(f"{pre}_toe", phi), 0.0)
        sw = np.array([b.curve(f"{pre}_x", phi), b.curve(f"{pre}_y", phi), b.curve(f"{pre}_z", phi)])
        fyaw, froll = b.curve(f"{pre}_yaw", phi), b.curve(f"{pre}_roll", phi)
        contact[side], mode[side] = False, "swing"
      else:
        nom = self.latched[side]
        heel, toe = self._stance_pitch(side)
        sw, fyaw, froll = np.zeros(3), 0.0, 0.0
        contact[side] = True
        mode[side] = "toe" if toe > 1e-3 else ("heel" if heel < -1e-3 else "flat")
      off, Rht = self._heel_toe(heel, toe)
      Rn = rotz(nom[2])
      Rsty = rotz(fyaw) @ Rht @ rotx(froll)
      foot_R[side] = Rn @ Rsty
      foot_pos[side] = np.array([nom[0], nom[1], 0.0]) + Rn @ (sw + rotz(fyaw) @ off)
    upper = {j: b.curve(f"joint:{j}", phi) for j in self.style.upper_joints}
    return PlannerOutput(
      pelvis_z=b.curve("pelvis_z", phi), pelvis_R=pel_R, foot_pos=foot_pos, foot_R=foot_R,
      contact=contact, contact_mode=mode, upper=upper, supports=self.preview(), phi=phi, path=self.path.copy(),
    )

  def preview(self):
    """Support schedule [(t0, t1, [(side, pose2d), ...]), ...] over the horizon."""
    evs = []  # (time, kind, side)
    for k in range(0, 8):
      for phi_e, kind, side in ((PHI_LTD, "TD", "l"), (PHI_RLO, "LO", "r"), (PHI_RTD, "TD", "r"), (1.0, "LO", "l")):
        t = self._time_to_phase(phi_e, k)
        if 1e-6 < t < self.horizon + 1.0:
          evs.append((t, kind, side))
    evs.sort()
    state = {s: not self.swinging(s) for s in ("l", "r")}
    pose = {s: (self.latched[s] if state[s] else self.target[s]) for s in ("l", "r")}
    td_count = {"l": 0, "r": 0}
    sup, t0 = [], 0.0
    for t, kind, side in evs:
      if t > t0 + 1e-6:
        sup.append((t0, min(t, self.horizon), [(s, pose[s].copy()) for s in ("l", "r") if state[s]]))
      if t >= self.horizon:
        t0 = self.horizon
        break
      t0 = t
      if kind == "LO":
        state[side] = False
      else:
        # The n-th upcoming touchdown of this side lands at its n-th foothold.
        state[side] = True
        pose[side] = self._predicted_foothold(side, td_count[side])
        td_count[side] += 1
    if t0 < self.horizon - 1e-6:
      sup.append((t0, self.horizon, [(s, pose[s].copy()) for s in ("l", "r") if state[s]]))
    return sup
