"""Model-based control stack: gait planner + CoM MPC + whole-body controller,
with the three gait-feedback mechanisms of Hopkins et al. Sec. 7.1:
velocity adjustment, step adjustment and time scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
import proxsuite

from disneyrepro.gait.mpc import ComMPC
from disneyrepro.gait.params import GaitStyle
from disneyrepro.gait.planner import FootGeom, GaitPlanner, PHI_LTD, PHI_RLO, PHI_RTD, rotz
from disneyrepro.gait.wbc import WbcGains, WbcRef, WholeBodyController, so3_log


@dataclass
class FeedbackGains:
  # Velocity adjustment: LPF(deadband(Kp e + Kd edot)), 0.15 Hz cut-off.
  vel_kp: float = 1.5
  vel_kd: float = 0.4
  vel_deadband: float = 0.01
  vel_cutoff: float = 0.15
  vel_max: float = 0.5
  # Step adjustment: deadband(Kp e + Kd edot), applied to the upcoming foothold.
  step_kp: float = 0.8
  step_kd: float = 0.25
  step_deadband: float = 0.005
  step_max: float = 0.12
  step_freeze: float = 1.1
  step_rate: float = 100.0
  # Time scaling: kappa = exp(kp eps + kd epsdot), clipped to support-duration limits.
  time_kp: float = 8.0
  time_kd: float = 1.0
  kappa_min: float = 0.5
  kappa_max: float = 2.0
  min_single_support: float = 0.12
  min_double_support: float = 0.02
  enabled: bool = True


def deadband(x, db):
  n = np.linalg.norm(x)
  if n <= db:
    return np.zeros_like(x)
  return x * (1 - db / n)


@dataclass
class ControllerConfig:
  foot: FootGeom
  foot_sites: dict
  contact_local: dict  # mode -> list of local contact points (sole frame)
  upper_joints: tuple = ()
  dt_sim: float = 0.001
  wbc_every: int = 2
  plan_every: int = 8
  joint_kp: float = 30.0
  joint_kd: float = 1.0
  max_q_err: float = 0.15
  latch_measured: bool = True
  mpc_beta_pos: float = 0.1
  mpc_beta_vel: float = 0.1
  wbc: WbcGains = field(default_factory=WbcGains)
  feedback: FeedbackGains = field(default_factory=FeedbackGains)


class GaitController:
  def __init__(self, model, data, style: GaitStyle, cfg: ControllerConfig, posture=None):
    self.m, self.cfg = model, cfg
    mujoco.mj_forward(model, data)
    com = data.subtree_com[1].copy()
    # Initial path frame: middle of the feet, heading of the pelvis.
    pl, pr = (data.site_xpos[model.site(cfg.foot_sites[s]).id] for s in ("l", "r"))
    R = data.xmat[1].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    mid = 0.5 * (pl + pr)
    self.planner = GaitPlanner(style, cfg.foot, init_path=(mid[0], mid[1], yaw))
    for s, p in (("l", pl), ("r", pr)):
      self.planner.latched[s] = np.array([p[0], p[1], yaw])
      self.planner.target[s] = self.planner.latched[s].copy()
    self.mpc = ComMPC(com[2], cfg.foot)
    self.wbc = WholeBodyController(model, cfg.foot_sites, cfg.wbc, posture=posture)
    self.na = model.nv - 6
    self.q_des = data.qpos[7:].copy()
    self.qd_des = np.zeros(self.na)
    self.tau = np.zeros(self.na)
    self.k = 0
    self.v_cmd = np.zeros(3)
    self.xi_ref = np.array([com[0], com[1], 0.0, 0.0])
    self.cop_ref = com[:2].copy()
    self.plan_t = 0.0
    self.out = None
    self.prev_out = None
    self.vel_adj = np.zeros(2)
    self.log = []
    self._plan(data)

  # ------------------------------------------------------------- planning
  def _feedback(self, data):
    fb = self.cfg.feedback
    pl = self.planner
    if not fb.enabled or self.out is None:
      pl.kappa, pl.v_offset[:] = 1.0, 0.0
      return
    x_ref, v_ref, _, _ = self.mpc.state(self.plan_t)
    com = data.subtree_com[1][:2]
    comv = self.wbc.last.get("comv", np.zeros(3))[:2]
    e, ed = com - x_ref, comv - v_ref
    dtp = self.cfg.dt_sim * self.cfg.plan_every
    # Velocity adjustment (body / path frame).
    raw = deadband(fb.vel_kp * e + fb.vel_kd * ed, fb.vel_deadband)
    a = 1 - math.exp(-2 * math.pi * fb.vel_cutoff * dtp)
    self.vel_adj += a * (raw - self.vel_adj)
    n = np.linalg.norm(self.vel_adj)
    if n > fb.vel_max:
      self.vel_adj *= fb.vel_max / n
    c, s = math.cos(pl.path[2]), math.sin(pl.path[2])
    pl.v_offset[:] = [c * self.vel_adj[0] + s * self.vel_adj[1], -s * self.vel_adj[0] + c * self.vel_adj[1]]
    # Step adjustment for the swing foot; frozen late in the swing (the foot
    # cannot follow a last-moment target jump) and rate limited.
    step = deadband(fb.step_kp * e + fb.step_kd * ed, fb.step_deadband)
    n = np.linalg.norm(step)
    if n > fb.step_max:
      step *= fb.step_max / n
    if self._swing_progress() < fb.step_freeze:
      d = step - pl.step_offset
      lim = fb.step_rate * dtp
      n = np.linalg.norm(d)
      if n > lim:
        d *= lim / n
      pl.step_offset[:] = pl.step_offset + d
    step = pl.step_offset.copy()
    # Time scaling along the axis from the current support to the next foothold.
    sw = [s_ for s_ in ("l", "r") if pl.swinging(s_)]
    if sw:
      other = "r" if sw[0] == "l" else "l"
      axis = pl.target[sw[0]][:2] - pl.latched[other][:2]
    else:
      axis = np.array([math.cos(pl.path[2]), math.sin(pl.path[2])])
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    kappa = math.exp(fb.time_kp * (e @ axis) + fb.time_kd * (ed @ axis))
    b = pl.blend
    t_ev = b.t_events
    ss = min(t_ev[1] - t_ev[0], t_ev[3] - t_ev[2])
    ds = min(t_ev[2] - t_ev[1], t_ev[4] - t_ev[3])
    kmax = min(fb.kappa_max, ss / fb.min_single_support, ds / fb.min_double_support if ds > 1e-6 else fb.kappa_max)
    pl.kappa = float(np.clip(kappa, fb.kappa_min, max(kmax, 1.0)))
    self.fb_state = {"e": e, "ed": ed, "kappa": pl.kappa, "step": step.copy(), "vel_adj": pl.v_offset.copy()}

  def _swing_progress(self):
    pl = self.planner
    b = pl.blend
    t = b.phase_map(pl.phi)
    for side, (t0, t1) in (("l", (0.0, b.t_events[1])), ("r", (b.t_events[2], b.t_events[3]))):
      if pl.swinging(side):
        return (t - t0) / max(t1 - t0, 1e-6)
    return 0.0

  def _foot_pose2d(self, data, side):
    sid = self.m.site(self.cfg.foot_sites[side]).id
    R = data.site_xmat[sid].reshape(3, 3)
    p = data.site_xpos[sid]
    return np.array([p[0], p[1], math.atan2(R[1, 0], R[0, 0])])

  def _plan(self, data):
    dtp = self.cfg.dt_sim * self.cfg.plan_every
    if self.out is not None:
      x, v, _, p = self.mpc.state(self.plan_t)
      self.ref_state_backup = (x.copy(), v.copy(), None, None)
      # Reference governor: pull the planned CoM state towards the measured one
      # (beta = 0 is a pure open-loop plan, beta = 1 replans from the estimate).
      com = data.subtree_com[1][:2]
      comv = self.wbc.last.get("comv", np.zeros(3))[:2]
      bp, bv = self.cfg.mpc_beta_pos, self.cfg.mpc_beta_vel
      x = x + bp * (com - x)
      v = v + bv * (comv - v)
      self.xi_ref = np.concatenate([x, v])
      self.cop_ref = p
      self._feedback(data)
      n_ev = len(self.planner.events)
      self.planner.update(dtp, self.v_cmd)
      # Footholds are latched at the measured foot pose at touch-down.
      for _, name in self.planner.events[n_ev:]:
        if name in ("LTD", "RTD") and self.cfg.latch_measured:
          side = "l" if name == "LTD" else "r"
          self.planner.latched[side] = self._foot_pose2d(data, side)
    self.prev_out = self.out
    self.out = self.planner.output()
    self.mpc.plan(self.xi_ref, self.cop_ref, self.out.supports)
    if self.mpc.status != proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED and self.out is not None and self.prev_out is not None:
      # Blended initial state not capturable with the planned steps: fall back
      # to the open-loop reference state (always feasible by construction).
      x, v, _, _ = self.ref_state_backup
      self.xi_ref = np.concatenate([x, v])
      self.mpc.plan(self.xi_ref, self.cop_ref, self.out.supports)
    self.plan_t = 0.0
    # Finite-difference planner velocities (the planner runs at 125 Hz).
    o, po = self.out, self.prev_out
    self.ref_vel = {}
    if po is None:
      self.ref_vel = {"z": 0.0, "w": np.zeros(3), "fv": {s: np.zeros(3) for s in "lr"}, "fw": {s: np.zeros(3) for s in "lr"}, "uq": {}}
    else:
      self.ref_vel["z"] = (o.pelvis_z - po.pelvis_z) / dtp
      self.ref_vel["w"] = so3_log(o.pelvis_R @ po.pelvis_R.T) / dtp
      self.ref_vel["fv"] = {s: (o.foot_pos[s] - po.foot_pos[s]) / dtp for s in "lr"}
      self.ref_vel["fw"] = {s: so3_log(o.foot_R[s] @ po.foot_R[s].T) / dtp for s in "lr"}
      self.ref_vel["uq"] = {j: (o.upper[j] - po.upper[j]) / dtp for j in o.upper}

  def _wbc_ref(self):
    o = self.out
    x, v, a, _ = self.mpc.state(self.plan_t)
    cl = self.cfg.contact_local
    pts = {}
    for s in "lr":
      if o.contact[s]:
        pts[s] = cl[o.contact_mode[s]]
    rv = self.ref_vel
    return WbcRef(
      com=x, com_vel=v, com_acc=a,
      pelvis_z=o.pelvis_z + rv["z"] * self.plan_t, pelvis_zd=rv["z"], pelvis_zdd=0.0,
      pelvis_R=o.pelvis_R, pelvis_w=rv["w"],
      foot_pos={s: o.foot_pos[s] + rv["fv"][s] * self.plan_t for s in "lr"},
      foot_vel=rv["fv"], foot_acc={s: np.zeros(3) for s in "lr"},
      foot_R=o.foot_R, foot_w=rv["fw"],
      contact_points=pts,
      upper_q={j: o.upper[j] + rv["uq"].get(j, 0.0) * self.plan_t for j in o.upper},
      upper_qd=rv["uq"],
    )

  # ----------------------------------------------------------------- step
  def step(self, data, v_cmd=None):
    if v_cmd is not None:
      self.v_cmd = np.asarray(v_cmd, float)
    cfg = self.cfg
    if self.k % cfg.plan_every == 0 and self.k > 0:
      self._plan(data)
    if self.k % cfg.wbc_every == 0:
      ref = self._wbc_ref()
      qdd, tau = self.wbc.compute(data, ref, cfg.dt_sim * cfg.wbc_every)
      self.tau = tau
      dt = cfg.dt_sim * cfg.wbc_every
      self.qd_des = data.qvel[6:] + qdd[6:] * dt
      self.q_des = np.clip(self.q_des + self.qd_des * dt, data.qpos[7:] - cfg.max_q_err, data.qpos[7:] + cfg.max_q_err)
      self.last_ref = ref
    self.k += 1
    self.plan_t += cfg.dt_sim
    u_dof = self.tau + cfg.joint_kp * (self.q_des - data.qpos[7:]) + cfg.joint_kd * (self.qd_des - data.qvel[6:])
    ctrl = np.zeros(self.m.nu)
    for a, dof in enumerate(self.wbc.dof_of_act):
      ctrl[a] = u_dof[dof - 6]
    lo, hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
    data.ctrl[:] = np.clip(ctrl, lo, hi)
    return data.ctrl
