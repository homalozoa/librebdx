"""Shared pieces of the Disney-character imitation RL formulation.

Grandia et al. (RSS 2024) Sec. V and Olaf (Mueller et al.) Sec. V:
  * kinematic reference x_t = (p, theta, v, omega, q, qdot, c_L, c_R) produced by a
    generator f(path_frame, g [, phase]),
  * path frame (x, y, yaw) that maps path-frame references to the world,
  * imitation + regularisation + survival rewards (Table I),
  * policy state s_t = (p^P, theta^P, v^T, omega^T, q, qdot, a_{t-1}, a_{t-2}).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_from_matrix,
)

from disneyrepro.kin.torch_kin import axis_angle_to_mat, so3_log


def yaw_mat(yaw):
  ez = torch.zeros(yaw.shape + (3,), device=yaw.device)
  ez[..., 2] = 1.0
  return axis_angle_to_mat(ez, yaw)


def wrap(a):
  return torch.atan2(torch.sin(a), torch.cos(a))


# ----------------------------------------------------------------------------
# Reference command base class
# ----------------------------------------------------------------------------


@dataclass(kw_only=True)
class CharacterRefCfg(CommandTermCfg):
  entity_name: str = "robot"
  foot_sites: tuple[str, str] = ("left_sole", "right_sole")
  pf_max_dist: float = 0.15
  """Path frame is projected to stay within this distance of the torso."""
  reset_noise_joint: float = 0.02
  init_yaw_range: tuple[float, float] = (-math.pi, math.pi)


class CharacterRef(CommandTerm):
  """Holds the path frame and the current kinematic reference (world frame).

  Subclasses implement ``_reference_path(env_ids)`` returning the path-frame
  reference: root_pos (B,3), root_R (B,3,3), q (B,nj in entity joint order),
  contacts (B,2) -- and the command/phase bookkeeping.
  """

  cfg: CharacterRefCfg

  def __init__(self, cfg: CharacterRefCfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    B, dev = self.num_envs, self.device
    self.nj = len(self.robot.joint_names)
    self.pf_pos = torch.zeros(B, 2, device=dev)
    self.pf_yaw = torch.zeros(B, device=dev)
    z = lambda *s: torch.zeros(B, *s, device=dev)
    self.ref_pos, self.ref_R = z(3), torch.eye(3, device=dev).repeat(B, 1, 1)
    self.ref_lin_vel, self.ref_ang_vel = z(3), z(3)
    self.ref_q, self.ref_qd = z(self.nj), z(self.nj)
    self.ref_contact = torch.ones(B, 2, device=dev)
    self._prev_pos, self._prev_R, self._prev_q = z(3), self.ref_R.clone(), z(self.nj)
    self.ref_pos_p, self.ref_R_p = z(3), self.ref_R.clone()
    self._prev_pos_p, self._prev_R_p = z(3), self.ref_R.clone()
    self._needs_init = torch.ones(B, dtype=torch.bool, device=dev)
    # Env ids waiting for (re)initialisation, as handed to reset(): avoids a
    # device-to-host sync (nonzero) on every step.
    self._pending = [torch.arange(B, device=dev)]
    self.foot_site_ids = self.robot.find_sites(list(cfg.foot_sites), preserve_order=True)[0]
    self.metrics["err_torso_pos"] = z()
    self.metrics["err_joint"] = z()
    self._err_sum = z(2)
    self._err_n = z()

  # Subclass API ---------------------------------------------------------------
  def _reference_path(self, env_ids):
    raise NotImplementedError

  def _advance(self, dt):
    """Advance command / phase / path-frame dynamics by dt (all envs)."""
    raise NotImplementedError

  def _init_episode(self, env_ids):
    """Called on reset before the reference is evaluated."""
    raise NotImplementedError

  def _init_velocities(self, env_ids):
    """Reference (lin, ang, joint) velocities for reference-state initialisation."""
    return None

  # ---------------------------------------------------------------------------
  def reset(self, env_ids):
    self._needs_init[env_ids] = True
    if env_ids is None or isinstance(env_ids, slice):
      self._pending.append(torch.arange(self.num_envs, device=self.device))
    else:
      self._pending.append(torch.as_tensor(env_ids, device=self.device).flatten())
    out = super().reset(env_ids)
    self._err_sum[env_ids] = 0.0
    self._err_n[env_ids] = 0.0
    return out

  def _to_world(self, root_pos_p, root_R_p, env_ids=slice(None)):
    Rpf = yaw_mat(self.pf_yaw[env_ids])
    pos = torch.zeros_like(root_pos_p)
    pos[:, :2] = self.pf_pos[env_ids]
    pos = pos + (Rpf @ root_pos_p.unsqueeze(-1)).squeeze(-1)
    pos[:, 2] = root_pos_p[:, 2] + self._env.scene.env_origins[env_ids, 2]
    return pos, Rpf @ root_R_p

  def feet_mid_and_yaw(self):
    sp = self.robot.data.site_pos_w[:, self.foot_site_ids]
    sq = self.robot.data.site_quat_w[:, self.foot_site_ids]
    mid = sp[:, :, :2].mean(1)
    fwd = matrix_from_quat(sq)[..., :, 0]  # site x axes (B,2,3)
    yaws = torch.atan2(fwd[..., 1], fwd[..., 0])
    yaw = torch.atan2(torch.sin(yaws).sum(1), torch.cos(yaws).sum(1))
    return mid, yaw

  def project_path_frame(self):
    torso = self.robot.data.root_link_pos_w[:, :2]
    d = self.pf_pos - torso
    n = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    self.pf_pos = torch.where(n > self.cfg.pf_max_dist, torso + d / n * self.cfg.pf_max_dist, self.pf_pos)

  def path_velocity(self):
    """Nominal path-frame velocity (world xy, yaw rate); the projection towards
    the torso is a correction and does not enter the reference velocity."""
    return torch.zeros(self.num_envs, 2, device=self.device), torch.zeros(self.num_envs, device=self.device)

  def _evaluate(self, env_ids=slice(None)):
    root_pos_p, root_R_p, q, contact = self._reference_path(env_ids)
    pos, R = self._to_world(root_pos_p, root_R_p, env_ids)
    self.ref_pos[env_ids], self.ref_R[env_ids] = pos, R
    self.ref_pos_p[env_ids], self.ref_R_p[env_ids] = root_pos_p, root_R_p
    self.ref_q[env_ids], self.ref_contact[env_ids] = q, contact

  def _update_command(self, env_ids):
    dt = self._env.step_dt
    if env_ids is None:
      self._advance(dt)
      self._evaluate()
      # Reference velocities from the generator: motion relative to the path
      # frame plus the nominal path-frame velocity (Eq. 1-3).
      Rpf = yaw_mat(self.pf_yaw)
      v_p = (self.ref_pos_p - self._prev_pos_p) / dt
      w_p = so3_log(self.ref_R_p @ self._prev_R_p.transpose(-1, -2)) / dt
      v_pf, w_pf = self.path_velocity()
      r = (Rpf @ self.ref_pos_p.unsqueeze(-1)).squeeze(-1)
      lin = (Rpf @ v_p.unsqueeze(-1)).squeeze(-1)
      lin[:, :2] += v_pf + torch.stack([-w_pf * r[:, 1], w_pf * r[:, 0]], -1)
      ang = (Rpf @ w_p.unsqueeze(-1)).squeeze(-1)
      ang[:, 2] += w_pf
      self.ref_lin_vel, self.ref_ang_vel = lin, ang
      self.ref_qd = (self.ref_q - self._prev_q) / dt
    init = torch.cat(self._pending) if len(self._pending) > 1 else (self._pending[0] if self._pending else None)
    self._pending = []
    if init is not None and init.numel() > 0:
      self._init_episode(init)
      self._evaluate(init)
      vel = self._init_velocities(init)
      if vel is None:
        self.ref_lin_vel[init] = 0.0
        self.ref_ang_vel[init] = 0.0
        self.ref_qd[init] = 0.0
      else:
        self.ref_lin_vel[init], self.ref_ang_vel[init], self.ref_qd[init] = vel
      self._write_state(init)
      self._needs_init[init] = False
      # Commands run after the env's forward(); refresh derived quantities.
      self._env.sim.forward()
    self._prev_pos, self._prev_R, self._prev_q = self.ref_pos.clone(), self.ref_R.clone(), self.ref_q.clone()
    self._prev_pos_p, self._prev_R_p = self.ref_pos_p.clone(), self.ref_R_p.clone()

  def _write_state(self, env_ids):
    q = self.ref_q[env_ids] + self.cfg.reset_noise_joint * torch.randn_like(self.ref_q[env_ids])
    lim = self.robot.data.soft_joint_pos_limits[env_ids]
    q = torch.clip(q, lim[..., 0], lim[..., 1])
    self.robot.write_joint_state_to_sim(q, self.ref_qd[env_ids], env_ids=env_ids)
    quat = quat_from_matrix(self.ref_R[env_ids])
    pos = self.ref_pos[env_ids].clone()
    pos[:, 2] += 0.002
    state = torch.cat([pos, quat, self.ref_lin_vel[env_ids], self.ref_ang_vel[env_ids]], -1)
    self.robot.write_root_state_to_sim(state, env_ids=env_ids)

  def _update_metrics(self):
    # Episode-averaged tracking errors (logged when the episode ends).
    p = self.robot.data.root_link_pos_w
    e = torch.stack([(p[:, :2] - self.ref_pos[:, :2]).norm(dim=-1), (self.robot.data.joint_pos - self.ref_q).abs().mean(-1)], -1)
    live = (~self._needs_init).float()
    self._err_sum += e * live[:, None]
    self._err_n += live
    avg = self._err_sum / self._err_n.clamp(min=1)[:, None]
    self.metrics["err_torso_pos"] = avg[:, 0]
    self.metrics["err_joint"] = avg[:, 1]

  # Policy-state helpers -------------------------------------------------------
  def torso_in_path_frame(self):
    Rpf = yaw_mat(self.pf_yaw)
    p = self.robot.data.root_link_pos_w
    d = torch.zeros_like(p)
    d[:, :2] = p[:, :2] - self.pf_pos
    p_rel = (Rpf.transpose(-1, -2) @ d.unsqueeze(-1)).squeeze(-1)
    R = Rpf.transpose(-1, -2) @ matrix_from_quat(self.robot.data.root_link_quat_w)
    return p_rel[:, :2], R


# ----------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------


def obs_torso_pos_path(env, command_name):
  return env.command_manager.get_term(command_name).torso_in_path_frame()[0]


def obs_torso_rot_path(env, command_name):
  R = env.command_manager.get_term(command_name).torso_in_path_frame()[1]
  return R[..., :, :2].reshape(R.shape[0], 6)


def obs_root_lin_vel_b(env, entity_name="robot"):
  r = env.scene[entity_name]
  return quat_apply_inverse(r.data.root_link_quat_w, r.data.root_link_lin_vel_w)


def obs_root_ang_vel_b(env, entity_name="robot"):
  r = env.scene[entity_name]
  return quat_apply_inverse(r.data.root_link_quat_w, r.data.root_link_ang_vel_w)


def obs_measured_joint_pos(env, entity_name="robot"):
  """Joint encoder readings with offset, backlash and noise (Eq. 19), minus the default pose."""
  from disneyrepro.rl.disney_actuator import DisneyActuator

  r = env.scene[entity_name]
  q, qd = r.data.joint_pos, r.data.joint_vel
  out = q.clone()
  for act in r.actuators:
    if isinstance(act, DisneyActuator):
      ids = act.target_ids
      out[:, ids] = act.measured_position(q[:, ids], qd[:, ids])
  return out - r.data.default_joint_pos


def obs_joint_pos_rel(env, entity_name="robot"):
  r = env.scene[entity_name]
  return r.data.joint_pos - r.data.default_joint_pos


def obs_joint_vel(env, entity_name="robot"):
  return env.scene[entity_name].data.joint_vel


def obs_prev_action(env):
  return env.action_manager.prev_action


def obs_prev_prev_action(env):
  return env.action_manager._prev_prev_action


def obs_command(env, command_name):
  return env.command_manager.get_command(command_name)


def obs_ref_error(env, command_name):
  """Privileged: reference torso pose / joint errors in the torso frame."""
  c = env.command_manager.get_term(command_name)
  r = c.robot
  R = matrix_from_quat(r.data.root_link_quat_w)
  dp = (R.transpose(-1, -2) @ (c.ref_pos - r.data.root_link_pos_w).unsqueeze(-1)).squeeze(-1)
  dR = so3_log(R.transpose(-1, -2) @ c.ref_R)
  return torch.cat([dp, dR, c.ref_q - r.data.joint_pos, c.ref_contact], -1)


def obs_foot_friction(env, entity_name="robot", geom_name=".*sole.*"):
  """Privileged: (randomised) sliding friction of the foot geoms."""
  key = "_disney_friction_gid"
  if not hasattr(env, key):
    r = env.scene[entity_name]
    local = r.find_geoms(geom_name)[0][0]
    setattr(env, key, int(r.indexing.geom_ids[local]))
  gid = getattr(env, key)
  fr = env.sim.model.geom_friction
  if fr.ndim == 2:
    return fr[gid, 0:1].expand(env.num_envs, 1)
  return fr[:, gid, 0:1]


# ----------------------------------------------------------------------------
# Rewards (Table I)
# ----------------------------------------------------------------------------


def _cmd(env, name):
  return env.command_manager.get_term(name)


def rew_torso_pos_xy(env, command_name, scale=200.0):
  c = _cmd(env, command_name)
  e = (c.robot.data.root_link_pos_w[:, :2] - c.ref_pos[:, :2]).square().sum(-1)
  return torch.exp(-scale * e)


def rew_torso_ori(env, command_name, scale=20.0):
  c = _cmd(env, command_name)
  R = matrix_from_quat(c.robot.data.root_link_quat_w)
  e = so3_log(c.ref_R.transpose(-1, -2) @ R).square().sum(-1)
  return torch.exp(-scale * e)


def rew_lin_vel_xy(env, command_name, scale=8.0):
  c = _cmd(env, command_name)
  e = (c.robot.data.root_link_lin_vel_w[:, :2] - c.ref_lin_vel[:, :2]).square().sum(-1)
  return torch.exp(-scale * e)


def rew_lin_vel_z(env, command_name, scale=8.0):
  c = _cmd(env, command_name)
  e = (c.robot.data.root_link_lin_vel_w[:, 2] - c.ref_lin_vel[:, 2]).square()
  return torch.exp(-scale * e)


def rew_ang_vel_xy(env, command_name, scale=2.0):
  c = _cmd(env, command_name)
  e = (c.robot.data.root_link_ang_vel_w[:, :2] - c.ref_ang_vel[:, :2]).square().sum(-1)
  return torch.exp(-scale * e)


def rew_ang_vel_z(env, command_name, scale=2.0):
  c = _cmd(env, command_name)
  e = (c.robot.data.root_link_ang_vel_w[:, 2] - c.ref_ang_vel[:, 2]).square()
  return torch.exp(-scale * e)


def rew_joint_pos(env, command_name, joint_ids):
  c = _cmd(env, command_name)
  return -(c.robot.data.joint_pos[:, joint_ids] - c.ref_q[:, joint_ids]).square().sum(-1)


def rew_joint_vel(env, command_name, joint_ids):
  c = _cmd(env, command_name)
  return -(c.robot.data.joint_vel[:, joint_ids] - c.ref_qd[:, joint_ids]).square().sum(-1)


def rew_contact(env, command_name, sensor_name):
  c = _cmd(env, command_name)
  found = env.scene.sensors[sensor_name].data.found > 0
  return (found.float() == c.ref_contact).float().sum(-1)


def rew_torques(env, entity_name="robot"):
  return -env.scene[entity_name].data.actuator_force.square().sum(-1)


def rew_joint_acc(env, entity_name="robot"):
  return -env.scene[entity_name].data.joint_acc.square().sum(-1)


def rew_action_rate(env, action_ids):
  a = env.action_manager.action[:, action_ids]
  a1 = env.action_manager.prev_action[:, action_ids]
  return -(a - a1).square().sum(-1)


def rew_action_acc(env, action_ids):
  a = env.action_manager.action[:, action_ids]
  a1 = env.action_manager.prev_action[:, action_ids]
  a2 = env.action_manager._prev_prev_action[:, action_ids]
  return -(a - 2 * a1 + a2).square().sum(-1)


def rew_survival(env):
  return torch.ones(env.num_envs, device=env.device)


# ----------------------------------------------------------------------------
# Terminations
# ----------------------------------------------------------------------------


def term_contact(env, sensor_name):
  return (env.scene.sensors[sensor_name].data.found > 0).any(-1)


# ----------------------------------------------------------------------------
# Disturbances (Table V)
# ----------------------------------------------------------------------------


@dataclass
class DisturbanceGroup:
  bodies: tuple[str, ...]
  force_xy: tuple[float, float]
  force_z: tuple[float, float]
  torque_xy: tuple[float, float]
  torque_z: tuple[float, float]
  on: tuple[float, float]
  off: tuple[float, float]


class Disturbances:
  """Random on/off force-torque disturbances per body, with a linear curriculum.

  Used as a 'step' event: ``EventTermCfg(func=Disturbances(...), mode="step")``.
  Each listed body has an independent timer; while 'on' a random force and
  torque (uniform magnitude per dimension, random sign) is applied.
  """

  def __init__(self, groups, curriculum_iters=1500, steps_per_iter=24, entity_name="robot"):
    self.groups = groups
    self.curriculum_steps = curriculum_iters * steps_per_iter
    self.entity_name = entity_name
    self._init = False

  def _setup(self, env):
    r = env.scene[self.entity_name]
    B, dev = env.num_envs, env.device
    self.slots = []
    for g in self.groups:
      for b in g.bodies:
        bid = r.find_bodies(b)[0][0]
        self.slots.append((g, bid))
    S = len(self.slots)
    self.timer = torch.rand(B, S, device=dev) * 2.0
    self.active = torch.zeros(B, S, dtype=torch.bool, device=dev)
    self.wrench = torch.zeros(B, S, 6, device=dev)
    self.body_ids = torch.tensor([b for _, b in self.slots], device=dev)
    # Per-slot ranges as tensors: (S, 6) magnitude bounds and (S,) on/off durations.
    lo = [[*([g.force_xy[0]] * 2), g.force_z[0], *([g.torque_xy[0]] * 2), g.torque_z[0]] for g, _ in self.slots]
    hi = [[*([g.force_xy[1]] * 2), g.force_z[1], *([g.torque_xy[1]] * 2), g.torque_z[1]] for g, _ in self.slots]
    self.w_lo, self.w_hi = torch.tensor(lo, device=dev), torch.tensor(hi, device=dev)
    self.on_lo = torch.tensor([g.on[0] for g, _ in self.slots], device=dev)
    self.on_hi = torch.tensor([g.on[1] for g, _ in self.slots], device=dev)
    self.off_lo = torch.tensor([g.off[0] for g, _ in self.slots], device=dev)
    self.off_hi = torch.tensor([g.off[1] for g, _ in self.slots], device=dev)
    self.used = sorted(set(int(b) for _, b in self.slots))
    self.used_t = torch.tensor(self.used, device=dev)
    self._init = True

  @staticmethod
  def _u(lo, hi, shape, dev):
    mag = lo + (hi - lo) * torch.rand(shape, device=dev)
    return mag * torch.sign(torch.rand(shape, device=dev) - 0.5)

  def __call__(self, env, env_ids, dt=None):
    """Vectorised over worlds and slots: no host-device synchronisation per step."""
    if not self._init:
      self._setup(env)
    dt = env.step_dt
    dev = env.device
    scale = min(1.0, env.common_step_counter / max(1, self.curriculum_steps))
    self.timer -= dt
    flip = self.timer <= 0
    turning_on = flip & ~self.active
    B, S = flip.shape
    w = self.w_lo + (self.w_hi - self.w_lo) * torch.rand(B, S, 6, device=dev)
    w = w * torch.sign(torch.rand(B, S, 6, device=dev) - 0.5)
    self.wrench = torch.where(flip[..., None], torch.where(turning_on[..., None], w, torch.zeros_like(w)), self.wrench)
    d_on = self.on_lo + (self.on_hi - self.on_lo) * torch.rand(B, S, device=dev)
    d_off = self.off_lo + (self.off_hi - self.off_lo) * torch.rand(B, S, device=dev)
    self.timer = torch.where(flip, torch.where(turning_on, d_on, d_off), self.timer)
    self.active = torch.where(flip, turning_on, self.active)
    r = env.scene[self.entity_name]
    # Sum the wrenches per body (a body may appear in several groups).
    total = torch.zeros(env.num_envs, r.num_bodies, 6, device=dev)
    total.index_add_(1, self.body_ids, self.wrench * scale)
    total = total[:, self.used_t]
    r.write_external_wrench_to_sim(total[..., :3], total[..., 3:], body_ids=self.used)

  def reset(self, env_ids=None):
    if self._init and env_ids is not None:
      self.wrench[env_ids] = 0.0
      self.active[env_ids] = False


def table_v_groups(hips, feet, pelvis, head, scale=1.0):
  """Disturbance parameters of Table V (forces optionally scaled to robot mass)."""
  s = scale
  return [
    DisturbanceGroup(tuple(hips) + tuple(feet), (0, 5 * s), (0, 5 * s), (0, 0.25 * s), (0, 0.25 * s), (0.25, 2.0), (1.0, 3.0)),
    DisturbanceGroup((pelvis, head), (0, 5 * s), (0, 5 * s), (0, 0.25 * s), (0, 0.25 * s), (2.0, 10.0), (1.0, 3.0)),
    DisturbanceGroup((pelvis,), (90 * s, 150 * s), (0, 10 * s), (0, 15 * s), (0, 15 * s), (0.1, 0.1), (12.0, 15.0)),
  ]


# ----------------------------------------------------------------------------
# Curriculum
# ----------------------------------------------------------------------------


def curriculum_reward_weight(env, env_ids, term, w0, w1, it0, it1, steps_per_iter=24):
  """Linearly ramp a reward weight from w0 to w1 between iterations it0 and it1."""
  it = env.common_step_counter / steps_per_iter
  a = min(max((it - it0) / max(it1 - it0, 1), 0.0), 1.0)
  w = w0 + a * (w1 - w0)
  env.reward_manager.get_term_cfg(term).weight = w
  return w
