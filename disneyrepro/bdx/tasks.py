"""BDX character tasks: perpetual standing, periodic walking, episodic motions."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from disneyrepro.bdx import robot_cfg as rc
from disneyrepro.bdx.standing_ref import CMD_HI, CMD_LO, Nominal
from disneyrepro.common.fitnet import FitNet
from disneyrepro.kin.torch_kin import euler_zyx_to_mat
from disneyrepro.rl import char_common as cc
from disneyrepro.rl.char_env import CharSpec, make_char_env_cfg, make_ppo_cfg

DATA = Path(__file__).parent / "data"

# Entity joint order equals the MJCF order.
JOINTS = [
  "Left_Hip_Yaw", "Left_Hip_Roll", "Left_Hip_Pitch", "Left_Knee", "Left_Ankle",
  "Right_Hip_Yaw", "Right_Hip_Roll", "Right_Hip_Pitch", "Right_Knee", "Right_Ankle",
  "Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll",
]
LEG_IDS = tuple(range(10))
NECK_IDS = tuple(range(10, 14))


# ----------------------------------------------------------------------------
# Standing (perpetual)
# ----------------------------------------------------------------------------


@dataclass(kw_only=True)
class StandingRefCfg(cc.CharacterRefCfg):
  net_path: str = str(DATA / "standing_net.pt")
  p_all_nominal: float = 0.15
  p_dim_active: float = 0.7
  tau_cmd: float = 0.25
  full_range_time: float = 0.8
  tau_path: float = 2.0

  def build(self, env):
    return StandingRef(self, env)


class StandingRef(cc.CharacterRef):
  cfg: StandingRefCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    dev = self.device
    self.net = FitNet.load(cfg.net_path, device=dev)
    for p in self.net.parameters():
      p.requires_grad_(False)
    assert list(self.robot.joint_names) == JOINTS, self.robot.joint_names
    self.lo = torch.tensor(CMD_LO, device=dev)
    self.hi = torch.tensor(CMD_HI, device=dev)
    self.g = torch.zeros(self.num_envs, 8, device=dev)
    self.g_target = torch.zeros_like(self.g)
    self.h0 = Nominal().torso_height

  @property
  def command(self):
    return 2 * (self.g - self.lo) / (self.hi - self.lo) - 1

  def _resample_command(self, env_ids):
    if getattr(self, "external", False):
      return
    n = len(env_ids)
    u = self.lo + torch.rand(n, 8, device=self.device) * (self.hi - self.lo)
    active = torch.rand(n, 8, device=self.device) < self.cfg.p_dim_active
    nominal = torch.rand(n, 1, device=self.device) < self.cfg.p_all_nominal
    self.g_target[env_ids] = torch.where(active & ~nominal, u, torch.zeros_like(u))

  def set_command(self, g, env_ids=slice(None)):
    """External (animation engine) command override; disables resampling."""
    self.external = True
    self.g_target[env_ids] = g

  def _advance(self, dt):
    a = 1 - math.exp(-dt / self.cfg.tau_cmd)
    rate = (self.hi - self.lo) / self.cfg.full_range_time * dt
    self.g = self.g + torch.clamp(a * (self.g_target - self.g), -rate, rate)
    # Path frame slowly converges towards the middle / heading of the feet.
    mid, yaw = self.feet_mid_and_yaw()
    b = 1 - math.exp(-dt / self.cfg.tau_path)
    self.pf_pos = self.pf_pos + b * (mid - self.pf_pos)
    self.pf_yaw = self.pf_yaw + b * cc.wrap(yaw - self.pf_yaw)
    self.project_path_frame()

  def _init_episode(self, env_ids):
    self.g[env_ids] = self.g_target[env_ids]
    self.pf_pos[env_ids] = self._env.scene.env_origins[env_ids, :2]
    lo, hi = self.cfg.init_yaw_range
    self.pf_yaw[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)

  def _reference_path(self, env_ids):
    g = self.g[env_ids]
    x = self.net(g)
    root = torch.cat([x[:, :2], (self.h0 + g[:, 4])[:, None]], -1)
    R = euler_zyx_to_mat(g[:, 5], g[:, 6], g[:, 7])
    return root, R, x[:, 2:], torch.ones(len(g), 2, device=self.device)


def _bdx_spec(command_cfg, num_envs=8192, action_scale=None, **kw) -> CharSpec:
  nominal = rc.nominal_joint_pos()
  return CharSpec(
    robot_cfg=rc.get_bdx_robot_cfg(nominal),
    command_cfg=command_cfg,
    leg_joints=LEG_IDS,
    neck_joints=NECK_IDS,
    foot_geoms=("left_foot_sole", "right_foot_sole"),
    fatal_geoms=("head_collision", "base_collision"),
    self_collision=("head_collision", "base_collision"),
    disturbance_bodies=dict(
      hips=("Left_Hip_Roll_Connector", "Right_Hip_Roll_Connector"),
      feet=("Left_Foot", "Right_Foot"),
      pelvis="base_link",
      head="Head_Base_Plate",
    ),
    action_scale=action_scale or rc.ACTION_SCALE,
    max_deviation=rc.MAX_DEVIATION,
    sim_dt=rc.SIM_DT,
    decimation=rc.DECIMATION,
    num_envs=num_envs,
    **kw,
  )


def obs_ref_q(env, command_name):
  c = env.command_manager.get_term(command_name)
  return c.ref_q - c.robot.data.default_joint_pos


def bdx_standing_env_cfg(play=False, num_envs=8192, disturbances=True, ref_obs=False, nominal_only=False):
  from mjlab.managers.observation_manager import ObservationTermCfg

  cmd = StandingRefCfg(resampling_time_range=(1.0, 4.0), p_all_nominal=1.0 if nominal_only else 0.15)
  extra = {"ref_q": ObservationTermCfg(func=obs_ref_q, params={"command_name": "ref"})} if ref_obs else {}
  cfg = make_char_env_cfg(_bdx_spec(cmd, num_envs=num_envs, extra_actor_terms=extra), play=play)
  if not disturbances:
    cfg.events.pop("disturbances", None)
  return cfg


# ----------------------------------------------------------------------------
# Walking (periodic)
# ----------------------------------------------------------------------------

# g_peri = (dh_head, yaw_head, pitch_head, roll_head, vx, vy, wz)  (Eq. 6)
WALK_LO = (-0.03, -0.6, -0.2, -0.3, -0.3, -0.4, -1.8)
WALK_HI = (0.01, 0.6, 0.4, 0.3, 0.7, 0.4, 1.8)  # Sec. VII-A: 0.7 m/s, 0.4 m/s, 1.8 rad/s


@dataclass(kw_only=True)
class WalkingRefCfg(cc.CharacterRefCfg):
  # DISNEY_WALK_TABLE=calm selects the calm gait (cadence grows with speed).
  table_path: str = str(DATA / ("walk_table_%s.npz" % os.environ["DISNEY_WALK_TABLE"] if os.environ.get("DISNEY_WALK_TABLE") else "walk_table.npz"))
  p_zero_vel: float = 0.1
  p_head_active: float = 0.5
  p_fast: float = 0.25
  sample_vy: float = 0.4   # full command range, as in the paper
  sample_wz: float = 1.8
  """BDX-R's hip roll / yaw ranges (+-0.26 rad) limit lateral and turning speed."""
  tau_cmd: float = 0.3
  vel_accel: tuple = (1.0, 1.0, 3.0)
  """Max rate of change of the path velocity command (m/s^2, m/s^2, rad/s^2)."""
  pf_max_dist: float = 0.15
  variant: str = "paper"
  """paper | vel_only | phase_contact | ref_input (Sec. VII-B alternatives)."""

  def build(self, env):
    return WalkingRef(self, env)


class WalkingRef(cc.CharacterRef):
  cfg: WalkingRefCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    from disneyrepro.rl.walk_table import WalkTable

    dev = self.device
    self.table = WalkTable(cfg.table_path, dev)
    assert self.table.joint_names == JOINTS
    self.lo = torch.tensor(WALK_LO, device=dev)
    self.hi = torch.tensor(WALK_HI, device=dev)
    self.g = torch.zeros(self.num_envs, 7, device=dev)
    self.g_target = torch.zeros_like(self.g)
    self.phi = torch.zeros(self.num_envs, device=dev)
    from disneyrepro.bdx.standing_ref import NECK_PER_M

    self.neck_per_m = NECK_PER_M

  @property
  def command(self):
    return 2 * (self.g - self.lo) / (self.hi - self.lo) - 1

  def _update_command(self, env_ids):
    super()._update_command(env_ids)
    if self.cfg.variant in ("vel_only", "phase_contact"):
      # Torso rewards track the commands directly (Hwangbo et al. style).
      from disneyrepro.rl.char_common import yaw_mat

      v = self.g[:, 4:]
      c, s = torch.cos(self.pf_yaw), torch.sin(self.pf_yaw)
      self.ref_pos[:, :2] = self.pf_pos
      self.ref_R = yaw_mat(self.pf_yaw)
      self.ref_lin_vel = torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], torch.zeros_like(c)], -1)
      self.ref_ang_vel = torch.stack([torch.zeros_like(c), torch.zeros_like(c), v[:, 2]], -1)

  def _init_velocities(self, env_ids, dt=0.02):
    from mjlab.utils.lab_api.math import matrix_from_quat

    from disneyrepro.kin.torch_kin import so3_log
    from disneyrepro.rl.char_common import yaw_mat

    v = self.g[env_ids, 4:]
    phi = self.phi[env_ids]
    a = self.table.query(v, phi)
    b = self.table.query(v, (phi + a["rate"] * dt) % 1.0)
    pa = torch.cat([a["root_xy"], a["root_z"]], -1)
    pb = torch.cat([b["root_xy"], b["root_z"]], -1)
    Ra, Rb = matrix_from_quat(a["root_quat"]), matrix_from_quat(b["root_quat"])
    Rpf = yaw_mat(self.pf_yaw[env_ids])
    lin = (Rpf @ ((pb - pa) / dt).unsqueeze(-1)).squeeze(-1)
    c, s = torch.cos(self.pf_yaw[env_ids]), torch.sin(self.pf_yaw[env_ids])
    lin[:, 0] += c * v[:, 0] - s * v[:, 1]
    lin[:, 1] += s * v[:, 0] + c * v[:, 1]
    ang = (Rpf @ (so3_log(Rb @ Ra.transpose(-1, -2)) / dt).unsqueeze(-1)).squeeze(-1)
    ang[:, 2] += v[:, 2]
    return lin, ang, (b["q"] - a["q"]) / dt

  def path_velocity(self):
    v = self.g[:, 4:]
    c, s = torch.cos(self.pf_yaw), torch.sin(self.pf_yaw)
    return torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], -1), v[:, 2]

  def future_q(self, dt):
    v = self.g[:, 4:]
    phi = (self.phi + self.table.query(v, self.phi)["rate"] * dt) % 1.0
    return self.table.query(v, phi)["q"] - self.robot.data.default_joint_pos

  def phase_features(self):
    a = 2 * math.pi * self.phi
    return torch.stack([torch.sin(a), torch.cos(a), torch.sin(2 * a), torch.cos(2 * a)], -1)

  def _resample_command(self, env_ids):
    if getattr(self, "external", False):
      return
    n = len(env_ids)
    u = self.lo + torch.rand(n, 7, device=self.device) * (self.hi - self.lo)
    head_on = torch.rand(n, 1, device=self.device) < self.cfg.p_head_active
    u[:, :4] = torch.where(head_on, u[:, :4], torch.zeros_like(u[:, :4]))
    # Velocity dims: sometimes zero, sometimes a single axis, otherwise mixed.
    r = torch.rand(n, 1, device=self.device)
    zero = r < self.cfg.p_zero_vel
    single = (r >= self.cfg.p_zero_vel) & (r < 0.45)
    axis = torch.randint(0, 3, (n,), device=self.device)
    mask = torch.nn.functional.one_hot(axis, 3).float()
    vel = u[:, 4:]
    vel = torch.where(single, vel * mask, vel)
    vel = torch.where(zero, torch.zeros_like(vel), vel)
    # Fast forward walking is rare under uniform sampling (and the ramp-up eats
    # into the segment); over-sample it so the top speed is trained as well.
    fast = (torch.rand(n, device=self.device) < self.cfg.p_fast) & ~zero[:, 0]
    vel[:, 0] = torch.where(fast, 0.45 + 0.25 * torch.rand(n, device=self.device), vel[:, 0])
    vel[:, 1] = vel[:, 1] * (self.cfg.sample_vy / WALK_HI[5])
    vel[:, 2] = vel[:, 2] * (self.cfg.sample_wz / WALK_HI[6])
    u[:, 4:] = vel
    self.g_target[env_ids] = u

  def set_command(self, g, env_ids=slice(None)):
    self.external = True
    self.g_target[env_ids] = g

  def _advance(self, dt):
    a = 1 - math.exp(-dt / self.cfg.tau_cmd)
    rate = torch.tensor((0.04 / 0.3, 0.6 / 0.3, 0.6 / 0.3, 0.6 / 0.3) + self.cfg.vel_accel, device=self.device) * dt
    self.g = self.g + torch.clamp(a * (self.g_target - self.g), -rate, rate)
    v = self.g[:, 4:]
    # Phase signal driven by the generator's phase rate (Eq. 3).
    self.phi = (self.phi + self.table.query(v, self.phi)["rate"] * dt) % 1.0
    # Path frame integrates the commanded path velocity.
    c, s = torch.cos(self.pf_yaw), torch.sin(self.pf_yaw)
    self.pf_pos = self.pf_pos + dt * torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], -1)
    self.pf_yaw = cc.wrap(self.pf_yaw + dt * v[:, 2])
    self.project_path_frame()

  def _init_episode(self, env_ids):
    self.g[env_ids] = self.g_target[env_ids]
    self.phi[env_ids] = torch.rand(len(env_ids), device=self.device)
    self.pf_pos[env_ids] = self._env.scene.env_origins[env_ids, :2]
    lo, hi = self.cfg.init_yaw_range
    self.pf_yaw[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)

  def _reference_path(self, env_ids):
    from mjlab.utils.lab_api.math import matrix_from_quat

    g = self.g[env_ids]
    ref = self.table.query(g[:, 4:], self.phi[env_ids])
    root = torch.cat([ref["root_xy"], ref["root_z"]], -1)
    R = matrix_from_quat(ref["root_quat"])
    q = ref["q"].clone()
    # Head commands are offsets relative to the nominal head motion of the walk cycle.
    q[:, 10] = q[:, 10] - self.neck_per_m * g[:, 0]
    q[:, 12] = q[:, 12] + g[:, 1]
    q[:, 11] = q[:, 11] + g[:, 2]
    q[:, 13] = q[:, 13] + g[:, 3]
    lim = self.robot.data.soft_joint_pos_limits[0]
    q = torch.clamp(q, lim[:, 0], lim[:, 1])
    contact = (ref["contact"] > 0.5).float()
    return root, R, q, contact


def obs_phase(env, command_name):
  return env.command_manager.get_term(command_name).phase_features()


def obs_ref_now_future(env, command_name):
  c = env.command_manager.get_term(command_name)
  return torch.cat([c.ref_q - c.robot.data.default_joint_pos, c.future_q(0.1), c.future_q(0.2)], -1)


def bdx_walking_env_cfg(play=False, num_envs=8192, variant="paper"):
  from mjlab.managers.observation_manager import ObservationTermCfg

  cmd = WalkingRefCfg(resampling_time_range=(2.0, 5.0), variant=variant)
  if variant in ("paper", "phase_contact"):
    extra = {"phase": ObservationTermCfg(func=obs_phase, params={"command_name": "ref"})}
  elif variant == "ref_input":
    extra = {"ref_poses": ObservationTermCfg(func=obs_ref_now_future, params={"command_name": "ref"})}
  else:
    extra = {}
  spec = _bdx_spec(cmd, num_envs=num_envs, extra_actor_terms=extra, action_scale=rc.ACTION_SCALE_REF)
  # Walking uses the stronger torso terms the same group used for Olaf's
  # walking policy (Mueller et al., Table III): with Table I's weights the
  # policy settles in a step-in-place local optimum on our compute budget.
  # DISNEY_TABLE1=1 keeps Table I's weights unchanged (the paper's setting).
  if __import__("os").environ.get("DISNEY_TABLE1") != "1":
    spec.weights.update(torso_pos_xy=4.0, torso_ori=1.5, lin_vel_xy=2.5)
  if variant in ("vel_only", "phase_contact"):
    for k in ("leg_joint_pos", "leg_joint_vel"):
      spec.weights[k] = 0.0
    if variant == "vel_only":
      spec.weights["contact"] = 0.0
  import os

  if os.environ.get("DISNEY_SURV"):
    spec.weights["survival"] = float(os.environ["DISNEY_SURV"])
  cfg = make_char_env_cfg(spec, play=play)
  if os.environ.get("DISNEY_SURV_CURRIC"):
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    w0, w1, i0, i1 = (float(x) for x in os.environ["DISNEY_SURV_CURRIC"].split(","))
    cfg.curriculum = {
      "survival_weight": CurriculumTermCfg(
        func=cc.curriculum_reward_weight, params=dict(term="survival", w0=w0, w1=w1, it0=i0, it1=i1)
      )
    }
  if os.environ.get("DISNEY_TERRAIN") == "1":
    # Walking policies are additionally trained on randomized terrain (Sec. V-E):
    # flat, mildly rough and small discrete obstacles (scaled to a 0.66 m robot).
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.terrains.config import discrete_obstacles, flat, random_rough
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    cfg.scene.terrain = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=10.0, num_rows=8, num_cols=8,
        sub_terrains={
          "flat": flat(proportion=0.4),
          "rough": random_rough(proportion=0.3, noise_range=(0.0, 0.015), noise_step=0.005),
          "obstacles": discrete_obstacles(proportion=0.3, obstacle_height_range=(0.005, 0.025), obstacle_width_range=(0.1, 0.4), num_obstacles=60),
        },
      ),
      max_init_terrain_level=None,
    )
  if os.environ.get("DISNEY_NODIST") == "1":
    cfg.events.pop("disturbances", None)
    cfg.observations["actor"].enable_corruption = False
  for k in [k for k, r in cfg.rewards.items() if r.weight == 0.0]:
    cfg.rewards.pop(k)
  return cfg


# ----------------------------------------------------------------------------
# Episodic motions
# ----------------------------------------------------------------------------

N_BASIS = 50


@dataclass(kw_only=True)
class EpisodicRefCfg(cc.CharacterRefCfg):
  clip_path: str = ""
  random_start: bool = True
  init_yaw_range: tuple = (-math.pi, math.pi)

  def build(self, env):
    return EpisodicRef(self, env)


class EpisodicRef(cc.CharacterRef):
  """Monotonic phase; path-frame trajectory relative to the start is part of the clip (Eq. 4)."""

  cfg: EpisodicRefCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    import numpy as np

    d = np.load(cfg.clip_path)
    dev = self.device
    assert [str(n) for n in d["joint_names"]] == JOINTS
    self.fps = float(d["fps"])
    t = lambda k: torch.tensor(d[k], dtype=torch.float32, device=dev)
    self.root_pos, self.root_quat, self.q, self.contact = t("root_pos"), t("root_quat"), t("q"), t("contact")
    self.n = self.q.shape[0]
    self.T = (self.n - 1) / self.fps
    self.emph = tuple(float(x) for x in d["emphasis"])
    self.t = torch.zeros(self.num_envs, device=dev)
    self.centres = torch.linspace(0, 1, N_BASIS, device=dev)

  @property
  def phase(self):
    return torch.clamp(self.t / self.T, 0.0, 1.0)

  @property
  def command(self):
    # Gaussian basis features of the phase (App. A), sigma = 1/N.
    return torch.exp(-0.5 * ((self.phase[:, None] - self.centres[None]) * N_BASIS) ** 2)

  def in_emphasis(self):
    a, b = self.emph
    return ((self.t >= a) & (self.t <= b)).float()

  def _resample_command(self, env_ids):
    pass

  def _init_velocities(self, env_ids, dt=0.02):
    from mjlab.utils.lab_api.math import matrix_from_quat

    from disneyrepro.kin.torch_kin import so3_log
    from disneyrepro.rl.char_common import yaw_mat

    f = torch.clamp(self.t[env_ids] * self.fps, 0, self.n - 2)
    i0 = torch.floor(f).long()
    Rpf = yaw_mat(self.pf_yaw[env_ids])
    lin = (Rpf @ ((self.root_pos[i0 + 1] - self.root_pos[i0]) * self.fps).unsqueeze(-1)).squeeze(-1)
    Ra, Rb = matrix_from_quat(self.root_quat[i0]), matrix_from_quat(self.root_quat[i0 + 1])
    ang = (Rpf @ (so3_log(Rb @ Ra.transpose(-1, -2)) * self.fps).unsqueeze(-1)).squeeze(-1)
    return lin, ang, (self.q[i0 + 1] - self.q[i0]) * self.fps

  def _advance(self, dt):
    self.t = self.t + dt

  def _init_episode(self, env_ids):
    if self.cfg.random_start:
      self.t[env_ids] = torch.rand(len(env_ids), device=self.device) * self.T * 0.9
    else:
      self.t[env_ids] = 0.0
    lo, hi = self.cfg.init_yaw_range
    yaw = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
    self.pf_yaw[env_ids] = yaw
    self.pf_pos[env_ids] = self._env.scene.env_origins[env_ids, :2]

  def _reference_path(self, env_ids):
    from mjlab.utils.lab_api.math import matrix_from_quat

    f = torch.clamp(self.t[env_ids] * self.fps, 0, self.n - 1)
    i0 = torch.floor(f).long().clamp(max=self.n - 2)
    e = (f - i0)[:, None]
    lerp = lambda A: (1 - e) * A[i0] + e * A[i0 + 1]
    quat = lerp(self.root_quat)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    contact = (lerp(self.contact) > 0.5).float()
    return lerp(self.root_pos), matrix_from_quat(quat), lerp(self.q), contact


def rew_emph_ang_vel(env, command_name):
  c = env.command_manager.get_term(command_name)
  return c.in_emphasis() * (cc.rew_ang_vel_xy(env, command_name) + cc.rew_ang_vel_z(env, command_name))


def rew_emph_jump(env, command_name):
  """Eq. 13 for the jump (App. A): inside the emphasis window, add a torso-height
  reward and extra weight on torso orientation and on the contact reward.

  The contact reward matches the reference contact schedule, so during the flight
  frames it pays for both feet being off the ground; without the extra weight the
  policy "cheats by keeping the toes on the ground".
  """
  c = env.command_manager.get_term(command_name)
  h = torch.exp(-((c.robot.data.root_link_pos_w[:, 2] - c.ref_pos[:, 2]) / JUMP_SIGMA_H) ** 2)
  ori = cc.rew_torso_ori(env, command_name)
  contact = cc.rew_contact(env, command_name, "feet_contact")
  return c.in_emphasis() * (JUMP_W_HEIGHT * h + JUMP_W_ORI * ori + JUMP_W_CONTACT * contact)


# Extra weights inside the jump window (Eq. 13: w = w0 + I[window] * w_extra). The paper gives no
# values. With a 7 cm height kernel and weights 3/3/4 the window paid <5% of the survival reward
# and the policy never left the ground (P1jump, 12.5k iterations). The kernel must tell a 6 cm rise from none.
JUMP_SIGMA_H = 0.015
JUMP_W_HEIGHT, JUMP_W_ORI, JUMP_W_CONTACT = 12.0, 3.0, 12.0


def bdx_episodic_env_cfg(name, play=False, num_envs=8192):
  from mjlab.managers.reward_manager import RewardTermCfg

  path = str(DATA / f"episodic_{name}.npz")
  cmd = EpisodicRefCfg(resampling_time_range=(1e9, 1e9), clip_path=path, random_start=not play)
  extra = {}
  if name == "excited":
    extra["emph_ang_vel"] = RewardTermCfg(func=rew_emph_ang_vel, weight=2.0, params={"command_name": "ref"})
  if name == "jump":
    extra["emph_jump"] = RewardTermCfg(func=rew_emph_jump, weight=1.0, params={"command_name": "ref"})
  import numpy as np

  T = (np.load(path)["q"].shape[0] - 1) / 50.0
  spec = _bdx_spec(cmd, num_envs=num_envs, extra_rewards=extra, episode_length_s=T + 0.4)
  import os

  if os.environ.get("DISNEY_SURV"):
    spec.weights["survival"] = float(os.environ["DISNEY_SURV"])
  cfg = make_char_env_cfg(spec, play=play)
  if play:
    cfg.episode_length_s = T + 0.4
  return cfg


EPISODIC = ("happy_dance", "excited", "jump", "tantrum")


def register():
  from mjlab.tasks.registry import register_mjlab_task

  register_mjlab_task(
    task_id="Disney-BDX-Standing",
    env_cfg=bdx_standing_env_cfg(),
    play_env_cfg=bdx_standing_env_cfg(play=True, num_envs=16),
    rl_cfg=make_ppo_cfg("bdx_standing", max_iterations=20000),
  )
  register_mjlab_task(
    task_id="Disney-BDX-Walking",
    env_cfg=bdx_walking_env_cfg(),
    play_env_cfg=bdx_walking_env_cfg(play=True, num_envs=16),
    rl_cfg=make_ppo_cfg("bdx_walking", max_iterations=20000),
  )
  for variant in ("vel_only", "phase_contact", "ref_input"):
    register_mjlab_task(
      task_id=f"Disney-BDX-Walking-{variant}",
      env_cfg=bdx_walking_env_cfg(variant=variant),
      play_env_cfg=bdx_walking_env_cfg(play=True, num_envs=16, variant=variant),
      rl_cfg=make_ppo_cfg(f"bdx_walking_{variant}", max_iterations=20000),
    )
  for name in EPISODIC:
    register_mjlab_task(
      task_id=f"Disney-BDX-Episodic-{name}",
      env_cfg=bdx_episodic_env_cfg(name),
      play_env_cfg=bdx_episodic_env_cfg(name, play=True, num_envs=16),
      rl_cfg=make_ppo_cfg(f"bdx_episodic_{name}", max_iterations=20000),
    )
  register_mjlab_task(
    task_id="Disney-BDX-Standing-Nominal",
    env_cfg=bdx_standing_env_cfg(nominal_only=True, disturbances=False),
    play_env_cfg=bdx_standing_env_cfg(play=True, num_envs=16, nominal_only=True),
    rl_cfg=make_ppo_cfg("bdx_standing_nominal", max_iterations=20000),
  )
  register_mjlab_task(
    task_id="Disney-BDX-Standing-RefObs",
    env_cfg=bdx_standing_env_cfg(ref_obs=True),
    play_env_cfg=bdx_standing_env_cfg(play=True, num_envs=16, ref_obs=True),
    rl_cfg=make_ppo_cfg("bdx_standing_refobs", max_iterations=20000),
  )
  register_mjlab_task(
    task_id="Disney-BDX-Standing-NoDist",
    env_cfg=bdx_standing_env_cfg(disturbances=False),
    play_env_cfg=bdx_standing_env_cfg(play=True, num_envs=16),
    rl_cfg=make_ppo_cfg("bdx_standing_nodist", max_iterations=20000),
  )
