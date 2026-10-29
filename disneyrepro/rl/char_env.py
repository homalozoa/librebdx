"""Environment builder shared by the BDX and Olaf imitation tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import terminations as mjterm
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from disneyrepro.rl import char_common as cc
from disneyrepro.rl.disney_action import DisneyJointActionCfg

# Reward weights of Table I (Grandia et al.).
BDX_WEIGHTS = dict(
  torso_pos_xy=1.0, torso_ori=1.0, lin_vel_xy=1.0, lin_vel_z=1.0, ang_vel_xy=0.5, ang_vel_z=0.5,
  leg_joint_pos=15.0, neck_joint_pos=100.0, leg_joint_vel=1.0e-3, neck_joint_vel=1.0, contact=1.0,
  torques=1.0e-3, joint_acc=2.5e-6, leg_action_rate=1.5, neck_action_rate=5.0,
  leg_action_acc=0.45, neck_action_acc=5.0, survival=20.0,
)


@dataclass
class CharSpec:
  robot_cfg: object
  command_cfg: object
  leg_joints: tuple[int, ...]
  neck_joints: tuple[int, ...]
  foot_geoms: tuple[str, str]
  fatal_geoms: tuple[str, ...]
  self_collision: tuple[str, str] | None
  disturbance_bodies: dict
  action_scale: dict
  max_deviation: dict
  sim_dt: float
  decimation: int
  weights: dict = field(default_factory=lambda: dict(BDX_WEIGHTS))
  episode_length_s: float = 20.0
  num_envs: int = 8192
  extra_actor_terms: dict = field(default_factory=dict)
  extra_rewards: dict = field(default_factory=dict)
  disturbance_scale: float = 1.0
  terrain: object = None
  viewer_body: str = "base_link"


def make_char_env_cfg(spec: CharSpec, play: bool = False) -> ManagerBasedRlEnvCfg:
  C = "ref"
  leg, neck = list(spec.leg_joints), list(spec.neck_joints)
  W = spec.weights

  actor = {
    "torso_pos_path": ObservationTermCfg(func=cc.obs_torso_pos_path, params={"command_name": C}, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "torso_rot_path": ObservationTermCfg(func=cc.obs_torso_rot_path, params={"command_name": C}, noise=Unoise(n_min=-0.02, n_max=0.02)),
    "lin_vel_b": ObservationTermCfg(func=cc.obs_root_lin_vel_b, noise=Unoise(n_min=-0.1, n_max=0.1)),
    "ang_vel_b": ObservationTermCfg(func=cc.obs_root_ang_vel_b, noise=Unoise(n_min=-0.15, n_max=0.15)),
    "joint_pos": ObservationTermCfg(func=cc.obs_measured_joint_pos),
    "joint_vel": ObservationTermCfg(func=cc.obs_joint_vel, noise=Unoise(n_min=-0.3, n_max=0.3)),
    "prev_action": ObservationTermCfg(func=cc.obs_prev_action),
    "prev_prev_action": ObservationTermCfg(func=cc.obs_prev_prev_action),
    "command": ObservationTermCfg(func=cc.obs_command, params={"command_name": C}),
    **spec.extra_actor_terms,
  }
  critic = {
    "torso_pos_path": ObservationTermCfg(func=cc.obs_torso_pos_path, params={"command_name": C}),
    "torso_rot_path": ObservationTermCfg(func=cc.obs_torso_rot_path, params={"command_name": C}),
    "lin_vel_b": ObservationTermCfg(func=cc.obs_root_lin_vel_b),
    "ang_vel_b": ObservationTermCfg(func=cc.obs_root_ang_vel_b),
    "joint_pos": ObservationTermCfg(func=cc.obs_joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=cc.obs_joint_vel),
    "prev_action": ObservationTermCfg(func=cc.obs_prev_action),
    "prev_prev_action": ObservationTermCfg(func=cc.obs_prev_prev_action),
    "command": ObservationTermCfg(func=cc.obs_command, params={"command_name": C}),
    **{k: ObservationTermCfg(func=v.func, params=v.params) for k, v in spec.extra_actor_terms.items()},
    "ref_error": ObservationTermCfg(func=cc.obs_ref_error, params={"command_name": C}),
    "friction": ObservationTermCfg(func=cc.obs_foot_friction),
  }

  def R(func, w, **params):
    return RewardTermCfg(func=func, weight=w, params=params)

  rewards = {
    "torso_pos_xy": R(cc.rew_torso_pos_xy, W["torso_pos_xy"], command_name=C),
    "torso_ori": R(cc.rew_torso_ori, W["torso_ori"], command_name=C),
    "lin_vel_xy": R(cc.rew_lin_vel_xy, W["lin_vel_xy"], command_name=C),
    "lin_vel_z": R(cc.rew_lin_vel_z, W["lin_vel_z"], command_name=C),
    "ang_vel_xy": R(cc.rew_ang_vel_xy, W["ang_vel_xy"], command_name=C),
    "ang_vel_z": R(cc.rew_ang_vel_z, W["ang_vel_z"], command_name=C),
    "leg_joint_pos": R(cc.rew_joint_pos, W["leg_joint_pos"], command_name=C, joint_ids=leg),
    "neck_joint_pos": R(cc.rew_joint_pos, W["neck_joint_pos"], command_name=C, joint_ids=neck),
    "leg_joint_vel": R(cc.rew_joint_vel, W["leg_joint_vel"], command_name=C, joint_ids=leg),
    "neck_joint_vel": R(cc.rew_joint_vel, W["neck_joint_vel"], command_name=C, joint_ids=neck),
    "contact": R(cc.rew_contact, W["contact"], command_name=C, sensor_name="feet_contact"),
    "torques": R(cc.rew_torques, W["torques"]),
    "joint_acc": R(cc.rew_joint_acc, W["joint_acc"]),
    "leg_action_rate": R(cc.rew_action_rate, W["leg_action_rate"], action_ids=leg),
    "neck_action_rate": R(cc.rew_action_rate, W["neck_action_rate"], action_ids=neck),
    "leg_action_acc": R(cc.rew_action_acc, W["leg_action_acc"], action_ids=leg),
    "neck_action_acc": R(cc.rew_action_acc, W["neck_action_acc"], action_ids=neck),
    "survival": R(cc.rew_survival, W["survival"]),
    **spec.extra_rewards,
  }
  # Weights in Table I are written with an explicit minus inside the term.
  for k in rewards:
    rewards[k].weight = abs(rewards[k].weight)

  sensors = [
    ContactSensorCfg(
      name="feet_contact",
      primary=ContactMatch(mode="geom", pattern=spec.foot_geoms, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="fatal_contact",
      primary=ContactMatch(mode="geom", pattern=spec.fatal_geoms, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  ]
  terminations = {
    "time_out": TerminationTermCfg(func=mjterm.time_out, time_out=True),
    "fatal_contact": TerminationTermCfg(func=cc.term_contact, params={"sensor_name": "fatal_contact"}),
  }
  if spec.self_collision is not None:
    sensors.append(
      ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="geom", pattern=spec.self_collision[0], entity="robot"),
        secondary=ContactMatch(mode="geom", pattern=spec.self_collision[1], entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
      )
    )
    terminations["self_collision"] = TerminationTermCfg(func=cc.term_contact, params={"sensor_name": "self_collision"})

  events = {
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=spec.foot_geoms),
        "operation": "abs",
        "ranges": (0.4, 1.2),
        "shared_random": True,
      },
    ),
    "body_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*",)), "operation": "scale", "ranges": (0.9, 1.1)},
    ),
    "torso_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(spec.viewer_body,)),
        "operation": "add",
        "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
      },
    ),
    # Reflected inertia +-20 % (App. B). Drawn per world at startup and redrawn for all
    # worlds together every 40 s of sim time: changing armature forces a model-constant
    # recomputation, which per-reset sampling triggered on almost every step (~45 % of
    # step time) for the same distribution over the batch.
    "armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "operation": "scale", "ranges": (0.8, 1.2)},
    ),
    "armature_redraw": EventTermCfg(
      mode="interval",
      interval_range_s=(40.0, 40.0),
      is_global_time=True,
      func=dr.joint_armature,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "operation": "scale", "ranges": (0.8, 1.2)},
    ),
    "disturbances": EventTermCfg(
      mode="step",
      func=cc.Disturbances(cc.table_v_groups(**spec.disturbance_bodies, scale=spec.disturbance_scale)),
    ),
  }
  if play:
    events.pop("disturbances")

  terrain = spec.terrain or TerrainEntityCfg(terrain_type="plane")
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=terrain, num_envs=spec.num_envs, sensors=tuple(sensors), entities={"robot": spec.robot_cfg}),
    observations={
      "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=not play),
      "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
    },
    actions={
      "joints": DisneyJointActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=spec.action_scale,
        use_default_offset=True,
        max_deviation=spec.max_deviation,
        physics_dt=spec.sim_dt,
        decimation=spec.decimation,
      )
    },
    commands={C: spec.command_cfg},
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=spec.viewer_body,
      distance=2.0, elevation=-10.0, azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=24,
      njmax=160,
      mujoco=MujocoCfg(timestep=spec.sim_dt, iterations=10, ls_iterations=20),
    ),
    decimation=spec.decimation,
    episode_length_s=spec.episode_length_s if not play else 1e9,
  )
  return cfg


def make_ppo_cfg(name: str, max_iterations: int = 20000) -> RslRlOnPolicyRunnerCfg:
  """PPO hyper-parameters of Table IV (Grandia et al.)."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 512, 512),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 512, 512), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.0,
      desired_kl=0.01,
      max_grad_norm=1.0,
      clip_param=0.2,
    ),
    experiment_name=name,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    save_interval=500,
    logger="tensorboard",
    upload_model=False,
  )
