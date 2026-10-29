"""Policy action -> joint setpoints, as in Grandia et al. Sec. V-D and App. A.

* a = 0 maps to the nominal joint positions, a = 1 to the expected range per
  joint (fixed linear output transform),
* the 50 Hz setpoints are up-sampled to the actuator rate with a first-order
  hold (linear interpolation between the previous and current setpoint) and a
  first-order low-pass filter (37.5 Hz cut-off),
* setpoints are clipped to a maximum deviation from the measured joint
  position, large enough that the actuator can still reach peak torque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class DisneyJointActionCfg(JointPositionActionCfg):
  max_deviation: dict | float = 3.0
  """Max |setpoint - q| per joint (regex dict or float)."""
  cutoff_hz: float = 37.5
  physics_dt: float = 1.0 / 600.0
  decimation: int = 12

  def build(self, env):
    return DisneyJointAction(self, env)


class DisneyJointAction(JointPositionAction):
  def __init__(self, cfg: DisneyJointActionCfg, env):
    super().__init__(cfg, env)
    from mjlab.utils.lab_api.string import resolve_matching_names_values

    if isinstance(cfg.max_deviation, dict):
      self._max_dev = torch.full((self.action_dim,), 3.0, device=self.device)
      idx, _, vals = resolve_matching_names_values(cfg.max_deviation, self._target_names)
      self._max_dev[idx] = torch.tensor(vals, device=self.device)
    else:
      self._max_dev = torch.full((self.action_dim,), float(cfg.max_deviation), device=self.device)
    self._alpha = 1.0 - math.exp(-2 * math.pi * cfg.cutoff_hz * cfg.physics_dt)
    n = self.num_envs
    default = self._entity.data.default_joint_pos[:, self._target_ids]
    self._sp_prev = default.clone()
    self._sp_cur = default.clone()
    self._filtered = default.clone()
    self._substep = 0
    self._fresh = torch.ones(n, dtype=torch.bool, device=self.device)

  def process_actions(self, actions):
    super().process_actions(actions)
    self._sp_prev = torch.where(self._fresh[:, None], self._processed_actions, self._sp_cur)
    self._filtered = torch.where(self._fresh[:, None], self._processed_actions, self._filtered)
    self._sp_cur = self._processed_actions.clone()
    self._fresh[:] = False
    self._substep = 0

  def apply_actions(self):
    self._substep += 1
    s = min(self._substep / self.cfg.decimation, 1.0)
    foh = self._sp_prev + s * (self._sp_cur - self._sp_prev)
    self._filtered = self._filtered + self._alpha * (foh - self._filtered)
    q = self._entity.data.joint_pos[:, self._target_ids]
    target = torch.clamp(self._filtered, q - self._max_dev, q + self._max_dev)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._fresh[env_ids] = True
