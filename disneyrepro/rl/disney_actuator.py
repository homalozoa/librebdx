"""Actuator model of Grandia et al., RSS 2024, Appendix B (Eqs. 14-20, Table VI).

  q~      = q + eps_q                                  (encoder offset, per episode)
  tau_m   = kP (a - q~) - kD qdot                      (quasi-direct drives, Eq. 14)
  tau_m   = kP (a - q~) + kD d/dt (a - q~)             (Dynamixel, Eq. 16)
  tau_f   = mu_s tanh(qdot / qdot_s) + mu_d qdot       (friction, Eq. 17)
  tau     = clamp(tau_m, tau_lo(qdot), tau_hi(qdot)) - tau_f   (Eq. 18)
  q_meas  = q~ + 0.5 b tanh(tau_m / tau_b) + N(0, (s0 + s1 |qdot|)^2)  (Eq. 19-20)

The torque limit is constant (tau_max) for braking and below qdot_taumax and
ramps linearly to zero at qdot_max when motoring. The reflected inertia I_m is
set as joint armature (randomised by a separate event).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg


@dataclass(frozen=True)
class MotorParams:
  kp: float
  kd: float
  tau_max: float
  qd_taumax: float
  qd_max: float
  mu_s: float
  mu_d: float
  b_min: float
  b_max: float
  eps_max: float
  sigma0: float
  sigma1: float
  armature: float
  error_derivative: bool = False


# Table VI of the paper.
UNITREE_A1 = MotorParams(15.0, 0.6, 34.0, 7.4, 20.0, 0.45, 0.023, 0.005, 0.015, 0.02, 1.80e-4, 3.61e-5, 0.011)
UNITREE_GO1 = MotorParams(10.0, 0.3, 23.7, 10.6, 28.8, 0.15, 0.016, 0.002, 0.005, 0.02, 1.89e-4, 5.47e-5, 0.0043)
DYNAMIXEL_XH540 = MotorParams(5.0, 0.2, 4.8, 0.2, 7.0, 0.05, 0.009, 0.002, 0.005, 0.02, 4.31e-4, 2.43e-5, 0.0058, True)


@dataclass(kw_only=True)
class DisneyActuatorCfg(IdealPdActuatorCfg):
  motor: MotorParams
  physics_dt: float
  qd_s: float = 0.1
  """Friction tanh activation velocity [rad/s] (not reported; assumed)."""
  tau_b: float = 1.0
  """Backlash tanh activation torque [Nm] (not reported; assumed)."""
  randomize: dict = field(default_factory=lambda: {"kp": 0.1, "kd": 0.1, "mu": 0.3, "tau_max": 0.05})
  """Per-episode relative uniform randomisation of the model parameters."""

  def build(self, entity, target_ids, target_names):
    return DisneyActuator(self, entity, target_ids, target_names)


def make_cfg(names, motor: MotorParams, physics_dt: float, **kw) -> DisneyActuatorCfg:
  # The MuJoCo motor's ctrlrange must admit tau_max plus friction compensation.
  limit = motor.tau_max + motor.mu_s + motor.mu_d * motor.qd_max
  return DisneyActuatorCfg(
    target_names_expr=tuple(names),
    stiffness=motor.kp,
    damping=motor.kd,
    effort_limit=limit,
    armature=motor.armature,
    motor=motor,
    physics_dt=physics_dt,
    **kw,
  )


class DisneyActuator(IdealPdActuator):
  def initialize(self, mj_model, model, data, device):
    super().initialize(mj_model, model, data, device)
    n, j = data.nworld, len(self._target_names)
    mp = self.cfg.motor
    z = lambda v: torch.full((n, j), float(v), device=device)
    self.kp, self.kd = z(mp.kp), z(mp.kd)
    self.tau_max, self.mu_s, self.mu_d = z(mp.tau_max), z(mp.mu_s), z(mp.mu_d)
    self.eps_q = z(0.0)
    self.backlash = z(0.5 * (mp.b_min + mp.b_max))
    self.tau_m = z(0.0)
    self.prev_err = z(0.0)
    self._has_prev = torch.zeros(n, dtype=torch.bool, device=device)
    self.applied = z(0.0)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    mp, rnd = self.cfg.motor, self.cfg.randomize
    shape = self.kp[env_ids].shape

    def u(scale):
      return 1.0 + (2 * torch.rand(shape, device=self.kp.device) - 1) * scale

    self.kp[env_ids] = mp.kp * u(rnd["kp"])
    self.kd[env_ids] = mp.kd * u(rnd["kd"])
    self.tau_max[env_ids] = mp.tau_max * u(rnd["tau_max"])
    self.mu_s[env_ids] = mp.mu_s * u(rnd["mu"])
    self.mu_d[env_ids] = mp.mu_d * u(rnd["mu"])
    self.eps_q[env_ids] = (2 * torch.rand(shape, device=self.kp.device) - 1) * mp.eps_max
    self.backlash[env_ids] = mp.b_min + torch.rand(shape, device=self.kp.device) * (mp.b_max - mp.b_min)
    self.tau_m[env_ids] = 0.0
    self._has_prev[env_ids] = False

  def torque_limits(self, qd):
    mp = self.cfg.motor
    span = mp.qd_max - mp.qd_taumax
    hi = self.tau_max * torch.clamp((mp.qd_max - qd) / span, 0.0, 1.0)
    lo = -self.tau_max * torch.clamp((mp.qd_max + qd) / span, 0.0, 1.0)
    return lo, hi

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    mp = self.cfg.motor
    q_t = cmd.pos + self.eps_q
    err = cmd.position_target - q_t
    if mp.error_derivative:
      prev = torch.where(self._has_prev[:, None], self.prev_err, err)
      tau_m = self.kp * err + self.kd * (err - prev) / self.cfg.physics_dt
      self.prev_err = err
      self._has_prev[:] = True
    else:
      tau_m = self.kp * err - self.kd * cmd.vel
    lo, hi = self.torque_limits(cmd.vel)
    tau_f = self.mu_s * torch.tanh(cmd.vel / self.cfg.qd_s) + self.mu_d * cmd.vel
    self.tau_m = tau_m
    self.applied = torch.clamp(tau_m, lo, hi) - tau_f
    return self.applied

  def measured_position(self, q, qd):
    """Encoder reading model (Eq. 19-20)."""
    mp = self.cfg.motor
    sigma = mp.sigma0 + mp.sigma1 * qd.abs()
    return q + self.eps_q + 0.5 * self.backlash * torch.tanh(self.tau_m / self.cfg.tau_b) + sigma * torch.randn_like(q)
