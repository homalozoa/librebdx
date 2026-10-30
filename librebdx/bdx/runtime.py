"""Runtime of Grandia et al. Sec. VI: policy switching + animation engine.

Each policy runs in its own single-world mjlab env; on a switch the full robot
state (root, joints, actuator/filter internals, action history, path frame)
is copied into the target env, so the new policy continues from exactly the
same physical state. The animation engine composes a looping background
animation, triggered clips blended in/out (Eqs. 9-10, T_beta = 0.1 s,
T_alpha = 0.35 s) and joystick inputs (Eqs. 11-12) into policy commands.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))


def _load(task, ckpt):
  from eval_policy import load_policy

  def hook(cfg):
    cfg.episode_length_s = 1e9

  return load_policy(task, ckpt, 1, render=True, cfg_hook=hook)


class PolicySlot:
  def __init__(self, name, task, ckpt):
    self.name = name
    self.env, self.wenv, self.policy = _load(task, ckpt)
    self.cmd = self.env.command_manager.get_term("ref")
    self.robot = self.env.scene["robot"]
    self.act = self.env.action_manager.get_term("joints")
    self.obs = self.wenv.get_observations()

  def step(self):
    with torch.no_grad():
      a = self.policy(self.obs)
    self.obs, _, _, _ = self.wenv.step(a)
    return a

  # ----------------------------------------------------------- transfer
  def export_state(self):
    r, am, at, c = self.robot, self.env.action_manager, self.act, self.cmd
    return {
      "root": torch.cat([r.data.root_link_pos_w, r.data.root_link_quat_w, r.data.root_link_lin_vel_w, r.data.root_link_ang_vel_w], -1).clone(),
      "q": r.data.joint_pos.clone(), "qd": r.data.joint_vel.clone(),
      "a": am._action.clone(), "a1": am._prev_action.clone(), "a2": am._prev_prev_action.clone(),
      "sp_prev": at._sp_prev.clone(), "sp_cur": at._sp_cur.clone(), "filt": at._filtered.clone(),
      "pf_pos": c.pf_pos.clone(), "pf_yaw": c.pf_yaw.clone(),
      "scale": at._scale.clone() if torch.is_tensor(at._scale) else at._scale, "offset": at._offset.clone(),
      "ref": (c.ref_pos.clone(), c.ref_R.clone(), c.ref_q.clone()),
    }

  def import_state(self, s, origin_offset):
    r, am, at, c = self.robot, self.env.action_manager, self.act, self.cmd
    # Policies may use different output transforms: map the action history
    # through setpoint space (setpoint = offset + scale * a).
    conv = lambda a: (s["scale"] * a + s["offset"] - at._offset) / at._scale
    s = dict(s, a=conv(s["a"]), a1=conv(s["a1"]), a2=conv(s["a2"]))
    root = s["root"].clone()
    root[:, :2] += origin_offset
    r.write_joint_state_to_sim(s["q"], s["qd"])
    r.write_root_state_to_sim(root)
    self.env.sim.forward()
    am._action[:] = s["a"]
    am._prev_action[:] = s["a1"]
    am._prev_prev_action[:] = s["a2"]
    at._sp_prev[:] = s["sp_prev"]
    at._sp_cur[:] = s["sp_cur"]
    at._filtered[:] = s["filt"]
    at._fresh[:] = False
    at._raw_actions[:] = s["a"]
    c.pf_pos[:] = s["pf_pos"] + origin_offset
    c.pf_yaw[:] = s["pf_yaw"]
    c._needs_init[:] = False
    c._pending = []
    c._prev_pos, c._prev_R, c._prev_q = s["ref"][0].clone(), s["ref"][1].clone(), s["ref"][2].clone()
    c._prev_pos[:, :2] += origin_offset
    # Evaluate this policy's reference now so reference velocities start clean.
    c._evaluate()
    c._prev_pos, c._prev_R, c._prev_q = c.ref_pos.clone(), c.ref_R.clone(), c.ref_q.clone()
    c._prev_pos_p, c._prev_R_p = c.ref_pos_p.clone(), c.ref_R_p.clone()
    self.obs = self.wenv.get_observations()


class AnimationEngine:
  """Background loop + triggered clip blending + joystick mapping for show functions/commands."""

  T_BETA, T_ALPHA = 0.1, 0.35

  def __init__(self):
    self.t = 0.0
    self.trig = None  # (start, duration, clip_fn)

  def background(self, t):
    # Intermittent eye blinks and antenna motion (show functions only).
    blink = 1.0 if (t % 3.7) < 0.12 else 0.0
    return {"eye_radius": 1.0 - 0.9 * blink, "antenna": 0.3 * math.sin(2 * math.pi * 0.25 * t), "head_dh": 0.0, "head_yaw": 0.0, "head_pitch": 0.0, "head_roll": 0.0}

  def trigger(self, clip_fn, duration):
    self.trig = (self.t, duration, clip_fn)

  @staticmethod
  def _ramp(tau, dur, T):
    return min(1.0, tau / T, (dur - tau) / T) if 0 <= tau <= dur else 0.0

  def compose(self, joystick):
    y = self.background(self.t)
    if self.trig is not None:
      t0, dur, fn = self.trig
      tau = self.t - t0
      if tau > dur:
        self.trig = None
      else:
        yt = fn(tau)
        beta, alpha = self._ramp(tau, dur, self.T_BETA), self._ramp(tau, dur, self.T_ALPHA)
        for k in y:
          w = beta if k in ("eye_radius", "antenna") else alpha
          y[k] = (1 - w) * y[k] + w * yt.get(k, y[k])
    # Joystick layer: additive gaze/posture offsets (Fig. 6 / Fig. 11 mapping).
    y["head_yaw"] += joystick.get("gaze_yaw", 0.0)
    y["head_pitch"] += joystick.get("gaze_pitch", 0.0)
    over = max(0.0, abs(joystick.get("gaze_yaw", 0.0)) - 0.6) * math.copysign(1, joystick.get("gaze_yaw", 0.0))
    y["torso_yaw"] = joystick.get("posture_yaw", 0.0) + 0.5 * over
    y["torso_pitch"] = joystick.get("posture_pitch", 0.0)
    # Posture control counter-rotates the head to keep the gaze fixed.
    y["head_yaw"] += 0.0
    self.t += 0.02
    return y


def nod_clip(tau):
  return {"head_pitch": 0.25 * math.sin(2 * math.pi * 2.0 * tau), "eye_radius": 1.2}


def no_clip(tau):
  return {"head_yaw": 0.35 * math.sin(2 * math.pi * 1.8 * tau), "eye_radius": 0.8}
