"""Sec. VI / VII-C: animation engine layers driving the standing policy.

Timeline: background only -> triggered "yes" nod -> joystick gaze (head follows,
torso joins at the end of the neck range) -> posture control (torso moves,
gaze fixed) -> triggered "no" while gazing. Records video + command/show plots.

python scripts/bdx_show.py STAND_CKPT
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_policy import load_policy  # noqa: E402

from disneyrepro.bdx.record import PoseRecorder  # noqa: E402
from disneyrepro.bdx.runtime import AnimationEngine, no_clip, nod_clip  # noqa: E402

OUT = Path("results/bdx")
env, wenv, policy = load_policy("Disney-BDX-Standing", sys.argv[1], 1, render=True, cfg_hook=lambda c: setattr(c, "episode_length_s", 1e9))
c = env.command_manager.get_term("ref")
eng = AnimationEngine()
obs = wenv.get_observations()
frames, log = [], []
rec = PoseRecorder()


def joystick(t):
  j = {}
  if 6.0 <= t < 10.0:  # gaze control: right stick yaw sweep
    j["gaze_yaw"] = 0.9 * np.sin(np.pi * (t - 6.0) / 4.0)
    j["gaze_pitch"] = 0.2 * np.sin(np.pi * (t - 6.0) / 2.0)
  if 10.5 <= t < 14.0:  # posture control: left stick, torso yaw / pitch, gaze fixed
    s = np.sin(np.pi * (t - 10.5) / 3.5)
    j["posture_yaw"] = 0.2 * s
    j["posture_pitch"] = 0.2 * s
  return j


t = 0.0
while t < 17.0:
  if abs(t - 3.0) < 1e-6 or (2.99 < t < 3.01):
    eng.trigger(nod_clip, 1.5)
  if 14.49 < t < 14.51:
    eng.trigger(no_clip, 1.6)
  y = eng.compose(joystick(t))
  g = torch.tensor([[y["head_dh"], y["head_yaw"], y["head_pitch"], y["head_roll"], 0.0, y["torso_yaw"], y["torso_pitch"], 0.0]], device=env.device)
  g = torch.maximum(torch.minimum(g, c.hi), c.lo)
  c.set_command(g)
  with torch.no_grad():
    obs, _, _, _ = wenv.step(policy(obs))
  frames.append(env.render())
  rec.add(env, eye_radius=y["eye_radius"], antenna=y["antenna"])
  log.append({"t": t, "g": g[0].tolist(), "eye_radius": y["eye_radius"], "antenna": y["antenna"], "q": env.scene["robot"].data.joint_pos[0].tolist()})
  t += 0.02

import mediapy  # noqa: E402

mediapy.write_video(OUT / "show_animation_engine.mp4", np.asarray(frames), fps=50)
json.dump(log, open(OUT / "show_animation_engine.json", "w"))
(OUT / "traj").mkdir(exist_ok=True)
rec.save(OUT / "traj" / "show_animation_engine.npz")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ts = np.array([r["t"] for r in log])
G = np.array([r["g"] for r in log])
fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for k, n in ((1, "head yaw"), (2, "head pitch")):
  ax[0].plot(ts, G[:, k], label=n)
for k, n in ((5, "torso yaw"), (6, "torso pitch")):
  ax[1].plot(ts, G[:, k], label=n)
ax[2].plot(ts, [r["eye_radius"] for r in log], label="eye radius")
ax[2].plot(ts, [r["antenna"] for r in log], label="antenna")
for a in ax:
  a.legend(fontsize=7)
  for x0, x1, lab in ((3, 4.5, "nod"), (6, 10, "gaze"), (10.5, 14, "posture"), (14.5, 16.1, "no")):
    a.axvspan(x0, x1, alpha=0.08)
ax[2].set_xlabel("Time [s]")
fig.tight_layout()
fig.savefig(OUT / "show_animation_engine.png", dpi=140)
print("saved", len(frames), "frames")
