"""Render demo videos of the final BDX policies (one env, 50 fps).

python scripts/bdx_videos.py STAND WALK HAPPY EXCITED JUMP TANTRUM   (an episodic checkpoint of "-" skips that clip)
"""

import sys
from pathlib import Path

import mediapy
import numpy as np
import torch

from librebdx.bdx.record import PoseRecorder

sys.path.insert(0, str(Path(__file__).parent))
from eval_policy import load_policy  # noqa: E402

TRAJ = Path("results/bdx/traj")
TRAJ.mkdir(parents=True, exist_ok=True)

OUT = Path("results/bdx/videos")
OUT.mkdir(parents=True, exist_ok=True)
stand, walk, happy, excited, jump, tantrum = sys.argv[1:7]


def run(task, ck, seconds, cmd_fn=None, name=None, hook=None, pre_step=None):
  env, wenv, policy = load_policy(task, ck, 1, render=True, cfg_hook=hook)
  c = env.command_manager.get_term("ref")
  obs = wenv.get_observations()
  frames = []
  rec = PoseRecorder()
  for i in range(int(seconds / env.step_dt)):
    if cmd_fn is not None:
      c.set_command(torch.tensor([cmd_fn(i * env.step_dt)], dtype=torch.float32, device=env.device))
    extra = pre_step(env, i) if pre_step is not None else {}
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
    frames.append(env.render())
    rec.add(env, **extra)
  rec.save(TRAJ / f"{name}.npz")
  mediapy.write_video(OUT / f"{name}.mp4", np.asarray(frames), fps=50)
  env.close()
  print("saved", name)


# Standing: sweep each command dimension (Fig. 4 style).
lo = [-0.03, -0.8, -0.2, -0.45, -0.05, -0.22, -0.3, -0.15]
hi = [0.01, 0.8, 0.5, 0.45, 0.03, 0.22, 0.15, 0.15]


def stand_cmd(t):
  g = [0.0] * 8
  k = int(t // 2.0)
  if k < 8:
    s = np.sin(np.pi * (t - 2.0 * k) / 2.0)
    g[k] = (hi[k] if s > 0 else -lo[k]) * s
  return g


run("Disney-BDX-Standing", stand, 17.0, stand_cmd, "standing_commands", hook=lambda c: setattr(c, "episode_length_s", 1e9))


def walk_cmd(t):
  segs = [(1.5, (0, 0, 0)), (3.5, (0.7, 0, 0)), (2.5, (0, 0.35, 0)), (2.5, (0, -0.35, 0)), (3.0, (0, 0, 0.9)), (3.0, (0.35, 0, -0.8)), (1.5, (0, 0, 0))]
  acc = 0.0
  for d, v in segs:
    acc += d
    if t < acc:
      return [0, 0, 0, 0, *v]
  return [0] * 7


run("Disney-BDX-Walking", walk, 17.5, walk_cmd, "walking_commands", hook=lambda c: setattr(c, "episode_length_s", 1e9))
# Force pushes (Sec. VII robustness) while walking at 0.3 m/s: 0.1 s, 90 N.
PUSHES = {150: (0.0, 90.0), 300: (0.0, -90.0), 450: (-90.0, 0.0)}


def push(env, i):
  r = env.scene["robot"]
  bid = r.find_bodies("base_link")[0]
  f = torch.zeros(1, 1, 3, device=env.device)
  for i0, (fx, fy) in PUSHES.items():
    if i0 <= i < i0 + 5:
      f[0, 0, 0], f[0, 0, 1] = fx, fy
  r.write_external_wrench_to_sim(f, torch.zeros_like(f), body_ids=bid)
  return {"force": f[0, 0].tolist()}


run("Disney-BDX-Walking", walk, 11.0, lambda t: [0, 0, 0, 0, 0.3, 0, 0], "walking_push", hook=lambda c: setattr(c, "episode_length_s", 1e9), pre_step=push)
for task, ck, name in (("happy_dance", happy, "episodic_happy_dance"), ("excited", excited, "episodic_excited"), ("jump", jump, "episodic_jump"), ("tantrum", tantrum, "episodic_tantrum")):
  if ck == "-":  # policy not trained yet
    continue
  import numpy as _np

  T = (_np.load(f"librebdx/bdx/data/episodic_{task}.npz")["q"].shape[0] - 1) / 50.0
  run(f"Disney-BDX-Episodic-{task}", ck, T + 0.3, None, name)
