"""Record the story beats of the film as policy rollouts (poses only, no frames).

python scripts/bdx_story.py STAND WALK
Commands: standing = head height/yaw/pitch/roll + torso height/yaw/pitch/roll,
walking = 4 head offsets + vx, vy, wz. Each clip carries 2 s of pre-roll that the
film trims away (the reset pose settles into the first command).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_policy import load_policy  # noqa: E402

from librebdx.bdx.record import PoseRecorder  # noqa: E402

TRAJ = Path("results/bdx/traj")
TRAJ.mkdir(parents=True, exist_ok=True)
stand, walk = sys.argv[1:3]
PRE = 2.0


def smooth(a, b, t, t0, t1):
  s = np.clip((t - t0) / (t1 - t0), 0, 1)
  s = s * s * (3 - 2 * s)
  return a + (b - a) * s


def run(task, ck, seconds, cmd_fn, name):
  env, wenv, policy = load_policy(task, ck, 1, cfg_hook=lambda c: setattr(c, "episode_length_s", 1e9))
  c = env.command_manager.get_term("ref")
  obs = wenv.get_observations()
  rec = PoseRecorder()
  falls = 0
  for i in range(int((seconds + PRE) / env.step_dt)):
    t = i * env.step_dt - PRE
    c.set_command(torch.tensor([cmd_fn(max(t, 0.0))], dtype=torch.float32, device=env.device))
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
    falls += int(env.termination_manager.terminated.sum())
    rec.add(env)
  rec.save(TRAJ / f"{name}.npz")
  env.close()
  print("saved", name, "falls", falls, flush=True)


def wake(t):
  """Bowed and low, then rising, then a slow curious look around."""
  hp = smooth(-0.2, 0.0, t, 1.0, 3.6)                     # head pitch: bowed -> level
  th = smooth(-0.05, 0.0, t, 0.8, 3.4)                    # torso height: crouched -> nominal
  tp = smooth(0.15, 0.0, t, 0.8, 3.4)                     # torso pitch: slumped -> upright
  yaw = 0.55 * np.sin(np.pi * np.clip((t - 4.2) / 3.0, 0, 2)) if t > 4.2 else 0.0
  roll = smooth(0.0, 0.3, t, 6.8, 7.6) - smooth(0.0, 0.3, t, 8.4, 9.2)  # a small head tilt
  return [0.0, yaw, hp, roll, th, 0.0, tp, 0.0]


def gaze_up(t):
  """Looking up at the ringed planet, then a tiny tilt."""
  hp = smooth(0.0, 0.5, t, 0.5, 3.0)
  tp = smooth(0.0, -0.12, t, 0.5, 3.0)
  roll = smooth(0.0, -0.25, t, 4.5, 5.5)
  return [0.01, 0.0, hp, roll, 0.0, 0.0, tp, 0.0]


def answer(t):
  """The second droid: still, then a slow nod that answers the signal."""
  hp = smooth(0.0, -0.18, t, 3.0, 3.5) - smooth(0.0, -0.18, t, 3.7, 4.3)
  hp += smooth(0.0, -0.12, t, 4.5, 4.9) - smooth(0.0, -0.12, t, 5.0, 5.5)
  return [0.0, smooth(0.0, 0.25, t, 0.5, 2.0), hp, 0.0, 0.0, 0.0, 0.0, 0.0]


def crossing(t):
  """From a standstill toward the signal: a gentle left curve, a long straight, a slow arrival."""
  vx = smooth(0.0, 0.45, t, 0.5, 2.0) + smooth(0.0, 0.25, t, 7.0, 8.0) - smooth(0.0, 0.7, t, 12.0, 14.0)
  wz = smooth(0.0, 0.18, t, 1.5, 2.5) - smooth(0.0, 0.18, t, 6.0, 7.0)
  head = smooth(0.0, 0.1, t, 2.0, 3.0)
  return [0.0, 0.0, head, 0.0, vx, 0.0, wz]


run("Disney-BDX-Standing", stand, 10.0, wake, "story_wake")
run("Disney-BDX-Standing", stand, 7.0, gaze_up, "story_gaze_up")
run("Disney-BDX-Standing", stand, 7.0, answer, "story_answer")
run("Disney-BDX-Walking", walk, 15.0, crossing, "story_crossing")
