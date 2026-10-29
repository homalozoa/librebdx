"""Where does one env step go? Times physics, actuator model, commands, rewards, obs (CUDA-synced).

python scripts/profile_step.py TASK [num_envs]
"""
import sys
import time
from collections import defaultdict

import torch

from disneyrepro.tasks import register_all

register_all()
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

task = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
cfg = load_env_cfg(task, play=False)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
T = defaultdict(float)


def timed(obj, name, key):
  f = getattr(obj, name)

  def w(*a, **k):
    t0 = time.perf_counter()  # host-side time only: launch/Python overhead (GPU runs async)
    r = f(*a, **k)
    T[key] += time.perf_counter() - t0
    return r

  setattr(obj, name, w)


timed(env.sim, "step", "physics (mujoco warp step)")
for nm, mgr in (("actions", env.action_manager), ("commands", env.command_manager), ("rewards", env.reward_manager),
                ("observations", env.observation_manager), ("terminations", env.termination_manager), ("events", env.event_manager)):
  for meth in ("apply_action", "process_action", "compute", "update", "apply"):
    if hasattr(mgr, meth):
      timed(mgr, meth, f"{nm}.{meth}")
c = env.command_manager.get_term("ref")
for meth, key in (("_advance", "cmd._advance"), ("_evaluate", "cmd._evaluate"), ("_init_episode", "cmd._init_episode"),
                  ("_write_state", "cmd._write_state"), ("_init_velocities", "cmd._init_velocities"), ("_update_metrics", "cmd._update_metrics")):
  if hasattr(c, meth):
    timed(c, meth, key)
timed(env.sim, "forward", "sim.forward")
ev = env.event_manager
a = torch.zeros(N, env.action_manager.total_action_dim, device="cuda")
for _ in range(5):
  env.step(a)
T.clear()
torch.cuda.synchronize()
t0 = time.perf_counter()
K = 48
for _ in range(K):
  env.step(a)
torch.cuda.synchronize()
tot = time.perf_counter() - t0
print(f"{task}: {tot / K * 1000:.1f} ms per policy step ({N} envs, decimation {cfg.decimation})")
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
  print(f"  {k:32s} {v / K * 1000:7.1f} ms  ({100 * v / tot:4.1f} %)")
