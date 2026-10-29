"""GPU time of one policy step's physics (sim.step x decimation), CUDA-event timed.

python scripts/physics_time.py TASK
"""
import sys

import torch

from disneyrepro.tasks import register_all

register_all()
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

cfg = load_env_cfg(sys.argv[1], play=False)
cfg.scene.num_envs = 8192
env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
a = torch.zeros(8192, env.action_manager.total_action_dim, device="cuda")
for _ in range(10):
  env.step(a)
ev = []
f = env.sim.step


def timed(*x, **k):
  s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  s.record()
  r = f(*x, **k)
  e.record()
  ev.append((s, e))
  return r


env.sim.step = timed
for _ in range(20):
  env.step(a)
torch.cuda.synchronize()
per = sum(s.elapsed_time(e) for s, e in ev) / 20
print(f"{sys.argv[1]}: physics GPU time {per:.1f} ms per policy step ({cfg.decimation} substeps, {per / cfg.decimation:.2f} ms each); "
      f"solver iterations {cfg.sim.mujoco.iterations}, ls {cfg.sim.mujoco.ls_iterations}, nconmax {cfg.sim.nconmax}")
