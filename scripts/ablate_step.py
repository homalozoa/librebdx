"""A/B step timing: total ms per policy step with one event term (or feature) removed at a time.

python scripts/ablate_step.py TASK
"""
import sys
import time

import torch

from disneyrepro.tasks import register_all

register_all()
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

task = sys.argv[1]
N = 8192


def bench(drop=None):
  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = N
  for d in (drop.split("+") if drop else []):
    cfg.events.pop(d, None)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
  a = torch.zeros(N, env.action_manager.total_action_dim, device="cuda")
  for _ in range(8):
    env.step(a)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(48):
    env.step(a)
  torch.cuda.synchronize()
  ms = (time.perf_counter() - t0) / 48 * 1000
  env.close()
  del env
  torch.cuda.empty_cache()
  return ms


base = bench()
print(f"baseline {base:.1f} ms/step", flush=True)
cfg = load_env_cfg(task, play=False)
for name in (sys.argv[2].split(",") if len(sys.argv) > 2 else list(cfg.events.keys())):
  ms = bench(name)
  print(f"  without {name:16s} {ms:6.1f} ms/step  (saves {base - ms:5.1f})", flush=True)
