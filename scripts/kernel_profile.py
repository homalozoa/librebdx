"""Top GPU kernels over 10 policy steps (torch.profiler).  python scripts/kernel_profile.py TASK"""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

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
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
  for _ in range(10):
    env.step(a)
  torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
