"""Host-side hot spots of env.step (cProfile).  python scripts/cpu_profile.py TASK"""
import cProfile
import pstats
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
torch.cuda.synchronize()
pr = cProfile.Profile()
pr.enable()
for _ in range(20):
  env.step(a)
torch.cuda.synchronize()
pr.disable()
st = pstats.Stats(pr)
st.sort_stats("tottime").print_stats(28)
