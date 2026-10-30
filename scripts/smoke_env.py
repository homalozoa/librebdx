"""Build a task env, step random/zero actions, report speed and basic stats."""

import sys
import time

import torch

from librebdx.tasks import register_all

register_all()
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

task = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
cfg = load_env_cfg(task)
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg, device="cuda")
obs, _ = env.reset()
print({k: v.shape for k, v in obs.items()})
act = torch.zeros(n, env.action_manager.total_action_dim, device="cuda")
t0 = time.time()
done_count = 0
rsum = 0.0
for i in range(steps):
  obs, rew, term, trunc, extras = env.step(act)
  done_count += int(term.sum())
  rsum += float(rew.mean())
torch.cuda.synchronize()
dt = time.time() - t0
print(f"{steps} steps x {n} envs: {dt:.2f}s -> {steps*n/dt:.0f} env-steps/s; terminations {done_count}; mean reward {rsum/steps:.3f}")
c = env.command_manager.get_term("ref")
print("err torso", float(c.metrics["err_torso_pos"].mean()), "err joint", float(c.metrics["err_joint"].mean()))
r = env.scene["robot"]
print("root z", r.data.root_link_pos_w[:4, 2].tolist())
for k, v in env.reward_manager._episode_sums.items():
  print(f"  {k:18s} {float(v.mean()):.3f}")
