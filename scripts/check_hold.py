"""Sanity check: can a hand-made high-gain policy hold the standing reference?"""

import sys

import torch

from disneyrepro.tasks import register_all

register_all()
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

task = sys.argv[1]
K = float(sys.argv[2])
cfg = load_env_cfg(task, play=len(sys.argv) > 3 and sys.argv[3] == "play")
cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg, device="cuda")
env.reset()
c = env.command_manager.get_term("ref")
r = env.scene["robot"]
act_term = env.action_manager.get_term("joints")
scale = act_term._scale
offset = act_term._offset
falls = 0
cnt = {}
for i in range(500):
  q = r.data.joint_pos
  sp = c.ref_q + K * (c.ref_q - q) - 0.3 * K * r.data.joint_vel * 0.05
  a = (sp - offset) / scale
  env.step(a)
  falls += int(env.termination_manager.terminated.sum())
  if i < 30 and env.termination_manager.terminated.any():
    print("step", i, "terminated", int(env.termination_manager.terminated.sum()), "lens", env.episode_length_buf[env.termination_manager.terminated][:5].tolist())
  tm = env.termination_manager
  for k in tm.active_terms:
    cnt[k] = cnt.get(k, 0) + int(tm.get_term(k).sum())
print(cnt)
print(f"K={K}: falls {falls}; joint err {(r.data.joint_pos - c.ref_q).abs().mean():.3f}; torso err {c.metrics['err_torso_pos'].mean():.3f}; z {r.data.root_link_pos_w[:, 2].mean():.3f}")
