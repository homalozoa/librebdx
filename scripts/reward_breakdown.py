"""Per-step reward term magnitudes under a hand-made tracking controller."""
import sys
import torch
from disneyrepro.tasks import register_all
register_all()
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
task, K = sys.argv[1], float(sys.argv[2])
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg, device="cuda"); env.reset()
c = env.command_manager.get_term("ref"); r = env.scene["robot"]
at = env.action_manager.get_term("joints")
rm = env.reward_manager
sums = {}
for i in range(300):
  q = r.data.joint_pos
  sp = c.ref_q + K * (c.ref_q - q) - 0.015 * K * r.data.joint_vel
  env.step((sp - at._offset) / at._scale)
  for name, cfg_ in zip(rm._term_names, rm._term_cfgs):
    v = cfg_.func(env, **cfg_.params) * cfg_.weight
    sums[name] = sums.get(name, 0) + float(v.mean()) / 300
for k, v in sorted(sums.items(), key=lambda x: x[1]):
  print(f"{k:18s} {v:+.3f}")
print("total", sum(sums.values()))
