"""Zero-command drift: walk, stop, then measure torso drift while commanded to stand.

python scripts/drift_test.py WALK_CKPT
"""
import json
import sys

import torch

sys.path.insert(0, "scripts")
from eval_policy import load_policy  # noqa: E402

N = 64
env, wenv, policy = load_policy("Disney-BDX-Walking", sys.argv[1], N)
c = env.command_manager.get_term("ref")
r = env.scene["robot"]
obs = wenv.get_observations()
dt = env.step_dt
out = {}
for name, v in (("after_forward", (0.5, 0, 0)), ("after_turn", (0, 0, 1.8))):
  g = torch.zeros(N, 7, device=env.device)
  g[:, 4:] = torch.tensor(v, device=env.device)
  for _ in range(int(3.0 / dt)):
    c.set_command(g)
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
  g[:] = 0
  for _ in range(int(2.0 / dt)):  # settle
    c.set_command(g)
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
  p0 = r.data.root_link_pos_w[:, :2].clone()
  yaw = []
  for _ in range(int(8.0 / dt)):
    c.set_command(g)
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
    yaw.append(r.data.root_link_ang_vel_w[:, 2].clone())
  d = (r.data.root_link_pos_w[:, :2] - p0).norm(dim=-1)
  out[name] = {"drift_m_per_s_mean": float(d.mean() / 8.0), "drift_m_per_s_max": float(d.max() / 8.0), "yaw_rate_mean": float(torch.stack(yaw).mean(0).abs().mean())}
print(json.dumps(out))
