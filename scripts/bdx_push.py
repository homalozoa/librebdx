"""Robustness: horizontal pushes on the torso (0.1 s) of increasing magnitude.

python scripts/bdx_push.py TASK CKPT [vx]
"""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_policy import load_policy  # noqa: E402

task, ck = sys.argv[1], sys.argv[2]
vx = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
N = 128
env, wenv, policy = load_policy(task, ck, N)
c = env.command_manager.get_term("ref")
r = env.scene["robot"]
bid = r.find_bodies("base_link")[0]
if "Walking" in task:
  c.set_command(torch.tensor([[0, 0, 0, 0, vx, 0, 0]] * N, device=env.device))
elif "Standing" in task:
  c.set_command(torch.zeros(N, 8, device=env.device))
res = {}
for F in (0, 30, 60, 90, 120, 150, 180):
  obs, _ = wenv.reset()
  ang = torch.rand(N, device=env.device) * 2 * math.pi
  fell = torch.zeros(N, dtype=torch.bool, device=env.device)
  for i in range(200):
    if 75 <= i < 80:  # 0.1 s push after 1.5 s
      f = torch.zeros(N, 1, 3, device=env.device)
      f[:, 0, 0], f[:, 0, 1] = F * torch.cos(ang), F * torch.sin(ang)
      r.write_external_wrench_to_sim(f, torch.zeros_like(f), body_ids=bid)
    elif i == 80:
      z = torch.zeros(N, 1, 3, device=env.device)
      r.write_external_wrench_to_sim(z, z, body_ids=bid)
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
    if i >= 75:
      fell |= env.termination_manager.terminated
  res[F] = float(1 - fell.float().mean())
  print(f"push {F:4d} N x 0.1 s: survival {res[F]:.2f}")
out = Path("results/bdx")
out.mkdir(parents=True, exist_ok=True)
name = task.replace("Disney-BDX-", "").lower() + (f"_vx{vx}" if vx else "")
json.dump(res, open(out / f"push_{name}.json", "w"), indent=1)
