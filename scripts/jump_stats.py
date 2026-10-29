"""Jump statistics over N worlds: flight time (both feet off), foot clearance, torso rise, knee torque.

python scripts/jump_stats.py JUMP_CKPT
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from eval_policy import load_policy  # noqa: E402

N = 64
env, wenv, policy = load_policy("Disney-BDX-Episodic-jump", sys.argv[1], N)
c = env.command_manager.get_term("ref")
r = env.scene["robot"]
s = env.scene.sensors["feet_contact"]
sole = r.find_sites(["left_sole", "right_sole"], preserve_order=True)[0]
obs = wenv.get_observations()
T = float(c.T)
air = torch.zeros(N, device=env.device)
ref_air = 0.0
z0 = None
zmax = torch.zeros(N, device=env.device)
clear = torch.zeros(N, device=env.device)
falls = 0
for i in range(int((T + 0.2) / env.step_dt)):
  with torch.no_grad():
    obs, _, _, _ = wenv.step(policy(obs))
  falls += int(env.termination_manager.terminated.sum())
  z = r.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  if z0 is None:
    z0 = z.clone()
  zmax = torch.maximum(zmax, z)
  off = (s.data.found == 0).all(-1)
  air += off.float() * env.step_dt
  ref_air += float((c.ref_contact[0] < 0.5).all()) * env.step_dt
  fz = (r.data.site_pos_w[:, sole, 2] - env.scene.env_origins[:, 2:3]).min(-1).values
  clear = torch.maximum(clear, fz * off.float())
res = {
  "flight_time_s": [float(air.mean()), float(air.min()), float(air.max())],
  "ref_flight_time_s": ref_air,
  "foot_clearance_cm": float(clear.mean() * 100),
  "torso_rise_cm": float((zmax - z0).mean() * 100),
  "worlds_airborne_frac": float((air > 0.02).float().mean()),
  "falls": falls,
}
print(json.dumps(res))
