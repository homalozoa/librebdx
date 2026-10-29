"""Sec. VII-B: compare RL formulations for walking at fixed commands.

Metrics per formulation (forward 0.3 m/s and 0.6 m/s, 64 envs x 10 s):
  step frequency [Hz] (touch-downs per foot per second), swing foot clearance [m],
  torso roll / pitch std [rad] (sway), contact-schedule agreement with the gait
  reference, joint MAE w.r.t. the gait reference [rad], velocity error, falls.

python scripts/bdx_ablation_eval.py NAME=TASK=CKPT [...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_policy import load_policy  # noqa: E402


def evaluate(task, ckpt, v, N=64, T=10.0):
  env, wenv, policy = load_policy(task, ckpt, N)
  c = env.command_manager.get_term("ref")
  r = env.scene["robot"]
  sens = env.scene.sensors["feet_contact"]
  sole = r.find_sites(["left_sole", "right_sole"], preserve_order=True)[0]
  c.set_command(torch.tensor([[0, 0, 0, 0, *v]] * N, device=env.device))
  obs = wenv.get_observations()
  steps = int(T / env.step_dt)
  prev = None
  td = torch.zeros(N, 2, device=env.device)
  clear = torch.zeros(N, 2, device=env.device)
  rp, agree, mae, verr = [], [], [], []
  falls = 0
  for i in range(steps):
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
    falls += int(env.termination_manager.terminated.sum())
    if i < 100:
      continue  # skip the start-up transient
    found = (sens.data.found > 0).float()
    if prev is not None:
      td += ((found - prev) > 0).float()
    prev = found
    z = r.data.site_pos_w[:, sole, 2] - env.scene.env_origins[:, 2:3]
    clear = torch.maximum(clear, z * (1 - found))
    R = r.data.root_link_quat_w
    from mjlab.utils.lab_api.math import matrix_from_quat

    M = matrix_from_quat(R)
    roll = torch.atan2(M[:, 2, 1], M[:, 2, 2])
    pitch = torch.asin(-M[:, 2, 0].clamp(-1, 1))
    rp.append(torch.stack([roll, pitch], -1))
    agree.append((found == c.ref_contact).float().mean(-1))
    mae.append((r.data.joint_pos - c.ref_q).abs().mean(-1))
    yaw = c.pf_yaw
    vw = r.data.root_link_lin_vel_w[:, :2]
    vx = torch.cos(yaw) * vw[:, 0] + torch.sin(yaw) * vw[:, 1]
    verr.append((vx - v[0]).abs())
  dur = (steps - 100) * env.step_dt
  rp = torch.stack(rp)
  return {
    "step_freq_hz": float(td.mean() / dur),
    "clearance_m": float(clear.mean()),
    "torso_roll_std": float(rp[..., 0].std(0).mean()),
    "torso_pitch_std": float(rp[..., 1].std(0).mean()),
    "contact_agreement": float(torch.stack(agree).mean()),
    "joint_mae_vs_gait_ref": float(torch.stack(mae).mean()),
    "vx_err": float(torch.stack(verr).mean()),
    "falls": falls,
  }


if __name__ == "__main__":
  out = {}
  for spec in sys.argv[1:]:
    name, task, ckpt = spec.split("=")
    out[name] = {}
    for v in ((0.3, 0.0, 0.0), (0.6, 0.0, 0.0)):
      res = evaluate(task, ckpt, v)
      out[name][str(v[0])] = res
      print(name, v[0], json.dumps({k: round(x, 3) for k, x in res.items()}), flush=True)
  Path("results/bdx").mkdir(parents=True, exist_ok=True)
  json.dump(out, open("results/bdx/ablations.json", "w"), indent=1)
