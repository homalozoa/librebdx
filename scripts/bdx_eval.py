"""Evaluations for Grandia et al. (RSS 2024).

  mae TASK CKPT          Table III: mean absolute joint tracking error over random commands
  veltrack CKPT          Fig. 7: commanded path velocities vs. measured torso velocities
  jump CKPT              Fig. 8: neck-pitch / left-knee torques vs. velocity-dependent limits
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_policy import load_policy  # noqa: E402

OUT = Path("results/bdx")
OUT.mkdir(parents=True, exist_ok=True)


def mae(task, ckpt, seconds=60.0, envs=256):
  env, wenv, policy = load_policy(task, ckpt, envs)
  obs = wenv.get_observations()
  c = env.command_manager.get_term("ref")
  r = env.scene["robot"]
  errs, falls = [], 0
  steps = int(seconds / env.step_dt)
  for _ in range(steps):
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
    falls += int(env.termination_manager.terminated.sum())
    alive = ~env.termination_manager.terminated
    errs.append((r.data.joint_pos - c.ref_q).abs()[alive].mean(0).cpu().numpy())
  e = np.array(errs)
  res = {"task": task, "ckpt": ckpt, "mae_rad": float(e.mean()), "per_joint": e.mean(0).tolist(),
         "falls": falls, "env_seconds": seconds * envs}
  print(json.dumps(res))
  name = task.replace("Disney-BDX-", "").lower()
  json.dump(res, open(OUT / f"mae_{name}.json", "w"), indent=1)
  return res


def veltrack(ckpt):
  from disneyrepro.bdx.tasks import WALK_HI, WALK_LO
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  env, wenv, policy = load_policy("Disney-BDX-Walking", ckpt, 1)
  c = env.command_manager.get_term("ref")
  r = env.scene["robot"]
  dt = env.step_dt
  # Command script similar to Fig. 7 (forward, lateral, turning, combined).
  segs = [(1.0, (0, 0, 0)), (3.0, (0.7, 0, 0)), (2.0, (0, 0.4, 0)), (2.0, (0, -0.4, 0)), (2.0, (0, 0, 1.0)), (2.0, (0.4, 0, -1.0)), (2.5, (0, 0, 1.8)), (2.5, (0, 0, -1.8)), (1.0, (0, 0, 0))]
  obs = wenv.get_observations()
  log = []
  t = 0.0
  for dur, v in segs:
    for _ in range(int(dur / dt)):
      g = torch.tensor([[0, 0, 0, 0, *v]], dtype=torch.float32, device=env.device)
      c.set_command(g)
      with torch.no_grad():
        obs, _, _, _ = wenv.step(policy(obs))
      yaw = c.pf_yaw[0]
      R = torch.tensor([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]], device=env.device)
      vw = r.data.root_link_lin_vel_w[0, :2]
      vp = R @ vw
      log.append([t, *c.g[0, 4:].tolist(), *vp.tolist(), float(r.data.root_link_ang_vel_w[0, 2])])
      t += dt
  a = np.array(log)
  np.save(OUT / "veltrack.npy", a)
  fig, ax = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
  for i, (lab, col) in enumerate((("x", "C0"), ("y", "C1"))):
    ax[0].plot(a[:, 0], a[:, 1 + i], "--", color=col)
    ax[0].plot(a[:, 0], np.convolve(a[:, 4 + i], np.ones(15) / 15, "same"), color=col, label=lab)
  ax[0].set_ylabel("v [m/s]")
  ax[0].legend()
  ax[1].plot(a[:, 0], a[:, 3], "--", color="C2")
  ax[1].plot(a[:, 0], np.convolve(a[:, 6], np.ones(15) / 15, "same"), color="C2", label="z")
  ax[1].set_ylabel("w [rad/s]")
  ax[1].set_xlabel("Time [s]")
  fig.tight_layout()
  fig.savefig(OUT / "fig7_veltrack.png", dpi=150)
  # Steady-state tracking error per segment (last half of each segment).
  rows, t0 = [], 0.0
  for dur, v in segs:
    m = (a[:, 0] > t0 + dur / 2) & (a[:, 0] < t0 + dur)
    rows.append({"cmd": v, "meas": [float(a[m, 4].mean()), float(a[m, 5].mean()), float(a[m, 6].mean())]})
    t0 += dur
  json.dump(rows, open(OUT / "veltrack.json", "w"), indent=1)
  print(json.dumps(rows))


def jump(ckpt):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  from disneyrepro.rl.disney_actuator import DisneyActuator

  env, wenv, policy = load_policy("Disney-BDX-Episodic-jump", ckpt, 1)
  r = env.scene["robot"]
  names = list(r.joint_names)
  acts = [a for a in r.actuators if isinstance(a, DisneyActuator)]
  obs = wenv.get_observations()
  log = []
  for i in range(int(1.6 / env.step_dt)):
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
    row = [i * env.step_dt]
    for jn in ("Neck_Pitch", "Left_Knee"):
      j = names.index(jn)
      for a in acts:
        ids = a.target_ids.tolist()
        if j in ids:
          k = ids.index(j)
          qd = r.data.joint_vel[:, j : j + 1]
          lo, hi = a.torque_limits(qd.expand(-1, len(ids)))
          row += [float(a.applied[0, k]), float(lo[0, k]), float(hi[0, k])]
    log.append(row)
  a = np.array(log)
  np.save(OUT / "jump_torques.npy", a)
  fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
  for k, lab in enumerate(("NP", "KP")):
    ax[k].plot(a[:, 0], a[:, 1 + 3 * k], label=f"tau {lab}")
    ax[k].plot(a[:, 0], a[:, 2 + 3 * k], "k--", lw=0.8)
    ax[k].plot(a[:, 0], a[:, 3 + 3 * k], "k--", lw=0.8)
    ax[k].set_ylabel("tau [Nm]")
    ax[k].legend(loc="upper right")
  ax[1].set_xlabel("Time [s]")
  fig.tight_layout()
  fig.savefig(OUT / "fig8_jump_torques.png", dpi=150)
  print("saved jump torque plot")


if __name__ == "__main__":
  cmd = sys.argv[1]
  if cmd == "mae":
    mae(sys.argv[2], sys.argv[3])
  elif cmd == "veltrack":
    veltrack(sys.argv[2])
  elif cmd == "jump":
    jump(sys.argv[2])
