"""Time-to-usable-behaviour curve: run the eval gates on every checkpoint of a run.

python scripts/gate_curve.py walk RUN_DIR [--envs 16] [--out results/gates/NAME.json]
python scripts/gate_curve.py jump RUN_DIR [--envs 64] [--out ...]

Each checkpoint gets gate metrics (walk: steady-state velocity error per command segment of Fig. 7,
falls; jump: flight time, airborne fraction, falls) and a pass/fail. The tensorboard event file of
the run gives wall-clock per iteration, so the first passing checkpoint becomes hours-to-gate.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from librebdx.tasks import register_all  # noqa: E402

register_all()
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

# Walk gate: steady-state (second half of each segment) command tracking, as in Fig. 7.
WALK_SEGS = [(1.0, (0, 0, 0)), (3.0, (0.7, 0, 0)), (2.0, (0, 0.4, 0)), (2.0, (0, -0.4, 0)),
             (2.0, (0, 0, 1.0)), (2.0, (0.4, 0, -1.0)), (2.5, (0, 0, 1.8)), (2.5, (0, 0, -1.8)), (1.0, (0, 0, 0))]
WALK_TOL = (0.07, 0.05, 0.2)  # m/s, m/s, rad/s: ~10% of the command range
JUMP_MIN_FLIGHT_S, JUMP_MIN_AIRBORNE = 0.16, 0.95


def build(task, envs):
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = envs
  agent_cfg = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(wenv, asdict(agent_cfg), device="cuda")
  return env, wenv, runner


def walk_gate(env, wenv, policy):
  c = env.command_manager.get_term("ref")
  r = env.scene["robot"]
  dt = env.step_dt
  obs = wenv.get_observations()
  falls, rows, t0 = 0, [], 0.0
  n = env.num_envs
  for dur, v in WALK_SEGS:
    acc = []
    steps = int(dur / dt)
    for k in range(steps):
      c.set_command(torch.tensor([[0, 0, 0, 0, *v]] * n, dtype=torch.float32, device=env.device))
      with torch.no_grad():
        obs, _, _, _ = wenv.step(policy(obs))
      falls += int(env.termination_manager.terminated.sum())
      if k >= steps // 2:
        yaw = c.pf_yaw
        cs, sn = torch.cos(yaw), torch.sin(yaw)
        vw = r.data.root_link_lin_vel_w[:, :2]
        vx = cs * vw[:, 0] + sn * vw[:, 1]
        vy = -sn * vw[:, 0] + cs * vw[:, 1]
        acc.append(torch.stack([vx, vy, r.data.root_link_ang_vel_w[:, 2]], -1).mean(0).cpu().numpy())
    m = np.mean(acc, 0)
    rows.append({"cmd": v, "meas": m.tolist(), "ok": bool(np.all(np.abs(m - np.array(v)) <= WALK_TOL))})
  ok = all(x["ok"] for x in rows) and falls == 0
  err = max(float(np.max(np.abs(np.array(x["meas"]) - np.array(x["cmd"])) / np.array(WALK_TOL))) for x in rows)
  return {"pass": ok, "falls": falls, "worst_err_over_tol": err, "segments": rows}


def jump_gate(env, wenv, policy):
  c = env.command_manager.get_term("ref")
  r = env.scene["robot"]
  s = env.scene.sensors["feet_contact"]
  n = env.num_envs
  obs = wenv.get_observations()
  air = torch.zeros(n, device=env.device)
  falls = 0
  for _ in range(int((float(c.T) + 0.2) / env.step_dt)):
    with torch.no_grad():
      obs, _, _, _ = wenv.step(policy(obs))
    falls += int(env.termination_manager.terminated.sum())
    air += (s.data.found == 0).all(-1).float() * env.step_dt
  fl, frac = float(air.mean()), float((air > 0.02).float().mean())
  return {"pass": fl >= JUMP_MIN_FLIGHT_S and frac >= JUMP_MIN_AIRBORNE and falls == 0,
          "flight_time_s": fl, "airborne_frac": frac, "falls": falls}


def wallclock(run_dir):
  """iteration -> hours since the first logged iteration, from the tensorboard event file(s)."""
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  pts = {}
  for ev in sorted(Path(run_dir).glob("events.out.tfevents.*")):
    acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
    acc.Reload()
    tag = next((t for t in acc.Tags()["scalars"] if "mean_reward" in t.lower() or t.endswith("Mean reward")), None)
    if tag:
      for e in acc.Scalars(tag):
        pts[e.step] = e.wall_time
  if not pts:
    return None
  steps = sorted(pts)
  return {s: (pts[s] - pts[steps[0]]) / 3600 for s in steps}


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("kind", choices=["walk", "jump"])
  ap.add_argument("run_dir")
  ap.add_argument("--envs", type=int, default=None)
  ap.add_argument("--out", default=None)
  ap.add_argument("--watch", action="store_true", help="early stop: poll RUN_DIR (a glob) and kill --pid after --patience passes in a row")
  ap.add_argument("--pid", type=int, default=None)
  ap.add_argument("--patience", type=int, default=2)
  a = ap.parse_args()
  task = "Disney-BDX-Walking" if a.kind == "walk" else "Disney-BDX-Episodic-jump"
  env, wenv, runner = build(task, a.envs or (16 if a.kind == "walk" else 64))
  if a.watch:
    import glob, os, signal, time

    while not glob.glob(a.run_dir + "/model_*.pt"):
      time.sleep(10)
    a.run_dir = str(Path(glob.glob(a.run_dir + "/model_*.pt")[0]).parent)
    run_start = os.path.getmtime(glob.glob(a.run_dir + "/events.out.tfevents.*")[0])
    seen, streak, curve = set(), 0, []
    while True:
      new = sorted((p for p in Path(a.run_dir).glob("model_*.pt") if p not in seen), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
      for p in new:
        time.sleep(3)  # checkpoint fully written
        seen.add(p)
        it = int(re.findall(r"\d+", p.stem)[0])
        runner.load(str(p), load_cfg={"actor": True}, strict=True, map_location="cuda")
        g = (walk_gate if a.kind == "walk" else jump_gate)(env, wenv, runner.get_inference_policy(device="cuda"))
        g.update(iter=it, hours=(time.time() - run_start) / 3600)
        curve.append(g)
        streak = streak + 1 if g["pass"] else 0
        print(it, g["pass"], {k: v for k, v in g.items() if k not in ("segments", "pass")}, flush=True)
        if streak >= a.patience:
          hit = curve[-a.patience]
          summary = {"run": a.run_dir, "early_stop_iter": it, "gate_first_pass_iter": hit["iter"], "gate_first_pass_hours": hit["hours"], "stopped_at_hours": g["hours"]}
          print(json.dumps(summary))
          out = Path(a.out or f"results/gates/{Path(a.run_dir).name}.json")
          out.parent.mkdir(parents=True, exist_ok=True)
          json.dump({"summary": summary, "curve": curve}, open(out, "w"), indent=1)
          if a.pid:
            os.kill(a.pid, signal.SIGINT)
          sys.exit(0)
      time.sleep(15)
  ckpts = sorted(Path(a.run_dir).glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
  wc = wallclock(a.run_dir)
  res = []
  for p in ckpts:
    it = int(re.findall(r"\d+", p.stem)[0])
    runner.load(str(p), load_cfg={"actor": True}, strict=True, map_location="cuda")
    policy = runner.get_inference_policy(device="cuda")
    g = (walk_gate if a.kind == "walk" else jump_gate)(env, wenv, policy)
    g.update(iter=it, hours=None if wc is None else wc.get(it))
    res.append(g)
    print(it, g["pass"], {k: v for k, v in g.items() if k not in ("segments", "pass")}, flush=True)
  first = next((g for g in res if g["pass"]), None)
  # "stable" pass: the first checkpoint from which every later one also passes.
  stable = next((res[i] for i in range(len(res)) if all(g["pass"] for g in res[i:])), None)
  summary = {"run": a.run_dir, "kind": a.kind, "first_pass_iter": first and first["iter"], "first_pass_hours": first and first["hours"],
             "stable_pass_iter": stable and stable["iter"], "stable_pass_hours": stable and stable["hours"],
             "final_iter": res[-1]["iter"], "final_hours": res[-1]["hours"]}
  print(json.dumps(summary))
  out = Path(a.out or f"results/gates/{Path(a.run_dir).name}.json")
  out.parent.mkdir(parents=True, exist_ok=True)
  json.dump({"summary": summary, "curve": res}, open(out, "w"), indent=1)
