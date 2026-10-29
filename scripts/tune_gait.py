"""Parallel random search over controller gains for a robot/style pair.

Scenarios: (A) omnidirectional velocity profile, (B) slow walk with a 25 N, 1 s
forward push and (C) a lateral push. Score = total survival time - 5 * mean CoM error.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")


def profile(vmax_x, vmax_y, vmax_w):
  def v_fn(t):
    if t < 2.0: return (0.0, 0.0, 0.0)
    if t < 6.0: return (min(0.75 * vmax_x, 0.4 * vmax_x * (t - 2.0)), 0.0, 0.0)
    if t < 9.0: return (0.5 * vmax_x, 0.0, 0.6 * vmax_w)
    if t < 12.0: return (0.0, 0.6 * vmax_y, 0.0)
    if t < 15.0: return (vmax_x, 0.0, 0.0)
    return (-0.3 * vmax_x, 0.0, -0.6 * vmax_w)
  return v_fn


def build_cfg(p):
  from disneyrepro.gait.controller import FeedbackGains
  from disneyrepro.gait.wbc import WbcGains
  wb = WbcGains(**{k[4:]: v for k, v in p.items() if k.startswith("wbc_")})
  fb = FeedbackGains(**{k[3:]: v for k, v in p.items() if k.startswith("fb_")})
  kw = {k: v for k, v in p.items() if not (k.startswith("wbc_") or k.startswith("fb_"))}
  return dict(wbc=wb, feedback=fb, **kw)


def evaluate(args):
  robot, style_name, p = args
  from disneyrepro.gait import sim
  from disneyrepro.gait import robots as R
  from disneyrepro.gait.styles import registry
  setup = getattr(R, robot)()
  style = registry.get(style_name)
  vx, _, vy, vw = style.max_velocity()
  score, errs, surv = 0.0, [], []
  scen = [
    ("profile", 18.0, profile(vx, vy, vw), None),
    ("push_fwd", 8.0, lambda t: (0.25 * vx, 0.0, 0.0), (3.0, 1.0, np.array([p.get("_push", 25.0), 0, 0]), setup_root(setup))),
    ("push_lat", 8.0, lambda t: (0.25 * vx, 0.0, 0.0), (3.0, 0.3, np.array([0, p.get("_push", 25.0) * 1.5, 0]), setup_root(setup))),
  ]
  pp = {k: v for k, v in p.items() if not k.startswith("_")}
  for name, dur, v_fn, push in scen:
    try:
      res = sim.run(setup, style, dur, v_fn, push=push, cfg_kw=build_cfg(pp), log_every=20)
    except Exception as e:  # noqa
      surv.append(0.0)
      continue
    surv.append(res["t_end"] if res["fell"] else dur)
    e = [np.linalg.norm(r["com"][:2] - r["com_ref"]) for r in res["log"] if r["com_ref"] is not None]
    errs.append(float(np.mean(e)) if e else 1.0)
  score = sum(surv) - 5.0 * float(np.mean(errs))
  return score, surv, errs, p


def setup_root(setup):
  import mujoco
  m = mujoco.MjModel.from_xml_path(setup.xml)
  return m.body(1).name


RANGES = {
  "wbc_com_kp": (20, 150), "wbc_com_kd": (5, 40), "wbc_w_com": (1, 50), "wbc_stance_kd": (10, 80),
  "wbc_foot_kp": (100, 800), "wbc_foot_kd": (10, 80), "wbc_pelvis_rot_kp": (50, 400), "wbc_pelvis_rot_kd": (10, 50),
  "wbc_pelvis_z_kp": (30, 300), "wbc_angmom_kd": (0, 20), "wbc_w_angmom": (0.0, 1.0),
  "joint_kp": (0, 80), "joint_kd": (0, 3),
  "fb_vel_kp": (0, 3), "fb_vel_kd": (0, 1), "fb_step_kp": (0, 2), "fb_step_kd": (0, 0.6),
  "fb_time_kp": (0, 15), "fb_time_kd": (0, 3), "mpc_beta_pos": (0, 0.3), "mpc_beta_vel": (0, 0.3),
}


def sample(rng, base=None, scale=1.0):
  p = {}
  for k, (lo, hi) in RANGES.items():
    if base is None:
      p[k] = float(rng.uniform(lo, hi))
    else:
      p[k] = float(np.clip(base[k] + rng.normal(0, scale * 0.15 * (hi - lo)), lo, hi))
  p["latch_measured"] = bool(rng.random() < 0.5) if base is None else base["latch_measured"]
  return p


if __name__ == "__main__":
  robot, style_name, n_iter = sys.argv[1], sys.argv[2], int(sys.argv[3])
  out = sys.argv[4]
  rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
  workers = int(os.environ.get("TUNE_WORKERS", "24"))
  best = []
  t0 = time.time()
  with mp.get_context("fork").Pool(workers) as pool:
    for it in range(n_iter):
      if it < 2 or not best:
        cands = [sample(rng) for _ in range(workers)]
      else:
        elites = [b[3] for b in sorted(best, key=lambda b: -b[0])[:6]]
        cands = [sample(rng, elites[i % len(elites)], scale=max(0.2, 1.0 - it / n_iter)) for i in range(workers - 2)]
        cands += [sample(rng) for _ in range(2)]
      res = pool.map(evaluate, [(robot, style_name, c) for c in cands])
      best += res
      best = sorted(best, key=lambda b: -b[0])[:40]
      b = best[0]
      print(f"iter {it} t={time.time()-t0:.0f}s best score {b[0]:.2f} surv {np.round(b[1],2)} err {np.round(b[2],3)}", flush=True)
      json.dump([{"score": s, "surv": sv, "err": e, "params": p} for s, sv, e, p in best], open(out, "w"), indent=1)
