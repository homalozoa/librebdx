"""Quick robustness check of controller options on the dino walk test."""
import sys
import numpy as np
from disneyrepro.gait import sim
from disneyrepro.gait.controller import FeedbackGains
from disneyrepro.gait.robots import dino
from disneyrepro.gait.styles.dino import dino_gait

def v_fn(t):
  if t < 2.0: return (0.0, 0.0, 0.0)
  if t < 6.0: return (min(0.3, 0.15 * (t - 2.0)), 0.0, 0.0)
  if t < 9.0: return (0.2, 0.0, 0.5)
  if t < 12.0: return (0.0, 0.1, 0.0)
  if t < 15.0: return (0.4, 0.0, 0.0)
  return (-0.15, 0.0, -0.5)
dur = float(sys.argv[1])
import json
variants = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {"default": {}}
for name, kw in variants.items():
  if "feedback" in kw:
    kw["feedback"] = FeedbackGains(**kw["feedback"])
  res = sim.run(dino(), dino_gait(), dur, v_fn, cfg_kw=kw)
  log = res["log"]
  e = np.array([np.linalg.norm(r["com"][:2] - r["com_ref"]) for r in log if r["com_ref"] is not None])
  print(f"{name:10s} fell={res['fell']} t_end={res['t_end']:.2f} com_err mean={e.mean():.3f} max={e.max():.3f}", flush=True)
