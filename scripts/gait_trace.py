import sys, json
import numpy as np
from disneyrepro.gait import sim
from disneyrepro.gait.controller import FeedbackGains
from disneyrepro.gait.robots import dino
from disneyrepro.gait.styles.dino import dino_gait
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
if "feedback" in kw: kw["feedback"] = FeedbackGains(**kw["feedback"])
def v_fn(t):
  if t < 2.0: return (0.0, 0.0, 0.0)
  if t < 6.0: return (min(0.3, 0.15 * (t - 2.0)), 0.0, 0.0)
  if t < 9.0: return (0.2, 0.0, 0.5)
  if t < 12.0: return (0.0, 0.1, 0.0)
  if t < 15.0: return (0.4, 0.0, 0.0)
  return (-0.15, 0.0, -0.5)
mode = kw.pop("mode", "walk")
res = sim.run(dino(), dino_gait(), float(sys.argv[1]), (lambda t: (0.0, 0.0, 0.0)) if mode == "inplace" else v_fn, cfg_kw=kw)
print("fell", res["fell"], res["t_end"])
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
for r in res["log"][::12]:
  if r["t"] < t0: continue
  e = r["com"][:2] - r["com_ref"] if r["com_ref"] is not None else np.zeros(2)
  fe = max(np.linalg.norm(r["foot"][s][:2] - r["foot_ref"][s][:2]) for s in "lr")
  print(f"t={r['t']:.2f} phi={r['phi']:.2f} e={np.round(e,3)} k={r['kappa']:.2f} step={np.round(r['step'],3)} foot_err={fe:.3f} pz={r['pelvis'][2]-r['pelvis_z_ref']:+.3f} yawerr={r['pelvis_yaw']-r['pelvis_yaw_ref']:+.3f} v={r['v_cmd']} w={ {k: round(v,2) for k,v in r['weights'].items()} }")
