import sys
import numpy as np
from disneyrepro.gait import sim
from disneyrepro.gait.robots import dino
from disneyrepro.gait.styles.dino import dino_gait

def v_fn(t):
  if t < 2.0: return (0.0, 0.0, 0.0)
  if t < 6.0: return (min(0.3, 0.15 * (t - 2.0)), 0.0, 0.0)
  if t < 9.0: return (0.2, 0.0, 0.5)
  return (0.0, 0.1, 0.0)
res = sim.run(dino(), dino_gait(), float(sys.argv[1]), v_fn, log_every=8)
print("fell", res["fell"], res["t_end"])
log = res["log"]
t0 = float(sys.argv[2])
for r in log:
  if r["t"] < t0: continue
  fe = {s: np.round(r["foot"][s] - r["foot_ref"][s], 3) for s in "lr"}
  print(f"t={r['t']:.3f} phi={r['phi']:.2f} k={r['kappa']:.2f} e={np.round(r['com'][:2]-r['com_ref'],3)} step={np.round(r['step'],3)} vadj={np.round(r['vel_adj'],2)} pz={r['pelvis'][2]:.3f}/{r['pelvis_z_ref']:.3f} yaw={r['pelvis_yaw']:.2f}/{r['pelvis_yaw_ref']:.2f} footerr={fe} c={r['contact_ref']} {r['wbc_status']}")
