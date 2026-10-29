"""First functional test of the gait stack on the dino: step in place, then walk."""

import sys
import time

import numpy as np

from disneyrepro.gait import sim
from disneyrepro.gait.robots import dino
from disneyrepro.gait.styles.dino import dino_gait

setup = dino()
style = dino_gait()
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0


def v_fn(t):
  if t < 2.0:
    return (0.0, 0.0, 0.0)
  if t < 6.0:
    return (min(0.3, 0.15 * (t - 2.0)), 0.0, 0.0)
  if t < 9.0:
    return (0.2, 0.0, 0.5)
  return (0.0, 0.1, 0.0)


t0 = time.time()
res = sim.run(setup, style, dur, v_fn, video="results/videos/dino_test.mp4")
print(f"wall {time.time()-t0:.1f}s for {res['t_end']:.2f}s sim; fell={res['fell']}")
log = res["log"]
for r in log[:: max(1, len(log) // 20)]:
  print(f"t={r['t']:.2f} phi={r['phi']:.2f} kappa={r['kappa']:.2f} com={np.round(r['com'],3)} com_ref={None if r['com_ref'] is None else np.round(r['com_ref'],3)} "
        f"pz={r['pelvis'][2]:.3f}/{r['pelvis_z_ref']:.3f} st={r['wbc_status']}")
print("events", res["events"][:12])
