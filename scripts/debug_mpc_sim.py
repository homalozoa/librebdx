import time
import numpy as np
from disneyrepro.gait import mpc as M
orig = M.ComMPC.plan
def timed(self, xi0, p0, supports):
  t0 = time.time()
  r = orig(self, xi0, p0, supports)
  dt = time.time() - t0
  if dt > 0.05:
    print(f"slow plan {dt:.2f}s status {self.status} xi0 {np.round(xi0,3)} p0 {np.round(p0,3)}")
    for a, b, f in supports:
      print("   ", round(a, 3), round(b, 3), [(s, np.round(p, 3)) for s, p in f])
    raise SystemExit
  return r
M.ComMPC.plan = timed
from disneyrepro.gait import sim
from disneyrepro.gait.robots import dino
from disneyrepro.gait.styles.dino import dino_gait
sim.run(dino(), dino_gait(), 0.3, lambda t: (0, 0, 0))
