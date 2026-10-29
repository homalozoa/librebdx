import time
import numpy as np
from disneyrepro.gait.mpc import ComMPC
from disneyrepro.gait.planner import FootGeom, GaitPlanner
from disneyrepro.gait.styles.dino import dino_gait
foot = FootGeom(-0.065, 0.065, 0.034)
pl = GaitPlanner(dino_gait(), foot)
out = pl.output()
mpc = ComMPC(0.37, foot)
print("supports", [(round(a,3), round(b,3), [s for s,_ in f]) for a,b,f in out.supports])
import proxsuite
t0=time.time(); mpc.plan(np.array([0,0,0,0.]), np.zeros(2), out.supports); print("plan", time.time()-t0, mpc.status)
import cProfile, pstats
cProfile.run("mpc.plan(np.array([0,0,0,0.]), np.zeros(2), out.supports)", "/tmp/mpc.prof")
pstats.Stats("/tmp/mpc.prof").sort_stats("tottime").print_stats(8)
