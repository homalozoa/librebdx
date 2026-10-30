"""Build the walking reference table for a character (gait tool + GPU IK).

python scripts/build_walk_refs.py bdx|olaf|olaf_flat
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from librebdx.gait.planner import FootGeom
from librebdx.gait.refgen import CycleSpec, build_table
from librebdx.gait.styles import characters as C
from librebdx.kin.leg_ik import LegIK
from librebdx.kin.torch_kin import mat_to_quat

ROOT = Path(__file__).parents[1] / "librebdx"
name = sys.argv[1]

if name in ("bdx", "bdx_calm"):
  from librebdx.bdx.standing_ref import XML, LEG_L, LEG_R

  style_fn = C.bdx_walk_calm if name == "bdx_calm" else C.bdx_walk
  foot = FootGeom(heel_x=-0.0675, toe_x=0.0675, half_width=0.008, line_contact=True)
  legs, feet, roll = LEG_L + LEG_R, ("left_sole", "right_sole"), False
  grid = (np.linspace(-0.3, 0.7, 11), np.linspace(-0.4, 0.4, 7), np.linspace(-1.8, 1.8, 9))
  out = ROOT / "bdx" / "data" / ("walk_table_calm.npz" if name == "bdx_calm" else "walk_table.npz")
else:
  XML = ROOT / "robots" / "olaf" / "olaf.xml"
  style_fn = C.olaf_walk if name == "olaf" else (lambda: C.olaf_walk(heel_toe=False))
  foot = FootGeom(heel_x=-0.07, toe_x=0.07, half_width=0.045)
  legs = [f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")]
  feet, roll = ("left_sole", "right_sole"), True
  grid = (np.linspace(-0.15, 0.3, 10), np.linspace(-0.12, 0.12, 5), np.linspace(-0.6, 0.6, 7))
  out = ROOT / "olaf" / "data" / ("walk_table.npz" if name == "olaf" else "walk_table_flat.npz")
out.parent.mkdir(parents=True, exist_ok=True)

dev = "cuda"
ik = LegIK(XML, legs, feet, foot_roll=roll, device=dev)
kin = ik.kin
# Nominal CoM height and the sole-site rotation at the zero pose.
q0 = torch.zeros(1, kin.nq, device=dev)
I = torch.eye(3, device=dev)[None]
P, R = kin.forward(torch.zeros(1, 3, device=dev), I, q0)
R_off = [kin.site(P, R, s)[1][0] for s in feet]
sole_z = kin.site(P, R, feet[0])[0][0, 2]
pz0 = style_fn().samples["in_place"].curves["pelvis_z"](0.0)
com_h = float(kin.com(P, R)[0, 2] - sole_z - (-sole_z - pz0))
print("LIPM CoM height", com_h)

t0 = time.time()
tab = build_table(style_fn, grid, CycleSpec(foot=foot, com_height=com_h))
print(f"gait tool: {time.time()-t0:.1f}s")
shape = tab["pelvis_z"].shape  # (Nx, Ny, Nw, Nphi)
N = int(np.prod(shape))
t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev).reshape(N, *a.shape[len(shape):])
upper_names = sorted(style_fn().upper_joints)
other_names = [kin.joint_names[i] for i in ik.other]
uidx = [upper_names.index(n) for n in other_names]
fR = t(tab["foot_R"]) @ torch.stack(R_off)[None]
aux = ik.pack(t(tab["pelvis_z"]), t(tab["pelvis_R"]), t(tab["upper"])[:, uidx], t(tab["com"]), t(tab["foot_p"]), fR)
# Solve phase-sequentially with warm starts (continuous IK branch along the
# cycle); two laps so that phase 0 is warm-started from the end of the cycle.
ncmd, nphi = N // shape[-1], shape[-1]
auxp = aux.reshape(ncmd, nphi, -1)
x = torch.zeros(ncmd, 2 + len(ik.leg), device=dev)
X = torch.zeros(ncmd, nphi, x.shape[-1], device=dev)
Rr = torch.zeros(ncmd, nphi, device=dev)
for lap in range(2):
  for k in range(nphi):
    x, r = ik.solve(auxp[:, k], x, iters=40 if (lap == 0 and k == 0) else 12)
    X[:, k], Rr[:, k] = x, r.abs().max(-1).values
x, res = X.reshape(N, -1), Rr.reshape(N)
print(f"IK residual: median {res.median():.2e} p99 {res.quantile(0.99):.2e} max {res.max():.2e}; frac>5mm {(res > 5e-3).float().mean():.3f}")
q = ik.full_q(x, t(tab["upper"])[:, uidx])
np.savez_compressed(
  out,
  vx=tab["vx"], vy=tab["vy"], wz=tab["wz"],
  root_xy=x[:, :2].cpu().numpy().reshape(shape + (2,)),
  root_z=tab["pelvis_z"],
  root_quat=mat_to_quat(t(tab["pelvis_R"])).cpu().numpy().reshape(shape + (4,)),
  q=q.cpu().numpy().reshape(shape + (kin.nq,)),
  contact=tab["contact"], rate=tab["rate"],
  foot_p=tab["foot_p"], com=tab["com"],
  joint_names=np.array(kin.joint_names), residual=res.cpu().numpy().reshape(shape),
)
print("saved", out, shape)
