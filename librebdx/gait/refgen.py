"""Dense periodic walking references from the gait tool (planner + CoM MPC + IK).

For each velocity command on a grid we run the procedural planner and the CoM
MPC open loop (kinematically) until the gait is periodic, record one two-step
cycle as a function of the gait phase phi in [0, 1), express it in the path
frame, and solve whole-body IK on the GPU. The result is a table that RL
environments interpolate (Grandia et al. Sec. V-A: "we densely sample the
reference generators and implement the reference look-up during RL as
interpolation of these samples").
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

from librebdx.gait.mpc import ComMPC
from librebdx.gait.planner import FootGeom, GaitPlanner


@dataclass
class CycleSpec:
  foot: FootGeom
  com_height: float
  dt: float = 0.008
  warmup: float = 4.0
  n_phase: int = 64


def simulate_cycle(style, v, spec: CycleSpec, stance_mid=(0.0, 0.0)):
  """Return per-phase path-frame samples of one periodic cycle at command v.

  The planner runs for a warm-up plus several cycles. The horizontal CoM is the
  bounded solution of the LIPM xdd = w^2 (x - p) for the designed CoP p(t):
  x(t) = int (w/2) exp(-w |t - s|) p(s) ds, i.e. the periodic, dynamically
  consistent CoM that a preview controller (MPC) converges to.
  """
  pl = GaitPlanner(style, spec.foot)
  w0 = style.blend(v).attr("stance_width")
  for s, sgn in (("l", 1), ("r", -1)):
    pl.latched[s] = np.array([0.0, sgn * 0.5 * w0, 0.0])
    pl.target[s] = pl.latched[s].copy()
  frames = []
  t = 0.0
  while t < spec.warmup + 6 * style.blend(v).T + 2.0:
    pl.update(spec.dt, v)
    t += spec.dt
    out = pl.output()
    frames.append((t, pl.phi, pl.path.copy(), out, pl.blend.phase_rate(pl.phi)))
  ts = np.array([f[0] for f in frames])
  cop = np.stack([_cop(f[3], spec.foot) for f in frames])
  omega = math.sqrt(9.81 / spec.com_height)
  # Convolution with the symmetric LIPM kernel (direct, O(n^2) is fine here).
  dtk = spec.dt
  lag = np.arange(-int(6 / omega / dtk), int(6 / omega / dtk) + 1) * dtk
  ker = 0.5 * omega * np.exp(-omega * np.abs(lag)) * dtk
  ker /= ker.sum()
  com = np.stack([np.convolve(np.pad(cop[:, j], (len(lag) // 2,), mode="edge"), ker, mode="valid") for j in range(2)], -1)
  comv = np.gradient(com, dtk, axis=0)
  # Record the first full cycle after the warm-up (LLO to LLO).
  phis = np.array([f[1] for f in frames])
  wraps = np.where((np.diff(phis) < 0) & (ts[1:] > spec.warmup))[0] + 1
  i0, i1 = wraps[0], wraps[1]
  # The RL phase advances uniformly in time (rate 1 / T): use the normalised
  # time since left lift-off rather than the planner's phase-map variable,
  # whose double-support segments are traversed almost instantly.
  T = ts[i1] - ts[i0]
  rec = [((ts[i] - ts[i0]) / T, frames[i][2], com[i], comv[i], frames[i][3], 1.0 / T) for i in range(i0, i1)]
  return _to_path_frame(rec, spec.n_phase)


def _cop(out, foot):
  """Designed CoP: centre of the stance foot, or the mean of both feet in double support."""
  pts = []
  for s in "lr":
    if out.contact[s]:
      pts.append(out.foot_pos[s][:2])
  return np.mean(pts, 0)


def _to_path_frame(rec, n):
  phis = np.array([r[0] for r in rec])
  order = np.argsort(phis)
  rows = []
  for i in order:
    phi, path, x, vv, out, rate = rec[i]
    c, s = math.cos(path[2]), math.sin(path[2])
    Rt = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    o = np.array([path[0], path[1], 0.0])
    row = {
      "phi": phi,
      "rate": rate,
      "com": (Rt @ (np.array([x[0], x[1], 0.0]) - o))[:2],
      "pelvis_z": out.pelvis_z,
      "pelvis_R": Rt @ out.pelvis_R,
      "foot_p": np.stack([Rt @ (out.foot_pos[s_] - o) for s_ in "lr"]),
      "foot_R": np.stack([Rt @ out.foot_R[s_] for s_ in "lr"]),
      "contact": np.array([out.contact["l"], out.contact["r"]], float),
      "upper": np.array([out.upper[j] for j in sorted(out.upper)]),
    }
    rows.append(row)
  # Resample on a uniform phase grid (linear interp; rotations via nearest + small angle).
  grid = (np.arange(n) + 0.0) / n
  ph = np.array([r["phi"] for r in rows])
  ph_ext = np.concatenate([ph - 1, ph, ph + 1])
  def interp(key):
    vals = np.stack([r[key] for r in rows])
    vals_ext = np.concatenate([vals, vals, vals])
    flat = vals_ext.reshape(len(vals_ext), -1)
    out = np.stack([np.interp(grid, ph_ext, flat[:, j]) for j in range(flat.shape[1])], -1)
    return out.reshape((n,) + vals.shape[1:])
  res = {k: interp(k) for k in ("rate", "com", "pelvis_z", "pelvis_R", "foot_p", "foot_R", "upper")}
  # Re-orthonormalise interpolated rotations.
  for k in ("pelvis_R", "foot_R"):
    U, _, Vt = np.linalg.svd(res[k])
    res[k] = U @ Vt
  # Contacts: nearest sample in phase.
  cont = np.stack([r["contact"] for r in rows])
  idx = np.abs(((grid[:, None] - ph[None]) + 0.5) % 1.0 - 0.5).argmin(1)
  res["contact"] = cont[idx]
  return res


def _worker(args):
  style_fn, v, spec = args
  return simulate_cycle(style_fn(), v, spec)


def build_table(style_fn, grid, spec: CycleSpec, workers=24):
  """grid: (vx_values, vy_values, wz_values) -> dict of arrays (Nx,Ny,Nw,Nphi,...)."""
  vx, vy, wz = grid
  cmds = [(a, b, c) for a in vx for b in vy for c in wz]
  with mp.get_context("fork").Pool(workers) as pool:
    cycles = pool.map(_worker, [(style_fn, v, spec) for v in cmds])
  shape = (len(vx), len(vy), len(wz))
  table = {}
  for k in cycles[0]:
    table[k] = np.stack([c[k] for c in cycles]).reshape(shape + cycles[0][k].shape)
  table["vx"], table["vy"], table["wz"] = np.array(vx), np.array(vy), np.array(wz)
  return table
