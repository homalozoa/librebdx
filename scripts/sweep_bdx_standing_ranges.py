"""Sweep each standing command dimension and report where the IK stays exact."""

import torch

from disneyrepro.bdx.standing_ref import CMD_NAMES, BdxStandingReference

ref = BdxStandingReference(device="cuda")
for dim, name in enumerate(CMD_NAMES):
  vals = torch.linspace(-1.2, 1.2, 241, device="cuda")
  g = torch.zeros(len(vals), 8, device="cuda")
  g[:, dim] = vals
  _, out = ref.solve(g, iters=25)
  res = out["residual"].abs()
  ok = res.max(-1).values < 2e-3
  okleg = res[:, :12].max(-1).values < 2e-3
  v_ok = vals[ok]
  # Contiguous feasible interval around 0.
  i0 = int((vals.abs()).argmin())
  lo = hi = i0
  while lo > 0 and ok[lo - 1]:
    lo -= 1
  while hi < len(vals) - 1 and ok[hi + 1]:
    hi += 1
  lo2 = hi2 = i0
  while lo2 > 0 and okleg[lo2 - 1]:
    lo2 -= 1
  while hi2 < len(vals) - 1 and okleg[hi2 + 1]:
    hi2 += 1
  print(f"{name:12s} feasible [{vals[lo]:+.3f}, {vals[hi]:+.3f}]  legs-only [{vals[lo2]:+.3f}, {vals[hi2]:+.3f}]")
