"""Densely sample the BDX standing IK and fit the reference interpolant."""

import time
from pathlib import Path

import torch

from disneyrepro.bdx.standing_ref import BdxStandingReference
from disneyrepro.common.fitnet import FitNet, fit

OUT = Path("disneyrepro/bdx/data")
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
ref = BdxStandingReference(device=dev)
N, B = 1_200_000, 16384
gs, xs, rs = [], [], []
t0 = time.time()
for i in range(0, N, B):
  g = ref.sample_commands(B)
  # Put 15% of each dimension exactly at nominal (many runtime commands are 0).
  mask = torch.rand_like(g) < 0.15
  g = torch.where(mask, torch.zeros_like(g), g)
  # And 5% of samples at a box corner/edge to cover the extremes.
  edge = torch.rand(B, 1, device=dev) < 0.05
  sgn = torch.where(torch.rand_like(g) < 0.5, ref.cmd_lo, ref.cmd_hi)
  g = torch.where(edge & (torch.rand_like(g) < 0.5), sgn, g)
  x, out = ref.solve(g, iters=25)
  gs.append(g)
  xs.append(x)
  rs.append(out["residual"].abs().max(-1).values)
  if i % (B * 10) == 0:
    print(f"{i}/{N} {time.time()-t0:.0f}s", flush=True)
G, X, Rmax = torch.cat(gs), torch.cat(xs), torch.cat(rs)
print("residual quantiles", Rmax.quantile(torch.tensor([0.5, 0.9, 0.99], device=dev)).tolist())
torch.save({"g": G.cpu(), "x": X.cpu(), "res": Rmax.cpu()}, OUT / "standing_samples.pt")

net = FitNet(ref.cmd_lo, ref.cmd_hi, X.mean(0), X.std(0).clamp(min=1e-3), hidden=(256, 256, 256)).to(dev)
ntr = int(0.95 * len(G))
fit(net, G[:ntr], X[:ntr], steps=30000)
with torch.no_grad():
  err = (net(G[ntr:]) - X[ntr:]).abs()
print("held-out abs err: root xy [m] mean %.2e p99 %.2e | joints [rad] mean %.2e p99 %.2e max %.2e" % (
  err[:, :2].mean(), err[:, :2].quantile(0.99), err[:, 2:].mean(), err[:, 2:].flatten().quantile(0.99), err[:, 2:].max()))
net.save(OUT / "standing_net.pt")
