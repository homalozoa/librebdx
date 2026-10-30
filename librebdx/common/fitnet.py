"""Small MLP used as a smooth interpolant of densely sampled reference generators.

The papers densely sample their (slow) reference generators offline and
interpolate during RL. For high-dimensional command spaces (8-D standing
commands) we fit an MLP to the samples instead of a tensor grid.
"""

from __future__ import annotations

import torch
from torch import nn


class FitNet(nn.Module):
  def __init__(self, in_lo, in_hi, out_mean, out_std, hidden=(256, 256, 256)):
    super().__init__()
    self.register_buffer("in_lo", torch.as_tensor(in_lo, dtype=torch.float32))
    self.register_buffer("in_hi", torch.as_tensor(in_hi, dtype=torch.float32))
    self.register_buffer("out_mean", torch.as_tensor(out_mean, dtype=torch.float32))
    self.register_buffer("out_std", torch.as_tensor(out_std, dtype=torch.float32))
    dims = [len(in_lo), *hidden, len(out_mean)]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
      layers += [nn.Linear(a, b), nn.SiLU()]
    self.net = nn.Sequential(*layers[:-1])
    self.hidden = hidden

  def forward(self, x):
    u = 2 * (x - self.in_lo) / (self.in_hi - self.in_lo) - 1
    return self.net(u) * self.out_std + self.out_mean

  def save(self, path):
    torch.save({"state": self.state_dict(), "hidden": self.hidden, "nin": len(self.in_lo), "nout": len(self.out_mean)}, path)

  @staticmethod
  def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    z = torch.zeros
    net = FitNet(z(ck["nin"]), torch.ones(ck["nin"]), z(ck["nout"]), torch.ones(ck["nout"]), tuple(ck["hidden"]))
    net.load_state_dict(ck["state"])
    return net.to(device).eval()


def fit(net, X, Y, steps=20000, batch=8192, lr=2e-3, log_every=2000):
  opt = torch.optim.Adam(net.parameters(), lr=lr)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
  n = X.shape[0]
  for it in range(steps):
    idx = torch.randint(0, n, (batch,), device=X.device)
    loss = ((net(X[idx]) - Y[idx]) / net.out_std).square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if it % log_every == 0 or it == steps - 1:
      print(f"  fit step {it} loss {loss.item():.2e}", flush=True)
  return net
