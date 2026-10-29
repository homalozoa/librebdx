"""Batched damped Gauss-Newton least squares with box bounds (torch)."""

from __future__ import annotations

import torch
from torch.func import jacfwd, vmap


class BatchedLSQ:
  """Minimise ||r(x, aux)||^2 per batch element, lo <= x <= hi.

  residual_fn(x, *aux) must accept batched inputs x (B,n), aux (B,...) and
  return (B,m). Jacobians are computed exactly with forward-mode AD.
  """

  def __init__(self, residual_fn, lo, hi, damping=1e-4):
    self.f = residual_fn
    self.lo, self.hi = lo, hi
    self.damping = damping

    def single(x, *aux):
      return residual_fn(x.unsqueeze(0), *[a.unsqueeze(0) for a in aux])[0]

    self._jac = vmap(jacfwd(single, argnums=0))

  def step(self, x, *aux):
    r = self.f(x, *aux)
    J = self._jac(x, *aux)
    n = x.shape[-1]
    JT = J.transpose(-1, -2)
    H = JT @ J + self.damping * torch.eye(n, device=x.device, dtype=x.dtype)
    g = (JT @ r.unsqueeze(-1)).squeeze(-1)
    dx = -torch.linalg.solve(H, g)
    x = torch.clamp(x + dx, self.lo, self.hi)
    return x, r

  def solve(self, x, *aux, iters=10, tol=1e-6):
    for _ in range(iters):
      x, r = self.step(x, *aux)
      if r.square().sum(-1).max() < tol:
        break
    r = self.f(x, *aux)
    return x, r
