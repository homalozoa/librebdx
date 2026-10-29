"""GPU look-up of the dense walking reference table: trilinear in the path
velocity command (vx, vy, wz) and linear (circular) in the gait phase."""

from __future__ import annotations

import numpy as np
import torch


class WalkTable:
  KEYS = ("root_xy", "root_z", "root_quat", "q", "contact", "rate")

  def __init__(self, path, device):
    d = np.load(path)
    self.device = device
    self.grid = [torch.tensor(d[k], dtype=torch.float32, device=device) for k in ("vx", "vy", "wz")]
    self.data = {}
    for k in self.KEYS:
      a = torch.tensor(d[k], dtype=torch.float32, device=device)
      if a.ndim == 4:
        a = a[..., None]
      self.data[k] = a  # (Nx, Ny, Nw, Nphi, D)
    # All fields packed along the last axis, so a query is 16 gathers instead of one
    # set per field (the per-field loop was launch-bound in training).
    self.slices, c = {}, 0
    for k, a in self.data.items():
      self.slices[k] = (c, c + a.shape[-1])
      c += a.shape[-1]
    self.packed = torch.cat(list(self.data.values()), -1).contiguous()
    self.nphi = self.data["rate"].shape[3]
    self.joint_names = [str(n) for n in d["joint_names"]]
    self.lo = torch.stack([g[0] for g in self.grid])
    self.hi = torch.stack([g[-1] for g in self.grid])

  def _idx(self, g, x):
    x = torch.clamp(x, g[0], g[-1])
    i = torch.clamp(torch.searchsorted(g, x, right=True) - 1, 0, len(g) - 2)
    a = (x - g[i]) / (g[i + 1] - g[i])
    return i, a

  def query(self, v, phi):
    """v (B,3), phi (B,) in [0,1) -> dict of (B, D) tensors."""
    (i, a), (j, b), (k, c) = (self._idx(g, v[:, n]) for n, g in enumerate(self.grid))
    p = phi * self.nphi
    l0 = torch.floor(p).long() % self.nphi
    e = p - torch.floor(p)
    l1 = (l0 + 1) % self.nphi
    A = self.packed
    acc = 0.0
    e = e[:, None]
    for di, wi in ((0, 1 - a), (1, a)):
      for dj, wj in ((0, 1 - b), (1, b)):
        for dk, wk in ((0, 1 - c), (1, c)):
          w = (wi * wj * wk)[:, None]
          ii, jj, kk = i + di, j + dj, k + dk
          acc = acc + w * torch.lerp(A[ii, jj, kk, l0], A[ii, jj, kk, l1], e)  # (B, C)
    out = {key: acc[:, s0:s1] for key, (s0, s1) in self.slices.items()}
    q = out["root_quat"]
    out["root_quat"] = q / q.norm(dim=-1, keepdim=True)
    out["rate"] = out["rate"][:, 0]
    return out
