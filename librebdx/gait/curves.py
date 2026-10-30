"""Function curves (cubic Hermite splines) and the gait phase map.

Curves are defined on a normalised interval u in [0, 1] (the CY / LS / RS
interval of a gait sample), so resizing an interval rescales the knot times as
in the editor of Hopkins et al. (Sec. 6.1).
"""

from __future__ import annotations

import numpy as np


class Curve:
  """Cubic Hermite spline through knots (u_i, v_i) with optional tangents.

  periodic=True: C1-continuous wrap-around (pelvis / upper-body curves on CY).
  clamped=True: value forced to zero at u=0 and u=1 (swing curves).
  Missing tangents are Catmull-Rom.
  """

  def __init__(self, knots, periodic=False, clamped=False, tangents=None):
    k = sorted((float(u), float(v)) for u, v in knots)
    if clamped:
      k = [(u, v) for u, v in k if 0.0 < u < 1.0]
      k = [(0.0, 0.0)] + k + [(1.0, 0.0)]
    if periodic:
      k = [(u, v) for u, v in k if u < 1.0]
    self.u = np.array([a for a, _ in k])
    self.v = np.array([b for _, b in k])
    self.periodic, self.clamped = periodic, clamped
    n = len(self.u)
    if tangents is not None:
      self.m = np.asarray(tangents, float)
    elif n == 1:
      self.m = np.zeros(1)
    else:
      self.m = np.zeros(n)
      for i in range(n):
        if periodic:
          up, vp = (self.u[i - 1] - (1.0 if i == 0 else 0.0)), self.v[i - 1]
          un, vn = (self.u[(i + 1) % n] + (1.0 if i == n - 1 else 0.0)), self.v[(i + 1) % n]
          self.m[i] = (vn - vp) / (un - up)
        elif 0 < i < n - 1:
          self.m[i] = (self.v[i + 1] - self.v[i - 1]) / (self.u[i + 1] - self.u[i - 1])
        else:
          self.m[i] = 0.0

  @staticmethod
  def constant(value=0.0, periodic=True):
    return Curve([(0.0, value)], periodic=periodic)

  def __call__(self, u):
    u = float(u)
    n = len(self.u)
    if n == 1:
      return float(self.v[0])
    if self.periodic:
      u = u % 1.0
      i = int(np.searchsorted(self.u, u, side="right") - 1)
      if i < 0:  # before first knot: segment from last knot (shifted by -1)
        u0, v0, m0 = self.u[-1] - 1.0, self.v[-1], self.m[-1]
        u1, v1, m1 = self.u[0], self.v[0], self.m[0]
      elif i == n - 1:
        u0, v0, m0 = self.u[-1], self.v[-1], self.m[-1]
        u1, v1, m1 = self.u[0] + 1.0, self.v[0], self.m[0]
      else:
        u0, v0, m0, u1, v1, m1 = self.u[i], self.v[i], self.m[i], self.u[i + 1], self.v[i + 1], self.m[i + 1]
    else:
      u = min(max(u, self.u[0]), self.u[-1])
      i = int(min(max(np.searchsorted(self.u, u, side="right") - 1, 0), n - 2))
      u0, v0, m0, u1, v1, m1 = self.u[i], self.v[i], self.m[i], self.u[i + 1], self.v[i + 1], self.m[i + 1]
    h = u1 - u0
    s = (u - u0) / h
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return float(h00 * v0 + h10 * h * m0 + h01 * v1 + h11 * h * m1)

  def to_dict(self):
    return {"knots": [[float(a), float(b)] for a, b in zip(self.u, self.v)], "periodic": self.periodic, "clamped": self.clamped}

  @staticmethod
  def from_dict(d):
    if isinstance(d, (int, float)):
      return Curve.constant(float(d))
    return Curve(d["knots"], periodic=d.get("periodic", False), clamped=d.get("clamped", False))


def f_soft(a):
  """Cubic smoothstep used for sample blend ratios (Eq. 8)."""
  a = min(max(a, 0.0), 1.0)
  return -2 * a**3 + 3 * a**2


# Event phases of the phase map (Sec. 5.1).
PHI_EVENTS = np.array([0.0, 0.1, 0.5, 0.6, 1.0])


class PhaseMap:
  """Monotone cubic (Fritsch-Carlson) map from phase phi to gait time t.

  Waypoints (0, 0), (0.1, t_LTD), (0.5, t_RLO), (0.6, t_RTD), (1, T_S).
  """

  def __init__(self, t_events):
    self.x = PHI_EVENTS
    self.y = np.asarray(t_events, float)
    d = np.diff(self.y) / np.diff(self.x)
    m = np.zeros(5)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, 4):
      m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 2.0 / (1.0 / d[i - 1] + 1.0 / d[i])
    # Periodic consistency of the slope at phi = 0 / 1 keeps the gait smooth.
    m0 = 2.0 / (1.0 / d[0] + 1.0 / d[-1])
    m[0] = m[-1] = m0
    for i in range(4):  # Fritsch-Carlson limiter
      a, b = m[i] / d[i], m[i + 1] / d[i]
      r = a * a + b * b
      if r > 9:
        t = 3 / np.sqrt(r)
        m[i], m[i + 1] = t * a * d[i], t * b * d[i]
    self.m = m

  def _seg(self, phi):
    return int(min(max(np.searchsorted(self.x, phi, side="right") - 1, 0), 3))

  def __call__(self, phi):
    i = self._seg(phi)
    x0, x1, y0, y1, m0, m1 = self.x[i], self.x[i + 1], self.y[i], self.y[i + 1], self.m[i], self.m[i + 1]
    h = x1 - x0
    s = (phi - x0) / h
    return (2 * s**3 - 3 * s**2 + 1) * y0 + (s**3 - 2 * s**2 + s) * h * m0 + (-2 * s**3 + 3 * s**2) * y1 + (s**3 - s**2) * h * m1

  def deriv(self, phi):
    i = self._seg(phi)
    x0, x1, y0, y1, m0, m1 = self.x[i], self.x[i + 1], self.y[i], self.y[i + 1], self.m[i], self.m[i + 1]
    h = x1 - x0
    s = (phi - x0) / h
    return ((6 * s**2 - 6 * s) * y0 + (3 * s**2 - 4 * s + 1) * h * m0 + (-6 * s**2 + 6 * s) * y1 + (3 * s**2 - 2 * s) * h * m1) / h

  def inverse(self, t):
    lo, hi = 0.0, 1.0
    for _ in range(40):
      mid = 0.5 * (lo + hi)
      if self(mid) < t:
        lo = mid
      else:
        hi = mid
    return 0.5 * (lo + hi)
