"""Centre-of-mass MPC of Hopkins et al. (Appendix, Eqs. 11-14).

Horizontal CoM dynamics with constant height and angular momentum (LIPM):
  xdd = omega^2 (x - x_cop).
Two CoP waypoints per support interval (middle: inside the active polygon,
end: inside the intersection with the next polygon), piecewise-linear CoP,
objective = final capture point near the last support centroid + waypoints
near polygon centroids + CoP velocity. Integration step 0.02 s.
"""

from __future__ import annotations

import math

import numpy as np
import proxsuite


def convex_hull(pts):
  pts = sorted(map(tuple, pts))
  if len(pts) <= 2:
    return np.array(pts)

  def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  lower, upper = [], []
  for p in pts:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
      lower.pop()
    lower.append(p)
  for p in reversed(pts):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
      upper.pop()
    upper.append(p)
  return np.array(lower[:-1] + upper[:-1])


def halfspaces(hull, margin=0.0):
  """CCW hull -> (N, c) with N p <= c."""
  n = len(hull)
  Ns, cs = [], []
  for i in range(n):
    a, b = hull[i], hull[(i + 1) % n]
    nv = np.array([b[1] - a[1], -(b[0] - a[0])])
    L = np.linalg.norm(nv)
    if L < 1e-9:
      continue
    nv /= L
    Ns.append(nv)
    cs.append(nv @ a - margin)
  return np.array(Ns), np.array(cs)


def foot_corners(pose, foot):
  x, y, yaw = pose
  c, s = math.cos(yaw), math.sin(yaw)
  hw = max(foot.half_width, 0.006)
  out = []
  for lx in (foot.heel_x, foot.toe_x):
    for ly in (-hw, hw):
      out.append((x + c * lx - s * ly, y + s * lx + c * ly))
  return out


class ComMPC:
  def __init__(self, com_height, foot, dt=0.02, g=9.81, w_T=100.0, w_c=1.0, w_v=0.01, margin=0.005):
    self.omega = math.sqrt(g / com_height)
    self.foot, self.dt = foot, dt
    self.w_T, self.w_c, self.w_v, self.margin = w_T, w_c, w_v, margin
    w, h = self.omega, dt
    ch, sh = math.cosh(w * h), math.sinh(w * h)
    self.A = np.array([[ch, sh / w], [w * sh, ch]])
    self.B = np.array([1 - ch, -w * sh])
    self.xi0 = None
    self.wp_t = None

  def _polygon(self, feet):
    pts = []
    for _, pose in feet:
      pts += foot_corners(pose, self.foot)
    return convex_hull(np.array(pts))

  def plan(self, xi0, p0, supports):
    """xi0 = (x, y, vx, vy); p0 = current CoP (x, y); supports from the planner."""
    supports = [s for s in supports if s[1] - s[0] > 1e-3 and len(s[2]) > 0]
    polys = [self._polygon(s[2]) for s in supports]
    wp_t, cons, cents = [], [], []
    for k, (t0, t1, _) in enumerate(supports):
      Hk = halfspaces(polys[k], self.margin)
      wp_t.append(0.5 * (t0 + t1))
      cons.append(Hk)
      cents.append(polys[k].mean(0))
      if k + 1 < len(supports):
        Hn = halfspaces(polys[k + 1], self.margin)
        cons.append((np.vstack([Hk[0], Hn[0]]), np.concatenate([Hk[1], Hn[1]])))
        both = np.vstack([polys[k], polys[k + 1]])
        cents.append(0.5 * (polys[k].mean(0) + polys[k + 1].mean(0)) if len(polys[k + 1]) else both.mean(0))
      else:
        cons.append(Hk)
        cents.append(polys[k].mean(0))
      wp_t.append(t1)
    N = len(wp_t)
    T = wp_t[-1]
    M = max(1, int(round(T / self.dt)))
    times = np.concatenate([[0.0], wp_t])
    # Interpolation weights of CoP at substep midpoints: p_j = L[j, 0] p0 + L[j, 1:] z.
    L = np.zeros((M, N + 1))
    for j in range(M):
      t = (j + 0.5) * T / M
      i = int(min(max(np.searchsorted(times, t, side="right") - 1, 0), N - 1))
      a = (t - times[i]) / max(times[i + 1] - times[i], 1e-6)
      L[j, i] += 1 - a
      L[j, i + 1] += a
    h = T / M
    w = self.omega
    # The capture point (divergent component) xi = x + xdot/omega obeys
    # xidot = omega (xi - p). Integrating it backwards in time from the free
    # final value xi_T is numerically stable (the paper's reverse-time
    # integration): xi_0 = e^{-omega T} xi_T + sum_j gamma_j p_j.
    decay = math.exp(-w * h)
    gamma = np.array([math.exp(-w * j * h) * (1 - decay) for j in range(M)])
    gl = gamma @ L  # weights on [p0, z]
    eT = math.exp(-w * T)
    # Decision x = [zx (N), zy (N), xiTx, xiTy].
    nz = 2 * N + 2
    P = np.zeros((nz, nz))
    q = np.zeros(nz)
    cT = cents[-1]
    Aeq = np.zeros((2, nz))
    beq = np.zeros(2)
    for ax in range(2):
      sl = slice(ax * N, (ax + 1) * N)
      xi_init = xi0[ax] + xi0[2 + ax] / w
      Aeq[ax, sl] = gl[1:]
      Aeq[ax, 2 * N + ax] = eT
      beq[ax] = xi_init - gl[0] * p0[ax]
      P[2 * N + ax, 2 * N + ax] += self.w_T
      q[2 * N + ax] -= self.w_T * cT[ax]
      P[sl, sl] += self.w_c * np.eye(N)
      q[sl] -= self.w_c * np.array([c[ax] for c in cents])
      # CoP rate penalty on consecutive waypoints (including p0).
      F = np.zeros((N, N))
      f0 = np.zeros(N)
      for i in range(N):
        dtw = max(times[i + 1] - times[i], 1e-2)
        F[i, i] = 1.0 / dtw
        if i > 0:
          F[i, i - 1] = -1.0 / dtw
        else:
          f0[i] = -p0[ax] / dtw
      P[sl, sl] += self.w_v * F.T @ F
      q[sl] += self.w_v * F.T @ f0
    rows, rhs = [], []
    for i, (Nh, c) in enumerate(cons):
      for n_, c_ in zip(Nh, c):
        r = np.zeros(nz)
        r[i], r[N + i] = n_[0], n_[1]
        rows.append(r)
        rhs.append(c_)
    C = np.array(rows)
    u = np.array(rhs)
    qp = proxsuite.proxqp.dense.QP(nz, 2, len(rows))
    qp.settings.eps_abs = 1e-7
    qp.settings.verbose = False
    qp.settings.max_iter = 2000
    qp.init(P + 1e-9 * np.eye(nz), q, Aeq, beq, C, -1e10 * np.ones(len(rows)), u)
    qp.solve()
    z = qp.results.x
    self.xi0 = np.asarray(xi0, float).copy()
    self.p0 = np.asarray(p0, float).copy()
    self.wp_t = times
    self.wp = np.vstack([self.p0, np.stack([z[:N], z[N : 2 * N]], -1)])
    self.xi_T = z[2 * N :]
    self.init_residual = Aeq @ z - beq
    self.status = qp.results.info.status
    return self.wp

  def cop(self, t):
    times = self.wp_t
    t = min(max(t, 0.0), times[-1])
    i = int(min(max(np.searchsorted(times, t, side="right") - 1, 0), len(times) - 2))
    a = (t - times[i]) / max(times[i + 1] - times[i], 1e-6)
    return (1 - a) * self.wp[i] + a * self.wp[i + 1]

  def state(self, t, h=0.002):
    """Reference CoM position, velocity, acceleration and CoP at time t after the plan."""
    x = self.xi0[:2].copy()
    v = self.xi0[2:].copy()
    w2 = self.omega**2
    n = int(t / h)
    tt = 0.0
    for _ in range(n):
      p = self.cop(tt + 0.5 * h)
      a = w2 * (x - p)
      x = x + v * h + 0.5 * a * h * h
      v = v + a * h
      tt += h
    rem = t - tt
    if rem > 1e-9:
      p = self.cop(tt + 0.5 * rem)
      a = w2 * (x - p)
      x = x + v * rem + 0.5 * a * rem * rem
      v = v + a * rem
    p = self.cop(t)
    return x, v, w2 * (x - p), p
