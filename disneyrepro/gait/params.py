"""Gait samples, styles and the velocity-based phase-space sample blending.

Hopkins et al., Sec. 4.3 (animation parameters) and Sec. 5 (gait sampler).
A sample S = (Pi_S, v_S) describes a periodic two-step walk cycle that starts
at left lift-off (LLO): left swing LS = (0, t_LTD), right swing
RS = (t_RLO, t_RTD), cycle CY = (0, T_S).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field

import numpy as np

from disneyrepro.gait.curves import Curve, PhaseMap, f_soft

FOOT_KEYS = ("x", "y", "z", "yaw", "heel", "toe", "roll")
CY_KEYS = ("pelvis_z", "pelvis_yaw", "pelvis_pitch", "pelvis_roll")
SAMPLE_NAMES = (
  "forward_fast", "forward_slow", "in_place", "reverse_slow", "reverse_fast",
  "left_strafe", "right_strafe", "left_turn", "right_turn",
)


@dataclass
class GaitSample:
  velocity: tuple[float, float, float]
  T: float
  t_ltd: float
  t_rlo: float
  t_rtd: float
  stance_width: float
  stance_splay: float = 0.0
  stance_x_bias: float = 0.0
  dt_psi_lo: float = 0.0
  dt_psi_td: float = 0.0
  curves: dict = field(default_factory=dict)

  def __post_init__(self):
    for k in CY_KEYS:
      if k not in self.curves:
        self.curves[k] = Curve.constant(0.0)
    for s in ("l", "r"):
      for k in FOOT_KEYS:
        key = f"{s}foot_{k}"
        if key not in self.curves:
          self.curves[key] = Curve([], clamped=True) if k not in ("heel", "toe") else Curve([(0, 0), (1, 0)])

  def events(self):
    return np.array([0.0, self.t_ltd, self.t_rlo, self.t_rtd, self.T])

  def validate(self, min_ss=0.05, min_ds=0.0):
    t = self.events()
    assert np.all(np.diff(t) > 0), f"event ordering violated: {t}"
    assert self.t_ltd >= min_ss and self.t_rtd - self.t_rlo >= min_ss
    assert self.t_rlo - self.t_ltd >= min_ds and self.T - self.t_rtd >= min_ds

  def to_dict(self):
    d = {k: getattr(self, k) for k in ("velocity", "T", "t_ltd", "t_rlo", "t_rtd", "stance_width", "stance_splay", "stance_x_bias", "dt_psi_lo", "dt_psi_td")}
    d["curves"] = {k: c.to_dict() for k, c in self.curves.items()}
    return d

  @staticmethod
  def from_dict(d):
    d = dict(d)
    d["curves"] = {k: Curve.from_dict(c) for k, c in d["curves"].items()}
    d["velocity"] = tuple(d["velocity"])
    return GaitSample(**d)


class BlendedGait:
  """Convex combination of samples evaluated in phase space (Sec. 5.2)."""

  def __init__(self, items):
    self.items = [(s, w) for s, w in items if w > 1e-9]
    tot = sum(w for _, w in self.items)
    self.items = [(s, w / tot) for s, w in self.items]
    self.maps = [PhaseMap(s.events()) for s, _ in self.items]
    self.t_events = sum(w * s.events() for s, w in self.items)
    self.phase_map = PhaseMap(self.t_events)

  @property
  def T(self):
    return self.t_events[-1]

  def attr(self, name):
    return sum(w * getattr(s, name) for s, w in self.items)

  def phase_rate(self, phi):
    return 1.0 / max(self.phase_map.deriv(phi), 1e-4)

  def curve(self, name, phi):
    """Value of a function curve at gait phase phi in [0, 1)."""
    out = 0.0
    side = name[0] if name[1:5] == "foot" else None
    for (s, w), pm in zip(self.items, self.maps):
      t = pm(phi)
      if side is None:  # cycle curve
        u = t / s.T
      elif side == "l":
        u = t / s.t_ltd if phi <= 0.1 else (0.0 if phi > 0.5 else 1.0)
      else:
        u = (t - s.t_rlo) / (s.t_rtd - s.t_rlo) if 0.5 <= phi <= 0.6 else (1.0 if phi > 0.6 else 0.0)
      c = s.curves.get(name)
      if c is None:
        continue
      out += w * c(u)
    return out

  def has_curve(self, name):
    return any(name in s.curves for s, _ in self.items)


@dataclass
class GaitStyle:
  name: str
  samples: dict
  eta: float = 0.5
  upper_joints: tuple = ()

  # --------------------------------------------------------------- blending
  def blend(self, v):
    """Three-step omnidirectional interpolation (Fig. 9, Eqs. 8-10)."""
    vx, vy, wz = v
    S = self.samples
    fr = [S[n] for n in ("reverse_fast", "reverse_slow", "in_place", "forward_slow", "forward_fast")]
    xs = [s.velocity[0] for s in fr]
    vxc = min(max(vx, xs[0]), xs[-1])
    i = int(min(max(np.searchsorted(xs, vxc, side="right") - 1, 0), 3))
    a = f_soft((vxc - xs[i]) / (xs[i + 1] - xs[i]))
    w_fr = {id(fr[i]): (fr[i], 1 - a), id(fr[i + 1]): (fr[i + 1], a)}

    ymax_l, ymax_r = S["left_strafe"].velocity[1], -S["right_strafe"].velocity[1]
    wmax_l, wmax_r = S["left_turn"].velocity[2], -S["right_turn"].velocity[2]
    yn = vy / ymax_l if vy >= 0 else vy / ymax_r
    wn = wz / wmax_l if wz >= 0 else wz / wmax_r
    rho = min(math.hypot(yn, wn), 1.0)
    beta = math.atan2(yn, wn) % (2 * math.pi)
    ring = [S["left_turn"], S["left_strafe"], S["right_turn"], S["right_strafe"], S["left_turn"]]
    k = int(min(beta // (math.pi / 2), 3))
    ab = f_soft((beta - k * math.pi / 2) / (math.pi / 2))
    w_ts = {}
    for s, w in ((ring[k], 1 - ab), (ring[k + 1], ab)):
      prev = w_ts.get(id(s), (s, 0.0))[1]
      w_ts[id(s)] = (s, prev + w)

    xmax = xs[-1] if vx >= 0 else -xs[0]
    xn = abs(vxc) / max(xmax, 1e-6)
    a_ts = f_soft(rho) * (1 - self.eta * f_soft(xn))
    total = {}
    for d, scale in ((w_fr, 1 - a_ts), (w_ts, a_ts)):
      for key, (s, w) in d.items():
        prev = total.get(key, (s, 0.0))[1]
        total[key] = (s, prev + scale * w)
    return BlendedGait(list(total.values()))

  def max_velocity(self):
    S = self.samples
    return (S["forward_fast"].velocity[0], S["reverse_fast"].velocity[0], S["left_strafe"].velocity[1], S["left_turn"].velocity[2])

  # -------------------------------------------------------------------- I/O
  def to_json(self, path):
    d = {"name": self.name, "eta": self.eta, "upper_joints": list(self.upper_joints), "samples": {k: s.to_dict() for k, s in self.samples.items()}}
    with open(path, "w") as f:
      json.dump(d, f, indent=1)

  @staticmethod
  def from_json(path):
    d = json.load(open(path))
    return GaitStyle(d["name"], {k: GaitSample.from_dict(s) for k, s in d["samples"].items()}, d.get("eta", 0.5), tuple(d.get("upper_joints", ())))

  def copy(self):
    return copy.deepcopy(self)
