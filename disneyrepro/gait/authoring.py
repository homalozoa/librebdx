"""Helpers to author gait samples in code (what the GUI editor does interactively)."""

from __future__ import annotations

import math

import numpy as np

from disneyrepro.gait.curves import Curve
from disneyrepro.gait.params import GaitSample, GaitStyle, SAMPLE_NAMES


def periodic(fn, n=16):
  """Periodic curve on CY from a function of u in [0, 1)."""
  return Curve([(i / n, fn(i / n)) for i in range(n)], periodic=True)


def swing(fn, n=8):
  """Swing curve on LS/RS, clamped to zero at both ends."""
  return Curve([(i / n, fn(i / n)) for i in range(1, n)], clamped=True)


def free(fn, n=8):
  """Unclamped swing curve (heel / toe pitch may be non-zero at the boundaries)."""
  return Curve([(i / n, fn(i / n)) for i in range(n + 1)])


def bump(height, peak=0.5, sharp=1.0):
  """Smooth swing-height profile peaking at `peak`."""

  def f(u):
    if u <= peak:
      s = u / peak
    else:
      s = (1 - u) / (1 - peak)
    return height * (math.sin(0.5 * math.pi * s) ** (2 * sharp))

  return f


def sample(v, T, swing_frac=0.4, left_swing=None, right_swing=None, width=0.17, splay=0.0, x_bias=0.0,
           pelvis_z=0.38, pelvis_curves=None, swing_height=0.04, foot_curves=None, upper=None,
           dt_psi_lo=0.0, dt_psi_td=0.0):
  """A symmetric (or asymmetric via left_swing/right_swing fractions) walk cycle."""
  ls = (left_swing if left_swing is not None else swing_frac) * T
  rs = (right_swing if right_swing is not None else swing_frac) * T
  ds_total = T - ls - rs
  t_ltd = ls
  t_rlo = ls + 0.5 * ds_total
  t_rtd = t_rlo + rs
  curves = {"pelvis_z": pelvis_z if isinstance(pelvis_z, Curve) else Curve.constant(pelvis_z)}
  for k, c in (pelvis_curves or {}).items():
    curves[k] = c
  for s in ("l", "r"):
    curves[f"{s}foot_z"] = swing(bump(swing_height))
  for k, c in (foot_curves or {}).items():
    curves[k] = c
  for j, c in (upper or {}).items():
    curves[f"joint:{j}"] = c
  smp = GaitSample(velocity=tuple(v), T=T, t_ltd=t_ltd, t_rlo=t_rlo, t_rtd=t_rtd, stance_width=width,
                   stance_splay=splay, stance_x_bias=x_bias, dt_psi_lo=dt_psi_lo, dt_psi_td=dt_psi_td, curves=curves)
  smp.validate()
  return smp


def style_from(name, samples: dict, upper_joints=(), eta=0.5):
  missing = set(SAMPLE_NAMES) - set(samples)
  assert not missing, missing
  return GaitStyle(name, samples, eta, tuple(upper_joints))
