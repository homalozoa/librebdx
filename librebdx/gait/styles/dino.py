"""Dino gait and injured gait (Hopkins et al. Sec. 8.1, Fig. 12).

Dino gait: symmetric swing intervals; the cycle duration drops from 0.95 s
(in place) to 0.65 s (0.2 m/s) and 0.45 s (0.4 m/s), i.e. the stepping
frequency more than doubles. Neck animated like a walking bird (head bob at
twice the gait frequency), tail sway inspired by theropod gait simulations.
Injured gait: asymmetric swing intervals (shorter left swing) and a positive
stance x-bias at the In Place and Forward Slow samples; Forward Fast equals
the dino gait so the character blends back to a symmetric gait at speed.
"""

import math

from librebdx.gait.authoring import periodic, sample, style_from

UPPER = ("neck_pitch1", "neck_yaw", "neck_pitch2", "head_pitch", "tail_yaw1", "tail_pitch2", "tail_yaw3", "tail_pitch4")
WIDTH = 0.17
Z = 0.375
TAU = 2 * math.pi


def upper_curves(amp=1.0, neck=True):
  """Bird-like head bob (2 per cycle) and tail sway (1 per cycle)."""
  a = amp
  return {
    "neck_pitch1": periodic(lambda u: 0.10 * a * math.sin(2 * TAU * u)),
    "neck_yaw": periodic(lambda u: 0.05 * a * math.sin(TAU * u + 0.5)),
    "neck_pitch2": periodic(lambda u: -0.18 * a * math.sin(2 * TAU * u + 0.35)),
    "head_pitch": periodic(lambda u: 0.08 * a * math.sin(2 * TAU * u + 0.7)),
    "tail_yaw1": periodic(lambda u: 0.12 * a * math.sin(TAU * u)),
    "tail_pitch2": periodic(lambda u: 0.05 * a * math.sin(2 * TAU * u - 0.4)),
    "tail_yaw3": periodic(lambda u: 0.20 * a * math.sin(TAU * u - 0.9)),
    "tail_pitch4": periodic(lambda u: 0.08 * a * math.sin(2 * TAU * u - 0.9)),
  }


def pelvis(amp_z=0.006, amp_roll=0.03, amp_yaw=0.03, phase=0.0):
  return dict(
    pelvis_z=periodic(lambda u: Z - amp_z * math.cos(2 * TAU * u + phase)),
    pelvis_curves={
      "pelvis_roll": periodic(lambda u: amp_roll * math.sin(TAU * u)),
      "pelvis_yaw": periodic(lambda u: amp_yaw * math.sin(TAU * u + 0.3)),
      "pelvis_pitch": periodic(lambda u: 0.02 * math.sin(2 * TAU * u)),
    },
  )


def dino_gait():
  S = {}
  S["in_place"] = sample((0, 0, 0), 0.95, 0.40, width=WIDTH, swing_height=0.045, upper=upper_curves(0.6), **pelvis(0.004))
  S["forward_slow"] = sample((0.2, 0, 0), 0.65, 0.40, width=WIDTH, swing_height=0.05, upper=upper_curves(0.9), **pelvis(0.006))
  S["forward_fast"] = sample((0.4, 0, 0), 0.45, 0.40, width=WIDTH, swing_height=0.05, upper=upper_curves(1.0), **pelvis(0.008))
  S["reverse_slow"] = sample((-0.1, 0, 0), 0.8, 0.40, width=WIDTH, swing_height=0.04, upper=upper_curves(0.6), **pelvis(0.004))
  S["reverse_fast"] = sample((-0.2, 0, 0), 0.65, 0.40, width=WIDTH, swing_height=0.045, upper=upper_curves(0.7), **pelvis(0.005))
  S["left_strafe"] = sample((0, 0.15, 0), 0.75, 0.38, width=WIDTH + 0.01, swing_height=0.045, upper=upper_curves(0.6), **pelvis(0.004))
  S["right_strafe"] = sample((0, -0.15, 0), 0.75, 0.38, width=WIDTH + 0.01, swing_height=0.045, upper=upper_curves(0.6), **pelvis(0.004))
  S["left_turn"] = sample((0, 0, 0.8), 0.75, 0.40, width=WIDTH, swing_height=0.045, upper=upper_curves(0.6), **pelvis(0.004))
  S["right_turn"] = sample((0, 0, -0.8), 0.75, 0.40, width=WIDTH, swing_height=0.045, upper=upper_curves(0.6), **pelvis(0.004))
  return style_from("dino_gait", S, UPPER)


def injured_gait():
  st = dino_gait()
  S = dict(st.samples)
  # Shorter left swing + positive stance x-bias: reads as an injured right leg.
  S["in_place"] = sample((0, 0, 0), 0.95, left_swing=0.26, right_swing=0.46, width=WIDTH, x_bias=0.04,
                         swing_height=0.04, upper=upper_curves(0.5), **pelvis(0.012, amp_roll=0.06))
  S["forward_slow"] = sample((0.2, 0, 0), 0.70, left_swing=0.28, right_swing=0.46, width=WIDTH, x_bias=0.04,
                             swing_height=0.045, upper=upper_curves(0.7), **pelvis(0.012, amp_roll=0.06))
  return style_from("injured_gait", S, UPPER)
