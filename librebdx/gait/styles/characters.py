"""Walk styles for the BDX-R character and Olaf, authored with the gait sampler.

BDX (Grandia et al.): top speeds 0.7 m/s forward, 0.4 m/s lateral and
1.8 rad/s turning on the paper's robot. BDX-R's hip-yaw range is only
+-0.26 rad, so its turning rate is limited to 1.0 rad/s here. Head bob at
twice the gait frequency (the reason for the second phase harmonic).
Olaf (Mueller et al.): characteristic heel-toe walk (toe-off before lift-off,
heel-strike at touch-down), small snowball feet, slow pace.
"""

import math

from librebdx.gait.authoring import free, periodic, sample, style_from, swing

TAU = 2 * math.pi

# ----------------------------------------------------------------------- BDX
BDX_Z = 0.222
BDX_W = 0.216
BDX_NECK = ("Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll")
BDX_NECK0 = (0.55, -0.55, 0.0, 0.0)


def _bdx_upper(a):
  n0, h0, _, _ = BDX_NECK0
  return {
    "Neck_Pitch": periodic(lambda u: n0 + 0.06 * a * math.sin(2 * TAU * u)),
    "Head_Pitch": periodic(lambda u: h0 - 0.08 * a * math.sin(2 * TAU * u + 0.4)),
    "Head_Yaw": periodic(lambda u: 0.05 * a * math.sin(TAU * u)),
    "Head_Roll": periodic(lambda u: -0.06 * a * math.sin(TAU * u + 0.3)),
  }


def _bdx_pelvis(a):
  return dict(
    pelvis_z=periodic(lambda u: BDX_Z - 0.006 * a * math.cos(2 * TAU * u)),
    pelvis_curves={
      "pelvis_roll": periodic(lambda u: 0.04 * a * math.sin(TAU * u)),
      "pelvis_yaw": periodic(lambda u: 0.05 * a * math.sin(TAU * u + 0.2)),
      "pelvis_pitch": periodic(lambda u: 0.02 * a * math.sin(2 * TAU * u)),
    },
  )


def bdx_walk(calm=False):
  def s(v, T, a=1.0, h=0.05, width=BDX_W):
    return sample(v, T, 0.42, width=width, swing_height=h, upper=_bdx_upper(a), **_bdx_pelvis(a))

  if calm:
    # A small character's unhurried walk: the cadence grows with speed, from 2 steps/s in
    # place to about 4 steps/s at 0.7 m/s (a hurried trot), instead of 3.2-4.4 steps/s.
    S = {
      "in_place": s((0, 0, 0), 1.0, 0.5, 0.04),
      "forward_slow": s((0.25, 0, 0), 0.92, 0.65, 0.045),
      "forward_mid": s((0.45, 0, 0), 0.7, 0.85, 0.05),
      "forward_fast": s((0.7, 0, 0), 0.48, 1.0, 0.055),
      "reverse_slow": s((-0.15, 0, 0), 0.9, 0.6),
      "reverse_fast": s((-0.3, 0, 0), 0.7, 0.7),
      "left_strafe": s((0, 0.4, 0), 0.46, 0.7, 0.05, BDX_W + 0.02),  # sideways steps stay short: leg reach
      "right_strafe": s((0, -0.4, 0), 0.46, 0.7, 0.05, BDX_W + 0.02),
      # Turning on the spot at 1.8 rad/s (Sec. VII-A): 0.56 s cycles, so each step
      # yaws 0.5 rad, i.e. +-0.25 rad of hip yaw per leg.
      "left_turn": s((0, 0, 1.8), 0.56, 0.7),
      "right_turn": s((0, 0, -1.8), 0.56, 0.7),
    }
    return style_from("bdx_walk_calm", S, BDX_NECK)

  S = {
    "in_place": s((0, 0, 0), 0.62, 0.6, 0.045),
    "forward_slow": s((0.35, 0, 0), 0.55, 0.8),
    "forward_fast": s((0.7, 0, 0), 0.46, 1.0, 0.055),
    "reverse_slow": s((-0.15, 0, 0), 0.6, 0.6),
    "reverse_fast": s((-0.3, 0, 0), 0.55, 0.7),
    "left_strafe": s((0, 0.4, 0), 0.52, 0.7, 0.05, BDX_W + 0.02),
    "right_strafe": s((0, -0.4, 0), 0.52, 0.7, 0.05, BDX_W + 0.02),
    "left_turn": s((0, 0, 1.0), 0.55, 0.7),
    "right_turn": s((0, 0, -1.0), 0.55, 0.7),
  }
  return style_from("bdx_walk", S, BDX_NECK)


# ---------------------------------------------------------------------- Olaf
OLAF_Z = 0.335
OLAF_W = 0.15
OLAF_NECK = ("neck_yaw", "neck_pitch", "neck_roll")


def _olaf_upper(a):
  return {
    "neck_yaw": periodic(lambda u: 0.06 * a * math.sin(TAU * u)),
    "neck_pitch": periodic(lambda u: 0.04 * a * math.sin(2 * TAU * u + 0.5)),
    "neck_roll": periodic(lambda u: -0.07 * a * math.sin(TAU * u + 0.4)),
  }


def _olaf_feet(heel_toe=True, toe=0.3, heel=-0.25):
  if not heel_toe:
    return {}
  out = {}
  for side in ("l", "r"):
    # Toe pitch at lift-off decays early in swing; heel pitch builds up for heel-strike.
    out[f"{side}foot_toe"] = free(lambda u: toe * max(0.0, 1 - u / 0.35) ** 2)
    out[f"{side}foot_heel"] = free(lambda u: heel * max(0.0, (u - 0.6) / 0.4) ** 1.5)
    out[f"{side}foot_z"] = swing(lambda u: 0.045 * math.sin(math.pi * u) ** 1.2)
  return out


def olaf_walk(heel_toe=True):
  def s(v, T, a=1.0, width=OLAF_W):
    return sample(
      v, T, 0.40, width=width, swing_height=0.04, upper=_olaf_upper(a), foot_curves=_olaf_feet(heel_toe),
      dt_psi_lo=0.12 if heel_toe else 0.0, dt_psi_td=0.10 if heel_toe else 0.0,
      pelvis_z=periodic(lambda u: OLAF_Z - 0.008 * a * math.cos(2 * TAU * u)),
      pelvis_curves={
        "pelvis_roll": periodic(lambda u: 0.05 * a * math.sin(TAU * u)),
        "pelvis_yaw": periodic(lambda u: 0.06 * a * math.sin(TAU * u + 0.2)),
      },
    )

  S = {
    "in_place": s((0, 0, 0), 0.8, 0.6),
    "forward_slow": s((0.15, 0, 0), 0.75, 0.8),
    "forward_fast": s((0.3, 0, 0), 0.65, 1.0),
    "reverse_slow": s((-0.08, 0, 0), 0.8, 0.6),
    "reverse_fast": s((-0.15, 0, 0), 0.75, 0.7),
    "left_strafe": s((0, 0.12, 0), 0.75, 0.7, OLAF_W + 0.02),
    "right_strafe": s((0, -0.12, 0), 0.75, 0.7, OLAF_W + 0.02),
    "left_turn": s((0, 0, 0.6), 0.75, 0.7),
    "right_turn": s((0, 0, -0.6), 0.75, 0.7),
  }
  return style_from("olaf_walk" if heel_toe else "olaf_walk_flat", S, OLAF_NECK)


def bdx_walk_calm():
  return bdx_walk(calm=True)
