"""Olaf robot of Mueller et al. (2025), built from the paper's description.

Paper facts used: 88.7 cm tall (without hair), 14.9 kg; 6-DoF legs that are
identical (not mirrored) and asymmetric -- the left leg has a rear-facing hip
roll actuator and a forward knee, the right leg a forward hip roll actuator
and a rear-facing knee -- hidden in a soft lower snowball (skirt); small
snowball feet; a large heavy head on a slim 3-DoF neck. Show functions (2-DoF
shoulders, jaw, eyebrow, eyes) have low inertia and are not part of the RL
backbone; they are modelled as fixed masses here. Dimensions/masses not given
in the paper are our own estimates.

Run: python -m librebdx.robots.olaf.build
"""

from pathlib import Path

import mujoco
import numpy as np

from librebdx.robots import builder as B

OUT = Path(__file__).parent / "olaf.xml"

ROOT_Z = 0.33
THIGH, SHANK = 0.12, 0.12
KNEE_X = 0.045
FOOT_HALF = (0.07, 0.045, 0.012)  # sole box half sizes
FOOT_CENTRE = (0.01, 0.0, -0.035)  # in ankle-roll frame
COL_WORLD, COL_FOOT, COL_BODY = 1, 2, 4


def leg(root, side, mats):
  white, dark = mats
  s = 1.0 if side == "left" else -1.0
  # Identical legs rotated by 180 deg about z would swap left/right geometry;
  # we keep the joint axes in the body frame and flip only the knee direction
  # and the hip-roll actuator placement as described in the paper.
  knee_fwd = 1.0 if side == "left" else -1.0
  hy = B.body(root, f"{side}_hip_yaw_link", (0.0, s * 0.075, -0.02))
  B.inertial(hy, 0.55, (0, 0, -0.02), size=(0.07, 0.07, 0.06))
  B.hinge(hy, f"{side}_hip_yaw", (0, 0, 1), (-0.6, 0.6))
  hr = B.body(hy, f"{side}_hip_roll_link", (0.0, 0.0, -0.035))
  # Hip-roll actuator body sits behind (left) or in front of (right) the axis.
  B.inertial(hr, 0.55, (-0.03 * knee_fwd, 0, 0), size=(0.07, 0.06, 0.06))
  B.hinge(hr, f"{side}_hip_roll", (1, 0, 0), (-0.45, 0.45))
  B.capsule(hr, f"{side}_hip_roll_geom", (-0.05 * knee_fwd, 0, 0), (0.0, 0, 0), 0.03, dark)
  th = B.body(hr, f"{side}_thigh", (0.0, 0.0, -0.01))
  B.inertial(th, 0.45, (0.5 * KNEE_X * knee_fwd, 0, -THIGH / 2), size=(0.05, 0.05, THIGH))
  B.hinge(th, f"{side}_hip_pitch", (0, 1, 0), (-1.3, 1.3))
  B.capsule(th, f"{side}_thigh_geom", (0, 0, 0), (KNEE_X * knee_fwd, 0, -THIGH), 0.024, dark)
  sh = B.body(th, f"{side}_shank", (KNEE_X * knee_fwd, 0.0, -THIGH))
  B.inertial(sh, 0.3, (-0.5 * KNEE_X * knee_fwd, 0, -SHANK / 2), size=(0.04, 0.04, SHANK))
  # Left: forward knee (bends like a human knee); right: rear-facing knee.
  rng = (-0.2, 1.6) if knee_fwd > 0 else (-1.6, 0.2)
  B.hinge(sh, f"{side}_knee", (0, 1, 0), rng)
  B.capsule(sh, f"{side}_shank_geom", (0, 0, 0), (-KNEE_X * knee_fwd, 0, -SHANK), 0.02, dark,
            collide=1, mask=(COL_FOOT, COL_FOOT))
  ap = B.body(sh, f"{side}_ankle_pitch_link", (-KNEE_X * knee_fwd, 0.0, -SHANK))
  B.inertial(ap, 0.1, (0, 0, 0), size=(0.03, 0.03, 0.03))
  B.hinge(ap, f"{side}_ankle_pitch", (0, 1, 0), (-0.9, 0.9))
  ft = B.body(ap, f"{side}_foot", (0.0, 0.0, 0.0))
  B.inertial(ft, 0.25, FOOT_CENTRE, size=(0.14, 0.09, 0.05))
  B.hinge(ft, f"{side}_ankle_roll", (1, 0, 0), (-0.45, 0.45))
  g = B.box(ft, f"{side}_foot_geom", FOOT_CENTRE, FOOT_HALF, None, mask=(COL_WORLD | COL_FOOT, COL_WORLD | COL_FOOT))
  g.rgba = [1, 1, 1, 0.0]
  g.condim = 3
  g.priority = 1
  g.friction = [0.9, 0.005, 0.0001]
  g.group = 3
  # Snowball foot (visual): a flattened white ellipsoid over the sole.
  cx, cy, cz = FOOT_CENTRE
  hx, hy_, hz = FOOT_HALF
  B.ellipsoid(ft, f"{side}_foot_vis", (cx, cy, cz + 0.02), (hx + 0.005, hy_ + 0.005, 0.035), white)
  zb = cz - hz
  ft.add_site(name=f"{side}_sole", pos=[cx, cy, zb], size=[0.01, 0, 0])
  ft.add_site(name=f"{side}_toe", pos=[cx + hx, cy, zb], size=[0.006, 0, 0])
  ft.add_site(name=f"{side}_heel", pos=[cx - hx, cy, zb], size=[0.006, 0, 0])


def build():
  spec = B.new_spec("olaf")
  white = B.add_material(spec, "snow", [0.96, 0.97, 1.0, 1])
  dark = B.add_material(spec, "mech", [0.25, 0.25, 0.28, 1])
  coal = B.add_material(spec, "coal", [0.08, 0.08, 0.08, 1])
  carrot = B.add_material(spec, "carrot", [0.95, 0.5, 0.1, 1])
  brown = B.add_material(spec, "twig", [0.45, 0.28, 0.12, 1])

  root = B.body(spec.worldbody, "pelvis", (0, 0, ROOT_Z))
  root.add_freejoint(name="root")
  # Lower body: hip actuators, battery, foam skirt (skirt is soft: visual only).
  B.inertial(root, 4.0, (0.0, 0, 0.05), diag=[0.06, 0.06, 0.05])
  B.ellipsoid(root, "skirt_vis", (0, 0, 0.0), (0.16, 0.16, 0.13), white)
  root.add_site(name="torso", pos=[0, 0, 0], size=[0.01, 0, 0])
  root.add_site(name="imu", pos=[0, 0, 0.05], size=[0.01, 0, 0])
  # Middle snowball: fixed to the pelvis (Olaf's torso has no waist joint).
  mid = B.body(root, "torso_upper", (0.0, 0, 0.17))
  B.inertial(mid, 3.0, (0, 0, 0), diag=[0.02, 0.02, 0.02])
  B.sphere(mid, "torso_geom", (0, 0, 0), 0.1, white, mask=(COL_BODY, COL_WORLD))
  for i, z in enumerate((0.07, 0.0, -0.07)):
    B.sphere(mid, f"button{i}", (0.112, 0, z), 0.012, coal)
  # Arms (twigs) as fixed masses (show functions, low inertia).
  for s in (1, -1):
    arm = B.body(mid, f"arm_{'l' if s > 0 else 'r'}", (0, s * 0.11, 0.03))
    B.inertial(arm, 0.25, (0, s * 0.08, 0.04), size=(0.03, 0.16, 0.03))
    B.capsule(arm, f"arm_geom_{s}", (0, 0, 0), (0.02, s * 0.2, 0.09), 0.01, brown)
  # Slim 3-DoF neck: yaw, pitch, roll.
  n1 = B.body(mid, "neck_yaw_link", (0.0, 0, 0.095))
  B.inertial(n1, 0.2, (0, 0, 0.015), size=(0.04, 0.04, 0.03))
  B.hinge(n1, "neck_yaw", (0, 0, 1), (-1.0, 1.0))
  B.capsule(n1, "neck_geom", (0, 0, 0), (0, 0, 0.02), 0.03, white)
  n2 = B.body(n1, "neck_pitch_link", (0.0, 0, 0.02))
  B.inertial(n2, 0.2, (0, 0, 0.01), size=(0.04, 0.04, 0.03))
  B.hinge(n2, "neck_pitch", (0, 1, 0), (-0.5, 0.7))
  hd = B.body(n2, "head", (0.0, 0, 0.015))
  B.inertial(hd, 2.6, (0.035, 0, 0.10), diag=[0.02, 0.018, 0.011])
  B.hinge(hd, "neck_roll", (1, 0, 0), (-0.4, 0.4))
  B.ellipsoid(hd, "head_geom", (0.03, 0, 0.10), (0.115, 0.11, 0.12), white, mask=(COL_BODY, COL_WORLD))
  B.ellipsoid(hd, "nose", (0.15, 0, 0.10), (0.06, 0.015, 0.015), carrot)
  for s in (1, -1):
    B.sphere(hd, f"eye{s}", (0.12, s * 0.045, 0.15), 0.02, coal)
  hd.add_site(name="head", pos=[0.03, 0, 0.10], size=[0.01, 0, 0])
  for side in ("left", "right"):
    leg(root, side, (white, dark))
  effort = {}
  for side in ("left", "right"):
    effort.update({f"{side}_hip_yaw": 23.7, f"{side}_hip_roll": 34.0, f"{side}_hip_pitch": 34.0, f"{side}_knee": 34.0,
                   f"{side}_ankle_pitch": 23.7, f"{side}_ankle_roll": 23.7})
  effort.update({"neck_yaw": 4.8, "neck_pitch": 6.0, "neck_roll": 4.8})
  B.add_motors(spec, effort)
  return spec


def main():
  spec = build()
  B.save(spec, OUT)
  m = mujoco.MjModel.from_xml_path(str(OUT))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  sole = d.site("left_sole").xpos[2]
  top = d.geom("head_geom").xpos[2] + 0.12
  print(f"olaf: mass {m.body_subtreemass[1]:.2f} kg, actuators {m.nu}, height {top - sole:.3f} m, root over sole {d.qpos[2]-sole:.3f}")
  print(f"  CoM x {d.subtree_com[1][0]:.3f} y {d.subtree_com[1][1]:.3f}, feet x {d.site('left_sole').xpos[0]:.3f}/{d.site('right_sole').xpos[0]:.3f}")


if __name__ == "__main__":
  main()
