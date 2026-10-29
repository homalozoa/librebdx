"""Small-scale dinosaur-inspired biped of Hopkins et al. (SIGGRAPH 2024), Sec. 8.1.

Paper specification: 20 actuators -- 6-DoF legs with reverse knees, a 4-DoF
bird-like neck, a 4-DoF segmented tail with alternating yaw and pitch joints;
615 mm tall, 16.9 kg (Table 4). No CAD is public, so the geometry below is our
own design matching those numbers. At q = 0 the robot is in its nominal stance.

Run: python -m disneyrepro.robots.dino.build
"""

from pathlib import Path

import mujoco
import numpy as np

from disneyrepro.robots import builder as B

OUT = Path(__file__).parent / "dino.xml"

ROOT_Z = 0.38
FOOT = dict(half=(0.065, 0.034, 0.008), centre=(0.0, 0.0, -0.027))  # in ankle frame

EFFORT = {
  "hip_yaw": 20.0, "hip_roll": 35.0, "hip_pitch": 35.0, "knee": 40.0, "ankle_pitch": 25.0, "ankle_roll": 15.0,
  "neck_pitch1": 8.0, "neck_yaw": 6.0, "neck_pitch2": 6.0, "head_pitch": 4.0,
  "tail_yaw1": 12.0, "tail_pitch2": 10.0, "tail_yaw3": 6.0, "tail_pitch4": 4.0,
}


def leg(spec, root, side, mat_a, mat_b):
  s = 1.0 if side == "left" else -1.0
  hy = B.body(root, f"{side}_hip_yaw_link", (0.0, s * 0.085, -0.04))
  B.inertial(hy, 0.45, (0, 0, -0.01), size=(0.06, 0.06, 0.06))
  B.hinge(hy, f"{side}_hip_yaw", (0, 0, 1), (-0.6, 0.6))
  B.capsule(hy, f"{side}_hip_yaw_geom", (0, 0, 0.0), (0, 0, -0.02), 0.03, mat_b)

  hr = B.body(hy, f"{side}_hip_roll_link", (0.0, 0.0, -0.02))
  B.inertial(hr, 0.45, (0, 0, 0), size=(0.06, 0.06, 0.06))
  B.hinge(hr, f"{side}_hip_roll", (1, 0, 0), (-0.5, 0.5))
  B.capsule(hr, f"{side}_hip_roll_geom", (-0.03, 0, 0), (0.03, 0, 0), 0.028, mat_b)

  th = B.body(hr, f"{side}_thigh", (0.0, 0.0, -0.015))
  B.inertial(th, 0.5, (-0.03, 0, -0.065), size=(0.05, 0.05, 0.14))
  B.hinge(th, f"{side}_hip_pitch", (0, 1, 0), (-1.2, 1.2))
  B.capsule(th, f"{side}_thigh_geom", (0, 0, 0), (-0.075, 0, -0.125), 0.026, mat_a, collide=1, mask=(2, 2))

  # Ankle placed under the CoM (the tail moves the CoM behind the hips).
  sh = B.body(th, f"{side}_shank", (-0.075, 0.0, -0.125))
  B.inertial(sh, 0.3, (0.02, 0, -0.072), size=(0.04, 0.04, 0.14))
  # Reverse knee: the knee points backwards, the shank runs forward-down.
  B.hinge(sh, f"{side}_knee", (0, 1, 0), (-1.0, 1.4))
  B.capsule(sh, f"{side}_shank_geom", (0, 0, 0), (0.04, 0, -0.145), 0.02, mat_a, collide=1, mask=(2, 2))

  ap = B.body(sh, f"{side}_ankle_pitch_link", (0.04, 0.0, -0.145))
  B.inertial(ap, 0.1, (0, 0, 0), size=(0.03, 0.03, 0.03))
  B.hinge(ap, f"{side}_ankle_pitch", (0, 1, 0), (-1.0, 1.0))
  ft = B.body(ap, f"{side}_foot", (0.0, 0.0, 0.0))
  B.inertial(ft, 0.2, FOOT["centre"], size=(0.13, 0.07, 0.02))
  B.hinge(ft, f"{side}_ankle_roll", (1, 0, 0), (-0.5, 0.5))
  g = B.box(ft, f"{side}_foot_geom", FOOT["centre"], FOOT["half"], mat_b, mask=(1, 1))
  g.condim = 3
  g.priority = 1
  g.friction = [0.9, 0.005, 0.0001]
  cx, cy, cz = FOOT["centre"]
  hx, hy_, hz = FOOT["half"]
  zb = cz - hz
  ft.add_site(name=f"{side}_sole", pos=[cx, cy, zb], size=[0.008, 0, 0])
  ft.add_site(name=f"{side}_toe", pos=[cx + hx, cy, zb], size=[0.005, 0, 0])
  ft.add_site(name=f"{side}_heel", pos=[cx - hx, cy, zb], size=[0.005, 0, 0])
  for i, (dx, dy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
    ft.add_site(name=f"{side}_corner{i}", pos=[cx + dx * hx, cy + dy * hy_, zb], size=[0.004, 0, 0])


def build():
  spec = B.new_spec("dino")
  green = B.add_material(spec, "skin", [0.35, 0.62, 0.32, 1])
  dark = B.add_material(spec, "dark", [0.18, 0.22, 0.2, 1])
  belly = B.add_material(spec, "belly", [0.85, 0.8, 0.6, 1])

  root = B.body(spec.worldbody, "pelvis", (0, 0, ROOT_Z))
  root.add_freejoint(name="root")
  B.inertial(root, 8.5, (0.01, 0, 0.01), diag=[0.05, 0.09, 0.09])
  B.ellipsoid(root, "torso_geom", (0.0, 0, 0.01), (0.17, 0.085, 0.085), green, mask=(4, 1))
  B.ellipsoid(root, "belly_geom", (0.02, 0, -0.02), (0.14, 0.07, 0.07), belly)
  root.add_site(name="torso", pos=[0, 0, 0], size=[0.01, 0, 0])
  root.add_site(name="imu", pos=[0, 0, 0.02], size=[0.01, 0, 0])

  for side in ("left", "right"):
    leg(spec, root, side, green, dark)

  # Neck: base pitch + yaw at the shoulder, mid pitch, head pitch (bird-like).
  n1 = B.body(root, "neck1", (0.15, 0, 0.05))
  B.inertial(n1, 0.45, (0.02, 0, 0.045), size=(0.05, 0.05, 0.1))
  B.hinge(n1, "neck_pitch1", (0, 1, 0), (-0.6, 0.9))
  n1y = B.body(n1, "neck1_yaw", (0, 0, 0))
  B.inertial(n1y, 0.1, (0, 0, 0.02), size=(0.03, 0.03, 0.03))
  B.hinge(n1y, "neck_yaw", (0, 0, 1), (-0.9, 0.9))
  B.capsule(n1y, "neck1_geom", (0, 0, 0), (0.04, 0, 0.09), 0.03, green)
  n2 = B.body(n1y, "neck2", (0.04, 0, 0.09))
  B.inertial(n2, 0.3, (0.015, 0, 0.03), size=(0.04, 0.04, 0.07))
  B.hinge(n2, "neck_pitch2", (0, 1, 0), (-0.9, 0.9))
  B.capsule(n2, "neck2_geom", (0, 0, 0), (0.03, 0, 0.06), 0.025, green)
  hd = B.body(n2, "head", (0.03, 0, 0.06))
  B.inertial(hd, 0.75, (0.05, 0, 0.0), size=(0.12, 0.07, 0.07))
  B.hinge(hd, "head_pitch", (0, 1, 0), (-0.9, 0.9))
  B.ellipsoid(hd, "head_geom", (0.05, 0, 0.0), (0.075, 0.042, 0.038), green, mask=(4, 1))
  B.sphere(hd, "eye_l", (0.08, 0.03, 0.015), 0.009, dark)
  B.sphere(hd, "eye_r", (0.08, -0.03, 0.015), 0.009, dark)
  hd.add_site(name="head", pos=[0.05, 0, 0], size=[0.01, 0, 0])

  # Tail: 4 segments, alternating yaw and pitch.
  parent, x0 = root, -0.16
  seg = [("tail_yaw1", (0, 0, 1), 1.0, 0.10, 0.04), ("tail_pitch2", (0, 1, 0), 0.8, 0.09, 0.032),
         ("tail_yaw3", (0, 0, 1), 0.6, 0.09, 0.025), ("tail_pitch4", (0, 1, 0), 0.4, 0.09, 0.018)]
  pos = (x0, 0, 0.01)
  for i, (jn, ax, mass, L, r) in enumerate(seg):
    tb = B.body(parent, f"tail{i+1}", pos)
    B.inertial(tb, mass, (-L / 2, 0, 0.0), size=(L, 2 * r, 2 * r))
    B.hinge(tb, jn, ax, (-0.7, 0.7))
    B.capsule(tb, f"tail{i+1}_geom", (0, 0, 0), (-L, 0, 0.012), r, green)
    parent, pos = tb, (-L, 0, 0.012)
  parent.add_site(name="tail_tip", pos=list(pos), size=[0.01, 0, 0])

  B.add_motors(spec, {j: e for j, e in _all_efforts().items()})
  return spec


def _all_efforts():
  out = {}
  for side in ("left", "right"):
    for k in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
      out[f"{side}_{k}"] = EFFORT[k]
  for k in ("neck_pitch1", "neck_yaw", "neck_pitch2", "head_pitch", "tail_yaw1", "tail_pitch2", "tail_yaw3", "tail_pitch4"):
    out[k] = EFFORT[k]
  return out


def main():
  spec = build()
  B.save(spec, OUT)
  m = mujoco.MjModel.from_xml_path(str(OUT))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  sole = d.site("left_sole").xpos[2]
  head_top = d.geom("head_geom").xpos[2] + 0.038
  print(f"dino: mass {m.body_subtreemass[1]:.2f} kg, actuators {m.nu}, height {head_top - sole:.3f} m, root over sole {d.qpos[2]-sole:.3f}")
  print(f"  CoM x {d.subtree_com[1][0]:.3f}, foot centre x {d.site('left_sole').xpos[0]:.3f}")
  for g in ("legs", "neck", "tail"):
    pass
  sub = lambda n: m.body_subtreemass[m.body(n).id]
  print(f"  legs {sub('left_hip_yaw_link'):.2f}x2, neck+head {sub('neck1'):.2f}, tail {sub('tail1'):.2f}")


if __name__ == "__main__":
  main()
