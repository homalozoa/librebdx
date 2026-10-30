"""Build the BDX-R model used as the stand-in for Disney's bipedal character.

Starting point is the open-source BDX-R MJCF (github.com/BDX-R/BDX-R-MjLab).
Changes so that it matches the robot described in Grandia et al. (RSS 2024):
  * mass of the torso / each leg / neck+head scaled to 5.8 / 3.6 / 2.4 kg
    (15.4 kg total, Sec. IV of the paper),
  * flat 7-capsule soles replaced by one rounded sole per foot so the robot can
    roll passively about the ankle (the paper's robot has no ankle roll and
    rounded soles),
  * collision capsules for the legs and a box for the head (used for
    self-collision and for the head/torso-ground termination),
  * sites at the soles, head and torso used by the reference generators,
  * the neck link lengthened by 6.6 cm so the robot stands 0.66 m tall without
    antennas (Sec. IV; BDX-R is 0.60 m),
  * hip yaw widened to +-0.6 rad and hip roll to -0.45..0.6 rad (mirrored): BDX-R's
    +-0.26 rad hip yaw caps turning near 1 rad/s, while the paper's robot turns at
    1.8 rad/s with a "large workspace of the two legs" (Sec. IV, VII-A).

Run: python -m librebdx.robots.bdxr.build
"""

from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "bdxr_orig.xml"
OUT = HERE / "bdxr.xml"

PAPER_MASS = {"torso": 5.8, "leg": 3.6, "head": 2.4}
NECK_EXTRA = 0.066  # m along the neck link; ~0.056 m of height at the nominal neck pitch
HIP_YAW = (-0.6, 0.6)
HIP_ROLL_LEFT = (-0.45, 0.6)  # the right leg mirrors it
SOLE_RADIUS = 0.02
# Collision bit masks: feet<->world, legs<->legs, head<->(world, legs, torso),
# torso<->(world, head). The torso capsule overlaps the hips, so it must not
# collide with the legs.
COL_WORLD, COL_LEG, COL_HEAD, COL_TORSO = 1, 2, 4, 8

LEG_JOINTS = ["Hip_Yaw", "Hip_Roll", "Hip_Pitch", "Knee", "Ankle"]
NECK_JOINTS = ["Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll"]


def _subtree_bodies(spec_body):
  out = [spec_body]
  for child in spec_body.bodies:
    out += _subtree_bodies(child)
  return out


def _scale_body(b, s):
  b.mass = b.mass * s
  b.fullinertia = np.asarray(b.fullinertia) * s
  b.inertia = np.asarray(b.inertia) * s


def _mesh_points_in_body(model, data, body_names):
  """Vertices of all visual meshes of the given bodies, in world frame."""
  pts = []
  for g in range(model.ngeom):
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    if model.body(model.geom_bodyid[g]).name not in body_names:
      continue
    mid = model.geom_dataid[g]
    v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[v0 : v0 + nv]
    R = data.geom_xmat[g].reshape(3, 3)
    pts.append(v @ R.T + data.geom_xpos[g])
  return np.concatenate(pts, 0)


def _to_body(data, model, body, p_world):
  bid = model.body(body).id
  R = data.xmat[bid].reshape(3, 3)
  return (p_world - data.xpos[bid]) @ R


def build() -> Path:
  spec = mujoco.MjSpec.from_file(str(SRC))
  root = spec.body("base_link")

  # --- mass redistribution -------------------------------------------------
  model = spec.compile()
  groups = {
    "leg_l": _subtree_bodies(spec.body("Left_Hip_Yaw_Motor")),
    "leg_r": _subtree_bodies(spec.body("Right_Hip_Yaw_Motor")),
    "head": _subtree_bodies(spec.body("Neck_Motor")),
  }
  in_group = {b.name for bs in groups.values() for b in bs}
  groups["torso"] = [b for b in _subtree_bodies(root) if b.name not in in_group]
  for key, bodies in groups.items():
    m_now = sum(model.body(b.name).mass[0] for b in bodies)
    target = PAPER_MASS["leg" if key.startswith("leg") else key]
    for b in bodies:
      _scale_body(b, target / m_now)

  # --- paper geometry: taller neck, wider hips ------------------------------
  head = spec.body("Head_Pitch_Motor")
  z0 = float(head.pos[2])
  k = (z0 + NECK_EXTRA) / z0
  head.pos = [head.pos[0], head.pos[1], z0 + NECK_EXTRA]
  neck = spec.body("Neck")
  for g in neck.geoms:
    if g.meshname == "Neck":
      g.pos = [g.pos[0], g.pos[1], g.pos[2] * k]
  mesh = spec.mesh("Neck")
  mesh.scale = [mesh.scale[0], mesh.scale[1], mesh.scale[2] * k]
  neck.ipos = [neck.ipos[0], neck.ipos[1], neck.ipos[2] * k]
  for side, mirror in (("Left", 1.0), ("Right", -1.0)):
    spec.joint(f"{side}_Hip_Yaw").range = list(HIP_YAW)
    lo, hi = HIP_ROLL_LEFT
    spec.joint(f"{side}_Hip_Roll").range = [lo, hi] if mirror > 0 else [-hi, -lo]

  # --- geometry fitting from the visual meshes -----------------------------
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[2] = 0.5
  mujoco.mj_forward(model, data)

  for side, prefix in (("left", "Left"), ("right", "Right")):
    pad = spec.body(f"{prefix}_Foot_Pad")
    olds = [g for g in pad.geoms if g.name.startswith(f"{side}_foot_collision")]
    ends = np.array([g.fromto for g in olds])
    mid = ends[len(ends) // 2]
    bottom = mid[2] - olds[0].size[0]
    # Express the sole line in the Foot frame (parent of the pad) so that the
    # sole site and the IK foot frame live in the same body.
    for g in olds:
      spec.delete(g)
    p0 = np.array([mid[0], mid[1], bottom + SOLE_RADIUS])
    p1 = np.array([mid[3], mid[4], bottom + SOLE_RADIUS])
    pad.add_geom(
      name=f"{side}_foot_sole",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=[SOLE_RADIUS, 0, 0],
      fromto=np.concatenate([p0, p1]),
      contype=1,
      conaffinity=1,
      condim=3,
      priority=1,
      friction=[0.8, 0.005, 0.0001],
      group=3,
      rgba=[1, 0.65, 0, 0.4],
    )
    centre = 0.5 * (p0 + p1)
    pad.add_site(name=f"{side}_sole", pos=[centre[0], centre[1], bottom], size=[0.01, 0, 0])
    pad.add_site(name=f"{side}_toe", pos=[max(p0[0], p1[0]), centre[1], bottom], size=[0.006, 0, 0])
    pad.add_site(name=f"{side}_heel", pos=[min(p0[0], p1[0]), centre[1], bottom], size=[0.006, 0, 0])

    # Leg capsules fitted to the upper / lower leg meshes (principal axis).
    for link in ("Upper_Leg", "Lower_Leg"):
      bname = f"{prefix}_{link}"
      names = [bname] + [f"{prefix}_{link}_Cover"]
      pts_w = _mesh_points_in_body(model, data, names)
      pts = _to_body(data, model, bname, pts_w)
      c = pts.mean(0)
      _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
      ax = vt[0]
      proj = (pts - c) @ ax
      lo, hi = np.percentile(proj, [3, 97])
      rad = np.percentile(np.linalg.norm((pts - c) - np.outer(proj, ax), axis=1), 70)
      rad = float(np.clip(rad, 0.025, 0.045))
      a = c + ax * (lo + rad)
      b = c + ax * (hi - rad)
      spec.body(bname).add_geom(
        name=f"{side}_{link.lower()}_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[rad, 0, 0],
        fromto=np.concatenate([a, b]),
        contype=COL_LEG,
        conaffinity=COL_LEG,
        condim=1,
        group=3,
        rgba=[1, 0.65, 0, 0.3],
      )

  # Head: an oriented box around all visual meshes of the head plate subtree.
  head_bodies = [b.name for b in _subtree_bodies(spec.body("Head_Base_Plate"))]
  pts = _to_body(data, model, "Head_Base_Plate", _mesh_points_in_body(model, data, head_bodies))
  lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
  hb = spec.body("Head_Base_Plate")
  hb.add_geom(
    name="head_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=0.5 * (hi - lo),
    pos=0.5 * (hi + lo),
    contype=COL_HEAD,
    conaffinity=COL_WORLD | COL_LEG | COL_TORSO,
    condim=1,
    group=3,
    rgba=[1, 0.65, 0, 0.3],
  )
  hb.add_site(name="head", pos=0.5 * (hi + lo), size=[0.01, 0, 0])
  # Torso collision also hits the head (self-collision) and the ground.
  base_col = [g for g in root.geoms if g.name == "base_collision"][0]
  base_col.contype = COL_TORSO
  base_col.conaffinity = COL_WORLD | COL_HEAD
  base_col.condim = 1
  root.add_site(name="torso", pos=[0, 0, 0], size=[0.01, 0, 0])

  # Write back the balanced, diagonalised inertias of the compiled model so the
  # exported XML loads without relying on compiler flags.
  model = spec.compile()
  for b in _subtree_bodies(root):
    bid = model.body(b.name).id
    b.fullinertia = [np.nan] * 6
    b.inertia = model.body_inertia[bid]
    b.iquat = model.body_iquat[bid]
    b.ipos = model.body_ipos[bid]
    b.explicitinertial = True
  spec.compiler.degree = False
  # Printed inertias are rounded, which can break A + B >= C by ~1e-12.
  xml = spec.to_xml().replace("<compiler ", '<compiler balanceinertia="true" ', 1)
  OUT.write_text(xml)
  return OUT


def summary(path: Path = OUT):
  m = mujoco.MjModel.from_xml_path(str(path))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  sub = lambda n: m.body_subtreemass[m.body(n).id]
  legs = sub("Left_Hip_Yaw_Motor"), sub("Right_Hip_Yaw_Motor")
  head = sub("Neck_Motor")
  print(f"total {m.body_subtreemass[1]:.2f} kg, legs {legs[0]:.2f}/{legs[1]:.2f}, "
        f"neck+head {head:.2f}, torso {m.body_subtreemass[1]-sum(legs)-head:.2f}")
  sole = d.site("left_sole").xpos[2]
  print(f"base height over sole at q=0: {d.qpos[2]-sole:.3f} m;"
        f" head site over sole {d.site('head').xpos[2]-sole:.3f} m")
  print("joints:", [m.joint(i).name for i in range(1, m.njnt)])


if __name__ == "__main__":
  build()
  summary()
