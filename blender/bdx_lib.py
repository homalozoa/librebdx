"""Blender scene kit for the BDX showcase (run inside Blender 5.x).

Robot: visual meshes exported by scripts/export_blender.py, painted with the
"Dune Scout" livery (ivory shell, ember accents, slate trim, gunmetal drives,
teal eyes) and optionally Meshy-generated shell textures. Animation comes from
recorded MuJoCo trajectories, resampled to the render frame rate.
Worlds: twin-sun desert at golden hour, and a night variant with a ringed moon.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "bdx" / "blender"
TEX = ROOT / "results" / "bdx" / "meshy"
BAKED = ROOT / "results" / "bdx" / "baked"


def hexrgb(h, a=1.0):
  h = h.lstrip("#")
  c = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
  lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
  return (*lin, a)


# ------------------------------------------------------------------ livery
PALETTE = {
  "ivory": "#E7DECB",
  "ember": "#C9532A",
  "slate": "#3B4754",
  "gunmetal": "#24272C",
  "rubber": "#141414",
  "teal": "#53E3E0",
}


SIDES = ("Left_", "Right_")
ROLES = {
  "eye": {"Left_Eye", "Right_Eye"},
  "rubber": {"Left_Foot_Pad", "Right_Foot_Pad"},
  "ember": {s + p for s in SIDES for p in ("Upper_Leg_Cover", "Front_Cover", "Ear")},
  "slate": {"Rear_Head_Cover", "Rear_Head_Plate", "Rear_Antenna_Box"} | {s + p for s in SIDES for p in ("Foot_Cover", "Ankle_Cover", "Upper_Leg_Detail")},
  "ivory": {"Face", "Front_Cover", "Rear_Cover", "Bottom_Cover", "Head_Base_Plate"} | {s + p for s in SIDES for p in ("Top_Cover", "Lower_Leg_Cover")},
}


def part_role(mesh):
  for role, names in ROLES.items():
    if mesh in names:
      return role
  return "gunmetal"


def _principled(name, color, rough, metal=0.0, coat=0.0):
  m = bpy.data.materials.new(name)
  m.use_nodes = True
  nt = m.node_tree
  b = nt.nodes["Principled BSDF"]
  b.inputs["Base Color"].default_value = hexrgb(color)
  b.inputs["Roughness"].default_value = rough
  b.inputs["Metallic"].default_value = metal
  if "Coat Weight" in b.inputs:
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Coat Roughness"].default_value = 0.25
  return m, nt, b


def _grime(nt, b, color, strength=0.25, scale=18.0):
  """Dusty, sun-faded paint: low-frequency noise darkens/desaturates the base."""
  tc = nt.nodes.new("ShaderNodeTexCoord")
  n = nt.nodes.new("ShaderNodeTexNoise")
  n.inputs["Scale"].default_value = scale
  n.inputs["Detail"].default_value = 8.0
  n.inputs["Roughness"].default_value = 0.6
  nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
  ramp = nt.nodes.new("ShaderNodeValToRGB")
  ramp.color_ramp.elements[0].position = 0.45
  ramp.color_ramp.elements[1].position = 0.75
  nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
  mix = nt.nodes.new("ShaderNodeMix")
  mix.data_type = "RGBA"
  mix.inputs["A"].default_value = hexrgb(color)
  dust = hexrgb("#9C8466")
  mix.inputs["B"].default_value = dust
  mul = nt.nodes.new("ShaderNodeMath")
  mul.operation = "MULTIPLY"
  mul.inputs[1].default_value = strength
  nt.links.new(ramp.outputs["Color"], mul.inputs[0])
  nt.links.new(mul.outputs[0], mix.inputs["Factor"])
  nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
  # Micro surface: fine bump so highlights break up like real paint.
  bump = nt.nodes.new("ShaderNodeBump")
  bump.inputs["Strength"].default_value = 0.04
  n2 = nt.nodes.new("ShaderNodeTexNoise")
  n2.inputs["Scale"].default_value = 900.0
  nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
  nt.links.new(n2.outputs["Fac"], bump.inputs["Height"])
  nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
  return mix


def _textured(name, png, rough):
  m = bpy.data.materials.new(name)
  m.use_nodes = True
  nt = m.node_tree
  b = nt.nodes["Principled BSDF"]
  img = nt.nodes.new("ShaderNodeTexImage")
  img.image = bpy.data.images.load(str(png), check_existing=True)
  nt.links.new(img.outputs["Color"], b.inputs["Base Color"])
  b.inputs["Roughness"].default_value = rough
  if "Coat Weight" in b.inputs:
    b.inputs["Coat Weight"].default_value = 0.15
  return m


def _baked(name, role):
  """Principled material from baked Meshy v2 PBR maps on the mesh's own 'baked' UVs."""
  m = bpy.data.materials.new(f"baked_{name}")
  m.use_nodes = True
  nt = m.node_tree
  b = nt.nodes["Principled BSDF"]
  uv = nt.nodes.new("ShaderNodeUVMap")
  uv.uv_map = "baked"

  def tex(kind, cs):
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(str(BAKED / f"{name}_{kind}.png"), check_existing=True)
    t.image.colorspace_settings.name = cs
    nt.links.new(uv.outputs["UV"], t.inputs["Vector"])
    return t

  col = tex("base_color", "sRGB").outputs["Color"]
  if role == "gunmetal":  # Meshy machined parts come back bright silver: darken to gunmetal, keep detail
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    mul.inputs["B"].default_value = hexrgb("#4A4E55")
    nt.links.new(col, mul.inputs["A"])
    col = mul.outputs["Result"]
  nt.links.new(col, b.inputs["Base Color"])
  nt.links.new(tex("roughness", "Non-Color").outputs["Color"], b.inputs["Roughness"])
  nt.links.new(tex("metallic", "Non-Color").outputs["Color"], b.inputs["Metallic"])
  nm = nt.nodes.new("ShaderNodeNormalMap")
  nm.uv_map = "baked"
  nm.inputs["Strength"].default_value = 0.8
  nt.links.new(tex("normal", "Non-Color").outputs["Color"], nm.inputs["Color"])
  nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
  if role != "gunmetal" and "Coat Weight" in b.inputs:  # enamel shell: a thin clear coat
    b.inputs["Coat Weight"].default_value = 0.25
    b.inputs["Coat Roughness"].default_value = 0.2
  return m


def make_materials():
  mats = {}
  for role, rough, coat, grime in (("ivory", 0.42, 0.25, 0.22), ("ember", 0.38, 0.3, 0.18), ("slate", 0.5, 0.1, 0.15)):
    m, nt, b = _principled(f"paint_{role}", PALETTE[role], rough, 0.0, coat)
    _grime(nt, b, PALETTE[role], grime)
    mats[role] = m
  m, nt, b = _principled("gunmetal", PALETTE["gunmetal"], 0.34, 1.0)
  if "Anisotropic" in b.inputs:
    b.inputs["Anisotropic"].default_value = 0.4
  mats["gunmetal"] = m
  mats["rubber"], _, _ = _principled("rubber", PALETTE["rubber"], 0.85)
  m, nt, b = _principled("eye", "#05080A", 0.05, 0.0, 1.0)
  b.inputs["Emission Color"].default_value = hexrgb(PALETTE["teal"])
  b.inputs["Emission Strength"].default_value = 6.0
  mats["eye"] = m
  return mats


# ------------------------------------------------------------------ robot
def build_robot(mats, coll_name="BDX"):
  """One object per visual geom; returns (objects, geom_mesh_names)."""
  z = np.load(DATA / "meshes.npz")
  coll = bpy.data.collections.new(coll_name)
  bpy.context.scene.collection.children.link(coll)
  objs = []
  meshy = {p.stem: p for p in TEX.glob("*.png")} if TEX.exists() else {}
  cache = {}
  for gname, mname in zip(z["geom_name"], z["geom_mesh"]):
    mname = str(mname)
    if mname not in cache:
      v, f = z[f"v_{mname}"], z[f"f_{mname}"]
      me = bpy.data.meshes.new(mname)
      me.from_pydata(v.tolist(), [], f.tolist())
      me.validate()
      me.shade_smooth()
      if hasattr(me, "set_sharp_from_angle"):
        me.set_sharp_from_angle(angle=math.radians(35))
      bk = BAKED / f"{mname}_uv.npy"
      if bk.exists():  # UVs of the bake (face-corner order matches from_pydata)
        uv = me.uv_layers.new(name="baked")
        uv.data.foreach_set("uv", np.load(bk).astype(np.float32).ravel())
      uvp = TEX / f"{mname}_uv.npy"
      if uvp.exists() and not bk.exists():  # per-corner UVs transferred from the Meshy result
        uv = me.uv_layers.new(name="meshy")
        uv.data.foreach_set("uv", np.load(uvp).astype(np.float32).ravel())
      cache[mname] = me
    me = cache[mname]
    ob = bpy.data.objects.new(str(gname) or mname, me)
    role = part_role(mname)
    if (BAKED / f"{mname}_uv.npy").exists():
      key = f"baked_{mname}"
      if key not in mats:
        mats[key] = _baked(mname, role)
      ob.data.materials.append(mats[key])
    elif mname in meshy and (TEX / f"{mname}_uv.npy").exists():
      key = f"meshy_{mname}"
      if key not in mats:
        mats[key] = _textured(key, meshy[mname], 0.45)
      ob.data.materials.append(mats[key])
    else:
      ob.data.materials.append(mats[role])
    ob.rotation_mode = "QUATERNION"
    coll.objects.link(ob)
    objs.append(ob)
  return objs, [str(m) for m in z["geom_mesh"]]


def _iris_material(name):
  """Procedural lens iris: thin outer ring, striated teal iris, dark pupil, inner ring.
  Resolution-independent (pure math on the disk's own coordinates)."""
  m = bpy.data.materials.new(name)
  m.use_nodes = True
  nt = m.node_tree
  nt.nodes.clear()
  tc = nt.nodes.new("ShaderNodeTexCoord")
  sep = nt.nodes.new("ShaderNodeSeparateXYZ")
  nt.links.new(tc.outputs["Object"], sep.inputs[0])
  vm = nt.nodes.new("ShaderNodeVectorMath")
  vm.operation = "LENGTH"
  comb = nt.nodes.new("ShaderNodeCombineXYZ")
  nt.links.new(sep.outputs["X"], comb.inputs["X"])
  nt.links.new(sep.outputs["Y"], comb.inputs["Y"])
  nt.links.new(comb.outputs[0], vm.inputs[0])
  r = vm.outputs["Value"]  # disk radius is 1 in object space

  def band(lo, hi, soft=0.012):
    a = nt.nodes.new("ShaderNodeMapRange")
    a.interpolation_type = "SMOOTHSTEP"
    a.inputs["From Min"].default_value, a.inputs["From Max"].default_value = lo - soft, lo + soft
    nt.links.new(r, a.inputs["Value"])
    b = nt.nodes.new("ShaderNodeMapRange")
    b.interpolation_type = "SMOOTHSTEP"
    b.inputs["From Min"].default_value, b.inputs["From Max"].default_value = hi + soft, hi - soft
    nt.links.new(r, b.inputs["Value"])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(a.outputs["Result"], mul.inputs[0])
    nt.links.new(b.outputs["Result"], mul.inputs[1])
    return mul.outputs[0]

  # Radial striations in the iris (angle-based noise), like machined aperture blades.
  ang = nt.nodes.new("ShaderNodeMath")
  ang.operation = "ARCTAN2"
  nt.links.new(sep.outputs["Y"], ang.inputs[0])
  nt.links.new(sep.outputs["X"], ang.inputs[1])
  wave = nt.nodes.new("ShaderNodeTexNoise")
  wave.inputs["Scale"].default_value = 9.0
  wave.inputs["Detail"].default_value = 3.0
  angv = nt.nodes.new("ShaderNodeCombineXYZ")
  nt.links.new(ang.outputs[0], angv.inputs["X"])
  nt.links.new(angv.outputs[0], wave.inputs["Vector"])
  stri = nt.nodes.new("ShaderNodeMapRange")
  stri.inputs["From Min"].default_value, stri.inputs["From Max"].default_value = 0.3, 0.7
  stri.inputs["To Min"].default_value, stri.inputs["To Max"].default_value = 0.55, 1.0
  nt.links.new(wave.outputs["Fac"], stri.inputs["Value"])

  outer = band(0.90, 0.965)
  iris = band(0.44, 0.80)
  inner = band(0.385, 0.405, 0.006)
  iris_s = nt.nodes.new("ShaderNodeMath")
  iris_s.operation = "MULTIPLY"
  nt.links.new(iris, iris_s.inputs[0])
  nt.links.new(stri.outputs["Result"], iris_s.inputs[1])
  # Iris colour shifts from deep teal (inside) to bright aqua (outside).
  grad = nt.nodes.new("ShaderNodeValToRGB")
  grad.color_ramp.elements[0].position, grad.color_ramp.elements[0].color = 0.44, hexrgb("#00B4B8")
  grad.color_ramp.elements[1].position, grad.color_ramp.elements[1].color = 0.80, hexrgb("#5FFFF4")
  nt.links.new(r, grad.inputs["Fac"])
  em_iris = nt.nodes.new("ShaderNodeEmission")
  nt.links.new(grad.outputs["Color"], em_iris.inputs["Color"])
  k = nt.nodes.new("ShaderNodeMath")
  k.operation = "MULTIPLY"
  k.inputs[1].default_value = 16.0
  nt.links.new(iris_s.outputs[0], k.inputs[0])
  nt.links.new(k.outputs[0], em_iris.inputs["Strength"])
  rings = nt.nodes.new("ShaderNodeMath")
  rings.operation = "ADD"
  nt.links.new(outer, rings.inputs[0])
  nt.links.new(inner, rings.inputs[1])
  em_ring = nt.nodes.new("ShaderNodeEmission")
  em_ring.inputs["Color"].default_value = hexrgb("#B8FFF9")
  k2 = nt.nodes.new("ShaderNodeMath")
  k2.operation = "MULTIPLY"
  k2.inputs[1].default_value = 26.0
  nt.links.new(rings.outputs[0], k2.inputs[0])
  nt.links.new(k2.outputs[0], em_ring.inputs["Strength"])
  base = nt.nodes.new("ShaderNodeBsdfPrincipled")  # dark pupil / gaps
  base.inputs["Base Color"].default_value = hexrgb("#030506")
  base.inputs["Roughness"].default_value = 0.35
  add1 = nt.nodes.new("ShaderNodeAddShader")
  add2 = nt.nodes.new("ShaderNodeAddShader")
  nt.links.new(em_iris.outputs[0], add1.inputs[0])
  nt.links.new(em_ring.outputs[0], add1.inputs[1])
  nt.links.new(add1.outputs[0], add2.inputs[0])
  nt.links.new(base.outputs[0], add2.inputs[1])
  # Master glow (keyframed for power-on, flicker and the animation engine's eye channel).
  glow = nt.nodes.new("ShaderNodeMixShader")
  glow.name = "glow"
  glow.inputs["Fac"].default_value = 1.0
  nt.links.new(base.outputs[0], glow.inputs[1])
  nt.links.new(add2.outputs[0], glow.inputs[2])
  o = nt.nodes.new("ShaderNodeOutputMaterial")
  nt.links.new(glow.outputs[0], o.inputs[0])
  return m


def add_eyes(objs, meshes, mats, frame=1):
  """Replace the flat emissive eyes with lens assemblies parented to each eye geom.

  The eye geom is thin along its local x axis; the outward side is the one facing
  away from the rear head cover. Returns [(iris_obj, glow_input, up_axis)]."""
  bpy.context.scene.frame_set(frame)
  idx = {m: i for i, m in enumerate(meshes)}
  face_c = objs[idx["Face"]].matrix_world.translation
  rear_c = objs[idx["Rear_Head_Cover"]].matrix_world.translation
  fwd = (face_c - rear_c).normalized()
  glass = bpy.data.materials.new("lens_glass")
  glass.use_nodes = True
  g = glass.node_tree.nodes["Principled BSDF"]
  g.inputs["Base Color"].default_value = hexrgb("#020304")
  g.inputs["Roughness"].default_value = 0.0
  g.inputs["Transmission Weight"].default_value = 1.0
  g.inputs["IOR"].default_value = 1.52
  if "Coat Weight" in g.inputs:
    g.inputs["Coat Weight"].default_value = 1.0
    g.inputs["Coat Roughness"].default_value = 0.01
  housing = bpy.data.materials.new("eye_housing")
  housing.use_nodes = True
  hb = housing.node_tree.nodes["Principled BSDF"]
  hb.inputs["Base Color"].default_value = hexrgb("#0B0C0E")
  hb.inputs["Roughness"].default_value = 0.22
  hb.inputs["Metallic"].default_value = 0.6
  out = []
  z = np.load(DATA / "meshes.npz")
  for side in ("Left_Eye", "Right_Eye"):
    eye = objs[idx[side]]
    eye.data.materials.clear()
    eye.data.materials.append(housing)
    v = z[f"v_{side}"]
    xw = eye.matrix_world.to_3x3() @ Vector((1, 0, 0))
    sgn = 1.0 if xw.dot(fwd) > 0 else -1.0
    f = z[f"f_{side}"]
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1) / 2
    nrm /= np.maximum(2 * area[:, None], 1e-12)
    xs = tri[:, :, 0].mean(1)
    front = (v[:, 0].max() if sgn > 0 else v[:, 0].min())
    # The visible opening: outward-facing faces within 2 mm of the front plane.
    ring = (nrm[:, 0] * sgn > 0.9) & (np.abs(xs - front) < 0.002)
    c = (tri[ring].mean(1) * area[ring, None]).sum(0) / area[ring].sum()
    cy, cz = c[1], c[2]
    reach = np.sqrt((tri[ring][:, :, 1] - cy) ** 2 + (tri[ring][:, :, 2] - cz) ** 2).max()
    rad = 0.8 * reach
    # Centre on the face plate's square opening instead: its inner walls are the Face
    # faces perpendicular to the plate that lie within ~45 mm of this eye.
    fo = objs[idx["Face"]]
    fv, ff = z["v_Face"], z["f_Face"]
    ftri = fv[ff]
    fn = np.cross(ftri[:, 1] - ftri[:, 0], ftri[:, 2] - ftri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    eye_in_face = fo.matrix_world.inverted() @ (eye.matrix_world @ Vector((front, cy, cz)))
    fc = ftri.mean(1)
    near = np.hypot(fc[:, 1] - eye_in_face.y, fc[:, 2] - eye_in_face.z) < 0.045
    wall = near & (np.abs(fn[:, 0]) < 0.2)
    if wall.sum() > 8:
      wy, wz = ftri[wall][:, :, 1].ravel(), ftri[wall][:, :, 2].ravel()
      hole = Vector((eye_in_face.x, (wy.max() + wy.min()) / 2, (wz.max() + wz.min()) / 2))
      in_eye = eye.matrix_world.inverted() @ (fo.matrix_world @ hole)
      cy, cz = in_eye.y, in_eye.z
      rad = 0.43 * min(np.ptp(wy), np.ptp(wz))
    front += sgn * 0.0006
    rot = Vector((0, 0, 1)).rotation_difference(Vector((sgn, 0, 0)))
    iris_mat = _iris_material(f"iris_{side}")
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=1.0, fill_type="NGON")
    iris = bpy.context.active_object
    iris.name = f"Iris_{side}"
    iris.data.materials.append(iris_mat)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    dome = bpy.context.active_object
    dome.name = f"Lens_{side}"
    dome.data.materials.append(glass)
    for ob in dome.data.polygons:
      ob.use_smooth = True
    for ob, loc, sc in ((iris, front, (rad, rad, 1.0)), (dome, front - sgn * 0.0002, (rad * 1.02, rad * 1.02, rad * 0.18))):
      ob.parent = eye
      ob.matrix_parent_inverse = Matrix.Identity(4)
      ob.rotation_mode = "QUATERNION"
      ob.rotation_quaternion = rot
      ob.location = (loc + (sgn * 0.0004 if ob is iris else 0.0), cy, cz)
      ob.scale = sc
      for c in ob.users_collection:
        c.objects.unlink(ob)
      eye.users_collection[0].objects.link(ob)
    # Head-up direction in the iris's own frame, for blinks.
    up_world = Vector((0, 0, 1))
    m3 = (eye.matrix_world.to_3x3() @ rot.to_matrix())
    up_local = m3.inverted() @ up_world
    up_axis = 0 if abs(up_local.x) > abs(up_local.y) else 1
    out.append((iris, iris_mat.node_tree.nodes["glow"].inputs["Fac"], up_axis, rad))
  return out


def animate_eyes(eyes, frames, glow=None, blinks=(), power_on=None):
  """glow: per-frame multiplier (or None for 1); blinks: frame numbers; power_on: (f0, f1) flicker-in."""
  n = len(frames)
  g = np.ones(n) if glow is None else np.asarray(glow, float)
  if power_on is not None:
    f0, f1 = power_on
    fr = np.asarray(frames, float)
    ramp = np.clip((fr - f0) / max(f1 - f0, 1), 0, 1)
    flick = np.where((fr > f0) & (fr < f1), (np.sin(fr * 2.7) > -0.2).astype(float), 1.0)
    g = g * np.where(fr < f0, 0.0, ramp * flick)
  for iris, fac, up_axis, rad in eyes:
    for f, val in zip(frames[::1], g[::1]):
      fac.default_value = float(np.clip(val, 0, 1))
      fac.keyframe_insert("default_value", frame=int(f))
    base = list(iris.scale)
    for b in blinks:  # 2-frame close, 3-frame open; squash along the head's vertical axis
      for df, s in ((-1, 1.0), (1, 0.08), (2, 0.08), (5, 1.0)):
        sc = list(base)
        sc[up_axis] = base[up_axis] * s
        iris.scale = sc
        iris.keyframe_insert("scale", frame=int(b + df))
    iris.scale = base


def bevel_all(mats, radius=0.0006):
  """Rounded-edge highlights (Cycles Bevel node) on every robot material."""
  for m in mats.values():
    if not m.use_nodes:
      continue
    nt = m.node_tree
    b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if b is None or any(n.type == "BEVEL" for n in nt.nodes):
      continue
    bev = nt.nodes.new("ShaderNodeBevel")
    bev.samples = 6
    bev.inputs["Radius"].default_value = radius
    if b.inputs["Normal"].links:  # keep an existing bump/normal map
      src = b.inputs["Normal"].links[0].from_socket
      nt.links.new(src, bev.inputs["Normal"])
    nt.links.new(bev.outputs["Normal"], b.inputs["Normal"])


def load_clip(name, fps_out=24, speed=1.0, t0=0.0, t1=None):
  """Resample a recorded clip at render-frame times (speed < 1 = slow motion)."""
  z = np.load(DATA / f"{name}.npz", allow_pickle=True)
  pos, quat, fps = z["pos"], z["quat"], float(z["fps"])
  T = pos.shape[0]
  t1 = (T - 1) / fps if t1 is None else t1
  n = int((t1 - t0) / speed * fps_out) + 1
  ts = t0 + np.arange(n) * speed / fps_out
  x = np.clip(ts * fps, 0, T - 1)
  i0 = np.floor(x).astype(int)
  i1 = np.minimum(i0 + 1, T - 1)
  w = (x - i0)[:, None, None]
  q0, q1 = quat[i0], quat[i1]
  q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
  q = (1 - w) * q0 + w * q1
  q /= np.linalg.norm(q, axis=-1, keepdims=True)
  out = {"pos": (1 - w) * pos[i0] + w * pos[i1], "quat": q, "t": ts, "n": n}
  rp = z["root_pos"]
  out["root"] = (1 - w[:, 0]) * rp[i0] + w[:, 0] * rp[i1]
  for k in z.files:
    if k not in ("pos", "quat", "fps", "root_pos"):
      a = z[k]
      out[k] = a[np.clip(np.round(x).astype(int), 0, len(a) - 1)] if a.ndim >= 1 and len(a) == T else a
  return out


def yaw_of(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def placement(clip, at=(0.0, 0.0), heading=0.0, frame=0):
  """(offset, yaw) for animate() so the robot is at `at` facing `heading` at `frame`."""
  yaw = heading - float(yaw_of(clip["root_quat"][frame]))
  c, s = math.cos(yaw), math.sin(yaw)
  p = clip["root"][frame]
  off = (at[0] - (c * p[0] - s * p[1]), at[1] - (s * p[0] + c * p[1]), 0.0)
  return off, yaw


def _fcurves(ob, n):
  """F-curves for location(3) + rotation_quaternion(4), compatible with layered actions."""
  ob.animation_data_create()
  act = bpy.data.actions.new(f"{ob.name}_act")
  ob.animation_data.action = act
  if hasattr(act, "layers"):  # Blender >= 4.4 layered actions
    slot = ob.animation_data.action_slot or act.slots.new(id_type="OBJECT", name=ob.name)
    ob.animation_data.action_slot = slot
    layer = act.layers.new("L") if not act.layers else act.layers[0]
    strip = layer.strips.new(type="KEYFRAME") if not layer.strips else layer.strips[0]
    cb = strip.channelbag(slot, ensure=True)
    new = cb.fcurves.new
  else:
    new = act.fcurves.new
  curves = [new("location", index=i) for i in range(3)] + [new("rotation_quaternion", index=i) for i in range(4)]
  for c in curves:
    c.keyframe_points.add(n)
  return curves


def animate(objs, clip, frame0=1, offset=(0, 0, 0), yaw=0.0):
  """Keyframe every object; offset/yaw place the clip in the set."""
  Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
  qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
  n = clip["n"]
  frames = frame0 + np.arange(n, dtype=np.float32)
  pos = clip["pos"] @ Rz.T + np.asarray(offset)
  q = clip["quat"]
  w1, v1 = qz[0], qz[1:]
  w2, v2 = q[..., :1], q[..., 1:]
  qw = np.concatenate([w1 * w2 - (v2 @ v1)[..., None], w1 * v2 + w2 * v1 + np.cross(v1, v2)], -1)
  for g, ob in enumerate(objs):
    curves = _fcurves(ob, n)
    for c, vals in zip(curves, [pos[:, g, i] for i in range(3)] + [qw[:, g, i] for i in range(4)]):
      co = np.empty(2 * n, np.float32)
      co[0::2], co[1::2] = frames, vals
      c.keyframe_points.foreach_set("co", co)
      c.keyframe_points.foreach_set("interpolation", [1] * n)  # LINEAR
      c.update()
  root = clip["root"] @ Rz.T + np.asarray(offset)
  return frames, root


# ------------------------------------------------------------------ world
def _world_gradient(zenith, horizon, ground, strength=1.0, stars=0.0):
  w = bpy.data.worlds.new("Sky")
  bpy.context.scene.world = w
  w.use_nodes = True
  nt = w.node_tree
  nt.nodes.clear()
  out = nt.nodes.new("ShaderNodeOutputWorld")
  bg = nt.nodes.new("ShaderNodeBackground")
  bg.inputs["Strength"].default_value = strength
  tc = nt.nodes.new("ShaderNodeTexCoord")
  sep = nt.nodes.new("ShaderNodeSeparateXYZ")
  nt.links.new(tc.outputs["Generated"], sep.inputs[0])
  ramp = nt.nodes.new("ShaderNodeValToRGB")
  r = ramp.color_ramp
  r.interpolation = "EASE"
  r.elements[0].position, r.elements[0].color = 0.47, hexrgb(ground)
  r.elements[1].position, r.elements[1].color = 0.505, hexrgb(horizon)
  e = r.elements.new(0.75)
  e.color = hexrgb(zenith)
  zf = nt.nodes.new("ShaderNodeMapRange")  # world direction z in [-1, 1] -> [0, 1]
  zf.inputs["From Min"].default_value = -1.0
  nt.links.new(sep.outputs["Z"], zf.inputs["Value"])
  nt.links.new(zf.outputs["Result"], ramp.inputs["Fac"])
  # Generated coords on the world map direction -> [0,1]; z = 0.5 at the horizon.
  col = ramp.outputs["Color"]
  if stars > 0:
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 700.0
    vor.feature = "DISTANCE_TO_EDGE" if False else "F1"
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = 0.11
    mr.inputs["To Min"].default_value = stars
    mr.inputs["To Max"].default_value = 0.0
    nt.links.new(tc.outputs["Generated"], vor.inputs["Vector"])
    nt.links.new(vor.outputs["Distance"], mr.inputs["Value"])
    mask = nt.nodes.new("ShaderNodeMapRange")  # no stars below the horizon haze
    mask.inputs["From Min"].default_value = 0.51
    mask.inputs["From Max"].default_value = 0.56
    nt.links.new(zf.outputs["Result"], mask.inputs["Value"])
    # Only ~8% of cells hold a star; brightness varies per cell.
    sepc = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(vor.outputs["Color"], sepc.inputs[0])
    pick = nt.nodes.new("ShaderNodeMapRange")
    pick.inputs["From Min"].default_value = 0.8
    pick.inputs["From Max"].default_value = 1.0
    pick.inputs["To Min"].default_value = 0.0
    pick.inputs["To Max"].default_value = 1.0
    nt.links.new(sepc.outputs[0], pick.inputs["Value"])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 2.0
    nt.links.new(pick.outputs["Result"], pw.inputs[0])
    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation = "MULTIPLY"
    nt.links.new(mr.outputs["Result"], m1.inputs[0])
    nt.links.new(pw.outputs[0], m1.inputs[1])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(m1.outputs[0], mul.inputs[0])
    nt.links.new(mask.outputs["Result"], mul.inputs[1])
    add = nt.nodes.new("ShaderNodeMix")
    add.data_type = "RGBA"
    add.blend_type = "ADD"
    add.inputs["Factor"].default_value = 1.0
    nt.links.new(col, add.inputs["A"])
    nt.links.new(mul.outputs[0], add.inputs["B"])
    col = add.outputs["Result"]
  nt.links.new(col, bg.inputs["Color"])
  nt.links.new(bg.outputs[0], out.inputs["Surface"])
  return w


def _world_sky(elev, azim, tint, strength=1.0):
  """Physically based multiple-scattering sky (dusty), warmed by a tint."""
  w = bpy.data.worlds.new("Sky")
  bpy.context.scene.world = w
  w.use_nodes = True
  nt = w.node_tree
  nt.nodes.clear()
  sky = nt.nodes.new("ShaderNodeTexSky")
  for t in ("MULTIPLE_SCATTERING", "NISHITA"):
    try:
      sky.sky_type = t
      break
    except TypeError:
      continue
  sky.sun_elevation = math.radians(elev)
  sky.sun_rotation = math.radians(azim + 90)
  sky.sun_disc = False
  for k, v in (("air_density", 1.4), ("aerosol_density", 6.0), ("dust_density", 6.0), ("ozone_density", 1.0), ("altitude", 300.0)):
    if hasattr(sky, k):
      setattr(sky, k, v)
  mix = nt.nodes.new("ShaderNodeMix")
  mix.data_type = "RGBA"
  mix.blend_type = "MULTIPLY"
  mix.inputs["Factor"].default_value = 1.0
  mix.inputs["B"].default_value = hexrgb(tint)
  nt.links.new(sky.outputs["Color"], mix.inputs["A"])
  bg = nt.nodes.new("ShaderNodeBackground")
  bg.inputs["Strength"].default_value = strength
  nt.links.new(mix.outputs["Result"], bg.inputs["Color"])
  out = nt.nodes.new("ShaderNodeOutputWorld")
  nt.links.new(bg.outputs[0], out.inputs["Surface"])
  return w


def _sun(name, elev, azim, color, strength, angle=0.6):
  d = bpy.data.lights.new(name, "SUN")
  d.energy = strength
  d.color = hexrgb(color)[:3]
  d.angle = math.radians(angle)
  ob = bpy.data.objects.new(name, d)
  ob.rotation_euler = Euler((math.radians(90 - elev), 0, math.radians(azim + 90)))
  bpy.context.scene.collection.objects.link(ob)
  return ob


def _disc(name, elev, azim, radius, color, strength, dist=900.0):
  """Visible sun/planet disc on the sky dome, facing the origin."""
  bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=radius)
  ob = bpy.context.active_object
  ob.name = name
  ob.data.shade_smooth()
  e, a = math.radians(elev), math.radians(azim)
  ob.location = (dist * math.cos(e) * math.cos(a), dist * math.cos(e) * math.sin(a), dist * math.sin(e))
  m = bpy.data.materials.new(name)
  m.use_nodes = True
  nt = m.node_tree
  nt.nodes.clear()
  em = nt.nodes.new("ShaderNodeEmission")
  em.inputs["Color"].default_value = hexrgb(color)
  em.inputs["Strength"].default_value = strength
  o = nt.nodes.new("ShaderNodeOutputMaterial")
  nt.links.new(em.outputs[0], o.inputs[0])
  ob.data.materials.append(m)
  ob.visible_shadow = False
  return ob


def _planet_shading(ob, light_az, light_el):
  """Gas giant: banded colour, crescent terminator, thin blue limb glow."""
  m = ob.data.materials[0]
  nt = m.node_tree
  nt.nodes.clear()
  geo = nt.nodes.new("ShaderNodeNewGeometry")
  a, e = math.radians(light_az), math.radians(light_el)
  dot = nt.nodes.new("ShaderNodeVectorMath")
  dot.operation = "DOT_PRODUCT"
  dot.inputs[1].default_value = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
  nt.links.new(geo.outputs["Normal"], dot.inputs[0])
  term = nt.nodes.new("ShaderNodeMapRange")
  term.inputs["From Min"].default_value = -0.15
  term.inputs["From Max"].default_value = 0.35
  nt.links.new(dot.outputs["Value"], term.inputs["Value"])
  tc = nt.nodes.new("ShaderNodeTexCoord")
  wave = nt.nodes.new("ShaderNodeTexWave")
  wave.bands_direction = "Z"
  wave.inputs["Scale"].default_value = 3.0
  wave.inputs["Distortion"].default_value = 4.0
  wave.inputs["Detail"].default_value = 3.0
  nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
  ramp = nt.nodes.new("ShaderNodeValToRGB")
  ramp.color_ramp.elements[0].color = hexrgb("#5E78B0")
  ramp.color_ramp.elements[1].color = hexrgb("#D8B48C")
  nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
  em = nt.nodes.new("ShaderNodeEmission")
  nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
  sm = nt.nodes.new("ShaderNodeMath")
  sm.operation = "MULTIPLY"
  sm.inputs[1].default_value = 0.85
  nt.links.new(term.outputs["Result"], sm.inputs[0])
  nt.links.new(sm.outputs[0], em.inputs["Strength"])
  lw = nt.nodes.new("ShaderNodeLayerWeight")
  lw.inputs["Blend"].default_value = 0.15
  limb = nt.nodes.new("ShaderNodeEmission")
  limb.inputs["Color"].default_value = hexrgb("#6FA8FF")
  nt.links.new(lw.outputs["Facing"], limb.inputs["Strength"])
  add = nt.nodes.new("ShaderNodeAddShader")
  nt.links.new(em.outputs[0], add.inputs[0])
  nt.links.new(limb.outputs[0], add.inputs[1])
  o = nt.nodes.new("ShaderNodeOutputMaterial")
  nt.links.new(add.outputs[0], o.inputs[0])


def _rings(planet, r):
  """Thin tilted ring system around the night planet."""
  bpy.ops.mesh.primitive_circle_add(vertices=128, radius=r * 2.3, fill_type="NGON", location=planet.location)
  ob = bpy.context.active_object
  ob.name = "Rings"
  ob.rotation_euler = (math.radians(10), math.radians(-22), math.radians(350 + 90))
  m = bpy.data.materials.new("rings")
  m.use_nodes = True
  nt = m.node_tree
  nt.nodes.clear()
  tc = nt.nodes.new("ShaderNodeTexCoord")
  grad = nt.nodes.new("ShaderNodeTexGradient")
  grad.gradient_type = "SPHERICAL"
  mp = nt.nodes.new("ShaderNodeMapping")
  mp.inputs["Scale"].default_value = (1 / (r * 2.3),) * 3
  nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
  nt.links.new(mp.outputs[0], grad.inputs["Vector"])
  ramp = nt.nodes.new("ShaderNodeValToRGB")
  e = ramp.color_ramp.elements
  e[0].position, e[0].color = 0.0, (0, 0, 0, 1)
  e[1].position, e[1].color = 0.52, (0, 0, 0, 1)
  for pos, val in ((0.3, 0.0), (0.36, 0.7), (0.4, 0.25), (0.45, 0.9), (0.5, 0.0)):
    el = e.new(pos)
    el.color = (val, val, val, 1)
  nt.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
  em = nt.nodes.new("ShaderNodeEmission")
  em.inputs["Color"].default_value = hexrgb("#C9D3EE")
  em.inputs["Strength"].default_value = 0.6
  tr = nt.nodes.new("ShaderNodeBsdfTransparent")
  mix = nt.nodes.new("ShaderNodeMixShader")
  nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
  nt.links.new(tr.outputs[0], mix.inputs[1])
  nt.links.new(em.outputs[0], mix.inputs[2])
  o = nt.nodes.new("ShaderNodeOutputMaterial")
  nt.links.new(mix.outputs[0], o.inputs[0])
  ob.data.materials.append(m)
  ob.visible_shadow = False
  return ob


def _haze_mix(nt, shader_out, haze_color, near=25.0, far=420.0):
  """Aerial perspective: blend toward the horizon colour with camera distance."""
  cam = nt.nodes.new("ShaderNodeCameraData")
  mr = nt.nodes.new("ShaderNodeMapRange")
  mr.inputs["From Min"].default_value = near
  mr.inputs["From Max"].default_value = far
  mr.inputs["To Max"].default_value = 0.85
  nt.links.new(cam.outputs["View Distance"], mr.inputs["Value"])
  em = nt.nodes.new("ShaderNodeEmission")
  em.inputs["Color"].default_value = hexrgb(haze_color)
  em.inputs["Strength"].default_value = 1.0
  mix = nt.nodes.new("ShaderNodeMixShader")
  nt.links.new(mr.outputs["Result"], mix.inputs["Fac"])
  nt.links.new(shader_out, mix.inputs[1])
  nt.links.new(em.outputs[0], mix.inputs[2])
  return mix.outputs[0]


def _dunes(size=3400.0, res=560, flat_r=14.0, seed=3):
  rng = np.random.default_rng(seed)
  xs = np.linspace(-size / 2, size / 2, res)
  # Non-uniform grid: denser near the centre where the robot walks.
  xs = np.sign(xs) * (np.abs(xs) / (size / 2)) ** 1.7 * (size / 2)
  X, Y = np.meshgrid(xs, xs)
  H = np.zeros_like(X)
  for k in range(7):
    th = rng.uniform(0, math.pi)
    lam = rng.uniform(25, 140)
    ph = rng.uniform(0, 2 * math.pi)
    u = X * math.cos(th) + Y * math.sin(th)
    H += rng.uniform(0.6, 2.2) * (lam / 80) * (0.5 + 0.5 * np.sin(2 * math.pi * u / lam + ph)) ** 2.2
  r = np.hypot(X, Y)
  ramp = np.clip((r - flat_r) / 60.0, 0, 1)
  # Taller dunes with distance, then settle under the ridge ring (r ~ 1400 m) so no edge shows.
  H = H * ramp**2 * (1 + 2.5 * np.clip((r - 120) / 300, 0, 1)) * (1 - 0.6 * np.clip((r - 900) / 500, 0, 1))
  V = np.stack([X.ravel(), Y.ravel(), H.ravel()], -1)
  F = []
  for i in range(res - 1):
    for j in range(res - 1):
      a = i * res + j
      F.append((a, a + 1, a + res + 1, a + res))
  me = bpy.data.meshes.new("dunes")
  me.from_pydata(V.tolist(), [], F)
  me.shade_smooth()
  ob = bpy.data.objects.new("Dunes", me)
  bpy.context.scene.collection.objects.link(ob)
  return ob


def _sand_material(base, haze, night=False):
  m = bpy.data.materials.new("sand")
  m.use_nodes = True
  nt = m.node_tree
  b = nt.nodes["Principled BSDF"]
  b.inputs["Roughness"].default_value = 0.92
  tc = nt.nodes.new("ShaderNodeTexCoord")
  # Wind ripples: distorted bands, plus grain.
  wave = nt.nodes.new("ShaderNodeTexWave")
  wave.inputs["Scale"].default_value = 8.0
  wave.inputs["Distortion"].default_value = 6.0
  wave.inputs["Detail"].default_value = 3.0
  wave.inputs["Detail Scale"].default_value = 1.5
  nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
  grain = nt.nodes.new("ShaderNodeTexNoise")
  grain.inputs["Scale"].default_value = 260.0
  nt.links.new(tc.outputs["Object"], grain.inputs["Vector"])
  big = nt.nodes.new("ShaderNodeTexNoise")
  big.inputs["Scale"].default_value = 0.08
  big.inputs["Detail"].default_value = 4.0
  nt.links.new(tc.outputs["Object"], big.inputs["Vector"])
  ramp = nt.nodes.new("ShaderNodeValToRGB")
  ramp.color_ramp.elements[0].color = hexrgb(base[0])
  ramp.color_ramp.elements[1].color = hexrgb(base[1])
  nt.links.new(big.outputs["Fac"], ramp.inputs["Fac"])
  nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
  add = nt.nodes.new("ShaderNodeMath")
  add.operation = "MULTIPLY_ADD"
  add.inputs[1].default_value = 0.25
  nt.links.new(grain.outputs["Fac"], add.inputs[0])
  nt.links.new(wave.outputs["Fac"], add.inputs[2])
  bump = nt.nodes.new("ShaderNodeBump")
  bump.inputs["Strength"].default_value = 0.14
  bump.inputs["Distance"].default_value = 0.008
  nt.links.new(add.outputs[0], bump.inputs["Height"])
  nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
  out = nt.nodes["Material Output"]
  nt.links.new(_haze_mix(nt, b.outputs[0], haze, 45.0, 700.0), out.inputs["Surface"])
  return m


def _rock_material(haze):
  m = bpy.data.materials.new("rock")
  m.use_nodes = True
  nt = m.node_tree
  b = nt.nodes["Principled BSDF"]
  b.inputs["Roughness"].default_value = 0.95
  tc = nt.nodes.new("ShaderNodeTexCoord")
  wave = nt.nodes.new("ShaderNodeTexWave")  # sedimentary strata
  wave.wave_type = "BANDS"
  wave.bands_direction = "Z"
  wave.inputs["Scale"].default_value = 0.06
  wave.inputs["Distortion"].default_value = 3.0
  wave.inputs["Detail"].default_value = 6.0
  nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
  ramp = nt.nodes.new("ShaderNodeValToRGB")
  ramp.color_ramp.elements[0].color = hexrgb("#6E4632")
  ramp.color_ramp.elements[1].color = hexrgb("#8E5E40")
  nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
  nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
  out = nt.nodes["Material Output"]
  nt.links.new(_haze_mix(nt, b.outputs[0], haze, 200.0, 1600.0), out.inputs["Surface"])
  return m


def _mesa(loc, radius, height, mat, seed):
  bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=height, location=(loc[0], loc[1], height / 2 - 2))
  ob = bpy.context.active_object
  ob.name = f"Mesa{seed}"
  sub = ob.modifiers.new("sub", "SUBSURF")
  sub.levels = sub.render_levels = 3
  tex = bpy.data.textures.new(f"mesa{seed}", "CLOUDS")
  tex.noise_scale = radius * 0.35
  disp = ob.modifiers.new("disp", "DISPLACE")
  disp.texture = tex
  disp.strength = radius * 0.25
  ob.scale = (1.0, 0.7 + 0.2 * (seed % 3), 1.0)
  ob.data.materials.append(mat)
  return ob


def _ridges(mat, seed=5, r0=1400.0, n=720):
  """Ring of distant ridge lines / mesas: flat-topped buttes, steep faces."""
  rng = np.random.default_rng(seed)
  th = np.linspace(0, 2 * math.pi, n, endpoint=False)
  h = np.zeros(n)
  for _ in range(9):
    c, w, top = rng.uniform(0, 2 * math.pi), rng.uniform(0.05, 0.22), rng.uniform(70, 230)
    d = np.angle(np.exp(1j * (th - c)))
    h = np.maximum(h, top * np.clip(1.25 - (np.abs(d) / w) ** 6, 0, 1) ** 0.25)
  h += 18 * np.abs(np.sin(th * 7 + 1.3)) + 10 * np.sin(th * 23) ** 2
  rings = [(r0 + 120, -5.0), (r0, 1.0), (r0 - 4, 1.0)]
  V = []
  for k, (r, frac) in enumerate(((r0 + 160, 0.0), (r0 + 40, 1.0), (r0, 0.96), (r0 - 30, 0.0))):
    z = h * frac - (8 if frac == 0 else 0)
    V += list(zip(r * np.cos(th), r * np.sin(th), z))
  F = []
  for k in range(3):
    for i in range(n):
      a, b = k * n + i, k * n + (i + 1) % n
      F.append((a, b, b + n, a + n))
  me = bpy.data.meshes.new("ridges")
  me.from_pydata(V, [], F)
  ob = bpy.data.objects.new("Ridges", me)
  bpy.context.scene.collection.objects.link(ob)
  ob.data.materials.append(mat)
  return ob


def _homestead(loc, r, mat, door_mat, rot=0.0):
  """A low adobe dome hut with a dark arched doorway, half-sunk into the sand."""
  bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=r, location=(loc[0], loc[1], -0.25 * r))
  dome = bpy.context.active_object
  dome.scale = (1.0, 1.0, 0.72)
  dome.data.shade_smooth()
  dome.data.materials.append(mat)
  bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.42 * r, depth=0.5 * r,
                                      location=(loc[0] + 0.92 * r * math.cos(rot), loc[1] + 0.92 * r * math.sin(rot), 0.12 * r),
                                      rotation=(math.pi / 2, 0, rot + math.pi / 2))
  arch = bpy.context.active_object
  arch.data.materials.append(mat)
  bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.3 * r, depth=0.06 * r,
                                      location=(loc[0] + 1.17 * r * math.cos(rot), loc[1] + 1.17 * r * math.sin(rot), 0.1 * r),
                                      rotation=(math.pi / 2, 0, rot + math.pi / 2))
  door = bpy.context.active_object
  door.data.materials.append(door_mat)
  return [dome, arch, door]


def _vaporator(loc, h=4.2, mat=None):
  """Tall moisture-farm tower silhouette: a slim mast with stacked collars."""
  parts = []
  bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.16, depth=h, location=(loc[0], loc[1], h / 2))
  parts.append(bpy.context.active_object)
  for k, (z, r) in enumerate(((0.35, 0.42), (h * 0.55, 0.3), (h * 0.62, 0.26), (h - 0.25, 0.22))):
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=r, depth=0.18 if k else 0.7, location=(loc[0], loc[1], z))
    parts.append(bpy.context.active_object)
  for p in parts:
    p.data.materials.append(mat)
  return parts


def desert_world(night=False):
  sc = bpy.context.scene
  if not night:
    _world_sky(elev=7.0, azim=205.0, tint="#FFC79A", strength=0.32)
    haze = "#C98B66"
    _sun("SunA", 8.0, 203.0, "#FFCB94", 3.6, 0.55)
    _sun("SunB", 5.0, 214.0, "#FF8A4C", 1.4, 0.9)
    _disc("DiscA", 8.0, 203.0, 11.0, "#FFF0D2", 45.0)
    _disc("DiscB", 5.0, 214.0, 7.0, "#FF9A5C", 35.0)
    sand = ("#B9875A", "#D8AE80")
  else:
    _world_gradient(zenith="#03060F", horizon="#1E2C52", ground="#1A1D2B", strength=1.0, stars=40.0)
    haze = "#121B33"
    _sun("Moon", 32.0, 150.0, "#9FB8FF", 1.1, 1.5)
    _sun("Rim", 4.0, 330.0, "#FF8A4A", 0.9, 2.0)
    pl = _disc("Planet", 15.0, 350.0, 95.0, "#B7C6E6", 1.6, 1500.0)
    _planet_shading(pl, light_az=110.0, light_el=15.0)
    _rings(pl, 95.0)
    sand = ("#5A4A47", "#7B6459")
  dunes = _dunes()
  sm = _sand_material(sand, haze, night)
  for nd in sm.node_tree.nodes:  # texture space = the dunes, so footprints share the same ripples
    if nd.type == "TEX_COORD":
      nd.object = dunes
  dunes.data.materials.append(sm)
  rock = _rock_material(haze)
  _ridges(rock, seed=5 if not night else 9)
  metal = bpy.data.materials.new("vapor")
  metal.use_nodes = True
  bb = metal.node_tree.nodes["Principled BSDF"]
  bb.inputs["Base Color"].default_value = hexrgb("#8C8378")
  bb.inputs["Metallic"].default_value = 0.6
  bb.inputs["Roughness"].default_value = 0.55
  for x, y in ((-9.5, 14.0), (16.0, 22.0), (-30.0, -18.0)):
    _vaporator((x, y), 4.4, metal)
  adobe = bpy.data.materials.new("adobe")
  adobe.use_nodes = True
  ab = adobe.node_tree.nodes["Principled BSDF"]
  ab.inputs["Base Color"].default_value = hexrgb("#C9A27A" if not night else "#5A4C48")
  ab.inputs["Roughness"].default_value = 0.95
  dark = bpy.data.materials.new("doorway")
  dark.use_nodes = True
  dark.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = hexrgb("#1A1210")
  for (x, y), r, rot in (((-15.0, 21.0), 2.6, -1.2), ((23.0, 31.0), 3.2, -2.0), ((-40.0, -25.0), 2.8, 0.6), ((60.0, -12.0), 3.0, 3.0)):
    _homestead((x, y), r, adobe, dark, rot)
  sc.view_settings.view_transform = "AgX"
  sc.view_settings.exposure = -0.35 if not night else 0.2
  for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - Punchy", "Punchy"):
    try:
      sc.view_settings.look = look
      break
    except TypeError:
      continue
  return haze


# ------------------------------------------------------------------ camera & render
def camera(name="Cam", lens=50.0, fstop=2.8):
  cd = bpy.data.cameras.new(name)
  cd.lens = lens
  cd.sensor_width = 36.0
  cd.dof.use_dof = True
  cd.dof.aperture_fstop = fstop
  cd.clip_end = 3000
  ob = bpy.data.objects.new(name, cd)
  bpy.context.scene.collection.objects.link(ob)
  bpy.context.scene.camera = ob
  return ob


def smooth(x, k=12):
  """Moving-average along time (camera follow lag)."""
  if len(x) < 3:
    return x
  pad = np.pad(x, ((k, k), (0, 0)), mode="edge")
  ker = np.ones(2 * k + 1) / (2 * k + 1)
  return np.stack([np.convolve(pad[:, i], ker, mode="valid") for i in range(x.shape[1])], -1)


def key_camera(cam, frames, eye, target, focus=None):
  """eye/target: [n,3] arrays. Keys location + look-at rotation per frame."""
  cam.rotation_mode = "QUATERNION"
  for f, e, t in zip(frames, eye, target):
    cam.location = Vector(e)
    d = Vector(t) - Vector(e)
    cam.rotation_quaternion = d.to_track_quat("-Z", "Y")
    cam.keyframe_insert("location", frame=int(f))
    cam.keyframe_insert("rotation_quaternion", frame=int(f))
    if focus is None:
      cam.data.dof.focus_distance = d.length
      cam.data.dof.keyframe_insert("focus_distance", frame=int(f))


def render_settings(w=1920, h=1080, fps=24, engine=None, samples=64, motion_blur=True):
  """EEVEE by default; BDX_ENGINE=CYCLES path-traces on Metal (macOS) or OptiX (NVIDIA)."""
  engine = engine or os.environ.get("BDX_ENGINE", "EEVEE")
  if engine == "CYCLES":
    samples = int(os.environ.get("BDX_SAMPLES", samples))
  sc = bpy.context.scene
  sc.render.resolution_x, sc.render.resolution_y = w, h
  sc.render.resolution_percentage = 100
  sc.render.fps = fps
  if engine == "CYCLES":
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL" if sys.platform == "darwin" else "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
      d.use = True
    sc.cycles.device = "GPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces = 6, 3, 3
    sc.cycles.transmission_bounces, sc.cycles.volume_bounces = 4, 0
    sc.cycles.use_auto_tile = True
    sc.render.use_persistent_data = True  # keep static geometry/textures between frames
  else:
    for e in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
      try:
        sc.render.engine = e
        break
      except TypeError:
        continue
    ee = sc.eevee
    ee.taa_render_samples = samples
    for attr, val in (("use_raytracing", True), ("use_shadows", True), ("shadow_ray_count", 2), ("shadow_step_count", 8), ("use_volumetric_shadows", True)):
      if hasattr(ee, attr):
        setattr(ee, attr, val)
    if hasattr(ee, "ray_tracing_options"):
      ee.ray_tracing_options.resolution_scale = "1"
  sc.render.use_motion_blur = motion_blur
  if hasattr(sc.render, "motion_blur_shutter"):
    sc.render.motion_blur_shutter = 0.5
  sc.render.film_transparent = False
  sc.render.image_settings.file_format = "PNG"
  sc.render.image_settings.color_mode = "RGB"
  compositor_glare()


def compositor_glare():
  """Bloom on emissive eyes / suns plus a soft vignette."""
  sc = bpy.context.scene
  try:
    if hasattr(sc, "compositing_node_group"):  # Blender 5.x
      ng = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
      ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
      sc.compositing_node_group = ng
      rl = ng.nodes.new("CompositorNodeRLayers")
      out = ng.nodes.new("NodeGroupOutput")
    else:
      sc.use_nodes = True
      ng = sc.node_tree
      ng.nodes.clear()
      rl = ng.nodes.new("CompositorNodeRLayers")
      out = ng.nodes.new("CompositorNodeComposite")
    gl = ng.nodes.new("CompositorNodeGlare")
    for key, vals in (("Type", ("Bloom", "BLOOM")), ("Quality", ("High", "HIGH"))):
      for v in vals:
        try:
          if key in gl.inputs:
            gl.inputs[key].default_value = v
          else:
            setattr(gl, key.lower() if key == "Quality" else "glare_type", v)
          break
        except (TypeError, AttributeError):
          continue
    for key, val in (("Threshold", 1.5), ("Size", 0.7), ("Strength", 0.35), ("Smoothness", 0.4)):
      if key in gl.inputs:
        gl.inputs[key].default_value = val
    if hasattr(gl, "threshold"):
      gl.threshold = 1.2
    ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
    ng.links.new(gl.outputs["Image"], out.inputs[0])
  except Exception as e:  # compositor API drift: render without glare
    print("compositor skipped:", e)


def reset_scene():
  bpy.ops.wm.read_factory_settings(use_empty=True)
