"""Bake Meshy v2 PBR textures onto the simulation meshes (Cycles, selected-to-active).

blender -b -P blender/bake.py -- [PART ...]
For every target mesh: smart-UV-unwrap the untouched simulation geometry, align the
Meshy source (axis/mirror-aware bounding-box fit), and bake base colour, roughness,
metallic and tangent-space normal into results/bdx/baked/<part>_<map>.png. The UVs go
to <part>_uv.npy in face-corner order, which build_robot() restores at render time.
"""

import os
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bdx_lib as L  # noqa: E402
from meshy_io import _align  # noqa: E402

SRC = L.ROOT / "results" / "bdx" / "meshy2" / "out"
OUT = L.ROOT / "results" / "bdx" / "baked"
OUT.mkdir(parents=True, exist_ok=True)


def sides(p):
  return [f"Left_{p}", f"Right_{p}"]


# Target meshes painted by each Meshy source (identical or mirrored shapes share one source).
TARGETS = {
  "Left_Hip_Yaw_Motor": sides("Hip_Yaw_Motor") + sides("Hip_Pitch_Motor"),
  "Left_Knee_Motor": sides("Knee_Motor"),
  "Left_Hip_Roll_Motor": sides("Hip_Roll_Motor"),
  "Left_Ankle_Motor": sides("Ankle_Motor"),
  "Neck_Motor": ["Neck_Motor"],
  "Head_Pitch_Motor": ["Head_Pitch_Motor", "Head_Yaw_Motor", "Head_Roll_Motor"] + sides("Ear_Motor"),
  "Left_Upper_Leg": sides("Upper_Leg"),
  "Left_Lower_Leg": sides("Lower_Leg"),
  "Left_Foot": sides("Foot"),
  "Left_Holder": sides("Holder"),
  "Left_Hip_Top_Holder": sides("Hip_Top_Holder"),
  "Left_Hip_Yaw_Support": sides("Hip_Yaw_Support"),
  "Front_Support": ["Front_Support"],
  "Neck": ["Neck"],
  "Left_Head_Support": sides("Head_Support"),
  "Head_Pitch_Motor_Support": ["Head_Pitch_Motor_Support"],
}
SHELL = sorted(n for r in ("ivory", "ember", "slate") for n in L.ROLES[r] if not n.endswith(("_Ear", "Upper_Leg_Detail")))
for n in SHELL:
  TARGETS[n] = [n]


def load_source(glb):
  before = set(bpy.data.objects)
  bpy.ops.import_scene.gltf(filepath=str(glb))
  obs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
  bpy.ops.object.select_all(action="DESELECT")
  for o in obs:
    o.select_set(True)
  bpy.context.view_layer.objects.active = obs[0]
  if len(obs) > 1:
    bpy.ops.object.join()
  src = bpy.context.view_layer.objects.active
  bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
  for o in [o for o in bpy.data.objects if o not in before and o != src]:
    bpy.data.objects.remove(o)
  return src


def emission_pass(mat, socket_name):
  """Route one Principled input (roughness / metallic) through Emission for an EMIT bake."""
  nt = mat.node_tree
  b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
  out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
  em = nt.nodes.new("ShaderNodeEmission")
  sock = b.inputs[socket_name]
  if sock.links:
    nt.links.new(sock.links[0].from_socket, em.inputs["Color"])
  else:
    v = sock.default_value
    em.inputs["Color"].default_value = (v, v, v, 1.0)
  keep = out.inputs["Surface"].links[0].from_socket
  nt.links.new(em.outputs[0], out.inputs["Surface"])
  return lambda: nt.links.new(keep, out.inputs["Surface"])


def bake_one(src_name, target, z):
  L.reset_scene()
  sc = bpy.context.scene
  sc.render.engine = "CYCLES"
  prefs = bpy.context.preferences.addons["cycles"].preferences
  prefs.compute_device_type = "METAL" if sys.platform == "darwin" else "OPTIX"
  prefs.get_devices()
  for d in prefs.devices:
    d.use = True
  sc.cycles.device = "GPU"
  sc.cycles.samples = 8
  # Target: the untouched simulation mesh, in millimetres (the Meshy fit lives in mm).
  v = z[f"v_{target}"] * 1000.0
  f = z[f"f_{target}"]
  me = bpy.data.meshes.new(target)
  me.from_pydata(v.tolist(), [], f.tolist())
  me.validate()
  me.shade_smooth()
  me.set_sharp_from_angle(angle=np.radians(35))
  tgt = bpy.data.objects.new(target, me)
  sc.collection.objects.link(tgt)
  bpy.ops.object.select_all(action="DESELECT")
  bpy.context.view_layer.objects.active = tgt
  tgt.select_set(True)
  bpy.ops.object.mode_set(mode="EDIT")
  bpy.ops.mesh.select_all(action="SELECT")
  bpy.ops.uv.smart_project(angle_limit=np.radians(60), island_margin=0.004, area_weight=0.6, scale_to_bounds=False)
  bpy.ops.object.mode_set(mode="OBJECT")
  size = np.linalg.norm(np.ptp(v, axis=0))
  res = 4096 if size > 200 else 2048
  # Source: Meshy GLB aligned onto the target.
  src = load_source(SRC / f"{src_name}.glb")
  sv = np.array([tuple(p.co) for p in src.data.vertices])
  fit, err = _align(sv, v)
  src.data.vertices.foreach_set("co", fit(sv).astype(np.float32).ravel())
  src.data.update()
  smat = src.material_slots[0].material
  # Bake target material: one image node per pass, the active node receives the bake.
  tm = bpy.data.materials.new("bake")
  tm.use_nodes = True
  tgt.data.materials.append(tm)
  img_node = tm.node_tree.nodes.new("ShaderNodeTexImage")
  tm.node_tree.nodes.active = img_node
  bk = sc.render.bake
  bk.use_selected_to_active = True
  bk.cage_extrusion = 1.2
  bk.max_ray_distance = 3.0
  bk.margin = 12
  bpy.ops.object.select_all(action="DESELECT")
  src.select_set(True)
  tgt.select_set(True)
  bpy.context.view_layer.objects.active = tgt
  out = {}
  for name, kind, cs in (("base_color", "DIFFUSE", "sRGB"), ("normal", "NORMAL", "Non-Color"), ("roughness", "EMIT", "Non-Color"), ("metallic", "EMIT", "Non-Color")):
    img = bpy.data.images.new(f"{target}_{name}", res, res, alpha=False, float_buffer=False)
    img.colorspace_settings.name = cs
    img_node.image = img
    restore = None
    if name in ("roughness", "metallic"):
      restore = emission_pass(smat, "Roughness" if name == "roughness" else "Metallic")
    if kind == "DIFFUSE":
      bk.use_pass_direct = bk.use_pass_indirect = False
      bk.use_pass_color = True
    bpy.ops.object.bake(type=kind)
    if restore:
      restore()
    path = OUT / f"{target}_{name}.png"
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    out[name] = path
  uv = np.zeros(len(tgt.data.loops) * 2, np.float32)
  tgt.data.uv_layers.active.data.foreach_get("uv", uv)
  np.save(OUT / f"{target}_uv.npy", uv.reshape(-1, 3, 2))
  print(f"baked {target} <- {src_name}: fit err {err:.3f}, {res}px", flush=True)


if __name__ == "__main__":
  argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
  z = np.load(L.DATA / "meshes.npz")
  for src_name, targets in TARGETS.items():
    if not (SRC / f"{src_name}.glb").exists():
      continue
    for t in targets:
      if argv and t not in argv and src_name not in argv:
        continue
      if (OUT / f"{t}_uv.npy").exists() and "--force" not in argv:
        continue
      bake_one(src_name, t, z)
