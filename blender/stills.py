"""Illustrations: key art, livery model sheet, blueprint readout.

blender -b -P blender/stills.py -- (keyart|keyart_night|sheet|blueprint) [--preview]
"""

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
import bdx_lib as L  # noqa: E402

OUT = L.ROOT / "results" / "bdx" / "stills"
OUT.mkdir(parents=True, exist_ok=True)


def robot_at(clip_name, t, heading, world=None, night=False):
  L.reset_scene()
  if world == "desert":
    L.desert_world(night)
  mats = L.make_materials()
  objs, meshes = L.build_robot(mats)
  clip = L.load_clip(clip_name, 50, 1.0, t, t + 0.04)
  off, yaw = L.placement(clip, (0, 0), heading, 0)
  _, root = L.animate(objs, clip, offset=off, yaw=yaw)
  eyes = L.add_eyes(objs, meshes, mats, 1)
  L.animate_eyes(eyes, [1])
  L.bevel_all(mats)
  bpy.context.scene.frame_set(1)
  return objs, meshes, root[0], mats


def look(cam, eye, tgt):
  cam.location = Vector(eye)
  cam.rotation_mode = "QUATERNION"
  cam.rotation_quaternion = (Vector(tgt) - Vector(eye)).to_track_quat("-Z", "Y")
  cam.data.dof.focus_distance = (Vector(tgt) - Vector(eye)).length


def keyart(night=False):
  objs, _, r, mats = robot_at("walking_commands", 3.1, math.radians(25 if not night else 170), "desert", night)
  cam = L.camera(lens=24, fstop=2.0 if not night else 5.6)  # night: stars stay pinpoints
  if not night:
    # Low hero angle facing the droid; the twin suns sit over its shoulder.
    look(cam, r + np.array([1.3, 0.62, -0.02]), r + np.array([-0.05, -0.5, 0.16]))
  else:
    look(cam, r + np.array([-1.45, 0.62, -0.18]), r + np.array([0, 0, 0.16]))
  return 2560, 1440


def studio(mats):
  w = bpy.data.worlds.new("Studio")
  bpy.context.scene.world = w
  w.use_nodes = True
  w.node_tree.nodes["Background"].inputs["Color"].default_value = L.hexrgb("#0B0D11")
  w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
  bpy.ops.mesh.primitive_plane_add(size=600)  # far edge sinks into darkness: no studio horizon line
  floor = bpy.context.active_object
  m, nt, b = L._principled("cyc", "#101318", 0.6)
  # Cyclorama look: the floor dissolves into the backdrop colour with distance.
  out = nt.nodes["Material Output"]
  nt.links.new(L._haze_mix(nt, b.outputs[0], "#0B0D11", 3.5, 9.0), out.inputs["Surface"])
  for nd in nt.nodes:
    if nd.type == "MAP_RANGE":
      nd.inputs["To Max"].default_value = 1.0
  floor.data.materials.append(m)
  for name, loc, energy, color, size in (("Key", (2.5, -2.0, 3.0), 650, "#FFE6C8", 2.0), ("Rim", (-3.0, 2.5, 2.2), 520, "#7FD8FF", 1.0), ("Fill", (-2.5, -3.5, 1.2), 160, "#FFB98A", 3.0)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.size = energy, size
    ld.color = L.hexrgb(color)[:3]
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (Vector((0, 0, 0.3)) - Vector(loc)).to_track_quat("-Z", "Y")
    bpy.context.scene.collection.objects.link(ob)
  bpy.context.scene.view_settings.view_transform = "AgX"


def sheet():
  """Four-view livery sheet: the droid turns on a turntable under fixed studio lights."""
  objs, _, r, mats = robot_at("standing_commands", 0.5, 0.0)
  studio(mats)
  table = bpy.data.objects.new("Turntable", None)
  table.location = (r[0], r[1], 0)
  bpy.context.scene.collection.objects.link(table)
  bpy.context.view_layer.update()
  for ob in objs:
    ob.parent = table
    ob.matrix_parent_inverse = table.matrix_world.inverted()
  cam = L.camera(lens=85, fstop=8.0)
  return [(az, r) for az in (0, 45, 90, 180)], cam, table


def blueprint():
  """Line-art technical readout: Freestyle outlines over flat blueprint blue."""
  objs, _, r, mats = robot_at("standing_commands", 0.5, 0.0)
  w = bpy.data.worlds.new("BP")
  bpy.context.scene.world = w
  w.use_nodes = True
  w.node_tree.nodes["Background"].inputs["Color"].default_value = L.hexrgb("#0B2A4A")
  flat = bpy.data.materials.new("bp")
  flat.use_nodes = True
  nt = flat.node_tree
  nt.nodes.clear()
  em = nt.nodes.new("ShaderNodeEmission")
  em.inputs["Color"].default_value = L.hexrgb("#0F3560")
  o = nt.nodes.new("ShaderNodeOutputMaterial")
  nt.links.new(em.outputs[0], o.inputs[0])
  for ob in objs:
    ob.data.materials.clear()
    ob.data.materials.append(flat)
  sc = bpy.context.scene
  sc.render.use_freestyle = True
  sc.render.line_thickness_mode = "ABSOLUTE"
  sc.render.line_thickness = 1.4
  vl = sc.view_layers[0]
  vl.use_freestyle = True
  ls = vl.freestyle_settings.linesets[0] if vl.freestyle_settings.linesets else vl.freestyle_settings.linesets.new("L")
  if ls.linestyle is None:
    ls.linestyle = bpy.data.linestyles.new("bp")
  ls.select_by_visibility = True
  ls.select_silhouette = ls.select_border = ls.select_crease = True
  ls.linestyle.color = L.hexrgb("#DDEEFF")[:3]
  ls.linestyle.thickness = 1.4
  vl.freestyle_settings.crease_angle = math.radians(140)
  cam = L.camera(lens=50, fstop=22)
  cam.data.type = "ORTHO"
  cam.data.ortho_scale = 0.95
  cam.data.dof.use_dof = False
  return [(0, r), (90, r)], cam


if __name__ == "__main__":
  argv = sys.argv[sys.argv.index("--") + 1 :]
  name = argv[0]
  preview = "--preview" in argv
  if name in ("keyart", "keyart_night"):
    w, h = keyart(name.endswith("night"))
    sc = bpy.context.scene
    L.render_settings(w // (2 if preview else 1), h // (2 if preview else 1), samples=24 if preview else 128, motion_blur=False)
    sc.render.filepath = str(OUT / f"{name}.png")
    bpy.ops.render.render(write_still=True)
  else:
    res = {"sheet": sheet, "blueprint": blueprint}[name]()
    views, cam, table = res if len(res) == 3 else (*res, None)
    sc = bpy.context.scene
    L.render_settings(900, 1200, samples=16 if preview else 96, motion_blur=False)
    if name == "blueprint":
      sc.view_settings.view_transform = "Standard"
      sc.render.resolution_x, sc.render.resolution_y = 1100, 1100
    for az, r in views:
      a = math.radians(az)
      c = r + np.array([0, 0, 0.12])
      d = 3.2 if name == "sheet" else 6.0
      if table is not None:  # rotate the droid, keep camera + lights fixed
        table.rotation_euler = (0, 0, -a)
        bpy.context.view_layer.update()
        a = math.radians(20)
      look(cam, c + d * np.array([math.cos(a), math.sin(a), 0.08 if name == "sheet" else 0.0]), c)
      sc.render.filepath = str(OUT / f"{name}_{az}.png")
      bpy.ops.render.render(write_still=True)
