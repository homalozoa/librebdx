"""BDX: A NEW GAIT, the showcase film, rendered shot by shot.

blender -b -P blender/film.py -- SHOT [--preview] [--frames a:b]
Frames go to results/bdx/film/<shot>/####.png; scripts/film_cut.sh assembles.
All robot motion is the recorded MuJoCo rollout of the trained policies.
"""

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
import bdx_lib as L  # noqa: E402

FPS = 24
OUT = L.ROOT / "results" / "bdx" / "film"


def ease(x):
  x = np.clip(x, 0, 1)
  return x * x * (3 - 2 * x)


def lerp(a, b, s):
  return np.asarray(a) * (1 - s[:, None]) + np.asarray(b) * s[:, None]


def setup(clip_name, night=False, speed=1.0, t0=0.0, t1=None, at=(0, 0), heading=0.0):
  L.reset_scene()
  L.desert_world(night)
  mats = L.make_materials()
  objs, meshes = L.build_robot(mats)
  clip = L.load_clip(clip_name, FPS, speed, t0, t1)
  off, yaw = L.placement(clip, at, heading, 0)
  frames, root = L.animate(objs, clip, offset=off, yaw=yaw)
  sc = bpy.context.scene
  sc.frame_start, sc.frame_end = 1, clip["n"]
  return clip, objs, meshes, frames, root, mats, (off, yaw)


def eye_pulse(mats, values, frames, base=6.0):
  """Eye emission follows the animation engine's eye channel (or a soft idle pulse)."""
  b = mats["eye"].node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
  for f, v in zip(frames[::2], values[::2]):
    b.default_value = float(base * v)
    b.keyframe_insert("default_value", frame=int(f))


# ------------------------------------------------------------------ shots
def shot_establish():
  """Twin suns, wide; slow push-in while the droid idles and nods."""
  # Backlit: camera opposite the suns (azimuth ~205 deg), droid facing camera.
  clip, objs, _, frames, root, mats, _ = setup("show_animation_engine", t0=0.0, t1=6.5, heading=math.radians(15))
  n = len(frames)
  s = ease(np.linspace(0, 1, n))
  r0 = root[0]
  u = np.array([math.cos(math.radians(15)), math.sin(math.radians(15)), 0.0])
  v = np.array([-u[1], u[0], 0.0])
  eye = lerp(r0 + 9.0 * u - 1.6 * v + [0, 0, 0.35], r0 + 3.6 * u - 0.7 * v + [0, 0, 0.12], s)
  tgt = lerp(r0 - 2.0 * v + [0, 0, 1.2], r0 - 0.35 * v + [0, 0, 0.3], s)
  cam = L.camera(lens=32, fstop=5.6)
  L.key_camera(cam, frames, eye, tgt)
  if "eye_radius" in clip:
    eye_pulse(mats, 0.6 + 0.6 * np.asarray(clip["eye_radius"], float), frames)


def shot_engine():
  """Animation engine: gaze sweep + posture, close and low."""
  clip, objs, _, frames, root, mats, _ = setup("show_animation_engine", t0=5.5, t1=16.5, heading=math.radians(180))
  n = len(frames)
  s = ease(np.linspace(0, 1, n))
  r0 = root[0]
  a = np.radians(-165 + 40 * s)  # stays in front of the face (droid heads toward -x)
  eye = r0 + np.stack([1.55 * np.cos(a), 1.55 * np.sin(a), 0.22 + 0.05 * s], -1)
  tgt = np.repeat([r0 + [0, 0, 0.3]], n, 0)
  cam = L.camera(lens=65, fstop=2.0)
  L.key_camera(cam, frames, eye, tgt)
  if "eye_radius" in clip:
    eye_pulse(mats, 0.6 + 0.6 * np.asarray(clip["eye_radius"], float), frames)


def shot_walk():
  """Tracking dolly alongside the walk: forward 0.7 m/s, strafes, turns."""
  clip, objs, _, frames, root, mats, _ = setup("walking_commands", t0=1.0, t1=16.5, heading=0.0)
  sm = L.smooth(root, 18)
  n = len(frames)
  s = np.linspace(0, 1, n)
  side = np.stack([np.zeros(n), -np.ones(n), np.zeros(n)], -1)
  back = np.stack([-np.ones(n), np.zeros(n), np.zeros(n)], -1)
  w = ease((s - 0.45) / 0.2)[:, None]
  off = (1 - w) * (side * 1.9 + back * 0.6) + w * (side * 1.2 + back * 2.0)
  eye = sm + off + np.array([0, 0, 0.1])
  tgt = sm + np.array([0.15, 0, 0.12])
  cam = L.camera(lens=45, fstop=2.8)
  L.key_camera(cam, frames, eye, tgt)


def shot_push():
  """Force pushes (90 N, 0.1 s) while walking at 0.3 m/s; a ripple marks each hit."""
  clip, objs, _, frames, root, mats, (off, yaw) = setup("walking_push", t0=2.2, t1=10.6, heading=0.0)
  sm = L.smooth(root, 14)
  n = len(frames)
  eye = sm + np.array([0.9, -2.3, 0.35])
  tgt = sm + np.array([0.2, 0, 0.1])
  cam = L.camera(lens=50, fstop=3.2)
  L.key_camera(cam, frames, eye, tgt)
  F = np.asarray(clip["force"], float)
  c, s_ = math.cos(yaw), math.sin(yaw)
  hits = [i for i in range(1, n) if np.linalg.norm(F[i]) > 1 and np.linalg.norm(F[i - 1]) < 1]
  for k, i in enumerate(hits):
    d = np.array([c * F[i, 0] - s_ * F[i, 1], s_ * F[i, 0] + c * F[i, 1], 0.0])
    d /= np.linalg.norm(d)
    ripple(root[i] - d * 0.16, d, frames[i], k)


def ripple(p, d, f0, k):
  """Expanding, fading ring of light where the Force push lands."""
  bpy.ops.mesh.primitive_torus_add(major_radius=0.1, minor_radius=0.012, location=tuple(p))
  ob = bpy.context.active_object
  ob.name = f"Ripple{k}"
  ob.rotation_mode = "QUATERNION"
  ob.rotation_quaternion = Vector(d).to_track_quat("Z", "Y")
  m = bpy.data.materials.new(f"ripple{k}")
  m.use_nodes = True
  nt = m.node_tree
  nt.nodes.clear()
  em = nt.nodes.new("ShaderNodeEmission")
  em.inputs["Color"].default_value = L.hexrgb("#9FE8FF")
  tr = nt.nodes.new("ShaderNodeBsdfTransparent")
  mix = nt.nodes.new("ShaderNodeMixShader")
  o = nt.nodes.new("ShaderNodeOutputMaterial")
  nt.links.new(tr.outputs[0], mix.inputs[1])
  nt.links.new(em.outputs[0], mix.inputs[2])
  nt.links.new(mix.outputs[0], o.inputs[0])
  ob.data.materials.append(m)
  ob.visible_shadow = False
  for df, sc, a, e in ((-1, 0.01, 0.0, 0.0), (0, 0.4, 1.0, 30.0), (4, 2.2, 0.6, 18.0), (10, 3.6, 0.0, 0.0)):
    ob.scale = (sc, sc, sc)
    ob.keyframe_insert("scale", frame=int(f0 + df))
    mix.inputs["Fac"].default_value = a
    mix.inputs["Fac"].keyframe_insert("default_value", frame=int(f0 + df))
    em.inputs["Strength"].default_value = e
    em.inputs["Strength"].keyframe_insert("default_value", frame=int(f0 + df))


def _night_dance(clip_name, az0, az1, lens=50, dist=1.9, h=0.25):
  clip, objs, _, frames, root, mats, _ = setup(clip_name, night=True, heading=math.radians(180))
  n = len(frames)
  s = ease(np.linspace(0, 1, n))
  r0 = root[0]
  a = np.radians(az0 + (az1 - az0) * s)
  eye = r0 + np.stack([dist * np.cos(a), dist * np.sin(a), h + 0 * s], -1)
  tgt = L.smooth(root, 10) * [0.5, 0.5, 0.0] + np.array([r0[0] * 0.5, r0[1] * 0.5, 0.42])
  cam = L.camera(lens=lens, fstop=4.0)
  L.key_camera(cam, frames, eye, tgt)
  bpy.context.scene.objects["Rim"].data.energy = 1.4


def shot_dance():
  _night_dance("episodic_happy_dance", 160, 200, lens=35, dist=1.9, h=0.08)


def shot_excited():
  _night_dance("episodic_excited", 205, 165, lens=32, dist=1.8, h=0.05)


def shot_tantrum():
  _night_dance("episodic_tantrum", 150, 185, lens=40, dist=2.1, h=0.12)


def shot_jump():
  """Ballistic jump in 0.4x slow motion, low angle into the suns."""
  clip, objs, _, frames, root, mats, _ = setup("episodic_jump", speed=0.4, heading=math.radians(30))
  n = len(frames)
  s = ease(np.linspace(0, 1, n))
  r0 = root[0]
  a = np.radians(35 + 25 * s)
  eye = r0 + np.stack([1.7 * np.cos(a), 1.7 * np.sin(a), -0.05 + 0.02 * s], -1)
  tgt = np.repeat([r0 + [0, 0, 0.2]], n, 0)
  cam = L.camera(lens=35, fstop=2.0)
  L.key_camera(cam, frames, eye, tgt)


def shot_transitions():
  """Stand -> walk -> stand -> dance -> stand in one continuous take, crane up."""
  clip, objs, _, frames, root, mats, _ = setup("transitions", heading=math.radians(0))
  sm = L.smooth(root, 24)
  n = len(frames)
  s = ease(np.linspace(0, 1, n))
  eye = sm + np.stack([0.4 + 1.2 * s, -2.6 + 0.4 * s, 0.3 + 1.3 * s], -1)
  tgt = sm + np.array([0.2, 0, 0.1])
  cam = L.camera(lens=40, fstop=4.0)
  L.key_camera(cam, frames, eye, tgt)


SHOTS = {
  "establish": shot_establish, "engine": shot_engine, "walk": shot_walk, "push": shot_push,
  "dance": shot_dance, "excited": shot_excited, "jump": shot_jump, "tantrum": shot_tantrum,
  "transitions": shot_transitions,
}

if __name__ == "__main__":
  argv = sys.argv[sys.argv.index("--") + 1 :]
  name = argv[0]
  SHOTS[name]()
  preview = "--preview" in argv
  if preview:
    L.render_settings(960, 540, FPS, samples=16, motion_blur=False)
  else:
    L.render_settings(1920, 1080, FPS, samples=48)
  sc = bpy.context.scene
  if "--frames" in argv:
    a, b = argv[argv.index("--frames") + 1].split(":")
    sc.frame_start, sc.frame_end = int(a), int(b)
  if "--step" in argv:
    sc.frame_step = int(argv[argv.index("--step") + 1])
  sc.render.filepath = str(OUT / (name + ("_preview" if preview else ""))) + "/"
  bpy.ops.render.render(animation=True)
