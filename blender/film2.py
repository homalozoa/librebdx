"""BDX: A New Gait, second cut. The story shots.

A scout droid wakes under two suns, hears a faint signal, crosses the dunes to a
relay tower, weathers the wind, leaps for joy, and arrives. Night falls; no one
answers, until a second pair of eyes lights up on a far dune.

blender -b -P blender/film2.py -- SHOT [--preview] [--frames a:b]
Every pose of every droid is a recorded rollout of a trained policy.
"""

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
import bdx_lib as L  # noqa: E402
from film import ease, lerp, ripple  # noqa: E402
from traces import traces_all  # noqa: E402

FPS = 24
OUT = L.ROOT / "results" / "bdx" / "film2"
RED = "#FF3B2F"


def droid(clip_name, at=(0, 0), heading=0.0, speed=1.0, t0=0.0, t1=None, frame0=1, mats=None, name="BDX"):
  mats = mats if mats is not None else L.make_materials()
  objs, meshes = L.build_robot(mats, name)
  clip = L.load_clip(clip_name, FPS, speed, t0, t1)
  off, yaw = L.placement(clip, at, heading, 0)
  frames, root = L.animate(objs, clip, frame0=frame0, offset=off, yaw=yaw)
  eyes = L.add_eyes(objs, meshes, mats, frame0)
  return {"objs": objs, "meshes": meshes, "clip": clip, "frames": frames, "root": root, "eyes": eyes, "mats": mats, "yaw": yaw, "speed": speed}


def world(night=False):
  L.reset_scene()
  L.desert_world(night)


def finish(d, n_frames, others=()):
  night = "Planet" in bpy.context.scene.objects
  traces_all([d, *others], night=night, wind=(0.12, 0.05) if night else (0.35, 0.12))
  L.bevel_all(d["mats"])
  sc = bpy.context.scene
  sc.frame_start, sc.frame_end = 1, n_frames


def heading_of(root, i, k=12):
  j = min(len(root) - 1, i + k)
  i0 = max(0, j - 2 * k)
  v = root[j] - root[i0]
  return math.atan2(v[1], v[0]) if np.hypot(v[0], v[1]) > 0.05 else None


def relay(loc, frames=None, period=2.0, on=0.35, h=5.2):
  """The signal: a relay mast (h metres) with a small dish and a blinking red lamp."""
  mat = bpy.data.materials.new("relay_metal")
  mat.use_nodes = True
  b = mat.node_tree.nodes["Principled BSDF"]
  b.inputs["Base Color"].default_value = L.hexrgb("#7E7468")
  b.inputs["Metallic"].default_value = 0.7
  b.inputs["Roughness"].default_value = 0.5
  parts = L._vaporator(loc, h, mat)
  bpy.ops.mesh.primitive_uv_sphere_add(radius=0.55 * h / 5.2, location=(loc[0] + 0.35 * h / 5.2, loc[1], 0.83 * h))
  dish = bpy.context.active_object
  dish.scale = (0.35, 1.0, 1.0)
  dish.data.materials.append(mat)
  lamp_m = bpy.data.materials.new("relay_lamp")
  lamp_m.use_nodes = True
  nt = lamp_m.node_tree
  nt.nodes.clear()
  em = nt.nodes.new("ShaderNodeEmission")
  em.inputs["Color"].default_value = L.hexrgb(RED)
  o = nt.nodes.new("ShaderNodeOutputMaterial")
  nt.links.new(em.outputs[0], o.inputs[0])
  bpy.ops.mesh.primitive_uv_sphere_add(radius=0.09, location=(loc[0], loc[1], h + 0.22))
  lamp = bpy.context.active_object
  lamp.data.materials.append(lamp_m)
  light = bpy.data.lights.new("relay_glow", "POINT")
  light.color = L.hexrgb(RED)[:3]
  light.shadow_soft_size = 0.1
  lo = bpy.data.objects.new("relay_glow", light)
  lo.location = (loc[0], loc[1], h + 0.22)
  bpy.context.scene.collection.objects.link(lo)
  if frames is not None:  # blink: short on, long off, like a beacon
    for f in frames:
      ph = ((f - 1) / FPS) % period
      v = 1.0 if ph < on else 0.0
      em.inputs["Strength"].default_value = 60.0 * v + 0.2
      em.inputs["Strength"].keyframe_insert("default_value", frame=int(f))
      light.energy = 40.0 * v
      light.keyframe_insert("energy", frame=int(f))
    for fc in _fcurves(nt) + _fcurves_light(light):
      for kp in fc.keyframe_points:
        kp.interpolation = "CONSTANT"
  return parts + [dish, lamp, lo]


def _fcurves(nt):
  ad = nt.animation_data
  if not ad or not ad.action:
    return []
  act = ad.action
  if hasattr(act, "layers") and act.layers:
    return list(act.layers[0].strips[0].channelbag(ad.action_slot).fcurves)
  return list(act.fcurves)


def _fcurves_light(light):
  ad = light.animation_data
  if not ad or not ad.action:
    return []
  act = ad.action
  if hasattr(act, "layers") and act.layers:
    return list(act.layers[0].strips[0].channelbag(ad.action_slot).fcurves)
  return list(act.fcurves)


def streaks(root, gusts, n=260, seed=4):
  """Wind-blown sand: thin grains racing past during each gust (motion blur draws the streak)."""
  rng = np.random.default_rng(seed)
  m = bpy.data.materials.new("sand_streak")
  m.use_nodes = True
  b = m.node_tree.nodes["Principled BSDF"]
  b.inputs["Base Color"].default_value = L.hexrgb("#D9B48A")
  b.inputs["Roughness"].default_value = 0.9
  b.inputs["Alpha"].default_value = 0.55
  for f0, f1, d in gusts:
    d = np.array([d[0], d[1], 0.0])
    d /= np.linalg.norm(d)
    side = np.array([-d[1], d[0], 0.0])
    c = root[min(f0, len(root) - 1)]
    for k in range(n // len(gusts)):
      bpy.ops.mesh.primitive_cylinder_add(vertices=5, radius=0.0012, depth=0.05)
      g = bpy.context.active_object
      g.data.materials.append(m)
      g.rotation_euler = (0, math.pi / 2, math.atan2(d[1], d[0]))
      start = c - d * rng.uniform(3, 7) + side * rng.uniform(-2.5, 2.5) + np.array([0, 0, rng.uniform(0.01, 0.6) ** 1.5])
      speed = rng.uniform(5, 9) / FPS
      fs = f0 + rng.integers(0, 4)
      fe = f1 + rng.integers(0, 6)
      for f, vis in ((fs - 1, 0.0), (fs, 1.0), (fe, 1.0), (fe + 1, 0.0)):
        p = start + d * speed * (f - fs)
        g.location = tuple(p)
        g.scale = (vis,) * 3
        g.keyframe_insert("location", frame=int(f))
        g.keyframe_insert("scale", frame=int(f))


# ------------------------------------------------------------------ shots
def shot_waking():
  """I. Waking: bowed and dark, eyes flicker on, it rises; the camera pulls back to two suns."""
  world()
  h = math.radians(25)
  d = droid("story_wake", heading=h, t0=2.0, t1=12.0)
  n = len(d["frames"])
  s = ease(np.linspace(0, 1, n))
  r0 = d["root"][0]
  u = np.array([math.cos(h), math.sin(h), 0.0])
  v = np.array([-u[1], u[0], 0.0])
  eye = lerp(r0 + 1.25 * u - 0.25 * v + [0, 0, 0.16], r0 + 3.3 * u - 0.9 * v + [0, 0, 0.28], s)
  tgt = lerp(r0 + [0, 0, 0.2], r0 - 0.65 * v + [0, 0, 0.55], s)
  cam = L.camera(lens=50, fstop=2.8)
  L.key_camera(cam, d["frames"], eye, tgt)
  L.animate_eyes(d["eyes"], d["frames"], blinks=(int(5.2 * FPS), int(8.4 * FPS)), power_on=(int(1.1 * FPS), int(2.3 * FPS)))
  finish(d, n)


def shot_listening():
  """II. Listening: the head turns; far behind it the relay blinks red."""
  world()
  d = droid("show_animation_engine", heading=math.radians(180), t0=5.5, t1=14.0)
  n = len(d["frames"])
  s = ease(np.linspace(0, 1, n))
  r0 = d["root"][0]
  a = np.radians(-165 + 40 * s)
  eye = r0 + np.stack([1.55 * np.cos(a), 1.55 * np.sin(a), 0.22 + 0.05 * s], -1)
  tgt = np.repeat([r0 + [0, 0, 0.3]], n, 0)
  relay((r0[0] + 34 * math.cos(math.radians(30)), r0[1] + 34 * math.sin(math.radians(30))), d["frames"])
  cam = L.camera(lens=65, fstop=2.4)
  L.key_camera(cam, d["frames"], eye, tgt)
  glow = 0.75 + 0.25 * np.asarray(d["clip"].get("eye_radius", np.ones(n)), float) if "eye_radius" in d["clip"] else None
  L.animate_eyes(d["eyes"], d["frames"], glow=glow, blinks=(int(2.0 * FPS), int(6.1 * FPS)))
  finish(d, n)


def shot_crossing():
  """III. Crossing: from behind (the relay ahead), then alongside as it walks."""
  world()
  d = droid("story_crossing2", heading=0.0, t0=2.0, t1=17.0)
  n = len(d["frames"])
  root = d["root"]
  sm = L.smooth(root, 20)
  hd = np.array([heading_of(sm, i) or 0.0 for i in range(n)])
  hd = np.unwrap(hd)
  hd = np.convolve(np.pad(hd, 24, mode="edge"), np.ones(49) / 49, mode="valid")
  fwd = np.stack([np.cos(hd), np.sin(hd), 0 * hd], -1)
  left = np.stack([-np.sin(hd), np.cos(hd), 0 * hd], -1)
  end_h = hd[-1]
  relay((root[-1][0] + 24 * math.cos(end_h), root[-1][1] + 24 * math.sin(end_h)), d["frames"])
  w = ease((np.linspace(0, 1, n) - 0.38) / 0.3)[:, None]
  # Swing around the droid on an arc (never through it): from behind-left to its right side.
  a0, a1 = math.atan2(0.55, -2.2), math.atan2(-1.9, 0.6)
  ang = a0 + ((a1 - a0 + math.pi) % (2 * math.pi) - math.pi) * w[:, 0]
  rad = (2.27 * (1 - w) + 1.99 * w)[:, 0]
  off = (np.cos(ang) * rad)[:, None] * fwd + (np.sin(ang) * rad)[:, None] * left
  eye = sm + off + np.array([0, 0, 0.32]) * (1 - w) + np.array([0, 0, 0.12]) * w
  tgt = sm + 1.1 * fwd * (1 - w) + 0.15 * fwd * w + np.array([0, 0, 0.18])
  cam = L.camera(lens=40, fstop=3.5)
  L.key_camera(cam, d["frames"], eye, tgt)
  L.animate_eyes(d["eyes"], d["frames"], blinks=(int(4.0 * FPS), int(11.5 * FPS)))
  finish(d, n)


def shot_wind():
  """IV. Wind: three gusts of blown sand; it staggers and keeps walking."""
  world()
  d = droid("story_wind2", heading=0.0, t0=2.2, t1=10.6)
  n = len(d["frames"])
  root = d["root"]
  sm = L.smooth(root, 14)
  eye = sm + np.array([0.9, -2.3, 0.35])
  tgt = sm + np.array([0.2, 0, 0.1])
  cam = L.camera(lens=50, fstop=3.2)
  L.key_camera(cam, d["frames"], eye, tgt)
  F = np.asarray(d["clip"]["force"], float)
  c, s_ = math.cos(d["yaw"]), math.sin(d["yaw"])
  hits = [i for i in range(1, n) if np.linalg.norm(F[i]) > 1 and np.linalg.norm(F[i - 1]) < 1]
  gusts = []
  for i in hits:
    dd = np.array([c * F[i, 0] - s_ * F[i, 1], s_ * F[i, 0] + c * F[i, 1]])
    gusts.append((int(d["frames"][max(0, i - 10)]), int(d["frames"][min(n - 1, i + 8)]), dd))
  streaks(root, gusts)
  relay((root[-1][0] + 30, root[-1][1] + 6), d["frames"])
  L.animate_eyes(d["eyes"], d["frames"], blinks=tuple(int(d["frames"][i]) + 2 for i in hits))
  finish(d, n)


def shot_leap():
  """V. Leap: one small flight in 0.4x, the two suns and the relay behind it."""
  world()
  d = droid("episodic_jump", speed=0.4, heading=math.radians(30))
  n = len(d["frames"])
  s = ease(np.linspace(0, 1, n))
  r0 = d["root"][0]
  a = np.radians(35 + 25 * s)
  eye = r0 + np.stack([1.7 * np.cos(a), 1.7 * np.sin(a), -0.05 + 0.02 * s], -1)
  tgt = np.repeat([r0 + [0, 0, 0.2]], n, 0)
  relay((r0[0] + 14 * math.cos(math.radians(228)), r0[1] + 14 * math.sin(math.radians(228))), d["frames"])
  cam = L.camera(lens=35, fstop=2.2)
  L.key_camera(cam, d["frames"], eye, tgt)
  L.animate_eyes(d["eyes"], d["frames"], blinks=(int(1.0 * FPS),))
  finish(d, n)


def shot_arrival():
  """VI. Arrival: walk, stop beside the relay at double support, and dance."""
  world()
  d = droid("story_arrival", heading=0.0)
  n = len(d["frames"])
  root = d["root"]
  sm = L.smooth(root, 24)
  stop = root[int(6.0 * FPS)]
  s = ease(np.linspace(0, 1, n))
  eye = sm + np.stack([0.4 + 1.0 * s, -2.7 + 0.3 * s, 0.32 + 0.38 * s], -1)
  # The relay stands on the camera's line of sight, 22 m beyond where the droid stops:
  # far enough that the whole mast and its blinking lamp stay in frame through the take.
  k = int(6.0 * FPS)
  v = (stop - eye[k]) * [1, 1, 0]
  v /= np.linalg.norm(v)
  relay((stop[0] + 22.0 * v[0] - 3.0 * v[1], stop[1] + 22.0 * v[1] + 3.0 * v[0]), d["frames"])
  tgt = sm + np.array([0.2, 0, 0.22]) + np.array([0, 0, 0.3]) * s[:, None]
  cam = L.camera(lens=32, fstop=4.0)
  L.key_camera(cam, d["frames"], eye, tgt)
  L.animate_eyes(d["eyes"], d["frames"], blinks=(int(6.5 * FPS), int(13.0 * FPS)))
  finish(d, n)


def _night(clip_name, az0, az1, lens, dist, h, blinks=(), relay_at=(-1.4, 1.6)):
  world(night=True)
  d = droid(clip_name, heading=math.radians(180))
  n = len(d["frames"])
  s = ease(np.linspace(0, 1, n))
  r0 = d["root"][0]
  a = np.radians(az0 + (az1 - az0) * s)
  eye = r0 + np.stack([dist * np.cos(a), dist * np.sin(a), h + 0 * s], -1)
  tgt = L.smooth(d["root"], 10) * [0.5, 0.5, 0.0] + np.array([r0[0] * 0.5, r0[1] * 0.5, 0.42])
  relay((r0[0] + relay_at[0], r0[1] + relay_at[1]), d["frames"])
  cam = L.camera(lens=lens, fstop=4.0)
  L.key_camera(cam, d["frames"], eye, tgt)
  bpy.context.scene.objects["Rim"].data.energy = 1.4
  L.animate_eyes(d["eyes"], d["frames"], blinks=blinks)
  return d, n


def shot_nightfall():
  """VII. Nightfall: the relay blinks, nothing answers, and patience runs out."""
  d, n = _night("episodic_tantrum", 150, 185, 40, 2.1, 0.12, blinks=(8,))
  finish(d, n)


def shot_answer():
  """VIII. The answer: over its shoulder, two small lights wake on a far dune and nod."""
  world(night=True)
  mats = L.make_materials()
  far = droid("story_answer", at=(13.0, 2.0), heading=math.radians(190), t0=2.0, t1=9.0, mats=mats, name="BDX2")
  near = droid("story_answer", at=(0.0, 0.0), heading=math.radians(10), t0=0.2, t1=7.2, mats=mats, name="BDX")
  n = len(near["frames"])
  relay((-1.3, -1.5), near["frames"])
  r0, rf = near["root"][0], far["root"][0]
  s = ease(np.linspace(0, 1, n))
  dvec = (rf - r0) * [1, 1, 0]
  dvec /= np.linalg.norm(dvec)
  perp = np.array([-dvec[1], dvec[0], 0.0])
  # Just behind the near droid's right shoulder, its head soft in the foreground.
  eye = lerp(r0 - 0.82 * dvec + 0.22 * perp + [0, 0, 0.40], r0 - 0.74 * dvec + 0.2 * perp + [0, 0, 0.40], s)
  tgt = np.repeat([rf + [0, 0, 0.22]], n, 0)
  cam = L.camera(lens=50, fstop=2.8)
  cam.data.dof.focus_distance = float(np.linalg.norm(tgt[0] - eye[0]))
  L.key_camera(cam, near["frames"], eye, tgt, focus=True)
  bpy.context.scene.objects["Rim"].data.energy = 1.2
  L.animate_eyes(near["eyes"], near["frames"])
  L.animate_eyes(far["eyes"], far["frames"], power_on=(int(1.6 * FPS), int(2.4 * FPS)), blinks=(int(5.6 * FPS),))
  finish(near, n, (far,))


def shot_excited():
  """IX. Joy: it is not alone."""
  d, n = _night("episodic_excited", 205, 165, 32, 1.8, 0.05, blinks=(30,))
  finish(d, n)


def shot_reunion():
  """IX. Reunion: it hurries 3.4 m across the sand to the other droid, stops, and cannot keep still."""
  world(night=True)
  mats = L.make_materials()
  near = droid("story_reunion", at=(0.0, 0.0), heading=0.0, t0=2.0, t1=14.6, mats=mats, name="BDX")
  far = droid("story_partner", at=(4.3, 0.55), heading=math.radians(195), t0=2.0, t1=14.6, mats=mats, name="BDX2")
  n = len(near["frames"])
  relay((-7.0, 10.0), near["frames"])
  pf = far["root"][0]
  s = ease(np.linspace(0, 1, n))
  # Side-on, wide enough to hold both droids from the first step to the dance.
  sm = L.smooth(near["root"], 20)
  gap = np.linalg.norm((pf - sm)[:, :2], axis=1)
  w = np.clip((3.6 - gap) / 2.6, 0, 1)[:, None]  # 0 while far apart, 1 once side by side
  look = (1 - w) * (0.62 * sm + 0.38 * pf) + w * (0.5 * sm + 0.5 * pf)  # partner stays in frame ahead
  look = L.smooth(look, 10)
  tgt = np.stack([look[:, 0], look[:, 1] + 0.2, 0.3 + 0 * s], -1)
  eye = tgt + np.stack([-0.35 + 0 * s, -3.6 + 0.6 * s, -0.02 + 0 * s], -1)
  cam = L.camera(lens=30, fstop=5.6)
  L.key_camera(cam, near["frames"], eye, tgt)
  bpy.context.scene.objects["Rim"].data.energy = 1.4
  L.animate_eyes(near["eyes"], near["frames"], blinks=(int(5.5 * FPS),))
  L.animate_eyes(far["eyes"], far["frames"], blinks=(int(2.2 * FPS), int(7.8 * FPS)))
  finish(near, n, (far,))


def shot_together():
  """X. Together: both look up at the ringed planet; the camera rises."""
  world(night=True)
  mats = L.make_materials()
  h = math.radians(350)
  a = droid("story_gaze_up", at=(0, 0), heading=h, t0=2.0, t1=9.0, mats=mats, name="BDX")
  b = droid("story_gaze_up", at=(2.6, -1.9), heading=h + 0.12, t0=1.6, t1=8.6, mats=mats, name="BDX2")
  n = len(a["frames"])
  relay((-2.2, 2.4), a["frames"])
  s = ease(np.linspace(0, 1, n))
  mid = (a["root"][0] + b["root"][0]) / 2
  u = np.array([math.cos(h), math.sin(h), 0.0])
  eye = lerp(mid - 4.0 * u + [0, 0, 0.3], mid - 5.6 * u + [0, 0, 0.95], s)
  tgt = lerp(mid + 6 * u + [0, 0, 0.9], mid + 10 * u + [0, 0, 1.25], s)
  cam = L.camera(lens=28, fstop=5.6)
  L.key_camera(cam, a["frames"], eye, tgt)
  bpy.context.scene.objects["Rim"].data.energy = 1.5
  L.animate_eyes(a["eyes"], a["frames"], blinks=(int(4.6 * FPS),))
  L.animate_eyes(b["eyes"], b["frames"], blinks=(int(3.1 * FPS),))
  finish(a, n, (b,))


SHOTS = {
  "waking": shot_waking, "listening": shot_listening, "crossing": shot_crossing, "wind": shot_wind,
  "leap": shot_leap, "arrival": shot_arrival, "nightfall": shot_nightfall, "answer": shot_answer,
  "excited": shot_excited, "reunion": shot_reunion, "together": shot_together,
}

if __name__ == "__main__":
  argv = sys.argv[sys.argv.index("--") + 1 :]
  name = argv[0]
  SHOTS[name]()
  preview = "--preview" in argv
  if preview:
    L.render_settings(960, 540, FPS, samples=24, motion_blur=False)
  else:
    L.render_settings(1920, 1080, FPS, samples=96)
  sc = bpy.context.scene
  if "--frames" in argv:
    a, b = argv[argv.index("--frames") + 1].split(":")
    sc.frame_start, sc.frame_end = int(a), int(b)
  if "--step" in argv:
    sc.frame_step = int(argv[argv.index("--step") + 1])
  sc.render.use_overwrite = False  # resume: frames already on disk are skipped
  sc.render.use_placeholder = True
  sc.render.filepath = str(OUT / (name + ("_preview" if preview else ""))) + "/"
  bpy.ops.render.render(animation=True)
