"""Footprints and sand dust from the simulated foot touchdowns.

traces_all([d, ...]) takes droid dicts from film2.droid() (objects already animated):

Footprints. Every touchdown of either foot pad becomes a print shaped from the pad's real
outline (its xy shadow at touchdown). Prints live on a 2.5 mm world lattice; each lattice
cell near the walk belongs to exactly one print (nearest outline), so patches tile the
corridor without overlapping. The dunes get a matching hole (a top-down mask in their
shader), so a print can sink below the old surface. Profile: floor pressed in by the
touchdown speed, walls at sand's angle of repose, a lumpy rim of displaced sand, a small
push-up at the toe, and fresh, ripple-free sand inside. Each patch lies flat (undisturbed)
until its touchdown frame.

Dust. Each touchdown throws sand grains (particles) and a soft volumetric puff that rises,
spreads, drifts downwind and fades; each lift-off lets a few grains trickle off the pad.
"""

import bpy
import numpy as np
from mathutils import Vector

import bdx_lib as L

CELL = 0.0025


def _hull(p):
  p = sorted(set(map(tuple, np.round(p, 5))))
  if len(p) < 3:
    return np.array(p)

  def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  lo, hi = [], []
  for q in p:
    while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
      lo.pop()
    lo.append(q)
  for q in reversed(p):
    while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0:
      hi.pop()
    hi.append(q)
  return np.array(lo[:-1] + hi[:-1])


def _sdist(xy, poly):
  """Signed distance (negative inside) from points to a convex polygon (CCW)."""
  d = np.full(len(xy), np.inf)
  inside = np.ones(len(xy), bool)
  n = len(poly)
  for i in range(n):
    a, b = poly[i], poly[(i + 1) % n]
    ab = b - a
    t = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
    d = np.minimum(d, np.linalg.norm(xy - (a + t[:, None] * ab), axis=1))
    inside &= (ab[0] * (xy[:, 1] - a[1]) - ab[1] * (xy[:, 0] - a[0])) >= 0
  return np.where(inside, -d, d)


def _vnoise(xy, scale, seed):
  """Smooth value noise in [0, 1] on a world lattice of the given cell size (m)."""
  g = xy / scale
  i = np.floor(g).astype(np.int64)
  f = g - i
  f = f * f * (3 - 2 * f)

  def h(a, b):
    n = (a * 374761393 + b * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    return (n & 0xFFFF) / 65535.0

  a, b = i[:, 0], i[:, 1]
  v00, v10, v01, v11 = h(a, b), h(a + 1, b), h(a, b + 1), h(a + 1, b + 1)
  return (v00 * (1 - f[:, 0]) + v10 * f[:, 0]) * (1 - f[:, 1]) + (v01 * (1 - f[:, 0]) + v11 * f[:, 0]) * f[:, 1]


def _ground_z(dunes, x, y):
  inv = dunes.matrix_world.inverted()
  o = inv @ Vector((x, y, 50.0))
  hit, loc, _, _ = dunes.ray_cast(o, (inv.to_3x3() @ Vector((0, 0, -1))).normalized())
  return (dunes.matrix_world @ loc).z if hit else 0.0


def _sand_mat():
  return next(m for m in bpy.data.materials if m.name.startswith("sand") and m.node_tree and
              any(n.type == "TEX_WAVE" for n in m.node_tree.nodes))


def _print_material(sand):
  """Sand material whose ripples fade inside fresh prints (per-vertex 'fresh' x object 'trodden')."""
  m = sand.copy()
  m.name = "sand_print"
  nt = m.node_tree
  bump = next(n for n in nt.nodes if n.type == "BUMP")
  a = nt.nodes.new("ShaderNodeAttribute")
  a.attribute_type = "GEOMETRY"
  a.attribute_name = "fresh"
  o = nt.nodes.new("ShaderNodeAttribute")
  o.attribute_type = "OBJECT"
  o.attribute_name = "trodden"
  mul = nt.nodes.new("ShaderNodeMath")
  mul.operation = "MULTIPLY"
  nt.links.new(a.outputs["Fac"], mul.inputs[0])
  nt.links.new(o.outputs["Fac"], mul.inputs[1])
  inv = nt.nodes.new("ShaderNodeMath")
  inv.operation = "SUBTRACT"
  inv.inputs[0].default_value = 1.0
  nt.links.new(mul.outputs[0], inv.inputs[1])
  k = nt.nodes.new("ShaderNodeMath")
  k.operation = "MULTIPLY"
  k.inputs[1].default_value = bump.inputs["Strength"].default_value
  nt.links.new(inv.outputs[0], k.inputs[0])
  nt.links.new(k.outputs[0], bump.inputs["Strength"])
  # Freshly turned sand is a touch darker and cooler (less sun-bleached, slightly damp from below).
  b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
  src = b.inputs["Base Color"].links[0].from_socket
  mix = nt.nodes.new("ShaderNodeMix")
  mix.data_type = "RGBA"
  mix.blend_type = "MULTIPLY"
  mix.inputs["B"].default_value = (0.86, 0.84, 0.82, 1)
  nt.links.new(mul.outputs[0], mix.inputs["Factor"])
  nt.links.new(src, mix.inputs["A"])
  nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
  return m


def _cut_dunes(sand, cells, i0, j0, ni, nj):
  """Make the dunes transparent over the print cells (eroded by one cell)."""
  img = bpy.data.images.new("trace_mask", ni, nj, alpha=False, float_buffer=False)
  px = np.zeros((nj, ni), np.float32)
  for (i, j) in cells:
    px[j - j0, i - i0] = 1.0
  er = px.copy()
  er[1:-1, 1:-1] = np.minimum.reduce([px[1:-1, 1:-1], px[:-2, 1:-1], px[2:, 1:-1], px[1:-1, :-2], px[1:-1, 2:]])
  er[[0, -1], :] = 0
  er[:, [0, -1]] = 0
  rgba = np.repeat(er.ravel()[:, None], 4, 1)
  rgba[:, 3] = 1
  img.pixels.foreach_set(rgba.astype(np.float32).ravel())
  img.colorspace_settings.name = "Non-Color"
  nt = sand.node_tree
  geo = nt.nodes.new("ShaderNodeNewGeometry")
  mp = nt.nodes.new("ShaderNodeMapping")
  mp.vector_type = "POINT"
  mp.inputs["Location"].default_value = (-i0 * CELL / (ni * CELL), -j0 * CELL / (nj * CELL), 0)
  mp.inputs["Scale"].default_value = (1 / (ni * CELL), 1 / (nj * CELL), 1)
  nt.links.new(geo.outputs["Position"], mp.inputs["Vector"])
  tex = nt.nodes.new("ShaderNodeTexImage")
  tex.image = img
  tex.interpolation = "Closest"
  tex.extension = "CLIP"
  nt.links.new(mp.outputs[0], tex.inputs["Vector"])
  tr = nt.nodes.new("ShaderNodeBsdfTransparent")
  mix = nt.nodes.new("ShaderNodeMixShader")
  out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
  surf = out.inputs["Surface"].links[0].from_socket
  nt.links.new(tex.outputs["Color"], mix.inputs["Fac"])
  nt.links.new(surf, mix.inputs[1])
  nt.links.new(tr.outputs[0], mix.inputs[2])
  nt.links.new(mix.outputs[0], out.inputs["Surface"])


def _grain_obj(night):
  if "sand_grain" in bpy.data.objects:
    return bpy.data.objects["sand_grain"]
  bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1.0, location=(0, 0, -50))
  g = bpy.context.active_object
  g.name = "sand_grain"
  m = bpy.data.materials.new("grain")
  m.use_nodes = True
  b = m.node_tree.nodes["Principled BSDF"]
  b.inputs["Base Color"].default_value = L.hexrgb("#6B5A54" if night else "#D2A574")
  b.inputs["Roughness"].default_value = 0.9
  g.data.materials.append(m)
  return g


def _puff_mat(night):
  m = bpy.data.materials.new("dust_puff")
  m.use_nodes = True
  nt = m.node_tree
  nt.nodes.clear()
  out = nt.nodes.new("ShaderNodeOutputMaterial")
  vol = nt.nodes.new("ShaderNodeVolumePrincipled")
  vol.inputs["Color"].default_value = L.hexrgb("#9A94A8" if night else "#F0D2AA")  # moonlit dust is pale
  vol.inputs["Anisotropy"].default_value = 0.45
  tc = nt.nodes.new("ShaderNodeTexCoord")
  noise = nt.nodes.new("ShaderNodeTexNoise")
  noise.inputs["Scale"].default_value = 2.5
  noise.inputs["Detail"].default_value = 6.0
  nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
  grad = nt.nodes.new("ShaderNodeTexGradient")
  grad.gradient_type = "SPHERICAL"
  nt.links.new(tc.outputs["Object"], grad.inputs["Vector"])
  cr = nt.nodes.new("ShaderNodeMapRange")
  cr.inputs["From Min"].default_value, cr.inputs["From Max"].default_value = 0.3, 0.7
  nt.links.new(noise.outputs["Fac"], cr.inputs["Value"])
  mul = nt.nodes.new("ShaderNodeMath")
  mul.operation = "MULTIPLY"
  nt.links.new(grad.outputs["Fac"], mul.inputs[0])
  nt.links.new(cr.outputs["Result"], mul.inputs[1])
  dens = nt.nodes.new("ShaderNodeMath")
  dens.name = "density"
  dens.operation = "MULTIPLY"
  nt.links.new(mul.outputs[0], dens.inputs[0])
  nt.links.new(dens.outputs[0], vol.inputs["Density"])
  nt.links.new(vol.outputs[0], out.inputs["Volume"])
  return m


def events(d, pad_names=("Left_Foot_Pad", "Right_Foot_Pad")):
  """Touchdowns and lift-offs: (frame, kind, pad object, speed m/s)."""
  out = []
  fps = bpy.context.scene.render.fps
  for name in pad_names:
    g = d["meshes"].index(name)
    z = d["clip"]["pos"][:, g, 2]
    ground = np.percentile(z, 5)
    down = z < ground + 0.006
    for i in np.where(down[1:] & ~down[:-1])[0] + 1:
      v = abs(z[i] - z[max(0, i - 2)]) * fps / 2 * d.get("speed", 1.0) ** -1
      out.append((int(d["frames"][i]), "down", d["objs"][g], float(v)))
    for i in np.where(~down[1:] & down[:-1])[0] + 1:
      out.append((int(d["frames"][i]), "up", d["objs"][g], 0.0))
  return sorted(out, key=lambda e: e[0])


def _burst(coll, k, c, gz, f, count, normal, rnd, life, grain, wind, size=0.0009, radius=0.05):
  bpy.ops.mesh.primitive_circle_add(vertices=24, radius=radius, fill_type="NGON", location=(c[0], c[1], gz + 0.004))
  em = bpy.context.active_object
  em.name = f"dust_emitter_{k}"
  for cc in em.users_collection:
    cc.objects.unlink(em)
  coll.objects.link(em)
  ps = em.modifiers.new("grains", "PARTICLE_SYSTEM").particle_system.settings
  ps.count = int(count)
  ps.frame_start, ps.frame_end = f, f + 2
  ps.lifetime, ps.lifetime_random = life, 0.5
  ps.normal_factor = normal
  ps.factor_random = rnd
  ps.object_align_factor = (wind[0], wind[1], 0.0)
  ps.render_type = "OBJECT"
  ps.instance_object = grain
  ps.particle_size, ps.size_random = size, 0.6
  em.show_instancer_for_render = False
  return em


def traces_all(droids, night=False, wind=(0.35, 0.12), max_puffs=80):
  sc = bpy.context.scene
  fps = sc.render.fps
  dunes = sc.objects["Dunes"]
  sand = _sand_mat()
  pm_sand = _print_material(sand)
  grain = _grain_obj(night)
  puff_m = _puff_mat(night)
  coll = bpy.data.collections.new("traces")
  sc.collection.children.link(coll)
  prints, lifts = [], []
  for d in droids:
    for (f, kind, pad, v) in events(d):
      sc.frame_set(min(f + 2, int(d["frames"][-1])) if kind == "down" else f)
      M = pad.matrix_world
      V = np.array([tuple(M @ vv.co) for vv in pad.data.vertices])
      if kind == "up":
        lifts.append((f, V[:, :2].mean(0), V[:, 2].min()))
        continue
      poly = _hull(V[:, :2])
      if len(poly) < 3:
        continue
      fwd = M.to_3x3() @ Vector((0, 0, 1))  # the pad's long axis
      fwd = np.array([fwd.x, fwd.y])
      fwd /= np.linalg.norm(fwd) + 1e-9
      c = poly.mean(0)
      prints.append({"f": f, "poly": poly, "c": c, "fwd": fwd, "v": v, "gz": _ground_z(dunes, *c)})
  # Toe direction: the pad axis that points along the droid's travel (toward its next print).
  for k, p in enumerate(prints):
    nxt = next((q for q in prints[k + 1 :] if np.linalg.norm(q["c"] - p["c"]) > 0.05), None)
    if nxt is not None and np.dot(nxt["c"] - p["c"], p["fwd"]) < 0:
      p["fwd"] = -p["fwd"]
  # ------------------------------------------------ lattice cells and ownership
  MARGIN = 0.03
  owner = {}
  for k, p in enumerate(prints):
    lo = np.floor((p["poly"].min(0) - MARGIN) / CELL).astype(int)
    hi = np.ceil((p["poly"].max(0) + MARGIN) / CELL).astype(int)
    I, J = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing="ij")
    ctr = np.stack([(I.ravel() + 0.5) * CELL, (J.ravel() + 0.5) * CELL], -1)
    sd = _sdist(ctr, p["poly"])
    for (i, j), s in zip(zip(I.ravel(), J.ravel()), sd):
      if s < MARGIN and (owner.get((i, j)) is None or s < owner[(i, j)][1]):
        owner[(i, j)] = (k, s)
  if not owner:
    return 0
  keys = np.array(list(owner.keys()))
  i0, j0 = keys.min(0) - 2
  ni, nj = keys.max(0) - keys.min(0) + 5
  _cut_dunes(sand, owner.keys(), i0, j0, ni, nj)
  rng = np.random.default_rng(7)
  for k, p in enumerate(prints):
    cells = [ij for ij, (o, _) in owner.items() if o == k]
    if not cells:
      continue
    vid, verts, faces = {}, [], []
    for (i, j) in cells:
      quad = []
      for (a, b) in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
        if (a, b) not in vid:
          vid[(a, b)] = len(verts)
          verts.append((a * CELL, b * CELL))
        quad.append(vid[(a, b)])
      faces.append(quad)
    xy = np.array(verts)
    sd = _sdist(xy, p["poly"])
    depth = float(np.clip(0.008 + 0.014 * p["v"], 0.008, 0.02))
    jag = 0.0035 * (_vnoise(xy, 0.012, k + 100) - 0.5) + 0.0006 * rng.standard_normal(len(xy))
    s = sd + jag
    wall = np.clip(-s / (depth / np.tan(np.radians(33))), 0, 1)       # angle-of-repose walls
    wall = wall * wall * (3 - 2 * wall)
    lump = 0.55 + 0.45 * _vnoise(xy, 0.014, k) + 0.25 * _vnoise(xy, 0.005, k + 50)
    rim = np.exp(-((s - 0.008) / 0.007) ** 2) * (s > -0.003) * lump
    along = (xy - p["c"]) @ p["fwd"]
    half = np.max((p["poly"] - p["c"]) @ p["fwd"])
    toe = np.exp(-((along - half - 0.01) / 0.012) ** 2) * np.exp(-(((xy - p["c"]) @ np.array([-p["fwd"][1], p["fwd"][0]])) / 0.03) ** 2)
    floor_noise = 0.0012 * (_vnoise(xy, 0.008, k + 200) - 0.5)
    H = -depth * wall + 0.35 * depth * rim + 0.25 * depth * toe + floor_noise * wall
    edge = np.clip((MARGIN - sd) / 0.012, 0, 1)                         # flat at the patch border
    H *= edge
    me = bpy.data.meshes.new(f"fp{k}")
    me.from_pydata([(x, y, h) for (x, y), h in zip(xy, H)], [], faces)
    me.shade_smooth()
    fresh = me.attributes.new("fresh", "FLOAT", "POINT")
    fresh.data.foreach_set("value", np.clip(wall + 0.6 * rim, 0, 1).astype(np.float32))
    fp = bpy.data.objects.new(f"footprint_{k}", me)
    fp.location = (0, 0, p["gz"] + 0.0002)
    fp.data.materials.append(pm_sand)
    coll.objects.link(fp)
    fp["trodden"] = 0.0
    for ff, vis in ((p["f"] - 1, 0.0), (p["f"], 1.0)):
      fp.scale = (1, 1, vis if vis else 1e-4)
      fp["trodden"] = vis
      fp.keyframe_insert("scale", frame=ff)
      fp.keyframe_insert('["trodden"]', frame=ff)
    # ---------------------------------------------- dust on touchdown
    if k >= max_puffs:
      continue
    strength = float(np.clip(p["v"] / 0.35, 0.4, 2.5))
    _burst(coll, k, p["c"], p["gz"], p["f"], 160 * strength, 0.55 * strength ** 0.5, 0.6, 22, grain, wind, size=0.0016)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1.0, location=(p["c"][0], p["c"][1], p["gz"] + 0.02))
    pf = bpy.context.active_object
    pf.name = f"dust_puff_{k}"
    for cc in pf.users_collection:
      cc.objects.unlink(pf)
    coll.objects.link(pf)
    pm = puff_m.copy()
    pf.data.materials.append(pm)
    pf.visible_shadow = False
    dn = pm.node_tree.nodes["density"].inputs[1]
    life = int(fps * (1.2 + 0.5 * strength))
    for ff, r, h, a in ((p["f"] - 1, 0.001, 0.0, 0.0), (p["f"], 0.05, 0.01, 1.0), (p["f"] + 4, 0.12 * strength ** 0.5, 0.035, 0.85),
                        (p["f"] + life // 2, 0.2 * strength ** 0.5, 0.07, 0.4), (p["f"] + life, 0.28 * strength ** 0.5, 0.1, 0.0)):
      t = (ff - p["f"]) / fps
      pf.location = (p["c"][0] + wind[0] * t, p["c"][1] + wind[1] * t, p["gz"] + 0.02 + h)
      pf.scale = (r, r, r * 0.5)
      pf.keyframe_insert("location", frame=ff)
      pf.keyframe_insert("scale", frame=ff)
      dn.default_value = (12.0 if night else 30.0) * a * min(strength, 1.6)
      dn.keyframe_insert("default_value", frame=ff)
  # ------------------------------------------------ grains trickling off as a pad lifts
  for k, (f, c, z) in enumerate(lifts[:max_puffs]):
    em = _burst(coll, 1000 + k, c, 0.0, f, 60, -0.05, 0.12, 18, grain, (0.05, 0.02), size=0.0014, radius=0.04)
    em.location.z = z + 0.004
  sc.frame_set(1)
  return len(prints)
