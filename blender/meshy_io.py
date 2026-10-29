"""Meshy texture round-trip for the BDX shell (Blender side).

  export:   blender -b -P blender/meshy_io.py -- export   -> results/bdx/meshy/in/<part>.obj (mm)
  transfer: blender -b -P blender/meshy_io.py -- transfer -> results/bdx/meshy/<part>.png + <part>_uv.npy

`transfer` imports Meshy's textured GLB, aligns it to our mesh (axis convention
+ bounding box), and gives every face corner of OUR mesh the UV of the closest
point on Meshy's surface, so simulation geometry stays untouched.
"""

import itertools
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform

sys.path.insert(0, str(Path(__file__).parent))
import bdx_lib as L  # noqa: E402

OUT = L.TEX
SHELL = sorted(n for r in ("ivory", "ember", "slate") for n in L.ROLES[r] if not n.endswith(("_Ear", "Upper_Leg_Detail")))


MECH_SRC = ["Left_Hip_Yaw_Motor", "Left_Knee_Motor", "Left_Hip_Roll_Motor", "Left_Ankle_Motor", "Neck_Motor",
            "Head_Pitch_Motor", "Left_Upper_Leg", "Left_Lower_Leg", "Left_Foot", "Left_Holder", "Left_Hip_Top_Holder",
            "Left_Hip_Yaw_Support", "Front_Support", "Neck", "Left_Head_Support", "Head_Pitch_Motor_Support"]


def export(out=OUT, parts=SHELL):
  z = np.load(L.DATA / "meshes.npz")
  (out / "in").mkdir(parents=True, exist_ok=True)
  for n in parts:
    v, f = z[f"v_{n}"] * 1000.0, z[f"f_{n}"] + 1
    with open(out / "in" / f"{n}.obj", "w") as fh:
      fh.writelines(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in v)
      fh.writelines(f"f {a} {b} {c}\n" for a, b, c in f)
  print("exported", len(parts), "parts:", " ".join(parts))


def _load_glb(path):
  before = set(bpy.data.objects)
  bpy.ops.import_scene.gltf(filepath=str(path))
  obs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
  verts, tris, uvs, img = [], [], [], None
  base = 0
  for o in obs:
    me = o.data
    me.calc_loop_triangles()
    M = o.matrix_world
    verts += [M @ v.co for v in me.vertices]
    uvl = me.uv_layers.active.data
    for t in me.loop_triangles:
      tris.append([base + i for i in t.vertices])
      uvs.append([tuple(uvl[li].uv) for li in t.loops])
    base += len(me.vertices)
    for slot in o.material_slots:
      for nd in slot.material.node_tree.nodes:
        if nd.type == "TEX_IMAGE" and img is None and any(l.to_socket.name == "Base Color" for l in nd.outputs[0].links):
          img = nd.image
  return np.array([tuple(v) for v in verts]), np.array(tris), np.array(uvs), img


def _align(src, dst):
  """Best axis permutation/sign + bbox fit mapping Meshy verts onto ours."""
  best = None
  dc, ds = (dst.max(0) + dst.min(0)) / 2, dst.max(0) - dst.min(0)
  sub = dst[np.random.default_rng(0).choice(len(dst), min(600, len(dst)), replace=False)]
  for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1, -1), repeat=3):
      p = src[:, perm] * np.array(sg)
      pc, ps = (p.max(0) + p.min(0)) / 2, p.max(0) - p.min(0)
      s = float(np.median(ds / np.maximum(ps, 1e-9)))
      q = (p - pc) * s + dc
      qs = q[np.random.default_rng(1).choice(len(q), min(1500, len(q)), replace=False)]
      d = np.sqrt(((sub[:, None] - qs[None]) ** 2).sum(-1)).min(1).mean()
      if best is None or d < best[0]:
        best = (d, perm, sg, pc, s)
  d, perm, sg, pc, s = best
  return (lambda x: (x[:, perm] * np.array(sg) - pc) * s + dc), d / float(np.linalg.norm(ds))


def transfer():
  z = np.load(L.DATA / "meshes.npz")
  for n in SHELL:
    glb = OUT / "out" / f"{n}.glb"
    if not glb.exists():
      continue
    L.reset_scene()
    mv, mt, muv, img = _load_glb(glb)
    v = z[f"v_{n}"] * 1000.0
    f = z[f"f_{n}"]
    fit, err = _align(mv, v)
    mv = fit(mv)
    bvh = BVHTree.FromPolygons([Vector(p) for p in mv], [tuple(t) for t in mt])
    uv = np.zeros((len(f), 3, 2), np.float32)
    for fi, tri in enumerate(f):
      a0, b0, c0 = (Vector(v[i]) for i in tri)
      nrm = (b0 - a0).cross(c0 - a0)
      nrm = nrm.normalized() if nrm.length > 1e-12 else Vector((0, 0, 1))
      for k, vi in enumerate(tri):
        # Sample slightly inside the face, and cast along the face normal from just
        # outside so thin double-sided panels take the texture of the matching side.
        p = Vector(v[vi] * 0.9 + v[tri].mean(0) * 0.1)
        loc, _, ti, d = bvh.ray_cast(p + nrm * 0.8, -nrm, 2.0)
        if loc is None:
          loc, _, ti, _ = bvh.find_nearest(p)
        a, b, c = (Vector(mv[i]) for i in mt[ti])
        ua, ub, uc = (Vector((*muv[ti][j], 0.0)) for j in range(3))
        w = barycentric_transform(loc, a, b, c, ua, ub, uc)
        uv[fi, k] = (w.x, w.y)
    np.save(OUT / f"{n}_uv.npy", uv)
    img.filepath_raw = str(OUT / f"{n}.png")
    img.file_format = "PNG"
    img.save()
    print(f"transfer {n}: align err {err:.3f}")


if __name__ == "__main__":
  cmd = sys.argv[sys.argv.index("--") + 1]
  if cmd == "export2":
    export(L.ROOT / "results" / "bdx" / "meshy2", SHELL + MECH_SRC)
  else:
    {"export": export, "transfer": transfer}[cmd]()
