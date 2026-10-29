"""Export BDX-R visual meshes and recorded trajectories for Blender.

  results/bdx/blender/meshes.npz      compiled visual meshes (MuJoCo frame) + geom table
  results/bdx/blender/<clip>.npz      per-frame world pose of every visual geom

python scripts/export_blender.py results/bdx/traj/*.npz
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

from disneyrepro.bdx.robot_cfg import get_spec

OUT = Path("results/bdx/blender")
OUT.mkdir(parents=True, exist_ok=True)
m = get_spec().compile()
d = mujoco.MjData(m)
G = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]


def name(obj, i):
  return mujoco.mj_id2name(m, obj, i) or f"{obj}_{i}"


meshes = {}
for g in G:
  k = int(m.geom_dataid[g])
  if k in meshes:
    continue
  v0, nv, f0, nf = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
  meshes[k] = (m.mesh_vert[v0 : v0 + nv].copy(), m.mesh_face[f0 : f0 + nf].copy())
blob = {f"v_{name(mujoco.mjtObj.mjOBJ_MESH, k)}": v for k, (v, _) in meshes.items()}
blob |= {f"f_{name(mujoco.mjtObj.mjOBJ_MESH, k)}": f for k, (_, f) in meshes.items()}
# Compiled vertices are re-centred: raw = mesh_quat * v + mesh_pos (before scaling back).
blob |= {f"p_{name(mujoco.mjtObj.mjOBJ_MESH, k)}": np.r_[m.mesh_pos[k], m.mesh_quat[k], m.mesh_scale[k]] for k in meshes}
np.savez_compressed(
  OUT / "meshes.npz",
  geom_name=np.array([name(mujoco.mjtObj.mjOBJ_GEOM, g) for g in G]),
  geom_mesh=np.array([name(mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) for g in G]),
  geom_body=np.array([name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for g in G]),
  geom_rgba=m.geom_rgba[G],
  **blob,
)

jadr = {}
for j in range(m.njnt):
  if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
    free = m.jnt_qposadr[j]
  else:
    jadr[name(mujoco.mjtObj.mjOBJ_JOINT, j)] = m.jnt_qposadr[j]

for path in sys.argv[1:]:
  tr = np.load(path)
  names = [str(n).split("/")[-1] for n in tr["joint_names"]]
  idx = [jadr[n] for n in names]
  T = tr["q"].shape[0]
  pos = np.zeros((T, len(G), 3), np.float32)
  quat = np.zeros((T, len(G), 4), np.float32)
  for t in range(T):
    d.qpos[free : free + 3] = tr["root_pos"][t]
    d.qpos[free + 3 : free + 7] = tr["root_quat"][t]
    d.qpos[idx] = tr["q"][t]
    mujoco.mj_kinematics(m, d)
    pos[t] = d.geom_xpos[G]
    for i, g in enumerate(G):
      q = np.zeros(4)
      mujoco.mju_mat2Quat(q, d.geom_xmat[g])
      quat[t, i] = q
  extra = {k: tr[k] for k in tr.files if k not in ("root_pos", "root_quat", "q", "joint_names", "fps")}
  np.savez_compressed(OUT / Path(path).name, pos=pos, quat=quat, fps=tr["fps"], root_pos=tr["root_pos"], root_quat=tr["root_quat"], **extra)
  print("exported", Path(path).stem, T, "frames")
