"""Record robot poses of env 0 for offline (Blender) rendering.

Each step stores the root pose relative to the env origin plus joint positions;
`save` writes an npz that scripts/export_blender.py turns into per-geom poses.
"""

from __future__ import annotations

import numpy as np


class PoseRecorder:
  def __init__(self):
    self.root_pos, self.root_quat, self.q, self.extra = [], [], [], {}
    self.joint_names = None

  def add(self, env, origin=None, **extra):
    r = env.scene["robot"]
    if self.joint_names is None:
      self.joint_names = list(r.joint_names)
    o = env.scene.env_origins[0] if origin is None else origin
    p = r.data.root_link_pos_w[0].clone()
    p[: o.shape[0]] -= o
    self.root_pos.append(p.cpu().numpy())
    self.root_quat.append(r.data.root_link_quat_w[0].cpu().numpy())
    self.q.append(r.data.joint_pos[0].cpu().numpy())
    for k, v in extra.items():
      self.extra.setdefault(k, []).append(v)

  def save(self, path, fps=50):
    np.savez_compressed(
      path, root_pos=np.asarray(self.root_pos), root_quat=np.asarray(self.root_quat), q=np.asarray(self.q),
      joint_names=np.asarray(self.joint_names), fps=fps, **{k: np.asarray(v) for k, v in self.extra.items()},
    )
