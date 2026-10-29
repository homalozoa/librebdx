"""Render a robot MJCF from a few views (visual + collision geometry)."""

import sys

import mujoco
import numpy as np

from disneyrepro.common.render import Renderer, add_floor, make_camera, save_image

xml, out = sys.argv[1], sys.argv[2]
height = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
spec = mujoco.MjSpec.from_file(xml)
add_floor(spec)
m = spec.compile()
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
if height == 0.0:
  # Lift so that the lowest geom touches the floor.
  zmin = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in range(m.ngeom) if m.geom_bodyid[g] > 0)
  height = d.qpos[2] - zmin
d.qpos[2] = height
mujoco.mj_forward(m, d)
com = d.subtree_com[1]
tiles = []
for show_col in (False, True):
  r = Renderer(m, 480, 480, show_collision=show_col)
  if show_col:
    for g in range(m.ngeom):
      if m.geom_group[g] == 2:
        m.geom_rgba[g, 3] = 0.15
  row = [r.render(d, make_camera(com, 1.6 * max(0.5, com[2] * 2.2), az, -12)) for az in (90, 150, 180)]
  tiles.append(np.concatenate(row, 1))
save_image(np.concatenate(tiles, 0), out)
print("saved", out, "base z", height)
