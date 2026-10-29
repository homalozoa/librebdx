"""Kinematic playback of a walking reference table for a few commands."""
import sys
import numpy as np
import mujoco
from disneyrepro.common.render import Renderer, add_floor, make_camera, save_video

xml, table, out = sys.argv[1], sys.argv[2], sys.argv[3]
cmds = [tuple(map(float, c.split(","))) for c in sys.argv[4:]]
d_ = np.load(table)
names = list(d_["joint_names"])
spec = mujoco.MjSpec.from_file(xml)
add_floor(spec)
m = spec.compile()
d = mujoco.MjData(m)
qadr = [m.joint(n).qposadr[0] for n in names]
r = Renderer(m, 480, 400)
frames = []

def interp(key, v, phi):
  vx, vy, wz = d_["vx"], d_["vy"], d_["wz"]
  def idx(g, x):
    x = np.clip(x, g[0], g[-1]); i = min(np.searchsorted(g, x, side="right") - 1, len(g) - 2)
    return i, (x - g[i]) / (g[i + 1] - g[i])
  (i, a), (j, b), (k, c) = idx(vx, v[0]), idx(vy, v[1]), idx(wz, v[2])
  n = d_["rate"].shape[-1]
  p = phi * n; l = int(p) % n; e = p - int(p); l2 = (l + 1) % n
  A = d_[key]
  acc = 0
  for di, wi in ((0, 1 - a), (1, a)):
    for dj, wj in ((0, 1 - b), (1, b)):
      for dk, wk in ((0, 1 - c), (1, c)):
        cell = A[i + di, j + dj, k + dk]
        acc = acc + wi * wj * wk * ((1 - e) * cell[l] + e * cell[l2])
  return acc

for v in cmds:
  pf = np.zeros(3); phi = 0.0; dt = 1 / 50
  for step in range(150):
    rate = interp("rate", v, phi)
    xy = interp("root_xy", v, phi); z = interp("root_z", v, phi); quat = interp("root_quat", v, phi); q = interp("q", v, phi)
    c, s = np.cos(pf[2]), np.sin(pf[2])
    d.qpos[0:2] = pf[:2] + np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])
    d.qpos[2] = z
    qz = np.array([np.cos(pf[2] / 2), 0, 0, np.sin(pf[2] / 2)])
    qq = np.zeros(4); mujoco.mju_mulQuat(qq, qz, quat / np.linalg.norm(quat)); d.qpos[3:7] = qq
    d.qpos[qadr] = q
    mujoco.mj_forward(m, d)
    frames.append(r.render(d, make_camera([d.qpos[0], d.qpos[1], 0.25], 1.5, 120, -15)))
    phi = (phi + rate * dt) % 1.0
    pf += dt * np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
r.close()
save_video(frames, out, fps=50)
