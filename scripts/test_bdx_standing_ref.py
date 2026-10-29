"""Check the BDX standing reference IK: accuracy, warm-start speed, and render Fig.4-style poses."""

import time

import mujoco
import numpy as np
import torch

from disneyrepro.bdx.standing_ref import CMD_HI, CMD_LO, CMD_NAMES, XML, BdxStandingReference
from disneyrepro.common.render import Renderer, add_floor, make_camera, save_image
from disneyrepro.kin.torch_kin import mat_to_quat

dev = "cuda"
ref = BdxStandingReference(device=dev)
print("nominal residual", ref.nominal_residual, "x_nominal", ref.x_nominal.cpu().numpy().round(3))
print("head height0", ref.head_height0)

# Cold-start accuracy on random commands.
g = ref.sample_commands(4096)
t0 = time.time()
x, out = ref.solve(g)
torch.cuda.synchronize()
r = out["residual"].abs().max(-1).values
print(f"cold solve 4096: {time.time()-t0:.3f}s, residual median {r.median():.2e} p95 {r.quantile(0.95):.2e} max {r.max():.2e}")

# Warm start along smooth command trajectories.
g0 = g.clone()
g1 = ref.sample_commands(4096)
worst = 0
t0 = time.time()
for i in range(1, 51):
  gi = g0 + (g1 - g0) * i / 50
  x, out = ref.solve(gi, x, iters=2)
  worst = max(worst, float(out["residual"].abs().max(-1).values.quantile(0.99)))
torch.cuda.synchronize()
print(f"warm 50 steps x 2 iters: {time.time()-t0:.3f}s, p99 residual worst {worst:.2e}")

# Render: nominal plus min/max of each command dimension (Fig. 4).
spec = mujoco.MjSpec.from_file(str(XML))
add_floor(spec)
m = spec.compile()
d = mujoco.MjData(m)
rend = Renderer(m, 320, 320)
names = ref.kin.joint_names
qadr = [m.joint(n).qposadr[0] for n in names]
tiles = []
for row in (range(0, 4), range(4, 8)):
  imgs = []
  for dim in row:
    for val in (CMD_LO[dim], CMD_HI[dim]):
      gg = torch.zeros(1, 8, device=dev)
      gg[0, dim] = val
      xx, o = ref.solve(gg)
      d.qpos[:3] = o["root_pos"][0].cpu().numpy()
      d.qpos[3:7] = mat_to_quat(o["root_R"])[0].cpu().numpy()
      d.qpos[qadr] = o["q"][0].cpu().numpy()
      mujoco.mj_forward(m, d)
      zmin = min(d.site_xpos[m.site("left_sole").id][2], d.site_xpos[m.site("right_sole").id][2])
      d.qpos[2] -= zmin
      mujoco.mj_forward(m, d)
      imgs.append(rend.render(d, make_camera([0, 0, 0.3], 1.2, 145, -10)))
      print(CMD_NAMES[dim], val, "res", float(o["residual"].abs().max()))
  tiles.append(np.concatenate(imgs, 1))
save_image(np.concatenate(tiles, 0), "results/figs/bdx_standing_ref_fig4.png")
