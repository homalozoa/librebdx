"""Fig. 9: policy transitions standing -> walking -> standing -> episodic -> standing.

python scripts/bdx_transitions.py STAND_CKPT WALK_CKPT EPI_TASK EPI_CKPT
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from disneyrepro.tasks import register_all

register_all()
from disneyrepro.bdx.record import PoseRecorder  # noqa: E402
from disneyrepro.bdx.runtime import PolicySlot  # noqa: E402

OUT = Path("results/bdx")
OUT.mkdir(parents=True, exist_ok=True)
stand_ck, walk_ck, epi_task, epi_ck = sys.argv[1:5]
slots = {
  "stand": PolicySlot("stand", "Disney-BDX-Standing", stand_ck),
  "walk": PolicySlot("walk", "Disney-BDX-Walking", walk_ck),
  "epi": PolicySlot("epi", epi_task, epi_ck),
}
origins = {k: s.env.scene.env_origins[0, :2].clone() for k, s in slots.items()}
dev = slots["stand"].env.device
for s in slots.values():
  s.cmd.external = True
slots["stand"].cmd.set_command(torch.zeros(1, 8, device=dev))

active = "stand"
log, frames = [], []
poses = PoseRecorder()
t = 0.0
dt = 0.02
pending_stop = False


def switch(to, init=None):
  global active
  st = slots[active].export_state()
  off = origins[to] - origins[active]
  slots[to].import_state(st, off)
  if init:
    init(slots[to])
  active = to


def walk_init(slot, wz):
  slot.cmd.phi[:] = 0.0 if wz >= 0 else 0.5  # left turn: start with a left step
  slot.cmd.g[:] = 0.0


def epi_init(slot):
  slot.cmd.t[:] = 0.0



script = [  # (time, event)
  (1.5, ("walk", (0.4, 0.0, 0.3))),
  (5.0, ("stop", None)),
  (7.5, ("epi", None)),
]
ev_i = 0
T_END = 7.5 + float(slots["epi"].cmd.T) + 1.5
epi_end = None
while t < T_END:
  if ev_i < len(script) and t >= script[ev_i][0]:
    kind, arg = script[ev_i][1]
    if kind == "walk":
      switch("walk", lambda s: walk_init(s, arg[2]))
      slots["walk"].cmd.set_command(torch.tensor([[0, 0, 0, 0, *arg]], dtype=torch.float32, device=dev))
    elif kind == "stop":
      slots["walk"].cmd.set_command(torch.zeros(1, 7, device=dev))
      pending_stop = True
    elif kind == "epi":
      switch("epi", epi_init)
      epi_end = t + float(slots["epi"].cmd.T)
    ev_i += 1
  if pending_stop and active == "walk":
    c = slots["walk"].cmd
    if bool((c.ref_contact[0] > 0.5).all()) and t > 5.3:  # next double support
      switch("stand")
      log.append({"t": t, "event": "walk->stand at double support"})
      pending_stop = False
  if epi_end is not None and active == "epi" and t >= epi_end:
    switch("stand")
    epi_end = None
  s = slots[active]
  a = s.step()
  # Joint setpoints (rad) = offset + scale * a: comparable across policies.
  sp = s.act._processed_actions[0]
  rec = {"t": t, "policy": active, "action": sp.cpu().tolist(), "q": s.robot.data.joint_pos[0].cpu().tolist()}
  if active == "walk":
    rec["phase"] = float(s.cmd.phi[0])
  log.append(rec)
  frames.append(s.env.render())
  poses.add(s.env, origins[active], policy=active)
  if bool(s.env.termination_manager.terminated.any()):
    print("FELL at", t, active)
    break
  t += dt

json.dump(log, open(OUT / "transitions.json", "w"))
(OUT / "traj").mkdir(exist_ok=True)
poses.save(OUT / "traj" / "transitions.npz")
import mediapy  # noqa: E402

mediapy.write_video(OUT / "transitions.mp4", np.asarray(frames), fps=50)

# Plot (Fig. 9 layout).
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

recs = [r for r in log if "action" in r]
ts = np.array([r["t"] for r in recs])
A = np.array([r["action"] for r in recs])
pol = [r["policy"] for r in recs]
col = {"stand": "#d8f0d8", "walk": "#f5d6d6", "epi": "#d6e2f5"}
fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for k in range(3):
  i0 = 0
  for i in range(1, len(pol) + 1):
    if i == len(pol) or pol[i] != pol[i0]:
      ax[k].axvspan(ts[i0], ts[min(i, len(ts) - 1)], color=col[pol[i0]], lw=0)
      i0 = i
names = ["HY", "HR", "HP", "KP", "AP", "", "", "", "", "", "NP", "HdP", "HdY", "HdR"]
for j in range(10, 14):
  ax[0].plot(ts, A[:, j], label=names[j])
for j in range(0, 5):
  ax[1].plot(ts, A[:, j], label=names[j])
ph = np.array([r.get("phase", np.nan) for r in recs])
ax[2].plot(ts, ph, "k")
ax[0].set_ylabel("neck setpoints [rad]")
ax[1].set_ylabel("left leg setpoints [rad]")
ax[2].set_ylabel("phase")
ax[2].set_xlabel("Time [s]")
ax[0].legend(ncol=4, fontsize=7)
ax[1].legend(ncol=5, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig9_transitions.png", dpi=150)
# Action continuity: max per-step action jump at switch instants vs. elsewhere.
jumps = np.abs(np.diff(A, axis=0)).max(1)
sw = [i for i in range(1, len(pol)) if pol[i] != pol[i - 1]]
print(json.dumps({"switch_jumps": [float(jumps[i - 1]) for i in sw], "median_jump": float(np.median(jumps)), "p99_jump": float(np.percentile(jumps, 99))}))
