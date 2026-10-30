"""Story beats for the calm-gait walking policy (W16calm): pace that changes with the story.

python scripts/bdx_story2.py STAND WALK EXCITED_CKPT
  story_crossing2 : slow and careful (0.15 m/s, ~2.2 steps/s), growing steadier (0.4 m/s)
  story_wind2     : 0.3 m/s walk with three 90 N, 0.1 s pushes
  story_reunion   : stand -> hurry over at 0.7 m/s (~4 steps/s) -> stop at double support -> excited
  story_partner   : the waiting partner (standing, small head motions), same length as the reunion
Poses only (results/bdx/traj/*.npz); 2 s of pre-roll is recorded and trimmed by the film.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from librebdx.tasks import register_all

register_all()
from librebdx.bdx.record import PoseRecorder  # noqa: E402
from librebdx.bdx.runtime import PolicySlot  # noqa: E402

TRAJ = Path("results/bdx/traj")
TRAJ.mkdir(parents=True, exist_ok=True)
stand_ck, walk_ck, exc_ck = sys.argv[1:4]
DT = 0.02
PRE = 2.0


def smooth(a, b, t, t0, t1):
  s = np.clip((t - t0) / (t1 - t0), 0, 1)
  return a + (b - a) * s * s * (3 - 2 * s)


slots = {
  "stand": PolicySlot("stand", "Disney-BDX-Standing", stand_ck),
  "walk": PolicySlot("walk", "Disney-BDX-Walking", walk_ck),
  "exc": PolicySlot("exc", "Disney-BDX-Episodic-excited", exc_ck),
}
origins = {k: s.env.scene.env_origins[0, :2].clone() for k, s in slots.items()}
dev = slots["stand"].env.device
for s in slots.values():
  s.cmd.external = True


class Run:
  def __init__(self):
    self.active = "stand"
    self.rec = PoseRecorder()
    self.falls = 0

  def switch(self, to, init=None):
    st = slots[self.active].export_state()
    slots[to].import_state(st, origins[to] - origins[self.active])
    if init:
      init(slots[to])
    self.active = to

  def step(self, **extra):
    s = slots[self.active]
    s.step()
    if bool(s.env.termination_manager.terminated.any()):
      self.falls += 1
      print("  fall at", round(self.rec_t, 2), "s in", self.active, flush=True)
    self.rec_t = getattr(self, "rec_t", -PRE) + DT
    self.rec.add(s.env, origins[self.active], policy=self.active, **extra)


def reset_all():
  for s in slots.values():
    s.env.reset()
    s.obs = s.wenv.get_observations()


def walk_cmd(vx, wz=0.0, head=0.0):
  slots["walk"].cmd.set_command(torch.tensor([[0, 0, head, 0, vx, 0, wz]], dtype=torch.float32, device=dev))


def walk_init(slot):
  slot.cmd.phi[:] = 0.0
  slot.cmd.g[:] = 0.0


ONLY = set(sys.argv[4:])  # optional: names of beats to record (crossing wind reunion partner)


def want(name):
  return not ONLY or name in ONLY


# ------------------------------------------------ crossing: slow and careful, then steadier
if want('crossing'):
  reset_all()
  R = Run()
  slots["stand"].cmd.set_command(torch.zeros(1, 8, device=dev))
  t = -PRE
  while t < 0.6:
    R.step()
    t += DT
  walk_cmd(0.0)
  R.switch("walk", walk_init)
  while t < 17.0:
    vx = smooth(0.0, 0.15, t, 0.6, 1.8) + smooth(0.0, 0.12, t, 5.5, 8.0) + smooth(0.0, 0.13, t, 10.5, 13.0)
    wz = smooth(0.0, 0.16, t, 2.0, 3.0) - smooth(0.0, 0.16, t, 6.0, 7.0)
    walk_cmd(vx, wz, smooth(0.0, 0.1, t, 2.0, 3.0))
    R.step()
    t += DT
  R.rec.save(TRAJ / "story_crossing2.npz")
  print("story_crossing2 falls", R.falls, flush=True)

# ------------------------------------------------ wind: three pushes at 0.3 m/s
if want('wind'):
  reset_all()
  R = Run()
  R.switch("walk", walk_init)
  r = slots["walk"].robot
  bid = r.find_bodies("base_link")[0]
  PUSH = {2.8: (0.0, 90.0), 5.8: (0.0, -90.0), 8.8: (-90.0, 0.0)}
  t = -PRE
  while t < 11.0:
    walk_cmd(smooth(0.0, 0.3, t, -PRE, -PRE + 1.0))
    f = torch.zeros(1, 1, 3, device=dev)
    for t0, (fx, fy) in PUSH.items():
      if t0 <= t < t0 + 0.1:
        f[0, 0, 0], f[0, 0, 1] = fx, fy
    r.write_external_wrench_to_sim(f, torch.zeros_like(f), body_ids=bid)
    R.step(force=f[0, 0].tolist())
    t += DT
  R.rec.save(TRAJ / "story_wind2.npz")
  print("story_wind2 falls", R.falls, flush=True)

# ------------------------------------------------ reunion: hurry over, stop, excited
if want('reunion'):
  reset_all()
  R = Run()
  slots["stand"].cmd.set_command(torch.tensor([[0, 0, 0.15, 0, 0, 0, 0, 0]], dtype=torch.float32, device=dev))
  t = -PRE
  xy0, run = None, 0.0
  phase = "look"
  while t < 13.0:
    s = slots[R.active]
    if phase == "look" and t >= 0.6:
      walk_cmd(0.0)
      R.switch("walk", walk_init)
      xy0 = (slots["walk"].robot.data.root_link_pos_w[0, :2] - origins["walk"]).clone()
      phase = "run"
    if phase == "run":
      v_run = 0.7 if run < 2.3 else 0.7 - 0.45 * min(1.0, (run - 2.3) / 0.9)  # ease off over the last metre
      walk_cmd(smooth(0.0, v_run, t, 0.6, 1.6))
      run = float(torch.linalg.norm(slots["walk"].robot.data.root_link_pos_w[0, :2] - origins["walk"] - xy0))
      if run > 3.4:  # planar distance: resets give the droid a random heading
        walk_cmd(0.0)
        phase = "stop"
    if phase == "stop" and bool((slots["walk"].cmd.ref_contact[0] > 0.5).all()):
      R.switch("stand")
      slots["stand"].cmd.set_command(torch.zeros(1, 8, device=dev))
      t_stop = t
      phase = "settle"
    if phase == "settle" and t >= t_stop + 1.0:
      R.switch("exc", lambda sl: sl.cmd.t.__setitem__(slice(None), 0.0))
      t_exc = t
      phase = "exc"
    if phase == "exc" and t >= t_exc + float(slots["exc"].cmd.T):
      R.switch("stand")
      slots["stand"].cmd.set_command(torch.zeros(1, 8, device=dev))
      phase = "after"
    R.step()
    t += DT
  R.rec.save(TRAJ / "story_reunion.npz")
  print("story_reunion falls", R.falls, "run distance", round(run, 2), "phase", phase, flush=True)

# ------------------------------------------------ the waiting partner
if want('partner'):
  reset_all()
  R = Run()
  t = -PRE
  while t < 13.0:
    yaw = smooth(0.0, 0.2, t, 0.5, 1.5) - smooth(0.0, 0.2, t, 4.0, 5.0)
    hp = smooth(0.0, 0.12, t, 3.6, 4.4) - smooth(0.0, 0.12, t, 4.6, 5.4)
    slots["stand"].cmd.set_command(torch.tensor([[0, yaw, hp, 0, 0, 0, 0, 0]], dtype=torch.float32, device=dev))
    R.step()
    t += DT
  R.rec.save(TRAJ / "story_partner.npz")
  print("story_partner falls", R.falls, flush=True)
