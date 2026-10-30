"""Episodic animations for the BDX character (stand-ins for the Maya clips).

The paper shows "happy dance", "excited motion", "jump" and "tantrum". We key
them procedurally in task space (torso, feet, head) and convert them to joint
trajectories with the same whole-body IK used for the other references.
Each clip starts and ends in the nominal standing pose. Stored at 50 Hz in
coordinates relative to the start path frame.

Run: python -m librebdx.bdx.episodic
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from librebdx.bdx.standing_ref import LEG_L, LEG_R, XML, NECK_PER_M, Nominal
from librebdx.kin.leg_ik import LegIK
from librebdx.kin.torch_kin import euler_zyx_to_mat, mat_to_quat

DATA = Path(__file__).parent / "data"
FPS = 50
NOM = Nominal()
W = NOM.stance_half_width


def smooth(x):
  x = np.clip(x, 0.0, 1.0)
  return x * x * (3 - 2 * x)


def window(t, t0, t1, ramp=0.3):
  """1 inside [t0, t1], smooth ramps outside."""
  return smooth((t - t0) / ramp + 1) * smooth((t1 - t) / ramp + 1)


def happy_dance(t):
  T = 6.0
  w = window(t, 0.6, T - 0.6)
  return dict(
    dz=w * (-0.012 - 0.018 * np.cos(2 * math.pi * 2.0 * t)),
    yaw=w * 0.18 * np.sin(2 * math.pi * 1.0 * t),
    pitch=w * 0.05 * np.sin(2 * math.pi * 2.0 * t),
    roll=w * 0.07 * np.sin(2 * math.pi * 1.0 * t + 0.5),
    head=dict(dh=w * 0.008 * np.sin(2 * math.pi * 2 * t), yaw=-w * 0.35 * np.sin(2 * math.pi * 1.0 * t), pitch=w * (0.1 + 0.12 * np.sin(2 * math.pi * 2 * t + 1)), roll=w * 0.2 * np.sin(2 * math.pi * 1.0 * t)),
    feet=None,
  ), T


def excited(t):
  T = 4.0
  w = window(t, 0.6, 3.2, 0.25)
  shake = window(t, 1.2, 2.6, 0.15)
  return dict(
    dz=w * (-0.01 - 0.01 * np.cos(2 * math.pi * 3 * t)),
    yaw=shake * 0.1 * np.sin(2 * math.pi * 4.0 * t),
    pitch=-w * 0.08,
    roll=shake * 0.1 * np.sin(2 * math.pi * 4.0 * t + 1.0),
    head=dict(dh=w * 0.01, yaw=shake * 0.3 * np.sin(2 * math.pi * 4 * t), pitch=w * 0.3, roll=shake * 0.15 * np.sin(2 * math.pi * 4 * t)),
    feet=None,
    emphasis=(1.2, 2.6),
  ), T


def jump(t):
  """Crouch, extend the legs in a constant-acceleration push, ballistic flight, land, recover.

  The torso drops 0.05 m (the knee range allows no deeper crouch with the torso
  pitched forward), then pushes to 0.03 m above the standing height in 0.15 s (legs
  nearly straight: nominal leg length 0.28 m, extended 0.34 m). Take-off speed is
  2 * 0.08 / 0.15 = 1.07 m/s: a 0.22 s flight and a 5.8 cm rise.
  The head pitches up during the flight, as described for Fig. 8.
  """
  T = 3.2
  g = 9.81
  t_c, t_p, d_c, d_up = 0.9, 0.15, 0.05, 0.03
  v0 = 2 * (d_c + d_up) / t_p
  t_to = t_c + t_p
  t_fl = 2 * v0 / g
  t_ld = t_to + t_fl
  crouch = -d_c * smooth((t - 0.4) / (t_c - 0.4))
  push = -d_c + 0.5 * (v0 / t_p) * (t - t_c) ** 2
  fly = d_up + v0 * (t - t_to) - 0.5 * g * (t - t_to) ** 2
  absorb = d_up - (0.05 + d_up) * np.sin(np.clip((t - t_ld) / 0.25, 0, 1) * math.pi / 2)
  recover = -0.05 * (1 - smooth((t - t_ld - 0.25) / 0.8))
  dz = np.where(t < t_c, crouch, np.where(t < t_to, push, np.where(t < t_ld, fly, np.where(t < t_ld + 0.25, absorb, recover))))
  # Feet leave with the body and tuck up 3 cm at the top of the flight.
  tuck = 0.03 * np.sin(np.clip((t - t_to) / t_fl, 0, 1) * math.pi)
  lift = np.where((t >= t_to) & (t < t_ld), fly - d_up + tuck, 0.0)
  return dict(
    dz=dz,
    yaw=0 * t,
    pitch=np.interp(t, [0, 0.4, t_c, t_to, t_ld, t_ld + 0.4, 2.6, T], [0, 0, 0.12, -0.02, -0.02, 0.1, 0, 0]),
    roll=0 * t,
    head=dict(dh=0 * t, yaw=0 * t, pitch=np.interp(t, [0, 0.8, t_to, t_ld, t_ld + 0.3, 2.2, T], [0, 0.2, -0.45, -0.4, -0.1, 0, 0]), roll=0 * t),
    feet=dict(lz=lift, rz=lift),
    emphasis=(t_c, t_ld + 0.3),
  ), T


def tantrum(t):
  T = 5.0
  w = window(t, 0.6, T - 0.6)
  stomp = lambda ph: np.maximum(0, np.sin(2 * math.pi * 1.5 * t + ph)) ** 2 * 0.045 * window(t, 1.0, 4.0, 0.2)
  lz, rz = stomp(0.0), stomp(math.pi)
  return dict(
    dz=w * -0.02,
    yaw=0 * t,
    pitch=w * 0.12,
    roll=w * 0.06 * np.sin(2 * math.pi * 1.5 * t),
    head=dict(dh=-w * 0.015, yaw=w * 0.45 * np.sin(2 * math.pi * 3.0 * t), pitch=w * -0.15, roll=0 * t),
    feet=dict(lz=lz, rz=rz),
  ), T


CLIPS = {"happy_dance": happy_dance, "excited": excited, "jump": jump, "tantrum": tantrum}


def build(name, fn, dev="cuda"):
  _, T = fn(np.zeros(1))
  t = np.arange(0, T + 1e-9, 1.0 / FPS)
  k, T = fn(t)
  n = len(t)
  ik = LegIK(XML, LEG_L + LEG_R, ("left_sole", "right_sole"), foot_roll=False, device=dev)
  kin = ik.kin
  tt = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float32, device=dev)
  z = NOM.torso_height + tt(k["dz"])
  R = euler_zyx_to_mat(tt(k["yaw"]), tt(k["pitch"]), tt(k["roll"]))
  # Neck joints: nominal + head offsets (same convention as the walking task).
  n0 = torch.tensor(NOM.neck, device=dev)
  h = k["head"]
  neck = n0.expand(n, 4).clone()
  neck[:, 0] -= NECK_PER_M * tt(h["dh"])
  neck[:, 1] += tt(h["pitch"])
  neck[:, 2] += tt(h["yaw"])
  neck[:, 3] += tt(h["roll"])
  P0, R0 = kin.forward(torch.zeros(1, 3, device=dev), torch.eye(3, device=dev)[None], torch.zeros(1, kin.nq, device=dev))
  R_off = [kin.site(P0, R0, s)[1][0] for s in ("left_sole", "right_sole")]
  fz = k["feet"] or {"lz": np.zeros(n), "rz": np.zeros(n)}
  fp = torch.zeros(n, 2, 3, device=dev)
  fp[:, 0, 1], fp[:, 1, 1] = W, -W
  fp[:, 0, 2], fp[:, 1, 2] = tt(fz["lz"]), tt(fz["rz"])
  fR = torch.stack(R_off)[None].expand(n, 2, 3, 3)
  # Shift the CoM towards the support foot while one foot is lifted.
  com = torch.zeros(n, 2, device=dev)
  if k["feet"] is not None:
    lz, rz = tt(fz["lz"]), tt(fz["rz"])
    com[:, 1] = 0.7 * W * torch.tanh((rz - lz) / 0.01)
  aux = ik.pack(z, R, neck, com, fp, fR)
  x0 = torch.zeros(n, 2 + 10, device=dev)
  x, r = ik.solve(aux, x0, iters=60)
  q = ik.full_q(x, neck)
  root = torch.cat([x[:, :2], z[:, None]], -1)
  contact = np.stack([np.asarray(fz["lz"]) < 0.002, np.asarray(fz["rz"]) < 0.002], -1).astype(np.float32)
  out = DATA / f"episodic_{name}.npz"
  np.savez_compressed(
    out, fps=FPS, root_pos=root.cpu().numpy(), root_quat=mat_to_quat(R).cpu().numpy(), q=q.cpu().numpy(),
    contact=contact, joint_names=np.array(kin.joint_names), emphasis=np.array(k.get("emphasis", (-1, -1)), float),
  )
  rf = r.abs().reshape(n, -1).max(-1).values.cpu().numpy()
  bad = np.nonzero(rf > 1e-3)[0]
  print(f"{name}: {T:.1f}s, IK residual max {r.abs().max():.2e}, saved {out}")
  if len(bad):
    print(f"  residual > 1 mm at t = {t[bad[0]]:.2f}..{t[bad[-1]]:.2f} s ({len(bad)} frames); dz there {np.asarray(k['dz'])[bad].min():+.3f}..{np.asarray(k['dz'])[bad].max():+.3f}")


if __name__ == "__main__":
  for name, fn in CLIPS.items():
    build(name, fn)
