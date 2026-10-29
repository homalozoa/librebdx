"""MuJoCo simulation harness for the model-based gait stack (with logging/video)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from disneyrepro.common.render import Renderer, add_floor, make_camera
from disneyrepro.gait.controller import ControllerConfig, GaitController
from disneyrepro.gait.planner import FootGeom


@dataclass
class RobotGaitSetup:
  xml: str
  foot: FootGeom
  foot_sites: dict
  upper_joints: tuple = ()
  posture: dict = field(default_factory=dict)  # joint name -> nominal angle
  friction: float = 1.0

  def contact_local(self):
    f = self.foot
    hw = f.half_width
    if f.line_contact:
      return {"flat": [np.array([f.heel_x, 0, 0]), np.array([f.toe_x, 0, 0])],
              "toe": [np.array([f.toe_x, 0, 0])], "heel": [np.array([f.heel_x, 0, 0])]}
    return {
      "flat": [np.array([x, y, 0.0]) for x in (f.heel_x, f.toe_x) for y in (-hw, hw)],
      "toe": [np.array([f.toe_x, y, 0.0]) for y in (-hw, hw)],
      "heel": [np.array([f.heel_x, y, 0.0]) for y in (-hw, hw)],
    }


def build_world(setup: RobotGaitSetup):
  spec = mujoco.MjSpec.from_file(setup.xml)
  add_floor(spec)
  spec.option.timestep = 0.001
  m = spec.compile()
  d = mujoco.MjData(m)
  d.qpos[3:7] = [1, 0, 0, 0]
  for jn, v in setup.posture.items():
    d.qpos[m.joint(jn).qposadr[0]] = v
  mujoco.mj_forward(m, d)
  zs = min(d.site_xpos[m.site(setup.foot_sites[s]).id][2] for s in "lr")
  d.qpos[2] -= zs - 0.0005
  mujoco.mj_forward(m, d)
  return m, d


def posture_vector(m, setup):
  out = np.zeros(m.nv - 6)
  for jn, v in setup.posture.items():
    out[m.joint(jn).dofadr[0] - 6] = v
  return out


def run(setup: RobotGaitSetup, style, duration, v_fn, push=None, video=None, cfg_kw=None, log_every=8):
  """push: (t0, duration, force_xyz, body_name). video: path or None."""
  m, d = build_world(setup)
  cfg = ControllerConfig(foot=setup.foot, foot_sites=setup.foot_sites, contact_local=setup.contact_local(),
                         upper_joints=setup.upper_joints, **(cfg_kw or {}))
  ctl = GaitController(m, d, style, cfg, posture=posture_vector(m, setup))
  names = {id(s): n for n, s in style.samples.items()}
  rend, frames = None, []
  if video:
    rend = Renderer(m, 640, 480)
  log = []
  n = int(duration / m.opt.timestep)
  push_bid = m.body(push[3]).id if push else None
  fell = False
  for k in range(n):
    t = k * m.opt.timestep
    v = v_fn(t)
    ctl.step(d, v)
    d.xfrc_applied[:] = 0
    if push and push[0] <= t < push[0] + push[1]:
      d.xfrc_applied[push_bid, :3] = push[2]
    mujoco.mj_step(m, d)
    if d.xpos[1][2] < 0.5 * ctl.out.pelvis_z:
      fell = True
      break
    if k % log_every == 0:
      pl = ctl.planner
      fb = getattr(ctl, "fb_state", {})
      ref = getattr(ctl, "last_ref", None)
      R = d.xmat[1].reshape(3, 3)
      rec = dict(
        t=t, phi=pl.phi, v_cmd=np.array(v, float), path=pl.path.copy(),
        kappa=pl.kappa, step=np.array(fb.get("step", np.zeros(2))), vel_adj=pl.v_offset.copy(),
        weights={names[id(s)]: w for s, w in pl.blend.items},
        com=d.subtree_com[1].copy(), com_ref=None if ref is None else ref.com.copy(),
        pelvis=d.xpos[1].copy(), pelvis_yaw=math.atan2(R[1, 0], R[0, 0]),
        pelvis_z_ref=ctl.out.pelvis_z, pelvis_yaw_ref=math.atan2(ctl.out.pelvis_R[1, 0], ctl.out.pelvis_R[0, 0]),
        q=d.qpos[7:].copy(), q_des=ctl.q_des.copy(), tau=ctl.tau.copy(),
        upper_ref=dict(ctl.out.upper), contact_ref=dict(ctl.out.contact),
        foot={s: d.site_xpos[m.site(setup.foot_sites[s]).id].copy() for s in "lr"},
        foot_ref={s: ctl.out.foot_pos[s].copy() for s in "lr"},
        qvel_root=d.qvel[:6].copy(), wbc_status=ctl.wbc.last.get("status"),
      )
      log.append(rec)
    if rend is not None and k % int(1 / (30 * m.opt.timestep)) == 0:
      cam = make_camera(d.subtree_com[1], 1.4 + 0.6 * d.subtree_com[1][2], 150, -12)
      frames.append(rend.render(d, cam))
  if rend is not None:
    rend.close()
  if video and frames:
    from disneyrepro.common.render import save_video

    save_video(frames, video, fps=30)
  return {"log": log, "fell": fell, "t_end": t, "events": list(ctl.planner.events), "model": m}
