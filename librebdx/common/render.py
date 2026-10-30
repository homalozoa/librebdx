"""Offscreen rendering helpers (EGL) for still images and videos."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def make_camera(lookat, distance=1.5, azimuth=135.0, elevation=-15.0):
  cam = mujoco.MjvCamera()
  cam.lookat[:] = lookat
  cam.distance = distance
  cam.azimuth = azimuth
  cam.elevation = elevation
  return cam


class Renderer:
  def __init__(self, model, width=640, height=480, show_collision=False):
    self.model = model
    self.r = mujoco.Renderer(model, height=height, width=width)
    self.opt = mujoco.MjvOption()
    if show_collision:
      self.opt.geomgroup[3] = 1
    self.opt.sitegroup[:] = 0

  def close(self):
    self.r.close()

  def render(self, data, cam):
    self.r.update_scene(data, camera=cam, scene_option=self.opt)
    return self.r.render().copy()


def add_floor(spec, size=20.0):
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[size, size, 0.05],
    rgba=[0.75, 0.78, 0.82, 1],
    contype=1,
    conaffinity=1,
  )
  spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
  spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
  return spec


def save_video(frames, path, fps=30):
  import mediapy

  mediapy.write_video(str(path), np.asarray(frames), fps=fps)


def save_image(img, path):
  import mediapy

  mediapy.write_image(str(path), img)
