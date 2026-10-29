"""Tiny helpers to build articulated robots procedurally with MjSpec."""

from __future__ import annotations

import mujoco
import numpy as np

G = mujoco.mjtGeom


def new_spec(name):
  spec = mujoco.MjSpec()
  spec.modelname = name
  spec.compiler.degree = False
  spec.compiler.autolimits = True
  spec.option.timestep = 0.001
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.default.geom.contype = 0
  spec.default.geom.conaffinity = 0
  spec.default.joint.armature = 0.01
  spec.default.joint.damping = [0.05, 0.0, 0.0]
  spec.default.joint.frictionloss = 0.02
  return spec


def add_material(spec, name, rgba):
  m = spec.add_material()
  m.name = name
  m.rgba = rgba
  return name


def body(parent, name, pos, quat=(1, 0, 0, 0)):
  return parent.add_body(name=name, pos=list(pos), quat=list(quat))


def inertial(b, mass, com=(0, 0, 0), diag=None, size=None):
  """Point inertia from a box of the given size (or explicit diag)."""
  b.mass = mass
  b.ipos = list(com)
  if diag is None:
    sx, sy, sz = size
    diag = [mass / 12 * (sy**2 + sz**2), mass / 12 * (sx**2 + sz**2), mass / 12 * (sx**2 + sy**2)]
  b.inertia = list(diag)
  b.explicitinertial = True


def hinge(b, name, axis, rng, armature=None):
  j = b.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis), range=list(rng))
  if armature is not None:
    j.armature = armature
  return j


def capsule(b, name, a, c, r, material=None, collide=0, mask=(0, 0), group=1):
  g = b.add_geom(name=name, type=G.mjGEOM_CAPSULE, fromto=list(a) + list(c), size=[r, 0, 0], group=group)
  g.contype, g.conaffinity = mask if collide else (0, 0)
  if material:
    g.material = material
  return g


def box(b, name, pos, half, material=None, mask=(0, 0), quat=(1, 0, 0, 0), group=1):
  g = b.add_geom(name=name, type=G.mjGEOM_BOX, pos=list(pos), size=list(half), quat=list(quat), group=group)
  g.contype, g.conaffinity = mask
  if material:
    g.material = material
  return g


def sphere(b, name, pos, r, material=None, mask=(0, 0), group=1):
  g = b.add_geom(name=name, type=G.mjGEOM_SPHERE, pos=list(pos), size=[r, 0, 0], group=group)
  g.contype, g.conaffinity = mask
  if material:
    g.material = material
  return g


def ellipsoid(b, name, pos, radii, material=None, mask=(0, 0), quat=(1, 0, 0, 0), group=1):
  g = b.add_geom(name=name, type=G.mjGEOM_ELLIPSOID, pos=list(pos), size=list(radii), quat=list(quat), group=group)
  g.contype, g.conaffinity = mask
  if material:
    g.material = material
  return g


def add_motors(spec, joints_effort: dict):
  for jn, eff in joints_effort.items():
    a = spec.add_actuator(name=jn, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.gainprm[0] = 1.0
    a.ctrlrange = [-eff, eff]
    a.ctrllimited = True


def quat_from_axis_angle(axis, angle):
  axis = np.asarray(axis, float)
  axis = axis / np.linalg.norm(axis)
  return [np.cos(angle / 2), *(np.sin(angle / 2) * axis)]


def save(spec, path):
  xml = spec.to_xml()
  open(path, "w").write(xml)
  return path


def summarize(path):
  m = mujoco.MjModel.from_xml_path(str(path))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  zs = []
  for g in range(m.ngeom):
    if m.geom_bodyid[g] == 0:
      continue
    zs.append((d.geom_xpos[g][2] + m.geom_rbound[g], d.geom_xpos[g][2] - m.geom_rbound[g]))
  top = max(z[0] for z in zs)
  joints = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != 0]
  print(f"{path}: mass {m.body_subtreemass[1]:.2f} kg, nu {m.nu}, top {top:.3f} m (root at {d.qpos[2]:.3f}), joints {joints}")
  return m
