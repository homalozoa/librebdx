"""Batched forward kinematics of a MuJoCo tree in torch.

The model is compiled with ``fusestatic`` so that only bodies with joints remain
(plus the floating root). Sites of fused bodies move to their parents, so the
sole/head sites used by the reference generators are still available.

Conventions: quaternions are (w, x, y, z); rotations are 3x3 matrices.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    [
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ],
    -1,
  ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
  """Rotation matrix -> unit quaternion (w >= 0), batched and differentiable."""
  m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
  m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
  m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
  qw = 0.5 * torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10))
  qx = 0.5 * torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10))
  qy = 0.5 * torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10))
  qz = 0.5 * torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10))
  qx = torch.copysign(qx, m21 - m12)
  qy = torch.copysign(qy, m02 - m20)
  qz = torch.copysign(qz, m10 - m01)
  q = torch.stack([qw, qx, qy, qz], -1)
  return q / q.norm(dim=-1, keepdim=True)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  """axis: (..., 3) unit, angle: (...) -> (..., 3, 3) (Rodrigues)."""
  x, y, z = axis.unbind(-1)
  c, s = torch.cos(angle), torch.sin(angle)
  C = 1 - c
  return torch.stack(
    [
      c + x * x * C, x * y * C - z * s, x * z * C + y * s,
      y * x * C + z * s, c + y * y * C, y * z * C - x * s,
      z * x * C - y * s, z * y * C + x * s, c + z * z * C,
    ],
    -1,
  ).reshape(angle.shape + (3, 3))


def euler_zyx_to_mat(yaw, pitch, roll):
  """R = Rz(yaw) Ry(pitch) Rx(roll)."""
  ez = torch.tensor([0.0, 0.0, 1.0], device=yaw.device, dtype=yaw.dtype)
  ey = torch.tensor([0.0, 1.0, 0.0], device=yaw.device, dtype=yaw.dtype)
  ex = torch.tensor([1.0, 0.0, 0.0], device=yaw.device, dtype=yaw.dtype)
  return (
    axis_angle_to_mat(ez.expand(yaw.shape + (3,)), yaw)
    @ axis_angle_to_mat(ey.expand(pitch.shape + (3,)), pitch)
    @ axis_angle_to_mat(ex.expand(roll.shape + (3,)), roll)
  )


def mat_to_euler_zyx(R):
  yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
  pitch = torch.asin(torch.clamp(-R[..., 2, 0], -1.0, 1.0))
  roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
  return yaw, pitch, roll


def so3_log(R: torch.Tensor) -> torch.Tensor:
  """Rotation vector of R (..., 3, 3)."""
  tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
  cos = torch.clamp((tr - 1) / 2, -1 + 1e-7, 1 - 1e-7)
  ang = torch.acos(cos)
  v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
  scale = torch.where(ang < 1e-4, 0.5 + ang**2 / 12, ang / (2 * torch.sin(ang)))
  return v * scale[..., None]


class TorchKinematics:
  """FK for the joint tree of an MjModel (hinge joints + optional free root)."""

  def __init__(self, xml_path: str, device="cpu", dtype=torch.float32):
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.compiler.fusestatic = True
    # Keep bodies referenced by name in the generators even if static.
    self.model = m = spec.compile()
    self.device, self.dtype = device, dtype
    t = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=dtype)
    self.nbody = m.nbody
    self.parent = [int(p) for p in m.body_parentid]
    self.body_pos = t(m.body_pos)
    self.body_rot = quat_to_mat(t(m.body_quat))
    self.mass = t(m.body_mass)
    self.ipos = t(m.body_ipos)
    self.total_mass = float(m.body_mass[1:].sum())
    self.body_names = [m.body(i).name for i in range(m.nbody)]
    # Hinge joints in qpos order (excluding the free joint).
    self.joint_names, self.body_joint = [], [-1] * m.nbody
    self.joint_axis, self.joint_pos, self.joint_range = [], [], []
    k = 0
    for j in range(m.njnt):
      if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        continue
      assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
      b = m.jnt_bodyid[j]
      assert self.body_joint[b] == -1, "one hinge per body supported"
      self.body_joint[b] = k
      self.joint_names.append(m.joint(j).name)
      self.joint_axis.append(m.jnt_axis[j])
      self.joint_pos.append(m.jnt_pos[j])
      self.joint_range.append(m.jnt_range[j])
      k += 1
    self.nq = k
    self.joint_axis = t(np.array(self.joint_axis))
    self.joint_pos = t(np.array(self.joint_pos))
    self.joint_range = t(np.array(self.joint_range))
    self.site_names = [m.site(i).name for i in range(m.nsite)]
    self.site_body = [int(b) for b in m.site_bodyid]
    self.site_pos = t(m.site_pos)
    self.site_rot = quat_to_mat(t(m.site_quat))

  def jidx(self, names):
    return [self.joint_names.index(n) for n in names]

  def sidx(self, name):
    return self.site_names.index(name)

  def forward(self, root_pos, root_rot, q):
    """root_pos (B,3), root_rot (B,3,3), q (B,nq) -> lists of (B,3), (B,3,3) per body."""
    P, R = [None] * self.nbody, [None] * self.nbody
    B = q.shape[0]
    P[0] = torch.zeros(B, 3, device=q.device, dtype=q.dtype)
    R[0] = torch.eye(3, device=q.device, dtype=q.dtype).expand(B, 3, 3)
    for b in range(1, self.nbody):
      if self.parent[b] == 0:
        P[b], R[b] = root_pos, root_rot
        continue
      p = self.parent[b]
      Pb = P[p] + (R[p] @ self.body_pos[b].unsqueeze(-1)).squeeze(-1)
      Rb = R[p] @ self.body_rot[b]
      j = self.body_joint[b]
      if j >= 0:
        Rj = axis_angle_to_mat(self.joint_axis[j].expand(B, 3), q[:, j])
        jp = self.joint_pos[j]
        # Rotation about an axis through jnt_pos (in body frame).
        Pb = Pb + (Rb @ (jp - (Rj @ jp.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        Rb = Rb @ Rj
      P[b], R[b] = Pb, Rb
    return P, R

  def com(self, P, R):
    c = 0.0
    for b in range(1, self.nbody):
      c = c + self.mass[b] * (P[b] + (R[b] @ self.ipos[b].unsqueeze(-1)).squeeze(-1))
    return c / self.total_mass

  def site(self, P, R, name):
    i = self.sidx(name)
    b = self.site_body[i]
    return P[b] + (R[b] @ self.site_pos[i].unsqueeze(-1)).squeeze(-1), R[b] @ self.site_rot[i]

  def body(self, P, R, name):
    i = self.body_names.index(name)
    return P[i], R[i]
