"""Task-space inverse-dynamics whole-body controller (Hopkins et al., Appendix ID).

Decision variables: joint accelerations qdd and non-negative friction-cone
basis weights rho (4 per contact point, Eq. 17). Constraints: floating-base
(centroidal) dynamics (Eq. 21), soft joint position limits (Eqs. 22-23),
rho >= 0 (Eq. 24), only active contacts carry force (Eq. 25), torque limits.
Objective: weighted task-acceleration errors + regularisation of qdd, rho and
the contact-induced torque (Eq. 20).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import proxsuite


def so3_log(R):
  c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
  a = np.arccos(c)
  v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  if a < 1e-6:
    return 0.5 * v
  return v * a / (2 * np.sin(a))


@dataclass
class WbcGains:
  com_kp: float = 60.0
  com_kd: float = 15.0
  pelvis_z_kp: float = 100.0
  pelvis_z_kd: float = 20.0
  pelvis_rot_kp: float = 150.0
  pelvis_rot_kd: float = 25.0
  foot_kp: float = 400.0
  foot_kd: float = 40.0
  stance_kd: float = 40.0
  stance_kp: float = 0.0
  joint_kp: float = 150.0
  joint_kd: float = 20.0
  angmom_kd: float = 5.0
  w_com: float = 10.0
  w_pelvis_z: float = 10.0
  w_pelvis_rot: float = 10.0
  w_swing: float = 10.0
  w_stance: float = 200.0
  w_joint: float = 1.0
  w_angmom: float = 0.05
  reg_qdd: float = 1e-4
  reg_rho: float = 1e-5
  reg_tau_c: float = 1e-5
  limit_k: float = 400.0
  limit_b: float = 40.0
  mu: float = 0.7
  posture_w: float = 0.02
  posture_kp: float = 20.0
  posture_kd: float = 6.0


@dataclass
class WbcRef:
  com: np.ndarray
  com_vel: np.ndarray
  com_acc: np.ndarray
  pelvis_z: float
  pelvis_zd: float
  pelvis_zdd: float
  pelvis_R: np.ndarray
  pelvis_w: np.ndarray
  foot_pos: dict
  foot_vel: dict
  foot_acc: dict
  foot_R: dict
  foot_w: dict
  contact_points: dict  # side -> list of local points (in foot site frame) carrying force
  upper_q: dict = field(default_factory=dict)
  upper_qd: dict = field(default_factory=dict)


class WholeBodyController:
  def __init__(self, model, foot_sites, gains=WbcGains(), posture=None):
    self.m = m = model
    self.g = gains
    self.nv = m.nv
    self.na = m.nv - 6
    self.foot_site = {s: m.site(n).id for s, n in foot_sites.items()}
    self.foot_body = {s: m.site_bodyid[i] for s, i in self.foot_site.items()}
    # actuated joint (dof) order = qvel[6:]; map actuators to dofs.
    self.dof_of_act = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
    self.act_of_dof = {d: a for a, d in enumerate(self.dof_of_act)}
    self.tau_max = np.zeros(self.na)
    for a in range(m.nu):
      self.tau_max[self.dof_of_act[a] - 6] = m.actuator_ctrlrange[a, 1]
    self.jnt_range = np.zeros((self.na, 2))
    for j in range(m.njnt):
      if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
        self.jnt_range[m.jnt_dofadr[j] - 6] = m.jnt_range[j]
    self.qadr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
    self.dadr = {m.joint(j).name: m.jnt_dofadr[j] for j in range(m.njnt)}
    self.posture = np.zeros(self.na) if posture is None else np.asarray(posture)
    self._prev = {}
    self._H = np.zeros((m.nv, m.nv))
    mu = gains.mu
    self.basis = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]], float)
    self.basis /= np.linalg.norm(self.basis, axis=1, keepdims=True)
    self.last = {}

  # ------------------------------------------------------------- jacobians
  def _jac(self, d, point, body):
    jp = np.zeros((3, self.nv))
    jr = np.zeros((3, self.nv))
    mujoco.mj_jac(self.m, d, jp, jr, point, body)
    return jp, jr

  def _jacdot(self, d, point, body):
    jp = np.zeros((3, self.nv))
    jr = np.zeros((3, self.nv))
    mujoco.mj_jacDot(self.m, d, jp, jr, point, body)
    return jp, jr

  def _fd(self, key, J, dt):
    prev = self._prev.get(key)
    self._prev[key] = J.copy()
    if prev is None or dt <= 0:
      return np.zeros_like(J)
    return (J - prev) / dt

  # ---------------------------------------------------------------- solve
  def compute(self, d, ref: WbcRef, dt):
    m, g, nv, na = self.m, self.g, self.nv, self.na
    mujoco.mj_fullM(m, d, self._H)
    H = self._H
    b = d.qfrc_bias.copy()
    qd = d.qvel.copy()

    tasks = []  # (J, desired J qdd, weight)

    # CoM (horizontal) and angular momentum.
    Jc = np.zeros((3, nv))
    mujoco.mj_jacSubtreeCom(m, d, Jc, 1)
    Jcd = self._fd("com", Jc, dt)
    com = d.subtree_com[1].copy()
    comv = Jc @ qd
    a_com = ref.com_acc + g.com_kp * (ref.com - com[:2]) + g.com_kd * (ref.com_vel - comv[:2])
    tasks.append((Jc[:2], a_com - (Jcd @ qd)[:2], g.w_com))
    Ak = np.zeros((3, nv))
    mujoco.mj_angmomMat(m, d, Ak, 1)
    Akd = self._fd("angmom", Ak, dt)
    k = Ak @ qd
    tasks.append((Ak, -g.angmom_kd * k - Akd @ qd, g.w_angmom))

    # Pelvis height and orientation (root body).
    pb = 1
    jp, jr = self._jac(d, d.xpos[pb], pb)
    jpd, jrd = self._jacdot(d, d.xpos[pb], pb)
    z, zd = d.xpos[pb][2], (jp @ qd)[2]
    az = ref.pelvis_zdd + g.pelvis_z_kp * (ref.pelvis_z - z) + g.pelvis_z_kd * (ref.pelvis_zd - zd)
    tasks.append((jp[2:3], np.array([az - (jpd @ qd)[2]]), g.w_pelvis_z))
    R = d.xmat[pb].reshape(3, 3)
    w = jr @ qd
    aw = g.pelvis_rot_kp * so3_log(ref.pelvis_R @ R.T) + g.pelvis_rot_kd * (ref.pelvis_w - w)
    tasks.append((jr, aw - jrd @ qd, g.w_pelvis_rot))

    # Feet.
    contacts = []  # (J_point (3,nv)) for force-carrying points
    for side, sid in self.foot_site.items():
      body = self.foot_body[side]
      p = d.site_xpos[sid].copy()
      Rf = d.site_xmat[sid].reshape(3, 3)
      jp, jr = self._jac(d, p, body)
      jpd, jrd = self._jacdot(d, p, body)
      v, wf = jp @ qd, jr @ qd
      pts = ref.contact_points.get(side, [])
      if pts:
        # Stance: track the planned (possibly rolling) foot pose softly, damp velocities.
        ap = g.stance_kd * (ref.foot_vel[side] - v) + g.stance_kp * (ref.foot_pos[side] - p)
        ar = g.stance_kd * (ref.foot_w[side] - wf) + g.stance_kp * so3_log(ref.foot_R[side] @ Rf.T)
        wgt = g.w_stance
        for lp in pts:
          pw = p + Rf @ lp
          jcp, _ = self._jac(d, pw, body)
          contacts.append(jcp)
      else:
        ap = ref.foot_acc[side] + g.foot_kp * (ref.foot_pos[side] - p) + g.foot_kd * (ref.foot_vel[side] - v)
        ar = g.foot_kp * so3_log(ref.foot_R[side] @ Rf.T) + g.foot_kd * (ref.foot_w[side] - wf)
        wgt = g.w_swing
      tasks.append((jp, ap - jpd @ qd, wgt))
      tasks.append((jr, ar - jrd @ qd, wgt))

    # Upper-body joints (joint space) and a weak posture task for all joints.
    for jn, qr in ref.upper_q.items():
      da, qa = self.dadr[jn], self.qadr[jn]
      J = np.zeros((1, nv))
      J[0, da] = 1.0
      qdr = ref.upper_qd.get(jn, 0.0)
      tasks.append((J, np.array([g.joint_kp * (qr - d.qpos[qa]) + g.joint_kd * (qdr - qd[da])]), g.w_joint))
    Jpost = np.zeros((na, nv))
    Jpost[:, 6:] = np.eye(na)
    qj = d.qpos[7:]
    tasks.append((Jpost, g.posture_kp * (self.posture - qj) - g.posture_kd * qd[6:], g.posture_w))

    # ------------------------------------------------------------- QP build
    nc = len(contacts)
    nr = 4 * nc
    nx = nv + nr
    Q = np.zeros((nx, nx))
    c = np.zeros(nx)
    for J, u, wgt in tasks:
      Q[:nv, :nv] += wgt * J.T @ J
      c[:nv] -= wgt * J.T @ u
    Q[:nv, :nv] += g.reg_qdd * np.eye(nv)
    # Contact force map: f_i = basis^T rho_i (world frame, z up); tau_c = sum J_i^T f_i.
    JcT_B = np.zeros((nv, nr))
    for i, Jp in enumerate(contacts):
      JcT_B[:, 4 * i : 4 * i + 4] = Jp.T @ self.basis.T
    Q[nv:, nv:] += g.reg_rho * np.eye(nr) + g.reg_tau_c * JcT_B[6:].T @ JcT_B[6:]
    # Floating-base dynamics: H_fb qdd + b_fb - JcT_B_fb rho = 0.
    Aeq = np.hstack([H[:6], -JcT_B[:6]])
    beq = -b[:6]
    # Inequalities: joint-limit accelerations, torque limits, rho >= 0.
    qj, qdj = d.qpos[7:], qd[6:]
    lo_acc = g.limit_k * (self.jnt_range[:, 0] - qj) - g.limit_b * qdj
    hi_acc = g.limit_k * (self.jnt_range[:, 1] - qj) - g.limit_b * qdj
    Cin = np.zeros((na + na + nr, nx))
    lin = np.zeros(na + na + nr)
    uin = np.zeros(na + na + nr)
    Cin[:na, 6:nv] = np.eye(na)
    lin[:na] = np.minimum(lo_acc, hi_acc)
    uin[:na] = np.maximum(lo_acc, hi_acc)
    # tau = H_a qdd + b_a - JcT_B_a rho within limits.
    Cin[na : 2 * na, :nv] = H[6:]
    Cin[na : 2 * na, nv:] = -JcT_B[6:]
    lin[na : 2 * na] = -self.tau_max - b[6:]
    uin[na : 2 * na] = self.tau_max - b[6:]
    Cin[2 * na :, nv:] = np.eye(nr)
    lin[2 * na :] = 0.0
    uin[2 * na :] = 1e6
    qp = proxsuite.proxqp.dense.QP(nx, 6, Cin.shape[0])
    qp.settings.eps_abs = 1e-6
    qp.settings.verbose = False
    qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
    qp.init(Q, c, Aeq, beq, Cin, lin, uin)
    qp.solve()
    x = qp.results.x
    qdd, rho = x[:nv], x[nv:]
    tau_full = H @ qdd + b - JcT_B @ rho
    tau = tau_full[6:]
    self.last = {"qdd": qdd, "rho": rho, "status": qp.results.info.status, "nc": nc, "com": com, "comv": comv}
    return qdd, tau
