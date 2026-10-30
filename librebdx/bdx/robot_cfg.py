"""mjlab entity configuration for the BDX-R stand-in of Disney's bipedal character.

Actuator types follow Sec. IV / Table VI of the paper: Unitree A1 drives at hip
roll, hip pitch and knee; Unitree Go1 drives at hip yaw, ankle and the lower
neck; Dynamixel XH540 servos for the three head joints.
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from librebdx.bdx.standing_ref import XML, Nominal
from librebdx.rl.disney_actuator import DYNAMIXEL_XH540, UNITREE_A1, UNITREE_GO1, make_cfg

# The paper's actuators run at 600 Hz with a 50 Hz policy (Sec. VI-D): 12 physics
# steps per policy step, with first-order hold and the 37.5 Hz low-pass in between.
SIM_HZ = int(__import__("os").environ.get("DISNEY_SIM_HZ", "600"))
SIM_DT = 1.0 / SIM_HZ
DECIMATION = SIM_HZ // 50  # 50 Hz policy

A1_JOINTS = (".*_Hip_Roll", ".*_Hip_Pitch", ".*_Knee")
GO1_JOINTS = (".*_Hip_Yaw", ".*_Ankle", "Neck_Pitch")
DXL_JOINTS = ("Head_Pitch", "Head_Yaw", "Head_Roll")
LEG_JOINT_EXPR = (".*_Hip_.*", ".*_Knee", ".*_Ankle")
NECK_JOINT_EXPR = ("Neck_Pitch", "Head_.*")

# Output transform: a = 1 maps to the expected range per joint (App. A). We use
# the larger of the half-range of the joint in the reference data (standing
# + walking) about the nominal pose and 0.25 * tau_max / kP (setpoint offset
# needed for a quarter of the peak torque with the low-gain drives).
_REF_HALF_RANGE = {
  ".*_Hip_Yaw": 0.22, ".*_Hip_Roll": 0.24, ".*_Hip_Pitch": 0.65, ".*_Knee": 0.56, ".*_Ankle": 0.52,
  "Neck_Pitch": 0.30, "Head_Pitch": 0.76, "Head_Yaw": 0.86, "Head_Roll": 0.52,
}
_TORQUE_RANGE = {
  **{n: 0.25 * UNITREE_A1.tau_max / UNITREE_A1.kp for n in A1_JOINTS},
  **{n: 0.25 * UNITREE_GO1.tau_max / UNITREE_GO1.kp for n in GO1_JOINTS},
  **{n: 0.25 * DYNAMIXEL_XH540.tau_max / DYNAMIXEL_XH540.kp for n in DXL_JOINTS},
}
# Standing / episodic policies were trained with the first transform, walking
# with the second (a = 1 <-> the joint's half-range in the references, App. A);
# the runtime converts action histories between them on policy switches.
ACTION_SCALE_TORQUE = {n: max(_REF_HALF_RANGE[n], _TORQUE_RANGE[n]) for n in _REF_HALF_RANGE}
ACTION_SCALE_REF = {n: max(_REF_HALF_RANGE[n], 0.3) for n in _REF_HALF_RANGE}
ACTION_SCALE = ACTION_SCALE_TORQUE


# Setpoints may deviate from the measured position by enough to reach tau_max.
MAX_DEVIATION = {
  **{n: 1.1 * UNITREE_A1.tau_max / UNITREE_A1.kp for n in A1_JOINTS},
  **{n: 1.1 * UNITREE_GO1.tau_max / UNITREE_GO1.kp for n in GO1_JOINTS},
  **{n: 1.1 * DYNAMIXEL_XH540.tau_max / DYNAMIXEL_XH540.kp for n in DXL_JOINTS},
}


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(XML))


def nominal_joint_pos() -> dict[str, float]:
  """Nominal standing pose (from the standing IK at zero command)."""
  from librebdx.bdx.standing_ref import BdxStandingReference

  ref = BdxStandingReference(device="cpu")
  q = ref.x_nominal[2:].tolist()
  return dict(zip(ref.kin.joint_names, q))


def get_bdx_robot_cfg(nominal: dict[str, float]) -> EntityCfg:
  import dataclasses
  import os

  motors = [UNITREE_A1, UNITREE_GO1, DYNAMIXEL_XH540]
  if os.environ.get("DISNEY_STIFF") == "1":  # diagnostic: 4x stiffer drives
    motors = [dataclasses.replace(m, kp=4 * m.kp, kd=2 * m.kd) for m in motors]
  actuators = (
    make_cfg(A1_JOINTS, motors[0], SIM_DT),
    make_cfg(GO1_JOINTS, motors[1], SIM_DT),
    make_cfg(DXL_JOINTS, motors[2], SIM_DT),
  )
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, Nominal().torso_height + 0.01),
      joint_pos=nominal,
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.98),
  )
