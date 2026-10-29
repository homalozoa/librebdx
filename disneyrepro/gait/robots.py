"""Robot setups for the model-based gait stack."""

from pathlib import Path

from disneyrepro.gait.planner import FootGeom
from disneyrepro.gait.sim import RobotGaitSetup

ROBOTS = Path(__file__).parents[1] / "robots"


def dino():
  from disneyrepro.robots.dino.build import FOOT

  hx, hy, _ = FOOT["half"]
  return RobotGaitSetup(
    xml=str(ROBOTS / "dino" / "dino.xml"),
    foot=FootGeom(heel_x=-hx, toe_x=hx, half_width=hy),
    foot_sites={"l": "left_sole", "r": "right_sole"},
    upper_joints=("neck_pitch1", "neck_yaw", "neck_pitch2", "head_pitch", "tail_yaw1", "tail_pitch2", "tail_yaw3", "tail_pitch4"),
  )
