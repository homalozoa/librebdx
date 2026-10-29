"""Look-dev stills: blender -b -P blender/lookdev.py -- OUTPREFIX [night] [frame] [az1,az2,..] [dist] [lens]"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bdx_lib as L  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1 :]
out = argv[0]
night = argv[1] == "night"
frame = int(argv[2]) if len(argv) > 2 else 40
azs = [float(a) for a in argv[3].split(",")] if len(argv) > 3 else [0.0]
dist = float(argv[4]) if len(argv) > 4 else 1.8
lens = float(argv[5]) if len(argv) > 5 else 50.0
clipname = argv[6] if len(argv) > 6 else "standing_commands"
tz = float(argv[7]) if len(argv) > 7 else 0.08  # camera target height above the root

L.reset_scene()
L.desert_world(night)
mats = L.make_materials()
objs, _ = L.build_robot(mats)  # _ = per-object mesh names
clip = L.load_clip(clipname, 24)
off, yaw = L.placement(clip, (0, 0), 0.0, frame)
frames, root = L.animate(objs, clip, offset=off, yaw=yaw)
eyes = L.add_eyes(objs, _, mats, frame)
L.animate_eyes(eyes, [frame])
L.bevel_all(mats)
cam = L.camera(lens=lens, fstop=4.0)
L.render_settings(1280, 720, samples=32)
import bpy  # noqa: E402

bpy.context.scene.frame_set(frame)
r = root[frame] + np.array([0, 0, tz])
for az in azs:
  a = math.radians(az)
  e = r + dist * np.array([math.cos(a), math.sin(a), 0.12])
  cam.location = e
  from mathutils import Vector  # noqa: E402

  cam.rotation_mode = "QUATERNION"
  cam.rotation_quaternion = (Vector(r) - Vector(e)).to_track_quat("-Z", "Y")
  cam.data.dof.focus_distance = dist
  bpy.context.scene.render.filepath = f"{out}_{int(az)}.png"
  bpy.ops.render.render(write_still=True)
