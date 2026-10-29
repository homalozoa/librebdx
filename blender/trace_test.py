"""Close test of footprints + dust: blender -b -P blender/trace_test.py -- OUT.png FRAME"""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
import bdx_lib as L  # noqa: E402
import film2 as F  # noqa: E402
from traces import events, traces_all  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1 :]
F.world()
d = F.droid("story_crossing", heading=0.0, t0=2.0, t1=17.0)
traces_all([d])
L.bevel_all(d["mats"])
ev = [e for e in events(d) if e[1] == "down"]
f = int(argv[1]) if len(argv) > 1 else ev[len(ev) // 2][0] + 3
bpy.context.scene.frame_set(f)
r = d["root"][list(d["frames"]).index(f)]
cam = L.camera(lens=35, fstop=5.6)
e = Vector((r[0] - 0.9, r[1] - 1.0, 0.45))
cam.location = e
cam.rotation_mode = "QUATERNION"
cam.rotation_quaternion = (Vector((r[0] - 0.25, r[1], 0.0)) - e).to_track_quat("-Z", "Y")
cam.data.dof.focus_distance = (Vector((r[0], r[1], 0)) - e).length
L.render_settings(1280, 720, 24, samples=64, motion_blur=True)
bpy.context.scene.frame_start = bpy.context.scene.frame_end = f
bpy.context.scene.render.filepath = argv[0]
bpy.ops.render.render(write_still=True)
print("touchdowns", [e[0] for e in ev][:12])
