"""Retexture BDX shell parts with Meshy (needs MESHY_API_KEY).

python3 blender/meshy_client.py [PART ...]     (default: all exported parts)
Writes results/bdx/meshy/out/<part>.glb and a task log; skips finished parts.
"""

import base64
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# MESHY_SET=v2: 4K PBR pass (cleaner BD-1-inspired shell + machined mechanical parts).
V2 = os.environ.get("MESHY_SET") == "v2"
ROOT = Path(__file__).resolve().parents[1] / "results" / "bdx" / ("meshy2" if V2 else "meshy")
API = "https://api.meshy.ai/openapi/v1/retexture"
KEY = os.environ["MESHY_API_KEY"]

# "Dune Scout" livery: one prompt per paint role, shared weathering language.
WEAR = (
  "satin enamel paint on a small film droid, crisp recessed panel lines, a few tiny stencilled maintenance "
  "markings and serial numbers in dark grey, fine desert sand dust settled in crevices, light scuffs and "
  "chipped paint on edges revealing bare brushed aluminium, believable used-universe weathering, clean and "
  "readable, physically based, no logos, no large text"
)
STYLE = {
  "ivory": f"Warm ivory bone-white (#E7DECB) {WEAR}",
  "ember": f"Solid burnt ember orange (#C9532A) paint covering the whole outer surface, one thin ivory pinstripe, no grey, no blue, {WEAR}",
  "slate": f"Dark slate blue-grey (#3B4754) {WEAR}",
}
ROLE = {
  "Face": "ivory", "Front_Cover": "ivory", "Rear_Cover": "ivory", "Bottom_Cover": "ivory", "Head_Base_Plate": "ivory",
  "Left_Top_Cover": "ivory", "Right_Top_Cover": "ivory", "Left_Lower_Leg_Cover": "ivory", "Right_Lower_Leg_Cover": "ivory",
  "Left_Upper_Leg_Cover": "ember", "Right_Upper_Leg_Cover": "ember", "Left_Front_Cover": "ember", "Right_Front_Cover": "ember",
  "Rear_Head_Cover": "slate", "Rear_Head_Plate": "slate", "Rear_Antenna_Box": "slate",
  "Left_Foot_Cover": "slate", "Right_Foot_Cover": "slate", "Left_Ankle_Cover": "slate", "Right_Ankle_Cover": "slate",
}


FINISH = (
  "fine crisp recessed panel lines, a few tiny technical decals and stencilled part numbers, delicate scuffs "
  "only on the outermost edges, otherwise a clean, cared-for finish, physically based, no logos, no large text"
)
MECH = (
  "dark gunmetal anodised aluminium robot actuator and frame, precisely machined faces, fine brushed grain, "
  "hex socket bolts, engraved part numbers, thin cable channels, faint edge wear revealing bright aluminium, "
  "physically based, no logos, no large text"
)
if V2:
  STYLE = {
    "ivory": f"Clean warm white enamel droid armour (#ECE6D8), satin finish, {FINISH}",
    "ember": f"Solid orange-red enamel droid armour (#D2502A) covering the whole outer surface, one thin white pinstripe, no grey, no blue, {FINISH}",
    "slate": f"Warm mid-grey painted droid armour (#7F817D), satin finish, {FINISH}",
    "mech": MECH,
  }
  MECH_SRC = ["Left_Hip_Yaw_Motor", "Left_Knee_Motor", "Left_Hip_Roll_Motor", "Left_Ankle_Motor", "Neck_Motor",
              "Head_Pitch_Motor", "Left_Upper_Leg", "Left_Lower_Leg", "Left_Foot", "Left_Holder", "Left_Hip_Top_Holder",
              "Left_Hip_Yaw_Support", "Front_Support", "Neck", "Left_Head_Support", "Head_Pitch_Motor_Support"]
  ROLE = {**ROLE, **{m: "mech" for m in MECH_SRC}}


def call(url, body=None):
  req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
  req.add_header("Authorization", f"Bearer {KEY}")
  req.add_header("Content-Type", "application/json")
  with urllib.request.urlopen(req, timeout=120) as r:
    return json.loads(r.read())


def retexture(part):
  out = ROOT / "out" / f"{part}.glb"
  if out.exists():
    return part, "cached"
  obj = (ROOT / "in" / f"{part}.obj").read_bytes()
  body = {
    "model_url": "data:application/octet-stream;base64," + base64.b64encode(obj).decode(),
    "text_style_prompt": STYLE[ROLE[part]],
    "ai_model": "meshy-6",
    "enable_original_uv": False,
    "enable_pbr": V2,
    "remove_lighting": True,
    "texture_resolution": "4k" if V2 else "2k",
    "target_formats": ["glb"],
  }
  tid = call(API, body)["result"]
  while True:
    t = call(f"{API}/{tid}")
    if t["status"] in ("SUCCEEDED", "FAILED", "CANCELED"):
      break
    time.sleep(10)
  log = ROOT / "out" / f"{part}.json"
  log.write_text(json.dumps({k: t.get(k) for k in ("id", "status", "consumed_credits", "task_error", "thumbnail_url")}, indent=1))
  if t["status"] != "SUCCEEDED":
    return part, f"{t['status']}: {t.get('task_error')}"
  urllib.request.urlretrieve(t["model_urls"]["glb"], out)
  if V2:  # keep the PBR maps next to the GLB (the GLB embeds them too)
    for k, u in (t.get("texture_urls") or [{}])[0].items():
      if isinstance(u, str) and u.startswith("http"):
        urllib.request.urlretrieve(u, ROOT / "out" / f"{part}_{k}.png")
  if t.get("thumbnail_url"):
    urllib.request.urlretrieve(t["thumbnail_url"], ROOT / "out" / f"{part}_thumb.png")
  return part, f"ok ({t.get('consumed_credits')} credits)"


if __name__ == "__main__":
  (ROOT / "out").mkdir(parents=True, exist_ok=True)
  parts = sys.argv[1:] or sorted(p.stem for p in (ROOT / "in").glob("*.obj"))
  with ThreadPoolExecutor(6) as ex:
    for part, msg in ex.map(retexture, parts):
      print(part, msg, flush=True)
