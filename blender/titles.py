"""Opening crawl and end card for the BDX film.

blender -b -P blender/titles.py -- (intro|title|crawl|end) [--preview]
"""

import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
import bdx_lib as L  # noqa: E402

FPS = 24
import os  # noqa: E402

LANG = os.environ.get("BDX_LANG", "en")
OUT = L.ROOT / "results" / "bdx" / (("film2" if LANG == "en" else f"film2_{LANG}") if os.environ.get("BDX_CUT") == "2" else "film")
ZH_H = os.path.expanduser("~/Library/Fonts/MiSans-Heavy.ttf")
ZH_B = os.path.expanduser("~/Library/Fonts/MiSans-Bold.ttf")
ZH_R = os.path.expanduser("~/Library/Fonts/MiSans-Medium.ttf")
ZH_PUNCT = "，。、；：？！）」』…"


def zh_wrap(text, n):
  """Break Chinese text into lines of about n characters (no punctuation at a line start)."""
  lines, cur = [], ""
  for ch in text:
    if len(cur) >= n and ch not in ZH_PUNCT:
      lines.append(cur)
      cur = ""
    cur += ch
  if cur:
    lines.append(cur)
  return "\n".join(lines)
FONT = "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"
FONT_C = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"
GOLD = "#FFD84A"
BLUE = "#55C8F0"

CRAWL = [
  ("EPISODE R", 0.4, "CENTER"),
  ("A NEW GAIT", 0.78, "CENTER"),
  ("", 0.5, "CENTER"),
  ("In the quiet of a simulator, thousands of copies of one small droid fall, reset and try again, "
   "all at the same time. From their mistakes a single policy is distilled: a way of walking.", 0.37, "JUSTIFY"),
  ("", 0.5, "CENTER"),
  ("Its gait was drawn by animators. One reward keeps it close to their drawing, and on its feet "
   "through pushes, slips and uneven ground.", 0.37, "JUSTIFY"),
  ("", 0.5, "CENTER"),
  ("Now the policy is given a body, and a desert under two suns. Somewhere past the dunes, "
   "a relay has begun to call....", 0.37, "JUSTIFY"),
]


CRAWL_ZH = [
  ("第 R 集", 0.4, "CENTER"),
  ("新 步 态", 0.78, "CENTER"),
  ("", 0.5, "CENTER"),
  ("在模拟器的寂静里，同一个小机器人的成千上万个分身同时跌倒、复位、再试一次。从这些失误中，提炼出唯一一个策略：一种走路的方式。", 0.36, "ZH"),
  ("", 0.5, "CENTER"),
  ("它的步态出自动画师之手。一条奖励让它贴近那份手稿，也让它在推搡、打滑与起伏的地面上站稳。", 0.36, "ZH"),
  ("", 0.5, "CENTER"),
  ("如今，这个策略有了身体，也有了一片双日照耀的沙漠。沙丘之外，一座中继塔已经开始呼唤……", 0.36, "ZH"),
]
if LANG == "zh":
  CRAWL = CRAWL_ZH


def base(seconds):
  L.reset_scene()
  sc = bpy.context.scene
  L._world_gradient(zenith="#000000", horizon="#000000", ground="#000000", strength=1.0, stars=14.0)
  w = bpy.context.scene.world.node_tree
  for nd in w.nodes:  # stars everywhere, not only above a horizon
    if nd.type == "MAP_RANGE" and abs(nd.inputs["From Min"].default_value - 0.51) < 1e-6:
      nd.inputs["From Min"].default_value, nd.inputs["From Max"].default_value = -1.0, -0.5
  sc.frame_start, sc.frame_end = 1, int(seconds * FPS)
  cam = L.camera(lens=35, fstop=22)
  cam.data.dof.use_dof = False
  return sc, cam


def emit(color, strength=2.0, name="txt"):
  m = bpy.data.materials.new(name)
  m.use_nodes = True
  nt = m.node_tree
  nt.nodes.clear()
  em = nt.nodes.new("ShaderNodeEmission")
  em.inputs["Color"].default_value = L.hexrgb(color)
  em.inputs["Strength"].default_value = strength
  tr = nt.nodes.new("ShaderNodeBsdfTransparent")
  mix = nt.nodes.new("ShaderNodeMixShader")
  mix.inputs["Fac"].default_value = 1.0
  o = nt.nodes.new("ShaderNodeOutputMaterial")
  nt.links.new(tr.outputs[0], mix.inputs[1])
  nt.links.new(em.outputs[0], mix.inputs[2])
  nt.links.new(mix.outputs[0], o.inputs[0])
  return m, mix.inputs["Fac"]


def text(body, size, align="CENTER", font=FONT, width=None, loc=(0, 0, 0), mat=None):
  cu = bpy.data.curves.new("t", "FONT")
  cu.body = body
  cu.font = bpy.data.fonts.load(font, check_existing=True)
  cu.size = size
  cu.align_x = align
  cu.align_y = "TOP"
  if width:
    cu.text_boxes[0].width = width
    cu.text_boxes[0].x = -width / 2
  cu.space_line = 1.15
  ob = bpy.data.objects.new("t", cu)
  ob.location = loc
  bpy.context.scene.collection.objects.link(ob)
  if mat:
    ob.data.materials.append(mat)
  return ob


def fade(fac, keys):
  for f, v in keys:
    fac.default_value = v
    fac.keyframe_insert("default_value", frame=f)


def intro():
  sc, cam = base(5.0)
  cam.location = (0, -10, 0)
  cam.rotation_euler = (math.radians(90), 0, 0)
  m, fac = emit(BLUE, 1.6)
  if LANG == "zh":
    t = text("不久以前，在一个遥远遥远的\n模拟器里……", 0.42, "LEFT", ZH_R, loc=(-3.6, 0, 0.5), mat=m)
  else:
    t = text("Not so long ago, in a simulator far,\nfar away....", 0.42, "LEFT", FONT, loc=(-3.6, 0, 0.5), mat=m)
  t.rotation_euler = (math.radians(90), 0, 0)
  fade(fac, [(1, 0.0), (20, 1.0), (96, 1.0), (118, 0.0)])


def title():
  sc, cam = base(7.0)
  cam.location = (0, -10, 0)
  cam.rotation_euler = (math.radians(90), 0, 0)
  m, fac = emit(GOLD, 3.0)
  t = text("BDX", 4.2, "CENTER", FONT_C, mat=m)
  t.data.align_y = "CENTER"
  t.data.extrude = 0.05
  t.rotation_euler = (math.radians(90), 0, 0)
  # Exponential recession: apparent size shrinks at a steady rate (fills the frame, then
  # drifts back to about 1/12 of its size over 7 s) instead of collapsing in the first second.
  d0, d1 = 7.6, 80.0
  for f in range(1, sc.frame_end + 1, 4):
    u = (f - 1) / (sc.frame_end - 1)
    t.location = (0, -10 + d0 * (d1 / d0) ** u, 0)
    t.keyframe_insert("location", frame=f)
  fade(fac, [(1, 1.0), (sc.frame_end - 30, 1.0), (sc.frame_end, 0.0)])


def _fcs(ob):
  act = ob.animation_data.action
  if hasattr(act, "layers") and act.layers:
    return act.layers[0].strips[0].channelbag(ob.animation_data.action_slot).fcurves
  return act.fcurves


def crawl():
  sc, cam = base(24.0)
  cam.location = (0, -7.5, 1.9)
  cam.rotation_euler = (math.radians(76), 0, 0)
  cam.data.lens = 42
  m, fac = emit(GOLD, 2.2)
  par = bpy.data.objects.new("crawl", None)
  sc.collection.objects.link(par)
  y = 0.0
  for body, size, align in CRAWL:
    if not body:
      y -= size
      continue
    if align == "ZH":
      t = text(zh_wrap(body, 12), size, "CENTER", ZH_B, loc=(0, y, 0), mat=m)
      t.data.space_line = 1.3
    else:
      t = text(body, size, align, ZH_H if LANG == "zh" else FONT_C, width=4.3, loc=(0, y, 0), mat=m)
    t.parent = par
    bpy.context.view_layer.update()
    y -= t.dimensions.y + 0.12  # measured height of the wrapped paragraph
  par.location = (0, -6.3, 0)
  par.keyframe_insert("location", frame=1)
  par.location = (0, 11.5, 0)
  par.keyframe_insert("location", frame=sc.frame_end)
  for fc in _fcs(par):
    for kp in fc.keyframe_points:
      kp.interpolation = "LINEAR"
  fade(fac, [(1, 1.0), (sc.frame_end - 36, 1.0), (sc.frame_end, 0.0)])


END = [
  ("A NEW GAIT", 0.62, GOLD, FONT_C),
  ("Every step in this film was taken by a learned policy.", 0.24, "#F1EBDD", FONT),
  ("", 0.3, None, None),
  ("STANDING 0.029 rad    WALKING 0.046 rad    JUMP 0.027 rad    0 FALLS IN 15,360 s PER POLICY", 0.22, BLUE, FONT_C),
  ("", 0.2, None, None),
  ("A reproduction of Design and Control of a Bipedal Robotic Character   /   Grandia et al., RSS 2024", 0.17, "#B9B3A5", FONT),
  ("Policies: PPO in MuJoCo Warp and mjlab   /   Robot: open-source BDX-R   /   Paint: Meshy   /   Rendering: Blender Cycles", 0.15, "#8F8A7E", FONT),
  ("Sand: desert field recordings by felix.blume and others on Freesound (CC0)   /   Score synthesized from the rollouts", 0.15, "#8F8A7E", FONT),
  ("", 0.3, None, None),
  ("MAY THE FORCE BE WITH YOU", 0.3, GOLD, FONT_C),
]
END_ZH = [
  ("新 步 态", 0.62, GOLD, ZH_H),
  ("这部影片里的每一步，都由学习得到的策略迈出。", 0.24, "#F1EBDD", ZH_R),
  ("", 0.3, None, None),
  ("站立 0.029 rad    行走 0.046 rad    跳跃 0.027 rad    每个策略 15,360 秒零摔倒", 0.22, BLUE, ZH_B),
  ("", 0.2, None, None),
  ("复现论文：Design and Control of a Bipedal Robotic Character   /   Grandia 等，RSS 2024", 0.17, "#B9B3A5", ZH_R),
  ("策略：PPO · MuJoCo Warp · mjlab   /   机器人：开源 BDX-R   /   涂装：Meshy   /   渲染：Blender Cycles", 0.15, "#8F8A7E", ZH_R),
  ("沙地声音：Freesound 上 felix.blume 等人的沙漠实录（CC0）   /   配乐由仿真数据合成", 0.15, "#8F8A7E", ZH_R),
  ("", 0.3, None, None),
  ("愿 原 力 与 你 同 在", 0.3, GOLD, ZH_H),
]
if LANG == "zh":
  END = END_ZH


def end():
  sc, cam = base(9.0)
  cam.location = (0, -10, 0)
  cam.rotation_euler = (math.radians(90), 0, 0)
  z = 1.7
  for i, (body, size, color, font) in enumerate(END):
    if not body:
      z -= size
      continue
    m, fac = emit(color, 2.0 if color == GOLD else 1.4, f"end{i}")
    t = text(body, size, "CENTER", font, loc=(0, 0, z), mat=m)
    t.rotation_euler = (math.radians(90), 0, 0)
    d = 6 * i
    fade(fac, [(1 + d, 0.0), (18 + d, 1.0), (sc.frame_end - 20, 1.0), (sc.frame_end, 0.0)])
    z -= size * 1.6


if __name__ == "__main__":
  argv = sys.argv[sys.argv.index("--") + 1 :]
  name = argv[0]
  {"intro": intro, "title": title, "crawl": crawl, "end": end}[name]()
  preview = "--preview" in argv
  L.render_settings(960 if preview else 1920, 540 if preview else 1080, FPS, samples=8 if preview else 24, motion_blur=False)
  sc = bpy.context.scene
  sc.view_settings.view_transform = "Standard"
  if "--step" in argv:
    sc.frame_step = int(argv[argv.index("--step") + 1])
  sc.render.filepath = str(OUT / (name + ("_preview" if preview else ""))) + "/"
  bpy.ops.render.render(animation=True)
