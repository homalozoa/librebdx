"""Assemble the second cut, BDX: A New Gait (the story).

python3 blender/film_cut2.py
Shots arrive from the GPU renderer as near-lossless MP4s in results/bdx/film2/shots/<shot>.mp4;
titles are local PNG frames in results/bdx/film2/<title>/. Each story shot gets a
three-line lower third: chapter, a line of verse, and the measured fact behind it.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILM = ROOT / "results" / "bdx" / "film2"
FPS = 24
FF = "/opt/homebrew/bin/ffmpeg"
PY = "/Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13"
F_TITLE = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"
F_VERSE = "/System/Library/Fonts/Avenir Next.ttc"
F_DATA = "/System/Library/Fonts/Menlo.ttc"

# (shot, transition in, seconds, source clip, clip t0, speed)
CUT = [
  ("intro", "fade", 0.0, None, 0, 1),
  ("title", "fade", 0.6, None, 0, 1),
  ("crawl", "fade", 0.6, None, 0, 1),
  ("waking", "fade", 1.4, "story_wake", 2.0, 1),
  ("listening", "wipeleft", 0.6, "show_animation_engine", 5.5, 1),
  ("crossing", "wiperight", 0.6, "story_crossing2", 2.0, 1),
  ("wind", "wipeleft", 0.6, "story_wind2", 2.2, 1),
  ("leap", "circleopen", 0.7, "episodic_jump", 0.0, 0.4),
  ("arrival", "wiperight", 0.6, "story_arrival", 0.0, 1),
  ("nightfall", "radial", 1.0, "episodic_tantrum", 0.0, 1),
  ("answer", "wipeleft", 0.6, "story_answer", 0.2, 1),
  ("reunion", "circleopen", 0.6, "story_reunion", 2.0, 1),
  ("together", "wiperight", 0.6, "story_gaze_up", 2.0, 1),
  ("end", "fade", 1.2, None, 0, 1),
]
# Chapter cards: the titles tip their hat to the saga; verse and measured fact underneath.
CARDS = {
  "en": {
    "waking": ('I  ·  AWAKENING', 'Two suns stand low. Its eyes flicker on, and it tries its weight on the sand.', 'standing policy  ·  8 command channels  ·  joint error 0.029 rad'),
    "listening": ('II  ·  THE SIGNAL', 'Beyond the dunes a red lamp blinks every two seconds. It turns its head to listen.', 'animation engine  ·  nod, gaze and posture over the standing policy'),
    "crossing": ('III  ·  THE CROSSING', 'It walks toward the light, carefully at first, then with more conviction. The sand keeps every step.', 'calm-gait walking policy  ·  0.15 to 0.4 m/s  ·  about 2 to 3 steps/s'),
    "wind": ('IV  ·  AGAINST THE WIND', 'Gusts shove it sideways. It catches itself and walks on.', '90 N gusts for 0.1 s  ·  calm-gait walking policy at 0.3 m/s'),
    "leap": ('V  ·  THE LEAP', 'Halfway there, for no reason it could name, it jumps.', 'jump policy  ·  shown at 0.4x  ·  joint error 0.027 rad'),
    "arrival": ('VI  ·  ARRIVAL', 'It stops short of the mast, settles onto both feet, and dances.', 'stand > walk > stand > dance  ·  full state handed over at each switch'),
    "nightfall": ('VII  ·  THE LONG NIGHT', 'Night falls. The lamp keeps blinking. Nothing answers, and it stamps its feet.', 'tantrum policy  ·  joint error 0.045 rad'),
    "answer": ('VIII  ·  AN ANSWER', 'Far off on a dune, a second pair of eyes comes on, and nods.', 'a second droid, 13 m away  ·  the same standing policy'),
    "reunion": ('IX  ·  A NEW HOPE', 'It hurries across the sand, stops beside the other, and cannot keep still.', 'walking at 0.7 m/s, about 4 steps/s  ·  stop  ·  excited policy'),
    "together": ('X  ·  BINARY STARS', 'Two small machines under a ringed planet, looking up.', 'head pitch command +0.5 rad  ·  every pose a policy rollout'),
  },
  "zh": {
    "waking": ('第一章 · 苏醒', '两颗太阳低悬。它的眼睛闪了几下亮起来，试着把重心交给脚下的沙。', '站立策略 · 8 个指令通道 · 关节误差 0.029 rad'),
    "listening": ('第二章 · 信号', '沙丘那头，一盏红灯每两秒闪一次。它转过头去听。', '动画引擎 · 点头、注视与姿态叠加在站立策略之上'),
    "crossing": ('第三章 · 远行', '它朝那盏灯走去，起初小心翼翼，渐渐笃定。沙地一步不落地记着。', '舒缓步态行走策略 · 0.15 至 0.4 m/s · 每秒约 2 至 3 步'),
    "wind": ('第四章 · 逆风', '阵风把它推向一侧。它稳住身子，继续往前。', '90 N 阵风持续 0.1 秒 · 舒缓步态行走策略，0.3 m/s'),
    "leap": ('第五章 · 一跃', '走到半路，它跳了一下，说不出为什么。', '跳跃策略 · 0.4 倍速 · 关节误差 0.027 rad'),
    "arrival": ('第六章 · 抵达', '它在塔前停下，两脚站稳，跳起舞来。', '站立 > 行走 > 站立 > 跳舞 · 每次切换完整交接状态'),
    "nightfall": ('第七章 · 长夜', '夜色降下来，红灯还在闪。没有回音，它跺起了脚。', '发脾气策略 · 关节误差 0.045 rad'),
    "answer": ('第八章 · 回应', '远处的沙丘上，又一双眼睛亮了，轻轻点了点头。', '第二个机器人，相距 13 米 · 同一个站立策略'),
    "reunion": ('第九章 · 新希望', '它急急地穿过沙地，在另一个身旁停下，高兴得停不下来。', '以 0.7 m/s、每秒约 4 步行走 · 停下 · 兴奋策略'),
    "together": ('第十章 · 双星', '两个小小的机器，并肩站在环星之下，抬着头。', '抬头指令 +0.5 rad · 每个姿态都来自策略运行'),
  },
}
FONTS = {
  "en": (F_TITLE, F_VERSE, F_DATA, 46, 34, 19),
  "zh": tuple(str(Path.home() / "Library/Fonts" / f) for f in ("MiSans-Heavy.ttf", "MiSans-Medium.ttf", "MiSans-Regular.ttf")) + (44, 34, 20),
}
TRIM = {"crawl": (80, 48)}  # frames dropped from head / tail


def run(cmd):
  subprocess.run(cmd, check=True)


def esc(t):
  return t.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def encode(shot, lang):
  card = CARDS[lang].get(shot)
  chapter, verse, fact = card if card else (None, None, None)
  out = FILM / f"seg_{lang}" / f"{shot}.mov"
  src = FILM / "shots" / f"{shot}.mp4"
  tdir = FILM if lang == "en" or shot == "title" else FILM.parent / f"film2_{lang}"
  if src.exists():
    inp = ["-i", str(src)]
    n = int(subprocess.run(["/opt/homebrew/bin/ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(src)],
                           capture_output=True, text=True).stdout.strip())
    a, b = TRIM.get(shot, (0, 0))
    sel = []
  else:
    frames = sorted((tdir / shot).glob("*.png"))
    a, b = TRIM.get(shot, (0, 0))
    frames = frames[a : len(frames) - b]
    inp = ["-framerate", str(FPS), "-start_number", frames[0].stem, "-i", str(tdir / shot / "%04d.png")]
    n = len(frames)
    a = b = 0
    sel = ["-frames:v", str(n)]
  vf = []
  if a or b:
    vf.append(f"trim=start_frame={a}:end_frame={n - b},setpts=PTS-STARTPTS")
    n = n - a - b
  vf.append("format=yuv444p10le")
  if chapter:
    ft, fv, fd, st, sv, sd = FONTS[lang]
    t0, t1 = 1.0, 5.2
    al = f"if(lt(t,{t0}),0,if(lt(t,{t0}+0.6),(t-{t0})/0.6,if(lt(t,{t1}),1,if(lt(t,{t1}+0.7),1-(t-{t1})/0.7,0))))"
    vf += [
      f"drawbox=x=96:y=ih-214:w=64:h=2:color=0xFFD84A@0.85:t=fill:enable='between(t,{t0},{t1}+0.7)'",
      f"drawtext=expansion=none:fontfile='{ft}':text='{esc(chapter)}':x=96:y=h-200:fontsize={st}:fontcolor=0xFFD84A:alpha='{al}'",
      f"drawtext=expansion=none:fontfile='{fv}':text='{esc(verse)}':x=97:y=h-146:fontsize={sv}:fontcolor=0xF1EBDD:alpha='{al}'",
      f"drawtext=expansion=none:fontfile='{fd}':text='{esc(fact)}':x=98:y=h-96:fontsize={sd}:fontcolor=0x9FD9D6:alpha='{al}*0.9'",
    ]
  run([FF, "-v", "error", "-y", *inp, *sel, "-vf", ",".join(vf), "-c:v", "prores_ks", "-profile:v", "3", str(out)])
  return n / FPS


def main(langs=("en", "zh")):
  for lang in langs:
    (FILM / f"seg_{lang}").mkdir(exist_ok=True)
    segs, t = [], 0.0
    for shot, trans, xf, clip, c0, speed in CUT:
      dur = encode(shot, lang)
      start = t - xf if segs else 0.0
      trim_a = TRIM.get(shot, (0, 0))[0] / FPS
      segs.append({"shot": shot, "start": start, "dur": dur, "xf": xf, "trans": trans, "clip": clip, "clip_t0": c0 + trim_a * speed, "speed": speed})
      t = start + dur
    tl = FILM / "timeline.json"
    tl.write_text(json.dumps({"fps": FPS, "length": t, "segments": segs}, indent=1))
    if lang == langs[0]:
      run([PY, str(Path(__file__).parent / "score2.py"), str(tl), str(FILM / "score.wav")])
    ins, chain, prev = [], "", "[0:v]"
    for i, sg in enumerate(segs):
      ins += ["-i", str(FILM / f"seg_{lang}" / f"{sg['shot']}.mov")]
      if i:
        chain += f"{prev}[{i}:v]xfade=transition={sg['trans']}:duration={sg['xf']}:offset={sg['start']:.3f}[v{i}];"
        prev = f"[v{i}]"
    chain += f"{prev}vignette=angle=PI/5.5:mode=forward,noise=alls=4:allf=t,format=yuv420p[vout]"
    master = FILM / f"bdx_a_new_gait_story_{lang}.mp4"
    run([FF, "-v", "error", "-y", *ins, "-i", str(FILM / "score.wav"), "-filter_complex", chain, "-map", "[vout]",
         "-map", f"{len(segs)}:a", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-tune", "grain", "-pix_fmt", "yuv420p",
         "-af", "acompressor=threshold=-22dB:ratio=3:attack=15:release=250:makeup=2,loudnorm=I=-16:TP=-1.5:LRA=9",
         "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(master)])
    web(t, master, lang)
    print("film:", master, f"{t:.1f} s", flush=True)


def web(length, master, lang):
  site = ROOT / "site" / "media"
  kbps = int(13.6 * 8192 / length) - 96
  for p in (1, 2):
    run([FF, "-v", "error", "-y", "-i", str(master), "-vf", "scale=1280:720:flags=lanczos,hqdn3d=2:2:4:4", "-c:v", "libx264",
         "-preset", "slower", "-b:v", f"{kbps}k", "-pass", str(p), "-passlogfile", str(FILM / f"x264pass_{lang}"),
         *(["-an", "-f", "mp4", "/dev/null"] if p == 1 else ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(site / f"film_{lang}.mp4")])])
  if lang != "en":
    return
  for shot, *_ in CUT:
    src = FILM / "shots" / f"{shot}.mp4"
    if not src.exists():
      continue
    run([FF, "-v", "error", "-y", "-i", str(src), "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-preset", "slower",
         "-crf", "25", "-maxrate", "2400k", "-bufsize", "4800k", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
         str(site / f"story_{shot}.mp4")])
    run([FF, "-v", "error", "-y", "-ss", "3", "-i", str(src), "-frames:v", "1", "-vf", "scale=1280:720", "-q:v", "4",
         str(site / f"story_{shot}.jpg")])


if __name__ == "__main__":
  import sys

  main(tuple(sys.argv[1:]) or ("en", "zh"))
