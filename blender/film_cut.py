"""Assemble BDX: A NEW GAIT from rendered frames.

python3 blender/film_cut.py
  1. encodes every shot (lower-third caption burned in),
  2. chains them with crossfades,
  3. writes timeline.json and scores it (blender/score.py, numpy via Blender's Python),
  4. grades (vignette + grain) and muxes -> results/bdx/film/bdx_a_new_gait.mp4
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILM = ROOT / "results" / "bdx" / "film"
FPS = 24
FF = "/opt/homebrew/bin/ffmpeg"
PY = "/Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13"
FONT = "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"
FONT_C = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"

# (shot, crossfade into it [s], caption title, caption detail, source clip, clip t0, speed)
CUT = [
  ("intro", 0.0, None, None, None, 0, 1),
  ("title", 0.6, None, None, None, 0, 1),
  ("crawl", 0.6, None, None, None, 0, 1),
  ("establish", 1.2, "STANDING", "perpetual policy   /   joint MAE 0.029 rad", "show_animation_engine", 0.0, 1),
  ("engine", 0.12, "ANIMATION ENGINE", "triggered nod   /   gaze and posture from the sticks", "show_animation_engine", 5.5, 1),
  ("walk", 0.12, "WALKING", "0.7 m/s forward   /   0.4 m/s strafe   /   1 rad/s turn", "walking_commands", 1.0, 1),
  ("push", 0.12, "FORCE PUSH", "90 N for 0.1 s, three directions   /   no fall", "walking_push", 2.2, 1),
  ("transitions", 0.12, "POLICY SWITCHING", "stand  >  walk  >  stand  >  dance  >  stand", "transitions", 0.0, 1),
  ("jump", 0.12, "JUMP", "ballistic reference   /   0.4x slow motion", "episodic_jump", 0.0, 0.4),
  ("dance", 0.8, "HAPPY DANCE", "episodic policy", "episodic_happy_dance", 0.0, 1),
  ("excited", 0.12, "EXCITED", "episodic policy", "episodic_excited", 0.0, 1),
  ("tantrum", 0.12, "TANTRUM", "episodic policy", "episodic_tantrum", 0.0, 1),
  ("end", 1.0, None, None, None, 0, 1),
]


# Frames dropped from the head / tail of a shot (dead air before the crawl, its long fade-out).
TRIM = {"crawl": (60, 48)}


def run(cmd):
  subprocess.run(cmd, check=True)


def esc(t):
  return t.replace(":", r"\:").replace("'", r"\'")


def encode(shot, title, detail):
  frames = sorted((FILM / shot).glob("*.png"))
  a, b = TRIM.get(shot, (0, 0))
  frames = frames[a : len(frames) - b]
  out = FILM / "seg" / f"{shot}.mov"
  vf = ["format=yuv444p10le"]
  if title:
    t0, t1 = 0.8, 4.6
    a = f"if(lt(t,{t0}),0,if(lt(t,{t0}+0.5),(t-{t0})/0.5,if(lt(t,{t1}),1,if(lt(t,{t1}+0.6),1-(t-{t1})/0.6,0))))"
    vf += [
      f"drawbox=x=96:y=ih-188:w=56:h=3:color=0xFFD84A@0.9:t=fill:enable='between(t,{t0},{t1}+0.6)'",
      f"drawtext=fontfile='{FONT_C}':text='{esc(title)}':x=96:y=h-172:fontsize=58:fontcolor=0xFFD84A:alpha='{a}'",
      f"drawtext=fontfile='{FONT}':text='{esc(detail)}':x=98:y=h-104:fontsize=26:fontcolor=0xEDE6D6:alpha='{a}'",
    ]
  run([FF, "-v", "error", "-y", "-framerate", str(FPS), "-start_number", frames[0].stem, "-i", str(FILM / shot / "%04d.png"),
       "-frames:v", str(len(frames)), "-vf", ",".join(vf), "-c:v", "prores_ks", "-profile:v", "3", str(out)])
  return len(frames) / FPS


def main():
  (FILM / "seg").mkdir(exist_ok=True)
  segs, t = [], 0.0
  for shot, xf, title, detail, clip, c0, speed in CUT:
    dur = encode(shot, title, detail)
    start = t - xf if segs else 0.0
    segs.append({"shot": shot, "start": start, "dur": dur, "xf": xf, "clip": clip, "clip_t0": c0, "speed": speed})
    t = start + dur
  (FILM / "timeline.json").write_text(json.dumps({"fps": FPS, "length": t, "segments": segs}, indent=1))
  # Video: xfade chain.
  ins, chain, prev = [], "", "[0:v]"
  for i, s in enumerate(segs):
    ins += ["-i", str(FILM / "seg" / f"{s['shot']}.mov")]
    if i == 0:
      continue
    lab = f"[v{i}]"
    chain += f"{prev}[{i}:v]xfade=transition=fade:duration={s['xf']}:offset={s['start']:.3f}{lab};"
    prev = lab
  grade = "vignette=angle=PI/5:mode=forward,noise=alls=5:allf=t,format=yuv420p"
  chain += f"{prev}{grade}[vout]"
  run([PY, str(Path(__file__).parent / "score.py"), str(FILM / "timeline.json"), str(FILM / "score.wav")])
  run([FF, "-v", "error", "-y", *ins, "-i", str(FILM / "score.wav"), "-filter_complex", chain, "-map", "[vout]",
       "-map", f"{len(segs)}:a", "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-tune", "grain", "-pix_fmt", "yuv420p",
       "-af", "acompressor=threshold=-22dB:ratio=3:attack=15:release=250:makeup=2,loudnorm=I=-16:TP=-1.5:LRA=9",
       "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest", str(FILM / "bdx_a_new_gait.mp4")])
  print("film:", FILM / "bdx_a_new_gait.mp4", f"{t:.1f} s")
  web(t)


def web(length):
  """Web cut: 720p, no grain, two-pass to stay under the 15 MB artifact file limit; plus shot loops."""
  site = ROOT / "site" / "media"
  site.mkdir(parents=True, exist_ok=True)
  kbps = int(13.5 * 8192 / length) - 96
  src = str(FILM / "bdx_a_new_gait.mp4")
  for p in (1, 2):
    run([FF, "-v", "error", "-y", "-i", src, "-vf", "scale=1280:720:flags=lanczos,hqdn3d=2:2:4:4", "-c:v", "libx264",
         "-preset", "slower", "-b:v", f"{kbps}k", "-pass", str(p), "-passlogfile", str(FILM / "x264pass"),
         *(["-an", "-f", "mp4", "/dev/null"] if p == 1 else ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(site / "film.mp4")])])
  for shot, *_ in CUT:
    if shot in ("intro", "title", "crawl", "end"):
      continue
    frames = sorted((FILM / shot).glob("*.png"))
    run([FF, "-v", "error", "-y", "-framerate", str(FPS), "-start_number", frames[0].stem, "-i", str(FILM / shot / "%04d.png"),
         "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-preset", "slower", "-crf", "25", "-maxrate", "2400k", "-bufsize", "4800k",
         "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(site / f"shot_{shot}.mp4")])
    run([FF, "-v", "error", "-y", "-i", str(frames[len(frames) // 2]), "-vf", "scale=1280:720", "-q:v", "4", str(site / f"shot_{shot}.jpg")])
  print("web media written to", site)


if __name__ == "__main__":
  main()
