"""Original synthesized score + sound design for the film (numpy only).

python score.py timeline.json out.wav
Layers: space pad (titles), desert wind, chord pad, title hit, crawl plucks,
footsteps + servo whine derived from the recorded rollouts, Force-push impacts,
droid chirps on the animation-engine clips, and a simple convolution reverb.
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "bdx" / "blender"
rng = np.random.default_rng(7)

tl = json.loads(Path(sys.argv[1]).read_text())
N = int((tl["length"] + 0.5) * SR)
t = np.arange(N) / SR
seg = {s["shot"]: s for s in tl["segments"]}
mix = np.zeros((N, 2))


def hz(note):  # MIDI -> Hz
  return 440.0 * 2 ** ((note - 69) / 12)


def fft_lowpass(x, fc, order=2):
  X = np.fft.rfft(x)
  f = np.fft.rfftfreq(len(x), 1 / SR)
  return np.fft.irfft(X / (1 + (f / fc) ** (2 * order)), len(x))


def fft_band(x, lo, hi):
  X = np.fft.rfft(x)
  f = np.fft.rfftfreq(len(x), 1 / SR)
  return np.fft.irfft(X * (1 / (1 + (lo / np.maximum(f, 1)) ** 4)) / (1 + (f / hi) ** 4), len(x))


def env(t0, t1, a=1.0, r=1.0):
  e = np.clip((t - t0) / max(a, 1e-3), 0, 1) * np.clip((t1 - t) / max(r, 1e-3), 0, 1)
  return e * e * (3 - 2 * e)


def add(sig, at, gain=1.0, pan=0.0):
  i = int(at * SR)
  if i >= N or i + len(sig) <= 0:
    return
  j0 = max(0, -i)
  s = sig[j0 : N - i] * gain
  mix[i + j0 : i + j0 + len(s), 0] += s * np.sqrt(0.5 * (1 - pan))
  mix[i + j0 : i + j0 + len(s), 1] += s * np.sqrt(0.5 * (1 + pan))


def voice(f, dur, harm=6, det=0.004):
  tt = np.arange(int(dur * SR)) / SR
  s = np.zeros_like(tt)
  for k in range(1, harm + 1):
    for d in (-det, det):
      s += np.sin(2 * np.pi * f * k * (1 + d) * tt + rng.uniform(0, 6.28)) / k**1.3
  return s / harm


# ------------------------------------------------------------------ space pad (titles)
space_end = seg["establish"]["start"] + 1.0
sp = np.zeros(N)
for n in (38, 45, 50, 57):  # D2 A2 D3 A3: open fifths, cold and wide
  sp += voice(hz(n), N / SR, harm=4, det=0.002)
sp = fft_lowpass(sp, 900)
sp *= env(0.3, space_end, 3.0, 2.5) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.07 * t))
mix += 0.22 * sp[:, None]

# Rising swell into the title, then the hit.
th = seg["title"]["start"]
sw = fft_band(rng.standard_normal(int(2.2 * SR)), 300, 6000) * np.linspace(0, 1, int(2.2 * SR)) ** 3
add(sw, th - 2.2, 0.25)
tt = np.arange(int(4 * SR)) / SR
boom = np.sin(2 * np.pi * (55 * tt - 12 * tt**2 / 2)) * np.exp(-tt / 0.9)
boom += 0.5 * fft_lowpass(rng.standard_normal(len(tt)), 400) * np.exp(-tt / 0.25)
add(boom, th, 0.9)
for n in (50, 57, 62, 66, 69):  # D major bloom under the title
  v = voice(hz(n), 6.0, 5) * np.exp(-np.arange(int(6 * SR)) / SR / 2.5) * np.minimum(1, np.arange(int(6 * SR)) / SR / 0.05)
  add(fft_lowpass(v, 2500), th, 0.09)

# Crawl: slow plucked arpeggio over the pad (D dorian).
c = seg["crawl"]
scale = [62, 64, 65, 69, 72, 74, 76, 77]
k = 0
for tp in np.arange(c["start"] + 0.5, c["start"] + c["dur"] - 1.0, 0.6):
  n = scale[(k * 5 + (k // 4)) % len(scale)]
  d = np.arange(int(1.6 * SR)) / SR
  pl = (np.sin(2 * np.pi * hz(n) * d) + 0.3 * np.sin(4 * np.pi * hz(n) * d)) * np.exp(-d / 0.35)
  add(pl, tp, 0.06, pan=np.sin(k * 1.7) * 0.5)
  k += 1

# ------------------------------------------------------------------ desert: wind + chord pad
desert0 = seg["establish"]["start"]
night0 = seg["dance"]["start"]
end0 = seg["end"]["start"]
wind = rng.standard_normal(N)
gust = 0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t + 1) * np.sin(2 * np.pi * 0.047 * t)
wind = fft_band(wind, 120, 1600) * gust
wl = fft_band(rng.standard_normal(N), 40, 300)
day = env(desert0, night0 + 0.5, 2.0, 1.2)
night = env(night0, end0 + 1.0, 1.0, 2.0)
mix[:, 0] += 0.05 * (wind * day + 0.4 * wind * night) + 0.08 * wl * day
mix[:, 1] += 0.05 * (np.roll(wind, 2400) * day + 0.4 * np.roll(wind, 2400) * night) + 0.08 * np.roll(wl, 1800) * day

# Chord pad: Dm9 - Bbmaj7 - Fadd9 - Csus2, 8 s each; night: Dm - Gm/D shimmer.
prog = [(50, 53, 57, 64), (46, 50, 53, 57), (41, 48, 53, 55), (48, 50, 55, 62)]
pad = np.zeros(N)
tc = desert0
i = 0
while tc < night0:
  notes = prog[i % 4]
  L = 8.6
  seglen = int(L * SR)
  v = sum(voice(hz(n), L, 6) for n in notes)
  e = np.minimum(1, np.arange(seglen) / SR / 2.5) * np.minimum(1, (L - np.arange(seglen) / SR) / 2.5)
  a = int(tc * SR)
  b = min(N, a + seglen)
  pad[a:b] += (v * e)[: b - a]
  tc += 8.0
  i += 1
pad = fft_lowpass(pad, 1400)
mix += 0.11 * pad[:, None]
nt = np.zeros(N)
for n in (62, 65, 69, 74):
  nt += voice(hz(n), N / SR, 3, 0.006)
nt = fft_lowpass(nt, 3000) * env(night0, end0 + 6.0, 2.5, 3.0) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))
mix += 0.06 * nt[:, None]
endpad = sum(voice(hz(n), N / SR, 4) for n in (38, 50, 57, 62, 66))
mix += 0.12 * (fft_lowpass(endpad, 1200) * env(end0, tl["length"], 1.5, 3.0))[:, None]

# ------------------------------------------------------------------ foley from the rollouts
meshes = np.load(DATA / "meshes.npz")
gname = [str(g) for g in meshes["geom_mesh"]]
feet = [gname.index("Left_Foot_Pad"), gname.index("Right_Foot_Pad")]
head = gname.index("Face")


def film_time(s, ct):
  return s["start"] + (ct - s["clip_t0"]) / s["speed"]


def thud(gain, pitch=1.0):
  d = np.arange(int(0.25 * SR)) / SR
  s = np.sin(2 * np.pi * 78 * pitch * d) * np.exp(-d / 0.05)
  s += 0.35 * fft_band(rng.standard_normal(len(d)), 800, 5000) * np.exp(-d / 0.012)
  return s * gain


for s in tl["segments"]:
  if not s["clip"]:
    continue
  z = np.load(DATA / f"{s['clip']}.npz", allow_pickle=True)
  fps = float(z["fps"])
  pos = z["pos"]
  T = len(pos)
  ct = np.arange(T) / fps
  ft = film_time(s, ct)
  inside = (ft >= s["start"] + 0.05) & (ft <= s["start"] + s["dur"] - 0.05)
  for k, g in enumerate(feet):
    fz = pos[:, g, 2]
    ground = np.percentile(fz, 5)
    down = fz < ground + 0.006
    td = np.where(down[1:] & ~down[:-1])[0] + 1
    for i in td:
      if inside[i]:
        vz = (fz[i] - fz[max(0, i - 2)]) * fps / 2
        add(thud(min(1.0, 0.25 + 1.2 * abs(vz)), 1.0 / max(s["speed"], 0.5) ** 0.3), ft[i], 0.5, pan=-0.3 if k == 0 else 0.3)
  # Servo whine follows how fast the body parts move relative to the torso.
  rel = pos - pos[:, :1]
  act = np.r_[0, np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean(-1) * fps]
  act = np.clip(act / 0.35, 0, 1)
  i0, i1 = int(max(0, s["start"]) * SR), int(min(tl["length"], s["start"] + s["dur"]) * SR)
  tt = t[i0:i1]
  a = np.interp(tt, ft, act) * env(s["start"], s["start"] + s["dur"], 0.3, 0.3)[i0:i1]
  f = 950 + 500 * a
  ph = 2 * np.pi * np.cumsum(f) / SR
  wh = (np.sin(ph) + 0.4 * np.sin(2 * ph)) * a**1.5
  mix[i0:i1] += 0.018 * wh[:, None]
  # Force-push impacts.
  if "force" in z.files:
    F = np.linalg.norm(z["force"], axis=-1)
    for i in np.where((F[1:] > 1) & (F[:-1] < 1))[0] + 1:
      if inside[i]:
        d = np.arange(int(0.9 * SR)) / SR
        whoosh = fft_band(rng.standard_normal(len(d)), 300, 3000) * np.exp(-((d - 0.08) ** 2) / 0.004)
        hit = np.sin(2 * np.pi * (60 - 25 * d) * d) * np.exp(-d / 0.18)
        add(0.5 * whoosh + 0.9 * hit, ft[i] - 0.08, 0.7)


def chirp(notes, dur=0.09, gain=0.12, glide=0.4):
  out = []
  for a, b in zip(notes[:-1], notes[1:]):
    d = np.arange(int(dur * SR)) / SR
    f = hz(a) * (hz(b) / hz(a)) ** np.minimum(1, d / (dur * glide))
    ph = 2 * np.pi * np.cumsum(f) / SR
    e = np.sin(np.pi * d / dur) ** 0.6
    out.append((np.sin(ph) + 0.25 * np.sign(np.sin(ph))) * e)
  return np.concatenate(out) * gain


# Droid voice on key beats.
add(chirp([84, 88, 86, 91], 0.08), film_time(seg["establish"], 3.0), 1.0, 0.1)
add(chirp([91, 86, 89, 84, 87], 0.07), film_time(seg["engine"], 6.2), 1.0, -0.1)
add(chirp([79, 77, 74, 72], 0.12, 0.1), film_time(seg["engine"], 14.5), 1.0)
add(chirp([84, 91, 96, 91, 98], 0.06), seg["dance"]["start"] + 0.4, 1.0, 0.2)
add(chirp([86, 93, 89, 96, 93, 100], 0.05), seg["excited"]["start"] + 0.2, 1.0, -0.2)
add(chirp([62, 60, 58, 55], 0.14, 0.12, 0.8), seg["tantrum"]["start"] + 0.3, 1.0)
add(chirp([88, 93, 98], 0.07), seg["jump"]["start"] + 1.5, 1.0)

# ------------------------------------------------------------------ reverb + master
d = np.arange(int(2.2 * SR)) / SR
ir = rng.standard_normal((len(d), 2)) * np.exp(-d / 0.55)[:, None]
ir[:, 0] = fft_lowpass(ir[:, 0], 5000)
ir[:, 1] = fft_lowpass(ir[:, 1], 5000)
L2 = 1 << int(np.ceil(np.log2(N + len(d))))
wet = np.stack([np.fft.irfft(np.fft.rfft(mix[:, c], L2) * np.fft.rfft(ir[:, c], L2), L2)[:N] for c in range(2)], -1)
out = mix + 0.06 * wet
out *= env(0.0, tl["length"], 0.2, 2.5)[:, None]
out = np.tanh(1.3 * out / (np.abs(out).max() + 1e-9)) / np.tanh(1.3) * 0.89
with wave.open(sys.argv[2], "wb") as w:
  w.setnchannels(2)
  w.setsampwidth(2)
  w.setframerate(SR)
  w.writeframes((out * 32767).astype(np.int16).tobytes())
print("score", sys.argv[2], f"{N / SR:.1f} s")
