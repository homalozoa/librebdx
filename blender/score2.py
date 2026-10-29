"""Score and sound design for the story cut (numpy only, Blender's bundled Python).

python score2.py timeline.json out.wav

Foley follows the recorded rollouts: every foot touchdown in the simulation becomes a
sand step made of (1) a real dry-sand crunch (Fantozzi's Footsteps, CC0) pitched up
and shortened for a small 15 kg droid, (2) a soft low pad press and (3) a short
trickle of displaced grains as the foot lifts. Loudness follows impact speed. The
relay's blinking lamp is a distant sonar ping; the music follows the story's arc.
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "bdx" / "blender"
AUD = ROOT / "results" / "bdx" / "audio"
rng = np.random.default_rng(11)

tl = json.loads(Path(sys.argv[1]).read_text())
N = int((tl["length"] + 0.5) * SR)
t = np.arange(N) / SR
seg = {s["shot"]: s for s in tl["segments"]}
mix = np.zeros((N, 2))


def hz(n):
  return 440.0 * 2 ** ((n - 69) / 12)


def lowpass(x, fc, order=2):
  X = np.fft.rfft(x)
  f = np.fft.rfftfreq(len(x), 1 / SR)
  return np.fft.irfft(X / (1 + (f / fc) ** (2 * order)), len(x))


def band(x, lo, hi):
  X = np.fft.rfft(x)
  f = np.fft.rfftfreq(len(x), 1 / SR)
  return np.fft.irfft(X / (1 + (lo / np.maximum(f, 1)) ** 4) / (1 + (f / hi) ** 4), len(x))


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


def chord(notes, t0, t1, gain, lp=1400, a=2.5, r=2.5, harm=6):
  L = t1 - t0
  if L <= 0:
    return
  v = sum(voice(hz(n), L, harm) for n in notes)
  n = len(v)
  e = np.minimum(1, np.arange(n) / SR / a) * np.minimum(1, (L - np.arange(n) / SR) / r)
  add(lowpass(v * np.clip(e, 0, 1), lp), t0, gain)


def S(name):
  return seg[name]["start"], seg[name]["start"] + seg[name]["dur"]


# ------------------------------------------------------------------ sand steps
def load_wav(p):
  with wave.open(str(p)) as w:
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768.0


def _soften(x):
  """One field step -> a soft sand step: flatten tonal ringing, tame the heel transient, ease the top."""
  n = len(x)
  X = np.fft.rfft(x)
  mag = np.abs(X)
  k = 31  # clamp spectral peaks to 2.5x their local median (removes the metallic ring)
  pad = np.pad(mag, k, mode="edge")
  med = np.array([np.median(pad[i : i + 2 * k + 1]) for i in range(0, len(mag))])
  X = X * np.minimum(1.0, 2.5 * med / np.maximum(mag, 1e-12))
  f = np.fft.rfftfreq(n, 1 / SR)
  X = X / (1 + (f / 6500) ** 4) / (1 + (260 / np.maximum(f, 1)) ** 4)  # 260 Hz-6.5 kHz: a small foot has no boom
  y = np.fft.irfft(X, n)
  # Transient tamer: fast envelope vs slow envelope, pull peaks down toward the body.
  a = np.abs(y)
  fast = np.convolve(a, np.ones(48) / 48, "same")
  slow = np.convolve(a, np.ones(1440) / 1440, "same")
  g = np.minimum(1.0, (2.2 * slow + 1e-6) / (fast + 1e-6)) ** 0.8
  y = y * g
  att = int(0.012 * SR)
  y[:att] *= np.linspace(0, 1, att) ** 1.5
  return y / (np.sqrt((y**2).mean()) + 1e-9)


def _bank():
  z = np.load(AUD / "sand_bank.npz")
  X = z["steps"].astype(np.float64)
  crest = np.abs(X).max(1) / np.sqrt((X**2).mean(1))
  F = np.abs(np.fft.rfft(X, axis=1))
  f = np.fft.rfftfreq(X.shape[1], 1 / SR)
  band_ = F[:, (f > 800) & (f < 8000)]
  peaky = band_.max(1) / np.median(band_, 1)
  keep = (crest < np.percentile(crest, 55)) & (peaky < np.percentile(peaky, 60))
  return [_soften(x) for x in X[keep]]


BANK = _bank()  # the softest, least ringing desert steps (CC0 field recordings), loudness-matched


def resample(x, ratio):
  n = int(len(x) / ratio)
  return np.interp(np.arange(n) * ratio, np.arange(len(x)), x)


def sand_step(vel, slow=1.0, side=0):
  """One step of a small droid on dry desert sand; identical voice for both feet.

  A softened real desert step, plus the sand compressing under the pad: a 150-700 Hz
  noise swell with a soft 25 ms rise. Slow-motion picture slows the sound with it."""
  x = BANK[rng.integers(0, len(BANK))]
  x = resample(x, 2 ** ((1.5 + rng.uniform(-0.3, 0.3)) / 12) * slow)  # a smaller foot: a touch higher
  L_ = int(0.2 / slow * SR)  # a light step is short
  x = x[:L_].copy()
  fo = int(0.08 / slow * SR)
  x[-fo:] *= np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
  d = np.arange(int(0.1 / slow * SR)) / SR
  sink = band(rng.standard_normal(len(d)), 350, 1400) * np.minimum(1, d / (0.02 / slow)) * np.exp(-d / (0.035 / slow))
  sink /= np.sqrt((sink**2).mean()) + 1e-9
  out = np.zeros(len(x) + 100)
  out[: len(x)] += x
  out[: len(sink)] += 0.15 * sink
  return out * 0.062 * 10 ** (rng.uniform(-1.0, 1.0) / 20)


def sand_lift(slow=1.0):
  """A pad leaving the sand: a soft, short pour of sand sliding back into the print."""
  L_ = int(0.25 / slow * SR)
  d = np.arange(L_) / SR
  pour = band(rng.standard_normal(L_), 600, 4500) * np.exp(-d / (0.07 / slow)) * np.minimum(1, d / 0.02)
  return 0.014 * pour / (np.sqrt((pour**2).mean()) + 1e-9)


# ------------------------------------------------------------------ music: the story arc
t_title = seg["title"]["start"]
crawl0, crawl1 = S("crawl")
day0 = seg["waking"]["start"]
night0 = seg["nightfall"]["start"]
end0 = seg["end"]["start"]
L_all = tl["length"]

# Space: cold open fifths under the intro and crawl.
sp = sum(voice(hz(n), N / SR, 4, 0.002) for n in (38, 45, 50, 57))
mix += 0.2 * (lowpass(sp, 900) * env(0.3, day0 + 1.5, 3.0, 3.0) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.07 * t)))[:, None]
sw = band(rng.standard_normal(int(2.2 * SR)), 300, 6000) * np.linspace(0, 1, int(2.2 * SR)) ** 3
add(sw, t_title - 2.2, 0.22)
tt = np.arange(int(4 * SR)) / SR
add(np.sin(2 * np.pi * (55 * tt - 6 * tt**2)) * np.exp(-tt / 0.9) + 0.5 * lowpass(rng.standard_normal(len(tt)), 400) * np.exp(-tt / 0.25), t_title, 0.85)
chord((50, 57, 62, 66, 69), t_title, t_title + 6.0, 0.05, 2500, 0.05, 3.5, 5)


def brass(note, dur, gain=1.0):
  """A small brass voice: bright saw harmonics whose brightness swells with the note."""
  n = int(dur * SR)
  tt = np.arange(n) / SR
  f = hz(note) * (1 + 0.004 * np.sin(2 * np.pi * 5.2 * tt) * np.minimum(1, tt / 0.3))
  ph = 2 * np.pi * np.cumsum(f) / SR
  bright = np.clip(tt / 0.08, 0, 1) * (0.55 + 0.45 * np.exp(-tt / 0.6))
  sig = sum(np.sin(k * ph) / k * np.clip(bright * 12 - k + 1, 0, 1) for k in range(1, 13))
  env = np.minimum(1, tt / 0.03) * np.minimum(1, (dur - tt) / 0.12)
  return sig * env * gain


# An original fanfare under the title (D major: rising arpeggio, a turn, the octave held).
FANFARE = [(62, 0.0, 0.32), (66, 0.32, 0.32), (69, 0.64, 0.6), (67, 1.24, 0.22), (66, 1.46, 0.22), (64, 1.68, 0.4), (74, 2.1, 2.6)]
for note, t0, dur in FANFARE:
  v = brass(note, dur) + 0.6 * brass(note - 12, dur)
  add(lowpass(v, 3800), t_title + 0.05 + t0, 0.11)
  add(lowpass(brass(note + 7, dur) * 0.5, 3000), t_title + 0.05 + t0, 0.05, 0.3)
chord((38, 50, 57, 62), t_title + 2.1, t_title + 5.4, 0.09, 1200, 0.1, 1.8, 6)
scale = [62, 64, 65, 69, 72, 74, 76, 77]
for k, tp in enumerate(np.arange(crawl0 + 0.5, crawl1 - 1.0, 0.6)):
  n = scale[(k * 5 + (k // 4)) % len(scale)]
  d = np.arange(int(1.6 * SR)) / SR
  add((np.sin(2 * np.pi * hz(n) * d) + 0.3 * np.sin(4 * np.pi * hz(n) * d)) * np.exp(-d / 0.35), tp, 0.055, np.sin(k * 1.7) * 0.5)

# Day: waking (alone, a single sustained tone opening into a chord), listening (suspense),
# crossing/wind (forward motion), leap (lift), arrival (resolution, then a playful dance).
w0, w1 = S("waking")
chord((50, 57), w0, w1 + 1, 0.10, 900, 3.0, 2.0)
chord((62, 69, 74), w0 + 2.2, w1 + 1, 0.06, 1800, 2.5, 2.0)
l0, l1 = S("listening")
chord((47, 54, 59, 62), l0, l1 + 0.5, 0.09, 1200, 1.5, 1.5)
c0, c1 = S("crossing")
prog = [(50, 53, 57, 64), (46, 50, 53, 57), (41, 48, 53, 55), (48, 50, 55, 62)]
k, tc = 0, c0
wd1 = S("wind")[1]
while tc < wd1:
  chord(prog[k % 4], tc, min(tc + 4.2, wd1 + 0.6), 0.1, 1500, 1.2, 1.2)
  tc += 4.0
  k += 1
lp0, lp1 = S("leap")
chord((53, 60, 65, 69, 72), lp0, lp1 + 0.3, 0.09, 2600, 0.8, 2.0)
a0, a1 = S("arrival")
chord((50, 57, 62, 66), a0, a1, 0.1, 1800, 1.5, 2.0)
for k, tp in enumerate(np.arange(a0 + 9.0, a1 - 0.5, 0.25)):  # the dance: bright plucks on the beat
  n = [74, 78, 81, 78, 76, 81, 83, 81][k % 8]
  d = np.arange(int(0.5 * SR)) / SR
  add(np.sin(2 * np.pi * hz(n) * d) * np.exp(-d / 0.12), tp, 0.05, 0.3 * np.sin(k))

# Night: cold and wide; a low grumble for the tantrum, then the answer and a warm rise to the end.
n0, n1 = S("nightfall")
chord((38, 45, 52), n0, n1 + 0.4, 0.11, 700, 1.5, 1.0)
an0, an1 = S("answer")
chord((50, 57, 64), an0, an1 + 0.5, 0.08, 1500, 1.0, 1.0)
ex0, ex1 = S("reunion")
run1 = ex0 + 5.6  # it reaches the other droid about here; the excited dance follows
for k, tp in enumerate(np.arange(ex0 + 0.6, run1, 0.22)):  # hurrying: a quickening, rising ostinato
  n = [62, 66, 69, 74][k % 4] + 2 * (k // 8)
  d = np.arange(int(0.25 * SR)) / SR
  add(np.sin(2 * np.pi * hz(n) * d) * np.exp(-d / 0.07), tp, 0.03 + 0.002 * k, 0.4 * np.sin(k))
chord((50, 57, 62, 66, 69), run1 - 0.3, ex1 + 0.3, 0.1, 2400, 0.4, 1.0)
for k, tp in enumerate(np.arange(run1, ex1 - 0.2, 0.125)):
  n = [74, 78, 81, 86][k % 4]
  d = np.arange(int(0.3 * SR)) / SR
  add(np.sin(2 * np.pi * hz(n) * d) * np.exp(-d / 0.08), tp, 0.035, 0.5 * np.sin(k * 0.9))
g0, g1 = S("together")
chord((38, 50, 57, 62, 66, 69), g0, L_all, 0.12, 1600, 2.0, 4.0)

# ------------------------------------------------------------------ air
wind = band(rng.standard_normal(N), 120, 1600) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t + 1) * np.sin(2 * np.pi * 0.047 * t))
low = band(rng.standard_normal(N), 40, 300)
day = env(day0, night0 + 0.6, 2.0, 1.2)
nite = env(night0, end0 + 1.0, 1.0, 2.0)
mix[:, 0] += 0.05 * wind * (day + 0.35 * nite) + 0.07 * low * day
mix[:, 1] += 0.05 * np.roll(wind, 2400) * (day + 0.35 * nite) + 0.07 * np.roll(low, 1800) * day

# ------------------------------------------------------------------ foley from the rollouts
meshes = np.load(DATA / "meshes.npz")
gname = [str(g) for g in meshes["geom_mesh"]]
feet = [gname.index("Left_Foot_Pad"), gname.index("Right_Foot_Pad")]


def film_time(s, ct):
  return s["start"] + (ct - s["clip_t0"]) / s["speed"]


DIST = {"answer": 0.55, "together": 0.6, "reunion": 0.85}  # quieter where the droid is farther from the camera
for s in tl["segments"]:
  if not s["clip"]:
    continue
  z = np.load(DATA / f"{s['clip']}.npz", allow_pickle=True)
  fps = float(z["fps"])
  pos = z["pos"]
  T = len(pos)
  ft = film_time(s, np.arange(T) / fps)
  inside = (ft >= s["start"] + 0.05) & (ft <= s["start"] + s["dur"] - 0.05)
  lvl = DIST.get(s["shot"], 1.0) * (0.75 if s["start"] >= night0 else 1.0)
  for k, g in enumerate(feet):
    fz = pos[:, g, 2]
    ground = np.percentile(fz, 5)
    down = fz < ground + 0.006
    for i in np.where(down[1:] & ~down[:-1])[0] + 1:
      if inside[i]:
        vz = abs(fz[i] - fz[max(0, i - 2)]) * fps / 2
        add(sand_step(vz, s["speed"]), ft[i], 0.62 * lvl * (1.3 if vz > 0.6 else 1.0), pan=0.0)
    for i in np.where(~down[1:] & down[:-1])[0] + 1:
      if inside[i]:
        add(sand_lift(s["speed"]), ft[i], 0.5 * lvl, pan=0.0)
  rel = pos - pos[:, :1]
  act = np.clip(np.r_[0, np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean(-1) * fps] / 0.35, 0, 1)
  i0, i1 = int(max(0, s["start"]) * SR), int(min(L_all, s["start"] + s["dur"]) * SR)
  a = np.interp(t[i0:i1], ft, act) * env(s["start"], s["start"] + s["dur"], 0.3, 0.3)[i0:i1]
  ph = 2 * np.pi * np.cumsum(950 + 500 * a) / SR
  mix[i0:i1] += (0.014 * lvl * (np.sin(ph) + 0.4 * np.sin(2 * ph)) * a**1.5)[:, None]
  if "force" in z.files:  # gusts: a rising rush of wind and sand, then the hit
    Fm = np.linalg.norm(z["force"], axis=-1)
    for i in np.where((Fm[1:] > 1) & (Fm[:-1] < 1))[0] + 1:
      if inside[i]:
        d = np.arange(int(1.4 * SR)) / SR
        rush = band(rng.standard_normal(len(d)), 400, 7000) * np.exp(-((d - 0.42) ** 2) / 0.03)
        hit = np.sin(2 * np.pi * (60 - 25 * d) * d) * np.exp(-d / 0.2) * (d > 0.4)
        add(0.55 * rush + 0.6 * hit, ft[i] - 0.4, 0.75, pan=0.4)

# ------------------------------------------------------------------ signal, eyes, voice
def ping(gain, far=True):
  d = np.arange(int(1.8 * SR)) / SR
  p = (np.sin(2 * np.pi * 1318.5 * d) + 0.25 * np.sin(2 * np.pi * 2637 * d)) * np.exp(-d / (0.5 if far else 0.25))
  p *= np.minimum(1, d / 0.004)
  return lowpass(p, 4500 if far else 9000) * gain


RELAY = {"listening": 0.05, "crossing": 0.05, "wind": 0.04, "leap": 0.06, "arrival": 0.1, "nightfall": 0.11, "answer": 0.08, "together": 0.06}
for name, g in RELAY.items():
  s0, s1 = S(name)
  for tp in np.arange(s0, s1, 2.0):  # matches the lamp: on for 0.35 s every 2 s
    add(ping(g, g < 0.08), tp, 1.0, 0.3)


def chirp(notes, dur=0.09, gain=0.12, glide=0.4):
  out = []
  for a_, b_ in zip(notes[:-1], notes[1:]):
    d = np.arange(int(dur * SR)) / SR
    f = hz(a_) * (hz(b_) / hz(a_)) ** np.minimum(1, d / (dur * glide))
    ph = 2 * np.pi * np.cumsum(f) / SR
    out.append((np.sin(ph) + 0.25 * np.sign(np.sin(ph))) * np.sin(np.pi * d / dur) ** 0.6)
  return np.concatenate(out) * gain


def power_on(gain):
  d = np.arange(int(0.9 * SR)) / SR
  f = 180 * 2 ** (d * 4.2)
  return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, d / 0.05) * np.exp(-((d - 0.55) ** 2) / 0.08) * gain


add(power_on(0.16), w0 + 1.1)
add(chirp([84, 88, 86, 91], 0.08), w0 + 2.6, 1.0, 0.1)            # awake
add(chirp([79, 84, 83], 0.1), l0 + 1.4, 1.0, -0.1)                # hears the signal
add(chirp([91, 86, 89, 84, 87], 0.07), c0 + 0.6, 0.9, 0.1)       # sets out
add(chirp([84, 91, 96, 91, 98], 0.06), a0 + 9.0, 1.0, 0.2)       # arrival: joy
add(chirp([62, 60, 58, 55], 0.14, 0.12, 0.8), n0 + 0.3, 1.0)      # nobody answers
add(power_on(0.07), an0 + 1.6 - 0.25, 1.0, 0.5)                   # far eyes wake
add(lowpass(chirp([88, 93, 91, 96], 0.08, 0.1), 3000), an0 + 2.6, 0.7, 0.6)  # the answer, from afar
add(chirp([86, 93, 89, 96, 93, 100], 0.05), ex0 + 0.4, 1.0, -0.2)    # sees the other: hurry
add(chirp([93, 98, 96, 101], 0.06), run1 + 0.2, 1.0, 0.1)              # arrives
add(chirp([81, 86, 90], 0.12, 0.09), g0 + 3.0, 1.0, 0.0)

# ------------------------------------------------------------------ reverb + master
d = np.arange(int(2.4 * SR)) / SR
ir = rng.standard_normal((len(d), 2)) * np.exp(-d / 0.6)[:, None]
ir = np.stack([lowpass(ir[:, c], 5000) for c in range(2)], -1)
L2 = 1 << int(np.ceil(np.log2(N + len(d))))
wet = np.stack([np.fft.irfft(np.fft.rfft(mix[:, c], L2) * np.fft.rfft(ir[:, c], L2), L2)[:N] for c in range(2)], -1)
out = (mix + 0.07 * wet) * env(0.0, L_all, 0.2, 2.5)[:, None]
out = np.tanh(1.3 * out / (np.abs(out).max() + 1e-9)) / np.tanh(1.3) * 0.89
with wave.open(sys.argv[2], "wb") as w:
  w.setnchannels(2)
  w.setsampwidth(2)
  w.setframerate(SR)
  w.writeframes((out * 32767).astype(np.int16).tobytes())
print("score", sys.argv[2], f"{N / SR:.1f} s")
