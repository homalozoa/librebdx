"""Cut a bank of single sand steps from the CC0 field recordings (Blender's Python + numpy).

python sand_bank.py  ->  results/bdx/audio/sand_bank.npz  (steps at 48 kHz, peak-normalized)
Each step starts 25 ms before its onset and runs 420 ms with a smooth fade. Steps are kept
only if they stand >= 18 dB above the local noise floor and have no second onset inside.
"""

import wave
from pathlib import Path

import numpy as np

SR = 48000
AUD = Path(__file__).resolve().parents[1] / "results" / "bdx" / "audio"
SRC = {"fs_705743_1661766": 1.0, "fs_655366_13723333": 1.0, "fs_511781_6753194": 1.0, "fs_164459_2320755": 1.0}


def load(p):
  with wave.open(str(p)) as w:
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768


def hp(x, fc=70):
  X = np.fft.rfft(x)
  f = np.fft.rfftfreq(len(x), 1 / SR)
  return np.fft.irfft(X / (1 + (fc / np.maximum(f, 1)) ** 4), len(x))


bank, meta = [], []
for name in SRC:
  x = hp(load(AUD / f"{name}.wav"))
  hop = 240
  e = np.sqrt(np.convolve(x**2, np.ones(hop) / hop, "same")[::hop])
  db = 20 * np.log10(e + 1e-9)
  floor = np.percentile(db, 20)
  last = -1000
  for i in range(4, len(db) - 100):
    if db[i] > floor + 12 and db[i - 4] < floor + 6 and i - last > 50:
      last = i
      s0 = i * hop - int(0.025 * SR)
      seg = x[s0 : s0 + int(0.42 * SR)].copy()
      if len(seg) < int(0.42 * SR):
        continue
      se = db[i : i + int(0.42 * SR / hop)]
      peak = se.max()
      if peak < floor + 18:
        continue
      # reject if a second strong onset appears inside the window (two steps merged)
      tail = se[int(0.12 * SR / hop) :]
      if len(tail) and tail.max() > peak - 4:
        continue
      env = np.ones(len(seg))
      fi = int(0.004 * SR)
      env[:fi] = np.linspace(0, 1, fi)
      fo = int(0.15 * SR)
      env[-fo:] = np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
      seg *= env
      seg /= np.abs(seg).max() + 1e-9
      X = np.abs(np.fft.rfft(seg))
      f = np.fft.rfftfreq(len(seg), 1 / SR)
      cen = float((X * f).sum() / X.sum())
      bank.append(seg.astype(np.float32))
      meta.append((name, i * hop / SR, peak - floor, cen))
meta = np.array(meta, dtype=object)
cen = np.array([m[3] for m in meta], float)
lo, hi = np.percentile(cen, [15, 85])
keep = [k for k in range(len(bank)) if lo <= cen[k] <= hi]  # drop spectral outliers (plane, twigs, thumps)
np.savez_compressed(AUD / "sand_bank.npz", steps=np.stack([bank[k] for k in keep]), src=np.array([meta[k][0] for k in keep]),
                    t=np.array([meta[k][1] for k in keep], float), snr=np.array([meta[k][2] for k in keep], float), centroid=cen[keep])
print("kept", len(keep), "of", len(bank), "centroid band", int(lo), "-", int(hi), "Hz")
from collections import Counter
print(Counter(meta[k][0] for k in keep))
