# Film pipeline

Turns recorded policy rollouts into *BDX: A New Gait*. Everything here runs inside Blender 5.1 (`blender -b -P <script> -- <args>`) except the cut, which needs Python 3 and ffmpeg. Outputs go to `results/bdx/`.

## Steps

```bash
# 1. Rollouts → per-mesh world poses (needs the GPU training environment)
python scripts/bdx_story.py STAND.pt WALK.pt
python scripts/bdx_story2.py STAND.pt WALK.pt EXCITED.pt
python scripts/export_blender.py results/bdx/traj/*.npz

# 2. Textures (optional; needs MESHY_API_KEY), then bake them onto the sim meshes
MESHY_SET=v2 python blender/meshy_client.py
blender -b -P blender/bake.py

# 3. Shots, with Cycles on a GPU (one chapter per call, or all of them)
BLENDER=/path/to/blender bash blender/render_story.sh [waking listening ...]

# 4. Footstep bank (CC0 recordings as WAVs in results/bdx/audio/), titles, and the final
#    cut; film_cut2.py writes the timeline and calls score2.py for the score
python blender/sand_bank.py
for l in en zh; do for t in intro title crawl end; do
  BDX_CUT=2 BDX_LANG=$l blender -b -P blender/titles.py -- $t; done; done
python blender/film_cut2.py en zh
```

## Files

| File | Role |
|---|---|
| `bdx_lib.py` | shared scene kit: robot import, materials, eyes, desert, lights, cameras |
| `film2.py` | the story chapters (current cut) |
| `traces.py` | footprints and sand dust from simulated touchdowns |
| `bake.py`, `meshy_client.py`, `meshy_io.py` | Meshy retexture round trip and bake |
| `sand_bank.py`, `score2.py` | footstep bank from CC0 recordings, score and sound design |
| `titles.py` | opening crawl and end card |
| `film_cut2.py` | lower thirds, wipes and the final EN/ZH encode |
| `stills.py`, `lookdev.py`, `trace_test.py` | key art, model sheet and look-dev tests |
| `film.py`, `film_cut.py`, `score.py`, `render_all.sh`, `render_v2.sh` | the first, showcase-style cut (kept for reference) |

Fonts are system fonts; the Chinese titles expect MiSans in `~/Library/Fonts`. Footstep recordings are not included: download the CC0 sounds listed in [NOTICE](../NOTICE).
