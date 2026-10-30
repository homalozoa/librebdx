# Project history

The repository was put under version control only at publication on Oct 29, 2026, so its first commit holds the finished state. This timeline was reconstructed from the working-session records and from earlier versions of the report. All dates are UTC, and times are approximate.

---

### Oct 22, 2026 — Setup and a first standing attempt
- Set up mjlab 1.6 / MuJoCo Warp with rsl_rl PPO on a single RTX 4090 workstation. Work began as part of a broader paper-reproduction effort; from Oct 23 it was limited to this paper.
- Chose the open-source BDX-R as the stand-in robot and re-massed it to the paper's distribution: torso 5.8 kg, legs 3.6 kg each, head 2.4 kg, 15.4 kg total.
- Built the core pieces:
  - batched torch IK for standing references, plus an MLP fitted to dense IK samples;
  - the Table VI actuator model;
  - the 50 Hz action pipeline (first-order hold, 37.5 Hz low-pass, clipping);
  - the gait-authoring tool reimplemented from *Stylized Gaits*.
- The walking reference table built cleanly: IK residual median 3e-8, with 1.3 % of extreme command combinations off by more than 5 mm.
- The first standing run (4,096 envs) failed. After 250 iterations, episodes lasted 1.6 s and the action std had collapsed to 0.16.
- An open-loop CoM MPC for the gait tool took about 1 s per solve and was ill-conditioned. It was later replaced by a DCM terminal constraint for the tool and LIPM convolution for the references.

### Oct 23 (morning) — Standing works; episodic tasks added
- Two causes of the standing plateau were found and fixed:
  - **Head action scale too small:** neck action penalties dominated. Action scaling now follows each joint's reference range, which raised the head joints from 0.24 to 0.52–0.86 rad.
  - **Misleading metric:** the logged tracking error was recorded at reset, the moment of falling. It is now the episode average.
- Entropy 0.005 was added; the paper uses 0.
- Run **S1** (standing, 2,000 iterations): joint MAE 0.028 rad with 0 falls (paper: 0.035).
- Added episodic tasks (happy dance, excited, tantrum, jump) with procedural clips and the App. A emphasis rewards. The tantrum clip was corrected so the CoM shifts over the stance foot.
- Training was split between the 4090 workstation and a shared 8-GPU server. Under other users' load the server ran at 11–12 s per iteration.

### Oct 23 (day) — Walking would not step
This was the longest debugging stretch. In order:
1. **Falls at high speed.** Speeds of |vx| ≥ 0.5 m/s took 7 % of training time but caused 62 % of falls. Fast forward commands were then over-sampled, which doubled episode length (306 → 631 steps).
2. **Stepping in place.** The policy stepped in place: the correlation between commanded and actual velocity was zero, even though a hand-written controller could move the robot. Two causes were fixed:
   - reference velocities were finite-differenced after the path-frame projection; they now come from the generator;
   - resets wrote zero velocities; reference-state initialization now includes reference velocities.
3. **Survival weight.** A diagnostic sweep showed that survival weight 20 (the paper's value) rewarded standing still, while weight 5 walked but fell.
4. **Over-large hip actions.** The hip yaw/roll action scale was about 2.5× the joints' range; it was fixed.
5. **The real bug (about 20:35 UTC).** The gait planner integrated phase with Euler steps and skipped about 45 ms at every lift-off, so the feet "teleported." With exact time advance through the phase map, the maximum reference joint speed fell from 124 to 25 rad/s.
- The main walking run (W14) then used a survival-weight curriculum from 5 to 20 and the torso weights from the group's Olaf paper. At 1,500 iterations the forward tracking slope was 0.73.

### Oct 23 (evening) — The jump stays on the ground
- The first jump (run E1) scored 0.031 rad MAE but kept its toes on the ground, the cheat the paper describes.
- The clip was rebuilt as a ballistic jump: 0.67 m/s take-off, about 0.14 s airborne. A later code sync from the workstation silently overwrote the rebuilt clip, so data directories are now excluded from syncs.
- A flight-frame airborne bonus was added, and the jump was fine-tuned with entropy 0.005 (run **E8air**).
- All four episodic policies finished with 0 falls. MAE: happy dance 0.052, excited 0.044, jump 0.027, tantrum 0.045 rad.

### Oct 24 (early) — First complete result set, at 400 Hz
- **W14** kept walking after the survival weight returned to 20: forward slope 0.91, lateral slope 0.87, MAE 0.055 rad, 0 falls.
- A 1,500-iteration rough-terrain fine-tune produced **W15terrain**: 0.046 rad MAE on both flat and rough ground.
- **Velocity tracking:** 0.699 m/s at a 0.7 m/s command. Turning was capped at 1.0 rad/s because BDX-R's hip yaw range is only ±0.26 rad (the paper reaches 1.8 rad/s).
- **Standing pushes:** 100 % survival up to 60 N; 62 % falling to 10 % across the paper's 90–150 N range.
- **Ablations** (Sec. VII-B; the ref-input variant was resumed as **C1cont**):
  - reference-as-input matched our formulation;
  - both reward-only variants learned to stand still instead of shuffling. This part was not reproduced.
- **Stated deviations at this point:** 400 Hz physics (paper: 600 Hz), entropy 0.005, and only about 2–8.5 % of the paper's 100,000-iteration budget.

### Oct 24 — Report, web page and the first film
- Wrote the Markdown report and a results web page with figures and videos.
- **First film:**
  - policy rollouts were exported as per-mesh world poses into Blender;
  - Meshy retextured 20 shell parts, and the textures were transferred onto the simulation meshes;
  - a twin-sun desert and a ringed-planet night scene were built;
  - the opening crawl was title-style.
- One shot's render was killed silently after 9 frames. A watcher now checks every shot's frame count and re-renders any missing range.
- The web cut was 117–118 s and 13.9 MB.
- A full-resolution review found five flaws, all fixed and re-rendered:
  - Meshy painted a thigh grey and blue, ignoring the prompt;
  - the torso side texture was smeared;
  - a bright gap showed at the dune edge;
  - the title receded too fast;
  - the night mesas looked flat.

### Oct 24–25 — Film v2: Cycles, 4K livery and ten chapters
- Rendering moved to Blender 5.1 Cycles on the 4090 workstation, with chapters split onto idle GPUs of the shared server.
- Meshy retextured 36 parts at 4K with full PBR maps. Each simulation mesh got its own UV unwrap, and the Meshy surface was baked onto it with Cycles (selected-to-active). This replaced per-corner UV transfer, which had smeared texture. All 53 target meshes were baked.
- Eyes became geometry: a glass dome over a procedural iris, with blinking.
- The story was restructured into ten chapters. The first footsteps came from a CC0 dry-sand recording pitched up for a 15 kg robot.
- The v2 film (122.9 s) was published on Oct 25.

### Oct 25 (morning) — Footprints, real sand, bilingual cut and the calm gait
- **Footsteps:** 92 single steps were cut by onset detection from CC0 desert recordings on Freesound.
  - The first version sounded "metallic" and the two feet sounded different.
  - The final light voice clamps spectral ringing, matches both feet within ±1 dB, and is about 5 dB quieter.
- **Footprints and dust** were added in Blender: prints shaped from the real foot-pad outline, sand grains thrown at touchdown, and a drifting dust puff.
- **Bilingual titles** (English and Chinese) were added with chapter wipes. "May the Force be with you" was kept verbatim, and the copy was rewritten in a plainer voice.
- **Calm gait:** the paper-matched walker (W15) stepped about 3.2–4.4 times per second, which looked hurried for a small character.
  - A new reference stretches the gait cycle at low speed: 1.0 s in place down to 0.48 s at 0.7 m/s.
  - IK failures first rose to 22 %; after fixes they were 10 %, all at extreme strafe-plus-speed corners.
  - W15 was fine-tuned on this reference for 2,000 iterations (run **W16calm**).
- **Reunion shot:** a run at 0.7 m/s that stops at double support and then switches policy. One take fell, and that take was discarded. A clean re-recording finished with 0 falls.
- The English cut was 127.3 s. Both language versions were published on Oct 25.

### Oct 25 (afternoon) — Honest audit and restart at paper settings
- Asked whether everything had been reproduced, a 64-world measurement showed it had not:
  - mean jump flight was 0.025 s, and only 41 % of worlds left the ground for more than 0.02 s;
  - turning was 1.0 rad/s against the paper's 1.8;
  - the training budget was a small fraction of the paper's.
- There was a brief proposal to switch to a different training method. It was reverted within minutes, before any code changed, and the decision was to follow the paper: PPO with Table IV hyper-parameters.
- **Geometry rebuild:**
  - the neck was lengthened by 6.6 cm, so the robot stands 0.66 m tall without antennas (BDX-R was about 0.60 m);
  - hip yaw was widened to ±0.6 rad and hip roll to −0.45…0.6 rad;
  - the mass distribution was kept.
- Every reference was regenerated for the new body:
  - standing MLP (loss 4e-6);
  - episodic clips, including a new ballistic jump with 1.07 m/s take-off and 0.22 s flight;
  - the walk table.
- **Speed work:**
  - Profiling showed the environment step was CPU-bound: 47 ms of GPU work against 194 ms of host time.
  - Three fixes made it 2.7× faster on an idle machine: armature randomization every 40 s instead of every reset, vectorized disturbances, and a packed walk table.
  - On the shared server, two runs per GPU gave 1.55× the throughput of one.
- **P1 batch:** six runs at 600 Hz (decimation 12), entropy 0, 8,192 envs and 100,000 iterations: **P1walk, P1walkH, P1jump, P1stand, P1happy, P1exc**.
  - The first launch started nothing because of a script bug; it was relaunched.
  - The main walker ran at 1.6 s per iteration on the 4090; shared-server runs ran at 2.3–4 s.

### Oct 26 — Entropy collapse at paper settings
- About 19 hours in, behaviour tests (not reward curves) showed the failures:
  - **P1walk** at 25k iterations stood still: a 0.7 m/s command gave 0 m/s, and 1.8 rad/s gave 0.03 rad/s;
  - **P1walkH** reached only 0.4 m/s and 0.2 rad/s;
  - **P1jump** had 0 s of flight.
- **Cause:** with entropy 0, the action std fell from 0.3 to 0.01–0.02 within 2,500 iterations.
- The two walkers and the jump were restarted from scratch with entropy 0.005, as **P2walk, P2walkH and P2jump**.
  - The jump emphasis reward was sharpened to σ = 1.5 cm, with extra weights of 12 (height), 3 (orientation) and 12 (contact). The weaker settings had paid under 5 % of the survival reward.
  - Standing and the two dances were unaffected and kept training.
- **Results the same day:**
  - P2walk at 17.5k iterations turned at +1.83 / −1.78 rad/s against the paper's ±1.8 rad/s target;
  - P2jump at 10k iterations flew 0.18–0.20 s with 8.8 cm foot clearance and 0 falls;
  - the knee peaked at about 15 Nm and did not reach its torque limit, unlike the paper's Fig. 8.

### Oct 27 — Where the compute went
- Earlier checkpoints were measured to see when each policy became usable:
  - the walker was within 10 % on every command by 7.5k iterations;
  - the jump reached its final flight level by 5k iterations;
  - more than 90 % of the 100k-iteration budget went to polishing that changed little.
- Sampling took 87 % of each iteration, and physics about 70 % of the total.
- Switching the remaining runs to a faster training method was considered again and declined. The runs continued unchanged.

### Oct 28 — Walker finished; jump moved
- **P2walk finished:**
  - MAE 0.043 rad with 0 falls (W16calm had 0.047);
  - the apparent sideways drift at zero command (0.07–0.18 m/s at mid-training) turned out to be settling after a stop. Settled drift is under 1.2 mm/s.
- **P2jump** was moved from the contended shared server to the idle 4090 and resumed from its checkpoint (resumed once, at about 65k iterations).

### Oct 29 — Final evaluation, re-recorded film and release
- All runs except tantrum had finished. Tantrum was not part of the six-run batch. **P2tantrum** started on the new geometry at about 1.3 s per iteration and was still training at release.
- **Final results** at 100,000 iterations, 600 Hz, 256 envs × 60 s each, none fell:

  | Policy | MAE (rad) | Paper |
  |---|---:|---:|
  | Standing | 0.017 | 0.035 |
  | Walking, flat | 0.043 | 0.123 |
  | Walking, rough terrain | 0.044 | — |
  | Excited | 0.028 | 0.029 |
  | Happy dance | 0.025 | 0.027 |
  | Jump | 0.022 | 0.043 |

  Turning reached +1.749 / −1.754 rad/s against a ±1.8 command.
- **Film:** every story beat was re-recorded with the final policies (0 falls) and the robot meshes were re-exported for the longer neck. Ten scenes were re-rendered, and the English and Chinese cuts came to 2:08 each. The tantrum scene still uses older footage until its policy finishes.
- **Release:** the repository was cleaned for open-sourcing (README, license, figures under `docs/`) and committed as a single commit.

---

## Run names

| Name | What it was |
|---|---|
| **S1** | First successful standing policy (Oct 23, 2,000 iterations, 400 Hz, original geometry) |
| **E1** | First batch of episodic policies: happy dance, excited, tantrum, jump (original geometry) |
| **E8air** | Jump fine-tune with the ballistic clip, airborne bonus and entropy 0.005 (Oct 23) |
| **W1…W13** | Walking debugging runs on Oct 23 (survival weight, reset velocities, action scale, phase bug) |
| **W14** | Main walking run with the survival curriculum 5 → 20, after the phase-integration fix |
| **W15 / W15terrain** | W14 fine-tuned on randomized rough terrain; the paper-matched walker of the first report |
| **W16calm** | W15 fine-tuned for 2,000 iterations on the calm gait reference; the walker in films v1–v2 |
| **C1cont** | Continuation of the reference-as-input ablation run |
| **P1stand, P1happy, P1exc** | Final standing, happy dance and excited policies (600 Hz, new geometry, 100k iterations, entropy 0) |
| **P1walk, P1walkH, P1jump** | Paper-faithful walking and jump runs with entropy 0; collapsed and abandoned on Oct 26 |
| **P2walk** | Final walker (Table I weights, entropy 0.005, calm gait); used in the final film |
| **P2walkH** | Second walking recipe (Olaf weights plus survival curriculum), kept as a hedge |
| **P2jump** | Final jump (sharpened Eq. 13 emphasis, entropy 0.005); resumed once at about 65k |
| **P2tantrum** | Tantrum retrained on the new geometry; finished Oct 30 (MAE 0.033 rad, 0 falls), released in v1.1.0 |
