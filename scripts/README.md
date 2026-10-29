# Scripts

Run from the repository root (`python scripts/<name>.py ...`). Each script's docstring has its full usage.

## Training

| Script | Purpose |
|---|---|
| `train.py` | mjlab training entry point with the BDX tasks registered |
| `smoke_env.py` | build a task, step it, report speed and basic stats |

## Paper results

| Script | Paper item |
|---|---|
| `bdx_eval.py mae TASK CKPT` | Table III, joint tracking MAE |
| `bdx_eval.py veltrack CKPT` | Fig. 7, commanded vs measured velocity |
| `bdx_eval.py jump CKPT` | Fig. 8, jump torques vs limits |
| `jump_stats.py CKPT` | flight time, clearance and torso rise over 64 worlds |
| `bdx_push.py TASK CKPT [vx]` | push robustness |
| `drift_test.py` | zero-command drift after walking and fast turns |
| `bdx_transitions.py` | Fig. 9, stand → walk → stand → episodic → stand |
| `bdx_show.py` | Sec. VI, the animation engine driving the standing policy |
| `bdx_ablation_eval.py` | Sec. VII-B, walking formulation ablation |
| `eval_policy.py TASK CKPT` | generic rollout, metrics and optional video |
| `bdx_videos.py` | demo videos of all final policies |

## References and data

| Script | Purpose |
|---|---|
| `fit_bdx_standing_net.py` | sample the standing IK and fit the reference MLP |
| `build_walk_refs.py` | build the walking reference table (gait tool + GPU IK) |

## Film

| Script | Purpose |
|---|---|
| `bdx_story.py`, `bdx_story2.py` | record the story beats as policy rollouts |
| `export_blender.py` | turn rollouts into per-mesh world poses for Blender |
