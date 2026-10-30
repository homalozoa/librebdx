# Changelog

How the project got here, stage by stage, is in [docs/HISTORY.md](docs/HISTORY.md).

## [Unreleased]

- Tantrum policy: 100,000 iterations on the final geometry; Table III MAE 0.033 rad (paper 0.032), 0 falls. Checkpoint `bdx_tantrum.pt` added to the release.
- Chapter VII ("The Long Night") re-recorded with the new policy and re-rendered; both film cuts rebuilt.

## [1.0.0] — 2026-10-29

First public release.

- Complete paper stack in mjlab / MuJoCo Warp: actuator model, action pipeline, standing / walking / episodic references, runtime with full state transfer, three-layer animation engine.
- BDX-R rebuilt to the paper's height (0.66 m), mass distribution and hip ranges; 600 Hz physics.
- Final policies, 100,000 iterations each at 8,192 × 24: standing, walking (two recipes), happy dance, excited, jump.
- Evaluations for Table III, Figs. 7–9, push robustness, zero-command drift and jump flight.
- Blender film pipeline and the bilingual short *BDX: A New Gait*.

Release assets: `checkpoints/*.pt` (rsl_rl format), the films in English and Chinese (1080p and 720p web cuts) and `SHA256SUMS`.
