#!/usr/bin/env bash
# Efficiency pilot: train the walker at a given physics rate with gate-based early stopping.
# usage: scripts/pilot_walk.sh SIM_HZ   (run from the repo root; one pilot at a time so timings are clean)
set -e
HZ=${1:-600}
PY=${PY:-python}
MAXIT=${2:-30000}
NAME=E_walk_${HZ}hz
export DISNEY_WALK_TABLE=calm DISNEY_TERRAIN=1 DISNEY_TABLE1=1 DISNEY_SIM_HZ=$HZ
$PY scripts/train.py Disney-BDX-Walking --env.scene.num-envs 8192 --agent.max-iterations $MAXIT \
  --agent.save-interval 500 --agent.algorithm.entropy-coef 0.005 --agent.run-name $NAME > logs_$NAME.log 2>&1 &
TRAIN=$!
$PY scripts/gate_curve.py walk "logs/rsl_rl/bdx_walking/*_$NAME" --watch --pid $TRAIN --patience 2 \
  --out results/gates/$NAME.json > logs_gate_$NAME.log 2>&1 || true
wait $TRAIN || true
