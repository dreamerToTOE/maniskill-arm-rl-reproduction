#!/usr/bin/env bash
# Full PickCube PPO run — ADAPTATION (not the literal official baseline):
#   script     ppo.py (readable, pinned blob 66802de0...)
#   task       PickCube-v1 / panda / state / pd_joint_delta_pos / physx_cuda
#   envs       2048 (official uses 4096) · num_steps 50 (official fast uses 4)
#   budget     50,000,000 env steps / seed · 3 seeds (9351 4796 1788, official set)
#   eval_freq  25 (upstream default) · no video · no wandb · no cudagraphs
#
# One supervised process per seed; stops on the first non-zero exit. Each seed
# writes its own summary and log so a failure cannot destroy earlier evidence.
#
# Expected wall time: ~40-45 min/seed (loop SPS ~22k at 2048 envs), ~2h total.
set -u
cd "$(dirname "$0")/.."
PY=./.venv-maniskill/bin/python
SEEDS="9351 4796 1788"
NUM_ENVS=2048
TOTAL=50000000
EVAL_FREQ=25
WALL_LIMIT=5400   # 90 min per seed, generous over the ~40 min expectation

FAILED=0
for s in $SEEDS; do
  echo "===== seed $s ====="
  "$PY" scripts/m2_stage_b_pilot.py \
    --num-envs "$NUM_ENVS" \
    --seed "$s" \
    --total-timesteps "$TOTAL" \
    --eval-freq "$EVAL_FREQ" \
    --wall-limit "$WALL_LIMIT" \
    --exp-name "PickCube-v1__ppo__${s}__n2048_50M" \
    --summary "results/m2_full_n2048_s${s}.json" \
    --log "results/raw/m2_full_n2048_s${s}.log"
  rc=$?
  echo "seed $s exit=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "STOPPING: seed $s failed with exit $rc"
    FAILED=1
    break
  fi
done
echo "full_run_done failed=$FAILED"
exit "$FAILED"
