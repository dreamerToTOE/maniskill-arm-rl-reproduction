#!/usr/bin/env bash
# Render the trained PickCube policies to evaluation videos (mp4).
#
# Runs the pinned upstream ppo.py in --evaluate mode for each seed's final_ckpt.pt.
# The upstream evaluate path wraps the eval envs in RecordEpisode, which renders
# rgb_array frames and writes mp4 videos under <checkpoint_dir>/test_videos/.
#
# Requires the GPU -> run in your terminal, not the agent (agent has no GPU).
# 8 eval envs x 50 steps per seed = 8 short clips per seed (~1.7 s each at 30 fps).
set -u
cd "$(dirname "$0")/.."
PY=./.venv-maniskill/bin/python
PPO=/home/ubuntu2004/.cache/ms-upstream-ppo/ppo.py

for s in 9351 4796 1788; do
  CKPT="runs/PickCube-v1__ppo__${s}__n2048_50M/final_ckpt.pt"
  [ -f "$CKPT" ] || { echo "missing $CKPT"; continue; }
  echo "===== rendering seed $s ====="
  "$PY" "$PPO" --env-id PickCube-v1 --evaluate --capture-video --checkpoint "$CKPT"
done

echo
echo "videos written under:"
for s in 9351 4796 1788; do
  d="runs/PickCube-v1__ppo__${s}__n2048_50M/test_videos"
  [ -d "$d" ] && echo "  $d  ($(ls "$d" 2>/dev/null | wc -l) files)"
done
