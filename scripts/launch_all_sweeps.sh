#!/usr/bin/env bash
# scripts/launch_all_sweeps.sh
# Spawn detached tmux sessions running run_steering_sweep.sh per recommended model.
#
# Usage:
#   bash scripts/launch_all_sweeps.sh                        # default 3 GPU layout
#   GPUS="0,0,0" bash scripts/launch_all_sweeps.sh           # all 3 sweeps on GPU 0 (sequential within tmux)
#
# Sessions: steer_dfc160, steer_cc45, steer_dfc45
# Detach with Ctrl-b d. Reattach: tmux a -t steer_dfc160
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-0,1,2}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"

declare -a MODELS=(
  "dfc-D8k-excl10-freeexcl-k160"
  "cc-D8k-nol1-k45"
  "dfc-D8k-excl10-k45"
)
declare -a SESSIONS=("steer_dfc160" "steer_cc45" "steer_dfc45")

mkdir -p logs

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  session="${SESSIONS[$i]}"
  gpu="${GPU_ARR[$i]:-${GPU_ARR[-1]}}"   # fall back to last GPU if list shorter

  rankings="data/rankings/${model}.csv"
  out="results/targeted_steering/${model}"
  log="logs/sweep_${model}.log"

  if [[ ! -f "$rankings" ]]; then
    echo "WARNING: rankings missing for $model → $rankings (skipping)" >&2
    continue
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists; skipping (kill it first to relaunch)"
    continue
  fi

  cmd="CUDA_VISIBLE_DEVICES=${gpu} bash run_steering_sweep.sh \
       --crosscoder antebe1/${model} \
       --rankings ${rankings} \
       --out ${out} \
       --device cuda \
       2>&1 | tee -a ${log}"
  tmux new -d -s "$session" "$cmd; echo '[done] $session'; bash"
  echo "launched $session on GPU $gpu  (model=$model)"
done

echo
echo "Sessions:"
tmux ls 2>/dev/null | grep -E "^(steer_dfc160|steer_cc45|steer_dfc45):" || true
echo "Tail logs/sweep_<model>.log to monitor."
