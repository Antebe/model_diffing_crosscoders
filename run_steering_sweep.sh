#!/usr/bin/env bash
# run_steering_sweep.sh
# Thin wrapper around run_steering_eval.py for one crosscoder.
# Designed to be launched inside a tmux session with a chosen CUDA device.
#
# Usage:
#   bash run_steering_sweep.sh \
#        --crosscoder antebe1/dfc-D8k-excl10-freeexcl-k160 \
#        --rankings  data/rankings/dfc-D8k-excl10-freeexcl-k160.csv \
#        --out       results/targeted_steering/dfc-D8k-excl10-freeexcl-k160
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
uv run python run_steering_eval.py "$@"
