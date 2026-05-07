#!/usr/bin/env bash
# run_ablation_l13.sh — DFC partition ablation + DFC-vs-CC at layer 13.
#
# Three targeted-steering sweeps:
#   1. dfc-shared : DFC, top-k% of shared partition  (GPU 0)
#   2. dfc-bexcl  : DFC, top-k% of B-exclusive partition  (GPU 1)
#   3. cc         : CrossCoder cc-D8k-k45 (full dict)  (sequential, after one frees)
#
# Existing dfc-D8k-excl10-k45-l13 (A-excl) sweep is reused as the baseline.
#
# Usage:
#   bash run_ablation_l13.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DFC_FULL_RANK="data/rankings/dfc-D8k-excl10-k45-full.csv"
CC_RANK="data/rankings/cc-D8k-k45.csv"

if [[ ! -f "$DFC_FULL_RANK" ]]; then
  echo "missing $DFC_FULL_RANK — run rank_features.py --rank-all-features first" >&2
  exit 1
fi
if [[ ! -f "$CC_RANK" ]]; then
  echo "missing $CC_RANK — run rank_features.py first" >&2
  exit 1
fi

run_sweep_bg() {
  local gpu="$1" partition="$2" out_suffix="$3" model="$4" rank_csv="$5" tag="$6"
  local out="results/targeted_steering/${out_suffix}"
  local log="logs/sweep_${tag}.log"
  echo "[launch] tag=${tag} gpu=${gpu} partition=${partition} → ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    uv run python run_steering_eval.py \
      --crosscoder "antebe1/${model}" \
      --rankings "${rank_csv}" \
      --partition "${partition}" \
      --out "${out}" \
      --n-prompts 40 \
      --device cuda 2>&1 | tee -a "${log}" &
  echo "  pid=$! log=${log}"
}

run_sweep_bg 0 shared dfc-D8k-excl10-k45-l13-shared dfc-D8k-excl10-k45 "$DFC_FULL_RANK" dfc_shared
PID_SHARED=$!
run_sweep_bg 1 b-excl dfc-D8k-excl10-k45-l13-bexcl  dfc-D8k-excl10-k45 "$DFC_FULL_RANK" dfc_bexcl
PID_BEXCL=$!

echo "Waiting for one DFC sweep to free a GPU before launching CC…"
wait -n
echo "  one DFC sweep finished; launching CC on freed GPU"

# Find which is still alive; CC takes the dead one's GPU.
if kill -0 "$PID_SHARED" 2>/dev/null; then
  CC_GPU=1
else
  CC_GPU=0
fi
echo "  CC will use GPU ${CC_GPU}"

run_sweep_bg "${CC_GPU}" all cc-D8k-k45 cc-D8k-k45 "$CC_RANK" cc_full

wait
echo "[done] all ablation sweeps complete"
