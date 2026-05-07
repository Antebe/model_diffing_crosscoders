#!/usr/bin/env bash
# run_missing_dfc.sh — Fill in missing |S| values for DFC partitions at l13.
#
# A-excl (n_tool=8): existing |S|∈{1,2,3,5,7}. Add k%∈{40,70} → |S|∈{4,6}.
#   |S|=8 maps to k=100, but that conflicts with existing l13 cell at |S|=7
#   (old ranking had n_tool=7) — leaving the cap at |S|=7 for A-excl.
#
# Shared (n_tool=38): existing |S|∈{2,4,7}. Add k%∈{2,7,13,15,20,22,25}
#   → |S|∈{1,3,5,6,8,9,10}. Result: full coverage |S|∈{1..10}.
#
# Outputs land in the SAME existing dirs (resume logic skips already-done
# cells) so plot_ablation_l13.py picks up the union.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DFC_FULL_RANK="data/rankings/dfc-D8k-excl10-k45-full.csv"
[[ -f "$DFC_FULL_RANK" ]] || { echo "missing $DFC_FULL_RANK" >&2; exit 1; }

run_bg() {
  local gpu="$1" partition="$2" out_dir="$3" k_list="$4" tag="$5"
  echo "[launch] tag=${tag} gpu=${gpu} partition=${partition} k%=${k_list} → ${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    uv run python run_steering_eval.py \
      --crosscoder antebe1/dfc-D8k-excl10-k45 \
      --rankings "$DFC_FULL_RANK" \
      --partition "$partition" \
      --k-list "$k_list" \
      --out "$out_dir" \
      --n-prompts 40 \
      --device cuda 2>&1 | tee -a "logs/sweep_${tag}.log" &
}

run_bg 0 a-excl  "results/targeted_steering/dfc-D8k-excl10-k45-l13-aexcl-fine" "40,70" dfc_aexcl_fine
run_bg 1 shared  "results/targeted_steering/dfc-D8k-excl10-k45-l13-shared-fine" "2,7,13,15,20,22,25" dfc_shared_fine

wait
echo "[done] missing DFC sweeps complete"
