#!/usr/bin/env bash
# run_all.sh — single-entry driver for the full pipeline.
#
# Stages:
#   1. setup    : verify rankings + models.jsonl in place
#   2. rank     : rank the other 2 models (skipped if CSVs already exist)
#   3. cache    : build feature cache for the lowest-side-effect model
#   4. parallel : launch in 4 tmux sessions:
#                   sweep_dfc160, sweep_cc45, sweep_dfc45, autointerp
#   5. wait     : block until all 4 sessions exit
#   6. post     : umap → cluster meta → figures → report
#
# Usage:
#   bash run_all.sh                 # full pipeline, GPUs 0,1,2,3
#   bash run_all.sh --smoke         # 2-prompt sweeps + 5-feat autointerp
#   bash run_all.sh --resume        # skip stages 1-4 and just do post
#   GPUS="0,1,2,3" bash run_all.sh  # custom GPU layout
#   SINGLE_GPU=1 bash run_all.sh    # serialise everything on one GPU
#   bash run_all.sh --two-gpu       # 2-GPU split: sweep_dfc160 + ranks on GPUS[0],
#                                   #   sweep_dfc45 + autointerp on GPUS[1]
#                                   #   (uses GPUS=0,1; sweep_cc45 pairs with dfc160)
#
# Detach the tmux sessions any time with Ctrl-b d. Reattach via
#   tmux a -t sweep_dfc160
# Logs land in logs/<stage>.log.
set -euo pipefail
cd "$(dirname "$0")"

source /home/cs29824/.venv/bin/activate

# Faster HF downloads (parallel byte chunks) + per-file byte tqdm bars through `tee`.
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

# ── Config ────────────────────────────────────────────────────────────────────
GPUS="${GPUS:-0,1,2,3}"
SINGLE_GPU="${SINGLE_GPU:-0}"
TWO_GPU="${TWO_GPU:-0}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"
N_PROMPTS="${N_PROMPTS:-40}"
N_RANK_PROMPTS="${N_RANK_PROMPTS:-4000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)  SMOKE=1; shift;;
    --resume) RESUME=1; shift;;
    --single-gpu) SINGLE_GPU=1; shift;;
    --two-gpu)    TWO_GPU=1; GPUS="${GPUS_OVERRIDE:-0,1}"; shift;;
    -h|--help) sed -n '1,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ "$SMOKE" == "1" ]]; then
  N_PROMPTS=2
  echo "[smoke] running with N_PROMPTS=$N_PROMPTS (sanity-check mode)"
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [[ "$SINGLE_GPU" == "1" ]]; then
  GPU_DFC160="${GPU_ARR[0]}"
  GPU_CC45="${GPU_ARR[0]}"
  GPU_DFC45="${GPU_ARR[0]}"
  GPU_AUTOINTERP="${GPU_ARR[0]}"
elif [[ "$TWO_GPU" == "1" ]]; then
  # One sweep per GPU. dfc160 (heavy, k=160) on GPUS[0]; dfc45 (lighter) on GPUS[1].
  # cc45 pairs with dfc160 on GPUS[0]; autointerp pairs with dfc45 on GPUS[1].
  # Stage-2 ranks reuse GPU_DFC160 → GPUS[0].
  GPU_DFC160="${GPU_ARR[0]}"
  GPU_CC45="${GPU_ARR[0]}"
  GPU_DFC45="${GPU_ARR[1]:-${GPU_ARR[0]}}"
  GPU_AUTOINTERP="${GPU_ARR[1]:-${GPU_ARR[0]}}"
else
  GPU_DFC160="${GPU_ARR[0]:-0}"
  GPU_CC45="${GPU_ARR[1]:-${GPU_ARR[0]}}"
  GPU_DFC45="${GPU_ARR[2]:-${GPU_ARR[0]}}"
  GPU_AUTOINTERP="${GPU_ARR[3]:-${GPU_ARR[0]}}"
fi

mkdir -p logs results/figures results/clusters
echo "GPUs: dfc160=$GPU_DFC160  cc45=$GPU_CC45  dfc45=$GPU_DFC45  autointerp=$GPU_AUTOINTERP"

# ── Stage 1 — setup checks ────────────────────────────────────────────────────
stage_setup() {
  echo "[stage 1] setup"
  for f in models.jsonl sweep_eval.py; do
    if [[ ! -f "$f" ]]; then
      echo "  MISSING: $f" >&2; exit 1
    fi
  done
  echo "  ✓ inputs present"
}

# ── Stage 2 — rank the other 2 models (~30 min each on GPU) ──────────────────
stage_rank() {
  echo "[stage 2] rank features for non-recommended models"
  for short in "dfc-D8k-excl10-freeexcl-k160" "dfc-D8k-excl10-k45"; do #"cc-D8k-nol1-k45"
    csv="data/rankings/${short}.csv"
    if [[ -f "$csv" ]]; then
      echo "  ✓ $csv exists, skipping"; continue
    fi
    log="logs/rank_${short}.log"
    echo "  → ranking ${short} → $csv  (log: $log)"
    CUDA_VISIBLE_DEVICES="$GPU_DFC160" python rank_features.py \
      --crosscoder "antebe1/${short}" --out "$csv" \
      --n-tool "$N_RANK_PROMPTS" --n-nontool "$N_RANK_PROMPTS" 2>&1 | tee "$log"

  done
}

# ── Stage 3 — build feature cache for autointerp model ───────────────────────
stage_cache() {
  echo "[stage 3] build feature cache for dfc-D8k-excl10-freeexcl-k160"
  out="cache/dfc-D8k-excl10-freeexcl-k160_features_toolrl"
  if [[ -f "$out/meta.json" ]]; then
    echo "  ✓ $out/meta.json exists, skipping"; return
  fi
  log="logs/feature_cache.log"
  CUDA_VISIBLE_DEVICES="$GPU_AUTOINTERP" python build_feature_cache.py \
      --crosscoder antebe1/dfc-D8k-excl10-freeexcl-k160 \
      --source-cache cache/toolrl --out "$out" 2>&1 | tee "$log"
}

# ── Stage 4 — launch parallel tmux sessions ──────────────────────────────────
launch_sweep() {
  local short="$1" gpu="$2" session="$3"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "  $session: tmux session already exists — skipping launch"
    return
  fi
  csv="data/rankings/${short}.csv"
  if [[ ! -f "$csv" ]]; then
    echo "  $session: SKIP (rankings missing: $csv)"; return
  fi
  log="logs/sweep_${short}.log"
  out="results/targeted_steering/${short}"
  cmd="source /home/cs29824/.venv/bin/activate && \
       CUDA_VISIBLE_DEVICES=${gpu} python run_steering_eval.py \
       --crosscoder antebe1/${short} \
       --rankings ${csv} \
       --out ${out} \
       --device cuda \
       --n-prompts ${N_PROMPTS} \
       2>&1 | tee -a ${log}; \
       echo '[done] ${session}'"
  tmux new -d -s "$session" "$cmd; exec bash"
  echo "  ✓ launched $session  (gpu=$gpu  short=$short)"
}

launch_autointerp() {
  local session="autointerp"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "  $session: tmux session already exists — skipping launch"
    return
  fi
  feat_cache="cache/dfc-D8k-excl10-freeexcl-k160_features_toolrl"
  out="results/autointerp_local/dfc-D8k-excl10-freeexcl-k160"
  log="logs/autointerp.log"
  subset_arg=""
  if [[ "$SMOKE" == "1" ]]; then
    # In smoke mode, leave the ranking floor permissive but cap at 5 features
    # via fire_rate_floor — easiest knob without a separate CLI flag.
    subset_arg="--fire-rate-floor 0.999"
  fi
  cmd="source /home/cs29824/.venv/bin/activate && \
       CUDA_VISIBLE_DEVICES=${GPU_AUTOINTERP} python run_autointerp_local.py \
       --crosscoder antebe1/dfc-D8k-excl10-freeexcl-k160 \
       --feat-cache ${feat_cache} \
       --rankings data/rankings/dfc-D8k-excl10-freeexcl-k160.csv \
       --out ${out} \
       --partition a_excl \
       --device cuda ${subset_arg} \
       2>&1 | tee -a ${log}; \
       echo '[done] ${session}'"
  tmux new -d -s "$session" "$cmd; exec bash"
  echo "  ✓ launched $session  (gpu=$GPU_AUTOINTERP)"
}

stage_parallel() {
  echo "[stage 4] launch parallel jobs in tmux"
  launch_sweep "dfc-D8k-excl10-freeexcl-k160" "$GPU_DFC160" "sweep_dfc160"
  launch_sweep "cc-D8k-nol1-k45"              "$GPU_CC45"   "sweep_cc45"
  launch_sweep "dfc-D8k-excl10-k45"           "$GPU_DFC45"  "sweep_dfc45"
  launch_autointerp
  echo "  monitor: tmux ls   (attach with tmux a -t <name>)"
}

# ── Stage 5 — block until tmux sessions exit ─────────────────────────────────
stage_wait() {
  echo "[stage 5] waiting for tmux sessions to exit (Ctrl-C aborts the wait, not the jobs)"
  local sessions=("sweep_dfc160" "sweep_cc45" "sweep_dfc45" "autointerp")
  while true; do
    local alive=()
    for s in "${sessions[@]}"; do
      if tmux has-session -t "$s" 2>/dev/null; then
        alive+=("$s")
      fi
    done
    if [[ ${#alive[@]} -eq 0 ]]; then
      echo "  ✓ all sessions exited"
      return
    fi
    printf "  alive: %s\n" "${alive[*]}"
    sleep 60
  done
}

# ── Stage 6 — post-processing (foreground) ───────────────────────────────────
stage_post() {
  echo "[stage 6] post-processing"

  # UMAP — needs the recommended model loaded
  echo "  → UMAP"
  CUDA_VISIBLE_DEVICES="$GPU_AUTOINTERP" python run_umap.py \
      --crosscoder antebe1/dfc-D8k-excl10-freeexcl-k160 \
      --out results/figures \
      --clusters-dir results/clusters 2>&1 | tee logs/umap.log

  # Cluster meta-autointerp — depends on autointerp results
  ai_dir="results/autointerp_local/dfc-D8k-excl10-freeexcl-k160"
  if [[ -d "$ai_dir" ]] && find "$ai_dir" -name "feat_*.json" -print -quit | grep -q .; then
    echo "  → cluster meta-autointerp"
    CUDA_VISIBLE_DEVICES="$GPU_AUTOINTERP" python run_cluster_meta.py \
        --autointerp "$ai_dir" \
        --clusters-csv results/clusters/aexcl_assignments.csv \
        --out results/clusters/cluster_meta.json \
        --figure-out results/figures/umap_aexcl_clusters_labeled.png \
        --device cuda 2>&1 | tee logs/cluster_meta.log
  else
    echo "  ⚠ skipping cluster meta — no autointerp results in $ai_dir"
  fi

  # Heatmaps + paper replications + line plots
  echo "  → steering figures"
  python build_steering_figures.py \
      --steering-root results/targeted_steering \
      --paper-jsonl "results/results_full (1).jsonl" \
      --out results/figures 2>&1 | tee logs/figures.log

  # Final report
  echo "  → REPORT.md"
  python build_report.py 2>&1 | tee logs/report.log
}

# ── Drive ────────────────────────────────────────────────────────────────────
if [[ "$RESUME" == "1" ]]; then
  echo "[resume] skipping stages 1-5; jumping to post-processing"
  stage_post
  exit 0
fi

stage_setup
stage_rank
stage_cache
stage_parallel
stage_wait
stage_post

echo
echo "============================================================"
echo "  DONE.  See results/REPORT.md and results/figures/"
echo "============================================================"
