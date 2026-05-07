#!/usr/bin/env bash
# run_all.sh — single-entry driver for the full pipeline.
#
# Stages:
#   1. setup    : verify models.jsonl + sweep_eval.py in place
#   2. rank     : rank features for each model in MODELS (skipped if CSVs already exist)
#   3. cache    : build feature cache for the autointerp target ($AUTOINTERP_MODEL)
#   4. parallel : launch one tmux sweep session per model + one autointerp session,
#                 round-robin across $GPUS
#   5. wait     : block until all launched sessions exit
#   6. post     : umap → cluster meta → figures → report
#
# Configure the model list by editing $MODELS at the top of this file, or
# by passing --models, or via the MODELS env var:
#
#   bash run_all.sh                                          # default 3-model sweep
#   bash run_all.sh --models antebe1/foo,antebe1/bar         # custom list
#   MODELS="antebe1/foo antebe1/bar" bash run_all.sh         # env var (space-separated)
#   AUTOINTERP_MODEL="antebe1/foo" bash run_all.sh           # which DFC gets autointerp
#   AUTOINTERP_MODEL="" bash run_all.sh                      # skip autointerp/UMAP entirely
#   bash run_all.sh --smoke                                  # 2-prompt sweeps + 5-feat autointerp
#   bash run_all.sh --resume                                 # skip stages 1-5, only post
#   GPUS="0,1,2,3" bash run_all.sh                           # custom GPU layout
#   SINGLE_GPU=1 bash run_all.sh                             # serialise on one GPU
#   bash run_all.sh --two-gpu                                # alternate models across 2 GPUs
#
# Detach the tmux sessions any time with Ctrl-b d. Reattach via
#   tmux a -t sweep_<name>
# Logs land in logs/<stage>.log.
set -euo pipefail
cd "$(dirname "$0")"

source /home/cs29824/.venv/bin/activate

# Faster HF downloads (parallel byte chunks) + per-file byte tqdm bars through `tee`.
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

# ── Config ────────────────────────────────────────────────────────────────────

# Default model list — sbhokare layer sweep of dfc-D8k-excl10-k45 across
# Qwen2.5-3B layers (full HF names: <user>/<short>).
# Override via --models or the MODELS env var to run a subset.
DEFAULT_MODELS=(
  "sbhokare/dfc-D8k-excl10-k45-l1"
  "sbhokare/dfc-D8k-excl10-k45-l5"
  "sbhokare/dfc-D8k-excl10-k45-l9"
  "sbhokare/dfc-D8k-excl10-k45-l14"
  "sbhokare/dfc-D8k-excl10-k45-l18"
  "sbhokare/dfc-D8k-excl10-k45-l20"
  "sbhokare/dfc-D8k-excl10-k45-l24"
  "sbhokare/dfc-D8k-excl10-k45-l28"
  "sbhokare/dfc-D8k-excl10-k45-l32"
)

# AUTOINTERP_MODEL: which model to run feature_cache + autointerp + UMAP +
# cluster_meta on. Must be a DFC (needs an a_excl partition). Set to "" to skip.
AUTOINTERP_MODEL="${AUTOINTERP_MODEL:-antebe1/dfc-D8k-excl10-freeexcl-k160}"

GPUS="${GPUS:-0,1}"
SINGLE_GPU="${SINGLE_GPU:-0}"
TWO_GPU="${TWO_GPU:-0}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"
N_PROMPTS="${N_PROMPTS:-40}"
N_RANK_PROMPTS="${N_RANK_PROMPTS:-4000}"

MODELS_OVERRIDE_SET=0
declare -a MODELS_OVERRIDE=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)  SMOKE=1; shift;;
    --resume) RESUME=1; shift;;
    --single-gpu) SINGLE_GPU=1; shift;;
    --two-gpu)    TWO_GPU=1; GPUS="${GPUS_OVERRIDE:-0,1}"; shift;;
    --models)
      IFS=',' read -ra MODELS_OVERRIDE <<< "$2"
      MODELS_OVERRIDE_SET=1
      shift 2;;
    -h|--help) sed -n '1,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

# Resolve MODELS: --models > $MODELS env var > DEFAULT_MODELS
if [[ "$MODELS_OVERRIDE_SET" == "1" ]]; then
  MODELS=("${MODELS_OVERRIDE[@]}")
elif [[ -n "${MODELS:-}" ]]; then
  # Env-var form: space-separated string. Convert to array.
  read -ra MODELS <<< "$MODELS"
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: MODELS list is empty" >&2; exit 1
fi

if [[ "$SMOKE" == "1" ]]; then
  N_PROMPTS=2
  echo "[smoke] running with N_PROMPTS=$N_PROMPTS (sanity-check mode)"
fi

# ── GPU layout ────────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}
LAST_GPU="${GPU_ARR[$((N_GPUS - 1))]}"

# gpu_for_idx <i>: round-robin GPU assignment for the i-th model in MODELS.
gpu_for_idx() {
  local idx=$1
  if [[ "$SINGLE_GPU" == "1" ]]; then echo "${GPU_ARR[0]}"; return; fi
  if [[ "$TWO_GPU" == "1" ]]; then
    echo "${GPU_ARR[$((idx % 2))]}"; return
  fi
  echo "${GPU_ARR[$((idx % N_GPUS))]}"
}

# Where rank + autointerp + UMAP + cluster_meta run.
if [[ "$SINGLE_GPU" == "1" ]]; then
  GPU_RANK="${GPU_ARR[0]}"
  GPU_AUTOINTERP="${GPU_ARR[0]}"
elif [[ "$TWO_GPU" == "1" ]]; then
  GPU_RANK="${GPU_ARR[0]}"
  GPU_AUTOINTERP="${GPU_ARR[1]:-${GPU_ARR[0]}}"
else
  GPU_RANK="${GPU_ARR[0]}"
  GPU_AUTOINTERP="${LAST_GPU}"
fi

mkdir -p logs results/figures results/clusters data/rankings
echo "GPUs: rank=$GPU_RANK  autointerp=$GPU_AUTOINTERP  sweep=round-robin($GPUS)"
echo "Models (${#MODELS[@]}):"
for m in "${MODELS[@]}"; do echo "  - $m"; done
echo "Autointerp target: ${AUTOINTERP_MODEL:-<disabled>}"

# Session list — populated by stage_parallel, consumed by stage_wait
declare -a SESSIONS=()

# safe_session_name <short> — turn an HF short name into a tmux-safe session id
safe_session_name() {
  echo "sweep_$(echo "$1" | tr -c 'a-zA-Z0-9' '_')"
}

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

# ── Stage 2 — rank features for each model in MODELS ─────────────────────────
# Uses rank_features_sweep.py to amortize the LLM forward pass: one shared
# pass over the seeded rank prompts snapshots `hidden_states[L+1]` for every
# unique layer in the sweep, then each crosscoder is encoded against its own
# layer's activations. ~Nx faster than looping rank_features.py per model.
stage_rank() {
  echo "[stage 2] rank features (${#MODELS[@]} models, shared LLM forwards)"
  local list
  list=$(printf "%s," "${MODELS[@]}")
  list="${list%,}"  # strip trailing comma
  local log="logs/rank_sweep.log"
  CUDA_VISIBLE_DEVICES="$GPU_RANK" python rank_features_sweep.py \
    --crosscoders "$list" \
    --out-dir data/rankings \
    --n-tool "$N_RANK_PROMPTS" --n-nontool "$N_RANK_PROMPTS" \
    --skip-existing 2>&1 | tee "$log"
}

# ── Stage 3 — build feature cache for the autointerp target ──────────────────
# Reads the autointerp crosscoder's training layer from its hparams.json
# (via a tiny inline python helper) and points --source-cache at the matching
# layer-aware raw cache: cache/toolrl_l<L>.
stage_cache() {
  if [[ -z "$AUTOINTERP_MODEL" ]]; then
    echo "[stage 3] skipped (AUTOINTERP_MODEL is empty)"; return
  fi
  local short="${AUTOINTERP_MODEL#*/}"
  local out="cache/${short}_features_toolrl"
  echo "[stage 3] build feature cache for ${short}"
  if [[ -f "$out/meta.json" ]]; then
    echo "  ✓ $out/meta.json exists, skipping"; return
  fi
  # Resolve the layer from hparams.json on the HF repo.
  local layer
  layer=$(uv run python -c "
from huggingface_hub import hf_hub_download
import json, sys
try:
    p = hf_hub_download(repo_id='${AUTOINTERP_MODEL}', filename='hparams.json')
    print(int(json.load(open(p))['layer']))
except Exception as e:
    print(13, file=sys.stderr); print(13)  # fall back to legacy default
")
  local source_cache="cache/toolrl_l${layer}"
  if [[ ! -f "${source_cache}/meta.json" ]]; then
    echo "  ⚠ raw cache ${source_cache}/meta.json missing — run run_cache.py --layer ${layer} first"
    return
  fi
  local log="logs/feature_cache.log"
  CUDA_VISIBLE_DEVICES="$GPU_AUTOINTERP" python build_feature_cache.py \
      --crosscoder "$AUTOINTERP_MODEL" \
      --source-cache "$source_cache" --out "$out" 2>&1 | tee "$log"
}

# ── Stage 4 — launch one tmux sweep per model + one autointerp session ───────
launch_sweep() {
  local full="$1" gpu="$2" session="$3"
  local short="${full#*/}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "  $session: tmux session already exists — skipping launch"
    return
  fi
  local csv="data/rankings/${short}.csv"
  if [[ ! -f "$csv" ]]; then
    echo "  $session: SKIP (rankings missing: $csv)"; return
  fi
  local log="logs/sweep_${short}.log"
  local out="results/targeted_steering/${short}"
  local cmd="source /home/cs29824/.venv/bin/activate && \
       CUDA_VISIBLE_DEVICES=${gpu} python run_steering_eval.py \
       --crosscoder ${full} \
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
  if [[ -z "$AUTOINTERP_MODEL" ]]; then
    echo "  autointerp: skipped (AUTOINTERP_MODEL is empty)"; return
  fi
  local full="$AUTOINTERP_MODEL"
  local short="${full#*/}"
  local session="autointerp"
  local out="results/autointerp_local/${short}"
  # Idempotent: if a previous run already wrote summary.json, don't re-launch.
  if [[ -f "${out}/summary.json" ]]; then
    echo "  autointerp: ${out}/summary.json exists, skipping"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "  $session: tmux session already exists — skipping launch"
    return
  fi
  local feat_cache="cache/${short}_features_toolrl"
  local log="logs/autointerp.log"
  local subset_arg=""
  if [[ "$SMOKE" == "1" ]]; then
    # In smoke mode, leave the ranking floor permissive but cap at 5 features
    # via fire_rate_floor — easiest knob without a separate CLI flag.
    subset_arg="--fire-rate-floor 0.999"
  fi
  local cmd="source /home/cs29824/.venv/bin/activate && \
       CUDA_VISIBLE_DEVICES=${GPU_AUTOINTERP} python run_autointerp_local.py \
       --crosscoder ${full} \
       --feat-cache ${feat_cache} \
       --rankings data/rankings/${short}.csv \
       --out ${out} \
       --partition a_excl \
       --device cuda ${subset_arg} \
       2>&1 | tee -a ${log}; \
       echo '[done] ${session}'"
  tmux new -d -s "$session" "$cmd; exec bash"
  echo "  ✓ launched $session  (gpu=$GPU_AUTOINTERP)"
  SESSIONS+=("$session")
}

stage_parallel() {
  # One serial-worker tmux session per GPU. Each worker loops through its
  # round-robin slice of MODELS, running run_steering_eval.py one at a time
  # so we never load >1 LLM-pair (~12 GiB fp16) per GPU concurrently.
  # Total wall time ≈ ⌈N_models / N_GPUs⌉ × per-sweep time.
  echo "[stage 4] launch sweeps (one serial worker per GPU)"

  # Build per-GPU queues by round-robin over MODELS.
  declare -A queue
  for gpu in "${GPU_ARR[@]}"; do queue[$gpu]=""; done
  local idx=0
  for full in "${MODELS[@]}"; do
    local g; g=$(gpu_for_idx "$idx")
    queue[$g]+="${full} "
    idx=$((idx + 1))
  done

  for gpu in "${GPU_ARR[@]}"; do
    local q="${queue[$gpu]}"
    [[ -z "$q" ]] && continue
    local session="sweep_gpu${gpu}"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "  $session: tmux session already exists — skipping launch"
      SESSIONS+=("$session")
      continue
    fi
    # Worker: sequential loop over the queue. A single run_steering_eval.py
    # invocation already implements per-cell resume, so partial cells from a
    # prior crash will be picked up on next start.
    local cmd="source /home/cs29824/.venv/bin/activate; \
for full in ${q}; do \
  short=\${full#*/}; \
  csv=\"data/rankings/\${short}.csv\"; \
  if [[ ! -f \"\$csv\" ]]; then echo \"  SKIP \$short — no rankings\"; continue; fi; \
  out=\"results/targeted_steering/\${short}\"; \
  log=\"logs/sweep_\${short}.log\"; \
  echo \"=== \$short on gpu ${gpu} (\$(date +%H:%M:%S)) ===\"; \
  CUDA_VISIBLE_DEVICES=${gpu} python run_steering_eval.py \
    --crosscoder \"\$full\" --rankings \"\$csv\" --out \"\$out\" \
    --device cuda --n-prompts ${N_PROMPTS} 2>&1 | tee -a \"\$log\" \
    || echo \"  FAILED \$short — moving on\"; \
done; \
echo \"[done] ${session}\""
    tmux new -d -s "$session" "$cmd; exec bash"
    SESSIONS+=("$session")
    echo "  ✓ launched $session  (gpu=$gpu  queue=$q)"
  done

  launch_autointerp
  echo "  monitor: tmux ls   (attach with tmux a -t sweep_gpu<N>)"
}

# ── Stage 5 — block until tmux sessions exit ─────────────────────────────────
stage_wait() {
  echo "[stage 5] waiting for tmux sessions to exit (Ctrl-C aborts the wait, not the jobs)"
  if [[ ${#SESSIONS[@]} -eq 0 ]]; then
    echo "  ✓ no sessions launched, nothing to wait on"; return
  fi
  while true; do
    local alive=()
    for s in "${SESSIONS[@]}"; do
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

  if [[ -n "$AUTOINTERP_MODEL" ]]; then
    local short="${AUTOINTERP_MODEL#*/}"

    # UMAP — needs the autointerp model loaded
    echo "  → UMAP for ${short}"
    CUDA_VISIBLE_DEVICES="$GPU_AUTOINTERP" python run_umap.py \
        --crosscoder "$AUTOINTERP_MODEL" \
        --out results/figures \
        --clusters-dir results/clusters 2>&1 | tee logs/umap.log

    # Cluster meta-autointerp — depends on autointerp results
    local ai_dir="results/autointerp_local/${short}"
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
  else
    echo "  ⚠ skipping UMAP + cluster_meta (AUTOINTERP_MODEL empty)"
  fi

  # Heatmaps + paper replications + line plots — aggregates over all sweep dirs
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
