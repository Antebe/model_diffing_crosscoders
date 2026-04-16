#!/usr/bin/env bash
# Configuration for neuron_identification/identify_neurons.py
# Edit values below and run:  bash neuron_identification/identify_neurons.sh

set -euo pipefail

# ─── Crosscoder ───────────────────────────────────────────────────────────────
export CROSSCODER_NAME="antebe1/dfc-D8k-excl10-freeexcl-k160"
export MODELS_JSONL="models.jsonl"

# ─── LLMs ─────────────────────────────────────────────────────────────────────
export MODEL_A_ID="chengq9/ToolRL-Qwen2.5-3B"
export MODEL_B_ID="Qwen/Qwen2.5-3B"

# ─── Activation extraction ────────────────────────────────────────────────────
export HIDDEN_STATES_IDX="14"
export MAX_LENGTH="512"
# DEVICE: where to place LLMs + crosscoder. cuda | mps | cpu.
export DEVICE="mps"

# ─── Datasets ─────────────────────────────────────────────────────────────────
# N_PER_DATASET: number of prompts sampled per corpus. 
export N_PER_DATASET="1000"
export TOOL_DATASET="emrecanacikgoz/ToolRL"
export TOOL_DATASET_SPLIT="train"
export NONTOOL_DATASET="HuggingFaceFW/fineweb"
export NONTOOL_DATASET_CONFIG="default"

# ─── Analysis + plots ─────────────────────────────────────────────────────────
# TOP_K: number of top-ranked neurons kept in the top-K CSVs and in the bar
export TOP_K="15"
export CORR_VLIM="0.5"

# ─── What counts as "fired" ──────────────────────────────────────────────────
# FIRE_THRESHOLD: a feature counts as fired on a prompt when its activation
# exceeds this value. TopK encoding fills its budget even with tiny values, so
# a literal ">0" treats trivial selections as firings. This threshold is used
# for fire_rate_* in the rankings and for which features appear per prompt in
# prompt_activations.jsonl. Raise to see only strong firings (e.g. 0.5, 1.0).
export FIRE_THRESHOLD="0.1"

# ─── Steering-candidate filter ───────────────────────────────────────────────
# Applied symmetrically to produce both tool_* and nontool_* candidate CSVs.
# A feature is a tool candidate when it fires on
#   >= CANDIDATE_MIN_FIRE_RATE of tool prompts   (specialty corpus)
#   <= CANDIDATE_MAX_FIRE_RATE of non-tool prompts (other corpus)
# and vice versa for non-tool candidates. "Fires" here uses FIRE_THRESHOLD
# above. Tighten (e.g. 0.5 / 0.05) for a stricter shortlist, or relax
# (e.g. 0.2 / 0.2) if little passes at small N.
export CANDIDATE_MIN_FIRE_RATE="0.3"
export CANDIDATE_MAX_FIRE_RATE="0.1"

# ─── Output ───────────────────────────────────────────────────────────────────
# Always auto-computes OUTPUT_DIR from crosscoder name + sample count, with a
# numeric suffix (_2, _3, …) if the folder already exists — so repeated runs
# never overwrite each other. Any OUTPUT_DIR inherited from the calling shell
# is deliberately ignored (common source of accidental overwrites). To pin a
# specific path, edit the export line below directly.
unset OUTPUT_DIR
CROSSCODER_SHORT="${CROSSCODER_NAME##*/}"
BASE="neuron_identification/runs/${CROSSCODER_SHORT}_n${N_PER_DATASET}"
CANDIDATE="$BASE"
idx=2
while [[ -e "$CANDIDATE" ]]; do
    CANDIDATE="${BASE}_${idx}"
    idx=$((idx + 1))
done
export OUTPUT_DIR="$CANDIDATE"
# SEED: seeds Python, numpy, and torch. Controls dataset sampling order.
export SEED="42"
# CACHE_ACTIVATIONS: "1" to save raw activations to a .npz file for faster re-running with the same prompts.
export CACHE_ACTIVATIONS="1"

# ─── Run ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
python "$SCRIPT_DIR/identify_neurons.py"
