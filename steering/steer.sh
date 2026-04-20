#!/usr/bin/env bash
# Configuration for steering/steer.py
# Edit values below and run:  bash steering/steer.sh

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
export MAX_NEW_TOKENS="512"
# LAYER: transformer layer to patch during generation (0-indexed).
export LAYER="13"
# DEVICE: where to place LLMs + crosscoder. cuda | mps | cpu.
export DEVICE="cuda"

# ─── Steering mode ────────────────────────────────────────────────────────────
# STEER_MODE: "scale" or "clamp"
#   scale — multiply each neuron's activation by SCALE_FACTOR
#   clamp — set each neuron's activation to exactly CLAMP_VALUE
export STEER_MODE="scale"

# NEURON_INDICES: comma-separated global feature indices from the crosscoder.
# These are the indices from identify_neurons output (feature_idx column).
#   A-exclusive: 0–818     (ToolRL-specific)
#   B-exclusive: 819–1637  (Base-specific)
#   Shared:      1638–8191
# Example: the top-3 A-exclusive tool neurons.
export NEURON_INDICES="5,12,100"

# SCALE_FACTOR: multiplier for "scale" mode. >1 amplifies, <1 dampens, 0 ablates.
export SCALE_FACTOR="10.0"

# CLAMP_VALUE: fixed activation value for "clamp" mode. The neuron is forced
# to this value regardless of what the crosscoder naturally produces.
# Use a value informed by the mean_tool column from your identification CSVs.
export CLAMP_VALUE="1.0"

# ─── Dataset ──────────────────────────────────────────────────────────────────
# CUSTOM_PROMPT: if set (non-empty), the script ignores the dataset entirely
# and runs steering on this single prompt. 
# Leave empty to use the dataset below instead.
export CUSTOM_PROMPT="What is the weather like in Tokyo right now?"

# Prompts to run steering on. Default: ToolRL (to see if steering helps/hurts
# on tool prompts). For non-tool prompts, switch to e.g. HuggingFaceFW/fineweb.
# Ignored when CUSTOM_PROMPT is set.
export DATASET="emrecanacikgoz/ToolRL"
export DATASET_SPLIT="train"
# DATASET_CONFIG: HuggingFace config name. Leave empty if unused.
export DATASET_CONFIG=""
export N_PROMPTS="2"

# ─── Output ───────────────────────────────────────────────────────────────────
unset OUTPUT_DIR
CROSSCODER_SHORT="${CROSSCODER_NAME##*/}"
BASE="steering/runs/${CROSSCODER_SHORT}_${STEER_MODE}_n${N_PROMPTS}"
CANDIDATE="$BASE"
idx=2
while [[ -e "$CANDIDATE" ]]; do
    CANDIDATE="${BASE}_${idx}"
    idx=$((idx + 1))
done
export OUTPUT_DIR="$CANDIDATE"

export SEED="42"

# ─── Run ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
python "$SCRIPT_DIR/steer.py"
