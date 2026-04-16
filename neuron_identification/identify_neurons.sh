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
export N_PER_DATASET="1"
export TOOL_DATASET="emrecanacikgoz/ToolRL"
export TOOL_DATASET_SPLIT="test"
export NONTOOL_DATASET="HuggingFaceFW/fineweb"
export NONTOOL_DATASET_CONFIG="default"

# ─── Analysis + plots ─────────────────────────────────────────────────────────
# TOP_K: number of top-ranked neurons kept in the top-K CSVs and in the bar
export TOP_K="15"
export CORR_VLIM="0.5"

# ─── Output ───────────────────────────────────────────────────────────────────
# OUTPUT_DIR: where CSVs, SVG plots, prompts.csv, run_config.json, and
# (optionally) the raw activations cache are written. Defaults to a per-run
# folder named from the crosscoder variant + sample count, e.g.
#   neuron_identification/runs/dfc-D8k-excl10-freeexcl-k160_n1000
# Override by setting OUTPUT_DIR before invoking this script.
CROSSCODER_SHORT="${CROSSCODER_NAME##*/}"
export OUTPUT_DIR="${OUTPUT_DIR:-neuron_identification/runs/${CROSSCODER_SHORT}_n${N_PER_DATASET}}"
# SEED: seeds Python, numpy, and torch. Controls dataset sampling order.
export SEED="42"
# CACHE_ACTIVATIONS: "1" to save raw activations to a .npz file for faster re-running with the same prompts.
export CACHE_ACTIVATIONS="1"

# ─── Run ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
python "$SCRIPT_DIR/identify_neurons.py"
