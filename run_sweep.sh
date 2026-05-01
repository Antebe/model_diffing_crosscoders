#!/usr/bin/env bash
# run_sweep.sh — Train 48 models (DFC + CrossCoder) and upload to HF.
#
# Sweep axes:
#   k:              [45, 90, 160]
#   dict_size:      [8192, 16384]
#   DFC excl%:      [0.03, 0.05, 0.10] x excl_sparsity [1e-3, 0]
#   CrossCoder:     sparsity [1e-3, 0]  (excl% = 0)
#
# Usage:
#   tmux new -s sweep
#   cd /home/cs29824/andre/xcdr3
#   bash run_sweep.sh 2>&1 | tee sweep.log
set -euo pipefail

cd "$(dirname "$0")"

# Fix wandb import shadow — always move wandb/ contents into wandb_runs/
if [ -d "./wandb" ]; then
    mkdir -p ./wandb_runs
    cp -a ./wandb/* ./wandb_runs/ 2>/dev/null || true
    rm -rf ./wandb
fi

# Prevent wandb from creating a wandb/ dir that shadows the Python package
export WANDB_DIR="./wandb_runs"
mkdir -p ./wandb_runs

# ── Config ──────────────────────────────────────────────────────────────
HF_USER="antebe1"
WANDB_PROJECT="dfc-crosscoder-sweep"
STEPS=9000
LR="1e-4"
TRAIN_BATCH=1024
LAYER=13
ACTIVATION_DIM=2048
MODEL_A="chengq9/ToolRL-Qwen2.5-3B"
MODEL_B="Qwen/Qwen2.5-3B"

COMPLETED=0
TOTAL=48

banner() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="
}

write_model_card() {
    local save_path="$1" name="$2" arch_type="$3" k="$4"
    local excl_a="$5" excl_b="$6" sparsity="$7" excl_sparsity="$8" dict_size="$9"

    if [ "$arch_type" = "DFC" ]; then
        if [ "$excl_sparsity" = "0" ]; then
            local loss_desc="MSE + L1 on shared features only (coef: ${sparsity}); exclusive features unpenalized"
        else
            local loss_desc="MSE + L1 sparsity (shared: ${sparsity}, exclusive: ${excl_sparsity})"
        fi
        local arch_desc="Dedicated Feature CrossCoder with partitioned dictionary (${excl_a}/${excl_b} A/B exclusive)"
        local n_excl_a=$(python3 -c "print(int(${dict_size} * ${excl_a}))")
        local n_excl_b=$(python3 -c "print(int(${dict_size} * ${excl_b}))")
        local n_shared=$(python3 -c "print(${dict_size} - int(${dict_size} * ${excl_a}) - int(${dict_size} * ${excl_b}))")
        local partition_desc="| A-exclusive features | ${n_excl_a} (${excl_a}) |
| B-exclusive features | ${n_excl_b} (${excl_b}) |
| Shared features | ${n_shared} |"
    else
        if [ "$sparsity" = "0" ]; then
            local loss_desc="MSE only (top-k enforces sparsity structurally, no L1 penalty)"
        else
            local loss_desc="MSE + L1 sparsity (coef: ${sparsity})"
        fi
        local arch_desc="Standard CrossCoder — all ${dict_size} features shared, no partition masks"
        local partition_desc="| Partitioning | None — all features shared |"
    fi

    # ── Copy supporting files ──────────────────────────────────────────
    cp dfc.py "${save_path}/dfc.py"
    cp demo.py "${save_path}/demo.py"
    cp requirements.txt "${save_path}/requirements.txt" 2>/dev/null || true

    # ── hparams.json ───────────────────────────────────────────────────
    cat > "${save_path}/hparams.json" <<HEOF
{
  "architecture": "${arch_type}",
  "model_a": "${MODEL_A}",
  "model_b": "${MODEL_B}",
  "layer": ${LAYER},
  "activation_dim": ${ACTIVATION_DIM},
  "dict_size": ${dict_size},
  "k": ${k},
  "model_a_exclusive_pct": ${excl_a},
  "model_b_exclusive_pct": ${excl_b},
  "steps": ${STEPS},
  "lr": ${LR},
  "train_batch": ${TRAIN_BATCH},
  "sparsity_coef": ${sparsity},
  "exclusive_sparsity_coef": ${excl_sparsity},
  "loss": "${loss_desc}",
  "wandb_project": "${WANDB_PROJECT}",
  "trained_at": "$(date -Iseconds)"
}
HEOF

    # ── README.md (HF model card) ─────────────────────────────────────
    cat > "${save_path}/README.md" <<MEOF
---
tags:
  - sparse-autoencoder
  - crosscoder
  - interpretability
  - qwen2
  - mechanistic-interpretability
  - dictionary-learning
license: mit
---

# ${name}

A **${arch_type}** sparse crosscoder trained to compare layer-${LAYER} activations between:
- **Model A (ToolRL)**: \`${MODEL_A}\` — fine-tuned with tool-use reinforcement learning
- **Model B (Base)**: \`${MODEL_B}\` — vanilla base model

## What is this?

This model learns a sparse dictionary of features from the internal representations of two language models. By comparing which features activate for which model, we can identify:
- **What the ToolRL fine-tuning changed** (A-exclusive features)
- **What remained the same** (shared features)
- **What the base model does that ToolRL suppressed** (B-exclusive features)

## Model Architecture

${arch_desc}

| Parameter | Value |
|-----------|-------|
| Dictionary size | ${dict_size} |
| Top-k active features | ${k} |
| Layer | ${LAYER} (middle layer of Qwen2.5-3B) |
| Activation dimension | ${ACTIVATION_DIM} |
${partition_desc}

### How it works

1. **Encode**: Takes stacked activations \`(batch, 2, ${ACTIVATION_DIM})\` from both models, applies per-model encoder weights, sums across models, and selects the top-${k} features via ReLU + top-k.
2. **Decode**: Reconstructs per-model activations from the sparse feature vector using per-model decoder weights.
3. **Partition masks** (DFC only): Hard binary masks zero out encoder/decoder weights to enforce that exclusive features cannot be used by the wrong model.

## Training

| Parameter | Value |
|-----------|-------|
| Loss function | ${loss_desc} |
| Training steps | ${STEPS} |
| Learning rate | ${LR} |
| Batch size | ${TRAIN_BATCH} |
| Sparsity coefficient (shared) | ${sparsity} |
| Exclusive sparsity coefficient | ${excl_sparsity} |
| Optimizer | Adam (grad clip 1.0) |
| W&B project | \`${WANDB_PROJECT}\` |

### Training Data

- **FineWeb**: ~40,000 general web text samples (from \`HuggingFaceFW/fineweb\` sample-10BT)
- **ToolRL**: ~40,000 tool-use conversation samples (from \`emrecanacikgoz/ToolRL\`, cycled)
- Activations extracted from layer ${LAYER}, last token per sample
- Both datasets concatenated and z-score normalized

## Usage

### Quick Start

\`\`\`python
import torch
from huggingface_hub import hf_hub_download

# Download model files
repo_id = "${HF_USER}/${name}"
for fname in ["model.pt", "config.json", "dfc.py"]:
    hf_hub_download(repo_id=repo_id, filename=fname, local_dir="./model")

# Load the crosscoder
import sys; sys.path.insert(0, "./model")
from dfc import DFCCrossCoder

dfc = DFCCrossCoder.load("./model", device="cuda")
print(f"Loaded: dict_size={dfc.dict_size}, k={dfc.k}")
\`\`\`

### Extract Features from Real Models

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load both models
model_a = AutoModelForCausalLM.from_pretrained("${MODEL_A}", device_map="cuda:0")
model_b = AutoModelForCausalLM.from_pretrained("${MODEL_B}", device_map="cuda:1")
tokenizer = AutoTokenizer.from_pretrained("${MODEL_B}")

# Get activations from layer ${LAYER}
# NOTE: hidden_states[0] = embeddings, hidden_states[i] = output of layer i-1
#       so layer ${LAYER} activations are at index ${LAYER}+1
text = "Use the search tool to find recent papers on RLHF"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    out_a = model_a(**inputs.to("cuda:0"), output_hidden_states=True)
    out_b = model_b(**inputs.to("cuda:1"), output_hidden_states=True)
    act_a = out_a.hidden_states[${LAYER} + 1][:, -1, :]  # last token, layer ${LAYER}
    act_b = out_b.hidden_states[${LAYER} + 1][:, -1, :]

# Stack and encode
activations = torch.stack([act_a.cpu(), act_b.cpu()], dim=1)  # (1, 2, ${ACTIVATION_DIM})
features = dfc.encode(activations.to(dfc.W_enc.device))

print(f"Active features: {(features > 0).sum().item()} / {dfc.dict_size}")
\`\`\`

### Analyze Partitions (DFC only)

\`\`\`python
stats = dfc.feature_stats(features)
print(f"L0 total:    {stats['l0_total']:.1f}")
print(f"L0 A-excl:   {stats['l0_a_excl']:.1f}")
print(f"L0 B-excl:   {stats['l0_b_excl']:.1f}")
print(f"L0 shared:   {stats['l0_shared']:.1f}")

# Check reconstruction quality
recon, feats = dfc(activations.to(dfc.W_enc.device))
mse = torch.nn.functional.mse_loss(recon.cpu(), activations)
print(f"Reconstruction MSE: {mse.item():.6f}")
\`\`\`

## Files

| File | Description |
|------|-------------|
| \`model.pt\` | PyTorch state dict (encoder/decoder weights + partition masks) |
| \`config.json\` | Architecture config: dict_size, k, partition sizes (n_a, n_b) |
| \`hparams.json\` | Full training hyperparameters including loss, lr, steps, etc. |
| \`dfc.py\` | \`DFCCrossCoder\` class definition — required to load model.pt |
| \`demo.py\` | Feature extraction demo (works with downloaded model) |
| \`requirements.txt\` | Python dependencies |

## Part of a Sweep

This model is one of 48 models in a hyperparameter sweep. See the full collection:

| Axis | Values |
|------|--------|
| k (top-k) | 45, 90, 160 |
| dict_size | 8,192 / 16,384 |
| Architecture | DFC (partitioned) / CrossCoder (all shared) |
| Exclusive % (DFC) | 3%, 5%, 10% |
| Exclusive sparsity | 1e-3 (penalized) / 0 (free) |
| CrossCoder L1 | with / without |

## Citation

\`\`\`bibtex
@misc{${name},
  title={${arch_type} CrossCoder: ToolRL vs Base Qwen2.5-3B},
  author={Andre Shportko},
  year={2026},
  url={https://huggingface.co/${HF_USER}/${name}}
}
\`\`\`
MEOF

    echo "  Wrote model card + hparams + supporting files to ${save_path}"
}

upload_to_hf() {
    local save_path="$1" repo_name="$2"
    uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
repo_id = '${HF_USER}/${repo_name}'
api.create_repo(repo_id, private=False, exist_ok=True, repo_type='model')
api.upload_folder(folder_path='${save_path}', repo_id=repo_id, commit_message='Upload trained model')
print(f'Uploaded to https://huggingface.co/{repo_id}')
"
}

repo_exists_on_hf() {
    # Returns 0 (true) if model.pt exists in the HF repo, 1 (false) otherwise
    uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    files = api.list_repo_files('${HF_USER}/$1')
    exit(0 if 'model.pt' in files else 1)
except Exception:
    exit(1)
"
}

train_model() {
    local name="$1" arch_type="$2" k="$3"
    local excl_a="$4" excl_b="$5" sparsity="$6" excl_sparsity="$7"
    local device="$8" dict_size="$9"
    local save_path="./checkpoints/${name}"

    # Skip if already uploaded to HF
    if repo_exists_on_hf "$name"; then
        echo "SKIP: ${name} already exists on HF — skipping train + upload"
        COMPLETED=$((COMPLETED + 1))
        return 0
    fi

    banner "[${COMPLETED}/${TOTAL}] ${name} (${arch_type}, k=${k}, D=${dict_size}, dev=${device})"

    rm -rf "$save_path"

    uv run python run_train.py \
        --k "$k" \
        --dict_size "$dict_size" \
        --model_a_excl "$excl_a" \
        --model_b_excl "$excl_b" \
        --sparsity_coef "$sparsity" \
        --exclusive_sparsity_coef "$excl_sparsity" \
        --save_path "$save_path" \
        --steps "$STEPS" \
        --train_batch "$TRAIN_BATCH" \
        --train_device "$device" \
        --wandb_project "$WANDB_PROJECT"

    write_model_card "$save_path" "$name" "$arch_type" "$k" "$excl_a" "$excl_b" "$sparsity" "$excl_sparsity" "$dict_size"
    upload_to_hf "$save_path" "$name"

    COMPLETED=$((COMPLETED + 1))
    banner "Done: ${name} (${COMPLETED}/${TOTAL})"
}

# ── Helper to run a pair on both GPUs ────────────────────────────────────
run_pair() {
    # Args: all args for model A (gpu0), then all args for model B (gpu1)
    train_model "$1" "$2" "$3" "$4" "$5" "$6" "$7" "cuda:0" "$8" &
    local PID0=$!
    train_model "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "cuda:1" "${16}" &
    local PID1=$!
    wait $PID0 || { echo "FAILED: $1"; exit 1; }
    wait $PID1 || { echo "FAILED: $9"; exit 1; }
}

# ══════════════════════════════════════════════════════════════════════════
banner "SWEEP START — ${TOTAL} models, 2 GPUs, batch=${TRAIN_BATCH}"
# ══════════════════════════════════════════════════════════════════════════

for DICT_SIZE in 8192 16384; do
    DS_TAG="D$(( DICT_SIZE / 1024 ))k"

    for K in 45 90 160; do

        # ── DFC: 3 excl% × 2 excl_sparsity = 6 models per (k, dict_size) ──
        # Run as 3 pairs: each pair = same excl% with esp=1e-3 vs esp=0

        for EXCL in 3 5 10; do
            EXCL_F="0.$(printf '%02d' $EXCL)"  # 0.03, 0.05, 0.10
            run_pair \
                "dfc-${DS_TAG}-excl${EXCL}-k${K}"          DFC "$K" "$EXCL_F" "$EXCL_F" 1e-3 1e-3 "$DICT_SIZE" \
                "dfc-${DS_TAG}-excl${EXCL}-freeexcl-k${K}" DFC "$K" "$EXCL_F" "$EXCL_F" 1e-3 0    "$DICT_SIZE"
        done

        # ── CrossCoder: 2 sparsity variants per (k, dict_size) ──────────
        run_pair \
            "cc-${DS_TAG}-k${K}"      CrossCoder "$K" 0.0 0.0 1e-3 0 "$DICT_SIZE" \
            "cc-${DS_TAG}-nol1-k${K}" CrossCoder "$K" 0.0 0.0 0    0 "$DICT_SIZE"

    done
done

banner "ALL ${TOTAL} MODELS COMPLETE"
