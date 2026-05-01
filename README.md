# DFC CrossCoder

Dedicated Feature CrossCoder for comparing ToolRL fine-tuned Qwen2.5-3B against the base model.

## Structure

```
dfc_project/
├── config.py       # All hyperparameters — edit this first
├── models.py       # LLM loading / unloading
├── cache.py        # Activation extraction + CachedActivationDataset
├── dfc.py          # DFCCrossCoder model
├── train.py        # Training loop with W&B logging
├── inspect.py      # Analysis tools
├── run_cache.py    # Step 1: extract activations to disk
├── run_train.py    # Step 2: train DFC from cache
└── notebook.ipynb  # Step 3: explore results
```

## Quickstart

```bash
pip install torch transformers datasets wandb tqdm matplotlib

# Step 1 — cache activations (run LLMs once, ~30 min)
python run_cache.py

# Step 2 — train DFC (no LLMs needed, fast)
python run_train.py

# Step 3 — explore
jupyter notebook notebook.ipynb
```

## CLI options

```bash
# Custom layer and sample count
python run_cache.py --layer 20 --fineweb_samples 100000 --toolrl_samples 50000

# Larger dict, no W&B
python run_train.py --dict_size 65536 --k 128 --steps 20000 --no_wandb

# Specific GPU
python run_train.py --train_device cuda:1
```

## HuggingFace Model

🤗 **Model available on HuggingFace**: Upload your trained model with:

```bash
python upload_to_hf.py
```

🚀 **Quick Start** (10 lines):
```python
# See quick_start.py for the minimal example
from dfc import DFCCrossCoder
dfc = DFCCrossCoder.load("./checkpoints/dfc2", device="cuda")
features = dfc.encode(activations)  # Your feature vector!
```

## Demo

```bash
python demo.py  # Interactive demo with feature analysis
```

## W&B metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Total loss (MSE + λ·L1) |
| `train/mse` | Reconstruction MSE |
| `train/l1` | L1 sparsity |
| `train/l0_total` | Mean active features per sample |
| `train/l0_a_excl` | Mean active A-exclusive features |
| `train/l0_b_excl` | Mean active B-exclusive features |
| `train/l0_shared` | Mean active shared features |
| `debug/enc_max_violation` | Partition mask integrity (should stay ~0) |

## Models

| Model | Role | HF ID |
|-------|------|--------|
| Model A | Fine-tuned (ToolRL) | `chengq9/ToolRL-Qwen2.5-3B` |
| Model B | Base | `Qwen/Qwen2.5-3B` |

Layer 13 is a reasonable default; try 16–20 for later representations.

## Cache size

| Samples | Disk (fp16) |
|---------|-------------|
| 50k + 50k | ~0.8 GB |
| 100k + 100k | ~1.6 GB |
