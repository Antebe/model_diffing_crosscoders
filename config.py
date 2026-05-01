"""
config.py — Central configuration for DFC CrossCoder experiments.
Edit this file to change models, layer, dict size, etc.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Models ────────────────────────────────────────────────────────
    model_a_name: str = "chengq9/ToolRL-Qwen2.5-3B"   # fine-tuned (tool-use RL)
    model_b_name: str = "Qwen/Qwen2.5-3B"              # base model
    tokenizer_name: str = "Qwen/Qwen2.5-3B"

    # ── Layer to probe ────────────────────────────────────────────────
    layer: int = 13          # which transformer layer's output to extract
    activation_dim: int = 2048   # Qwen2.5-3B hidden_size (auto-detected at runtime)

    # ── Data sources ─────────────────────────────────────────────────
    fineweb_samples: int = 40_000    # how many FineWeb texts to cache
    toolrl_samples:  int = 40_000    # how many ToolRL texts to cache
    min_text_len:    int = 3        # skip very short texts

    # ── Cache ─────────────────────────────────────────────────────────
    cache_dir:       str = "./cache"
    fineweb_cache:   str = "./cache/fineweb"
    toolrl_cache:    str = "./cache/toolrl"
    shard_size:      int = 2_048     # samples per shard file
    extract_batch:   int = 8         # tokenizer batch for LLM forward pass

    # ── DFC architecture ──────────────────────────────────────────────
    dict_size:              int   = 16_384   # total features (scale to 65536+ for real runs)
    k:                      int   = 90       # top-k active features
    model_a_exclusive_pct:  float = 0.05    # 5% A-only features
    model_b_exclusive_pct:  float = 0.05    # 5% B-only features

    # ── Training ──────────────────────────────────────────────────────
    steps:          int   = 9_000
    lr:             float = 1e-4
    sparsity_coef:  float = 1e-3
    exclusive_sparsity_coef: float = 2e-4
    train_batch:    int   = 64
    num_workers:    int   = 2
    log_every:      int   = 50
    save_path:      str   = "./checkpoints/dfc2"

    # ── Devices ───────────────────────────────────────────────────────
    device_a:    str = "cuda:0"
    device_b:    str = "cuda:1"
    train_device: str = "cuda:0"

    # ── W&B ───────────────────────────────────────────────────────────
    wandb_project:  str  = "dfc-crosscoder-3"
    wandb_entity:   str  = ""        # leave blank to use default entity
    wandb_enabled:  bool = True

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.save_path).mkdir(parents=True, exist_ok=True)


# Default config used by scripts
cfg = Config()
