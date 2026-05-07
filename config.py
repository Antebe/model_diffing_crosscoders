"""
config.py — Central configuration for DFC CrossCoder experiments.
Edit this file to change models, layer, dict size, etc.

Path conventions
────────────────
All on-disk caches are layered: every path includes ``_l<L>`` so multiple
layers can coexist on disk.

  raw activations        cache/{dataset}_l{L}/
  crosscoder features    cache/{dataset}_features_l{L}/                (one cc per layer)
                         cache/{short}_features_{dataset}_l{L}/        (model-tagged)

Use ``raw_cache_path`` / ``feature_cache_path`` / ``layer_from_hparams``
instead of hand-formatting paths so the convention has a single source of
truth. ``Config.fineweb_cache`` and ``Config.toolrl_cache`` are properties
derived from ``Config.layer`` — assigning ``cfg.layer = L`` automatically
updates both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Path helpers (single source of truth for cache layout) ────────────────

def raw_cache_path(dataset: str, layer: int, root: str = "./cache") -> str:
    """Canonical path for raw LLM activations at (dataset, layer).

    Layout: ``{root}/{dataset}_l{layer}/`` — produced by ``run_cache.py``.
    """
    return f"{root}/{dataset}_l{layer}"


def feature_cache_path(
    dataset: str,
    crosscoder_short: str,
    root: str = "./cache",
) -> str:
    """Canonical path for crosscoder feature codes for (crosscoder, dataset).

    Layout: ``{root}/{crosscoder_short}_features_{dataset}/``.

    The short name (e.g. ``"dfc-D8k-excl10-k45-l9"``) already encodes the
    training layer via its ``-l<L>`` suffix, so feature caches do not need a
    separate ``_l<L>`` token in the path.
    """
    return f"{root}/{crosscoder_short}_features_{dataset}"


def layer_from_hparams(hparams_path) -> int:
    """Read the training layer from a crosscoder's ``hparams.json``.

    Raises ``ValueError`` if the file is missing or has no ``layer`` field —
    callers should catch this and decide on a fallback.
    """
    p = Path(hparams_path)
    if not p.exists():
        raise ValueError(f"hparams.json not found at {p}")
    data = json.loads(p.read_text())
    if "layer" not in data:
        raise ValueError(f"hparams.json at {p} has no 'layer' field")
    return int(data["layer"])


# ─── Main config ───────────────────────────────────────────────────────────

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

    # ── Cache root ───────────────────────────────────────────────────
    # Per-dataset paths are derived from `layer` via the properties below.
    cache_dir:       str = "./cache"
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

    # Layer-derived cache paths. Properties so they always reflect the
    # current value of `self.layer` — assigning cfg.layer = L re-routes
    # cfg.fineweb_cache / cfg.toolrl_cache automatically.
    @property
    def fineweb_cache(self) -> str:
        return raw_cache_path("fineweb", self.layer, self.cache_dir)

    @property
    def toolrl_cache(self) -> str:
        return raw_cache_path("toolrl", self.layer, self.cache_dir)

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.save_path).mkdir(parents=True, exist_ok=True)


# Default config used by scripts
cfg = Config()
