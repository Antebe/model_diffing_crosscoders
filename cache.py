"""
cache.py — Extract and store last-token activations from both models.

Workflow
────────
1. `cache_activations()`  — runs LLMs once, writes sharded .pt files
2. `CachedActivationDataset` — lazy PyTorch Dataset backed by those shards
3. `build_combined_dataset()` — merges FineWeb + ToolRL datasets for training

Disk layout
───────────
cache/
  fineweb/
    meta.json
    shard_00000.pt   # (shard_size, 2, d) fp16
    shard_00001.pt
    …
  toolrl/
    meta.json
    shard_00000.pt
    …
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Extraction helpers
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_last_token_acts(
    model,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    device: str,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Forward `texts` through `model` and return the final real token's hidden
    state at `layer_idx` for each text.

    Returns: (N, hidden_dim) float32 CPU tensor.
    """
    return extract_last_token_acts_multi(
        model, tokenizer, texts, [layer_idx], device, batch_size,
    )[layer_idx]


@torch.no_grad()
def extract_last_token_acts_multi(
    model,
    tokenizer,
    texts: list[str],
    layer_indices: list[int],
    device: str,
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """
    Forward `texts` through `model` ONCE and return the final real token's
    hidden state at *each* requested layer.

    Causal-LM hidden_states are computed top-to-bottom in a single forward
    pass; pulling N layers costs the same as pulling 1 (just N more index
    lookups + .cpu() copies).

    Returns: ``{layer_idx: (N, hidden_dim) float32 CPU tensor}`` keyed by
    the original ``layer_indices``.
    """
    per_layer: dict[int, list[torch.Tensor]] = {L: [] for L in layer_indices}
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        last_idx = attn_mask.sum(dim=1) - 1               # (B,)
        arange = torch.arange(len(batch), device=last_idx.device)
        for L in layer_indices:
            # hidden_states[0] = embedding, [i+1] = after layer i
            hidden = out.hidden_states[L + 1]             # (B, seq, d)
            acts = hidden[arange, last_idx]               # (B, d)
            per_layer[L].append(acts.float().cpu())

    return {L: torch.cat(per_layer[L], dim=0) for L in layer_indices}


# ──────────────────────────────────────────────────────────────────────
# Caching
# ──────────────────────────────────────────────────────────────────────

def cache_activations(
    model_a,
    model_b,
    tokenizer,
    data_iter: Iterator[str],
    layer_idx: int,
    cache_dir: str,
    device_a: str = "cuda:0",
    device_b: str = "cuda:1",
    extract_batch: int = 8,
    shard_size: int = 2_048,
    max_samples: int | None = None,
    min_text_len: int = 30,
) -> str:
    """Single-layer wrapper around ``cache_activations_multi``.

    Kept for backward compatibility. New callers should use
    ``cache_activations_multi`` directly to amortize the LLM forward pass
    across multiple layers.
    """
    cache_activations_multi(
        model_a=model_a,
        model_b=model_b,
        tokenizer=tokenizer,
        data_iter=data_iter,
        layer_indices=[layer_idx],
        cache_dirs={layer_idx: cache_dir},
        device_a=device_a,
        device_b=device_b,
        extract_batch=extract_batch,
        shard_size=shard_size,
        max_samples=max_samples,
        min_text_len=min_text_len,
    )
    return cache_dir


def cache_activations_multi(
    model_a,
    model_b,
    tokenizer,
    data_iter: Iterator[str],
    layer_indices: list[int],
    cache_dirs: dict[int, str],
    device_a: str = "cuda:0",
    device_b: str = "cuda:1",
    extract_batch: int = 8,
    shard_size: int = 2_048,
    max_samples: int | None = None,
    min_text_len: int = 30,
) -> dict[int, str]:
    """
    Run both models over ``data_iter`` ONCE and write one cache per layer.

    For each shard-sized buffer of texts, performs *one* forward pass per
    model and snapshots ``hidden_states[L+1]`` for every ``L`` in
    ``layer_indices`` — so 9 layers cost the same forward time as 1, plus
    a small cost for the extra index/copy/save ops.

    Each layer ``L`` gets its own cache directory at ``cache_dirs[L]`` with
    the same on-disk layout as the single-layer ``cache_activations``:
    ``shard_NNNNN.pt`` (N, 2, d) fp16 + ``texts/shard_NNNNN.jsonl`` +
    ``meta.json``.

    Returns ``cache_dirs`` for chaining.
    """
    if set(layer_indices) != set(cache_dirs.keys()):
        raise ValueError(
            f"layer_indices ({layer_indices}) and cache_dirs.keys "
            f"({list(cache_dirs.keys())}) must match"
        )

    # Per-layer paths and shard bookkeeping.
    paths: dict[int, Path] = {}
    texts_dirs: dict[int, Path] = {}
    shards_written: dict[int, list[str]] = {L: [] for L in layer_indices}
    text_shards_written: dict[int, list[str]] = {L: [] for L in layer_indices}
    for L in layer_indices:
        p = Path(cache_dirs[L])
        p.mkdir(parents=True, exist_ok=True)
        td = p / "texts"
        td.mkdir(exist_ok=True)
        paths[L] = p
        texts_dirs[L] = td

    buf: list[str] = []
    shard_idx = 0
    total = 0

    pbar = tqdm(
        desc=f"Caching {len(layer_indices)}L → {[cache_dirs[L] for L in layer_indices]}",
        unit="sample",
        dynamic_ncols=True,
    )

    def _flush(texts: list[str]) -> None:
        nonlocal shard_idx, total
        # Two forward passes total (one per model), N layers extracted from each.
        acts_a = extract_last_token_acts_multi(
            model_a, tokenizer, texts, layer_indices, device_a, extract_batch,
        )
        acts_b = extract_last_token_acts_multi(
            model_b, tokenizer, texts, layer_indices, device_b, extract_batch,
        )
        for L in layer_indices:
            shard = torch.stack([acts_a[L], acts_b[L]], dim=1).half()  # (N, 2, d)
            act_fname = f"shard_{shard_idx:05d}.pt"
            torch.save(shard, paths[L] / act_fname)
            shards_written[L].append(act_fname)

            text_fname = f"shard_{shard_idx:05d}.jsonl"
            text_path = texts_dirs[L] / text_fname
            with open(text_path, "w") as f:
                for text in texts:
                    f.write(json.dumps({"text": text}) + "\n")
            text_shards_written[L].append(text_fname)

        shard_idx += 1
        total += len(texts)
        pbar.update(len(texts))

    for text in data_iter:
        if len(text.strip()) < min_text_len:
            continue
        buf.append(text)
        if len(buf) >= shard_size:
            _flush(buf)
            buf = []
        if max_samples and total >= max_samples:
            break

    if buf:
        _flush(buf)

    pbar.close()

    for L in layer_indices:
        meta = dict(
            layer_idx=L,
            shard_size=shard_size,
            total_samples=total,
            shards=shards_written[L],
            text_shards=text_shards_written[L],
        )
        json.dump(meta, open(paths[L] / "meta.json", "w"), indent=2)
        print(f"[Cache] L={L}: {total:,} samples → {shard_idx} shards in {cache_dirs[L]}")

    return cache_dirs


def cache_exists(cache_dir: str) -> bool:
    return (Path(cache_dir) / "meta.json").exists()


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class CachedActivationDataset(Dataset):
    """
    Lazy-loading Dataset backed by sharded .pt files from cache_activations().

    • One shard in memory at a time (evict-on-miss).
    • Optional per-channel z-score normalisation (estimated from first shard).
    • Each item: (2, d) float32 tensor ready for DFCCrossCoder.
    • Carries a `source_label` tag so callers know which dataset a batch came from.
    """

    def __init__(
        self,
        cache_dir: str,
        normalize: bool = True,
        source_label: str = "unknown",
    ):
        self.cache_dir = Path(cache_dir)
        self.source_label = source_label
        meta = json.load(open(self.cache_dir / "meta.json"))
        self.shard_files = [self.cache_dir / s for s in meta["shards"]]
        self.total = meta["total_samples"]

        # Build flat index without loading data
        self._index: list[tuple[int, int]] = []
        self._shard_sizes: list[int] = []
        self._loaded: tuple[int, torch.Tensor] | None = None

        # Preload all shards into one contiguous tensor (~1GB for 72k samples)
        all_shards = []
        for fpath in self.shard_files:
            all_shards.append(
                torch.load(fpath, map_location="cpu", weights_only=True).float()
            )
        self._data = torch.cat(all_shards, dim=0)  # (total, 2, d)
        assert len(self._data) == self.total

        # Normalisation stats (from first shard)
        if normalize:
            self.mean: torch.Tensor | None = self._data.mean(dim=0)            # (2, d)
            self.std:  torch.Tensor | None = self._data.std(dim=0).clamp(min=1e-6)
            self._data = (self._data - self.mean) / self.std
        else:
            self.mean = self.std = None

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]


def build_combined_dataset(
    fineweb_cache: str,
    toolrl_cache: str,
    normalize: bool = True,
) -> ConcatDataset:
    """Merge FineWeb and ToolRL caches into one shuffled dataset."""
    fw = CachedActivationDataset(fineweb_cache, normalize=normalize, source_label="fineweb")
    tr = CachedActivationDataset(toolrl_cache,  normalize=normalize, source_label="toolrl")
    combined = ConcatDataset([fw, tr])
    print(f"[Dataset] FineWeb: {len(fw):,} | ToolRL: {len(tr):,} | Total: {len(combined):,}")
    return combined
