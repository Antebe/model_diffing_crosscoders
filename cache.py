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
    all_acts = []
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
        # hidden_states[0] = embedding, [i+1] = after layer i
        hidden = out.hidden_states[layer_idx + 1]        # (B, seq, d)
        last_idx = attn_mask.sum(dim=1) - 1              # (B,)
        acts = hidden[torch.arange(len(batch)), last_idx] # (B, d)
        all_acts.append(acts.float().cpu())

    return torch.cat(all_acts, dim=0)


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
    """
    Run both models over `data_iter` once. Saves (N, 2, d) fp16 shards and corresponding text.

    Returns `cache_dir` for chaining into CachedActivationDataset.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Create texts subdirectory
    texts_dir = cache_path / "texts"
    texts_dir.mkdir(exist_ok=True)

    buf: list[str] = []
    shard_idx = 0
    total = 0
    shards_written: list[str] = []
    text_shards_written: list[str] = []

    pbar = tqdm(desc=f"Caching → {cache_dir}", unit="sample", dynamic_ncols=True)

    def _flush(texts: list[str]) -> None:
        nonlocal shard_idx, total
        a = extract_last_token_acts(model_a, tokenizer, texts, layer_idx, device_a, extract_batch)
        b = extract_last_token_acts(model_b, tokenizer, texts, layer_idx, device_b, extract_batch)
        shard = torch.stack([a, b], dim=1).half()   # (N, 2, d) — fp16
        
        # Save activations
        act_fname = f"shard_{shard_idx:05d}.pt"
        torch.save(shard, cache_path / act_fname)
        shards_written.append(act_fname)
        
        # Save corresponding text
        text_fname = f"shard_{shard_idx:05d}.jsonl"
        text_path = texts_dir / text_fname
        with open(text_path, 'w') as f:
            for text in texts:
                f.write(json.dumps({"text": text}) + "\n")
        text_shards_written.append(text_fname)
        
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

    meta = dict(
        layer_idx=layer_idx,
        shard_size=shard_size,
        total_samples=total,
        shards=shards_written,
        text_shards=text_shards_written,
    )
    json.dump(meta, open(cache_path / "meta.json", "w"), indent=2)
    print(f"[Cache] {total:,} samples → {shard_idx} shards in {cache_dir}")
    return cache_dir


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
