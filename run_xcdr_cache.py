"""
cache_crosscoder_activations.py
────────────────────────────────
Encodes already-cached LLM activations through the trained DFC and saves the
resulting sparse feature vectors to disk.  No LLMs needed — just the DFC
checkpoint and the shard files from run_cache.py.

Run in tmux:
    tmux new -s dfc_cache
    python cache_crosscoder_activations.py
    # Ctrl-B D  to detach, tail -f dfc_feature_cache.log to monitor

Output layout:
    cache/
      fineweb_features/
        meta.json          <- shard list + feature stats
        shard_00000.pt     <- (shard_size, dict_size) float16
        ...
      toolrl_features/
        meta.json
        shard_00000.pt
        ...

Each shard row is the DFC feature vector for one text sample.
Shared / A-exclusive / B-exclusive partitions are preserved by index
(see DFCCrossCoder.a_end, b_end).
"""

import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dfc import DFCCrossCoder
from cache import CachedActivationDataset

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("dfc_cache")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
for handler in [logging.FileHandler("dfc_feature_cache.log", mode="a"),
                logging.StreamHandler(sys.stdout)]:
    handler.setFormatter(fmt)
    log.addHandler(handler)

cfg = Config()


# ── Load cached text ──────────────────────────────────────────────────────────

def load_cached_texts(cache_dir: str):
    """Load text from cache directory, preserving order to match activations."""
    cache_path = Path(cache_dir)
    texts_dir = cache_path / "texts"
    
    if not texts_dir.exists():
        raise FileNotFoundError(f"No texts directory found in {cache_dir}. Cache may be from older version.")
    
    # Load meta to get text shards in order
    meta_path = cache_path / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    
    text_shards = meta.get("text_shards", [])
    if not text_shards:
        # Fallback: infer text shard names from activation shards
        text_shards = [shard.replace(".pt", ".jsonl") for shard in meta["shards"]]
    
    texts = []
    for shard_name in text_shards:
        shard_path = texts_dir / shard_name
        if shard_path.exists():
            with open(shard_path) as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data["text"])
    
    return texts


# ── Encode one dataset ────────────────────────────────────────────────────────

@torch.no_grad()
def encode_dataset(
    dfc: DFCCrossCoder,
    llm_cache_dir: str,
    out_dir: str,
    text_iter=None,  # Unused - kept for compatibility
    batch_size: int = 256,
    device: str = "cuda:0",
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create texts subdirectory for autointerp
    texts_dir = out_path / "texts"
    texts_dir.mkdir(exist_ok=True)

    meta_out = out_path / "meta.json"
    if meta_out.exists():
        log.info(f"  Already cached -> {out_dir}  (skipping)")
        return

    log.info(f"  Source : {llm_cache_dir}")
    log.info(f"  Output : {out_dir}")

    dataset = CachedActivationDataset(llm_cache_dir, normalize=True)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    total          = len(dataset)
    shard_size     = cfg.shard_size
    shard_idx      = 0
    buf            = []
    text_buf       = []
    shards_written = []
    n_written      = 0

    sum_l0   = 0.0
    sum_l0_a = 0.0
    sum_l0_b = 0.0
    sum_l0_s = 0.0

    pbar = tqdm(total=total, unit="sample", dynamic_ncols=True, file=sys.stdout)
    
    # Load cached text that corresponds to this activation cache
    text_list = load_cached_texts(llm_cache_dir)
    if len(text_list) != total:
        log.warning(f"Text count ({len(text_list)}) doesn't match activation count ({total})")
    text_idx = 0

    def _flush(rows, texts):
        nonlocal shard_idx, n_written
        # Save feature activations
        shard = torch.cat(rows, dim=0).half()
        fname = f"shard_{shard_idx:05d}.pt"
        torch.save(shard, out_path / fname)
        shards_written.append(fname)
        
        # Save corresponding text
        text_fname = f"shard_{shard_idx:05d}.jsonl"
        text_path = texts_dir / text_fname
        with open(text_path, 'w') as f:
            for text in texts:
                f.write(json.dumps({"text": text}) + "\n")
        
        shard_idx += 1
        n_written += len(shard)

    for batch in loader:
        feats = dfc.encode(batch.to(device)).cpu()   # (B, dict_size)
        batch_size = len(feats)
        
        # Get corresponding text batch
        batch_texts = text_list[text_idx:text_idx + batch_size]
        text_idx += batch_size

        sum_l0   += (feats > 0).float().sum(dim=-1).mean().item()
        sum_l0_a += (feats[:, :dfc.a_end] > 0).float().sum(dim=-1).mean().item()
        sum_l0_b += (feats[:, dfc.a_end:dfc.b_end] > 0).float().sum(dim=-1).mean().item()
        sum_l0_s += (feats[:, dfc.b_end:] > 0).float().sum(dim=-1).mean().item()

        buf.append(feats)
        text_buf.extend(batch_texts)
        pbar.update(len(feats))

        # Check if we need to flush (using number of samples)
        buf_size = sum(len(t) for t in buf)
        if buf_size >= shard_size:
            _flush(buf, text_buf[:buf_size])
            text_buf = text_buf[buf_size:]
            buf = []

    if buf:
        _flush(buf, text_buf)

    pbar.close()

    n_batches = max(len(loader), 1)
    meta = {
        "source_cache":   llm_cache_dir,
        "total_samples":  n_written,
        "dict_size":      dfc.dict_size,
        "n_a":            dfc.n_a,
        "n_b":            dfc.n_b,
        "n_shared":       dfc.n_shared,
        "shards":         shards_written,
        "mean_l0_total":  round(sum_l0   / n_batches, 3),
        "mean_l0_a_excl": round(sum_l0_a / n_batches, 3),
        "mean_l0_b_excl": round(sum_l0_b / n_batches, 3),
        "mean_l0_shared": round(sum_l0_s / n_batches, 3),
    }
    json.dump(meta, open(meta_out, "w"), indent=2)

    log.info(f"  + {n_written:,} samples -> {shard_idx} shards")
    log.info(f"    mean L0: total={meta['mean_l0_total']}  "
             f"A={meta['mean_l0_a_excl']}  "
             f"B={meta['mean_l0_b_excl']}  "
             f"shared={meta['mean_l0_shared']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  DFC Feature Activation Caching")
    log.info(f"  Checkpoint : {cfg.save_path}")
    log.info(f"  Device     : {cfg.train_device}")
    log.info("=" * 60)

    log.info("Loading DFC checkpoint...")
    dfc = DFCCrossCoder.load(cfg.save_path, device=cfg.train_device)
    dfc.eval()
    log.info(f"  dict_size={dfc.dict_size}  k={dfc.k}  "
             f"n_a={dfc.n_a}  n_b={dfc.n_b}  n_shared={dfc.n_shared}")

    # Feature caches use the model-tagged convention:
    # cache/{checkpoint_name}_features_{dataset}/. Raw caches are layered
    # via cfg.fineweb_cache / cfg.toolrl_cache (cache/{dataset}_l{layer}/).
    from config import feature_cache_path
    short = Path(cfg.save_path).name
    datasets = [
        ("FineWeb", cfg.fineweb_cache, feature_cache_path("fineweb", short, root=cfg.cache_dir)),
        ("ToolRL",  cfg.toolrl_cache,  feature_cache_path("toolrl",  short, root=cfg.cache_dir)),
    ]

    for name, llm_cache, feat_cache in datasets:
        log.info("")
        log.info(f"-- {name} --")

        if not (Path(llm_cache) / "meta.json").exists():
            log.warning(f"  LLM cache not found at {llm_cache} — run run_cache.py first.")
            continue

        encode_dataset(
            dfc=dfc,
            llm_cache_dir=llm_cache,
            out_dir=feat_cache,
            text_iter=None,  # Will load from cache
            batch_size=256,
            device=cfg.train_device,
        )

    log.info("")
    log.info("=" * 60)
    log.info("  All done. Feature caches written to:")
    for _, _, feat_cache in datasets:
        log.info(f"    {feat_cache}/")
        log.info(f"    {feat_cache}/texts/  (text files for autointerp)")
    log.info("  Ready for autointerp tomorrow.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()