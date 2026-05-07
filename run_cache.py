"""
run_cache.py — Extract and cache activations from FineWeb and ToolRL datasets.

Run this once (or once per layer-set) before training. LLMs are unloaded
after caching.

Usage:
    python run_cache.py                                          # default cfg.layer
    python run_cache.py --layer 20 --fineweb_samples 100000      # single layer
    python run_cache.py --layers 1,5,9,14,18,20,24,28,32         # multi-layer
                                                                 # one forward pass total

Multi-layer mode amortizes the LLM forward pass across all requested layers
(causal-LM hidden_states are computed top-to-bottom in a single pass; pulling
N layers costs the same forward time as pulling 1). Each layer's cache is
written to ``cache/{dataset}_l{L}/`` exactly as the single-layer mode would.
"""

import argparse
from itertools import islice

from datasets import load_dataset

from cache import cache_activations_multi, cache_exists
from config import Config, raw_cache_path
from models import load_both_models, unload_models


def chunk_text(text: str, target_tokens: int = 30, min_tokens: int = 1) -> list[str]:
    """
    Split text into semantic chunks based on sentences.
    
    Args:
        text: Input text to chunk
        target_tokens: Target tokens per chunk (rough estimate: ~0.75 tokens per word)
        min_tokens: Minimum tokens for a valid chunk
    
    Returns:
        List of text chunks
    """
    # Rough token estimation: ~0.75 tokens per word
    target_words = int(target_tokens / 0.75)
    min_words = int(min_tokens / 0.75)
    
    # Split on sentence boundaries
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        word_count = len(sentence.split())
        
        # If adding this sentence exceeds target, finish current chunk
        if current_word_count > 0 and current_word_count + word_count > target_words:
            chunk_text = ' '.join(current_chunk)
            if current_word_count >= min_words:
                chunks.append(chunk_text)
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    # Add final chunk
    if current_chunk and current_word_count >= min_words:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks if chunks else [text]  # Fallback to original text


def get_fineweb_iter(n: int):
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT", 
        split="train",
        streaming=True,
    )
    
    count = 0
    for ex in ds:
        if count >= n:
            break
        
        text = ex["text"]
        if text and len(text.strip()) >= 3:  # Minimum text length
            chunks = chunk_text(text)
            for chunk in chunks:
                if count >= n:
                    break
                yield chunk
                count += 1


def get_toolrl_iter(n: int):
    """
    emrecanacikgoz/ToolRL — 4k rows with columns: instruction, input, output.
    Concatenates all three into a single text, then chunks it.
    """
    ds = load_dataset("emrecanacikgoz/ToolRL", split="train")  # ~15 MB, load fully

    print(f"[ToolRL] Columns: {ds.column_names}")
    print(f"[ToolRL] Total rows: {len(ds)}")

    def _format(ex):
        parts = []
        for col in ("instruction", "input", "output"):
            val = ex.get(col, "")
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        return "\n\n".join(parts)

    # Cycle the 4k dataset until we have n samples
    from itertools import cycle
    count = 0
    for ex in cycle(ds):
        if count >= n:
            break
            
        text = _format(ex)
        if text and len(text.strip()) >= 3:
            chunks = chunk_text(text)
            for chunk in chunks:
                if count >= n:
                    break
                yield chunk
                count += 1


def _cache_dataset(
    cfg: Config,
    name: str,
    samples: int,
    iter_fn,
    layers: list[int],
    model_a,
    model_b,
    tokenizer,
) -> None:
    """Cache one dataset across all `layers` in a single forward pass."""
    cache_dirs = {L: raw_cache_path(name, L, root=cfg.cache_dir) for L in layers}

    # Skip layers whose meta.json already exists; only run if at least one is missing.
    pending = [L for L in layers if not cache_exists(cache_dirs[L])]
    skipped = [L for L in layers if L not in pending]
    if skipped:
        print(f"\n[Cache] {name}: skipping {len(skipped)} layer(s) already on disk: {skipped}")
    if not pending:
        print(f"[Cache] {name}: nothing to do.")
        return

    print(f"\n[Cache] {name}: caching {samples:,} samples × {len(pending)} layer(s) {pending} "
          f"(one forward pass per shard) …")
    cache_activations_multi(
        model_a=model_a,
        model_b=model_b,
        tokenizer=tokenizer,
        data_iter=iter_fn(samples),
        layer_indices=pending,
        cache_dirs={L: cache_dirs[L] for L in pending},
        device_a=cfg.device_a,
        device_b=cfg.device_b,
        extract_batch=cfg.extract_batch,
        shard_size=cfg.shard_size,
        min_text_len=cfg.min_text_len,
    )


def main(cfg: Config, layers: list[int]):
    print("=" * 60)
    print(f"  DFC Activation Caching — layers {layers}")
    print("=" * 60)

    # ── Load models ──────────────────────────────────────────────────
    model_a, model_b, tokenizer = load_both_models(cfg)

    _cache_dataset(cfg, "fineweb", cfg.fineweb_samples, get_fineweb_iter, layers,
                   model_a, model_b, tokenizer)
    _cache_dataset(cfg, "toolrl",  cfg.toolrl_samples,  get_toolrl_iter,  layers,
                   model_a, model_b, tokenizer)

    # ── Unload LLMs ──────────────────────────────────────────────────
    unload_models(model_a, model_b)
    print("\n[Cache] All done. Run python run_train.py to train the DFC.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--layer",  type=int, default=None,
                   help="Single transformer layer to cache (sets cfg.layer)")
    g.add_argument("--layers", type=str, default=None,
                   help="Comma-separated list of layers (e.g. '1,5,9,14,18,20,24,28,32') "
                        "— extracted in a single forward pass per shard")
    parser.add_argument("--fineweb_samples",  type=int, default=None)
    parser.add_argument("--toolrl_samples",   type=int, default=None)
    parser.add_argument("--shard_size",       type=int, default=None)
    parser.add_argument("--extract_batch",    type=int, default=None)
    parser.add_argument("--device_a",         type=str, default=None)
    parser.add_argument("--device_b",         type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.layer           is not None: cfg.layer            = args.layer
    if args.fineweb_samples is not None: cfg.fineweb_samples  = args.fineweb_samples
    if args.toolrl_samples  is not None: cfg.toolrl_samples   = args.toolrl_samples
    if args.shard_size      is not None: cfg.shard_size        = args.shard_size
    if args.extract_batch   is not None: cfg.extract_batch     = args.extract_batch
    if args.device_a        is not None: cfg.device_a          = args.device_a
    if args.device_b        is not None: cfg.device_b          = args.device_b

    if args.layers is not None:
        layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
        if not layers:
            raise SystemExit("--layers parsed to an empty list")
    else:
        layers = [cfg.layer]

    main(cfg, layers)
