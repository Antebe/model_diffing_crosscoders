"""
build_feature_cache.py — encode an existing raw-activation cache through
a HuggingFace crosscoder and write per-shard sparse feature tensors.

Inputs:
  --crosscoder    HF repo id (e.g. ``antebe1/dfc-D8k-excl10-freeexcl-k160``).
  --source-cache  Directory of raw activation shards
                  (e.g. ``cache/toolrl/``) produced by ``cache.py``.
                  Each shard is ``(N, 2, d)`` fp16 with h_a / h_b stacked.
  --out           Output directory; written as ``shard_NNNNN.pt`` files
                  containing ``(N, dict_size)`` fp16 tensors plus a
                  ``meta.json`` and a copy of the source ``texts/`` shards.

The output layout matches what ``autointerp.TopKFinder`` and
``autointerp.TextStore`` expect.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from sweep_eval import load_crosscoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crosscoder", required=True)
    p.add_argument("--source-cache", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--models-jsonl", default="models.jsonl")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    src: Path = args.source_cache
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    src_meta = json.loads((src / "meta.json").read_text())
    shards = src_meta["shards"]
    print(f"[cache] source={src}  shards={len(shards)}  total_samples={src_meta.get('total_samples', '?')}")

    # Load crosscoder
    entries = [json.loads(l) for l in Path(args.models_jsonl).read_text().splitlines() if l.strip()]
    matches = [e for e in entries if e["name"] == args.crosscoder]
    if not matches:
        raise SystemExit(f"crosscoder {args.crosscoder!r} not in {args.models_jsonl}")
    cc = load_crosscoder(matches[0], device=args.device)
    if cc is None:
        raise SystemExit("crosscoder load failed")

    n_a = int(getattr(cc, "n_a", 0))
    n_b = int(getattr(cc, "n_b", 0))
    n_shared = int(cc.dict_size) - n_a - n_b
    written: list[str] = []
    total = 0

    for sname in tqdm(shards, desc="encode"):
        shard = torch.load(src / sname, map_location=args.device, weights_only=False)
        x = shard.float().to(args.device)               # (N, 2, d)
        with torch.no_grad():
            feats = cc.encode(x)                        # (N, dict_size)
        feats_h = feats.half().cpu()                    # save as fp16
        torch.save(feats_h, out / sname)
        written.append(sname)
        total += int(feats_h.shape[0])
        del shard, x, feats, feats_h

    # Copy the parallel texts/ subdir if present
    src_texts = src / "texts"
    if src_texts.is_dir():
        dst_texts = out / "texts"
        dst_texts.mkdir(exist_ok=True)
        for fp in src_texts.iterdir():
            shutil.copy2(fp, dst_texts / fp.name)
        print(f"[cache] copied {len(list(dst_texts.iterdir()))} text shards → {dst_texts}")
    else:
        print("[cache] WARNING: source has no texts/ subdir; autointerp will use placeholders")

    meta = {
        "source_cache": str(src),
        "crosscoder": args.crosscoder,
        "total_samples": total,
        "dict_size": int(cc.dict_size),
        "n_a": n_a,
        "n_b": n_b,
        "n_shared": n_shared,
        "shards": written,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[cache] wrote {len(written)} shards / {total:,} samples → {out}")
    print(f"[cache] meta: dict={cc.dict_size}  n_a={n_a}  n_b={n_b}  n_shared={n_shared}")


if __name__ == "__main__":
    main()
