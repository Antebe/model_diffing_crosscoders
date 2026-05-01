"""
run_autointerp_local.py — entry point for A-exclusive autointerp using
local Gemma-2-9B-it.

Selects feature subset by partition (default ``a_excl``) optionally
intersected with non-dead-on-eval-corpus rows from the discriminative
ranking CSV, then runs the full Topk → Explain → Detect pipeline via
``LocalAutoInterpPipeline``.

Output: ``<out>/<bucket>/feat_NNNNNNN.json`` per feature plus
``<out>/summary.json`` after the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from autointerp_local import LocalAutoInterpPipeline


def select_features(
    feat_cache_meta: dict,
    rankings_csv: Path | None,
    partition: str,
    fire_rate_floor: float,
) -> list[int]:
    """Resolve feature indices for the chosen partition."""
    n_a = int(feat_cache_meta["n_a"])
    n_b = int(feat_cache_meta["n_b"])
    dict_size = int(feat_cache_meta["dict_size"])

    if partition == "a_excl":
        candidates = list(range(n_a))
    elif partition == "b_excl":
        candidates = list(range(n_a, n_a + n_b))
    elif partition == "shared":
        candidates = list(range(n_a + n_b, dict_size))
    elif partition == "all":
        candidates = list(range(dict_size))
    else:
        raise ValueError(f"unknown partition: {partition!r}")

    if rankings_csv is None or fire_rate_floor <= 0:
        return candidates

    keep: set[int] = set()
    with rankings_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["feature_idx"])
                fr = float(row["fire_rate_tool"])
            except (KeyError, ValueError):
                continue
            if fr >= fire_rate_floor:
                keep.add(idx)
    candidates_set = set(candidates)
    filtered = sorted(candidates_set & keep)
    print(
        f"[select] partition={partition} candidates={len(candidates)} "
        f"after fire_rate_tool>={fire_rate_floor}: {len(filtered)}"
    )
    return filtered


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crosscoder", required=True,
                   help="HF repo id (used only for logging — pipeline reads from --feat-cache)")
    p.add_argument("--feat-cache", required=True, type=Path,
                   help="output of build_feature_cache.py")
    p.add_argument("--rankings", type=Path, default=None,
                   help="optional ranking CSV; required when --fire-rate-floor > 0")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--partition", default="a_excl",
                   choices=["a_excl", "b_excl", "shared", "all"])
    p.add_argument("--fire-rate-floor", type=float, default=0.005,
                   help="drop features whose fire_rate_tool is below this (skip dead-on-eval)")
    p.add_argument("--gemma-model", default="google/gemma-2-9b-it")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.8,
                   help="detection-accuracy threshold for 'interpretable' label")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    feat_cache_meta = json.loads((args.feat_cache / "meta.json").read_text())

    feature_subset = select_features(
        feat_cache_meta, args.rankings, args.partition, args.fire_rate_floor,
    )
    print(f"[run] crosscoder={args.crosscoder}")
    print(f"[run] partition={args.partition}  features={len(feature_subset)}")

    pipeline = LocalAutoInterpPipeline(
        feat_cache_dir=str(args.feat_cache),
        source_cache_dir=str(args.feat_cache),         # texts/ lives next to feature shards
        results_dir=str(args.out),
        gemma_model_id=args.gemma_model,
        gemma_device=args.device,
        top_k=args.top_k,
        n_random_for_detection=args.n_random,
        max_concurrent=1,
        interpretability_threshold=args.threshold,
        feature_subset=feature_subset,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
