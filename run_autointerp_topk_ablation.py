"""run_autointerp_topk_ablation.py — autointerp the top-10 tool-discriminative
features per partition × model at l13.

Reads ``results/figures/ablation_l13/autointerp_features.json`` (frozen
top-10 lists), then dispatches the local-Gemma autointerp pipeline once
per (model, partition) bucket. Writes results under
``results/autointerp/l13_ablation/<model>_<partition>/``.

Usage:
    uv run python run_autointerp_topk_ablation.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from autointerp_local import LocalAutoInterpPipeline


CONFIG_PATH = Path("results/figures/ablation_l13/autointerp_features.json")
OUT_ROOT = Path("results/autointerp/l13_ablation")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gemma-model", default="google/gemma-2-9b-it")
    p.add_argument("--top-k", type=int, default=10,
                   help="max-activating examples per feature for explanation")
    p.add_argument("--n-random", type=int, default=10,
                   help="random examples for the detection score")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    config = json.loads(CONFIG_PATH.read_text())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for model_key, model_info in config.items():
        if not isinstance(model_info, dict) or model_key.startswith("_"):
            continue
        feat_cache = Path(f"cache/{model_key}_features_toolrl")
        if not feat_cache.is_dir():
            print(f"[skip] {model_key}: no feature cache at {feat_cache}")
            continue
        for partition_key, info in model_info.items():
            if not isinstance(info, dict) or partition_key.startswith("_"):
                continue
            runs.append({
                "model":       model_key,
                "partition":   partition_key,
                "feat_cache":  feat_cache,
                "features":    [int(x) for x in info["feature_idx"]],
            })

    if not runs:
        raise SystemExit("No runs derived from autointerp_features.json")

    print(f"[plan] {len(runs)} (model, partition) buckets to interpret:")
    for r in runs:
        print(f"  {r['model']} / {r['partition']:<10}  "
              f"({len(r['features'])} feats: {r['features']})")

    for r in runs:
        out = OUT_ROOT / f"{r['model']}_{r['partition']}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n══════════ {r['model']} / {r['partition']} ══════════")
        print(f"feat_cache = {r['feat_cache']}")
        print(f"out        = {out}")
        pipeline = LocalAutoInterpPipeline(
            feat_cache_dir=str(r["feat_cache"]),
            source_cache_dir=str(r["feat_cache"]),
            results_dir=str(out),
            gemma_model_id=args.gemma_model,
            gemma_device=args.device,
            top_k=args.top_k,
            n_random_for_detection=args.n_random,
            max_concurrent=1,
            feature_subset=r["features"],
        )
        pipeline.run()
        print(f"[done] {r['model']} / {r['partition']}")

    print(f"\n[all done] outputs under {OUT_ROOT}")


if __name__ == "__main__":
    main()
