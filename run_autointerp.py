# run_autointerp_exclusive.py
#
# Runs autointerp against a (crosscoder, dataset) feature cache.
# Defaults are derived from --crosscoder + --dataset + --layer via the
# layer-aware path helpers in config.py:
#   FEAT_CACHE   = cache/{crosscoder_short}_features_{dataset}/
#   SOURCE_CACHE = cache/{dataset}_l{layer}/
import argparse
import json
from pathlib import Path

from tqdm import tqdm
from autointerp import AutoInterpPipeline, TopKFinder
from config import feature_cache_path, raw_cache_path, layer_from_hparams

_p = argparse.ArgumentParser()
_p.add_argument("--crosscoder-short", default="dfc2",
                help="Crosscoder short name (used to locate the feature cache)")
_p.add_argument("--dataset", default="toolrl")
_p.add_argument("--layer", type=int, default=None,
                help="Override the layer (default: read from <feat_cache>/hparams.json or fall back to 13)")
_p.add_argument("--feat-cache", default=None, help="Override feature cache path")
_p.add_argument("--source-cache", default=None, help="Override raw cache path")
_p.add_argument("--results-dir", default=None)
_args, _ = _p.parse_known_args()

FEAT_CACHE = _args.feat_cache or feature_cache_path(_args.dataset, _args.crosscoder_short)
# Resolve layer: explicit > feature-cache hparams sidecar > default 13
if _args.layer is not None:
    LAYER = _args.layer
else:
    try:
        LAYER = layer_from_hparams(Path(FEAT_CACHE) / "hparams.json")
    except Exception:
        LAYER = 13
SOURCE_CACHE = _args.source_cache or raw_cache_path(_args.dataset, LAYER)
RESULTS_DIR = _args.results_dir or f"./results/autointerp/{_args.dataset}"

# ── 1. Find non-dead features via TopKFinder ──────────────────────────────
meta = json.loads((Path(FEAT_CACHE) / "meta.json").read_text())
a_end = meta["n_a"]
b_end = meta["n_a"] + meta["n_b"]

finder = TopKFinder(FEAT_CACHE, k=15)
finder.run()

#add tqdm progress bar to the following loop

exclusive_active = [
    feat_idx
    for feat_idx, examples in tqdm(finder.results.items(), desc="Processing features")
    if feat_idx < a_end          # a_excl (0–a_end) or b_excl (a_end–b_end)
    and len(examples) > 3        # non-dead
]

exclusive_active = [ 20,
    435,
    546,
    651,
    93,
    432,
    326,
    503,
    696,
    421,
    665,
    95,
    167] 
print(f"Non-dead exclusive features: {len(exclusive_active):,}")

# ── 2. Run pipeline with pre-built subset ─────────────────────────────────
pipeline = AutoInterpPipeline(
    feat_cache_dir=FEAT_CACHE,
    source_cache_dir=SOURCE_CACHE,
    results_dir=RESULTS_DIR,
    feature_subset=exclusive_active,
)
# Reuse the TopKFinder results we already computed — skip re-running it
pipeline.topk_results   = finder.results
pipeline._n_a           = meta["n_a"]
pipeline._n_b           = meta["n_b"]
pipeline._dict_size     = meta["dict_size"]
pipeline._a_end         = a_end
pipeline._b_end         = b_end

pipeline.run()