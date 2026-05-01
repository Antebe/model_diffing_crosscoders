# run_autointerp_exclusive.py
import json
from pathlib import Path

from tqdm import tqdm
from autointerp import AutoInterpPipeline, TopKFinder

FEAT_CACHE   = "./cache/toolrl_features"
SOURCE_CACHE = "./cache/toolrl"
RESULTS_DIR  = "./results/autointerp/toolrl"

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