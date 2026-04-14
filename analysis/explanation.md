# What I Did

I first inspected the repo and confirmed that the three relevant files for this task were:

- `results_full.jsonl`
- `results_steer.jsonl`
- `sweep_eval.py`

Their roles are:

- `results_full.jsonl`: the main post-reconstruction sweep results file.
- `results_steer.jsonl`: a separate steering-results file, inspected only for context.
- `sweep_eval.py`: the reference file that defines the evaluation modes, metric naming, and confirms that `exclusive_pct == 0` should be treated as the CrossCoder baseline.

## Analysis Pipeline

I wrote `analysis/run_xcoder_hparams_analysis.py` to make the analysis reproducible.

The script does the following:

1. Loads `results_full.jsonl`.
2. Parses each model variant from both:
   - the JSON `hyperparameters`
   - the model name string
3. Validates that those two sources agree on:
   - architecture
   - dictionary size
   - TopK
   - exclusive percentage
4. Extracts structured metadata for each model:
   - `TopK`
   - `dict_size`
   - model A exclusive %
   - model B exclusive %
   - CrossCoder baseline vs DFC
   - L1 usage
   - whether exclusive features are free or penalized
5. Auto-detects the actual post-reconstruction metrics from the file schema.
6. Builds:
   - a flat per-model table
   - a tidy metric table
7. Computes grouped summaries for:
   - metric vs TopK
   - metric vs exclusive %
   - metric vs regime
   - matched TopK comparisons
   - matched DFC-vs-CrossCoder comparisons
8. Writes:
   - CSV tables to `analysis/tables/`
   - plots to `analysis/figures/`
   - a written summary to `analysis/summary.md`

## Why The Script Is Zero-Dependency

The environment did not have `pandas`, `numpy`, or `matplotlib` installed, so I rewrote the analysis as a zero-dependency Python script using only the standard library.

Because of that:

- the tables are still written as CSV
- the plots are written as `.svg` files instead of using `matplotlib`

## Metrics Actually Selected

The script selected these post-reconstruction metrics from the real result schema:

- `format_accuracy_A_post`
- `format_accuracy_B_post`
- `overall_score_A_post`
- `overall_score_B_post`
- `tool_correctness_A_post`
- `tool_correctness_B_post`

These were treated as the main post-reconstruction metrics for the analysis.

## How I Controlled For Confounders

I did not rely only on raw averages, because those are confounded by other sweep settings.

Instead, I added controlled comparisons:

- For TopK effects:
  I compared matched variants within the same
  `(dict_size, exclusive_pct, regime)`
  and measured changes such as:
  - `45 -> 90`
  - `90 -> 160`
  - `45 -> 160`

- For exclusive-percentage effects:
  I compared each DFC variant against the matched CrossCoder baseline with the same:
  `(dict_size, TopK, uses_l1)`

That is what the following tables contain:

- `analysis/tables/controlled_topk_pairwise_summary.csv`
- `analysis/tables/controlled_exclusive_vs_baseline_summary.csv`

## Main Output Files

- `analysis/run_xcoder_hparams_analysis.py`
- `analysis/summary.md`
- `analysis/README.md`
- `analysis/tables/`
- `analysis/figures/`

## One Important Sweep Limitation

The sweep is not perfectly complete.

There are `47` variants in `results_full.jsonl`, not a full `48`.
The missing combination is:

- `dict_size = 16384`
- `TopK = 90`
- `exclusive_pct = 10%`
- `DFC penalized-exclusive`

So the conclusions are descriptive and useful, but not as strong as they would be with a complete factorial sweep.
