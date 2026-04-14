# X-Coder Hyperparameter Analysis

## Files Used

- `results_full.jsonl`: full post-reconstruction sweep results and the primary analysis input.
- `results_steer.jsonl`: steering-only results, inspected for context but not centered in this analysis.
- `sweep_eval.py`: source of truth for evaluation modes, metric naming, and the `exclusive_pct == 0` CrossCoder baseline convention.

## Sweep Dimensions

- `dict_size`: 8192, 16384
- `TopK`: 45, 90, 160
- `exclusive_pct`: 0%, 3%, 5%, 10%
- `regimes`: CrossCoder + L1, CrossCoder no-L1, DFC free-exclusive, DFC penalized-exclusive
- `n_model_variants`: 47

## Main Post-Reconstruction Metrics

The script auto-selected these `*_post` fields: `format_accuracy_A_post`, `format_accuracy_B_post`, `overall_score_A_post`, `overall_score_B_post`, `tool_correctness_A_post`, `tool_correctness_B_post`.

Metric families:

- `format_accuracy`
- `overall_score`
- `tool_correctness`

## Main Trends

- Matched overall_score_A_post: average change from TopK 45 to 160 was +0.059.
- Matched format_accuracy_A_post: average change from TopK 45 to 160 was +2.250.
- Matched tool_correctness_A_post: average change from TopK 45 to 160 was +1.812.
- Matched overall_score_B_post: average change from TopK 45 to 160 was -0.049.
- Matched vs CrossCoder baseline, the best DFC setting was DFC free-exclusive at 10% exclusive (+0.017), while the worst was DFC penalized-exclusive at 5% (-0.270).
- Matched vs CrossCoder baseline, the best DFC setting was DFC free-exclusive at 10% exclusive (+1.000), while the worst was DFC penalized-exclusive at 5% (-9.000).
- Matched vs CrossCoder baseline, the best DFC setting was DFC free-exclusive at 10% exclusive (+0.333), while the worst was DFC penalized-exclusive at 5% (-9.000).
- Matched vs CrossCoder baseline, the best DFC setting was DFC penalized-exclusive at 10% exclusive (+0.028), while the worst was DFC free-exclusive at 5% (-0.043).

## Interpretation

- Higher TopK does not show a universal monotonic benefit. The controlled pairwise summaries are the cleanest answer to whether larger TopK helps.
- Nonzero exclusive partitions sometimes beat the 0-exclusive CrossCoder baseline, but the sign and size depend heavily on dictionary size and whether exclusive features are regularized.
- Model A is the clearer signal. Model B remains much weaker overall, so its post metrics move less and are noisier to interpret.

## Caveats

- The conclusions are descriptive rather than causal. The sweep has 47 variants, not a perfectly complete factorial design, and `dict_size` plus sparsity regime confound raw averages. The matched pair tables reduce that by comparing only within shared `(dict_size, exclusive_pct, regime)` for TopK effects and within shared `(dict_size, TopK, uses_l1)` for DFC-vs-baseline effects.
- The steering file was inspected only for context and summarized into 18 grouped rows; it was not used as the main evidence for reconstruction conclusions.
- `layer`, `steps`, `lr`, and `train_batch` are constant in this sweep, so the main practical confounders are `dict_size`, shared L1 usage, and whether exclusive features are free or regularized.
- `format_accuracy_B_post` is zero for most variants, so it has limited discriminative power.

## Artifacts

- Figures: `analysis/figures/`
- Tables: `analysis/tables/`
- Repro script: `analysis/run_xcoder_hparams_analysis.py`
