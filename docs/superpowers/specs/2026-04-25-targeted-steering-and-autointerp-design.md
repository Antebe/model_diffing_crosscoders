# Targeted-Neuron Steering, Local-Gemma Autointerp, and Decoder UMAP — Design

**Date:** 2026-04-25
**Author:** scaffolded with Claude
**Supersedes:** N/A (extends `dfc_toolrl_sweep_paper.pdf` §8.1, §8.2, §8.3)

---

## 1. Motivation and hypotheses

The sweep paper (`results/dfc_toolrl_sweep_paper.pdf`) closes with three
explicitly-flagged "next experiment" placeholders (§8.1, §8.2, §8.3). This
design realises §8.1 (autointerp of the A-exclusive partition) and §8.2
(targeted-neuron steering on a selected subset), and writes up §8.3 as a
research design without code.

**Steering target: Model A (ToolRL), not Model B.** The metric we care
about is whether selectively amplifying A-exclusive features through
*Model A's* decoder direction can *improve Model A's* tool-calling
behaviour above its post-reconstruction baseline (paper Table 1: Δ_A
tool-correctness = +32.9% CrossCoder, +30.6% DFC). Spillover into Model
B is not the question here.

The paper's §6 is critical context: scaling all A-excl features by a
positive constant *pre-top-k* is a mathematical no-op under top-k SAEs.
Therefore the new sweep applies steering as a **post-top-k delta** on a
**selected subset** of A-excl neurons, ranked by discriminative score.
This avoids re-deriving Fig 7's null result.

### Hypotheses (testable)

- **H1 (steering improvement on Model A).** There exists a
  `(model, k%, α)` cell where steering the top-k% most-discriminative
  A-excl features by α *strictly improves* Model A's `tool_correctness`
  above the full-reconstruction baseline (paper Table 1: ≈+30 pp on top
  of the 19% clean baseline → ~50% post-recon). Win condition:
  `Δ_A_tool_correctness ≥ +35 pp` at small k% (≤ 8%) and modest α
  (≤ 16). Parity with full recon at smaller k% is also interesting
  (sparser intervention, same effect). Evidence against H1 = flat or
  monotone-decreasing surface vs full recon.
- **H2 (non-monotone in k%).** The k% axis at fixed α will be non-monotone:
  k%=1 undershoots (too few features carry the capability), k%=100
  approaches the full-partition uniform case (post-top-k delta on every
  active A-excl feature, which most closely matches the §6 inert
  baseline), and the optimum sits at k% ∈ [4, 32]. Evidence against H2 =
  monotone curves.
- **H3 (semantic clustering).** A-excl decoder vectors of the
  lowest-side-effect model (`dfc-D8k-excl10-freeexcl-k160`) form a small
  number (3-7) of interpretable clusters when projected by UMAP and
  clustered by HDBSCAN, and Gemma-2-9B-it can name those clusters
  consistently from member-feature descriptions.
- **H4 (RL-only, no code).** Hesitation and stubbornness boundaries
  (Pan et al. 2026 cited in §8.3) are observable behavioural endpoints
  that should respond to targeted steering of tool-calling neurons.
  Writeup proposes the experimental protocol; no execution in this spec.

---

## 2. Sub-projects

| ID | Sub-project | Outputs | Depends on |
|---|---|---|---|
| **P0** | Bring `models.jsonl` + `tool_neurons_A_full.csv` into working tree | `models.jsonl`, `data/rankings/<model>.csv` | — |
| **P1** | Targeted steering sweep (post-top-k delta) | `results/targeted_steering/*.jsonl`, `results/targeted_steering/summary.csv` | P0 |
| **P2** | Local-Gemma autointerp on A-excl + co-firing shared | `results/autointerp_local/<model>/feat_*.json` | P0, feature caches |
| **P3** | Decoder UMAP figures | `results/figures/umap_partitions.png`, `results/figures/umap_aexcl_clusters.png` | P2 |
| **P4** | Cluster-level meta-autointerp | `results/clusters/cluster_meta.json`, `results/figures/umap_aexcl_clusters_labeled.png` | P3 |
| **P5** | Sweep visualizations | `results/figures/heatmap_*.png`, `results/figures/paper_repro_*.png` | P1 |
| **P6** | Final markdown report | `results/REPORT.md` | P1–P5 |
| **P7** | RL writeup (no code) | `docs/rl_boundaries.md` | — |

P1, P2, P7 can run in parallel from the start. P3→P4→P6 chain after P2.
P5 chains after P1. P6 is the integration step.

---

## 3. P0 — Setup

Files to bring in from the analysis zip:

- `model_diffing_crosscoders-main/models.jsonl` → `models.jsonl` (root) —
  needed by `sweep_eval.py` and the new steering script.
- `model_diffing_crosscoders-main/neuron_identification/runs/dfc-D8k-excl10-freeexcl-k160_n1000/tool_neurons_A_full.csv`
  → `data/rankings/dfc-D8k-excl10-freeexcl-k160.csv`. This already encodes
  the **discriminative score** (Cohen's d on tool vs nontool firing) that
  Q1 asked for. We use the `cohens_d` column descending; ties broken by
  `fire_rate_tool` descending.
- For the other two recommended models (`cc-D8k-nol1-k45`,
  `dfc-D8k-excl10-k45`) the zip's `neuron_identification/` only has the
  D=8k-excl10-freeexcl-k160 run. We need to **rebuild rankings** for the
  other two via a new script `rank_features.py`. CrossCoders have no
  exclusive partition (`a_end=0`), so for `cc-D8k-nol1-k45` we rank the
  *full dictionary* by the same score and treat the top-N (N matched to
  the DFC's `n_a` for fairness, e.g. ~819) as the "A-excl-equivalent"
  pool. This is documented explicitly in REPORT.md as a methodological
  caveat.

**Rebuild `rank_features.py` algorithm:**

1. Load HF crosscoder via `sweep_eval.load_crosscoder`.
2. Sample N=1000 ToolRL prompts and N=1000 FineWeb prompts.
3. For each prompt, compute layer-13 last-token activations for both
   models, run `crosscoder.encode`, store the feature vector.
4. For each feature i, compute Cohen's d of activations on tool vs
   nontool, AUROC, fire rate on each class.
5. Write CSV to `data/rankings/<model>.csv` with the same schema as the
   zip's file.

This script is also useful as a sanity check for the zip's CSV (we should
reproduce the existing ranking on `dfc-D8k-excl10-freeexcl-k160` to within
noise).

---

## 4. P1 — Targeted steering sweep

**Script:** `run_steering_eval.py` (single-cell runner) +
`run_steering_sweep.sh` (wrapper).

### Sweep grid

- **Models** (3): `dfc-D8k-excl10-freeexcl-k160`, `cc-D8k-nol1-k45`,
  `dfc-D8k-excl10-k45`.
- **k%** (8): `[1, 2, 4, 8, 16, 32, 64, 100]`.
- **α** (6): `[0, 1, 6, 16, 32, 64]` — α=1 is a no-op anchor for sanity
  (delta should be exactly zero); α=0 ablates the selected subset
  (full suppression test).
- **Prompts:** 100 ToolRL prompts (same hold-out as the paper, seed=42).

Total: 3 × 8 × 6 = 144 cells × 100 prompts = 14,400 evaluations. With
greedy decoding and `max_new_tokens=200` on a single GPU, expect ~10–14
hours per model, so plan on running each model in its own tmux pane.

### Per-cell algorithm

Adapts `steering/steer.py::compute_steering_delta` to **Model A** (the
existing helper hard-codes Model B; we mirror its math but flip the
decoder index):

1. Tokenize prompt; extract `h_a` (layer-13 last token, Model A) and
   `h_b` (Model B).
2. Encode `(h_a, h_b)` through the crosscoder → sparse `features`
   (top-k applied here).
3. For the model's top-k%-most-discriminative subset `S` (selected from
   `data/rankings/<model>.csv`, rows with `sign == "tool"`, sorted by
   `cohens_d` desc, prefix of length `ceil(0.01 · k_pct · |A-excl|)`):
   `delta = sum_{i ∈ S} (α − 1) · features[i] · W_dec[i, 0, :]`
4. `patched_a = h_a + delta`.
5. Generate from **Model A** with the layer-13 last-token activation
   patched to `patched_a` for the prefill pass only.
6. Score generation with `score_response` (paper's ToolRL rubric).

### Baselines per row

For each prompt also compute and store:

- `clean_score`: unsteered Model A generation (the "pre-recon" Model A
  baseline ≈ 19% tool-correctness in the paper).
- `recon_score`: full reconstruction of Model A without targeted
  steering — patches `recon_a = crosscoder.decode(features)[0, 0, :]`
  into Model A. This is the paper's "post-recon Model A" condition
  (Table 1, ≈ +30 pp). Computed once per prompt and reused across cells.
- `steered_score`: from step 5 above.

The CrossCoder variant (`cc-D8k-nol1-k45`) has `a_end == 0`, so its
"top-k%-of-A-excl" is interpreted as top-k% of the discriminative-ranked
*full dictionary*. Documented as a methodological caveat in REPORT.md.

### Output schema

One JSONL file per (model, k%, α):
`results/targeted_steering/<model_name>/k<pct>_a<coef>.jsonl`

Each line:
```json
{
  "prompt_index": int,
  "prompt": str,
  "subset_indices": [int, ...],
  "subset_size": int,
  "k_pct": float,
  "alpha": float,
  "clean_score": {format_accuracy, tool_correctness, overall_score},
  "recon_score":  {...},
  "steered_score": {...},
  "subset_activations": {"original": {idx: val}, "steered": {idx: val}}
}
```

A summary roll-up `results/targeted_steering/summary.csv` aggregates per
cell: `model, k_pct, alpha, n, mean_clean_*, mean_recon_*, mean_steered_*,
delta_vs_clean_*, delta_vs_recon_*`.

### Crash-safety + resume

- Each (model, k%, α) cell writes to its own JSONL, so partial progress
  is preserved.
- Wrapper script skips a cell if its JSONL exists and has 100 lines.
- LLMs and crosscoder are loaded once per model and held across all
  48 (k%, α) cells for that model — never reload mid-sweep.

---

## 5. P2 — Local-Gemma autointerp

**Files:**
- New `autointerp_local.py` — same orchestration interface as
  `autointerp.py`, but `LocalGemmaClient` replaces `AsyncClaude`.
- New `run_autointerp_local.py` — entry point.
- New `build_feature_cache.py` — re-uses logic in `run_xcdr_cache.py` but
  parametrised by HF crosscoder name.

### Feature cache for the recommended model

Need a feature cache for `dfc-D8k-excl10-freeexcl-k160` covering the same
texts as `cache/toolrl/` and `cache/fineweb/`. The existing
`cache/toolrl_features` is from a *different* DFC (D=16k p=5%) and cannot
be reused.

`build_feature_cache.py` algorithm:

1. Load HF crosscoder.
2. For each shard in `cache/toolrl/` (raw activations): re-encode
   `(h_a, h_b)` → store sparse features as `shard_NNNNN.pt` in
   `cache/<model_name>_features_toolrl/`.
3. Same for `cache/fineweb/`.
4. Write `meta.json` matching the existing schema.

### Subset selection (A-excl only)

Per user direction, the shared partition is **out of scope** for
autointerp; we do not depend on the stale A-20 / A-665 anchors.

1. Load `data/rankings/dfc-D8k-excl10-freeexcl-k160.csv`.
2. Autointerp set = all 819 A-excl features (indices `[0, n_a)` per the
   crosscoder's `a_end`).
3. Optionally skip features with `fire_rate_tool < 0.005` (effectively
   dead on the eval corpus) to save compute. Documented in the run log.

No shared-partition computation; no co-firing pass; no Pearson matrix.

### Local-Gemma backend

- Model: `google/gemma-2-9b-it` (already cached locally per
  `~/.cache/huggingface/hub/`).
- Synchronous batched generation (no async) — the locality removes any
  reason for async semaphores.
- Same `EXPLAIN_SYSTEM`, `EXPLAIN_USER_TEMPLATE`, `DETECT_SYSTEM`,
  `DETECT_USER_TEMPLATE` prompts as the existing `autointerp.py`.
- **Scoring metric:** keep the existing detection-accuracy (10 max + 10
  random, accuracy = correct/20) for now, since rewriting to
  precision/recall is the *separate* HypotheSAEs scaffold task that
  prompted this conversation. Cross-reference: that gets its own design
  doc later.
- Output: `results/autointerp_local/<model>/feat_NNNNNNN.json` with the
  same `FeatureRecord` schema as `autointerp.py`.

### Estimated cost

819 A-excl features (minus any dead). Each feature = 2 Gemma calls
(explain + detect), ~2k tokens total. On a 9B model with 80GB GPU and
batch size 8, expect ~30s per feature → ~7h total. Run in tmux.

---

## 6. P3 — Decoder UMAP figures

**Script:** `run_umap.py`. Single model: `dfc-D8k-excl10-freeexcl-k160`.

### Figure 1: 3-partition UMAP

- Input: `dfc.W_dec[:, 0, :]` and `dfc.W_dec[:, 1, :]` per feature
  (concatenated → `(dict_size, 2 × activation_dim) = (8192, 4096)`).
- UMAP to 2D with `n_neighbors=30`, `min_dist=0.1`, `metric='cosine'`,
  `random_state=42`.
- Colour each point by partition: A-excl (red), B-excl (blue),
  shared (grey). The shared partition is huge so use `alpha=0.2` for
  shared, `alpha=0.9` for A/B-excl.
- Save `results/figures/umap_partitions.png` (300 DPI).

### Figure 2: A-excl-only UMAP with HDBSCAN clusters

- Input: `W_dec[:a_end, 0, :]` (A-side decoder of A-excl features only),
  shape `(819, 2048)`.
- UMAP same parameters.
- HDBSCAN on the 2D UMAP embedding: `min_cluster_size=20`,
  `min_samples=5`. Expect 3–7 clusters + noise.
- Save `results/figures/umap_aexcl_clusters.png`. Save cluster
  assignments to `results/clusters/aexcl_assignments.csv`
  (feature_idx, cluster_id, umap_x, umap_y).

---

## 7. P4 — Cluster-level meta-autointerp

**Script:** `run_cluster_meta.py`.

For each cluster from P3:

1. Read all member features' `explanation` fields from P2's JSONs.
2. Truncate each to 200 chars; concatenate as a numbered list.
3. Prompt Gemma:
   > "The following N feature descriptions were grouped together based on
   > decoder-direction similarity. Identify the common theme. Reply on
   > exactly two lines: `[CLUSTER_NAME]: <2-5 word label>` and
   > `[CLUSTER_SUMMARY]: <one sentence>`."
4. Save `results/clusters/cluster_meta.json`:
   `{cluster_id: {name, summary, n_members, members}}`.
5. Re-render Figure 2 with cluster names as legend entries:
   `results/figures/umap_aexcl_clusters_labeled.png`.

---

## 8. P5 — Sweep visualizations

**Script:** `build_steering_figures.py`.

Per model (3 heatmaps as requested):

- `heatmap_<model>_dA_tool_vs_clean.png`: 8 × 6 cell heatmap of
  `Δ_A_tool_correctness` (steered − clean) — primary deliverable.
  Cell text = raw delta. Diverging colormap centred at 0.
- `heatmap_<model>_dA_tool_vs_recon.png`: 8 × 6 heatmap of
  `Δ_A_tool_correctness` (steered − full-recon) — answers H1: does
  targeted steering beat full reconstruction?
- `heatmap_<model>_dA_overall_vs_clean.png`: same shape, overall_score
  metric.
- `lineplot_<model>_dA_vs_kpct.png`: one line per α, x = k%, y = Δ A
  tool-correctness vs clean.
- `lineplot_<model>_dA_vs_alpha.png`: one line per k%, x = α, y = Δ A
  tool-correctness vs clean.

Overall:

- `compare_models_best_cell.png`: bar chart of best `(k%, α)` cell per
  model vs paper's full-reconstruction baseline (Table 1, Δ_A
  ≈ +30 pp).
- Replicate paper figs from existing data — useful for direct comparison
  in REPORT.md:
  - Fig 1 (pre/post boxplots Model A side only) — read
    `results/results_full (1).jsonl`.
  - Fig 4 (top-k vs delta, Model A side only) — same source.
  - Fig 7 (uniform-scale boxplot) — same source. Overlay the targeted
    sweep's best cells to make the "selective ≠ uniform" point visually.

All figures saved to `results/figures/` at 300 DPI.

---

## 9. P6 — Final report

**File:** `results/REPORT.md`.

Structure:

1. **Abstract** — restates H1–H3 and the headline finding.
2. **Recap of paper finding** — embed Fig 1, Fig 7 from the paper.
3. **Targeted steering results** — embed all heatmaps and line plots
   from P5; quote the best `(model, k%, α)` cell per H1; comment on H2
   monotonicity per model.
4. **Decoder geometry** — embed UMAP figures from P3+P4; list cluster
   names from P4.
5. **A-excl autointerp summary** — table of top-20 A-excl features by
   discriminative score with their Gemma explanations and
   detection-accuracy.
6. **Discussion** — does targeted steering improve over uniform recon?
   Which clusters dominate the helpful-cell pattern?
7. **Open questions** — references `docs/rl_boundaries.md`.

---

## 10. P7 — RL writeup (no code)

**File:** `docs/rl_boundaries.md`.

Web research scope:

- Pan et al. 2026 (cited in §8.3) — find and summarise.
- Recent RL-tuning work on tool-call abstention / hesitation
  (search: "tool calling abstention", "RLHF hesitation",
  "knowledge boundary").
- Stubbornness in RL-tuned LLMs (search: "RL stubbornness",
  "policy entropy collapse", "exploration RL LLM").

Deliverable structure:

1. **Definitions.** Hesitation = refuses to call when call would help.
   Stubbornness = persists with a wrong call after evidence of failure.
2. **Existing work.** ≤ 1 paragraph per relevant paper; honest
   discussion of what they measured vs what we want to measure.
3. **Proposed protocol.**
   - Hesitation probe: prompts that nominally need a tool, but with
     ambiguous benefit. Measure call rate clean vs steered (modulating
     the best targeted-steering subset from P1, derived from the
     discriminative ranking).
   - Stubbornness probe: 2-turn prompts where turn-1 receives an
     explicit failure signal. Measure rate of repeated identical call.
4. **Predictions.** Targeted amplification of the top-k% A-excl subset
   identified by P1 should *decrease hesitation* (more calls); ablation
   (α=0) should *increase hesitation*. Stubbornness is less predictable.
5. **Failure modes.** Confounds with format-spillover; rubric only
   measures one call per response, so multi-turn rubric extension may
   be needed.

This document is **review-only**; no implementation in this spec.

---

## 11. File layout (final)

```
xcdr3/
├── docs/
│   ├── superpowers/specs/2026-04-25-...-design.md   (this doc)
│   └── rl_boundaries.md                              (P7)
├── data/
│   └── rankings/
│       ├── dfc-D8k-excl10-freeexcl-k160.csv          (P0, from zip)
│       ├── cc-D8k-nol1-k45.csv                       (P0, rebuilt)
│       └── dfc-D8k-excl10-k45.csv                    (P0, rebuilt)
├── cache/
│   ├── toolrl/                                       (existing)
│   ├── fineweb/                                      (existing, used by P0 ranking only)
│   └── dfc-D8k-excl10-freeexcl-k160_features_toolrl/ (P2)
├── results/
│   ├── targeted_steering/
│   │   ├── dfc-D8k-excl10-freeexcl-k160/k01_a00.jsonl ...
│   │   ├── cc-D8k-nol1-k45/...
│   │   ├── dfc-D8k-excl10-k45/...
│   │   └── summary.csv
│   ├── autointerp_local/dfc-D8k-excl10-freeexcl-k160/feat_*.json
│   ├── clusters/
│   │   ├── aexcl_assignments.csv
│   │   └── cluster_meta.json
│   ├── figures/                                       (P3, P4, P5)
│   └── REPORT.md                                      (P6)
├── models.jsonl                                       (P0, from zip)
├── rank_features.py                                   (P0)
├── build_feature_cache.py                             (P2)
├── autointerp_local.py                                (P2 lib)
├── run_autointerp_local.py                            (P2 entry)
├── run_steering_eval.py                               (P1 single-cell)
├── run_steering_sweep.sh                              (P1 wrapper)
├── run_umap.py                                        (P3)
├── run_cluster_meta.py                                (P4)
├── build_steering_figures.py                          (P5)
└── build_report.py                                    (P6)
```

---

## 12. Tmux command sequence

Conventions:

- All scripts run via `uv run` per project convention.
- Each long-running step in its own tmux pane so failures are isolated
  and you can detach (`Ctrl-b d`) and reattach (`tmux a -t <name>`).
- Heavy GPU steps: P1 needs Model A + Model B + crosscoder ≈ 14GB per
  model; P2 needs Gemma-9B ≈ 18GB. Sequence below assumes 4 visible GPUs
  (sweep × 3 + autointerp × 1). If you have fewer, drop the
  `CUDA_VISIBLE_DEVICES` lines and run sweeps sequentially in a single
  pane — total walltime grows roughly linearly. With 1 GPU, plan
  ~50 hours total (3 × ~14h sweeps + ~8h autointerp); with 2 GPUs,
  pair the smallest-VRAM jobs.

```bash
# ─── P0: setup (foreground, ~30s) ──────────────────────────────────────────
cd /home/cs29824/andre/xcdr3
mkdir -p data/rankings cache results/{targeted_steering,autointerp_local,clusters,figures}
cp /tmp/xcdr_zip/model_diffing_crosscoders-main/models.jsonl ./models.jsonl
cp /tmp/xcdr_zip/model_diffing_crosscoders-main/neuron_identification/runs/dfc-D8k-excl10-freeexcl-k160_n1000/tool_neurons_A_full.csv \
   data/rankings/dfc-D8k-excl10-freeexcl-k160.csv

# Rank the other two models (≈30 min each on GPU)
CUDA_VISIBLE_DEVICES=0 uv run python rank_features.py \
    --crosscoder antebe1/cc-D8k-nol1-k45         --out data/rankings/cc-D8k-nol1-k45.csv
CUDA_VISIBLE_DEVICES=0 uv run python rank_features.py \
    --crosscoder antebe1/dfc-D8k-excl10-k45      --out data/rankings/dfc-D8k-excl10-k45.csv


# ─── P1: targeted steering sweep — TMUX ────────────────────────────────────
# One tmux session per model so they parallelise across GPUs.

tmux new -d -s steer_dfc160 \
  "CUDA_VISIBLE_DEVICES=0 uv run bash run_steering_sweep.sh \
       --crosscoder antebe1/dfc-D8k-excl10-freeexcl-k160 \
       --rankings data/rankings/dfc-D8k-excl10-freeexcl-k160.csv \
       --out results/targeted_steering/dfc-D8k-excl10-freeexcl-k160 \
       2>&1 | tee logs/sweep_dfc160.log"

tmux new -d -s steer_cc45 \
  "CUDA_VISIBLE_DEVICES=1 uv run bash run_steering_sweep.sh \
       --crosscoder antebe1/cc-D8k-nol1-k45 \
       --rankings data/rankings/cc-D8k-nol1-k45.csv \
       --out results/targeted_steering/cc-D8k-nol1-k45 \
       2>&1 | tee logs/sweep_cc45.log"

tmux new -d -s steer_dfc45 \
  "CUDA_VISIBLE_DEVICES=2 uv run bash run_steering_sweep.sh \
       --crosscoder antebe1/dfc-D8k-excl10-k45 \
       --rankings data/rankings/dfc-D8k-excl10-k45.csv \
       --out results/targeted_steering/dfc-D8k-excl10-k45 \
       2>&1 | tee logs/sweep_dfc45.log"

# Watch any: tmux a -t steer_dfc160     Detach: Ctrl-b d


# ─── P2: feature cache + local autointerp — TMUX ───────────────────────────
# A-excl only (no shared, no co-firing). Single feature cache (toolrl).
tmux new -d -s autointerp \
  "CUDA_VISIBLE_DEVICES=3 uv run python build_feature_cache.py \
       --crosscoder antebe1/dfc-D8k-excl10-freeexcl-k160 \
       --source-cache cache/toolrl --out cache/dfc-D8k-excl10-freeexcl-k160_features_toolrl \
   && CUDA_VISIBLE_DEVICES=3 uv run python run_autointerp_local.py \
       --crosscoder antebe1/dfc-D8k-excl10-freeexcl-k160 \
       --feat-cache cache/dfc-D8k-excl10-freeexcl-k160_features_toolrl \
       --rankings   data/rankings/dfc-D8k-excl10-freeexcl-k160.csv \
       --partition  a_excl \
       --backend    google/gemma-2-9b-it \
       --out        results/autointerp_local/dfc-D8k-excl10-freeexcl-k160 \
       2>&1 | tee logs/autointerp.log"


# ─── P7: RL writeup — runs anytime, no GPU ─────────────────────────────────
# This is a manual research + writing task. Drive via a fresh Claude
# session pointed at docs/rl_boundaries.md if you want.


# ─── P3, P4, P5, P6: post-processing — FOREGROUND, after P1+P2 done ────────
# Wait for tmux sessions to exit (or just run when ready):

uv run python run_umap.py \
    --crosscoder antebe1/dfc-D8k-excl10-freeexcl-k160 \
    --out results/figures
uv run python run_cluster_meta.py \
    --autointerp results/autointerp_local/dfc-D8k-excl10-freeexcl-k160 \
    --clusters   results/clusters/aexcl_assignments.csv \
    --out        results/clusters/cluster_meta.json \
    --backend    google/gemma-2-9b-it
uv run python build_steering_figures.py \
    --steering-root results/targeted_steering \
    --paper-results "results/results_full (1).jsonl" \
    --out results/figures
uv run python build_report.py --out results/REPORT.md
```

---

## 13. Risks and mitigations

- **Risk:** Model A's recon baseline is already ≈ +30 pp; targeted
  steering may struggle to beat it (H1 partially disconfirmed). Even a
  null is informative if matched at smaller k% (sparser intervention,
  same effect). *Mitigation:* report parity outcomes prominently; the
  paper itself has a similar null in §6 and that is fine.
- **Risk:** the post-top-k delta with the discriminative subset shows
  zero improvement. *Mitigation:* honest reporting; §6 sets the
  precedent for null results being publishable.
- **Risk:** Gemma-2-9B-it underperforms on autointerp prompts vs
  Claude/GPT. *Mitigation:* spot-check the first 50 features manually;
  if quality is poor, add an Anthropic fallback path (cheap because most
  features are already named locally).
- **Risk:** the zip's `tool_neurons_A_full.csv` was generated with a
  different random seed for the prompt sample than what we use; rankings
  may shift. *Mitigation:* `rank_features.py` reproduces the ranking
  with seed=42; compare first 20 rows; tolerate ≤ 5 swaps.
- **Risk:** cache building blocks autointerp; a single bug in
  `build_feature_cache.py` halts P2. *Mitigation:* dry-run on 1 shard
  before kicking off the full job.
- **Risk:** UMAP cluster structure is a noise artefact and Gemma names
  spurious themes. *Mitigation:* report silhouette score alongside;
  treat cluster meta-names as suggestive, not definitive.

---

## 14. Out of scope

- HypotheSAEs precision/recall scoring (separate scaffold; this is a
  follow-up doc).
- Re-training any crosscoder.
- Retraining the base or ToolRL Qwen models.
- Implementing the RL hesitation/stubbornness probes (P7 is design-only).
- Multi-turn or agentic eval — paper's single-turn rubric only.
