
# Project documentation index

## Files in this directory

* [`paper.md`](paper.md) — the **consolidated living report**. Folds the prior sweep paper, prior descriptive report, and the current targeted-steering / autointerp / UMAP results into one structured narrative, with all formulas inlined.
* [`rl_boundaries.md`](rl_boundaries.md) — pre-existing design notes on the RL hesitation / stubbornness probe.
* [`superpowers/`](superpowers/) — pre-existing specs and plans (referenced from `paper.md` as needed).

## Relationship to other authoritative artifacts

| Source of truth | Used by `paper.md` for |
|---|---|
| [`results/dfc_toolrl_sweep_paper.pdf`](../results/dfc_toolrl_sweep_paper.pdf) | sweep-level numbers (47 crosscoders, +31 pp Δ_A, +6.7 pp Δ_B, p-values, paper Figs 1, 4, 7) |
| [`results/Agent_AI_Final_Report.pdf`](../results/Agent_AI_Final_Report.pdf) | descriptive feature inventory (A-20, A-665, etc.), pre-/post-recon experiments at 512 / 2048 token truncation |
| [`results/REPORT.md`](../results/REPORT.md) | targeted-steering best cells (+55, +65 pp Δ_A) and the autointerp top-20 table from the prior 40-prompt run |
| [`results/results_full (1).jsonl`](<../results/results_full (1).jsonl>) | the actual per-prompt metrics from the prior sweep |
| [`results/targeted_steering/<short>/`](../results/targeted_steering/) | per-cell `(k%, α)` jsonl files (in flux during the in-progress 40-prompt sweep) |
| [`results/autointerp_local/dfc-D8k-excl10-freeexcl-k160/summary.json`](../results/autointerp_local/dfc-D8k-excl10-freeexcl-k160/summary.json) | autointerp counts and partition breakdown |
| [`results/clusters/`](../results/clusters/) | UMAP + HDBSCAN + Gemma meta-autointerp cluster meta |
| [`data/rankings/<short>.csv`](../data/rankings/) | Cohen's-d rankings per crosscoder, used by both autointerp and steering |

When numbers in `paper.md` need refreshing, regenerate from the corresponding source-of-truth file rather than editing prose by hand.

## Tag conventions

* `<!-- TODO: ... -->` — work still needed (e.g. "regenerate Sec 5 after stage-4 completes").
* `<!-- NOTE: ... -->` — provenance / staleness annotations.
* `<!-- FEEDBACK: ... -->` — open question for collaborators.

Find all open items: `grep -nE 'TODO|NOTE|FEEDBACK' docs/paper.md`.

## How figures are referenced

All figures are referenced by **relative path from `docs/`**, e.g. `![Caption](../results/figures/foo.png)`. When `pandoc`-converting to LaTeX or HTML these paths resolve correctly from the build directory if the build is run from `docs/`.

## Conversion plan

When a camera-ready version is needed:

```bash
cd /home/cs29824/andre/xcdr3/docs
pandoc paper.md -o paper.pdf --pdf-engine=xelatex
```

(Add a CSL file and `--citeproc` if/when the references are migrated to BibTeX; currently they are inline.)

## Regenerating after the in-progress sweep finishes

The 40-prompt targeted-steering sweep started at 23:57 UTC Apr 27 and is expected to complete in ~6 h. Once `tmux kill-session -t sweep_dfc160 sweep_dfc45 autointerp` exits naturally:

```bash
cd /home/cs29824/andre/xcdr3
python build_steering_figures.py --steering-root results/targeted_steering --out results/figures
python plot_steering_curves.py --steering-dir results/targeted_steering/dfc-D8k-excl10-freeexcl-k160 --out results/figures/steering_curves_dfc-D8k-excl10-freeexcl-k160.png
python plot_steering_curves.py --steering-dir results/targeted_steering/dfc-D8k-excl10-k45 --out results/figures/steering_curves_dfc-D8k-excl10-k45.png
python build_report.py            # rewrites results/REPORT.md
# Then re-read results/REPORT.md and update the affected numbers in docs/paper.md (Sec 5.1, Sec 7.1, Sec 9).
```
