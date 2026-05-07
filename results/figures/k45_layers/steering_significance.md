# Does steering help? — paired tests on dfc-D8k-excl10-k45

Layers used (full 30-cell grid): `dfc-D8k-excl10-k45-l1, dfc-D8k-excl10-k45-l5, dfc-D8k-excl10-k45-l9, dfc-D8k-excl10-k45-l13, dfc-D8k-excl10-k45-l14, dfc-D8k-excl10-k45-l18, dfc-D8k-excl10-k45-l20, dfc-D8k-excl10-k45-l24, dfc-D8k-excl10-k45-l28`
Excluded (incomplete grid): `dfc-D8k-excl10-k45-l32`

Paired observations are (clean tool-correctness, steered tool-correctness). One-sided H1: steered > clean.

## 1. Prompt-level (every individual prompt)

Contingency over 10800 paired prompts:

|                | steered=0 | steered=1 |
|----------------|-----------|-----------|
| **clean=0**    |      6051 |      2319 (rescued) |
| **clean=1**    |       677 (broken)  |      1753 |

- McNemar (exact binomial, one-sided H1: rescued > broken): rescued=2319, broken=677, p = 6.07e-209
### Paired tests on prompt-level Δ (in %-points) (n = 10800)

- mean Δ = +15.204 %-points  (95% CI [+14.253, +16.155])
- Cohen's d_z = 0.301
- paired t (one-sided, H1: steered > clean): t = 31.331, p = 1.27e-206
- Wilcoxon signed-rank (one-sided): W = 3475021.5, p = 5.10e-198

## 2. Cell-level (per (layer, k%, α) cell mean)

### Paired tests on cell-mean Δ tool-corr (%) (n = 270)

- mean Δ = +15.204 %-points  (95% CI [+12.501, +17.906])
- Cohen's d_z = 0.674
- paired t (one-sided, H1: steered > clean): t = 11.077, p = 4.83e-24
- Wilcoxon signed-rank (one-sided): W = 11770.0, p = 7.92e-23

## 3. Best cell per layer

| layer | best k% | best α | Δ tool-corr (%) |
|-------|---------|--------|-----------------|
| dfc-D8k-excl10-k45-l1 | 4 | 16 | +67.5 |
| dfc-D8k-excl10-k45-l5 | 4 | 64 | +35.0 |
| dfc-D8k-excl10-k45-l9 | 4 | 64 | +52.5 |
| dfc-D8k-excl10-k45-l13 | 4 | 32 | +65.0 |
| dfc-D8k-excl10-k45-l14 | 4 | 16 | +52.5 |
| dfc-D8k-excl10-k45-l18 | 4 | 1 | +0.0 |
| dfc-D8k-excl10-k45-l20 | 4 | 6 | +70.0 |
| dfc-D8k-excl10-k45-l24 | 4 | 1 | +0.0 |
| dfc-D8k-excl10-k45-l28 | 4 | 64 | +47.5 |

### Paired tests on per-layer best Δ (%) (n = 9)

- mean Δ = +43.333 %-points  (95% CI [+22.681, +63.986])
- Cohen's d_z = 1.613
- paired t (one-sided, H1: steered > clean): t = 4.839, p = 0.0006
- Wilcoxon signed-rank (one-sided): W = 28.0, p = 0.0078

## Verdict

At every aggregation level, the mean Δ is positive and the one-sided paired t and Wilcoxon both reject the null at the 5% level (prompt p=1.27e-206, cell p=4.83e-24, best-per-layer p=0.0006).

Caveats: cells share underlying prompts and (k%, α) sweep cells within a layer are not independent draws, so prompt- and cell-level p-values overstate independent evidence. The best-cell-per-layer test (n=9) is the most conservative — every layer gets credited only at its peak — but multiple-comparison corrected by virtue of taking only the max per layer (selection bias inflates the mean Δ). Treat the verdict as robust *because* all three levels agree, not because any single one is definitive.
