# Layer-13 ablation — does targeted A-excl steering beat the baselines?

Conditions:

- `dfc-aexcl` — DFC · A-exclusive (`dfc-D8k-excl10-k45-l13`)  ✓
- `dfc-shared` — DFC · shared (`dfc-D8k-excl10-k45-l13-shared`)  ✓
- `dfc-bexcl` — DFC · B-exclusive (`dfc-D8k-excl10-k45-l13-bexcl`)  ✓
- `dfc-combo` — DFC · A-excl ∪ shared (`dfc-D8k-excl10-k45-l13-combo`)  ✓
- `cc` — CrossCoder · all (`cc-D8k-k45`)  ✓

Pairings: per (k%, α) cell, A-excl Δ tool-corr − baseline Δ tool-corr. One-sided H1: A-excl > baseline.

## Best cell per condition

| condition | best Δ (%) | 95% CI | |S| | α | n_prompts |
|-----------|-----------:|--------|----:|--:|----------:|
| DFC · A-exclusive | +65.0 | [+47.9, +82.1] | 1 | 32 | 40 |
| DFC · shared | +62.5 | [+43.8, +81.2] | 38 | 6 | 40 |
| DFC · B-exclusive | +0.0 | [+0.0, +0.0] | 1 | 1 | 40 |
| DFC · A-excl ∪ shared | +52.5 | [+34.8, +70.2] | 31 | 64 | 40 |
| CrossCoder · all | +70.0 | [+53.5, +86.5] | 33 | 6 | 40 |

## Paired tests (cell-level)

### DFC-A-excl vs DFC · shared (n = 30)

- mean Δ = +10.500 %-points  (95% CI [+0.659, +20.341])
- Cohen's d_z = 0.398
- paired t (one-sided, H1: A-excl > baseline): t = 2.182, p = 0.0187
- Wilcoxon signed-rank (one-sided): W = 217.0, p = 0.0277

### DFC-A-excl vs DFC · B-exclusive (n = 30)

- mean Δ = +24.333 %-points  (95% CI [+15.501, +33.165])
- Cohen's d_z = 1.029
- paired t (one-sided, H1: A-excl > baseline): t = 5.635, p = 2.18e-06
- Wilcoxon signed-rank (one-sided): W = 268.0, p = 3.79e-05

### DFC-A-excl vs DFC · A-excl ∪ shared (n = 30)

- mean Δ = +15.667 %-points  (95% CI [+5.226, +26.108])
- Cohen's d_z = 0.560
- paired t (one-sided, H1: A-excl > baseline): t = 3.069, p = 0.0023
- Wilcoxon signed-rank (one-sided): W = 242.0, p = 0.0043

### DFC-A-excl vs CrossCoder · all (n = 30)

- mean Δ = -0.083 %-points  (95% CI [-7.764, +7.597])
- Cohen's d_z = -0.004
- paired t (one-sided, H1: A-excl > baseline): t = -0.022, p = 0.5088
- Wilcoxon signed-rank (one-sided): W = 123.0, p = 0.5453

## Prompt-level paired tests

For every (k%, α, prompt_index) row in both conditions, compute steered-tool − clean-tool ∈ {−1, 0, +1}, then test (A-excl Δ) − (baseline Δ) > 0.

### Prompt-level DFC-A-excl vs DFC · shared (n = 1200)

- mean Δ = +10.500 %-points  (95% CI [+6.959, +14.041])
- Cohen's d_z = 0.168
- paired t (one-sided, H1: A-excl > baseline): t = 5.817, p = 3.84e-09
- Wilcoxon signed-rank (one-sided): W = 73416.0, p = 4.76e-09

### Prompt-level DFC-A-excl vs DFC · B-exclusive (n = 1200)

- mean Δ = +24.333 %-points  (95% CI [+21.158, +27.509])
- Cohen's d_z = 0.434
- paired t (one-sided, H1: A-excl > baseline): t = 15.034, p = 3.15e-47
- Wilcoxon signed-rank (one-sided): W = 83065.0, p = 1.35e-43

### Prompt-level DFC-A-excl vs DFC · A-excl ∪ shared (n = 1200)

- mean Δ = +15.667 %-points  (95% CI [+12.081, +19.252])
- Cohen's d_z = 0.247
- paired t (one-sided, H1: A-excl > baseline): t = 8.573, p = 1.54e-17
- Wilcoxon signed-rank (one-sided): W = 89169.5, p = 4.22e-17

### Prompt-level DFC-A-excl vs CrossCoder · all (n = 1200)

- mean Δ = -0.083 %-points  (95% CI [-3.487, +3.320])
- Cohen's d_z = -0.001
- paired t (one-sided, H1: A-excl > baseline): t = -0.048, p = 0.5192
- Wilcoxon signed-rank (one-sided): W = 46872.0, p = 0.5192
