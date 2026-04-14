# Model Selection Recommendation

## Selection Rule

The sweep results do not include direct reconstruction loss, so I used model A post-reconstruction quality as a practical proxy for "low loss."

Primary model A preservation metrics:

- `overall_score_A_post`
- `tool_correctness_A_post`
- `format_accuracy_A_post`

Primary model B side-effect metrics:

- `overall_score_B_post`
- `tool_correctness_B_post`

Lower side effects on model B means:

- more negative / less improved `overall_score_B_post`
- lower `tool_correctness_B_post`

## Recommended Model: Lowest Loss + Lowest Side Effects On Model B

**Model:** `antebe1/dfc-D8k-excl10-freeexcl-k160`

Why this model:

- `overall_score_A_post = 1.49`
- `tool_correctness_A_post = 83`
- `format_accuracy_A_post = 83`
- `overall_score_B_post = -1.0`
- `tool_correctness_B_post = 0`

Interpretation:

- This is the strongest model A preservation result in the sweep.
- It also shows essentially no measurable spillover into model B.
- If the goal is high reconstruction quality with minimal unintended B-side activation, this is the best candidate in the current file.

## Recommended Model: Lowest Loss + Highest Side Effects On Model B

**Model:** `antebe1/cc-D8k-nol1-k45`

Why this model:

- `overall_score_A_post = 1.1`
- `tool_correctness_A_post = 70`
- `format_accuracy_A_post = 70`
- `overall_score_B_post = -0.66`
- `tool_correctness_B_post = 17`

Interpretation:

- This model still preserves model A very well.
- At the same time, it produces strong model B side effects relative to the cleaner candidates.
- This makes it a good choice if the goal is to study spillover while keeping model A performance high.

## More Aggressive Spillover Alternative

If the goal is to maximize spillover more aggressively, even at some cost to model A preservation, then another useful candidate is:

**Model:** `antebe1/dfc-D8k-excl10-k45`

Metrics:

- `overall_score_A_post = 0.7`
- `tool_correctness_A_post = 57`
- `format_accuracy_A_post = 56`
- `overall_score_B_post = -0.6`
- `tool_correctness_B_post = 20`

Interpretation:

- This appears to produce even stronger B-side spillover than `cc-D8k-nol1-k45`.
- But it does so with weaker model A preservation.
- So it is better as a "high spillover probe" than as a balanced A-preserving model.

## Final Recommendation

If you want one clean pair for follow-up work:

- **Minimal-spillover model:** `antebe1/dfc-D8k-excl10-freeexcl-k160`
- **Spillover-study model:** `antebe1/cc-D8k-nol1-k45`

If you want a stronger spillover stress test:

- **Aggressive spillover probe:** `antebe1/dfc-D8k-excl10-k45`
