# Knowledge / Hesitation / Stubbornness Boundaries — Experiment Design

**Status:** design only, no code.
**Date:** 2026-04-25.
**Scope:** §8.3 of `results/dfc_toolrl_sweep_paper.pdf`. Builds on the
targeted-steering work in
`docs/superpowers/specs/2026-04-25-targeted-steering-and-autointerp-design.md`.

---

## 1. Definitions

We define three behavioural endpoints that are observable from completions
without retraining either model.

- **Hesitation boundary.** The decision surface separating prompts where
  the model emits a `<tool_call>` from prompts where it does not, given
  that a tool would be helpful. *Hesitation* = refuses to call when a
  call would help. Concretely, on a prompt that obviously calls for a
  tool (e.g. "what's the weather in Boston *right now*"), does the model
  invoke the tool, or does it answer from priors?
- **Stubbornness boundary.** Persistence of an initial choice in the
  presence of contradicting evidence. *Stubbornness* = the model
  re-issues the same call (or doubles down on the same wrong answer)
  after a turn-1 failure signal that should have caused it to adapt.
- **Knowledge boundary** (background concept; not directly probed
  here). The set of factual queries the model can answer reliably from
  internal weights. We use it as the conceptual frame for the other two
  endpoints — both hesitation and stubbornness are *failures of
  calibrated self-knowledge* about whether to call vs answer / whether
  to retry vs revise.

---

## 2. Existing work

The following are the closest published anchors. Each item is a one-paragraph
honest summary — what they actually measured, not what they claim.

- **Knowledge Boundary Discovery (KBD)**
  ([arxiv:2603.21022][kbd]). Defines knowledge boundary via two automatically
  generated question pools (within / beyond model knowledge), then trains
  with RL to push the model toward calibrated abstention. Useful as a
  conceptual frame; the boundary detector is question-generation based,
  not activation based.
- **Reinforced Internal-External Knowledge Synergistic** ([arxiv:2505.07596][rieks]).
  Argues LLMs have *blurred* perception of which questions need external
  retrieval vs internal answer; trains with RL to sharpen that
  perception. Closest existing operationalisation of "hesitation
  boundary" as a learned threshold over self-knowledge.
- **TruthRL ternary reward** ([recent abstention survey][knowyourlimits]).
  Distinguishes correct / hallucinated / abstained; rewards abstention
  on unanswerable questions. RL recipe whose abstention behaviour we
  could ablate via targeted steering rather than retraining.
- **Reasoning fine-tuning degrades abstention**
  ([AbstentionBench, arxiv:2506.09038][abstentionbench]). Shows that
  GRPO / verifier-driven RL on math + IF tasks *erodes* abstention on
  unanswerable inputs. Strong empirical motivation: ToolRL fine-tuning
  may have similarly eroded the *hesitation* signal in Model A. The DFC
  exclusive-partition gives us a tool to revert that without retraining.
- **Stubbornness measurement**
  ([Medium write-up][stubbornness]). Operationalises stubbornness as
  the slope of an LLM's tendency to revert to its priors under
  counter-evidence. Behavioural metric, not mechanistic; we adopt the
  measurement protocol but probe with steering instead of prompting.
- **AbstentionBench** ([github][ab-github]). 20 datasets aggregated for
  abstention evaluation; the natural held-out for the hesitation probe
  if we want a comparison axis beyond ToolRL prompts.

[kbd]: https://arxiv.org/html/2603.21022
[rieks]: https://arxiv.org/pdf/2505.07596
[knowyourlimits]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754/131566/Know-Your-Limits-A-Survey-of-Abstention-in-Large
[abstentionbench]: https://arxiv.org/pdf/2506.09038
[stubbornness]: https://medium.com/@hadiazouni/measuring-the-stubbornness-of-llms-d45c51bc1911
[ab-github]: https://github.com/facebookresearch/AbstentionBench

What's missing from the literature: every cited work changes behaviour by
re-training (RL) or re-prompting. None of them probe the boundary by
*activation steering* of the SAE features that the RL fine-tune itself
created. That is the gap we propose to fill.

---

## 3. Proposed protocols

Both probes reuse the same 3-model crosscoder set as the steering sweep
(`dfc-D8k-excl10-freeexcl-k160`, `cc-D8k-nol1-k45`, `dfc-D8k-excl10-k45`)
and the same `(k%, α)` grid. The intervention vehicle is the
post-top-k delta from the existing `run_steering_eval.py`.

### 3.1 Hesitation probe

**Construction of the eval set.** Two prompt buckets, ~50 prompts each:

- **Tool-needed bucket.** Prompts that materially require a tool to
  answer correctly (current weather, live stock price, fresh news,
  arithmetic with large numbers, file content lookup, etc.). Source:
  hand-curate from ToolRL train slice plus injected "today's date"
  questions; cross-check with AbstentionBench's "outdated information"
  subset.
- **Tool-not-needed bucket.** Prompts that are answerable from priors
  alone (well-known facts, reasoning puzzles, opinion). Same dataset
  origins, filtered for "no temporal context" and "no external data".

**Per-prompt measurement.** On Model A only:

1. `clean` — generate without intervention.
2. `recon` — generate with full crosscoder reconstruction patched.
3. `steered` — for each `(k%, α)` cell, generate with the targeted
   subset patched.

**Outcomes.** For each completion, two binary signals:

- `called_tool`: contains `<tool_call>`.
- `tool_was_needed`: belongs to bucket 1.

Aggregate to a 2 × 2 confusion matrix per cell. Hesitation rate =
`P(¬called_tool | tool_was_needed)` — the false-negative rate on the
tool-needed bucket.

**Sweep deliverable.** Heatmap `hesitation_rate` (k% × α) per model,
delta vs `clean`. Mirror format to the steering-fidelity heatmaps so
they can sit side by side in the report.

### 3.2 Stubbornness probe

**Construction of the eval set.** ~40 multi-turn prompts where turn-1
calls a wrong tool / wrong arguments and the simulated environment
returns an explicit failure observation. Three sub-styles:

- **Same-name-typo:** correct tool exists with similar name; the tool
  the model first called returns "function not found".
- **Wrong-args:** correct tool name, malformed argument; environment
  returns a parse error with the offending field highlighted.
- **Refusing-environment:** environment returns a permission denial that
  suggests trying a different tool.

**Per-prompt measurement.** Two-turn rollout:

1. Turn-1 generation under each condition (clean / recon / steered per
   cell).
2. Insert a synthetic environment turn with the explicit failure signal.
3. Turn-2 generation under the same condition (steering re-applied at
   turn-2 prefill, *not* during turn-1 — we want to measure how
   steering changes the model's response to evidence, not the initial
   call).

**Outcomes.**

- `same_call_repeated`: turn-2 emits an identical `<tool_call>` to
  turn-1 (lexical comparison after JSON normalisation).
- `revised_correctly`: turn-2 emits a tool call that matches the
  intended correct tool / args.

Aggregate to per-cell `stubbornness_rate = P(same_call_repeated)` and
`recovery_rate = P(revised_correctly)`. Recovery is a more useful signal
than stubbornness alone because a model can avoid stubbornness by simply
giving up (refusing to call). Both should be reported.

### 3.3 Cross-condition control

Add a permutation control: replicate every condition with a *random*
A-excl subset of the same size as the targeted top-k%. A real effect of
targeted steering means random subsets should not produce the same
shift; if they do, we are measuring a generic activation-energy effect
not a feature-specific one. This is cheap (one extra (k%, α) column) and
strong evidence against non-specific confounds.

---

## 4. Predictions

These are the predictions we will pre-register before running the probes.
They are intentionally falsifiable — an opposite outcome is informative
and would be reported in `REPORT.md`.

| Probe | Intervention | Predicted direction | Mechanism |
|---|---|---|---|
| Hesitation | α > 1 on best top-k% from P1 | ↓ hesitation rate (more calls on tool-needed) | Amplifying tool-binding features lowers the threshold for emitting `<tool_call>` |
| Hesitation | α = 0 (ablation) | ↑ hesitation rate | Suppressing tool-binding features removes the surface convention |
| Hesitation | α > 1 on tool-not-needed bucket | weakly ↑ false-positive rate | Amplification is non-specific to need, only to surface form |
| Stubbornness | α > 1 (turn-2 only) | ↑ same_call_repeated | Stronger activation of the binding circuit overrides counter-evidence |
| Stubbornness | α = 0 (turn-2 only) | ↑ recovery_rate | Suppressing binding lets evidence drive the next call |
| Random-subset control | matched k% size | ≈ no shift | Targeted features are causally implicated; random ones are not |

If the random-subset control shows non-zero shifts of similar magnitude,
the interpretation collapses to "activation energy matters, identity
doesn't" and we report that honestly.

---

## 5. Failure modes and confounds

- **Format spillover.** Even if a tool call is *intended*, the rubric
  requires the literal `<tool_call>` token. Steering may invert the
  intent without changing the surface, or vice versa. Decouple by
  reporting `intended_tool_call` (regex on JSON-name field) separately
  from `format_accuracy`. The paper's rubric already does this (see
  `score_response`); reuse it.
- **Single-turn rubric on multi-turn task.** `score_response` was
  built for single-turn ToolRL. The stubbornness probe needs at minimum
  a two-turn-aware variant. We propose a thin extension that scores
  turn-2 under the same rubric and adds the `same_call_repeated`
  comparator; this is implementation work for follow-up, not part of
  this design.
- **Underpowered buckets.** 50 hand-curated tool-needed prompts may not
  surface small effects. Bootstrap CIs over the per-prompt outcomes;
  flag any cell whose CI crosses zero.
- **Confound with reconstruction.** Patching at all (even with α=1,
  which is delta = 0) still goes through the crosscoder encode/decode
  path. We compare against `recon_a` (full reconstruction patched
  back) — not against `clean` only — to isolate the steering-specific
  effect from the reconstruction-baseline effect.
- **Crosscoder-architecture confound.** CrossCoder runs lack a
  partition; "top-k%-of-A-excl" collapses to "top-k%-of-full-dict".
  This is documented in the steering-sweep spec but worth re-flagging
  here: any cross-architecture effect on hesitation may be an artefact
  of pool definition, not of steering content.
- **Pre-registration drift.** If we look at the data first and then pick
  the cells to report, we will mistake noise for signal. The protocol
  fixes the cells to (best cell from P1) + (matched random control) +
  (α=0 ablation) before any probe data is generated.

---

## 6. Sequencing into the broader project

This work is **downstream of** the steering sweep (P1) and the targeted
autointerp (P2 / P4). Specifically we need:

1. P1 sweep complete → defines the "best (k%, α)" cell per model.
2. P2 + P4 results → cluster names give us a candidate breakdown of
   *which* A-excl cluster carries the binding circuit (e.g. "JSON
   format binding" vs "tool-name lookup"). Lets us steer one cluster
   at a time and report cluster-level hesitation/stubbornness deltas.
3. Then this probe is implemented as `run_boundary_probe.py` reusing
   the steering helpers.

Estimated additional code: ~400 lines (probe runner + 2-turn rubric
extension). Estimated walltime per model: ~6 GPU-hours if run with the
same `(k%, α)` grid, ~2 hours if restricted to 4 cells per model.

---

## 7. Out of scope (this writeup)

- Re-training Model A or Model B with RL.
- Replacing the rubric with an LLM-as-judge.
- Multi-tool agentic loops > 2 turns.
- Probing the shared partition (the steering sweep itself excludes it
  per the parent spec).
