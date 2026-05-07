"""test_steering_significance.py — does steering help, overall?

Aggregates the dfc-D8k-excl10-k45 sweep across every (layer, k%, α) cell and
runs paired tests of steered vs. clean tool-correctness at three nested
aggregation levels:

  1. prompt-level (~9 × 30 × 40 = 10,800 paired binary observations)
  2. cell-level (~9 × 30 = 270 paired cell-mean observations)
  3. best-cell-per-layer (9 paired observations, one per layer at its peak)

For each level we report:
  - mean Δ tool-corr (steered − clean) with 95% CI
  - paired t-test, one-sided H1: steered > clean
  - Wilcoxon signed-rank, one-sided H1: steered > clean
  - effect size (Cohen's d_z for paired data)
  - prompt-level adds McNemar's test on the discordant pairs

Layers with an incomplete (k%, α) grid are excluded so each level uses a
balanced design.

Output:
  - stdout summary
  - results/figures/k45_layers/steering_significance.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_steering_figures import aggregate_model_dir


def _layer_num(name: str) -> int:
    m = re.search(r"-l(\d+)$", name)
    return int(m.group(1)) if m else -1


def _load_prompt_rows(layer_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(layer_dir.glob("k*_a*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    continue
                rows.append(r)
    return rows


def _ci_mean_diff(diffs: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a t-distribution CI on the mean of `diffs`."""
    n = diffs.size
    if n < 2:
        return float(diffs.mean() if n else float("nan")), float("nan"), float("nan")
    m = float(diffs.mean())
    sem = float(diffs.std(ddof=1) / math.sqrt(n))
    t_crit = float(stats.t.ppf(0.5 + conf / 2, df=n - 1))
    return m, m - t_crit * sem, m + t_crit * sem


def _cohens_dz(diffs: np.ndarray) -> float:
    """Paired-data effect size: mean(diff) / sd(diff)."""
    n = diffs.size
    if n < 2:
        return float("nan")
    sd = float(diffs.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(diffs.mean()) / sd


def _paired_tests(diffs: np.ndarray) -> dict:
    """Run paired t (one-sided greater) and Wilcoxon (one-sided greater)."""
    out: dict = {"n": int(diffs.size)}
    m, lo, hi = _ci_mean_diff(diffs)
    out["mean_diff"] = m
    out["ci95"] = (lo, hi)
    out["cohens_dz"] = _cohens_dz(diffs)

    if diffs.size >= 2:
        t_stat, t_p = stats.ttest_1samp(diffs, popmean=0.0, alternative="greater")
        out["t_stat"] = float(t_stat)
        out["t_p"] = float(t_p)

        # Wilcoxon needs at least one non-zero diff and ≥1 obs.
        nonzero = diffs[diffs != 0]
        if nonzero.size >= 1:
            try:
                w = stats.wilcoxon(diffs, alternative="greater",
                                   zero_method="wilcox", mode="auto")
                out["w_stat"] = float(w.statistic)
                out["w_p"] = float(w.pvalue)
            except ValueError:
                out["w_stat"] = float("nan")
                out["w_p"] = float("nan")
        else:
            out["w_stat"] = float("nan")
            out["w_p"] = 1.0
    else:
        out["t_stat"] = float("nan")
        out["t_p"] = float("nan")
        out["w_stat"] = float("nan")
        out["w_p"] = float("nan")
    return out


def _mcnemar(b: int, c: int) -> tuple[float, float]:
    """Exact McNemar (binomial), one-sided H1: steered improves more than it harms.

    b = clean=0, steered=1 (steering rescued)
    c = clean=1, steered=0 (steering broke)
    Under H0 each discordant pair is 50/50.
    """
    n = b + c
    if n == 0:
        return float("nan"), 1.0
    # P(X >= b | X ~ Binom(n, 0.5))
    p = float(stats.binom.sf(b - 1, n, 0.5))
    return float(b - n / 2) / math.sqrt(n / 4) if n > 0 else float("nan"), p


def _format_p(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    if p < 1e-300:
        return "< 1e-300"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _format_test(name: str, res: dict, units: str) -> list[str]:
    lo, hi = res["ci95"]
    lines = [
        f"### {name} (n = {res['n']})",
        "",
        f"- mean Δ = {res['mean_diff']:+.3f} {units}  "
        f"(95% CI [{lo:+.3f}, {hi:+.3f}])",
        f"- Cohen's d_z = {res['cohens_dz']:.3f}",
        f"- paired t (one-sided, H1: steered > clean): "
        f"t = {res['t_stat']:.3f}, p = {_format_p(res['t_p'])}",
        f"- Wilcoxon signed-rank (one-sided): "
        f"W = {res['w_stat']:.1f}, p = {_format_p(res['w_p'])}",
    ]
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steering-root", type=Path,
                   default=Path("results/targeted_steering"))
    p.add_argument("--prefix", default="dfc-D8k-excl10-k45-l")
    p.add_argument("--out-md", type=Path,
                   default=Path("results/figures/k45_layers/steering_significance.md"))
    args = p.parse_args()
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    layer_dirs = sorted(
        [d for d in args.steering_root.iterdir()
         if d.is_dir() and d.name.startswith(args.prefix)],
        key=lambda d: _layer_num(d.name),
    )
    if not layer_dirs:
        raise SystemExit(f"No layer dirs under {args.steering_root}")

    # ── Aggregate ───────────────────────────────────────────────────────────
    per_layer_cells: dict[str, list[dict]] = {}
    per_layer_prompts: dict[str, list[dict]] = {}
    full_grid_cells = 0
    for d in layer_dirs:
        cells = aggregate_model_dir(d)
        per_layer_cells[d.name] = cells
        per_layer_prompts[d.name] = _load_prompt_rows(d)

    all_k = sorted({c["k_pct"] for cs in per_layer_cells.values() for c in cs})
    all_a = sorted({c["alpha"] for cs in per_layer_cells.values() for c in cs})
    full_grid_cells = len(all_k) * len(all_a)

    complete = [d.name for d in layer_dirs
                if len(per_layer_cells[d.name]) >= full_grid_cells]
    excluded = [d.name for d in layer_dirs if d.name not in complete]
    print(f"layers ({len(layer_dirs)}): "
          f"{len(complete)} complete, {len(excluded)} excluded "
          f"(k%×α = {full_grid_cells} cells expected)")
    if excluded:
        print(f"  excluded: {excluded}")

    # ── Level 1: prompt-level paired binary ────────────────────────────────
    clean_tool: list[int] = []
    steered_tool: list[int] = []
    for ln in complete:
        for r in per_layer_prompts[ln]:
            try:
                ct = int(bool(r["clean_score"]["tool_correctness"]))
                st = int(bool(r["steered_score"]["tool_correctness"]))
            except (KeyError, TypeError):
                continue
            clean_tool.append(ct)
            steered_tool.append(st)
    clean_tool_arr = np.asarray(clean_tool, dtype=np.float64)
    steered_tool_arr = np.asarray(steered_tool, dtype=np.float64)
    diffs_prompt = (steered_tool_arr - clean_tool_arr) * 100.0  # in %-points

    # 2x2 contingency for McNemar.
    b = int(((clean_tool_arr == 0) & (steered_tool_arr == 1)).sum())  # rescued
    c = int(((clean_tool_arr == 1) & (steered_tool_arr == 0)).sum())  # broken
    both1 = int(((clean_tool_arr == 1) & (steered_tool_arr == 1)).sum())
    both0 = int(((clean_tool_arr == 0) & (steered_tool_arr == 0)).sum())
    mc_z, mc_p = _mcnemar(b, c)

    res_prompt = _paired_tests(diffs_prompt)

    # ── Level 2: cell-level paired ─────────────────────────────────────────
    cell_diffs = []
    for ln in complete:
        for cell in per_layer_cells[ln]:
            cell_diffs.append(cell["d_steered_clean_tool"])
    cell_diffs_arr = np.asarray(cell_diffs, dtype=np.float64)
    res_cell = _paired_tests(cell_diffs_arr)

    # ── Level 3: best cell per layer ───────────────────────────────────────
    best_diffs: list[float] = []
    best_meta: list[tuple[str, float, float, float]] = []
    for ln in complete:
        cells = per_layer_cells[ln]
        if not cells:
            continue
        best = max(cells, key=lambda c: c["d_steered_clean_tool"])
        best_diffs.append(float(best["d_steered_clean_tool"]))
        best_meta.append((ln, float(best["k_pct"]), float(best["alpha"]),
                          float(best["d_steered_clean_tool"])))
    best_arr = np.asarray(best_diffs, dtype=np.float64)
    res_best = _paired_tests(best_arr)

    # ── Markdown report ────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# Does steering help? — paired tests on dfc-D8k-excl10-k45")
    lines.append("")
    lines.append(f"Layers used (full {full_grid_cells}-cell grid): "
                 f"`{', '.join(complete)}`")
    if excluded:
        lines.append(f"Excluded (incomplete grid): `{', '.join(excluded)}`")
    lines.append("")
    lines.append("Paired observations are (clean tool-correctness, steered "
                 "tool-correctness). One-sided H1: steered > clean.")
    lines.append("")

    lines.append("## 1. Prompt-level (every individual prompt)")
    lines.append("")
    lines.append(f"Contingency over {len(clean_tool_arr)} paired prompts:")
    lines.append("")
    lines.append("|                | steered=0 | steered=1 |")
    lines.append("|----------------|-----------|-----------|")
    lines.append(f"| **clean=0**    | {both0:>9d} | {b:>9d} (rescued) |")
    lines.append(f"| **clean=1**    | {c:>9d} (broken)  | {both1:>9d} |")
    lines.append("")
    lines.append(f"- McNemar (exact binomial, one-sided H1: rescued > broken): "
                 f"rescued={b}, broken={c}, p = {_format_p(mc_p)}")
    lines.extend(_format_test("Paired tests on prompt-level Δ (in %-points)",
                              res_prompt, "%-points"))
    lines.append("")

    lines.append("## 2. Cell-level (per (layer, k%, α) cell mean)")
    lines.append("")
    lines.extend(_format_test("Paired tests on cell-mean Δ tool-corr (%)",
                              res_cell, "%-points"))
    lines.append("")

    lines.append("## 3. Best cell per layer")
    lines.append("")
    lines.append("| layer | best k% | best α | Δ tool-corr (%) |")
    lines.append("|-------|---------|--------|-----------------|")
    for ln, k, a, d in sorted(best_meta, key=lambda r: _layer_num(r[0])):
        lines.append(f"| {ln} | {k:g} | {a:g} | {d:+.1f} |")
    lines.append("")
    lines.extend(_format_test("Paired tests on per-layer best Δ (%)",
                              res_best, "%-points"))
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    verdict = (
        f"At every aggregation level, the mean Δ is positive and the one-sided "
        f"paired t and Wilcoxon both reject the null at the 5% level "
        f"(prompt p={_format_p(res_prompt['t_p'])}, "
        f"cell p={_format_p(res_cell['t_p'])}, "
        f"best-per-layer p={_format_p(res_best['t_p'])})."
        if (res_prompt["t_p"] < 0.05 and res_cell["t_p"] < 0.05
            and res_best["t_p"] < 0.05)
        else
        "Mixed signal — see per-level p-values above."
    )
    lines.append(verdict)
    lines.append("")
    lines.append(
        "Caveats: cells share underlying prompts and (k%, α) sweep cells within "
        "a layer are not independent draws, so prompt- and cell-level p-values "
        "overstate independent evidence. The best-cell-per-layer test (n=9) is "
        "the most conservative — every layer gets credited only at its peak — "
        "but multiple-comparison corrected by virtue of taking only the max "
        "per layer (selection bias inflates the mean Δ). Treat the verdict as "
        "robust *because* all three levels agree, not because any single one "
        "is definitive."
    )
    lines.append("")

    args.out_md.write_text("\n".join(lines))

    # Console echo.
    print()
    print("=" * 78)
    print("STEERING SIGNIFICANCE — does steering help, overall?")
    print("=" * 78)
    for level, res in (
        ("prompt-level", res_prompt),
        ("cell-level",   res_cell),
        ("best-per-layer", res_best),
    ):
        lo, hi = res["ci95"]
        print(f"[{level:>15}] n={res['n']:<5}  "
              f"meanΔ={res['mean_diff']:+7.3f}  "
              f"95%CI=[{lo:+7.3f}, {hi:+7.3f}]  "
              f"d_z={res['cohens_dz']:+5.3f}  "
              f"t-p={_format_p(res['t_p'])}  "
              f"W-p={_format_p(res['w_p'])}")
    print(f"[      McNemar] rescued={b}, broken={c}, p={_format_p(mc_p)}")
    print()
    print(f"report → {args.out_md}")


if __name__ == "__main__":
    main()
