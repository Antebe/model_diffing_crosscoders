"""plot_ablation_l13.py — partition ablation + DFC-vs-CC comparison at l13.

Compares four conditions on layer 13 of the dfc-D8k-excl10-k45 / cc-D8k-k45
sweep:

  1. dfc-aexcl  : DFC, top-k% selected from A-exclusive features
                  (existing dir: results/targeted_steering/dfc-D8k-excl10-k45-l13)
  2. dfc-shared : DFC, top-k% selected from the shared partition
                  (results/targeted_steering/dfc-D8k-excl10-k45-l13-shared)
  3. dfc-bexcl  : DFC, top-k% selected from B-exclusive features
                  (results/targeted_steering/dfc-D8k-excl10-k45-l13-bexcl)
  4. cc         : CrossCoder (no partition; full-dictionary top-k)
                  (results/targeted_steering/cc-D8k-k45)

Outputs in --out:
  - heatmap_<cond>_dA_tool_vs_clean.png  (one per condition)
  - ablation_envelope_vs_alpha.png       (best Δ over k% per condition)
  - ablation_envelope_vs_kpct.png        (best Δ over α per condition)
  - ablation_best_cell_bars.png
  - ablation_significance.md             (paired-cell tests vs DFC-A-excl)

Stat tests (one-sided H1: DFC-A-excl > baseline) pair the (k%, α) cells
between conditions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_steering_figures import (
    aggregate_model_dir,
    grid_from_rows,
    heatmap,
)


CONDITIONS = [
    ("dfc-aexcl",  "DFC · A-exclusive",       "dfc-D8k-excl10-k45-l13"),
    ("dfc-shared", "DFC · shared",            "dfc-D8k-excl10-k45-l13-shared"),
    ("dfc-bexcl",  "DFC · B-exclusive",       "dfc-D8k-excl10-k45-l13-bexcl"),
    ("dfc-combo",  "DFC · A-excl ∪ shared",   "dfc-D8k-excl10-k45-l13-combo"),
    ("cc",         "CrossCoder · all",        "cc-D8k-k45"),
]

CONDITION_COLORS = {
    "dfc-aexcl":  "#1f77b4",
    "dfc-shared": "#2ca02c",
    "dfc-bexcl":  "#9467bd",
    "dfc-combo":  "#ff7f0e",
    "cc":         "#d62728",
}

METRIC = "d_steered_clean_tool"
METRIC_PRETTY = "Δ tool-corr (steered − clean), %"


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


def _format_p(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    if p < 1e-300:
        return "< 1e-300"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _ci_mean_diff(x: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    n = x.size
    if n < 2:
        return float(x.mean() if n else float("nan")), float("nan"), float("nan")
    m = float(x.mean())
    sem = float(x.std(ddof=1) / math.sqrt(n))
    t_crit = float(stats.t.ppf(0.5 + conf / 2, df=n - 1))
    return m, m - t_crit * sem, m + t_crit * sem


def _paired_test(diffs: np.ndarray) -> dict:
    out = {"n": int(diffs.size)}
    m, lo, hi = _ci_mean_diff(diffs)
    out["mean_diff"] = m
    out["ci95"] = (lo, hi)
    if diffs.size >= 2:
        sd = float(diffs.std(ddof=1))
        out["cohens_dz"] = m / sd if sd > 0 else float("nan")
        t_stat, t_p = stats.ttest_1samp(diffs, popmean=0.0, alternative="greater")
        out["t_stat"] = float(t_stat)
        out["t_p"] = float(t_p)
        if (diffs != 0).any():
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
        out["cohens_dz"] = float("nan")
        out["t_stat"] = float("nan")
        out["t_p"] = float("nan")
        out["w_stat"] = float("nan")
        out["w_p"] = float("nan")
    return out


def _best_per_axis(rows: list[dict], axis: str, axis_values: list[float],
                    metric: str) -> list[float]:
    bucket: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        bucket[float(r[axis])].append(float(v))
    return [
        max(bucket[x]) if bucket[x] else float("nan") for x in axis_values
    ]


def _best_with_ci_per_axis(
    rows: list[dict], axis: str, axis_values: list[float], metric: str,
) -> tuple[list[float], list[float], list[float]]:
    """For each axis value, return (best Δ, lo, hi) where lo/hi are the 95% CI
    of the cell that achieved the best Δ. Cells with metric `d_steered_clean_tool`
    have CI under d_tool_ci_lo / d_tool_ci_hi."""
    ci_lo_key = "d_tool_ci_lo" if metric == "d_steered_clean_tool" else None
    ci_hi_key = "d_tool_ci_hi" if metric == "d_steered_clean_tool" else None
    best: dict[float, dict] = {}
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        x = float(r[axis])
        cur = best.get(x)
        if cur is None or float(v) > cur["v"]:
            best[x] = {
                "v": float(v),
                "lo": float(r.get(ci_lo_key, float("nan"))) if ci_lo_key else float("nan"),
                "hi": float(r.get(ci_hi_key, float("nan"))) if ci_hi_key else float("nan"),
            }
    ys, los, his = [], [], []
    for x in axis_values:
        b = best.get(x)
        if b is None:
            ys.append(float("nan")); los.append(float("nan")); his.append(float("nan"))
        else:
            ys.append(b["v"]); los.append(b["lo"]); his.append(b["hi"])
    return ys, los, his


def _plot_envelope(
    per_cond_best: dict[str, list[float]],
    cond_labels: dict[str, str],
    x_values: list[float],
    x_label: str,
    title: str,
    out_path: Path,
    per_cond_ci: dict[str, tuple[list[float], list[float]]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    xs = np.asarray(x_values, dtype=np.float64)
    for cond, ys_list in per_cond_best.items():
        ys = np.asarray(ys_list, dtype=np.float64)
        finite = np.isfinite(ys)
        if not finite.any():
            continue
        color = CONDITION_COLORS.get(cond, "grey")
        if per_cond_ci is not None and cond in per_cond_ci:
            los_list, his_list = per_cond_ci[cond]
            los = np.asarray(los_list, dtype=np.float64)
            his = np.asarray(his_list, dtype=np.float64)
            err_finite = finite & np.isfinite(los) & np.isfinite(his)
            if err_finite.any():
                yerr_lo = ys[err_finite] - los[err_finite]
                yerr_hi = his[err_finite] - ys[err_finite]
                ax.errorbar(
                    xs[err_finite], ys[err_finite],
                    yerr=[yerr_lo, yerr_hi],
                    fmt="o", linewidth=2.0, markersize=7,
                    color=color, ecolor=color, capsize=3, alpha=0.95,
                    label=cond_labels.get(cond, cond),
                )
                # Connect the points with a line on top.
                ax.plot(
                    xs[finite], ys[finite],
                    color=color, linewidth=2.0, alpha=0.7,
                )
                continue
        ax.plot(
            xs[finite], ys[finite],
            marker="o", linewidth=2.0, markersize=7,
            color=color,
            label=cond_labels.get(cond, cond),
        )
    ax.set_xscale("log")
    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{v:g}" for v in x_values])
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"max of {METRIC_PRETTY}")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[envelope] {out_path}")


def _plot_best_cell_bars(
    per_cond_best_cell: dict[str, dict],
    cond_labels: dict[str, str],
    out_path: Path,
    title: str = "Layer 13: best cell across the sweep, per condition",
) -> None:
    conds = list(per_cond_best_cell.keys())
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    xs = np.arange(len(conds))
    ys = [per_cond_best_cell[c]["best_d"] if per_cond_best_cell[c] else 0.0
          for c in conds]
    colors = [CONDITION_COLORS.get(c, "grey") for c in conds]

    # Error bars from per-cell paired-t 95% CI on the steered − clean diff.
    yerr_lo = []
    yerr_hi = []
    for c in conds:
        info = per_cond_best_cell[c]
        if info is None:
            yerr_lo.append(0.0); yerr_hi.append(0.0)
            continue
        lo = info.get("ci_lo")
        hi = info.get("ci_hi")
        if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
            yerr_lo.append(0.0); yerr_hi.append(0.0)
        else:
            yerr_lo.append(max(0.0, info["best_d"] - lo))
            yerr_hi.append(max(0.0, hi - info["best_d"]))

    ax.bar(xs, ys, color=colors, alpha=0.9,
           yerr=[yerr_lo, yerr_hi], capsize=4,
           error_kw={"elinewidth": 1.4, "ecolor": "black"})

    label_pad = max(yerr_hi + [0.0]) + (max(ys) * 0.03 if max(ys) else 0.5)
    for x, c in zip(xs, conds):
        info = per_cond_best_cell[c]
        if info is None:
            continue
        S = info.get("subset_size")
        S_str = f"\n|S|={S}" if S is not None else ""
        ax.text(
            x, info["best_d"] + label_pad,
            f"{info['best_d']:+.1f}{S_str}\nα={info['alpha']:g}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([cond_labels.get(c, c) for c in conds],
                       rotation=15, ha="right")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(f"best cell {METRIC_PRETTY}")
    ax.set_title(title + "\n(error bars: 95% paired-t CI over n=40 prompts)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    if max(ys) > 0:
        ax.set_ylim(
            min(0, min(ys)) - max(yerr_lo + [0.0]) - 5,
            max(ys) * 1.30,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[best-bars] {out_path}")


def _format_test_md(label: str, res: dict, units: str = "%-points") -> list[str]:
    lo, hi = res["ci95"]
    return [
        f"### {label} (n = {res['n']})",
        "",
        f"- mean Δ = {res['mean_diff']:+.3f} {units}  "
        f"(95% CI [{lo:+.3f}, {hi:+.3f}])",
        f"- Cohen's d_z = {res['cohens_dz']:.3f}",
        f"- paired t (one-sided, H1: A-excl > baseline): "
        f"t = {res['t_stat']:.3f}, p = {_format_p(res['t_p'])}",
        f"- Wilcoxon signed-rank (one-sided): "
        f"W = {res['w_stat']:.1f}, p = {_format_p(res['w_p'])}",
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steering-root", type=Path,
                   default=Path("results/targeted_steering"))
    p.add_argument("--out", type=Path,
                   default=Path("results/figures/ablation_l13"))
    p.add_argument("--max-abs-k", type=int, default=10,
                   help="Cap |S| (absolute neurons steered) for the "
                        "absolute-k report. Cells with subset_size beyond "
                        "this are dropped from the absk plots/stats.")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cond_labels = {key: label for key, label, _ in CONDITIONS}
    per_cond_cells: dict[str, list[dict]] = {}
    per_cond_prompt_rows: dict[str, list[dict]] = {}
    missing: list[str] = []
    for key, label, dirname in CONDITIONS:
        d = args.steering_root / dirname
        if not d.is_dir():
            missing.append(f"{key} ({d})")
            continue
        per_cond_cells[key] = aggregate_model_dir(d)
        per_cond_prompt_rows[key] = _load_prompt_rows(d)
        print(f"[load] {key:>11s}: {len(per_cond_cells[key])} cells, "
              f"{len(per_cond_prompt_rows[key])} prompt rows  ({dirname})")
    if missing:
        print("[warn] missing condition dirs:")
        for m in missing:
            print(f"  - {m}")

    if "dfc-aexcl" not in per_cond_cells:
        raise SystemExit("DFC A-excl baseline missing; cannot run comparisons.")

    all_k = sorted({c["k_pct"] for cs in per_cond_cells.values() for c in cs})
    all_a = sorted({c["alpha"] for cs in per_cond_cells.values() for c in cs})
    full_grid = len(all_k) * len(all_a)
    print(f"k%: {all_k}  α: {all_a}  full grid = {full_grid}")

    # 1) Per-condition heatmap (Δ tool-corr vs clean) — re-keyed on |S|×α so
    # missing cells render as transparent gaps and the visible row labels are
    # absolute neuron counts. Each condition's |S| axis is just the |S|
    # values that appear in its sweep — no padding to a global union.
    for cond, rows in per_cond_cells.items():
        if not rows:
            continue
        S_vals_cond = sorted({int(r["subset_size"]) for r in rows})
        grid_S = np.full((len(S_vals_cond), len(all_a)), np.nan)
        S_idx = {S: i for i, S in enumerate(S_vals_cond)}
        a_idx = {a: j for j, a in enumerate(all_a)}
        for r in rows:
            S = int(r["subset_size"])
            a = float(r["alpha"])
            v = r.get(METRIC)
            if v is None or not np.isfinite(v):
                continue
            grid_S[S_idx[S], a_idx[a]] = float(v)
        heatmap(
            grid_S, S_vals_cond, all_a,
            title=f"{cond_labels[cond]}\nΔ Model-A tool vs clean (l13)",
            out_path=args.out / f"heatmap_{cond}_dA_tool_vs_clean.png",
            fmt="{:+.1f}",
            ylabel="|S| (absolute # of neurons steered)",
            yticklabel_fmt="{:d}",
        )

    # 2) Envelope plots, one line per condition (with per-cell 95% CIs).
    best_vs_alpha: dict[str, list[float]] = {}
    ci_vs_alpha: dict[str, tuple[list[float], list[float]]] = {}
    best_vs_k: dict[str, list[float]] = {}
    ci_vs_k: dict[str, tuple[list[float], list[float]]] = {}
    for cond, rows in per_cond_cells.items():
        ys, los, his = _best_with_ci_per_axis(rows, "alpha", all_a, METRIC)
        best_vs_alpha[cond] = ys
        ci_vs_alpha[cond] = (los, his)
        ys, los, his = _best_with_ci_per_axis(rows, "k_pct", all_k, METRIC)
        best_vs_k[cond] = ys
        ci_vs_k[cond] = (los, his)

    _plot_envelope(
        best_vs_alpha, cond_labels, all_a,
        x_label="α (steering coefficient)",
        title="Layer 13 ablation — max Δ tool-corr over |S|, per condition\n"
              "(error bars: 95% paired-t CI of the chosen cell, n=40)",
        out_path=args.out / "ablation_envelope_vs_alpha.png",
        per_cond_ci=ci_vs_alpha,
    )
    _plot_envelope(
        best_vs_k, cond_labels, all_k,
        x_label="k% (top-k feature subset)",
        title="Layer 13 ablation — max Δ tool-corr over α, per condition\n"
              "(error bars: 95% paired-t CI of the chosen cell, n=40)",
        out_path=args.out / "ablation_envelope_vs_kpct.png",
        per_cond_ci=ci_vs_k,
    )

    # 3) Best-cell-per-condition bar chart (with per-cell 95% CI).
    per_cond_best_cell: dict[str, dict | None] = {}
    for cond, rows in per_cond_cells.items():
        if not rows:
            per_cond_best_cell[cond] = None
            continue
        best = max(rows, key=lambda r: r[METRIC])
        per_cond_best_cell[cond] = {
            "best_d": float(best[METRIC]),
            "k_pct": float(best["k_pct"]),
            "alpha": float(best["alpha"]),
            "subset_size": int(best.get("subset_size", 0)),
            "n": int(best.get("n", 0)),
            "ci_lo": float(best.get("d_tool_ci_lo", float("nan"))),
            "ci_hi": float(best.get("d_tool_ci_hi", float("nan"))),
        }
    _plot_best_cell_bars(
        per_cond_best_cell, cond_labels,
        args.out / "ablation_best_cell_bars.png",
    )

    # 4) Paired stat tests vs DFC-A-excl baseline (per cell).
    base_rows = per_cond_cells["dfc-aexcl"]
    base_by_cell = {(r["k_pct"], r["alpha"]): r[METRIC] for r in base_rows}

    md = ["# Layer-13 ablation — does targeted A-excl steering beat the baselines?",
          "",
          "Conditions:",
          ""]
    for key, label, dirname in CONDITIONS:
        marker = "✓" if key in per_cond_cells else "✗ MISSING"
        md.append(f"- `{key}` — {label} (`{dirname}`)  {marker}")
    md.append("")
    md.append(
        "Pairings: per (k%, α) cell, A-excl Δ tool-corr − baseline Δ tool-corr. "
        "One-sided H1: A-excl > baseline."
    )
    md.append("")

    md.append("## Best cell per condition")
    md.append("")
    md.append("| condition | best Δ (%) | 95% CI | |S| | α | n_prompts |")
    md.append("|-----------|-----------:|--------|----:|--:|----------:|")
    for key, label, _ in CONDITIONS:
        info = per_cond_best_cell.get(key)
        if info is None:
            md.append(f"| {label} | — | — | — | — | — |")
        else:
            lo = info.get("ci_lo", float("nan"))
            hi = info.get("ci_hi", float("nan"))
            ci_str = (f"[{lo:+.1f}, {hi:+.1f}]"
                      if np.isfinite(lo) and np.isfinite(hi) else "—")
            md.append(
                f"| {label} | {info['best_d']:+.1f} | {ci_str} | "
                f"{info.get('subset_size', '—')} | {info['alpha']:g} | "
                f"{info['n']} |"
            )
    md.append("")

    md.append("## Paired tests (cell-level)")
    md.append("")
    for key, label, _ in CONDITIONS:
        if key == "dfc-aexcl" or key not in per_cond_cells:
            continue
        other_by_cell = {
            (r["k_pct"], r["alpha"]): r[METRIC] for r in per_cond_cells[key]
        }
        diffs = []
        for cell, base_v in base_by_cell.items():
            if cell in other_by_cell:
                diffs.append(base_v - other_by_cell[cell])
        diffs = np.asarray(diffs, dtype=np.float64)
        res = _paired_test(diffs)
        md.extend(_format_test_md(f"DFC-A-excl vs {label}", res))
        md.append("")

    # 5) Prompt-level paired tests (more powerful but cells are correlated).
    md.append("## Prompt-level paired tests")
    md.append("")
    md.append(
        "For every (k%, α, prompt_index) row in both conditions, "
        "compute steered-tool − clean-tool ∈ {−1, 0, +1}, then test "
        "(A-excl Δ) − (baseline Δ) > 0."
    )
    md.append("")
    base_prompt = {
        (r["k_pct"], r["alpha"], r["prompt_index"]): (
            int(bool(r["steered_score"]["tool_correctness"]))
            - int(bool(r["clean_score"]["tool_correctness"]))
        )
        for r in per_cond_prompt_rows.get("dfc-aexcl", [])
        if "steered_score" in r and "clean_score" in r
    }
    for key, label, _ in CONDITIONS:
        if key == "dfc-aexcl" or key not in per_cond_prompt_rows:
            continue
        other_prompt = {
            (r["k_pct"], r["alpha"], r["prompt_index"]): (
                int(bool(r["steered_score"]["tool_correctness"]))
                - int(bool(r["clean_score"]["tool_correctness"]))
            )
            for r in per_cond_prompt_rows[key]
            if "steered_score" in r and "clean_score" in r
        }
        keys_shared = set(base_prompt) & set(other_prompt)
        diffs = np.asarray(
            [(base_prompt[k] - other_prompt[k]) * 100.0 for k in keys_shared],
            dtype=np.float64,
        )
        if diffs.size == 0:
            continue
        res = _paired_test(diffs)
        md.extend(_format_test_md(
            f"Prompt-level DFC-A-excl vs {label}", res, units="%-points",
        ))
        md.append("")

    md_path = args.out / "ablation_significance.md"
    md_path.write_text("\n".join(md))
    print(f"[stats] {md_path}")

    # ── Absolute-k report (cap |S| ≤ args.max_abs_k) ──────────────────────
    max_S = int(args.max_abs_k)
    print(f"\n=== Absolute-k report (|S| ≤ {max_S}) ===")

    per_cond_cells_capped: dict[str, list[dict]] = {
        cond: [c for c in cells if int(c.get("subset_size", 0)) <= max_S]
        for cond, cells in per_cond_cells.items()
    }
    for cond, cells in per_cond_cells_capped.items():
        kept = sorted({(int(c["subset_size"]), float(c["alpha"])) for c in cells})
        print(f"  {cond:>11s}: kept {len(cells)}/30 cells; "
              f"|S| values present: "
              f"{sorted({int(c['subset_size']) for c in cells})}")

    # Envelope vs absolute |S| (max over α per |S|), tracking CI of the cell
    # that achieved the max.
    all_S = sorted({
        int(c["subset_size"])
        for cells in per_cond_cells_capped.values() for c in cells
    })
    best_vs_S: dict[str, list[float]] = {}
    ci_vs_S: dict[str, tuple[list[float], list[float]]] = {}
    for cond, cells in per_cond_cells_capped.items():
        bucket: dict[int, dict] = {}
        for c in cells:
            v = c.get(METRIC)
            if v is None or not np.isfinite(v):
                continue
            S = int(c["subset_size"])
            cur = bucket.get(S)
            if cur is None or float(v) > cur["v"]:
                bucket[S] = {
                    "v": float(v),
                    "lo": float(c.get("d_tool_ci_lo", float("nan"))),
                    "hi": float(c.get("d_tool_ci_hi", float("nan"))),
                }
        best_vs_S[cond] = [
            bucket[s]["v"] if s in bucket else float("nan") for s in all_S
        ]
        ci_vs_S[cond] = (
            [bucket[s]["lo"] if s in bucket else float("nan") for s in all_S],
            [bucket[s]["hi"] if s in bucket else float("nan") for s in all_S],
        )

    # Reuse _plot_envelope but with absolute-k axis. Force linear (no log) by
    # scaling — easiest is to monkey-patch via a small helper that does the
    # same thing minus xscale("log"). Re-implement inline to keep the change
    # local.
    def _plot_envelope_linear(
        per_cond_best, cond_labels_, x_values, x_label, title, out_path,
        per_cond_ci=None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        xs = np.asarray(x_values, dtype=np.float64)
        for cond, ys_list in per_cond_best.items():
            ys = np.asarray(ys_list, dtype=np.float64)
            finite = np.isfinite(ys)
            if not finite.any():
                continue
            color = CONDITION_COLORS.get(cond, "grey")
            if per_cond_ci is not None and cond in per_cond_ci:
                los_list, his_list = per_cond_ci[cond]
                los = np.asarray(los_list, dtype=np.float64)
                his = np.asarray(his_list, dtype=np.float64)
                err_finite = finite & np.isfinite(los) & np.isfinite(his)
                if err_finite.any():
                    yerr_lo = ys[err_finite] - los[err_finite]
                    yerr_hi = his[err_finite] - ys[err_finite]
                    ax.errorbar(
                        xs[err_finite], ys[err_finite],
                        yerr=[yerr_lo, yerr_hi],
                        fmt="o", linewidth=2.0, markersize=7,
                        color=color, ecolor=color, capsize=3, alpha=0.95,
                        label=cond_labels_.get(cond, cond),
                    )
                    ax.plot(
                        xs[finite], ys[finite],
                        color=color, linewidth=2.0, alpha=0.7,
                    )
                    continue
            ax.plot(
                xs[finite], ys[finite],
                marker="o", linewidth=2.0, markersize=7,
                color=color,
                label=cond_labels_.get(cond, cond),
            )
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xticks(x_values)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"max of {METRIC_PRETTY}")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[envelope-absk] {out_path}")

    _plot_envelope_linear(
        best_vs_S, cond_labels, all_S,
        x_label=f"|S| (absolute # of neurons steered, cap = {max_S})",
        title=(f"Layer 13 ablation — max Δ over α, by absolute |S| (≤ {max_S})\n"
               f"(error bars: 95% paired-t CI of the chosen cell, n=40)"),
        out_path=args.out / "ablation_envelope_vs_absk.png",
        per_cond_ci=ci_vs_S,
    )

    # Envelope vs α with cells filtered by |S| ≤ max_S (track CI of best cell).
    best_vs_alpha_capped: dict[str, list[float]] = {}
    ci_vs_alpha_capped: dict[str, tuple[list[float], list[float]]] = {}
    for cond, cells in per_cond_cells_capped.items():
        bucket: dict[float, dict] = {}
        for c in cells:
            v = c.get(METRIC)
            if v is None or not np.isfinite(v):
                continue
            a = float(c["alpha"])
            cur = bucket.get(a)
            if cur is None or float(v) > cur["v"]:
                bucket[a] = {
                    "v": float(v),
                    "lo": float(c.get("d_tool_ci_lo", float("nan"))),
                    "hi": float(c.get("d_tool_ci_hi", float("nan"))),
                }
        best_vs_alpha_capped[cond] = [
            bucket[a]["v"] if a in bucket else float("nan") for a in all_a
        ]
        ci_vs_alpha_capped[cond] = (
            [bucket[a]["lo"] if a in bucket else float("nan") for a in all_a],
            [bucket[a]["hi"] if a in bucket else float("nan") for a in all_a],
        )
    _plot_envelope(
        best_vs_alpha_capped, cond_labels, all_a,
        x_label="α (steering coefficient)",
        title=(f"Layer 13 ablation — max Δ over |S|≤{max_S}, vs α\n"
               f"(error bars: 95% paired-t CI of the chosen cell, n=40)"),
        out_path=args.out / "ablation_envelope_vs_alpha_absk.png",
        per_cond_ci=ci_vs_alpha_capped,
    )

    # Best cell among |S| ≤ max_S (with CI).
    best_cell_capped: dict[str, dict | None] = {}
    for cond, cells in per_cond_cells_capped.items():
        if not cells:
            best_cell_capped[cond] = None
            continue
        best = max(cells, key=lambda r: r[METRIC])
        best_cell_capped[cond] = {
            "best_d": float(best[METRIC]),
            "k_pct":  float(best["k_pct"]),
            "alpha":  float(best["alpha"]),
            "n":      int(best.get("n", 0)),
            "subset_size": int(best.get("subset_size", 0)),
            "ci_lo":  float(best.get("d_tool_ci_lo", float("nan"))),
            "ci_hi":  float(best.get("d_tool_ci_hi", float("nan"))),
        }
    _plot_best_cell_bars(
        best_cell_capped, cond_labels,
        args.out / "ablation_best_cell_bars_absk.png",
        title=(f"Layer 13: best cell at |S| ≤ {max_S} "
               f"(absolute # of neurons steered)"),
    )

    # Stats — pair by (k%, α) but only where max(|S|_A, |S|_other) ≤ max_S.
    base_with_S = {
        (r["k_pct"], r["alpha"]): (r[METRIC], int(r.get("subset_size", 0)))
        for r in per_cond_cells["dfc-aexcl"]
    }
    md_absk: list[str] = []
    md_absk.append(f"# Layer-13 ablation — absolute-|S| report (|S| ≤ {max_S})")
    md_absk.append("")
    md_absk.append(
        f"Same data as `ablation_significance.md`, re-keyed by absolute "
        f"number of neurons steered (`subset_size`) and capped at "
        f"|S| ≤ {max_S}. Only existing cells are reported — no "
        f"interpolation. Pairing across conditions still uses the (k%, α) "
        f"cell key, with the additional filter that BOTH conditions' |S| "
        f"≤ {max_S} for that cell."
    )
    md_absk.append("")

    md_absk.append("## |S| values present per condition (after cap)")
    md_absk.append("")
    md_absk.append("| condition | |S| values present | n cells kept |")
    md_absk.append("|-----------|---------------------|-------------:|")
    for key, label, _ in CONDITIONS:
        cells = per_cond_cells_capped.get(key, [])
        sizes = sorted({int(c["subset_size"]) for c in cells})
        md_absk.append(f"| {label} | {sizes} | {len(cells)} |")
    md_absk.append("")

    md_absk.append("## Best cell at |S| ≤ " + str(max_S))
    md_absk.append("")
    md_absk.append("| condition | best Δ (%) | 95% CI | |S| | α | n_prompts |")
    md_absk.append("|-----------|-----------:|--------|----:|--:|----------:|")
    for key, label, _ in CONDITIONS:
        info = best_cell_capped.get(key)
        if info is None:
            md_absk.append(f"| {label} | — | — | — | — | — |")
        else:
            lo = info.get("ci_lo", float("nan"))
            hi = info.get("ci_hi", float("nan"))
            ci_str = (f"[{lo:+.1f}, {hi:+.1f}]"
                      if np.isfinite(lo) and np.isfinite(hi) else "—")
            md_absk.append(
                f"| {label} | {info['best_d']:+.1f} | {ci_str} | "
                f"{info['subset_size']} | {info['alpha']:g} | "
                f"{info['n']} |"
            )
    md_absk.append("")

    md_absk.append("## Per-(|S|, α) cell table per condition")
    md_absk.append("")
    md_absk.append("Δ tool-corr (steered − clean), %, with 95% paired-t CI "
                   f"over n=40 prompts. Only |S| ≤ {max_S}.")
    md_absk.append("")
    for key, label, _ in CONDITIONS:
        cells = per_cond_cells_capped.get(key, [])
        if not cells:
            continue
        md_absk.append(f"### {label}")
        md_absk.append("")
        md_absk.append("| |S| | α | Δ (%) | 95% CI |")
        md_absk.append("|----:|--:|------:|--------|")
        for c in sorted(cells, key=lambda r: (int(r["subset_size"]), float(r["alpha"]))):
            lo = float(c.get("d_tool_ci_lo", float("nan")))
            hi = float(c.get("d_tool_ci_hi", float("nan")))
            ci_str = (f"[{lo:+.1f}, {hi:+.1f}]"
                      if np.isfinite(lo) and np.isfinite(hi) else "—")
            md_absk.append(
                f"| {int(c['subset_size'])} | {float(c['alpha']):g} | "
                f"{float(c[METRIC]):+.1f} | {ci_str} |"
            )
        md_absk.append("")

    md_absk.append(f"## Paired tests at |S| ≤ {max_S}")
    md_absk.append("")
    md_absk.append(
        "Pair by (k%, α) cell; include only if BOTH conditions' "
        f"|S| ≤ {max_S} for that cell. One-sided H1: A-excl > baseline."
    )
    md_absk.append("")
    for key, label, _ in CONDITIONS:
        if key == "dfc-aexcl" or key not in per_cond_cells:
            continue
        other_with_S = {
            (r["k_pct"], r["alpha"]): (r[METRIC], int(r.get("subset_size", 0)))
            for r in per_cond_cells[key]
        }
        diffs = []
        kept_cells = []
        for cell, (base_v, base_S) in base_with_S.items():
            if cell not in other_with_S:
                continue
            other_v, other_S = other_with_S[cell]
            if max(base_S, other_S) > max_S:
                continue
            diffs.append(base_v - other_v)
            kept_cells.append((cell, base_S, other_S))
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        md_absk.append(f"### DFC-A-excl vs {label} ({len(kept_cells)} cells kept)")
        md_absk.append("")
        if diffs_arr.size == 0:
            md_absk.append("(no cells left after the |S| cap)")
            md_absk.append("")
            continue
        res = _paired_test(diffs_arr)
        md_absk.extend(_format_test_md(
            f"DFC-A-excl vs {label} (|S|≤{max_S}, paired by k%, α)",
            res,
        ))
        md_absk.append("")

    md_absk_path = args.out / "ablation_significance_absk.md"
    md_absk_path.write_text("\n".join(md_absk))
    print(f"[stats-absk] {md_absk_path}")

    # ── Saturation curve: full |S| range, log-x — DFC saturates, CC climbs ──
    print("\n=== Saturation curve (all |S|, log axis) ===")
    sat_per_S: dict[str, dict[int, dict]] = {}
    for cond, cells in per_cond_cells.items():
        bucket: dict[int, dict] = {}
        for c in cells:
            v = c.get(METRIC)
            S = int(c.get("subset_size", 0))
            if v is None or not np.isfinite(v) or S < 1:
                continue
            cur = bucket.get(S)
            if cur is None or float(v) > cur["v"]:
                bucket[S] = {
                    "v": float(v),
                    "lo": float(c.get("d_tool_ci_lo", float("nan"))),
                    "hi": float(c.get("d_tool_ci_hi", float("nan"))),
                }
        sat_per_S[cond] = bucket

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=150)
    for cond in [k for k, _, _ in CONDITIONS]:
        if cond not in sat_per_S or not sat_per_S[cond]:
            continue
        Ss = sorted(sat_per_S[cond])
        ys = [sat_per_S[cond][S]["v"] for S in Ss]
        los = [sat_per_S[cond][S]["lo"] for S in Ss]
        his = [sat_per_S[cond][S]["hi"] for S in Ss]
        color = CONDITION_COLORS.get(cond, "grey")
        ys_arr = np.asarray(ys, dtype=np.float64)
        los_arr = np.asarray(los, dtype=np.float64)
        his_arr = np.asarray(his, dtype=np.float64)
        good = np.isfinite(los_arr) & np.isfinite(his_arr)
        if good.any():
            yerr_lo = np.where(good, ys_arr - los_arr, 0.0)
            yerr_hi = np.where(good, his_arr - ys_arr, 0.0)
            ax.errorbar(Ss, ys, yerr=[yerr_lo, yerr_hi],
                        fmt="o", linewidth=2.0, markersize=7,
                        color=color, ecolor=color, capsize=3, alpha=0.95,
                        label=cond_labels.get(cond, cond))
            ax.plot(Ss, ys, color=color, linewidth=2.0, alpha=0.7)
        else:
            ax.plot(Ss, ys, marker="o", linewidth=2.0, markersize=7,
                    color=color, label=cond_labels.get(cond, cond))
        # Annotate the peak.
        peak_S = Ss[int(np.argmax(ys))]
        peak_v = max(ys)
        ax.annotate(f"peak +{peak_v:.0f}\n@|S|={peak_S}",
                    (peak_S, peak_v), xytext=(8, 8),
                    textcoords="offset points", fontsize=8,
                    color=color)

    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("|S| (absolute # of neurons steered, log scale)")
    ax.set_ylabel(f"max over α of {METRIC_PRETTY}")
    ax.set_title("Layer 13 saturation curve — DFC A-excl plateaus at 1 neuron; "
                 "CC climbs slowly\n(error bars: 95% paired-t CI of the "
                 "best-α cell at each |S|, n=40)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    sat_path = args.out / "ablation_saturation_curve.png"
    fig.savefig(sat_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saturation] {sat_path}")

    # ── Aggregate (|S|, α) tradeoff heatmap ─────────────────────────────
    # Pool live conditions (drop B-excl which is structurally dead) and at
    # each (|S|, α) coordinate take both the mean and the max Δ across
    # conditions. Mean shows the typical behaviour; max shows what's
    # achievable. Empty (|S|, α) coordinates render as transparent gaps.
    print("\n=== Aggregate (|S|, α) tradeoff ===")
    LIVE_CONDS = [k for k in ("dfc-aexcl", "dfc-shared", "cc")
                   if k in per_cond_cells]
    pool_cells: list[dict] = []
    for cond in LIVE_CONDS:
        for r in per_cond_cells[cond]:
            v = r.get(METRIC)
            if v is None or not np.isfinite(v):
                continue
            pool_cells.append({
                "S": int(r["subset_size"]),
                "alpha": float(r["alpha"]),
                "v": float(v),
                "ci_lo": float(r.get("d_tool_ci_lo", float("nan"))),
                "ci_hi": float(r.get("d_tool_ci_hi", float("nan"))),
                "cond": cond,
            })

    S_axis = sorted({c["S"] for c in pool_cells if c["S"] >= 1})
    a_axis = sorted({c["alpha"] for c in pool_cells})
    S_idx = {S: i for i, S in enumerate(S_axis)}
    a_idx = {a: j for j, a in enumerate(a_axis)}

    grid_mean = np.full((len(S_axis), len(a_axis)), np.nan)
    grid_max = np.full((len(S_axis), len(a_axis)), np.nan)
    grid_count = np.zeros((len(S_axis), len(a_axis)), dtype=int)
    for c in pool_cells:
        i, j = S_idx[c["S"]], a_idx[c["alpha"]]
        prev = grid_mean[i, j]
        if np.isnan(prev):
            grid_mean[i, j] = c["v"]
            grid_max[i, j] = c["v"]
        else:
            n = grid_count[i, j]
            grid_mean[i, j] = (prev * n + c["v"]) / (n + 1)
            if c["v"] > grid_max[i, j]:
                grid_max[i, j] = c["v"]
        grid_count[i, j] += 1

    heatmap(
        grid_mean, S_axis, a_axis,
        title=("Aggregate tradeoff: mean Δ across "
               f"{{{', '.join(LIVE_CONDS)}}}\n"
               f"(empty cells = no condition swept that |S|×α)"),
        out_path=args.out / "ablation_tradeoff_mean_S_alpha.png",
        fmt="{:+.1f}",
        ylabel="|S| (absolute # of neurons steered)",
        yticklabel_fmt="{:d}",
    )
    heatmap(
        grid_max, S_axis, a_axis,
        title=("Aggregate tradeoff: best achievable Δ across "
               f"{{{', '.join(LIVE_CONDS)}}}\n(over A-excl, shared, CC)"),
        out_path=args.out / "ablation_tradeoff_max_S_alpha.png",
        fmt="{:+.1f}",
        ylabel="|S| (absolute # of neurons steered)",
        yticklabel_fmt="{:d}",
    )

    # 1D marginals: best Δ vs |S| (max over α) and vs α (max over |S|).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)

    # vs |S|
    best_per_S: list[float] = []
    best_alpha_at_S: list[float] = []
    for S in S_axis:
        vs = [c["v"] for c in pool_cells if c["S"] == S]
        alphas = [c["alpha"] for c in pool_cells if c["S"] == S]
        if vs:
            best_per_S.append(max(vs))
            best_alpha_at_S.append(alphas[int(np.argmax(vs))])
        else:
            best_per_S.append(float("nan"))
            best_alpha_at_S.append(float("nan"))
    ax1.plot(S_axis, best_per_S, marker="o", color="#1f77b4", linewidth=2)
    ax1.set_xscale("log")
    for x, y, a in zip(S_axis, best_per_S, best_alpha_at_S):
        if np.isfinite(y) and np.isfinite(a):
            ax1.annotate(f"α={a:g}", (x, y), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=7)
    ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax1.set_xlabel("|S| (log scale)")
    ax1.set_ylabel(f"max over α & condition of {METRIC_PRETTY}")
    ax1.set_title("Tradeoff vs |S| — best α annotated per |S|")
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)

    # vs α
    best_per_a: list[float] = []
    best_S_at_a: list[float] = []
    for a in a_axis:
        vs = [c["v"] for c in pool_cells if c["alpha"] == a]
        Ss = [c["S"] for c in pool_cells if c["alpha"] == a]
        if vs:
            best_per_a.append(max(vs))
            best_S_at_a.append(Ss[int(np.argmax(vs))])
        else:
            best_per_a.append(float("nan"))
            best_S_at_a.append(float("nan"))
    ax2.plot(a_axis, best_per_a, marker="o", color="#d62728", linewidth=2)
    ax2.set_xscale("log")
    for x, y, S in zip(a_axis, best_per_a, best_S_at_a):
        if np.isfinite(y) and np.isfinite(S):
            ax2.annotate(f"|S|={int(S)}", (x, y), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=7)
    ax2.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax2.set_xlabel("α (log scale)")
    ax2.set_ylabel(f"max over |S| & condition of {METRIC_PRETTY}")
    ax2.set_title("Tradeoff vs α — best |S| annotated per α")
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.suptitle(f"Aggregate |S|–α balance across {{{', '.join(LIVE_CONDS)}}}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out / "ablation_tradeoff_marginals.png",
                bbox_inches="tight")
    plt.close(fig)
    print(f"[tradeoff] {args.out / 'ablation_tradeoff_mean_S_alpha.png'}")
    print(f"[tradeoff] {args.out / 'ablation_tradeoff_max_S_alpha.png'}")
    print(f"[tradeoff] {args.out / 'ablation_tradeoff_marginals.png'}")

    # Identify the "balance point": (|S|, α) with peak max-Δ, and the
    # smallest-|S| / smallest-α coordinate that achieves within 5pp of peak.
    finite_max = np.nanmax(grid_max) if np.isfinite(grid_max).any() else float("nan")
    if np.isfinite(finite_max):
        peak_i, peak_j = np.unravel_index(np.nanargmax(grid_max), grid_max.shape)
        within_5 = np.where(grid_max >= finite_max - 5)
        # Find the "minimum cost" cell within 5pp of peak (smallest |S|, then α).
        candidates = sorted(
            ((S_axis[i], a_axis[j], grid_max[i, j])
             for i, j in zip(*within_5)),
            key=lambda x: (x[0], x[1]),
        )
        cheapest = candidates[0] if candidates else None
        balance_md = [
            "## Aggregate |S|–α balance",
            "",
            f"Pooled across {{{', '.join(LIVE_CONDS)}}} at l13.",
            "",
            f"- **Peak achievable Δ** = {finite_max:+.1f}% at "
            f"|S|={S_axis[peak_i]}, α={a_axis[peak_j]:g}",
        ]
        if cheapest is not None:
            balance_md.append(
                f"- **Cheapest balance point** (within 5pp of peak): "
                f"|S|={cheapest[0]}, α={cheapest[1]:g} → Δ = {cheapest[2]:+.1f}%"
            )
        balance_md.append("")
        balance_md.append(
            "Reading: high α with too few neurons under-steers; "
            "high |S| with low α under-steers; the balance point lies in the "
            "low-|S|, mid-α region."
        )
        (args.out / "ablation_tradeoff_summary.md").write_text(
            "\n".join(balance_md)
        )
        print(f"[tradeoff-md] {args.out / 'ablation_tradeoff_summary.md'}")

    print(f"[done] outputs in {args.out}")


if __name__ == "__main__":
    main()
