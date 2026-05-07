"""
build_steering_figures.py — figures for the targeted-steering sweep + paper
replications.

Per model produces (under ``--out``):
  - heatmap_<short>_dA_tool_vs_clean.png
  - heatmap_<short>_dA_tool_vs_recon.png
  - heatmap_<short>_dA_overall_vs_clean.png
  - boxplot_<short>_dA_vs_kpct.png    (one box per k%, points = per-cell Δs across α)
  - boxplot_<short>_dA_vs_alpha.png   (one box per α,  points = per-cell Δs across k%)

Cross-model:
  - compare_models_best_cell.png

Paper replications (from --paper-jsonl):
  - paper_repro_fig1_modelA_pre_post_box.png
  - paper_repro_fig4_topk_vs_dA.png
  - paper_repro_fig7_uniform_scale.png  (overlays best targeted cells per model)

Reads per-model JSONL written by run_steering_eval.py from
``<steering-root>/<short>/k*_a*.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as _scipy_stats


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate per-cell JSONLs → tidy rows
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_model_dir(model_dir: Path) -> list[dict]:
    """One row per (k_pct, alpha) with mean clean / recon / steered scores.

    Also accumulates per-prompt paired diffs for tool-correctness so each row
    carries a 95% paired-t CI on `d_steered_clean_tool` (in %-points):
        - d_tool_ci_lo / d_tool_ci_hi : CI bounds (% points)
        - d_tool_sem                  : standard error of the mean (% points)
    """
    cells: dict[tuple[float, float], dict] = {}
    for p in sorted(model_dir.glob("k*_a*.jsonl")):
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
                key = (float(r["k_pct"]), float(r["alpha"]))
                c = cells.setdefault(key, {
                    "n": 0,
                    "subset_size": int(r.get("subset_size", 0)),
                    "clean_tool": 0,    "recon_tool": 0,    "steered_tool": 0,
                    "clean_format": 0,  "recon_format": 0,  "steered_format": 0,
                    "clean_overall": 0.0, "recon_overall": 0.0, "steered_overall": 0.0,
                    "diff_tool": [],
                    "diff_overall": [],
                })
                c["n"] += 1
                for tag in ("clean", "recon", "steered"):
                    s = r[f"{tag}_score"]
                    c[f"{tag}_tool"]    += int(bool(s["tool_correctness"]))
                    c[f"{tag}_format"]  += int(bool(s["format_accuracy"]))
                    c[f"{tag}_overall"] += float(s["overall_score"])
                # Paired per-prompt diffs for CI.
                steer_t = int(bool(r["steered_score"]["tool_correctness"]))
                clean_t = int(bool(r["clean_score"]["tool_correctness"]))
                c["diff_tool"].append(steer_t - clean_t)
                c["diff_overall"].append(
                    float(r["steered_score"]["overall_score"])
                    - float(r["clean_score"]["overall_score"])
                )
    rows = []
    for (k_pct, alpha), c in sorted(cells.items()):
        n = max(c["n"], 1)
        row = {"k_pct": k_pct, "alpha": alpha, "n": c["n"], "subset_size": c["subset_size"]}
        for tag in ("clean", "recon", "steered"):
            row[f"{tag}_tool_pct"]    = 100.0 * c[f"{tag}_tool"]    / n
            row[f"{tag}_format_pct"]  = 100.0 * c[f"{tag}_format"]  / n
            row[f"{tag}_overall_avg"] = c[f"{tag}_overall"] / n
        row["d_steered_clean_tool"]   = row["steered_tool_pct"]    - row["clean_tool_pct"]
        row["d_steered_recon_tool"]   = row["steered_tool_pct"]    - row["recon_tool_pct"]
        row["d_steered_clean_overall"] = row["steered_overall_avg"] - row["clean_overall_avg"]

        # 95% paired-t CI on the per-prompt tool diff (× 100 for %-points).
        diffs = np.asarray(c["diff_tool"], dtype=np.float64)
        if diffs.size >= 2:
            mean_pct = float(diffs.mean()) * 100.0
            sem_pct = float(diffs.std(ddof=1) / math.sqrt(diffs.size)) * 100.0
            t_crit = float(_scipy_stats.t.ppf(0.975, df=diffs.size - 1))
            row["d_tool_sem"]   = sem_pct
            row["d_tool_ci_lo"] = mean_pct - t_crit * sem_pct
            row["d_tool_ci_hi"] = mean_pct + t_crit * sem_pct
        else:
            row["d_tool_sem"]   = float("nan")
            row["d_tool_ci_lo"] = float("nan")
            row["d_tool_ci_hi"] = float("nan")

        # Same for overall_score (continuous, but same form).
        diffs_o = np.asarray(c["diff_overall"], dtype=np.float64)
        if diffs_o.size >= 2:
            sem_o = float(diffs_o.std(ddof=1) / math.sqrt(diffs_o.size))
            t_crit_o = float(_scipy_stats.t.ppf(0.975, df=diffs_o.size - 1))
            mean_o = float(diffs_o.mean())
            row["d_overall_sem"]   = sem_o
            row["d_overall_ci_lo"] = mean_o - t_crit_o * sem_o
            row["d_overall_ci_hi"] = mean_o + t_crit_o * sem_o
        else:
            row["d_overall_sem"]   = float("nan")
            row["d_overall_ci_lo"] = float("nan")
            row["d_overall_ci_hi"] = float("nan")

        rows.append(row)
    return rows


def grid_from_rows(
    rows: list[dict],
    metric: str,
    k_pcts: list[float],
    alphas: list[float],
) -> np.ndarray:
    """Lay rows out as a (k, α) grid; missing cells = NaN."""
    out = np.full((len(k_pcts), len(alphas)), np.nan, dtype=np.float64)
    by_key = {(r["k_pct"], r["alpha"]): r for r in rows}
    for i, k in enumerate(k_pcts):
        for j, a in enumerate(alphas):
            r = by_key.get((k, a))
            if r is not None:
                out[i, j] = float(r[metric])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Heatmaps + line plots
# ──────────────────────────────────────────────────────────────────────────────

def heatmap(
    grid: np.ndarray,
    k_pcts: list[float],
    alphas: list[float],
    title: str,
    out_path: Path,
    cmap: str = "RdBu_r",
    fmt: str = "{:+.1f}",
    centre_zero: bool = True,
    ylabel: str = "k% of A-excl ranked subset",
    yticklabel_fmt: str = "{:g}%",
) -> None:
    """Heatmap with NaN cells rendered as a clear gap (no fill, no text)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    if centre_zero and np.isfinite(grid).any():
        m = float(np.nanmax(np.abs(grid)))
        vmin, vmax = -m, m
    else:
        vmin, vmax = None, None
    # Mask NaN cells so they render as background (no fill).
    masked = np.ma.masked_invalid(grid)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="white", alpha=0.0)
    ax.set_facecolor("#f0f0f0")
    im = ax.imshow(masked, aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_yticks(range(len(k_pcts)))
    ax.set_yticklabels([yticklabel_fmt.format(k) for k in k_pcts])
    ax.set_xlabel("alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Mark missing cells with a light dashed border + diagonal hatch so the
    # gap is unambiguous (vs. the row/col scaffold).
    from matplotlib.patches import Rectangle
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > (vmax or 1) * 0.55 else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color=color, fontsize=8)
            else:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#ffffff", edgecolor="#cccccc",
                    hatch="//", linewidth=0.4, alpha=0.65, zorder=2,
                ))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Δ")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[heatmap] {out_path}")


def _boxplot_grouped(
    rows: list[dict],
    metric: str,
    x_axis: str,            # "k_pct" or "alpha"
    x_values: list[float],  # tick order on x-axis (the grouping axis)
    title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    """Generic helper. One box per x_value; box's points = per-cell metric values
    across the *other* axis. Mean overlaid as a red line+marker.
    """
    bucket: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        bucket[float(r[x_axis])].append(float(v))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    positions = list(range(1, len(x_values) + 1))
    data = [np.asarray(bucket.get(x, []), dtype=np.float64) for x in x_values]
    counts = [d.size for d in data]
    means = [float(d.mean()) if d.size else np.nan for d in data]

    bp = ax.boxplot(
        [d if d.size else [np.nan] for d in data],
        positions=positions, widths=0.55,
        manage_ticks=False, showfliers=True, patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#5b9bd5"); patch.set_alpha(0.4)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.5)

    # Scatter of underlying points (jittered) so single-point boxes are visible.
    rng = np.random.RandomState(0)
    for pos, d in zip(positions, data):
        if not d.size:
            continue
        jitter = (rng.rand(d.size) - 0.5) * 0.18
        ax.scatter(np.full(d.size, pos) + jitter, d,
                   s=14, alpha=0.55, c="#1f77b4", edgecolors="none", zorder=3)

    ax.plot(positions, means, color="crimson", marker="o", linestyle="-",
            label="mean", zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{v:g}" + (f"\n(n={n})" if n else "") for v, n in zip(x_values, counts)],
        fontsize=9,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[box] {out_path}")


def boxplot_vs_kpct(
    rows: list[dict],
    metric: str,
    k_pcts: list[float],
    alphas: list[float],   # kept in signature for call-site compatibility
    title: str,
    out_path: Path,
    ylabel: str,
) -> None:
    _boxplot_grouped(
        rows, metric, x_axis="k_pct", x_values=k_pcts,
        title=title, out_path=out_path,
        xlabel="k% (subset of A-excl) — box = per-cell Δs across α",
        ylabel=ylabel,
    )


def boxplot_vs_alpha(
    rows: list[dict],
    metric: str,
    k_pcts: list[float],   # kept in signature for call-site compatibility
    alphas: list[float],
    title: str,
    out_path: Path,
    ylabel: str,
) -> None:
    _boxplot_grouped(
        rows, metric, x_axis="alpha", x_values=alphas,
        title=title, out_path=out_path,
        xlabel="α (steering coefficient) — box = per-cell Δs across k%",
        ylabel=ylabel,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Paper replications (from results_full jsonl)
# ──────────────────────────────────────────────────────────────────────────────

def paper_fig1_box(paper_rows: list[dict], out_path: Path) -> None:
    """Pre vs post tool-correctness on Model A across the sweep."""
    pre = [r["tool_correctness_A_pre"] for r in paper_rows if "tool_correctness_A_pre" in r]
    post = [r["tool_correctness_A_post"] for r in paper_rows if "tool_correctness_A_post" in r]
    if not pre or not post:
        print("[fig1-paper] missing tool_correctness fields; skipping")
        return
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    ax.boxplot([pre, post], tick_labels=["pre-recon", "post-recon"])
    for x_pos, vals in zip([1, 2], [pre, post]):
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.15
        ax.scatter(np.full(len(vals), x_pos) + jitter, vals,
                   alpha=0.4, s=14, c="#1f77b4")
    ax.set_ylabel("Tool correctness (%)")
    ax.set_title("Model A — paper replication (Fig 1, left)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig1-paper] {out_path}")


def paper_fig4_topk(paper_rows: list[dict], out_path: Path) -> None:
    """Δ Model-A tool-correctness vs top-k, split by architecture."""
    by_arch_topk: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in paper_rows:
        if "tool_correctness_A_pre" not in r or "tool_correctness_A_post" not in r:
            continue
        hp = r.get("hyperparameters", {})
        arch = hp.get("architecture", "?")
        k = int(hp.get("k", 0))
        by_arch_topk[(arch, k)].append(
            r["tool_correctness_A_post"] - r["tool_correctness_A_pre"]
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    archs = sorted({a for a, _ in by_arch_topk})
    ks = sorted({k for _, k in by_arch_topk})
    colors = {"CrossCoder": "#d62728", "DFC": "#1f77b4"}
    for arch in archs:
        means, sems = [], []
        for k in ks:
            vals = by_arch_topk.get((arch, k), [])
            if vals:
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals) / np.sqrt(len(vals))))
            else:
                means.append(np.nan); sems.append(0.0)
        ax.errorbar(ks, means, yerr=sems, marker="o",
                    label=arch, color=colors.get(arch, "grey"))
    ax.set_xticks(ks)
    ax.set_xlabel("top-k active features")
    ax.set_ylabel("Δ Model A tool-corr. (%)")
    ax.set_title("Model A fidelity vs k (paper Fig 4, left)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig4-paper] {out_path}")


def fig7_overlay(
    paper_rows: list[dict],
    best_per_model: dict[str, float],
    out_path: Path,
) -> None:
    """Paper Fig 7 (uniform pre-top-k scaling, Model B) overlay with our
    targeted post-top-k Model-A best cells, to make the
    'selective ≠ uniform' point visually."""
    uniform = [
        r["tool_correctness_B_post"] - r["tool_correctness_B_pre"]
        for r in paper_rows
        if "tool_correctness_B_pre" in r and "tool_correctness_B_post" in r
    ]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    if uniform:
        ax.boxplot([uniform], positions=[1],
                   tick_labels=["uniform pre-top-k\n(paper §6)"])
        ax.scatter(np.full(len(uniform), 1) + (np.random.RandomState(0).rand(len(uniform)) - 0.5) * 0.15,
                   uniform, alpha=0.4, s=14, c="#9467bd")
    if best_per_model:
        names = list(best_per_model.keys())
        vals = list(best_per_model.values())
        ax.scatter(range(2, 2 + len(names)), vals, s=70, c="#ff7f0e",
                   marker="*", label="targeted post-top-k (best cell, Δ A)")
        for i, (n, v) in enumerate(zip(names, vals)):
            ax.annotate(n, (i + 2, v), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=7)
        ax.set_xticks([1] + list(range(2, 2 + len(names))))
        ax.set_xticklabels(["uniform pre-top-k\n(paper §6)"] + ["targeted"] * len(names))
        ax.legend(loc="best")
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.set_ylabel("Δ tool-correctness (%)")
    ax.set_title("Selective post-top-k vs uniform pre-top-k")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig7-overlay] {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-model best-cell bar chart
# ──────────────────────────────────────────────────────────────────────────────

def compare_models(
    best_per_model: dict[str, dict],
    out_path: Path,
    paper_recon_baseline: float = 30.6,   # DFC mean Δ from paper Table 1
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    names = list(best_per_model.keys())
    vals = [m["d_steered_clean_tool"] for m in best_per_model.values()]
    bars = ax.bar(range(len(names)), vals, color="#1f77b4", alpha=0.85)
    ax.axhline(paper_recon_baseline, color="#d62728", ls="--",
               label=f"paper recon baseline (DFC mean +{paper_recon_baseline:g}%)")
    for i, m in enumerate(best_per_model.values()):
        ax.annotate(
            f"k%={m['k_pct']:g}\nα={m['alpha']:g}\nn={m['n']}",
            (i, vals[i]), xytext=(0, 4),
            textcoords="offset points", ha="center", fontsize=8,
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Best cell Δ Model-A tool-correctness (%)")
    ax.set_title("Best targeted-steering cell per model vs paper recon baseline")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[compare] {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steering-root", required=True, type=Path)
    p.add_argument("--paper-jsonl", default="results/results_full (1).jsonl")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    model_dirs = [d for d in args.steering_root.iterdir() if d.is_dir()]
    if not model_dirs:
        raise SystemExit(f"no model subdirs under {args.steering_root}")

    best_per_model: dict[str, dict] = {}
    all_rows_per_model: dict[str, list[dict]] = {}

    for md in sorted(model_dirs):
        short = md.name
        rows = aggregate_model_dir(md)
        if not rows:
            print(f"[skip] {short}: no rows")
            continue
        all_rows_per_model[short] = rows

        k_pcts = sorted({r["k_pct"]   for r in rows})
        alphas = sorted({r["alpha"]   for r in rows})

        # 3 heatmaps
        for metric, suffix, fmt in [
            ("d_steered_clean_tool",   "dA_tool_vs_clean",    "{:+.1f}"),
            ("d_steered_recon_tool",   "dA_tool_vs_recon",    "{:+.1f}"),
            ("d_steered_clean_overall","dA_overall_vs_clean", "{:+.2f}"),
        ]:
            grid = grid_from_rows(rows, metric, k_pcts, alphas)
            heatmap(
                grid, k_pcts, alphas,
                title=f"{short}\nΔ Model-A {suffix.replace('_', ' ')}",
                out_path=args.out / f"heatmap_{short}_{suffix}.png",
                fmt=fmt,
            )

        # 2 boxplots (replacing prior line plots — points = per-cell Δs)
        boxplot_vs_kpct(
            rows, "d_steered_clean_tool", k_pcts, alphas,
            title=f"{short} — Δ A tool-corr. vs clean by k%",
            out_path=args.out / f"boxplot_{short}_dA_vs_kpct.png",
            ylabel="Δ Model-A tool-corr. (%)",
        )
        boxplot_vs_alpha(
            rows, "d_steered_clean_tool", k_pcts, alphas,
            title=f"{short} — Δ A tool-corr. vs clean by α",
            out_path=args.out / f"boxplot_{short}_dA_vs_alpha.png",
            ylabel="Δ Model-A tool-corr. (%)",
        )

        # Best cell
        best = max(rows, key=lambda r: r["d_steered_clean_tool"])
        best_per_model[short] = best
        print(f"[best] {short}: k%={best['k_pct']:g} α={best['alpha']:g}  "
              f"Δ A tool={best['d_steered_clean_tool']:+.2f}%  n={best['n']}")

    if best_per_model:
        compare_models(best_per_model, args.out / "compare_models_best_cell.png")

    # Paper replications
    paper_path = Path(args.paper_jsonl)
    if paper_path.exists():
        paper_rows = []
        with paper_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    paper_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        paper_fig1_box(paper_rows, args.out / "paper_repro_fig1_modelA_pre_post_box.png")
        paper_fig4_topk(paper_rows, args.out / "paper_repro_fig4_topk_vs_dA.png")
        fig7_overlay(
            paper_rows,
            {n: m["d_steered_clean_tool"] for n, m in best_per_model.items()},
            args.out / "paper_repro_fig7_uniform_scale_overlay.png",
        )
    else:
        print(f"[paper-repro] {paper_path} missing; skipping replications")


if __name__ == "__main__":
    main()
