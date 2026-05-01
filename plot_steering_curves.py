"""Two-panel steering boxplots over per-cell means.

Per-cell mean = average steered score across all valid prompts in cell (k%, α).
Cells with zero valid prompts are dropped.

Left panel:  one box per k%; the box's points are the per-cell means across α
             (so each k% box contains up to len(α-grid) values).
Right panel: one box per α;  points are per-cell means across k% (up to
             len(k%-grid) values per box).

Scores of -1 (invalid / unscored) are dropped before averaging.

Usage:
    python plot_steering_curves.py \
        --steering-dir results/targeted_steering/dfc-D8k-excl10-freeexcl-k160 \
        --out results/figures/steering_curves_dfc-D8k-excl10-freeexcl-k160.png \
        [--metric overall_score]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VALID_METRICS = ("overall_score", "format_accuracy", "tool_correctness")


def load_rows(steering_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(steering_dir.glob("*.jsonl")):
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def cell_means(
    rows: list[dict], metric: str,
) -> dict[tuple[float, float], float]:
    """Returns {(k_pct, alpha) -> mean valid steered score across prompts}."""
    bucket: dict[tuple[float, float], list[float]] = {}
    for r in rows:
        s = r.get("steered_score", {}).get(metric, None)
        if s is None:
            continue
        v = float(s) if not isinstance(s, bool) else float(s)
        if v < 0:  # -1 sentinel = invalid / unscored
            continue
        bucket.setdefault((float(r["k_pct"]), float(r["alpha"])), []).append(v)
    return {k: float(np.mean(v)) for k, v in bucket.items() if v}


def collect_by_axis(
    means: dict[tuple[float, float], float], axis: str,
) -> dict[float, list[float]]:
    """Group per-cell means by one axis value (k% or α)."""
    idx = 0 if axis == "k_pct" else 1
    bucket: dict[float, list[float]] = {}
    for key, m in means.items():
        bucket.setdefault(key[idx], []).append(m)
    return bucket


def plot_box_vs_x(ax, bucket: dict[float, list[float]], x_axis: str) -> None:
    """Boxplot at each x value; box collapses over the other axis + all prompts.

    x positions placed on a log-scale axis. Mean overlaid as a small marker so
    the trend is readable when boxes are tall.
    """
    x_vals = sorted(bucket.keys())
    if not x_vals:
        return
    data = [np.asarray(bucket[x], dtype=np.float64) for x in x_vals]
    means = [float(d.mean()) if d.size else np.nan for d in data]
    counts = [d.size for d in data]

    # Log axis can't show x=0 — shift zero values to a small placeholder so they
    # remain visible on the left edge.
    plot_x = np.array(x_vals, dtype=np.float64)
    zero_mask = plot_x <= 0
    if zero_mask.any():
        positives = plot_x[~zero_mask]
        eps = positives.min() / 4.0 if positives.size else 0.5
        plot_x[zero_mask] = eps

    # Width per box on a log axis: use a fixed log-width.
    widths = plot_x * 0.25

    bp = ax.boxplot(
        data, positions=plot_x, widths=widths, manage_ticks=False,
        showfliers=True, patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#5b9bd5"); patch.set_alpha(0.4)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.5)

    ax.plot(plot_x, means, color="crimson", marker="o", linestyle="-",
            label="mean", zorder=5)

    ax.set_xscale("log")
    ax.set_xticks(plot_x)
    ax.set_xticklabels(
        [f"{v:g}" + (f"\n(n={n})" if n else "") for v, n in zip(x_vals, counts)],
        fontsize="x-small",
    )
    if x_axis == "k_pct":
        ax.set_xlabel("k% (top-k feature subset) — box = per-cell means across α")
        ax.set_title("score vs k%")
    else:
        ax.set_xlabel("α (steering coefficient) — box = per-cell means across k%")
        ax.set_title("score vs coefficient")
    ax.set_ylabel("per-cell mean steered score")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize="x-small")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steering-dir", type=Path, required=True,
                   help="Directory of k*_a*.jsonl cell files")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PNG path")
    p.add_argument("--metric", default="overall_score", choices=VALID_METRICS)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.steering_dir)
    if not rows:
        raise SystemExit(f"No rows found under {args.steering_dir}")

    means = cell_means(rows, args.metric)
    if not means:
        raise SystemExit(f"No valid scored rows for metric={args.metric!r}")
    by_k     = collect_by_axis(means, "k_pct")
    by_alpha = collect_by_axis(means, "alpha")
    print(f"loaded {len(rows)} rows → {len(means)} non-empty cells  "
          f"({len(by_k)} k% values, {len(by_alpha)} α values, "
          f"metric={args.metric})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    plot_box_vs_x(axes[0], by_k,     x_axis="k_pct")
    plot_box_vs_x(axes[1], by_alpha, x_axis="alpha")
    fig.suptitle(
        f"Targeted steering: {args.steering_dir.name}  (metric={args.metric})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
