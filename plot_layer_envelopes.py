"""plot_layer_envelopes.py — per-layer figures for the dfc-D8k-excl10-k45 sweep.

For every ``dfc-D8k-excl10-k45-l<N>`` directory under ``--steering-root`` this:

  1. Re-uses ``build_steering_figures.aggregate_model_dir`` + ``heatmap`` to
     write the same three heatmaps the main figure script produces:
       - heatmap_<layer>_dA_tool_vs_clean.png
       - heatmap_<layer>_dA_tool_vs_recon.png
       - heatmap_<layer>_dA_overall_vs_clean.png

  2. Builds two combined "envelope" line plots that summarise every layer at
     once on a shared axis:
       - envelope_best_over_kpct_vs_alpha.png
            x = α, y = max over k% of Δ tool-corr (clean→steered) per layer.
            One arc per layer connects that layer's best cell at each α.
            Black dashed line + grey band = mean ± 95% CI across layers per α.
       - envelope_best_over_alpha_vs_kpct.png
            x = k%, y = max over α of Δ tool-corr per layer. Same overlay.

Layers with an incomplete (k%, α) grid (e.g. l32 from a partial sweep) are
listed and excluded from the two envelope plots — partial coverage would
distort the mean / CI band.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Reuse the canonical aggregator + heatmap helpers from the main figure script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_steering_figures import aggregate_model_dir, grid_from_rows, heatmap


def _setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.color": "#dddddd",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "font.family": "DejaVu Sans",
    })


def _layer_num(name: str) -> int:
    m = re.search(r"-l(\d+)$", name)
    return int(m.group(1)) if m else -1


def _per_layer_best(
    rows: list[dict],
    metric: str,
    axis_key: str,
    axis_values: list[float],
) -> list[float]:
    """For each value of ``axis_key`` return max of ``metric`` across the other
    axis. Missing combinations produce NaN."""
    bucket: dict[float, list[float]] = {}
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        bucket.setdefault(float(r[axis_key]), []).append(float(v))
    out: list[float] = []
    for x in axis_values:
        vals = bucket.get(x, [])
        out.append(max(vals) if vals else float("nan"))
    return out


def _ci_across_layers(
    arr: np.ndarray, conf: float = 0.95, min_layers_for_band: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column mean and ±t-CI half-width using only finite entries.

    arr: (n_layers, n_x). Returns (mean, lo, hi). Columns with fewer than
    `min_layers_for_band` contributing layers get NaN for lo/hi (the
    uncertainty estimate is too noisy with df=0,1 and the band balloons
    visually). Mean is still returned whenever ≥1 layer contributes.
    """
    means = np.full(arr.shape[1], np.nan)
    lo = np.full(arr.shape[1], np.nan)
    hi = np.full(arr.shape[1], np.nan)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        m = float(col.mean())
        means[j] = m
        if col.size >= min_layers_for_band:
            sem = float(col.std(ddof=1) / np.sqrt(col.size))
            t_crit = float(stats.t.ppf(0.5 + conf / 2, df=col.size - 1))
            half = t_crit * sem
            lo[j] = m - half
            hi[j] = m + half
    return means, lo, hi


def _plot_envelope(
    per_layer: dict[str, list[float]],
    x_values: list[float],
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
    x_log: bool = True,
) -> None:
    """Arc per layer (faint), black mean line + grey 95% CI band across layers."""
    layer_names = sorted(per_layer.keys(), key=_layer_num)
    fig, ax = plt.subplots(figsize=(10.0, 4.8))

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(layer_names), 2)))
    xs = np.asarray(x_values, dtype=np.float64)
    for color, name in zip(colors, layer_names):
        ys = np.asarray(per_layer[name], dtype=np.float64)
        finite = np.isfinite(ys)
        if not finite.any():
            continue
        ax.plot(
            xs[finite], ys[finite],
            marker="o", color=color, label=f"l{_layer_num(name)}",
            alpha=0.55, linewidth=1.1, markersize=3.5,
        )

    arr = np.array(
        [per_layer[name] for name in layer_names], dtype=np.float64
    )
    means, lo, hi = _ci_across_layers(arr, conf=0.95)
    finite = np.isfinite(means)
    if finite.any():
        band_finite = finite & np.isfinite(lo) & np.isfinite(hi)
        if band_finite.any():
            ax.fill_between(
                xs[band_finite], lo[band_finite], hi[band_finite],
                color="#444444", alpha=0.14, linewidth=0,
                label="95% CI across layers", zorder=2,
            )
        ax.plot(
            xs[finite], means[finite],
            color="black", linewidth=2.2, marker="s", markersize=5,
            markeredgecolor="white", markeredgewidth=0.8,
            label="mean across layers", zorder=10,
        )

    if x_log:
        ax.set_xscale("log")
    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{v:g}" for v in x_values])
    ax.axhline(0, color="#888888", linewidth=0.6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[envelope] {out_path}")


def _best_cell(
    rows: list[dict], metric: str
) -> tuple[float, float, float] | None:
    best: tuple[float, float, float] | None = None
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        if best is None or float(v) > best[0]:
            best = (float(v), float(r["k_pct"]), float(r["alpha"]))
    return best


def _best_cell_full(rows: list[dict], metric: str) -> dict | None:
    """Return the row that achieved the max metric value (so callers can pull
    CI fields too), or None if no finite values."""
    best: dict | None = None
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        if best is None or float(v) > float(best[metric]):
            best = r
    return best


def _plot_top_per_layer_bars(
    per_layer_rows: dict[str, list[dict]],
    layer_names: list[str],
    metric: str,
    metric_pretty: str,
    out_path: Path,
    title: str,
) -> None:
    """Bar chart: top Δ per layer (annotated with |S|, α) + per-cell 95% CI."""
    layer_names = sorted(layer_names, key=_layer_num)
    if not layer_names:
        return

    full_rows: list[tuple[str, dict | None]] = []
    for name in layer_names:
        full_rows.append((name, _best_cell_full(per_layer_rows[name], metric)))

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    xs = np.arange(len(layer_names))
    ys = np.array([
        float(b[metric]) if b is not None else np.nan for _, b in full_rows
    ], dtype=np.float64)

    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    for (_, best), y in zip(full_rows, ys):
        if best is None or not np.isfinite(y):
            yerr_lo.append(0.0); yerr_hi.append(0.0); continue
        lo = float(best.get("d_tool_ci_lo", float("nan")))
        hi = float(best.get("d_tool_ci_hi", float("nan")))
        if np.isfinite(lo) and np.isfinite(hi) and metric == "d_steered_clean_tool":
            yerr_lo.append(max(0.0, y - lo)); yerr_hi.append(max(0.0, hi - y))
        else:
            yerr_lo.append(0.0); yerr_hi.append(0.0)

    bar_colors = ["#4a90c2" if (np.isfinite(y) and y >= 0) else "#bbbbbb"
                  for y in ys]
    ax.bar(xs, np.where(np.isfinite(ys), ys, 0.0),
           color=bar_colors, edgecolor="white", linewidth=0.8, width=0.66,
           yerr=[yerr_lo, yerr_hi], capsize=0,
           error_kw={"elinewidth": 1.0, "ecolor": "#444444", "alpha": 0.9})

    finite_max = float(np.nanmax(ys)) if np.isfinite(ys).any() else 1.0
    pad = max(abs(finite_max) * 0.04, 0.5)
    for x, (_, best), eh in zip(xs, full_rows, yerr_hi):
        if best is None:
            ax.text(x, 0.0, "n/a", ha="center", va="bottom",
                    fontsize=7.5, color="#888")
            continue
        v = float(best[metric])
        S = int(best.get("subset_size", 0))
        a = float(best["alpha"])
        ax.text(
            x, v + eh + pad * 0.6,
            f"{v:+.1f}\n|S|={S} α={a:g}",
            ha="center", va="bottom", fontsize=7.5, color="#222",
        )

    if np.isfinite(ys).any():
        win = int(np.nanargmax(ys))
        ax.bar(xs[win], ys[win], color="#d96a6a",
               edgecolor="white", linewidth=0.8, width=0.66)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"l{_layer_num(n)}" for n in layer_names])
    ax.axhline(0, color="#888888", linewidth=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel(metric_pretty)
    ax.set_title(title)
    ax.grid(True, axis="y")
    if metric == "d_steered_clean_tool":
        ax.text(0.99, -0.16,
                "error bars: 95% paired-t CI of the best cell (n=40)",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=7.5, color="#666666", style="italic")
    ymax = float(np.nanmax(ys)) if np.isfinite(ys).any() else 1.0
    ymin = float(np.nanmin(ys)) if np.isfinite(ys).any() else 0.0
    ax.set_ylim(min(ymin, 0) - pad, ymax + pad * 6 + max(yerr_hi + [0.0]))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[top-per-layer] {out_path}")


def _plot_best_cell_per_layer(
    per_layer_rows: dict[str, list[dict]],
    complete: list[str],
    metrics: list[str],
    metric_pretty: dict[str, str],
    out_path: Path,
    title: str,
) -> None:
    """Bar/line chart: best Δ across the full (k%, α) grid per layer, per metric."""
    layer_names = sorted(complete, key=_layer_num)
    layer_idx = [_layer_num(n) for n in layer_names]
    if not layer_names:
        return

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = {
        "d_steered_clean_tool":    "#1f77b4",
        "d_steered_recon_tool":    "#2ca02c",
        "d_steered_clean_overall": "#d62728",
    }
    for metric in metrics:
        ys: list[float] = []
        Ss: list[int | float] = []
        alphas: list[float] = []
        los: list[float] = []
        his: list[float] = []
        for name in layer_names:
            best = _best_cell_full(per_layer_rows[name], metric)
            if best is None:
                ys.append(np.nan); Ss.append(np.nan); alphas.append(np.nan)
                los.append(np.nan); his.append(np.nan)
            else:
                ys.append(float(best[metric]))
                Ss.append(int(best.get("subset_size", 0)))
                alphas.append(float(best["alpha"]))
                los.append(float(best.get("d_tool_ci_lo", float("nan"))))
                his.append(float(best.get("d_tool_ci_hi", float("nan"))))
        c = colors.get(metric, "grey")
        ys_arr = np.asarray(ys, dtype=np.float64)
        if metric == "d_steered_clean_tool":
            los_arr = np.asarray(los, dtype=np.float64)
            his_arr = np.asarray(his, dtype=np.float64)
            band = np.isfinite(ys_arr) & np.isfinite(los_arr) & np.isfinite(his_arr)
            if band.any():
                ax.fill_between(np.asarray(layer_idx)[band],
                                los_arr[band], his_arr[band],
                                color=c, alpha=0.13, linewidth=0)
            ax.plot(layer_idx, ys_arr, marker="o", color=c,
                    label=metric_pretty.get(metric, metric),
                    markeredgecolor="white", markeredgewidth=0.8)
        else:
            ax.plot(layer_idx, ys_arr, marker="o", color=c,
                    label=metric_pretty.get(metric, metric),
                    markeredgecolor="white", markeredgewidth=0.8)
        # Annotate (|S|, α) for the primary metric only on top-3 layers.
        if metric == "d_steered_clean_tool" and any(np.isfinite(ys_arr)):
            top3 = np.argsort(-ys_arr)[:3]
            for k in top3:
                x = layer_idx[k]; y = ys[k]; S = Ss[k]; a = alphas[k]
                if np.isfinite(y):
                    ax.annotate(
                        f"|S|={S} α={a:g}",
                        (x, y), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", fontsize=7, color=c,
                    )

    ax.set_xticks(layer_idx)
    ax.set_xticklabels([f"l{i}" for i in layer_idx])
    ax.axhline(0, color="#888888", linewidth=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("best Δ across the sweep grid")
    ax.set_title(title)
    ax.grid(True, axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[best-cell] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--steering-root", type=Path,
        default=Path("results/targeted_steering"),
    )
    p.add_argument("--prefix", default="dfc-D8k-excl10-k45-l")
    p.add_argument(
        "--out", type=Path,
        default=Path("results/figures/k45_layers"),
    )
    p.add_argument(
        "--metrics",
        default="d_steered_clean_tool,d_steered_recon_tool,d_steered_clean_overall",
        help="comma-separated list of row metrics to envelope-plot",
    )
    p.add_argument("--max-abs-k", type=int, default=10,
                   help="Cap |S| (absolute neurons steered) for the absolute-k "
                        "cross-layer plots. Cells with subset_size beyond this "
                        "are dropped from the absk plots.")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _setup_style()

    layer_dirs = sorted(
        [d for d in args.steering_root.iterdir()
         if d.is_dir() and d.name.startswith(args.prefix)
         and re.fullmatch(re.escape(args.prefix) + r"\d+", d.name)],
        key=lambda d: _layer_num(d.name),
    )
    if not layer_dirs:
        raise SystemExit(
            f"No layer dirs found under {args.steering_root} "
            f"with prefix {args.prefix!r}"
        )

    per_layer_rows = {d.name: aggregate_model_dir(d) for d in layer_dirs}

    all_k = sorted({r["k_pct"] for rows in per_layer_rows.values() for r in rows})
    all_a = sorted({r["alpha"] for rows in per_layer_rows.values() for r in rows})
    expected = len(all_k) * len(all_a)
    print(f"[scan] layers ({len(layer_dirs)}): "
          f"{[d.name for d in layer_dirs]}")
    print(f"[scan] k% values: {all_k}")
    print(f"[scan] α values:  {all_a}")
    print(f"[scan] full grid = {expected} cells per layer")

    # 1) Heatmaps per layer.
    heatmap_specs = [
        ("d_steered_clean_tool",    "dA_tool_vs_clean",     "{:+.1f}"),
        ("d_steered_recon_tool",    "dA_tool_vs_recon",     "{:+.1f}"),
        ("d_steered_clean_overall", "dA_overall_vs_clean",  "{:+.2f}"),
    ]
    for d in layer_dirs:
        rows = per_layer_rows[d.name]
        if not rows:
            print(f"[skip] {d.name}: no rows")
            continue
        for metric, suffix, fmt in heatmap_specs:
            grid = grid_from_rows(rows, metric, all_k, all_a)
            heatmap(
                grid, all_k, all_a,
                title=f"{d.name}\nΔ Model-A {suffix.replace('_', ' ')}",
                out_path=args.out / f"heatmap_{d.name}_{suffix}.png",
                fmt=fmt,
            )

    # 2) Envelope plots — keep layers covering the modal (= per-layer) grid.
    # Some layers got fine-grained k% extensions (e.g. l13 with k=40, 70),
    # which inflates the union grid beyond what each layer individually has.
    # A layer is "complete" if it has at least 30 cells (the original 6×5 grid).
    MODAL_COMPLETE = 30
    complete: list[str] = []
    for d in layer_dirs:
        n = len(per_layer_rows[d.name])
        if n >= MODAL_COMPLETE:
            complete.append(d.name)
        else:
            print(f"[note] excluding {d.name} from envelope plots: "
                  f"{n}/{MODAL_COMPLETE} cells")

    if not complete:
        raise SystemExit("No layer has full coverage; refusing to plot envelopes.")

    label = args.prefix.removesuffix("-l")

    metric_pretty = {
        "d_steered_clean_tool":    "Δ tool-corr (steered − clean), %",
        "d_steered_recon_tool":    "Δ tool-corr (steered − recon), %",
        "d_steered_clean_overall": "Δ overall score (steered − clean)",
    }
    metric_short = {
        "d_steered_clean_tool":    "tool_vs_clean",
        "d_steered_recon_tool":    "tool_vs_recon",
        "d_steered_clean_overall": "overall_vs_clean",
    }

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for metric in metrics:
        ylabel_alpha = f"max over k% of {metric_pretty.get(metric, metric)}"
        ylabel_k = f"max over α of {metric_pretty.get(metric, metric)}"
        suffix = metric_short.get(metric, metric)

        best_vs_alpha = {
            ln: _per_layer_best(per_layer_rows[ln], metric, "alpha", all_a)
            for ln in complete
        }
        _plot_envelope(
            best_vs_alpha, all_a,
            x_label="α (steering coefficient)",
            y_label=ylabel_alpha,
            title=(f"{label} [{suffix}]: best-cell envelope vs α "
                   f"(arc per layer; mean ± 95% CI across layers)"),
            out_path=args.out / f"envelope_{suffix}_best_over_kpct_vs_alpha.png",
        )

        best_vs_k = {
            ln: _per_layer_best(per_layer_rows[ln], metric, "k_pct", all_k)
            for ln in complete
        }
        _plot_envelope(
            best_vs_k, all_k,
            x_label="k% (top-k feature subset)",
            y_label=ylabel_k,
            title=(f"{label} [{suffix}]: best-cell envelope vs k% "
                   f"(arc per layer; mean ± 95% CI across layers)"),
            out_path=args.out / f"envelope_{suffix}_best_over_alpha_vs_kpct.png",
        )

    # Per-layer best-cell summary across the full (k%, α) grid.
    _plot_best_cell_per_layer(
        per_layer_rows, complete, metrics, metric_pretty,
        out_path=args.out / "best_cell_per_layer.png",
        title=f"{label}: best (k%, α) cell per layer",
    )

    # Single-metric top-per-layer bar chart (one PNG per metric).
    for metric in metrics:
        suffix = metric_short.get(metric, metric)
        _plot_top_per_layer_bars(
            per_layer_rows, complete, metric,
            metric_pretty=metric_pretty.get(metric, metric),
            out_path=args.out / f"top_per_layer_{suffix}.png",
            title=(f"{label} [{suffix}]: top performance per layer "
                   f"(best across the full k% × α grid)"),
        )

    # ── Absolute-k cross-layer report (cap |S| ≤ args.max_abs_k) ──────────
    max_S = int(args.max_abs_k)
    print(f"\n=== Absolute-k cross-layer report (|S| ≤ {max_S}) ===")

    per_layer_capped = {
        ln: [c for c in per_layer_rows[ln]
             if int(c.get("subset_size", 0)) <= max_S]
        for ln in complete
    }
    for ln in sorted(per_layer_capped, key=_layer_num):
        sizes = sorted({int(c["subset_size"]) for c in per_layer_capped[ln]})
        print(f"  {ln}: kept {len(per_layer_capped[ln])}/30 cells; "
              f"|S| present: {sizes}")

    # Drop |S|=0 cells (no neurons selected = trivial no-op).
    for ln in list(per_layer_capped):
        per_layer_capped[ln] = [
            c for c in per_layer_capped[ln] if int(c.get("subset_size", 0)) >= 1
        ]
    all_S = sorted({
        int(c["subset_size"])
        for cells in per_layer_capped.values() for c in cells
    })

    for metric in metrics:
        suffix = metric_short.get(metric, metric)
        ylabel = metric_pretty.get(metric, metric)

        # Per-layer arc: max over α at each |S| value (only existing cells).
        best_vs_S: dict[str, list[float]] = {}
        for ln in complete:
            bucket: dict[int, list[float]] = {}
            for c in per_layer_capped[ln]:
                v = c.get(metric)
                if v is None or not np.isfinite(v):
                    continue
                bucket.setdefault(int(c["subset_size"]), []).append(float(v))
            best_vs_S[ln] = [
                max(bucket[s]) if bucket.get(s) else float("nan")
                for s in all_S
            ]
        _plot_envelope(
            best_vs_S, all_S,
            x_label=f"|S| (absolute # of neurons steered, cap = {max_S})",
            y_label=f"max over α of {ylabel}",
            title=(f"{label} [{suffix}]: |S|-budgeted envelope (≤ {max_S}) "
                   f"vs |S| — arc per layer"),
            out_path=args.out / f"envelope_{suffix}_absk_vs_absk.png",
            x_log=False,
        )

        # Per-layer arc: max over |S|≤cap at each α.
        best_vs_alpha_capped: dict[str, list[float]] = {}
        for ln in complete:
            bucket_a: dict[float, list[float]] = {}
            for c in per_layer_capped[ln]:
                v = c.get(metric)
                if v is None or not np.isfinite(v):
                    continue
                bucket_a.setdefault(float(c["alpha"]), []).append(float(v))
            best_vs_alpha_capped[ln] = [
                max(bucket_a[a]) if bucket_a.get(a) else float("nan")
                for a in all_a
            ]
        _plot_envelope(
            best_vs_alpha_capped, all_a,
            x_label="α (steering coefficient)",
            y_label=f"max over |S|≤{max_S} of {ylabel}",
            title=(f"{label} [{suffix}]: |S|-budgeted envelope (≤ {max_S}) "
                   f"vs α — arc per layer"),
            out_path=args.out / f"envelope_{suffix}_absk_vs_alpha.png",
        )

        # Top-per-layer bar chart restricted to |S| ≤ cap.
        _plot_top_per_layer_bars(
            per_layer_capped, complete, metric,
            metric_pretty=metric_pretty.get(metric, metric),
            out_path=args.out / f"top_per_layer_{suffix}_absk.png",
            title=(f"{label} [{suffix}]: top performance per layer "
                   f"under |S| ≤ {max_S}"),
        )

    print(f"[done] outputs in {args.out}")


if __name__ == "__main__":
    main()
