#!/usr/bin/env python3
"""
Zero-dependency analysis for the x-coder hyperparameter sweep.

Outputs:
  - analysis/tables/*.csv
  - analysis/figures/*.svg
  - analysis/summary.md
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_RESULTS = REPO_ROOT / "results_full.jsonl"
STEER_RESULTS = REPO_ROOT / "results_steer.jsonl"

ANALYSIS_DIR = REPO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
SUMMARY_PATH = ANALYSIS_DIR / "summary.md"

NAME_RE = re.compile(
    r"antebe1/"
    r"(?P<prefix>cc|dfc)-"
    r"D(?P<dict_k>\d+)k"
    r"(?P<nol1>-nol1)?"
    r"(?P<excl>-excl(?P<excl_pct>\d+))?"
    r"(?P<freeexcl>-freeexcl)?"
    r"-k(?P<topk>\d+)$"
)

SVG_COLORS = [
    "#1b6ca8",
    "#c94c4c",
    "#2f855a",
    "#b7791f",
    "#6b46c1",
    "#0f766e",
    "#9f1239",
    "#4a5568",
]


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {path} line {line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} contained no rows")
    return rows


def parse_model_name(name: str) -> dict:
    match = NAME_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Unrecognized model name format: {name}")
    parsed = match.groupdict()
    return {
        "parsed_architecture": "CrossCoder" if parsed["prefix"] == "cc" else "DFC",
        "parsed_dict_size": int(parsed["dict_k"]) * 1024,
        "parsed_topk": int(parsed["topk"]),
        "parsed_exclusive_pct": int(parsed["excl_pct"]) / 100 if parsed["excl_pct"] else 0.0,
        "parsed_use_l1": parsed["nol1"] is None,
        "parsed_free_exclusive": parsed["freeexcl"] is not None,
    }


def float_eq(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def infer_regime(row: dict) -> str:
    if row["is_crosscoder_baseline"]:
        return "CrossCoder + L1" if row["sparsity_coef"] > 0 else "CrossCoder no-L1"
    return "DFC free-exclusive" if row["free_exclusive"] else "DFC penalized-exclusive"


def validate_and_flatten(rows: list[dict]) -> list[dict]:
    flat_rows = []
    for row in rows:
        hp = dict(row.get("hyperparameters", {}))
        parsed = parse_model_name(row["name"])

        if hp["architecture"] != parsed["parsed_architecture"]:
            raise ValueError(f"Architecture mismatch for {row['name']}")
        if int(hp["dict_size"]) != parsed["parsed_dict_size"]:
            raise ValueError(f"dict_size mismatch for {row['name']}")
        if int(hp["k"]) != parsed["parsed_topk"]:
            raise ValueError(f"k mismatch for {row['name']}")

        a_excl = float(hp.get("model_a_exclusive_pct", 0.0))
        b_excl = float(hp.get("model_b_exclusive_pct", 0.0))
        if not float_eq(a_excl, b_excl):
            raise ValueError(f"Asymmetric exclusive percentages in {row['name']}")
        if not float_eq(a_excl, parsed["parsed_exclusive_pct"]):
            raise ValueError(f"Exclusive % mismatch for {row['name']}")

        flat = {k: v for k, v in row.items() if k != "hyperparameters"}
        flat.update(hp)
        flat.update(parsed)
        flat["exclusive_pct"] = a_excl
        flat["is_crosscoder_baseline"] = float_eq(a_excl, 0.0)
        flat["variant_family"] = "CrossCoder baseline" if flat["is_crosscoder_baseline"] else "DFC"
        flat["free_exclusive"] = (not flat["is_crosscoder_baseline"]) and float_eq(
            flat["exclusive_sparsity_coef"], 0.0
        )
        flat["uses_l1"] = float(flat["sparsity_coef"]) > 0.0
        flat["uses_exclusive_l1"] = float(flat["exclusive_sparsity_coef"]) > 0.0
        flat["regime"] = infer_regime(flat)
        flat["exclusive_pct_label"] = f"{int(round(flat['exclusive_pct'] * 100))}%"
        flat_rows.append(flat)
    return sorted(flat_rows, key=lambda x: x["name"])


def identify_metric_columns(rows: list[dict]) -> dict[str, list[str]]:
    stages = {"pre": [], "post": [], "delta": []}
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())
    for column in sorted(all_columns):
        for stage in stages:
            if column.endswith(f"_{stage}"):
                stages[stage].append(column)
    if not stages["post"]:
        raise ValueError("No *_post columns found")
    return stages


def metric_base_and_side(metric: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+)_(A|B)_(pre|post|delta)", metric)
    if not match:
        raise ValueError(f"Unexpected metric format: {metric}")
    return match.group(1), match.group(2)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer fieldnames for empty CSV: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tidy_metrics(rows: list[dict], metric_columns: dict[str, list[str]]) -> list[dict]:
    tidy = []
    for row in rows:
        for stage, columns in metric_columns.items():
            for column in columns:
                metric_family, side = metric_base_and_side(column)
                tidy.append(
                    {
                        "name": row["name"],
                        "architecture": row["architecture"],
                        "variant_family": row["variant_family"],
                        "regime": row["regime"],
                        "dict_size": row["dict_size"],
                        "topk": row["k"],
                        "exclusive_pct": row["exclusive_pct"],
                        "exclusive_pct_label": row["exclusive_pct_label"],
                        "model_side": side,
                        "metric_family": metric_family,
                        "stage": stage,
                        "value": row[column],
                    }
                )
    return tidy


def summary_stats(values: list[float]) -> dict:
    vals = [float(v) for v in values]
    return {
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
        "count": len(vals),
    }


def grouped_post_summary(tidy: list[dict], group_cols: list[str]) -> list[dict]:
    buckets = defaultdict(list)
    for row in tidy:
        if row["stage"] != "post":
            continue
        key = tuple(row[col] for col in group_cols + ["metric_family", "model_side"])
        buckets[key].append(float(row["value"]))

    out = []
    for key in sorted(buckets):
        stats = summary_stats(buckets[key])
        record = {}
        for idx, col in enumerate(group_cols + ["metric_family", "model_side"]):
            record[col] = key[idx]
        record.update(stats)
        out.append(record)
    return out


def compute_controlled_topk_deltas(rows: list[dict], post_metrics: list[str]) -> tuple[list[dict], list[dict]]:
    row_map = {
        (row["dict_size"], row["exclusive_pct"], row["regime"], row["k"]): row
        for row in rows
    }
    pairs = [(45, 90), (90, 160), (45, 160)]
    raw = []
    for dict_size in sorted({row["dict_size"] for row in rows}):
        for exclusive_pct in sorted({row["exclusive_pct"] for row in rows}):
            for regime in sorted({row["regime"] for row in rows}):
                for k_lo, k_hi in pairs:
                    lo = row_map.get((dict_size, exclusive_pct, regime, k_lo))
                    hi = row_map.get((dict_size, exclusive_pct, regime, k_hi))
                    if lo is None or hi is None:
                        continue
                    for metric in post_metrics:
                        raw.append(
                            {
                                "dict_size": dict_size,
                                "exclusive_pct": exclusive_pct,
                                "regime": regime,
                                "metric": metric,
                                "comparison": f"{k_lo}->{k_hi}",
                                "delta": float(hi[metric]) - float(lo[metric]),
                            }
                        )

    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["metric"], row["comparison"])].append(row["delta"])

    summary = []
    for key in sorted(grouped):
        stats = summary_stats(grouped[key])
        summary.append(
            {
                "metric": key[0],
                "comparison": key[1],
                **stats,
            }
        )
    return raw, summary


def compute_controlled_exclusive_vs_baseline(rows: list[dict], post_metrics: list[str]) -> tuple[list[dict], list[dict]]:
    baseline_rows = {
        (row["dict_size"], row["k"], row["uses_l1"]): row
        for row in rows
        if row["is_crosscoder_baseline"]
    }
    raw = []
    for row in rows:
        if row["is_crosscoder_baseline"]:
            continue
        baseline = baseline_rows.get((row["dict_size"], row["k"], row["uses_l1"]))
        if baseline is None:
            continue
        for metric in post_metrics:
            raw.append(
                {
                    "dict_size": row["dict_size"],
                    "topk": row["k"],
                    "uses_l1": row["uses_l1"],
                    "regime": row["regime"],
                    "exclusive_pct": row["exclusive_pct"],
                    "metric": metric,
                    "delta_vs_baseline": float(row[metric]) - float(baseline[metric]),
                }
            )

    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["metric"], row["regime"], row["exclusive_pct"])].append(row["delta_vs_baseline"])

    summary = []
    for key in sorted(grouped):
        stats = summary_stats(grouped[key])
        summary.append(
            {
                "metric": key[0],
                "regime": key[1],
                "exclusive_pct": key[2],
                **stats,
            }
        )
    return raw, summary


def inspect_steering_context() -> list[dict]:
    rows = load_jsonl(STEER_RESULTS)
    flat = []
    for row in rows:
        hp = dict(row["hyperparameters"])
        flat.append({**{k: v for k, v in row.items() if k != "hyperparameters"}, **hp})

    grouped = defaultdict(lambda: {"overall": [], "tool": []})
    for row in flat:
        key = (row["k"], row["model_a_exclusive_pct"], row["exclusive_sparsity_coef"])
        grouped[key]["overall"].append(float(row["overall_score_B_steer_delta"]))
        grouped[key]["tool"].append(float(row["tool_correctness_B_steer_delta"]))

    summary = []
    for key in sorted(grouped):
        summary.append(
            {
                "topk": key[0],
                "exclusive_pct": key[1],
                "exclusive_sparsity_coef": key[2],
                "overall_score_B_steer_delta_mean": mean(grouped[key]["overall"]),
                "tool_correctness_B_steer_delta_mean": mean(grouped[key]["tool"]),
                "n_rows": len(grouped[key]["overall"]),
            }
        )
    return summary


def linear_scale(value: float, domain_min: float, domain_max: float, out_min: float, out_max: float) -> float:
    if domain_max == domain_min:
        return (out_min + out_max) / 2
    ratio = (value - domain_min) / (domain_max - domain_min)
    return out_min + ratio * (out_max - out_min)


def svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_line_chart_svg(
    panels: list[dict],
    x_values: list[float],
    path: Path,
    title: str,
    x_label: str,
) -> None:
    panel_w = 400
    panel_h = 240
    cols = 2
    rows = math.ceil(len(panels) / cols)
    total_w = cols * panel_w
    total_h = rows * panel_h + 50

    all_y = []
    for panel in panels:
        for series in panel["series"]:
            all_y.extend([y for _, y in series["points"]])
    y_min = min(all_y)
    y_max = max(all_y)
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{total_w / 2}" y="24" text-anchor="middle" font-size="20" font-family="monospace">{svg_escape(title)}</text>',
    ]

    for idx, panel in enumerate(panels):
        col = idx % cols
        row = idx // cols
        x0 = col * panel_w
        y0 = row * panel_h + 40
        left = x0 + 60
        right = x0 + panel_w - 20
        top = y0 + 25
        bottom = y0 + panel_h - 45

        parts.append(f'<rect x="{x0 + 8}" y="{y0 + 5}" width="{panel_w - 16}" height="{panel_h - 16}" fill="none" stroke="#ddd"/>')
        parts.append(f'<text x="{x0 + panel_w / 2}" y="{y0 + 18}" text-anchor="middle" font-size="14" font-family="monospace">{svg_escape(panel["title"])}</text>')
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>')

        for tick in x_values:
            px = linear_scale(tick, min(x_values), max(x_values), left, right)
            parts.append(f'<line x1="{px}" y1="{bottom}" x2="{px}" y2="{bottom + 4}" stroke="black"/>')
            parts.append(f'<text x="{px}" y="{bottom + 18}" text-anchor="middle" font-size="11" font-family="monospace">{svg_escape(tick)}</text>')

        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = y_min + frac * (y_max - y_min)
            py = linear_scale(val, y_min, y_max, bottom, top)
            parts.append(f'<line x1="{left}" y1="{py}" x2="{right}" y2="{py}" stroke="#eee"/>')
            parts.append(f'<text x="{left - 8}" y="{py + 4}" text-anchor="end" font-size="10" font-family="monospace">{val:.2f}</text>')

        parts.append(f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h - 8}" text-anchor="middle" font-size="11" font-family="monospace">{svg_escape(x_label)}</text>')

        legend_y = top
        for s_idx, series in enumerate(panel["series"]):
            color = SVG_COLORS[s_idx % len(SVG_COLORS)]
            points = []
            for x, y in series["points"]:
                px = linear_scale(x, min(x_values), max(x_values), left, right)
                py = linear_scale(y, y_min, y_max, bottom, top)
                points.append((px, py))
            polyline = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}"/>')
            for px, py in points:
                parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}"/>')
            lx = right - 120
            ly = legend_y + s_idx * 14
            parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 16}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<text x="{lx + 20}" y="{ly + 4}" font-size="10" font-family="monospace">{svg_escape(series["label"])}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def build_bar_chart_svg(rows: list[dict], path: Path, title: str) -> None:
    metrics = ["overall_score", "format_accuracy", "tool_correctness"]
    sides = ["A", "B"]
    regimes = sorted({row["regime"] for row in rows})
    dict_sizes = sorted({row["dict_size"] for row in rows})
    width = 1200
    height = 900
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="20" font-family="monospace">{svg_escape(title)}</text>',
    ]
    cell_w = 560
    cell_h = 250
    all_vals = []
    for row in rows:
        for metric in metrics:
            for side in sides:
                all_vals.append(float(row[f"{metric}_{side}_post"]))
    y_min = min(all_vals)
    y_max = max(all_vals)
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    for r_idx, metric in enumerate(metrics):
        for c_idx, side in enumerate(sides):
            x0 = 20 + c_idx * cell_w
            y0 = 50 + r_idx * cell_h
            left = x0 + 50
            right = x0 + cell_w - 20
            top = y0 + 25
            bottom = y0 + cell_h - 45
            parts.append(f'<rect x="{x0}" y="{y0}" width="{cell_w - 20}" height="{cell_h - 10}" fill="none" stroke="#ddd"/>')
            parts.append(f'<text x="{x0 + (cell_w - 20) / 2}" y="{y0 + 18}" text-anchor="middle" font-size="14" font-family="monospace">{metric} model {side}</text>')
            parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>')
            parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>')

            group_width = (right - left) / max(1, len(regimes))
            bar_width = group_width / (len(dict_sizes) + 1)
            for i, regime in enumerate(regimes):
                gx = left + i * group_width
                parts.append(
                    f'<text x="{gx + group_width / 2}" y="{bottom + 18}" text-anchor="middle" font-size="10" font-family="monospace">{svg_escape(regime)}</text>'
                )
                for j, dict_size in enumerate(dict_sizes):
                    match = next((row for row in rows if row["regime"] == regime and row["dict_size"] == dict_size), None)
                    if match is None:
                        continue
                    value = float(match[f"{metric}_{side}_post"])
                    x = gx + 8 + j * bar_width
                    y = linear_scale(value, y_min, y_max, bottom, top)
                    h = bottom - y
                    color = SVG_COLORS[j % len(SVG_COLORS)]
                    parts.append(f'<rect x="{x}" y="{y}" width="{bar_width - 6}" height="{h}" fill="{color}"/>')
            for frac in [0.0, 0.5, 1.0]:
                val = y_min + frac * (y_max - y_min)
                py = linear_scale(val, y_min, y_max, bottom, top)
                parts.append(f'<line x1="{left}" y1="{py}" x2="{right}" y2="{py}" stroke="#eee"/>')
                parts.append(f'<text x="{left - 8}" y="{py + 4}" text-anchor="end" font-size="10" font-family="monospace">{val:.2f}</text>')

    legend_x = width - 180
    legend_y = height - 60
    for idx, dict_size in enumerate(dict_sizes):
        color = SVG_COLORS[idx % len(SVG_COLORS)]
        parts.append(f'<rect x="{legend_x}" y="{legend_y + idx * 18}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y + idx * 18 + 12}" font-size="11" font-family="monospace">D{dict_size}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def make_line_plots(rows: list[dict], metric_bases: list[str]) -> None:
    dict_sizes = sorted({row["dict_size"] for row in rows})
    regimes = sorted({row["regime"] for row in rows})
    topks = sorted({row["k"] for row in rows})
    excls = sorted({row["exclusive_pct"] for row in rows})

    for dict_size in dict_sizes:
        subset = [row for row in rows if row["dict_size"] == dict_size]
        panels = []
        for metric in metric_bases:
            for side in ["A", "B"]:
                series = []
                for regime in regimes:
                    regime_rows = [row for row in subset if row["regime"] == regime]
                    for excl in excls:
                        line_rows = sorted(
                            [row for row in regime_rows if float_eq(row["exclusive_pct"], excl)],
                            key=lambda x: x["k"],
                        )
                        if not line_rows:
                            continue
                        label = f"{regime}, excl={int(round(excl * 100))}%"
                        series.append(
                            {
                                "label": label,
                                "points": [(row["k"], float(row[f"{metric}_{side}_post"])) for row in line_rows],
                            }
                        )
                panels.append({"title": f"{metric} model {side}", "series": series})
        build_line_chart_svg(
            panels,
            topks,
            FIGURES_DIR / f"metric_vs_topk_D{dict_size}.svg",
            f"Post-Reconstruction Metrics vs TopK (dict_size={dict_size})",
            "TopK",
        )

        panels = []
        for metric in metric_bases:
            for side in ["A", "B"]:
                series = []
                for regime in regimes:
                    regime_rows = [row for row in subset if row["regime"] == regime]
                    for topk in topks:
                        line_rows = sorted(
                            [row for row in regime_rows if row["k"] == topk],
                            key=lambda x: x["exclusive_pct"],
                        )
                        if not line_rows:
                            continue
                        label = f"{regime}, k={topk}"
                        series.append(
                            {
                                "label": label,
                                "points": [
                                    (row["exclusive_pct"] * 100, float(row[f"{metric}_{side}_post"]))
                                    for row in line_rows
                                ],
                            }
                        )
                panels.append({"title": f"{metric} model {side}", "series": series})
        build_line_chart_svg(
            panels,
            [x * 100 for x in excls],
            FIGURES_DIR / f"metric_vs_exclusive_pct_D{dict_size}.svg",
            f"Post-Reconstruction Metrics vs Exclusive % (dict_size={dict_size})",
            "Exclusive %",
        )


def make_baseline_comparison_plot(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["dict_size"])].append(row)

    averaged = []
    metrics = ["overall_score", "format_accuracy", "tool_correctness"]
    for key in sorted(grouped):
        regime, dict_size = key
        out = {"regime": regime, "dict_size": dict_size}
        for metric in metrics:
            for side in ["A", "B"]:
                vals = [float(row[f"{metric}_{side}_post"]) for row in grouped[key]]
                out[f"{metric}_{side}_post"] = mean(vals)
        averaged.append(out)

    build_bar_chart_svg(
        averaged,
        FIGURES_DIR / "crosscoder_vs_dfc_regimes.svg",
        "Average Post-Reconstruction Metrics by Regime and Dictionary Size",
    )


def get_summary_value(summary_rows: list[dict], **filters) -> dict | None:
    for row in summary_rows:
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    return None


def metric_trend_text(topk_summary: list[dict], exclusive_summary: list[dict], metric: str) -> tuple[str, str]:
    topk = get_summary_value(topk_summary, metric=metric, comparison="45->160")
    if topk:
        topk_text = f"Matched {metric}: average change from TopK 45 to 160 was {topk['mean']:+.3f}."
    else:
        topk_text = f"Matched {metric}: insufficient complete TopK pairs."

    rows = [row for row in exclusive_summary if row["metric"] == metric]
    if not rows:
        excl_text = f"Matched {metric} vs baseline: no DFC-to-baseline matches found."
    else:
        best = max(rows, key=lambda x: x["mean"])
        worst = min(rows, key=lambda x: x["mean"])
        excl_text = (
            f"Matched vs CrossCoder baseline, the best DFC setting was {best['regime']} at "
            f"{int(round(best['exclusive_pct'] * 100))}% exclusive ({best['mean']:+.3f}), while the worst was "
            f"{worst['regime']} at {int(round(worst['exclusive_pct'] * 100))}% ({worst['mean']:+.3f})."
        )
    return topk_text, excl_text


def write_summary(
    rows: list[dict],
    metric_columns: dict[str, list[str]],
    topk_summary: list[dict],
    exclusive_summary: list[dict],
    steering_summary: list[dict],
) -> None:
    post_metrics = metric_columns["post"]
    metric_bases = sorted({metric_base_and_side(col)[0] for col in post_metrics})

    overall_a_topk, overall_a_excl = metric_trend_text(topk_summary, exclusive_summary, "overall_score_A_post")
    overall_b_topk, overall_b_excl = metric_trend_text(topk_summary, exclusive_summary, "overall_score_B_post")
    format_a_topk, format_a_excl = metric_trend_text(topk_summary, exclusive_summary, "format_accuracy_A_post")
    tool_a_topk, tool_a_excl = metric_trend_text(topk_summary, exclusive_summary, "tool_correctness_A_post")

    dict_sizes = sorted({row["dict_size"] for row in rows})
    topks = sorted({row["k"] for row in rows})
    excls = sorted({row["exclusive_pct"] for row in rows})
    regimes = sorted({row["regime"] for row in rows})

    robustness_note = (
        "The conclusions are descriptive rather than causal. The sweep has 47 variants, not a perfectly complete "
        "factorial design, and `dict_size` plus sparsity regime confound raw averages. The matched pair tables reduce "
        "that by comparing only within shared `(dict_size, exclusive_pct, regime)` for TopK effects and within shared "
        "`(dict_size, TopK, uses_l1)` for DFC-vs-baseline effects."
    )

    summary_lines = [
        "# X-Coder Hyperparameter Analysis",
        "",
        "## Files Used",
        "",
        "- `results_full.jsonl`: full post-reconstruction sweep results and the primary analysis input.",
        "- `results_steer.jsonl`: steering-only results, inspected for context but not centered in this analysis.",
        "- `sweep_eval.py`: source of truth for evaluation modes, metric naming, and the `exclusive_pct == 0` CrossCoder baseline convention.",
        "",
        "## Sweep Dimensions",
        "",
        f"- `dict_size`: {', '.join(str(x) for x in dict_sizes)}",
        f"- `TopK`: {', '.join(str(x) for x in topks)}",
        f"- `exclusive_pct`: {', '.join(f'{int(round(x * 100))}%' for x in excls)}",
        f"- `regimes`: {', '.join(regimes)}",
        f"- `n_model_variants`: {len(rows)}",
        "",
        "## Main Post-Reconstruction Metrics",
        "",
        f"The script auto-selected these `*_post` fields: {', '.join(f'`{m}`' for m in post_metrics)}.",
        "",
        "Metric families:",
        "",
        *[f"- `{metric}`" for metric in metric_bases],
        "",
        "## Main Trends",
        "",
        f"- {overall_a_topk}",
        f"- {format_a_topk}",
        f"- {tool_a_topk}",
        f"- {overall_b_topk}",
        f"- {overall_a_excl}",
        f"- {format_a_excl}",
        f"- {tool_a_excl}",
        f"- {overall_b_excl}",
        "",
        "## Interpretation",
        "",
        "- Higher TopK does not show a universal monotonic benefit. The controlled pairwise summaries are the cleanest answer to whether larger TopK helps.",
        "- Nonzero exclusive partitions sometimes beat the 0-exclusive CrossCoder baseline, but the sign and size depend heavily on dictionary size and whether exclusive features are regularized.",
        "- Model A is the clearer signal. Model B remains much weaker overall, so its post metrics move less and are noisier to interpret.",
        "",
        "## Caveats",
        "",
        f"- {robustness_note}",
        f"- The steering file was inspected only for context and summarized into {len(steering_summary)} grouped rows; it was not used as the main evidence for reconstruction conclusions.",
        "- `layer`, `steps`, `lr`, and `train_batch` are constant in this sweep, so the main practical confounders are `dict_size`, shared L1 usage, and whether exclusive features are free or regularized.",
        "- `format_accuracy_B_post` is zero for most variants, so it has limited discriminative power.",
        "",
        "## Artifacts",
        "",
        "- Figures: `analysis/figures/`",
        "- Tables: `analysis/tables/`",
        "- Repro script: `analysis/run_xcoder_hparams_analysis.py`",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    ensure_dirs()
    rows = validate_and_flatten(load_jsonl(FULL_RESULTS))
    metric_columns = identify_metric_columns(rows)
    tidy = tidy_metrics(rows, metric_columns)

    write_csv(TABLES_DIR / "model_variants_flat.csv", rows)
    write_csv(TABLES_DIR / "metrics_tidy.csv", tidy)
    write_csv(
        TABLES_DIR / "schema_summary.csv",
        [
            {"quantity": "n_model_variants", "value": len(rows)},
            {"quantity": "dict_sizes", "value": ", ".join(str(x) for x in sorted({row['dict_size'] for row in rows}))},
            {"quantity": "topk_values", "value": ", ".join(str(x) for x in sorted({row['k'] for row in rows}))},
            {
                "quantity": "exclusive_pct_values",
                "value": ", ".join(f"{int(round(x * 100))}%" for x in sorted({row['exclusive_pct'] for row in rows})),
            },
            {"quantity": "regimes", "value": ", ".join(sorted({row['regime'] for row in rows}))},
            {"quantity": "post_metrics", "value": ", ".join(metric_columns["post"])},
        ],
        fieldnames=["quantity", "value"],
    )

    post_by_topk = grouped_post_summary(tidy, ["topk"])
    post_by_excl = grouped_post_summary(tidy, ["exclusive_pct"])
    post_by_dict_regime_topk = grouped_post_summary(tidy, ["dict_size", "regime", "topk"])
    post_by_dict_regime_excl = grouped_post_summary(tidy, ["dict_size", "regime", "exclusive_pct"])
    write_csv(TABLES_DIR / "post_summary_by_topk.csv", post_by_topk)
    write_csv(TABLES_DIR / "post_summary_by_exclusive_pct.csv", post_by_excl)
    write_csv(TABLES_DIR / "post_summary_by_dict_regime_topk.csv", post_by_dict_regime_topk)
    write_csv(TABLES_DIR / "post_summary_by_dict_regime_exclusive_pct.csv", post_by_dict_regime_excl)

    topk_raw, topk_summary = compute_controlled_topk_deltas(rows, metric_columns["post"])
    excl_raw, excl_summary = compute_controlled_exclusive_vs_baseline(rows, metric_columns["post"])
    steer_summary = inspect_steering_context()
    write_csv(TABLES_DIR / "controlled_topk_pairwise_raw.csv", topk_raw)
    write_csv(TABLES_DIR / "controlled_topk_pairwise_summary.csv", topk_summary)
    write_csv(TABLES_DIR / "controlled_exclusive_vs_baseline_raw.csv", excl_raw)
    write_csv(TABLES_DIR / "controlled_exclusive_vs_baseline_summary.csv", excl_summary)
    write_csv(TABLES_DIR / "steering_context_summary.csv", steer_summary)

    metric_bases = sorted({metric_base_and_side(col)[0] for col in metric_columns["post"]})
    make_line_plots(rows, metric_bases)
    make_baseline_comparison_plot(rows)
    write_summary(rows, metric_columns, topk_summary, excl_summary, steer_summary)

    print("Loaded full results rows:", len(rows))
    print("Selected post metrics:", ", ".join(metric_columns["post"]))
    print("Regimes:", ", ".join(sorted({row['regime'] for row in rows})))
    print("Wrote tables to:", TABLES_DIR)
    print("Wrote figures to:", FIGURES_DIR)
    print("Wrote summary to:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
