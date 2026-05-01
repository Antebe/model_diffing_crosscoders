"""
summarize_steering.py
─────────────────────
Roll up per-cell JSONLs from results/targeted_steering/<model>/k*_a*.jsonl
into a single summary CSV.

Output columns:
    model, k_pct, alpha, n,
    mean_clean_overall, mean_clean_format, mean_clean_tool,
    mean_recon_overall, mean_recon_format, mean_recon_tool,
    mean_steered_overall, mean_steered_format, mean_steered_tool,
    delta_overall_vs_clean, delta_overall_vs_recon,
    delta_tool_vs_clean,    delta_tool_vs_recon,
    delta_format_vs_clean,  delta_format_vs_recon

Usage:
    python summarize_steering.py \
        --steering-root results/targeted_steering \
        --out results/targeted_steering/summary.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

CELL_RE = re.compile(r"^k(\d+)_a(\d+)\.jsonl$")


def _collect_cell(path: Path) -> dict:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            rows.append(r)
    if not rows:
        return dict(n=0)
    df = pd.DataFrame(rows)
    out = dict(n=len(df))
    for which in ("clean", "recon", "steered"):
        scores = pd.json_normalize(df[f"{which}_score"])
        out[f"mean_{which}_overall"] = float(scores["overall_score"].mean())
        out[f"mean_{which}_format"] = float(
            scores["format_accuracy"].astype(float).mean()
        )
        out[f"mean_{which}_tool"] = float(
            scores["tool_correctness"].astype(float).mean()
        )
    for metric in ("overall", "format", "tool"):
        out[f"delta_{metric}_vs_clean"] = (
            out[f"mean_steered_{metric}"] - out[f"mean_clean_{metric}"]
        )
        out[f"delta_{metric}_vs_recon"] = (
            out[f"mean_steered_{metric}"] - out[f"mean_recon_{metric}"]
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering-root", default="results/targeted_steering")
    parser.add_argument("--out", default="results/targeted_steering/summary.csv")
    args = parser.parse_args()

    root = Path(args.steering_root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    rows: list[dict] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cell in sorted(model_dir.iterdir()):
            m = CELL_RE.match(cell.name)
            if not m:
                continue
            k_pct = int(m.group(1))
            alpha = int(m.group(2))
            stats = _collect_cell(cell)
            row = dict(model=model_dir.name, k_pct=k_pct, alpha=alpha, **stats)
            rows.append(row)

    if not rows:
        raise SystemExit(f"No cell files found under {root}")

    df = (
        pd.DataFrame(rows)
        .sort_values(["model", "k_pct", "alpha"])
        .reset_index(drop=True)
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} cell summaries to {args.out}")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
