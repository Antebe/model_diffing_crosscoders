"""Tests for build_steering_figures aggregation helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_steering_figures import aggregate_model_dir, grid_from_rows


def _write_cell(path: Path, *, k_pct: float, alpha: float, n: int,
                steered_tool_pct: float, clean_tool_pct: float = 20.0,
                recon_tool_pct: float = 50.0):
    """Write n synthetic per-prompt rows so aggregation produces target percentages."""
    n_steered_correct = round(steered_tool_pct / 100.0 * n)
    n_clean_correct   = round(clean_tool_pct   / 100.0 * n)
    n_recon_correct   = round(recon_tool_pct   / 100.0 * n)

    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "prompt_index": i,
                "k_pct": k_pct, "alpha": alpha, "subset_size": 4,
                "clean_score":   {
                    "tool_correctness": int(i < n_clean_correct),
                    "format_accuracy": 1, "overall_score": 1.0,
                },
                "recon_score":   {
                    "tool_correctness": int(i < n_recon_correct),
                    "format_accuracy": 1, "overall_score": 1.5,
                },
                "steered_score": {
                    "tool_correctness": int(i < n_steered_correct),
                    "format_accuracy": 1, "overall_score": 1.5,
                },
            }) + "\n")


def test_aggregate_single_cell(tmp_path: Path):
    p = tmp_path / "k08_a16.jsonl"
    _write_cell(p, k_pct=8.0, alpha=16.0, n=10,
                steered_tool_pct=70.0, clean_tool_pct=20.0, recon_tool_pct=50.0)
    rows = aggregate_model_dir(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["k_pct"] == 8.0
    assert r["alpha"] == 16.0
    assert r["n"] == 10
    assert r["steered_tool_pct"] == 70.0
    assert r["clean_tool_pct"] == 20.0
    assert r["recon_tool_pct"] == 50.0
    assert r["d_steered_clean_tool"] == 50.0
    assert r["d_steered_recon_tool"] == 20.0


def test_aggregate_skips_error_rows(tmp_path: Path):
    p = tmp_path / "k01_a01.jsonl"
    with p.open("w") as f:
        for i in range(3):
            f.write(json.dumps({
                "prompt_index": i, "k_pct": 1.0, "alpha": 1.0, "subset_size": 1,
                "clean_score":   {"tool_correctness": 1, "format_accuracy": 1, "overall_score": 1.0},
                "recon_score":   {"tool_correctness": 1, "format_accuracy": 1, "overall_score": 1.0},
                "steered_score": {"tool_correctness": 1, "format_accuracy": 1, "overall_score": 1.0},
            }) + "\n")
        f.write(json.dumps({"error": "boom", "prompt_index": 99,
                            "k_pct": 1.0, "alpha": 1.0}) + "\n")
    rows = aggregate_model_dir(p.parent)
    assert len(rows) == 1
    assert rows[0]["n"] == 3            # the error row is excluded


def test_grid_from_rows_layout():
    rows = [
        {"k_pct": 1.0,  "alpha": 0.0, "v": 1.0},
        {"k_pct": 1.0,  "alpha": 1.0, "v": 2.0},
        {"k_pct": 32.0, "alpha": 1.0, "v": 5.0},
        # (32, 0) is missing → NaN
    ]
    grid = grid_from_rows(rows, "v", k_pcts=[1.0, 32.0], alphas=[0.0, 1.0])
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 1.0
    assert grid[0, 1] == 2.0
    assert grid[1, 1] == 5.0
    assert np.isnan(grid[1, 0])
