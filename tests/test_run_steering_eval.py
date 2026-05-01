"""Tests for run_steering_eval.py — deterministic helpers only."""
from __future__ import annotations

import json
from pathlib import Path

from run_steering_eval import (
    N_PROMPTS,
    SWEEP_ALPHAS,
    SWEEP_KS,
    cell_path,
    is_cell_complete,
)


def test_sweep_grid_matches_spec():
    assert SWEEP_KS    == [1, 2, 4, 8, 16, 32, 64, 100]
    assert SWEEP_ALPHAS == [0, 1, 6, 16, 32, 64]
    assert N_PROMPTS    == 100


def test_cell_path_format(tmp_path):
    p = cell_path(tmp_path, k_pct=8, alpha=16)
    assert p.name == "k08_a16.jsonl"
    p2 = cell_path(tmp_path, k_pct=100, alpha=0)
    assert p2.name == "k100_a00.jsonl"
    p3 = cell_path(tmp_path, k_pct=1, alpha=64)
    assert p3.name == "k01_a64.jsonl"


def test_is_cell_complete_missing(tmp_path):
    p = tmp_path / "k01_a01.jsonl"
    assert not is_cell_complete(p, expected_n=100)


def test_is_cell_complete_partial(tmp_path):
    p = tmp_path / "k01_a01.jsonl"
    with open(p, "w") as f:
        for i in range(50):
            f.write(json.dumps({"prompt_index": i}) + "\n")
    assert not is_cell_complete(p, expected_n=100)


def test_is_cell_complete_full(tmp_path):
    p = tmp_path / "k01_a01.jsonl"
    with open(p, "w") as f:
        for i in range(100):
            f.write(json.dumps({"prompt_index": i}) + "\n")
    assert is_cell_complete(p, expected_n=100)
