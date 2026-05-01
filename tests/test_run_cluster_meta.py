"""Tests for run_cluster_meta — pure-python helpers."""
from __future__ import annotations

import json
from pathlib import Path

from run_cluster_meta import (
    _truncate,
    load_assignments,
    load_explanations,
    parse_meta,
)


def test_truncate_short_string_unchanged():
    assert _truncate("hello world") == "hello world"


def test_truncate_long_string_appends_ellipsis():
    s = "x" * 300
    out = _truncate(s, n=50)
    assert len(out) == 51                # 50 chars + "…"
    assert out.endswith("…")


def test_truncate_strips_newlines():
    out = _truncate("line1\nline2\n line3")
    assert "\n" not in out


def test_parse_meta_well_formed():
    raw = "[CLUSTER_NAME]: tool argument binding\n[CLUSTER_SUMMARY]: features track JSON arg fields"
    name, summary = parse_meta(raw)
    assert name == "tool argument binding"
    assert summary == "features track JSON arg fields"


def test_parse_meta_falls_back_to_first_line():
    raw = "totally malformed response from the model"
    name, summary = parse_meta(raw)
    assert name.startswith("totally malformed")
    assert summary == "(no summary parsed)"


def test_parse_meta_empty_input():
    name, summary = parse_meta("")
    assert name == "(unnamed)"
    assert summary == "(no summary parsed)"


def test_load_assignments_basic(tmp_path: Path):
    csv = tmp_path / "a.csv"
    csv.write_text(
        "feature_idx,cluster_id,umap_x,umap_y\n"
        "0,2,1.5,2.5\n"
        "1,-1,3.0,4.0\n"
        "2,2,1.0,2.0\n"
    )
    out = load_assignments(csv)
    assert out == {0: 2, 1: -1, 2: 2}


def test_load_explanations_skips_dead_and_empty(tmp_path: Path):
    base = tmp_path / "ai" / "0000000"
    base.mkdir(parents=True)
    (base / "feat_0000001.json").write_text(json.dumps({
        "feat_idx": 1, "is_dead": False, "explanation": "JSON tool name binding",
    }))
    (base / "feat_0000002.json").write_text(json.dumps({
        "feat_idx": 2, "is_dead": True, "explanation": "should be skipped",
    }))
    (base / "feat_0000003.json").write_text(json.dumps({
        "feat_idx": 3, "is_dead": False, "explanation": "   ",
    }))
    out = load_explanations(tmp_path / "ai")
    assert out == {1: "JSON tool name binding"}
