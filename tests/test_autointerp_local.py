"""Tests for autointerp_local + run_autointerp_local — pure-Python helpers
only (no model loads, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_autointerp_local import select_features


@pytest.fixture
def feat_cache_meta_dfc() -> dict:
    return {"dict_size": 100, "n_a": 10, "n_b": 10, "n_shared": 80}


@pytest.fixture
def feat_cache_meta_cc() -> dict:
    return {"dict_size": 100, "n_a": 0, "n_b": 0, "n_shared": 100}


def test_select_partition_a_excl(feat_cache_meta_dfc):
    out = select_features(
        feat_cache_meta_dfc, rankings_csv=None, partition="a_excl",
        fire_rate_floor=0.0,
    )
    assert out == list(range(10))


def test_select_partition_b_excl(feat_cache_meta_dfc):
    out = select_features(
        feat_cache_meta_dfc, rankings_csv=None, partition="b_excl",
        fire_rate_floor=0.0,
    )
    assert out == list(range(10, 20))


def test_select_partition_shared(feat_cache_meta_dfc):
    out = select_features(
        feat_cache_meta_dfc, rankings_csv=None, partition="shared",
        fire_rate_floor=0.0,
    )
    assert out[0] == 20
    assert out[-1] == 99


def test_select_partition_all(feat_cache_meta_dfc):
    out = select_features(
        feat_cache_meta_dfc, rankings_csv=None, partition="all",
        fire_rate_floor=0.0,
    )
    assert len(out) == 100


def test_select_unknown_partition_raises(feat_cache_meta_dfc):
    with pytest.raises(ValueError, match="unknown partition"):
        select_features(
            feat_cache_meta_dfc, rankings_csv=None, partition="bogus",
            fire_rate_floor=0.0,
        )


def test_fire_rate_floor_filters_out_dead_features(tmp_path, feat_cache_meta_dfc):
    """Features below the floor are excluded; above the floor stay."""
    csv_path = tmp_path / "rank.csv"
    csv_path.write_text(
        "rank,feature_idx,cohens_d,auroc,fire_rate_tool,fire_rate_nontool,diff,mean_tool,mean_nontool,sign\n"
        "0,0,1.0,0.9,0.5,0.0,1.0,1.0,0.0,tool\n"   # alive
        "1,1,0.9,0.8,0.001,0.0,0.5,0.5,0.0,tool\n" # below floor
        "2,2,0.8,0.7,0.5,0.0,1.0,1.0,0.0,tool\n"   # alive
        "3,3,0.7,0.6,0.0,0.0,0.0,0.0,0.0,tool\n"   # below floor
        "4,4,0.6,0.5,0.5,0.5,0.0,0.5,0.5,tool\n"   # alive
    )
    out = select_features(
        feat_cache_meta_dfc, rankings_csv=csv_path, partition="a_excl",
        fire_rate_floor=0.005,
    )
    assert set(out) == {0, 2, 4}


def test_local_gemma_client_call_signature():
    """LocalGemmaClient.call should be an async coroutine accepting (system, user)."""
    import inspect
    from autointerp_local import LocalGemmaClient
    assert inspect.iscoroutinefunction(LocalGemmaClient.call)
    sig = inspect.signature(LocalGemmaClient.call)
    assert list(sig.parameters) == ["self", "system", "user"]
