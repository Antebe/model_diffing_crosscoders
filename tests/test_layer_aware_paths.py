"""Tests for the layer-aware path helpers introduced in the refactor.

Covers:
  - raw_cache_path / feature_cache_path string layout
  - layer_from_hparams happy path + error cases
  - Config.fineweb_cache / Config.toolrl_cache property semantics
    (in particular: re-derive when cfg.layer is reassigned)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import (
    Config,
    feature_cache_path,
    layer_from_hparams,
    raw_cache_path,
)


class TestRawCachePath:
    def test_default_root(self):
        assert raw_cache_path("fineweb", 13) == "./cache/fineweb_l13"
        assert raw_cache_path("toolrl", 0) == "./cache/toolrl_l0"
        assert raw_cache_path("toolrl", 35) == "./cache/toolrl_l35"

    def test_custom_root(self):
        assert raw_cache_path("fineweb", 9, root="/tmp/x") == "/tmp/x/fineweb_l9"


class TestFeatureCachePath:
    def test_model_tagged(self):
        # Feature caches are always model-tagged; no _l<L> suffix because the
        # short name already encodes layer (e.g. "...-l9").
        assert (
            feature_cache_path("toolrl", "dfc-D8k-excl10-k45-l9")
            == "./cache/dfc-D8k-excl10-k45-l9_features_toolrl"
        )

    def test_custom_root(self):
        assert (
            feature_cache_path("fineweb", "abc", root="/scratch")
            == "/scratch/abc_features_fineweb"
        )


class TestLayerFromHparams:
    def test_reads_layer(self, tmp_path: Path):
        hp = tmp_path / "hparams.json"
        hp.write_text(json.dumps({"layer": 14, "dict_size": 8192}))
        assert layer_from_hparams(hp) == 14

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not found"):
            layer_from_hparams(tmp_path / "nope.json")

    def test_missing_layer_field(self, tmp_path: Path):
        hp = tmp_path / "hparams.json"
        hp.write_text(json.dumps({"dict_size": 8192}))
        with pytest.raises(ValueError, match="no 'layer' field"):
            layer_from_hparams(hp)


class TestConfigCacheProperties:
    def test_default_layer_13(self):
        c = Config()
        assert c.fineweb_cache == "./cache/fineweb_l13"
        assert c.toolrl_cache == "./cache/toolrl_l13"

    def test_layer_reassignment_re_derives_paths(self):
        # The whole point of properties: cfg.layer = L → caches follow.
        c = Config()
        c.layer = 32
        assert c.fineweb_cache == "./cache/fineweb_l32"
        assert c.toolrl_cache == "./cache/toolrl_l32"

    def test_custom_cache_dir(self, tmp_path):
        c = Config(cache_dir=str(tmp_path))
        c.layer = 5
        assert c.fineweb_cache == f"{tmp_path}/fineweb_l5"
        assert c.toolrl_cache == f"{tmp_path}/toolrl_l5"
