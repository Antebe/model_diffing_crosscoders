"""Tests for run_umap figure plumbing — no UMAP, no clustering, just IO."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from run_umap import figure_aexcl_clusters, figure_partitions


def test_figure_partitions_writes_png(tmp_path: Path):
    n_a, n_b, dict_size = 5, 5, 30
    coords = np.random.RandomState(0).rand(dict_size, 2)
    out = tmp_path / "umap_partitions.png"
    figure_partitions(coords, n_a, n_b, dict_size, out)
    assert out.exists()
    assert out.stat().st_size > 1024


def test_figure_aexcl_clusters_writes_png(tmp_path: Path):
    coords = np.random.RandomState(0).rand(30, 2)
    labels = np.array([0]*10 + [1]*10 + [-1]*10)
    out = tmp_path / "umap_clusters.png"
    figure_aexcl_clusters(coords, labels, out)
    assert out.exists()
    assert out.stat().st_size > 1024


def test_figure_aexcl_clusters_with_names(tmp_path: Path):
    coords = np.random.RandomState(0).rand(20, 2)
    labels = np.array([0]*10 + [1]*10)
    names = {0: "tool name binding", 1: "JSON formatting"}
    out = tmp_path / "umap_named.png"
    figure_aexcl_clusters(coords, labels, out, cluster_names=names)
    assert out.exists()
