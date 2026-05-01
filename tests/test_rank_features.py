"""Tests for rank_features.py metric helpers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from rank_features import (
    auroc,
    cohens_d,
    fire_rate,
    rank_features,
)


def test_cohens_d_known_value():
    """Means differ by 1.0, equal variance ⇒ d = -1.0 (since a's mean < b's)."""
    a = np.array([0.0, 1.0, 2.0])  # mean 1.0, var 1.0
    b = np.array([1.0, 2.0, 3.0])  # mean 2.0, var 1.0
    assert math.isclose(cohens_d(a, b), -1.0, abs_tol=1e-6)


def test_cohens_d_zero_when_means_equal():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([2.0, 1.0, 0.0])
    assert abs(cohens_d(a, b)) < 1e-9


def test_cohens_d_handles_zero_std():
    """Both groups constant and equal ⇒ 0."""
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    assert cohens_d(a, b) == 0.0


def test_auroc_perfect_separation():
    pos = np.array([1.0, 2.0, 3.0])
    neg = np.array([-1.0, -2.0, -3.0])
    assert auroc(pos, neg) == 1.0


def test_auroc_in_unit_interval():
    pos = np.array([1.0, 2.0])
    neg = np.array([1.5, 2.5])
    assert 0.0 <= auroc(pos, neg) <= 1.0


def test_fire_rate_basic():
    acts = np.array([0.0, 0.5, 0.0, 1.0])
    assert fire_rate(acts) == 0.5
    assert fire_rate(acts, threshold=0.6) == 0.25


def test_rank_features_writes_expected_columns():
    """rank_features returns a dataframe with the zip's CSV column schema."""
    rng = np.random.default_rng(0)
    n_features = 5
    n_tool = 50
    n_nontool = 50
    tool = rng.normal(loc=1.0, scale=1.0, size=(n_tool, n_features))
    tool[:, 0] += 2.0  # feature 0 strongly tool-discriminative
    nontool = rng.normal(loc=0.0, scale=1.0, size=(n_nontool, n_features))

    df = rank_features(tool_acts=tool, nontool_acts=nontool)

    expected_cols = {
        "rank", "feature_idx", "cohens_d", "auroc",
        "fire_rate_tool", "fire_rate_nontool",
        "diff", "mean_tool", "mean_nontool", "sign",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == n_features
    top = df.iloc[0]
    assert top["feature_idx"] == 0
    assert top["sign"] == "tool"
