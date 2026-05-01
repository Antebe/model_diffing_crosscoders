"""Tests for steering/steer.py — Model A post-top-k targeted steering."""
from __future__ import annotations

import pytest
import torch

from sweep_eval import DFCCrossCoder
from steering.steer import (
    compute_steering_delta,
    select_top_subset,
)


@pytest.fixture
def dfc():
    """Tiny DFC for fast tests: dict_size=20, n_a=4, n_b=4, n_shared=12."""
    torch.manual_seed(0)
    cc = DFCCrossCoder(
        activation_dim=8,
        dict_size=20,
        k=5,
        model_a_exclusive_pct=0.20,  # n_a = 4
        model_b_exclusive_pct=0.20,  # n_b = 4
    ).eval()
    assert cc.a_end == 4
    assert cc.b_end == 8
    return cc


@pytest.fixture
def features(dfc):
    """A sparse top-k feature vector with known A-exclusive activations."""
    f = torch.zeros(1, dfc.dict_size)
    f[0, 0] = 3.0   # A-excl, will be in subset
    f[0, 1] = 2.0   # A-excl, will be in subset
    f[0, 2] = 1.0   # A-excl, NOT in subset
    f[0, 10] = 5.0  # shared, irrelevant for this test
    return f


def test_alpha_one_is_noop(dfc, features):
    """α=1 ⇒ (α−1)=0 ⇒ delta must be the zero vector."""
    delta = compute_steering_delta(
        features=features, subset_indices=[0, 1], alpha=1.0, dfc=dfc, model_idx=0
    )
    assert delta.shape == (dfc.activation_dim,)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_empty_subset_is_noop(dfc, features):
    """No selected features ⇒ delta is zero regardless of α."""
    delta = compute_steering_delta(
        features=features, subset_indices=[], alpha=10.0, dfc=dfc, model_idx=0
    )
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_alpha_zero_ablates_subset(dfc, features):
    """α=0 ⇒ delta must equal negative of subset's Model-A recon contribution."""
    subset = [0, 1]
    delta = compute_steering_delta(
        features=features, subset_indices=subset, alpha=0.0, dfc=dfc, model_idx=0
    )
    # Expected: -1 * sum_{i in subset} features[0,i] * W_dec[i, 0, :]
    expected = torch.zeros(dfc.activation_dim)
    for i in subset:
        expected -= features[0, i].item() * dfc.W_dec[i, 0, :].detach()
    assert torch.allclose(delta, expected, atol=1e-6)


def test_single_feature_direction(dfc, features):
    """With one feature i, delta points along (α−1) · features[i] · W_dec[i, 0]."""
    i = 0
    alpha = 4.0
    delta = compute_steering_delta(
        features=features, subset_indices=[i], alpha=alpha, dfc=dfc, model_idx=0
    )
    expected = (alpha - 1.0) * features[0, i].item() * dfc.W_dec[i, 0, :].detach()
    assert torch.allclose(delta, expected, atol=1e-6)


def test_inactive_feature_contributes_nothing(dfc, features):
    """Subset members with features[i]==0 should not move delta — post-top-k semantics."""
    delta_with = compute_steering_delta(
        features=features, subset_indices=[0, 3], alpha=5.0, dfc=dfc, model_idx=0
    )
    delta_without = compute_steering_delta(
        features=features, subset_indices=[0], alpha=5.0, dfc=dfc, model_idx=0
    )
    assert torch.allclose(delta_with, delta_without, atol=1e-6)


def test_select_top_subset_basic():
    """select_top_subset reads ranking CSV and returns top k% feature indices."""
    import pandas as pd
    df = pd.DataFrame(
        {
            "rank":            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "feature_idx":     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            "cohens_d":        [1.0, 0.9, 0.8, 0.7, 0.6, -0.5, 0.4, 0.3, 0.2, 0.1],
            "sign":            ["tool"] * 5 + ["nontool"] + ["tool"] * 4,
        }
    )
    # 9 tool-signed rows; 20% → ceil(0.2*9) = 2 → top 2 by cohens_d among tool rows
    subset = select_top_subset(df, k_pct=20.0, n_a=10)
    assert subset == [9, 8]


def test_select_top_subset_excludes_indices_outside_a_excl():
    """tool-signed rows whose feature_idx >= n_a are dropped."""
    import pandas as pd
    df = pd.DataFrame(
        {
            "rank":         [0, 1, 2],
            "feature_idx":  [3, 7, 1],   # 7 is outside a_excl when n_a=4
            "cohens_d":     [1.0, 0.9, 0.5],
            "sign":         ["tool", "tool", "tool"],
        }
    )
    subset = select_top_subset(df, k_pct=100.0, n_a=4)
    assert 7 not in subset
    assert set(subset) == {3, 1}


def test_select_top_subset_kpct_100_returns_all_tool():
    import pandas as pd
    df = pd.DataFrame(
        {
            "rank":        [0, 1, 2, 3],
            "feature_idx": [0, 1, 2, 3],
            "cohens_d":    [1.0, 0.9, 0.8, 0.7],
            "sign":        ["tool", "tool", "nontool", "tool"],
        }
    )
    subset = select_top_subset(df, k_pct=100.0, n_a=4)
    assert set(subset) == {0, 1, 3}
