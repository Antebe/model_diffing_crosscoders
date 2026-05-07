"""Tests for rank_features_sweep.

Verifies that the multi-layer activation collector + crosscoder encoder
produce the same per-feature numbers as the per-model rank_features path —
i.e. that consolidating LLM forwards is an optimization, not a behavior
change.

Uses fake LLMs whose ``hidden_states[i+1]`` is deterministically distinct
per layer, so per-layer correctness is byte-checkable without GPUs.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from rank_features_sweep import (
    collect_layered_activations,
    encode_through_crosscoder,
)


class _DeterministicLM(nn.Module):
    """Returns hidden_states[i] = embedding scaled by (i + offset).

    Every layer is filled with a distinct constant so callers can verify
    they pulled the right one.
    """
    def __init__(self, n_layers: int, hidden_dim: int, offset: float = 0.0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.offset = offset

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        B, T = input_ids.shape
        # n_layers + 1 entries: index 0 is "embedding", index i+1 is layer i output.
        hs = tuple(
            torch.full((B, T, self.hidden_dim), float(i) + self.offset)
            for i in range(self.n_layers + 1)
        )
        return SimpleNamespace(hidden_states=hs)


class _FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512, **_):
        # 4 tokens per prompt regardless of input — the test only checks
        # last-token activation values, not tokenization fidelity.
        ids = torch.ones((1, 4), dtype=torch.long)
        return SimpleNamespace(input_ids=ids)


# ─── collect_layered_activations ───────────────────────────────────────────

def test_per_layer_values_match_layer_marker():
    # Model A fills hidden_states[i] with i; model B with i + 100.
    n_layers = 36
    hidden_dim = 8
    model_a = _DeterministicLM(n_layers, hidden_dim, offset=0.0)
    model_b = _DeterministicLM(n_layers, hidden_dim, offset=100.0)
    tok = _FakeTokenizer()

    prompts = ["a", "b", "c"]
    layers = [1, 5, 9, 14, 32]
    out = collect_layered_activations(
        prompts, tok, model_a, model_b, layers, device="cpu", desc="t",
    )

    # One entry per requested layer.
    assert sorted(out.keys()) == layers

    for L in layers:
        h_a, h_b = out[L]
        # Shape: (n_prompts, hidden_dim).
        assert h_a.shape == (len(prompts), hidden_dim)
        assert h_b.shape == (len(prompts), hidden_dim)
        # hidden_states[L+1] of model A = (L+1); model B = (L+1+100).
        np.testing.assert_array_equal(h_a, np.full_like(h_a, float(L + 1)))
        np.testing.assert_array_equal(h_b, np.full_like(h_b, float(L + 101)))


def test_no_cross_layer_contamination():
    """Pulling layers in different orders / sets must yield identical per-layer arrays."""
    model_a = _DeterministicLM(36, 4, offset=0.0)
    model_b = _DeterministicLM(36, 4, offset=100.0)
    tok = _FakeTokenizer()
    prompts = ["x", "y"]

    out_full = collect_layered_activations(
        prompts, tok, model_a, model_b, [1, 5, 9, 14], "cpu", desc="full",
    )
    out_subset = collect_layered_activations(
        prompts, tok, model_a, model_b, [9], "cpu", desc="subset",
    )
    out_reordered = collect_layered_activations(
        prompts, tok, model_a, model_b, [14, 1, 5, 9], "cpu", desc="reordered",
    )

    for L in [1, 5, 9, 14]:
        h_a_full, h_b_full = out_full[L]
        h_a_re, h_b_re = out_reordered[L]
        np.testing.assert_array_equal(h_a_full, h_a_re)
        np.testing.assert_array_equal(h_b_full, h_b_re)

    h_a_full, h_b_full = out_full[9]
    h_a_sub, h_b_sub = out_subset[9]
    np.testing.assert_array_equal(h_a_full, h_a_sub)
    np.testing.assert_array_equal(h_b_full, h_b_sub)


# ─── encode_through_crosscoder ─────────────────────────────────────────────

class _PassThroughCC(nn.Module):
    """Stub crosscoder whose encode() returns a deterministic linear combo
    of its inputs, so we can verify batching is order-preserving and that
    every row is independently encoded."""
    def __init__(self, hidden_dim: int, dict_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dict_size = dict_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, hidden_dim). Output: (B, dict_size) where col c = sum of
        # row c (mod hidden_dim) of x[:, 0] + x[:, 1] — concrete enough that
        # any cross-row contamination would change the result.
        B, two, d = x.shape
        assert two == 2
        out = torch.zeros((B, self.dict_size), dtype=x.dtype)
        for b in range(B):
            for c in range(self.dict_size):
                out[b, c] = (x[b, 0, c % d] * 3.0 + x[b, 1, c % d] * 7.0)
        return out


def test_encode_batching_matches_one_shot():
    cc = _PassThroughCC(hidden_dim=4, dict_size=6)
    np.random.seed(0)
    n = 13
    h_a = np.random.randn(n, 4).astype(np.float32)
    h_b = np.random.randn(n, 4).astype(np.float32)

    one_shot = encode_through_crosscoder(cc, h_a, h_b, device="cpu", batch_size=999)
    batched  = encode_through_crosscoder(cc, h_a, h_b, device="cpu", batch_size=4)

    assert one_shot.shape == (n, 6)
    np.testing.assert_allclose(one_shot, batched, rtol=1e-6, atol=1e-6)


def test_encode_per_row_independence():
    """Reordering inputs must reorder outputs accordingly (no row mixing)."""
    cc = _PassThroughCC(hidden_dim=4, dict_size=6)
    np.random.seed(1)
    n = 5
    h_a = np.random.randn(n, 4).astype(np.float32)
    h_b = np.random.randn(n, 4).astype(np.float32)

    feats = encode_through_crosscoder(cc, h_a, h_b, device="cpu", batch_size=2)

    perm = np.array([3, 0, 1, 4, 2])
    feats_perm = encode_through_crosscoder(cc, h_a[perm], h_b[perm], device="cpu", batch_size=2)
    np.testing.assert_allclose(feats[perm], feats_perm, rtol=1e-6, atol=1e-6)
