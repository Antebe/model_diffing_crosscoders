"""
steering/steer.py
─────────────────
Targeted-neuron steering on **Model A** (post-top-k delta).

Per spec §4 P1, for a sparse post-top-k feature vector ``features`` and a
subset ``S`` of A-exclusive feature indices selected by discriminative
ranking::

    delta = sum_{i in S} (alpha - 1) * features[i] * W_dec[i, 0, :]
    patched_a = h_a + delta

Because the subset is applied *post* top-k, features that did not survive
top-k for this prompt have ``features[i] == 0`` and therefore do not
contribute — this is precisely the property that distinguishes the
targeted sweep from the §6 pre-top-k uniform scaling, which is a no-op
under top-k SAEs.
"""
from __future__ import annotations

import math
from typing import Sequence

import pandas as pd
import torch

from sweep_eval import CrossCoder, DFCCrossCoder


@torch.no_grad()
def compute_steering_delta(
    features: torch.Tensor,
    subset_indices: Sequence[int],
    alpha: float,
    dfc: "DFCCrossCoder | CrossCoder",
    model_idx: int = 0,
) -> torch.Tensor:
    """Return additive activation-space delta for Model A targeted steering.

    Args:
        features: Sparse top-k feature vector, shape ``(1, dict_size)`` or
            ``(dict_size,)``. Must be on same device as ``dfc``.
        subset_indices: Feature indices to steer. Indices not satisfying
            ``features[i] != 0`` contribute nothing (post-top-k semantics).
        alpha: Multiplicative scale on the selected features' contribution.
            ``alpha=1`` ⇒ no-op (delta = 0). ``alpha=0`` ⇒ ablate subset.
        dfc: Loaded crosscoder. ``W_dec`` shape is ``(dict_size, 2, d)``.
        model_idx: 0 for Model A's decoder, 1 for Model B's. Default 0.

    Returns:
        ``(d,)`` float tensor on the same device/dtype as ``W_dec[:, model_idx]``.
    """
    if features.dim() == 2:
        if features.shape[0] != 1:
            raise ValueError(
                f"features batch dim must be 1, got {features.shape[0]}"
            )
        f = features[0]
    elif features.dim() == 1:
        f = features
    else:
        raise ValueError(f"features must be 1D or 2D, got shape {features.shape}")

    device = dfc.W_dec.device
    d = dfc.activation_dim
    delta = torch.zeros(d, device=device, dtype=dfc.W_dec.dtype)

    if not subset_indices or alpha == 1.0:
        return delta

    coef = alpha - 1.0
    for i in subset_indices:
        val = f[i].item()
        if val == 0.0:
            continue
        delta += coef * val * dfc.W_dec[i, model_idx, :]
    return delta


def select_top_subset(
    rankings_df: pd.DataFrame,
    k_pct: float,
    n_a: int,
    partition: str = "a-excl",
    a_end: int | None = None,
    b_end: int | None = None,
) -> list[int]:
    """Select top-k% feature indices from a ranking dataframe, by partition.

    Args:
        rankings_df: Must have columns ``feature_idx``, ``cohens_d``, ``sign``.
            Same schema as ``tool_neurons_A_full.csv``. For partitions other
            than ``a-excl`` the dataframe must contain rows for the selected
            partition's feature indices.
        k_pct: Percentage of *partition* tool-aligned features to select.
        n_a: A-exclusive partition size (DFC's ``a_end``). Used as the
            default upper bound when ``partition == "a-excl"``. Pass
            ``cc.dict_size`` for CrossCoder variants and partition will be
            forced to "all".
        partition: One of:
            - ``"a-excl"`` (default, back-compat): feature_idx ∈ [0, a_end).
            - ``"shared"``: feature_idx ∈ [b_end, dict_size). Requires
              ``b_end`` to be passed.
            - ``"b-excl"``: feature_idx ∈ [a_end, b_end). Requires both
              ``a_end`` and ``b_end``.
            - ``"all"``: no index filter (CrossCoder / full-dictionary mode).
        a_end: Override for A/B partition boundary. Defaults to ``n_a``.
        b_end: B-exclusive end (start of shared). Required for shared/b-excl.

    Returns:
        Feature indices, sorted by descending ``cohens_d`` (largest first).
    """
    if k_pct <= 0:
        return []
    if a_end is None:
        a_end = n_a

    if partition == "a-excl+shared":
        if b_end is None:
            raise ValueError("partition='a-excl+shared' requires b_end")
        a_part = select_top_subset(
            rankings_df, k_pct=k_pct, n_a=n_a,
            partition="a-excl", a_end=a_end, b_end=b_end,
        )
        s_part = select_top_subset(
            rankings_df, k_pct=k_pct, n_a=n_a,
            partition="shared", a_end=a_end, b_end=b_end,
        )
        # Preserve A-excl-first ordering, then shared (deterministic).
        seen = set()
        out: list[int] = []
        for i in list(a_part) + list(s_part):
            if i not in seen:
                seen.add(i)
                out.append(int(i))
        return out

    df = rankings_df[rankings_df["sign"] == "tool"]
    if partition == "a-excl":
        df = df[df["feature_idx"] < a_end]
    elif partition == "b-excl":
        if b_end is None:
            raise ValueError("partition='b-excl' requires b_end")
        df = df[(df["feature_idx"] >= a_end) & (df["feature_idx"] < b_end)]
    elif partition == "shared":
        if b_end is None:
            raise ValueError("partition='shared' requires b_end")
        df = df[df["feature_idx"] >= b_end]
    elif partition == "all":
        pass
    else:
        raise ValueError(f"unknown partition {partition!r}")

    df = df.sort_values("cohens_d", ascending=False)
    n_tool = len(df)
    if n_tool == 0:
        return []
    n_select = max(1, math.ceil(k_pct / 100.0 * n_tool))
    n_select = min(n_select, n_tool)
    return df["feature_idx"].iloc[:n_select].tolist()
