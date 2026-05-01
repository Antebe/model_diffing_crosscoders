#!/usr/bin/env python3
"""
run_experiments.py — ToolRL Neuron Evolution Analysis.

Focused analysis: track how the most influential ToolRL-specific features
in the DFC latent space map back to model A (ToolRL) and model B (base),
revealing where new capabilities come from.

Pipeline:
  1. Select top-K most active A-exclusive features on ToolRL data
  2. Find co-activating shared features for each top feature
  3. Analyze decoder weight structure: which hidden neurons are implicated
  4. Compare shared feature decoders across model A vs B (evolution signal)
  5. Visualize the full story with publication-quality figures

Usage:
  uv run run_experiments.py
  uv run run_experiments.py --checkpoint ./checkpoints/dfc2 --top_k 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dfc import DFCCrossCoder


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_feature_shards(cache_dir: str, device: str = "cpu") -> torch.Tensor:
    """Load all feature shards and concatenate into (N, dict_size)."""
    meta = json.load(open(Path(cache_dir) / "meta.json"))
    shards = []
    for fname in tqdm(meta["shards"], desc=f"Loading {Path(cache_dir).name}"):
        s = torch.load(Path(cache_dir) / fname, map_location=device, weights_only=True)
        shards.append(s.float())
    return torch.cat(shards, dim=0)


def load_autointerp(results_dir: Path, feat_idx: int) -> dict | None:
    """Try to load autointerp explanation for a feature."""
    for subdir in ["toolrl", "fineweb"]:
        path = results_dir / subdir / "0000000" / f"feat_{feat_idx:07d}.json"
        if path.exists():
            return json.load(open(path))
    return None


def partition_label(idx: int, dfc: DFCCrossCoder) -> str:
    if idx < dfc.a_end:
        return "A-excl"
    elif idx < dfc.b_end:
        return "B-excl"
    return "Shared"


# ──────────────────────────────────────────────────────────────────────
# Step 1: Select top ToolRL-active A-exclusive features
# ──────────────────────────────────────────────────────────────────────

def select_top_toolrl_features(
    tr_feats: torch.Tensor,
    fw_feats: torch.Tensor,
    dfc: DFCCrossCoder,
    top_k: int = 30,
) -> dict:
    """
    Rank A-exclusive features by mean activation on ToolRL data.
    Returns dict with indices, activation stats, and differential scores.
    """
    print("\n" + "=" * 70)
    print("  Step 1: Selecting Top ToolRL-Specific Features")
    print("=" * 70)

    a_excl_tr = tr_feats[:, :dfc.a_end]  # (N_tr, n_a)
    a_excl_fw = fw_feats[:, :dfc.a_end]  # (N_fw, n_a)

    # Mean activation (only counting nonzero)
    tr_fire_mask = a_excl_tr > 0
    tr_fire_rate = tr_fire_mask.float().mean(dim=0)  # fraction of samples firing
    tr_mean_act = a_excl_tr.sum(dim=0) / tr_fire_mask.sum(dim=0).clamp(min=1)

    fw_fire_mask = a_excl_fw > 0
    fw_fire_rate = fw_fire_mask.float().mean(dim=0)

    # Score: fire_rate on ToolRL * mean activation (combined influence)
    influence_score = tr_fire_rate * tr_mean_act

    # Top-K by influence
    top_vals, top_local_idx = torch.topk(influence_score, min(top_k, dfc.n_a))
    top_global_idx = top_local_idx  # A-excl starts at 0

    print(f"\n  Top {len(top_global_idx)} A-exclusive features by ToolRL influence:")
    print(f"  {'Rank':<6} {'Feat#':<8} {'TR fire%':>10} {'FW fire%':>10} {'TR mean':>10} {'Score':>10}")
    print(f"  {'─' * 56}")
    for rank, gidx in enumerate(top_global_idx):
        i = gidx.item()
        print(
            f"  {rank+1:<6} {i:<8} "
            f"{tr_fire_rate[i].item():>10.4f} "
            f"{fw_fire_rate[i].item():>10.4f} "
            f"{tr_mean_act[i].item():>10.3f} "
            f"{influence_score[i].item():>10.3f}"
        )

    return {
        "top_global_idx": top_global_idx.numpy(),
        "tr_fire_rate": tr_fire_rate.numpy(),
        "fw_fire_rate": fw_fire_rate.numpy(),
        "tr_mean_act": tr_mean_act.numpy(),
        "influence_score": influence_score.numpy(),
    }


# ──────────────────────────────────────────────────────────────────────
# Step 2: Find co-activating shared features
# ──────────────────────────────────────────────────────────────────────

def find_coactivating_features(
    tr_feats: torch.Tensor,
    dfc: DFCCrossCoder,
    anchor_indices: np.ndarray,
    n_coact: int = 10,
) -> dict[int, np.ndarray]:
    """
    For each anchor A-exclusive feature, find the shared features that
    most often co-fire on the same ToolRL samples.

    Returns {anchor_idx: array of top shared feature global indices}.
    """
    print("\n" + "=" * 70)
    print("  Step 2: Finding Co-activating Shared Features")
    print("=" * 70)

    shared_feats = tr_feats[:, dfc.b_end:]  # (N, n_shared)
    shared_fire = (shared_feats > 0).float()

    coact_map: dict[int, np.ndarray] = {}

    for anchor in anchor_indices:
        anchor_mask = (tr_feats[:, anchor] > 0).float()  # (N,)
        n_anchor = anchor_mask.sum().clamp(min=1)

        # P(shared fires | anchor fires)
        cofire = (shared_fire * anchor_mask.unsqueeze(1)).sum(dim=0) / n_anchor
        top_vals, top_local = torch.topk(cofire, min(n_coact, dfc.n_shared))
        top_global = top_local.numpy() + dfc.b_end  # offset to global index
        coact_map[int(anchor)] = top_global

        print(f"  Anchor feat {anchor}: top co-activators = "
              f"{top_global[:5].tolist()}... (max P={top_vals[0]:.3f})")

    return coact_map


# ──────────────────────────────────────────────────────────────────────
# Step 3: Decoder weight analysis — mapping to hidden neurons
# ──────────────────────────────────────────────────────────────────────

def analyze_decoder_weights(
    dfc: DFCCrossCoder,
    top_indices: np.ndarray,
    coact_map: dict[int, np.ndarray],
    out_dir: Path,
) -> dict:
    """
    For each top feature, extract decoder weights W_dec[feat, model, :].

    For A-exclusive features: only model A decoder is active.
    For shared co-activators: both decoders are active → compare evolution.

    Returns structured analysis dict.
    """
    print("\n" + "=" * 70)
    print("  Step 3: Decoder Weight Analysis")
    print("=" * 70)

    W_dec = dfc.W_dec.detach().cpu()  # (dict_size, 2, d)
    d = dfc.activation_dim

    results = {
        "top_features": {},
        "shared_evolution": {},
        "neuron_importance_A": np.zeros(d),
        "neuron_importance_B": np.zeros(d),
    }

    # Accumulate which hidden neurons are most implicated
    neuron_weight_A = np.zeros(d)
    neuron_weight_B = np.zeros(d)

    for feat_idx in top_indices:
        feat_idx = int(feat_idx)
        dec_A = W_dec[feat_idx, 0, :].numpy()  # (d,)
        dec_B = W_dec[feat_idx, 1, :].numpy()  # (d,) — zeroed for A-excl

        # Top neurons for model A reconstruction
        top_neurons_A = np.argsort(np.abs(dec_A))[-20:][::-1]

        results["top_features"][feat_idx] = {
            "dec_A_norm": float(np.linalg.norm(dec_A)),
            "dec_B_norm": float(np.linalg.norm(dec_B)),
            "top_neurons_A": top_neurons_A.tolist(),
            "top_neuron_weights_A": dec_A[top_neurons_A].tolist(),
        }

        neuron_weight_A += np.abs(dec_A)

        # Analyze shared co-activators
        if feat_idx in coact_map:
            for shared_idx in coact_map[feat_idx]:
                shared_idx = int(shared_idx)
                if shared_idx in results["shared_evolution"]:
                    continue  # already analyzed

                s_dec_A = W_dec[shared_idx, 0, :].numpy()
                s_dec_B = W_dec[shared_idx, 1, :].numpy()

                cos_sim = float(F.cosine_similarity(
                    W_dec[shared_idx, 0, :].unsqueeze(0),
                    W_dec[shared_idx, 1, :].unsqueeze(0),
                ).item())

                # Weight difference: how model A diverged from model B
                diff = s_dec_A - s_dec_B
                top_diff_neurons = np.argsort(np.abs(diff))[-20:][::-1]

                results["shared_evolution"][shared_idx] = {
                    "cos_sim_A_B": cos_sim,
                    "dec_A_norm": float(np.linalg.norm(s_dec_A)),
                    "dec_B_norm": float(np.linalg.norm(s_dec_B)),
                    "diff_norm": float(np.linalg.norm(diff)),
                    "top_diverged_neurons": top_diff_neurons.tolist(),
                    "top_diverged_weights": diff[top_diff_neurons].tolist(),
                }

                neuron_weight_A += np.abs(s_dec_A)
                neuron_weight_B += np.abs(s_dec_B)

    results["neuron_importance_A"] = neuron_weight_A
    results["neuron_importance_B"] = neuron_weight_B

    # Summary
    shared_sims = [v["cos_sim_A_B"] for v in results["shared_evolution"].values()]
    if shared_sims:
        print(f"\n  Shared feature decoder similarity (A vs B):")
        print(f"    Mean cosine sim: {np.mean(shared_sims):.4f}")
        print(f"    Min:  {np.min(shared_sims):.4f}")
        print(f"    Max:  {np.max(shared_sims):.4f}")
        print(f"    Std:  {np.std(shared_sims):.4f}")

    # Top 20 most implicated neurons in model A
    top_neurons_global = np.argsort(neuron_weight_A)[-20:][::-1]
    print(f"\n  Top 20 most ToolRL-implicated neurons in model A (hidden dim):")
    for i, n in enumerate(top_neurons_global):
        print(f"    #{i+1}: neuron {n} (cumulative |weight| = {neuron_weight_A[n]:.4f})")

    return results


# ──────────────────────────────────────────────────────────────────────
# Step 4: Visualization
# ──────────────────────────────────────────────────────────────────────

def visualize_evolution(
    dfc: DFCCrossCoder,
    top_result: dict,
    coact_map: dict[int, np.ndarray],
    decoder_result: dict,
    tr_feats: torch.Tensor,
    fw_feats: torch.Tensor,
    out_dir: Path,
    autointerp_dir: Path | None = None,
):
    """Create publication-quality multi-panel figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    print("\n" + "=" * 70)
    print("  Step 4: Generating Visualizations")
    print("=" * 70)

    # ── Color palette ──
    C_TOOLRL = "#E63946"     # red
    C_BASE   = "#457B9D"     # steel blue
    C_SHARED = "#2A9D8F"     # teal
    C_ACCENT = "#F4A261"     # amber
    C_BG     = "#FAFAFA"
    C_GRID   = "#E0E0E0"
    C_TEXT   = "#2D3436"

    top_idx = top_result["top_global_idx"]
    W_dec = dfc.W_dec.detach().cpu()

    # Load autointerp labels if available
    feat_labels: dict[int, str] = {}
    if autointerp_dir and autointerp_dir.exists():
        for subdir in ["toolrl", "fineweb"]:
            interp_dir = autointerp_dir / subdir / "0000000"
            if interp_dir.exists():
                for f in interp_dir.glob("feat_*.json"):
                    data = json.load(open(f))
                    feat_labels[data["feat_idx"]] = data.get("explanation", "")[:60]

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 1: Feature Overview — Top ToolRL Features
    # ══════════════════════════════════════════════════════════════════

    n_top = len(top_idx)
    fig = plt.figure(figsize=(20, 14), facecolor=C_BG)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.06, right=0.96, top=0.92, bottom=0.06)

    fig.suptitle("ToolRL Neuron Evolution: From Base Model to Tool-Use",
                 fontsize=18, fontweight="bold", color=C_TEXT, y=0.97)

    # ── Panel A: Influence scores bar chart ──
    ax = fig.add_subplot(gs[0, 0])
    scores = top_result["influence_score"][top_idx]
    y_pos = np.arange(min(n_top, 25))
    display_idx = top_idx[:25]
    labels = []
    for idx in display_idx:
        lbl = feat_labels.get(int(idx), "")
        labels.append(f"F{idx}" + (f" | {lbl[:30]}" if lbl else ""))

    bars = ax.barh(y_pos, scores[:25], color=C_TOOLRL, alpha=0.85, height=0.7,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, color=C_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Influence Score (fire rate × mean activation)", fontsize=9)
    ax.set_title("A  Top ToolRL-Exclusive Features", fontsize=12, fontweight="bold",
                 color=C_TOOLRL, loc="left")
    ax.set_facecolor(C_BG)
    ax.grid(axis="x", color=C_GRID, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: Fire rate comparison (ToolRL vs FineWeb) ──
    ax = fig.add_subplot(gs[0, 1])
    tr_rates = top_result["tr_fire_rate"][top_idx[:25]]
    fw_rates = top_result["fw_fire_rate"][top_idx[:25]]
    x = np.arange(len(tr_rates))
    w = 0.35
    ax.bar(x - w/2, tr_rates, w, color=C_TOOLRL, alpha=0.85, label="ToolRL", edgecolor="white")
    ax.bar(x + w/2, fw_rates, w, color=C_BASE, alpha=0.85, label="FineWeb (base)", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i}" for i in top_idx[:25]], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Firing Rate (fraction of samples)", fontsize=9)
    ax.set_title("B  ToolRL vs Base Firing Rates", fontsize=12, fontweight="bold",
                 color=C_TEXT, loc="left")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel C: Decoder norm comparison (A-excl features, model A only) ──
    ax = fig.add_subplot(gs[0, 2])
    dec_norms = []
    for idx in top_idx[:25]:
        dec_norms.append(np.linalg.norm(W_dec[int(idx), 0, :].numpy()))
    ax.bar(np.arange(len(dec_norms)), dec_norms, color=C_TOOLRL, alpha=0.8,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.arange(len(dec_norms)))
    ax.set_xticklabels([f"F{i}" for i in top_idx[:25]], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Decoder L2 Norm", fontsize=9)
    ax.set_title("C  Decoder Magnitude → Model A", fontsize=12, fontweight="bold",
                 color=C_TOOLRL, loc="left")
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel D: Shared feature evolution — cosine similarity histogram ──
    ax = fig.add_subplot(gs[1, 0])
    cos_sims = [v["cos_sim_A_B"] for v in decoder_result["shared_evolution"].values()]
    if cos_sims:
        ax.hist(cos_sims, bins=30, color=C_SHARED, alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(cos_sims), color=C_ACCENT, linewidth=2, linestyle="--",
                   label=f"Mean = {np.mean(cos_sims):.3f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("Cosine Similarity (Model A vs Model B decoder)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("D  Shared Feature Decoder Evolution", fontsize=12, fontweight="bold",
                 color=C_SHARED, loc="left")
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel E: Neuron importance heatmap (top 50 neurons, A vs B) ──
    ax = fig.add_subplot(gs[1, 1])
    imp_A = decoder_result["neuron_importance_A"]
    imp_B = decoder_result["neuron_importance_B"]

    # Select neurons that are most different
    diff = imp_A - imp_B
    top_diverged = np.argsort(np.abs(diff))[-50:][::-1]

    data = np.stack([imp_A[top_diverged], imp_B[top_diverged]], axis=0)
    # Normalize per row for visibility
    row_max = data.max(axis=1, keepdims=True).clip(min=1e-8)
    data_norm = data / row_max

    cmap = LinearSegmentedColormap.from_list("toolrl", ["#FFFFFF", C_TOOLRL, "#8B0000"])
    im = ax.imshow(data_norm, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Model A\n(ToolRL)", "Model B\n(Base)"], fontsize=9)
    ax.set_xlabel("Top diverged hidden neurons (sorted by |A - B|)", fontsize=9)

    # Show every 5th neuron label
    tick_positions = np.arange(0, len(top_diverged), 5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(top_diverged[i]) for i in tick_positions], fontsize=6, rotation=45)

    ax.set_title("E  Hidden Neuron Importance: A vs B", fontsize=12, fontweight="bold",
                 color=C_TEXT, loc="left")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Relative importance")

    # ── Panel F: Neuron importance difference (A minus B) ──
    ax = fig.add_subplot(gs[1, 2])
    diff_sorted = diff[top_diverged]
    colors_bar = [C_TOOLRL if d > 0 else C_BASE for d in diff_sorted]
    ax.bar(np.arange(len(diff_sorted)), diff_sorted, color=colors_bar, alpha=0.8,
           edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Top diverged neurons (same order as Panel E)", fontsize=9)
    ax.set_ylabel("Importance(A) - Importance(B)", fontsize=9)
    ax.set_title("F  ToolRL Gain vs Base", fontsize=12, fontweight="bold",
                 color=C_TEXT, loc="left")

    patch_a = mpatches.Patch(color=C_TOOLRL, label="ToolRL gained")
    patch_b = mpatches.Patch(color=C_BASE, label="Base stronger")
    ax.legend(handles=[patch_a, patch_b], fontsize=8, framealpha=0.9)
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(out_dir / "toolrl_evolution_overview.png", dpi=200, facecolor=C_BG)
    plt.close()
    print(f"  Saved: {out_dir / 'toolrl_evolution_overview.png'}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 2: Three-column evolution flow
    #   Model B neurons → DFC latent features → Model A neurons
    # ══════════════════════════════════════════════════════════════════

    W_enc = dfc.W_enc.detach().cpu()  # (2, d, dict_size)

    # --- Collect the features to show (mix of A-excl + top shared) ---
    n_excl_show = min(8, n_top)
    # Pick the most-diverged shared features from co-activation analysis
    shared_by_div = sorted(
        decoder_result["shared_evolution"].items(),
        key=lambda x: x[1]["diff_norm"], reverse=True,
    )
    n_shared_show = min(6, len(shared_by_div))
    shared_show_idx = [int(k) for k, _ in shared_by_div[:n_shared_show]]

    feat_show = list(top_idx[:n_excl_show].astype(int)) + shared_show_idx
    n_feat = len(feat_show)

    # --- For each feature, find top-3 Model-B encoder neurons and top-3 Model-A decoder neurons ---
    b_neurons_agg: dict[int, float] = {}
    a_neurons_agg: dict[int, float] = {}
    # (feat_row, neuron, weight)
    links_B: list[tuple[int, int, float]] = []
    links_A: list[tuple[int, int, float]] = []

    for row, fidx in enumerate(feat_show):
        # Model A decoder weights
        dec_a = W_dec[fidx, 0, :].numpy()
        top_a = np.argsort(np.abs(dec_a))[-3:][::-1]
        for n in top_a:
            links_A.append((row, int(n), float(dec_a[n])))
            a_neurons_agg[int(n)] = a_neurons_agg.get(int(n), 0) + abs(dec_a[n])

        # Model B encoder weights (for shared features) or decoder (for shared)
        if fidx < dfc.a_end:
            # A-exclusive: model B has no access → encoder weights are zero
            # Show model A encoder instead (where it reads from in A-space)
            enc_a = W_enc[0, :, fidx].numpy()  # (d,)
            top_b = np.argsort(np.abs(enc_a))[-3:][::-1]
            for n in top_b:
                links_B.append((row, int(n), float(enc_a[n])))
                b_neurons_agg[int(n)] = b_neurons_agg.get(int(n), 0) + abs(enc_a[n])
        else:
            # Shared: model B decoder shows B's representation
            dec_b = W_dec[fidx, 1, :].numpy()
            top_b = np.argsort(np.abs(dec_b))[-3:][::-1]
            for n in top_b:
                links_B.append((row, int(n), float(dec_b[n])))
                b_neurons_agg[int(n)] = b_neurons_agg.get(int(n), 0) + abs(dec_b[n])

    # Select top neurons to display per side
    n_neurons_show = 15
    top_b_neurons = sorted(b_neurons_agg.items(), key=lambda x: -x[1])[:n_neurons_show]
    top_a_neurons = sorted(a_neurons_agg.items(), key=lambda x: -x[1])[:n_neurons_show]
    b_order = {n: i for i, (n, _) in enumerate(top_b_neurons)}
    a_order = {n: i for i, (n, _) in enumerate(top_a_neurons)}

    # --- Draw ---
    fig, ax = plt.subplots(figsize=(22, 14), facecolor=C_BG)
    ax.set_xlim(-0.3, 5.3)
    y_max = max(n_feat, n_neurons_show) * 1.3 + 1
    ax.set_ylim(-2.0, y_max)
    ax.axis("off")
    fig.suptitle(
        "Neuron Evolution: Model B → DFC Latent → Model A",
        fontsize=18, fontweight="bold", color=C_TEXT, y=0.97,
    )
    ax.text(2.5, y_max - 0.3,
            "How base-model neurons map through shared/exclusive features to ToolRL neurons",
            ha="center", fontsize=11, color="#636E72", style="italic")

    col_x = [0.0, 2.15, 4.3]   # left (B neurons), center (features), right (A neurons)
    box_w = [0.9, 1.1, 0.9]
    box_h = 0.65

    def y_pos(row: int, total: int) -> float:
        return (total - 1 - row) * 1.2

    # ---- Column 1: Model B / source neurons ----
    for neuron, row_i in b_order.items():
        y = y_pos(row_i, n_neurons_show)
        imp = b_neurons_agg[neuron]
        max_imp = top_b_neurons[0][1]
        alpha = 0.2 + 0.8 * (imp / max(max_imp, 1e-8))
        box = mpatches.FancyBboxPatch(
            (col_x[0] - box_w[0] / 2, y - box_h / 2), box_w[0], box_h,
            boxstyle="round,pad=0.06", facecolor=C_BASE, alpha=alpha,
            edgecolor=C_BASE, linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(col_x[0], y, f"n{neuron}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    # ---- Column 2: DFC features ----
    for row, fidx in enumerate(feat_show):
        y = y_pos(row, n_feat)
        is_excl = fidx < dfc.a_end
        color = C_TOOLRL if is_excl else C_SHARED
        tag = "A-excl" if is_excl else "Shared"
        lbl = feat_labels.get(fidx, "")
        text = f"F{fidx}  [{tag}]"
        if lbl:
            text += f"\n{lbl[:32]}"

        score = top_result["influence_score"][fidx] if is_excl else decoder_result["shared_evolution"].get(fidx, {}).get("diff_norm", 0)
        max_score = max(top_result["influence_score"][top_idx[:n_excl_show]].max(), 1e-8) if is_excl else max((v["diff_norm"] for v in decoder_result["shared_evolution"].values()), default=1)
        alpha = 0.25 + 0.75 * (score / max(max_score, 1e-8))

        box = mpatches.FancyBboxPatch(
            (col_x[1] - box_w[1] / 2, y - box_h / 2), box_w[1], box_h,
            boxstyle="round,pad=0.06", facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(col_x[1], y, text, ha="center", va="center",
                fontsize=6.5, color=C_TEXT, fontweight="bold")

    # ---- Column 3: Model A neurons ----
    for neuron, row_i in a_order.items():
        y = y_pos(row_i, n_neurons_show)
        imp = a_neurons_agg[neuron]
        max_imp = top_a_neurons[0][1]
        alpha = 0.2 + 0.8 * (imp / max(max_imp, 1e-8))
        box = mpatches.FancyBboxPatch(
            (col_x[2] - box_w[2] / 2, y - box_h / 2), box_w[2], box_h,
            boxstyle="round,pad=0.06", facecolor=C_TOOLRL, alpha=alpha,
            edgecolor=C_TOOLRL, linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(col_x[2], y, f"n{neuron}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    # ---- Draw links: B neurons → features ----
    for feat_row, neuron, weight in links_B:
        if neuron not in b_order:
            continue
        y_start = y_pos(b_order[neuron], n_neurons_show)
        y_end = y_pos(feat_row, n_feat)
        is_excl = feat_show[feat_row] < dfc.a_end
        color = C_TOOLRL if is_excl else C_BASE
        lw = max(0.4, abs(weight) * 40)
        alpha = min(0.85, abs(weight) / 0.04 + 0.08)
        ax.annotate(
            "", xy=(col_x[1] - box_w[1] / 2, y_end),
            xytext=(col_x[0] + box_w[0] / 2, y_start),
            arrowprops=dict(arrowstyle="-|>", color=color, alpha=alpha,
                            linewidth=lw, connectionstyle="arc3,rad=0.15",
                            mutation_scale=8),
        )

    # ---- Draw links: features → A neurons ----
    for feat_row, neuron, weight in links_A:
        if neuron not in a_order:
            continue
        y_start = y_pos(feat_row, n_feat)
        y_end = y_pos(a_order[neuron], n_neurons_show)
        is_excl = feat_show[feat_row] < dfc.a_end
        color = C_TOOLRL if is_excl else C_SHARED
        lw = max(0.4, abs(weight) * 40)
        alpha = min(0.85, abs(weight) / 0.04 + 0.08)
        ax.annotate(
            "", xy=(col_x[2] - box_w[2] / 2, y_end),
            xytext=(col_x[1] + box_w[1] / 2, y_start),
            arrowprops=dict(arrowstyle="-|>", color=color, alpha=alpha,
                            linewidth=lw, connectionstyle="arc3,rad=0.15",
                            mutation_scale=8),
        )

    # ---- Column headers ----
    header_y = -1.2
    ax.text(col_x[0], header_y, "Model B Neurons\n(Base, d=2048)", ha="center",
            fontsize=12, fontweight="bold", color=C_BASE)
    ax.text(col_x[1], header_y, "DFC Latent Features\n(A-excl + Shared)", ha="center",
            fontsize=12, fontweight="bold", color=C_TEXT)
    ax.text(col_x[2], header_y, "Model A Neurons\n(ToolRL, d=2048)", ha="center",
            fontsize=12, fontweight="bold", color=C_TOOLRL)

    # Legend
    legend_patches = [
        mpatches.Patch(color=C_TOOLRL, alpha=0.7, label="A-exclusive path"),
        mpatches.Patch(color=C_SHARED, alpha=0.7, label="Shared path (A side)"),
        mpatches.Patch(color=C_BASE, alpha=0.7, label="Shared path (B side)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10, framealpha=0.9)

    plt.savefig(out_dir / "evolution_flow.png", dpi=200, facecolor=C_BG,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'evolution_flow.png'}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 3: Shared feature evolution — detailed comparison
    # ══════════════════════════════════════════════════════════════════

    shared_data = decoder_result["shared_evolution"]
    if shared_data:
        n_shared_show = min(30, len(shared_data))
        sorted_shared = sorted(shared_data.items(), key=lambda x: x[1]["diff_norm"], reverse=True)
        top_shared = sorted_shared[:n_shared_show]

        fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=C_BG)
        fig.suptitle("Shared Feature Evolution: How ToolRL Reshapes Base Model Representations",
                     fontsize=16, fontweight="bold", color=C_TEXT, y=0.98)

        # Panel A: Scatter — cos_sim vs diff_norm
        ax = axes[0]
        sims = [v["cos_sim_A_B"] for _, v in sorted_shared]
        diffs = [v["diff_norm"] for _, v in sorted_shared]
        sc = ax.scatter(sims, diffs, c=diffs, cmap="YlOrRd", s=40, alpha=0.8,
                        edgecolors="white", linewidth=0.5)
        ax.set_xlabel("Cosine Similarity (A vs B decoder)", fontsize=10)
        ax.set_ylabel("Decoder Difference Norm", fontsize=10)
        ax.set_title("A  Alignment vs Divergence", fontsize=12, fontweight="bold",
                     color=C_SHARED, loc="left")
        plt.colorbar(sc, ax=ax, shrink=0.7, label="Diff norm")
        ax.set_facecolor(C_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel B: Top most-diverged shared features
        ax = axes[1]
        feat_ids = [f"F{k}" for k, _ in top_shared[:20]]
        diff_norms = [v["diff_norm"] for _, v in top_shared[:20]]
        cos_sims_bar = [v["cos_sim_A_B"] for _, v in top_shared[:20]]
        y = np.arange(len(feat_ids))
        bars = ax.barh(y, diff_norms, color=C_SHARED, alpha=0.8, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(feat_ids, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Decoder Difference Norm (A - B)", fontsize=10)
        ax.set_title("B  Most Diverged Shared Features", fontsize=12, fontweight="bold",
                     color=C_SHARED, loc="left")

        # Annotate with cosine similarity
        for i, (norm, sim) in enumerate(zip(diff_norms, cos_sims_bar)):
            ax.text(norm + 0.002, i, f"cos={sim:.2f}", va="center", fontsize=7, color=C_TEXT)

        ax.set_facecolor(C_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel C: Decoder weight comparison for top 5 most diverged
        ax = axes[2]
        n_detail = min(5, len(top_shared))
        for rank, (feat_idx, info) in enumerate(top_shared[:n_detail]):
            dec_A = W_dec[feat_idx, 0, :].numpy()
            dec_B = W_dec[feat_idx, 1, :].numpy()

            # Show top 10 neurons by divergence
            diff_vec = dec_A - dec_B
            top_n = np.argsort(np.abs(diff_vec))[-10:][::-1]

            x_offset = rank * 12
            ax.bar(np.arange(10) + x_offset, dec_A[top_n], width=0.4,
                   color=C_TOOLRL, alpha=0.8, label="Model A" if rank == 0 else "")
            ax.bar(np.arange(10) + x_offset + 0.4, dec_B[top_n], width=0.4,
                   color=C_BASE, alpha=0.8, label="Model B" if rank == 0 else "")

            # Group label
            ax.text(x_offset + 5, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.02,
                    f"F{feat_idx}", ha="center", fontsize=8, fontweight="bold")

        ax.set_title("C  Decoder Weights: A vs B (Top Diverged Neurons)", fontsize=12,
                     fontweight="bold", color=C_TEXT, loc="left")
        ax.set_ylabel("Decoder Weight", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_facecolor(C_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(out_dir / "shared_feature_evolution.png", dpi=200, facecolor=C_BG)
        plt.close()
        print(f"  Saved: {out_dir / 'shared_feature_evolution.png'}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE 4: Neuron fingerprint heatmap (annotated) + A-vs-B strip
    # ══════════════════════════════════════════════════════════════════

    n_feat_show = min(20, n_top)
    n_neuron_show = 40

    imp_A = decoder_result["neuron_importance_A"]
    imp_B = decoder_result["neuron_importance_B"]
    top_neurons = np.argsort(imp_A)[-n_neuron_show:][::-1]

    # Build the heatmap: features × neurons (Model A decoder weights)
    heatmap_data = np.zeros((n_feat_show, n_neuron_show))
    for i, feat_idx in enumerate(top_idx[:n_feat_show]):
        dec_A = W_dec[int(feat_idx), 0, :].numpy()
        heatmap_data[i, :] = dec_A[top_neurons]

    # Two rows: main heatmap + A-vs-B comparison strip
    fig, axes = plt.subplots(
        2, 1, figsize=(max(18, n_neuron_show * 0.55), 10), facecolor=C_BG,
        gridspec_kw={"height_ratios": [5, 1.2], "hspace": 0.08},
        sharex=True,
    )

    # ---- Top: main fingerprint heatmap ----
    ax = axes[0]
    vmax = np.abs(heatmap_data).max()
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(np.arange(n_feat_show))
    feat_ylabels = []
    for idx in top_idx[:n_feat_show]:
        lbl = feat_labels.get(int(idx), "")
        feat_ylabels.append(f"F{idx}" + (f"  {lbl[:28]}" if lbl else ""))
    ax.set_yticklabels(feat_ylabels, fontsize=7)
    ax.set_ylabel("ToolRL-Exclusive Feature", fontsize=11)
    ax.set_title("ToolRL Feature → Model A Neuron Fingerprint (Decoder Weights)",
                 fontsize=14, fontweight="bold", color=C_TEXT, pad=10)

    # Annotate every column
    ax.set_xticks(np.arange(n_neuron_show))
    ax.set_xticklabels([str(n) for n in top_neurons], fontsize=6.5, rotation=90)

    # Light gridlines between cells
    for x in np.arange(-0.5, n_neuron_show, 1):
        ax.axvline(x, color=C_GRID, linewidth=0.3, alpha=0.5)
    for y in np.arange(-0.5, n_feat_show, 1):
        ax.axhline(y, color=C_GRID, linewidth=0.3, alpha=0.5)

    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label="Decoder weight (A)")

    # ---- Bottom: A-vs-B importance comparison per neuron ----
    ax2 = axes[1]
    a_vals = imp_A[top_neurons]
    b_vals = imp_B[top_neurons]
    # Normalize for comparison
    scale = max(a_vals.max(), b_vals.max(), 1e-8)
    a_norm = a_vals / scale
    b_norm = b_vals / scale

    x_pos = np.arange(n_neuron_show)
    bar_w = 0.38
    ax2.bar(x_pos - bar_w / 2, a_norm, bar_w, color=C_TOOLRL, alpha=0.85,
            label="Model A (ToolRL)", edgecolor="white", linewidth=0.3)
    ax2.bar(x_pos + bar_w / 2, b_norm, bar_w, color=C_BASE, alpha=0.85,
            label="Model B (Base)", edgecolor="white", linewidth=0.3)

    ax2.set_xticks(np.arange(n_neuron_show))
    ax2.set_xticklabels([str(n) for n in top_neurons], fontsize=6.5, rotation=90)
    ax2.set_xlabel("Hidden Neuron Index", fontsize=11)
    ax2.set_ylabel("Rel. Importance", fontsize=9)
    ax2.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)
    ax2.set_facecolor(C_BG)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", color=C_GRID, linewidth=0.4)

    plt.savefig(out_dir / "neuron_fingerprint_heatmap.png", dpi=200, facecolor=C_BG,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'neuron_fingerprint_heatmap.png'}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ToolRL Neuron Evolution Analysis")
    parser.add_argument("--checkpoint", default="./checkpoints/dfc2",
                        help="Path to DFC checkpoint directory")
    parser.add_argument("--toolrl_features", default="./cache/toolrl_features",
                        help="Path to cached ToolRL feature vectors")
    parser.add_argument("--fineweb_features", default="./cache/fineweb_features",
                        help="Path to cached FineWeb feature vectors")
    parser.add_argument("--autointerp_dir", default="./results/autointerp",
                        help="Path to autointerp results (optional)")
    parser.add_argument("--out_dir", default="./results/evolution",
                        help="Output directory for figures and data")
    parser.add_argument("--top_k", type=int, default=30,
                        help="Number of top A-exclusive features to analyze")
    parser.add_argument("--n_coact", type=int, default=10,
                        help="Number of co-activating shared features per anchor")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading DFC from {args.checkpoint}...")
    dfc = DFCCrossCoder.load(args.checkpoint, device="cpu")

    # Load features
    print("Loading cached feature vectors...")
    tr_feats = load_feature_shards(args.toolrl_features)
    fw_feats = load_feature_shards(args.fineweb_features)
    print(f"  ToolRL: {tr_feats.shape}  FineWeb: {fw_feats.shape}")

    # Step 1: Select top ToolRL features
    top_result = select_top_toolrl_features(tr_feats, fw_feats, dfc, top_k=args.top_k)

    # Step 2: Find co-activating shared features
    coact_map = find_coactivating_features(
        tr_feats, dfc, top_result["top_global_idx"], n_coact=args.n_coact
    )

    # Step 3: Decoder weight analysis
    decoder_result = analyze_decoder_weights(
        dfc, top_result["top_global_idx"], coact_map, out_dir
    )

    # Step 4: Visualize
    autointerp_dir = Path(args.autointerp_dir) if args.autointerp_dir else None
    visualize_evolution(
        dfc, top_result, coact_map, decoder_result,
        tr_feats, fw_feats, out_dir, autointerp_dir,
    )

    # Save raw results
    summary = {
        "top_features": top_result["top_global_idx"].tolist(),
        "coactivation_map": {str(k): v.tolist() for k, v in coact_map.items()},
        "decoder_analysis": {
            "top_features": {
                str(k): {kk: vv for kk, vv in v.items()}
                for k, v in decoder_result["top_features"].items()
            },
            "shared_evolution": {
                str(k): v for k, v in decoder_result["shared_evolution"].items()
            },
        },
    }
    with open(out_dir / "evolution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {out_dir / 'evolution_summary.json'}")

    print("\n" + "=" * 70)
    print("  Done! All outputs in:", out_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
