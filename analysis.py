"""
inspect.py — Tools for analysing a trained DFCCrossCoder.

Functions
─────────
  inspect_latents(dfc, x)              — partition breakdown for one sample
  compare_on_texts(...)                — batch of texts → per-sample breakdown
  top_exclusive_features(dfc, x, ...)  — highest-firing exclusive features
  decoder_similarity(dfc)              — cosine sim of shared feature decoders
  feature_frequency(dfc, dataset, ...) — how often each feature fires across corpus
  plot_loss(history)                   — matplotlib training curve
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm import tqdm

if TYPE_CHECKING:
    from dfc import DFCCrossCoder
    from cache import CachedActivationDataset


# ──────────────────────────────────────────────────────────────────────
# Single-sample inspection
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def inspect_latents(
    dfc: "DFCCrossCoder",
    x: torch.Tensor,
    top_n: int = 20,
) -> dict:
    """
    x : (2, d) or (1, 2, d).
    Prints a partition breakdown and top active features.
    Returns a dict with features, reconstruction, per-partition counts, mse.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    dev = next(dfc.parameters()).device
    x = x.to(dev)

    recon, feats = dfc(x)
    f = feats[0]
    a_end, b_end = dfc.a_end, dfc.b_end
    fa, fb, fs = f[:a_end], f[a_end:b_end], f[b_end:]

    def pct(n, d):
        return f"{100*n/max(d,1):.1f}%"

    print(f"\n{'═'*56}")
    print(f"  Latent Space Inspection")
    print(f"{'═'*56}")
    print(f"  A-exclusive  active : {(fa>0).sum():>5} / {dfc.n_a:<6} {pct((fa>0).sum(), dfc.n_a)}")
    print(f"  B-exclusive  active : {(fb>0).sum():>5} / {dfc.n_b:<6} {pct((fb>0).sum(), dfc.n_b)}")
    print(f"  Shared       active : {(fs>0).sum():>5} / {dfc.n_shared:<6} {pct((fs>0).sum(), dfc.n_shared)}")
    print(f"  Total        active : {(f>0).sum():>5} / {dfc.dict_size}")

    vals, idxs = torch.topk(f, min(top_n, dfc.k))
    print(f"\n  {'Feat':>7}  {'Value':>9}  Partition")
    print(f"  {'─'*35}")
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if v <= 0:
            break
        kind = "A-exclusive" if i < a_end else ("B-exclusive" if i < b_end else "Shared")
        print(f"  {i:>7}  {v:>9.4f}  {kind}")

    mse = F.mse_loss(recon[0], x[0]).item()
    print(f"\n  Reconstruction MSE : {mse:.6f}")
    print(f"{'═'*56}")

    return {
        "features":        f.cpu(),
        "reconstruction":  recon[0].cpu(),
        "n_active_a":      (fa > 0).sum().item(),
        "n_active_b":      (fb > 0).sum().item(),
        "n_active_shared": (fs > 0).sum().item(),
        "mse":             mse,
    }


# ──────────────────────────────────────────────────────────────────────
# Multi-text comparison
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compare_on_texts(
    dfc: "DFCCrossCoder",
    model_a,
    model_b,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    device_a: str,
    device_b: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract activations for `texts` on the fly, run DFC, print partition breakdown.
    Returns (features, reconstruction) on CPU.
    """
    from cache import extract_last_token_acts

    acts_a = extract_last_token_acts(model_a, tokenizer, texts, layer_idx, device_a, batch_size=4)
    acts_b = extract_last_token_acts(model_b, tokenizer, texts, layer_idx, device_b, batch_size=4)
    x = torch.stack([acts_a, acts_b], dim=1)

    dev = next(dfc.parameters()).device
    recon, feats = dfc(x.to(dev))
    feats, recon, x = feats.cpu(), recon.cpu(), x.cpu()

    a_end, b_end = dfc.a_end, dfc.b_end
    print(f"\n{'═'*70}")
    print(f"  Model A ({device_a}) vs Model B ({device_b})  |  layer {layer_idx}  |  {len(texts)} samples")
    print(f"{'═'*70}")
    for i, text in enumerate(texts):
        f = feats[i]
        na = (f[:a_end] > 0).sum().item()
        nb = (f[a_end:b_end] > 0).sum().item()
        ns = (f[b_end:] > 0).sum().item()
        mse = F.mse_loss(recon[i], x[i]).item()
        print(f"\n  [{i:02d}] {text[:80].replace(chr(10),' ')!r}")
        print(f"        A-excl={na:>3} | B-excl={nb:>3} | shared={ns:>4} | MSE={mse:.5f}")

    return feats, recon


# ──────────────────────────────────────────────────────────────────────
# Per-model exclusive feature top-k
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def top_exclusive_features(
    dfc: "DFCCrossCoder",
    x: torch.Tensor,
    model: str = "a",
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """
    Return the top-n most activated exclusive features for `model` ('a' or 'b').
    x : (2, d) single pair.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    dev = next(dfc.parameters()).device
    _, feats = dfc(x.to(dev))
    f = feats[0]

    excl   = f[: dfc.a_end]     if model == "a" else f[dfc.a_end : dfc.b_end]
    offset = 0                   if model == "a" else dfc.a_end

    active = (excl > 0).sum().item()
    k = min(top_n, active)
    if k == 0:
        print(f"  No exclusive-{model.upper()} features active.")
        return []

    vals, idxs = torch.topk(excl, k)
    result = [(offset + int(i), float(v)) for v, i in zip(vals, idxs)]

    print(f"\n  Top {k} Model-{model.upper()} exclusive features  (of {active} active):")
    print(f"  {'Feat':>7}  {'Value':>9}")
    for idx, val in result:
        print(f"  {idx:>7}  {val:>9.4f}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Decoder alignment
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decoder_similarity(dfc: "DFCCrossCoder") -> torch.Tensor:
    """
    For shared features: cosine similarity between Model-A and Model-B decoder directions.
    Values near 1 → same direction (feature behaves identically across models).
    Values near 0 → orthogonal (feature has diverged).
    """
    W  = dfc.W_dec[dfc.b_end:]          # (n_shared, 2, d)
    wa, wb = W[:, 0, :], W[:, 1, :]
    cos = F.cosine_similarity(wa, wb, dim=-1)

    print(f"\n{'═'*50}")
    print(f"  Shared-feature decoder cosine similarity  (n={dfc.n_shared})")
    print(f"{'═'*50}")
    print(f"  mean  : {cos.mean():.4f}")
    print(f"  std   : {cos.std():.4f}")
    print(f"  min   : {cos.min():.4f}")
    print(f"  max   : {cos.max():.4f}")
    print(f"  >0.9  : {(cos>0.9).float().mean()*100:.1f}%  (highly aligned)")
    print(f"  0.5–0.9: {((cos>=0.5)&(cos<=0.9)).float().mean()*100:.1f}%  (partially aligned)")
    print(f"  <0.5  : {(cos<0.5).float().mean()*100:.1f}%  (diverged)")
    return cos


# ──────────────────────────────────────────────────────────────────────
# Feature frequency over corpus
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def feature_frequency(
    dfc: "DFCCrossCoder",
    dataset: "CachedActivationDataset",
    n_samples: int = 4_096,
    batch_size: int = 128,
) -> torch.Tensor:
    """
    Count how many times each feature fires over `n_samples` from `dataset`.
    Returns (dict_size,) int64 tensor.

    Useful for finding dead features and dominant features per partition.
    """
    from torch.utils.data import DataLoader, Subset
    import random

    dev = next(dfc.parameters()).device
    idxs = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    sub  = Subset(dataset, idxs)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0)

    freq = torch.zeros(dfc.dict_size, dtype=torch.long)
    for batch in tqdm(loader, desc="Feature freq", leave=False):
        feats = dfc.encode(batch.to(dev))
        freq += (feats > 0).long().sum(dim=0).cpu()

    # Print summary by partition
    fa = freq[: dfc.a_end]
    fb = freq[dfc.a_end : dfc.b_end]
    fs = freq[dfc.b_end:]
    total = len(idxs)

    def dead(f):
        return (f == 0).sum().item()

    print(f"\n  Feature frequency over {total} samples:")
    print(f"  A-exclusive : dead={dead(fa)}/{dfc.n_a}  max={fa.max().item()}  mean={fa.float().mean():.1f}")
    print(f"  B-exclusive : dead={dead(fb)}/{dfc.n_b}  max={fb.max().item()}  mean={fb.float().mean():.1f}")
    print(f"  Shared      : dead={dead(fs)}/{dfc.n_shared}  max={fs.max().item()}  mean={fs.float().mean():.1f}")

    return freq


# ──────────────────────────────────────────────────────────────────────
# Training curve
# ──────────────────────────────────────────────────────────────────────

def plot_loss(history: list[dict], smooth: int = 50):
    """Plot training loss / mse / l1. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed.")
        return

    steps = [r["step"]        for r in history]
    loss  = [r["train/loss"]  for r in history]
    mse   = [r["train/mse"]   for r in history]
    l1    = [r["train/l1"]    for r in history]

    def smooth_vals(v):
        import numpy as np
        return np.convolve(v, np.ones(smooth)/smooth, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, label in zip(axes, [loss, mse, l1], ["Total Loss", "MSE", "L1"]):
        ax.plot(steps[smooth-1:], smooth_vals(vals), linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=120)
    plt.show()
    print("[Plot] Saved training_curve.png")
