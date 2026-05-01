"""
dfc.py — Dedicated Feature CrossCoder (DFC) model.

Feature layout in dict_size
────────────────────────────
  ┌─────────────────────┬─────────────────────┬──────────────────────────┐
  │  A-exclusive (n_a)  │  B-exclusive (n_b)  │     Shared (n_shared)    │
  └─────────────────────┴─────────────────────┴──────────────────────────┘
  idx:  0 ─────── a_end ──────── b_end ───────────────────── dict_size

Constraints (enforced by gradient masking + _apply_masks every step)
──────────────────────────────────────────────────────────────────────
  • Model A cannot encode/decode B-exclusive features
  • Model B cannot encode/decode A-exclusive features
  • Shared features are accessible to both
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFCCrossCoder(nn.Module):

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        model_a_exclusive_pct: float = 0.05,
        model_b_exclusive_pct: float = 0.05,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k

        self.n_a     = int(dict_size * model_a_exclusive_pct)
        self.n_b     = int(dict_size * model_b_exclusive_pct)
        self.n_shared = dict_size - self.n_a - self.n_b
        self.a_end   = self.n_a
        self.b_end   = self.n_a + self.n_b

        print(
            f"[DFC] dict={dict_size} k={k} | "
            f"A-excl={self.n_a} B-excl={self.n_b} shared={self.n_shared}"
        )

        # Encoder: W_enc[model, d_in, dict_size]
        self.W_enc = nn.Parameter(
            torch.randn(2, activation_dim, dict_size) / (activation_dim ** 0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(dict_size))

        # Decoder: W_dec[dict_size, model, d_in]
        self.W_dec = nn.Parameter(
            torch.randn(dict_size, 2, activation_dim) / (dict_size ** 0.5)
        )
        self.b_dec = nn.Parameter(torch.zeros(2, activation_dim))

        # ── Partition masks (move with .to(device)) ───────────────────
        # enc_mask[model, dict_size]
        enc_mask = torch.ones(2, dict_size)
        enc_mask[1, : self.a_end] = 0                   # B cannot encode A-excl
        enc_mask[0, self.a_end : self.b_end] = 0        # A cannot encode B-excl
        self.register_buffer("enc_mask", enc_mask)

        # dec_mask[dict_size, model]
        dec_mask = torch.ones(dict_size, 2)
        dec_mask[: self.a_end, 1] = 0                   # A-excl: B decoder = 0
        dec_mask[self.a_end : self.b_end, 0] = 0        # B-excl: A decoder = 0
        self.register_buffer("dec_mask", dec_mask)

        self._apply_masks()

    # ── Weight enforcement ────────────────────────────────────────────

    @torch.no_grad()
    def _apply_masks(self):
        """Zero forbidden weights. Call after every optimiser step."""
        for m in range(2):
            self.W_enc.data[m] *= self.enc_mask[m].unsqueeze(0)
        for m in range(2):
            self.W_dec.data[:, m, :] *= self.dec_mask[:, m].unsqueeze(1)

    # ── Forward ───────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, d) → features: (B, dict_size) sparse top-k."""
        W = self.W_enc * self.enc_mask.unsqueeze(1)         # (2, d, dict)
        pre = torch.einsum("bmd,mdf->bf", x, W) + self.b_enc
        pre = F.relu(pre)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        features = torch.zeros_like(pre)
        features.scatter_(-1, topk_idx, topk_vals)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, dict_size) → (B, 2, d)."""
        W = self.W_dec * self.dec_mask.unsqueeze(-1)        # (dict, 2, d)
        return torch.einsum("bf,fmd->bmd", features, W) + self.b_dec

    def forward(self, x: torch.Tensor):
        """x: (B, 2, d) → (reconstruction, features)."""
        features = self.encode(x)
        recon    = self.decode(features)
        return recon, features

    def loss(
        self, 
        x: torch.Tensor, 
        sparsity_coef: float = 1e-3,
        exclusive_sparsity_coef: float = 1e-3  # Lower penalty for exclusive features
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MSE + weighted L1 sparsity. Returns (total, mse, l1_shared, l1_exclusive)."""
        recon, features = self.forward(x)
        mse = F.mse_loss(recon, x)
        
        # Split features by partition
        # fa = features[:, :self.a_end]          # A-exclusive
        # fb = features[:, self.a_end:self.b_end] # B-exclusive 
        fs = features[:, self.b_end:]          # Shared

        # A sees: A-exclusive + shared
        fa = torch.cat([features[:, :self.a_end], features[:, self.b_end:]], dim=-1)   # A-exclusive + shared
        fb = torch.cat([features[:, self.a_end:self.b_end], features[:, self.b_end:]], dim=-1)  # B-exclusive + shared
        
        # Separate sparsity penalties
        l1_shared = fs.abs().mean()
        l1_exclusive = (fa.abs().mean() + fb.abs().mean()) / 2
        total = mse + exclusive_sparsity_coef * l1_exclusive + sparsity_coef * l1_shared
        
        return total, mse, l1_shared, l1_exclusive

    # ── Diagnostics ───────────────────────────────────────────────────

    @torch.no_grad()
    def verify_partition_integrity(self) -> dict[str, float]:
        """Max absolute value in weights that should be zero."""
        if self.n_a == 0 and self.n_b == 0:
            return {"enc_max_violation": 0.0, "dec_max_violation": 0.0}
        enc_viol  = (self.W_enc.abs() * (1 - self.enc_mask).unsqueeze(1)).max().item()
        dec_viol_a = self.W_dec[: self.a_end, 1, :].abs().max().item() if self.n_a > 0 else 0.0
        dec_viol_b = self.W_dec[self.a_end : self.b_end, 0, :].abs().max().item() if self.n_b > 0 else 0.0
        return {
            "enc_max_violation": enc_viol,
            "dec_max_violation": max(dec_viol_a, dec_viol_b),
        }

    @torch.no_grad()
    def feature_stats(self, features: torch.Tensor) -> dict[str, float]:
        """Partition-level activation stats for a batch of features."""
        fa = features[:, : self.a_end]
        fb = features[:, self.a_end : self.b_end]
        fs = features[:, self.b_end :]
        return {
            "l0_total":    (features > 0).float().sum(dim=-1).mean().item(),
            "l0_a_excl":   (fa > 0).float().sum(dim=-1).mean().item(),
            "l0_b_excl":   (fb > 0).float().sum(dim=-1).mean().item(),
            "l0_shared":   (fs > 0).float().sum(dim=-1).mean().item(),
            "mean_a_excl": fa.mean().item(),
            "mean_b_excl": fb.mean().item(),
            "mean_shared": fs.mean().item(),
        }

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model with disk space checking and error handling."""
        import shutil
        import tempfile
        
        # Check available disk space
        free_space = shutil.disk_usage(Path(path).parent or ".").free
        if free_space < 100_000_000:  # Less than 100MB
            raise RuntimeError(f"Insufficient disk space: {free_space / 1e9:.2f}GB available. Need at least 0.1GB.")
        
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save to temporary file first, then move to avoid corruption
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            try:
                torch.save(self.state_dict(), tmp_file.name)
                shutil.move(tmp_file.name, f"{path}/model.pt")
            except Exception as e:
                # Clean up temp file on error
                if Path(tmp_file.name).exists():
                    Path(tmp_file.name).unlink()
                raise RuntimeError(f"Failed to save model: {e}")
        
        # Save config
        config_data = dict(
            activation_dim=self.activation_dim,
            dict_size=self.dict_size,
            k=self.k,
            n_a=self.n_a,
            n_b=self.n_b,
        )
        with open(f"{path}/config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        print(f"[DFC] Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DFCCrossCoder":
        cfg = json.load(open(f"{path}/config.json"))
        model = cls(
            activation_dim=cfg["activation_dim"],
            dict_size=cfg["dict_size"],
            k=cfg["k"],
            model_a_exclusive_pct=cfg["n_a"] / cfg["dict_size"],
            model_b_exclusive_pct=cfg["n_b"] / cfg["dict_size"],
        )
        model.load_state_dict(
            torch.load(f"{path}/model.pt", map_location=device, weights_only=True)
        )
        return model.to(device)
