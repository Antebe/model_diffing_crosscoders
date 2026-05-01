"""
train.py — DFC training loop with W&B logging.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dfc import DFCCrossCoder


def train(
    dfc: DFCCrossCoder,
    dataset: Dataset,
    cfg,
    run_name: str = "dfc-run",
    tags: list[str] | None = None,
) -> tuple[DFCCrossCoder, list]:
    """
    Train DFCCrossCoder from a CachedActivationDataset.

    W&B metrics logged each step:
      train/loss, train/mse, train/l1
      train/l0_total, train/l0_a_excl, train/l0_b_excl, train/l0_shared
      debug/enc_max_violation, debug/dec_max_violation

    Returns (trained_dfc, history_list).
    """
    # ── W&B ──────────────────────────────────────────────────────────
    wandb_run = None
    if cfg.wandb_enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
                name=run_name,
                tags=tags or [],
                config={
                    "model_a": cfg.model_a_name,
                    "model_b": cfg.model_b_name,
                    "layer": cfg.layer,
                    "dict_size": cfg.dict_size,
                    "k": cfg.k,
                    "n_a": dfc.n_a,
                    "n_b": dfc.n_b,
                    "n_shared": dfc.n_shared,
                    "steps": cfg.steps,
                    "lr": cfg.lr,
                    "sparsity_coef": cfg.sparsity_coef,
                    "train_batch": cfg.train_batch,
                    "dataset_size": len(dataset),
                },
            )
            print(f"[W&B] Run: {wandb_run.url}")
        except ImportError:
            print("[W&B] wandb not installed, skipping.")

    # ── Setup ─────────────────────────────────────────────────────────
    dfc = dfc.to(cfg.train_device)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    opt = torch.optim.Adam(dfc.parameters(), lr=cfg.lr)
    history: list[dict] = []
    step = 0

    print(
        f"[Train] {cfg.steps} steps | lr={cfg.lr} | λ={cfg.sparsity_coef} "
        f"| k={cfg.k} | batch={cfg.train_batch} | dataset={len(dataset):,}"
    )
    pbar = tqdm(total=cfg.steps, unit="step", dynamic_ncols=True)

    # ── Loop ──────────────────────────────────────────────────────────
    while step < cfg.steps:
        for batch in loader:
            if step >= cfg.steps:
                break

            batch = batch.to(cfg.train_device)
            opt.zero_grad()
            total, mse, l1_shared, l1_exclusive = dfc.loss(batch, cfg.sparsity_coef, cfg.exclusive_sparsity_coef)
            total.backward()
            torch.nn.utils.clip_grad_norm_(dfc.parameters(), 1.0)
            opt.step()
            dfc._apply_masks()

            step += 1

            # ── Metrics ───────────────────────────────────────────────
            with torch.no_grad():
                _, feats = dfc(batch)
                fstats = dfc.feature_stats(feats)

            row = {
                "step": step,
                "train/loss": total.item(),
                "train/mse":  mse.item(),
                "train/l1_shared":   l1_shared.item(),
                "train/l1_exclusive": l1_exclusive.item(),
                "train/l1_total": l1_shared.item() + l1_exclusive.item(),
                **{f"train/{k}": v for k, v in fstats.items()},
            }

            # Partition integrity check (cheap — just looks at weights)
            if step % cfg.log_every == 0:
                vi = dfc.verify_partition_integrity()
                row.update({
                    "debug/enc_max_violation": vi["enc_max_violation"],
                    "debug/dec_max_violation": vi["dec_max_violation"],
                })

            history.append(row)

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{total.item():.4f}",
                mse=f"{mse.item():.4f}",
                l1_shared=f"{l1_shared.item():.4f}",
                l1_excl=f"{l1_exclusive.item():.4f}",
            )

            # ── W&B log ───────────────────────────────────────────────
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in row.items() if k != "step"}, step=step)

            # ── Console log ───────────────────────────────────────────
            if step % cfg.log_every == 0:
                vi_str = f"enc_viol={row.get('debug/enc_max_violation', 0):.1e}"
                tqdm.write(
                    f"step {step:>6} | loss {total.item():.4f} "
                    f"| mse {mse.item():.4f} | l1 {(l1_shared + l1_exclusive).item():.4f} "
                    f"| l0={fstats['l0_total']:.1f} "
                    f"(A={fstats['l0_a_excl']:.1f} B={fstats['l0_b_excl']:.1f} "
                    f"S={fstats['l0_shared']:.1f}) | {vi_str}"
                )

            # ── Checkpoint ────────────────────────────────────────────
            if step % 1_000 == 0:
                dfc.save(cfg.save_path)
                if wandb_run is not None:
                    wandb_run.save(f"{cfg.save_path}/model.pt")

    pbar.close()
    dfc.save(cfg.save_path)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"[Train] Done — {step} steps. Checkpoint: {cfg.save_path}")
    return dfc, history
