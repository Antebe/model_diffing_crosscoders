"""
run_train.py — Train DFCCrossCoder from cached activations.

Requires run_cache.py to have been run first.

Usage:
    python run_train.py
    python run_train.py --steps 20000 --dict_size 32768 --k 128
    python run_train.py --no_wandb
"""

import argparse

from cache import CachedActivationDataset, build_combined_dataset, cache_exists
from config import Config
from dfc import DFCCrossCoder
from train import train


def main(cfg: Config):
    print("=" * 60)
    print("  DFC CrossCoder Training")
    print("=" * 60)

    # ── Check caches exist ───────────────────────────────────────────
    missing = []
    if not cache_exists(cfg.fineweb_cache): missing.append("FineWeb")
    if not cache_exists(cfg.toolrl_cache):  missing.append("ToolRL")
    if missing:
        print(f"[Error] Missing caches: {missing}. Run python run_cache.py first.")
        return

    # ── Load datasets ────────────────────────────────────────────────
    dataset = build_combined_dataset(
        fineweb_cache=cfg.fineweb_cache,
        toolrl_cache=cfg.toolrl_cache,
        normalize=True,
    )

    # ── Build DFC ────────────────────────────────────────────────────
    # If a checkpoint exists, resume from it
    import json
    from pathlib import Path

    ckpt_cfg = Path(cfg.save_path) / "config.json"
    if ckpt_cfg.exists():
        print(f"[Train] Resuming from checkpoint: {cfg.save_path}")
        dfc = DFCCrossCoder.load(cfg.save_path, device=cfg.train_device)
    else:
        # Infer activation_dim from cache metadata
        import torch
        meta_path = Path(cfg.fineweb_cache) / "meta.json"
        sample = torch.load(
            Path(cfg.fineweb_cache) / json.load(open(meta_path))["shards"][0],
            map_location="cpu",
            weights_only=True,
        )
        cfg.activation_dim = sample.shape[-1]
        print(f"[Train] activation_dim from cache: {cfg.activation_dim}")

        dfc = DFCCrossCoder(
            activation_dim=cfg.activation_dim,
            dict_size=cfg.dict_size,
            k=cfg.k,
            model_a_exclusive_pct=cfg.model_a_exclusive_pct,
            model_b_exclusive_pct=cfg.model_b_exclusive_pct,
        )

    # ── Train ────────────────────────────────────────────────────────
    # Use checkpoint folder name + sparsity info for unique W&B run name
    save_name = Path(cfg.save_path).name
    run_name = f"{save_name}-sp{cfg.sparsity_coef}-esp{cfg.exclusive_sparsity_coef}"

    dfc, history = train(
        dfc=dfc,
        dataset=dataset,
        cfg=cfg,
        run_name=run_name,
        tags=["fineweb", "toolrl", f"layer-{cfg.layer}"],
    )

    print(f"\n[Done] Model saved to {cfg.save_path}")
    print(f"       To inspect, run the notebook: jupyter notebook notebook.ipynb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",               type=int,   default=None)
    parser.add_argument("--dict_size",           type=int,   default=None)
    parser.add_argument("--k",                   type=int,   default=None)
    parser.add_argument("--lr",                  type=float, default=None)
    parser.add_argument("--sparsity_coef",       type=float, default=None)
    parser.add_argument("--train_batch",         type=int,   default=None)
    parser.add_argument("--layer",               type=int,   default=None)
    parser.add_argument("--model_a_excl",        type=float, default=None)
    parser.add_argument("--model_b_excl",        type=float, default=None)
    parser.add_argument("--train_device",        type=str,   default=None)
    parser.add_argument("--save_path",            type=str,   default=None)
    parser.add_argument("--exclusive_sparsity_coef", type=float, default=None)
    parser.add_argument("--no_wandb",            action="store_true")
    parser.add_argument("--wandb_project",       type=str,   default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.steps          is not None: cfg.steps                 = args.steps
    if args.dict_size      is not None: cfg.dict_size             = args.dict_size
    if args.k              is not None: cfg.k                     = args.k
    if args.lr             is not None: cfg.lr                    = args.lr
    if args.sparsity_coef  is not None: cfg.sparsity_coef         = args.sparsity_coef
    if args.train_batch    is not None: cfg.train_batch           = args.train_batch
    if args.layer          is not None: cfg.layer                 = args.layer
    if args.model_a_excl   is not None: cfg.model_a_exclusive_pct = args.model_a_excl
    if args.model_b_excl   is not None: cfg.model_b_exclusive_pct = args.model_b_excl
    if args.train_device   is not None: cfg.train_device          = args.train_device
    if args.save_path      is not None: cfg.save_path              = args.save_path
    if args.exclusive_sparsity_coef is not None: cfg.exclusive_sparsity_coef = args.exclusive_sparsity_coef
    if args.no_wandb:                   cfg.wandb_enabled          = False
    if args.wandb_project  is not None: cfg.wandb_project         = args.wandb_project

    print("Hyperparameters:")
    for key, value in vars(cfg).items():
        print(f"  {key}: {value}")
    main(cfg)


