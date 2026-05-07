"""
rank_features_sweep.py
──────────────────────
Multi-crosscoder rank pass with shared LLM forward passes.

Stage 2 of run_all.sh used to invoke ``rank_features.py`` once per crosscoder,
each invocation re-running both 3B LLMs over the same seeded rank prompts.
This driver consolidates the LLM work:

  1. Sample tool + nontool prompts deterministically (seed=42) — ONCE.
  2. Forward both LLMs over those prompts ONCE, snapshotting
     ``hidden_states[L+1]`` for every unique layer ``L`` referenced by the
     supplied crosscoders.
  3. For each crosscoder, encode the matching layer's stacked activations
     through it and write the same per-feature CSV that ``rank_features.py``
     produces (Cohen's d, AUROC, fire-rate, etc.).

The encode step is a single linear projection + top-k — tiny relative to
the LLM forward pass — so the wall-clock cost is roughly that of one
``rank_features.py`` invocation regardless of how many crosscoders share
those LLM forwards.

Usage:
    python rank_features_sweep.py \
        --crosscoders sbhokare/dfc-D8k-excl10-k45-l1,sbhokare/dfc-D8k-excl10-k45-l5,... \
        [--out-dir data/rankings] [--n-tool 4000] [--n-nontool 4000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rank_features import (
    DATASET_ID,
    FINEWEB_ID,
    FINEWEB_NAME,
    _sample_prompts,
    rank_features,
)
from sweep_eval import (
    MAX_LENGTH,
    MODEL_A_ID,
    MODEL_B_ID,
    hidden_states_idx_for,
    load_crosscoder,
)


# ─── Activation collection (one forward pass, multi-layer snapshot) ────────

@torch.no_grad()
def collect_layered_activations(
    prompts: list[str],
    tokenizer,
    model_a,
    model_b,
    layers: list[int],
    device: str,
    desc: str = "Forward",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """One forward pass per LLM per prompt; snapshot last-token activations
    at each requested layer.

    Returns: ``{layer: (h_a, h_b)}`` where each tensor is ``(N, hidden_dim)``
    float32 numpy. Index alignment matches ``prompts``.
    """
    by_layer_a: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    by_layer_b: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    for p in tqdm(prompts, desc=desc, unit="prompt"):
        ids = tokenizer(
            p, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
        ).input_ids.to(device)
        out_a = model_a(input_ids=ids, output_hidden_states=True)
        out_b = model_b(input_ids=ids, output_hidden_states=True)
        for L in layers:
            idx = hidden_states_idx_for(L)
            by_layer_a[L].append(out_a.hidden_states[idx][0, -1, :].float().cpu().numpy())
            by_layer_b[L].append(out_b.hidden_states[idx][0, -1, :].float().cpu().numpy())
    return {
        L: (np.stack(by_layer_a[L], axis=0), np.stack(by_layer_b[L], axis=0))
        for L in layers
    }


@torch.no_grad()
def encode_through_crosscoder(
    cc, h_a: np.ndarray, h_b: np.ndarray, device: str, batch_size: int = 256,
) -> np.ndarray:
    """Stack (h_a, h_b) along the model axis and encode through ``cc``.
    Returns ``(N, dict_size)`` float32 numpy."""
    n = len(h_a)
    out = np.zeros((n, cc.dict_size), dtype=np.float32)
    for i in range(0, n, batch_size):
        a = torch.from_numpy(h_a[i:i + batch_size]).to(device)
        b = torch.from_numpy(h_b[i:i + batch_size]).to(device)
        x = torch.stack([a, b], dim=1)            # (B, 2, hidden_dim)
        feats = cc.encode(x)
        out[i:i + batch_size] = feats.float().cpu().numpy()
    return out


# ─── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Rank features for multiple crosscoders, sharing LLM forwards."
    )
    p.add_argument("--crosscoders", required=True,
                   help="Comma-separated HF model IDs.")
    p.add_argument("--out-dir", default="data/rankings",
                   help="Directory for per-crosscoder CSV outputs.")
    p.add_argument("--models-jsonl", default="models.jsonl")
    p.add_argument("--n-tool", type=int, default=1000)
    p.add_argument("--n-nontool", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip crosscoders whose CSV already exists.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cc_names = [s.strip() for s in args.crosscoders.split(",") if s.strip()]
    if not cc_names:
        raise SystemExit("--crosscoders parsed to empty list")

    with open(args.models_jsonl) as f:
        entries = {
            e["name"]: e
            for e in (json.loads(line) for line in f if line.strip())
        }
    missing = [n for n in cc_names if n not in entries]
    if missing:
        raise SystemExit(
            f"crosscoders missing from {args.models_jsonl}: {missing}"
        )

    # Filter out crosscoders whose CSV already exists, when requested.
    todo: list[str] = []
    for n in cc_names:
        short = n.split("/", 1)[-1]
        csv = out_dir / f"{short}.csv"
        if args.skip_existing and csv.exists():
            print(f"  ✓ {csv} exists, skipping")
            continue
        todo.append(n)
    if not todo:
        print("All rankings already exist; nothing to do.")
        return

    # Resolve the unique layers needed up front (from models.jsonl).
    layers = sorted({
        int(entries[n]["hyperparameters"]["layer"]) for n in todo
    })
    print(f"Crosscoders to rank: {len(todo)}")
    for n in todo:
        L = entries[n]["hyperparameters"]["layer"]
        print(f"  - {n}  (layer {L})")
    print(f"Unique layers: {layers}")

    # Sample prompts deterministically — same set seen by every crosscoder.
    print(f"\nSampling {args.n_tool} tool + {args.n_nontool} nontool prompts (seed={args.seed}) …")
    tool_prompts = _sample_prompts(DATASET_ID, None, args.n_tool, args.seed)
    nontool_prompts = _sample_prompts(
        FINEWEB_ID, FINEWEB_NAME, args.n_nontool, args.seed,
    )
    print(f"  tool={len(tool_prompts)}  nontool={len(nontool_prompts)}")

    # Load LLMs once, do one forward pass per LLM per prompt, snapshot all layers.
    print(f"\nLoading tokenizer + LLMs (device={args.device}) …")
    tok = AutoTokenizer.from_pretrained(MODEL_B_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A_ID, torch_dtype=torch.float16, device_map=args.device,
    ).eval()
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B_ID, torch_dtype=torch.float16, device_map=args.device,
    ).eval()

    t0 = time.time()
    tool_acts = collect_layered_activations(
        tool_prompts, tok, model_a, model_b, layers, args.device,
        desc="Tool prompts",
    )
    print(f"  tool acts ready ({time.time() - t0:.0f}s)")
    t0 = time.time()
    nontool_acts = collect_layered_activations(
        nontool_prompts, tok, model_a, model_b, layers, args.device,
        desc="Nontool prompts",
    )
    print(f"  nontool acts ready ({time.time() - t0:.0f}s)")

    # Free LLMs — they aren't needed past this point.
    del model_a, model_b
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Encode through each crosscoder, compute stats, write CSV.
    for full_name in todo:
        short = full_name.split("/", 1)[-1]
        csv = out_dir / f"{short}.csv"
        print(f"\n=== {short} ===")
        cc = load_crosscoder(entries[full_name], device=args.device)
        if cc is None:
            print(f"  ⚠ failed to load {full_name}; skipping")
            continue
        L = int(getattr(cc, "layer", entries[full_name]["hyperparameters"]["layer"]))
        if L not in tool_acts:
            print(f"  ⚠ no snapshot at layer {L}; expected one of {sorted(tool_acts)}")
            del cc
            continue

        t_h_a, t_h_b = tool_acts[L]
        n_h_a, n_h_b = nontool_acts[L]
        t0 = time.time()
        tool_feats = encode_through_crosscoder(cc, t_h_a, t_h_b, args.device)
        nontool_feats = encode_through_crosscoder(cc, n_h_a, n_h_b, args.device)
        print(f"  encoded: tool={tool_feats.shape}  nontool={nontool_feats.shape}  ({time.time()-t0:.0f}s)")

        n_a = int(getattr(cc, "a_end", 0))
        if n_a > 0:
            feat_indices = list(range(n_a))
            tool_feats = tool_feats[:, :n_a]
            nontool_feats = nontool_feats[:, :n_a]
        else:
            feat_indices = list(range(cc.dict_size))

        df = rank_features(tool_feats, nontool_feats, feat_indices)
        df.to_csv(csv, index=False)
        print(f"  wrote {len(df)} rows → {csv}")

        del cc
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
