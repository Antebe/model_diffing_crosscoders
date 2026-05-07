"""
run_steering_eval.py
────────────────────
Single-model targeted-steering sweep driver (spec §4 P1).

Loads Model A, Model B, and the named crosscoder ONCE, then iterates the
8 × 6 (k%, α) grid for ``--n-prompts`` ToolRL prompts. Per-cell output is
``<out>/k<pct>_a<alpha>.jsonl`` (one line per prompt). Cells with already
``--n-prompts`` lines are skipped on resume; partial cells resume at
prompt-index granularity.

For each prompt and cell we compute three scores:
  - ``clean_score``  : unsteered Model A generation (constant per prompt;
                       cached and reused across all 48 cells of one prompt).
  - ``recon_score``  : Model A with full crosscoder reconstruction patched
                       in (post-recon Model A baseline; constant per prompt).
  - ``steered_score``: Model A with ``h_a + delta`` patched, where
                       ``delta`` is the post-top-k targeted delta.

Note on per-prompt baseline caching: clean and recon scores are recomputed
*per cell* (not cross-cell cached) because each cell file is independently
resumable. This costs an extra ~2× compute relative to cross-cell caching;
the trade-off is dramatically simpler crash recovery.

Usage:
    python run_steering_eval.py \
        --crosscoder antebe1/dfc-D8k-excl10-freeexcl-k160 \
        --rankings data/rankings/dfc-D8k-excl10-freeexcl-k160.csv \
        --out results/targeted_steering/dfc-D8k-excl10-freeexcl-k160 \
        [--n-prompts 100] [--seed 42] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sweep_eval import (
    DATASET_ID,
    MAX_LENGTH,
    MODEL_A_ID,
    MODEL_B_ID,
    extract_prompt,
    generate_clean,
    generate_with_patch,
    get_last_token_activation,
    load_crosscoder,
    score_response,
)
from steering.steer import compute_steering_delta, select_top_subset

SWEEP_KS = [4, 8, 16, 32, 64, 100]
SWEEP_ALPHAS = [1, 6, 16, 32, 64]
N_PROMPTS = 100
# Layer is now read from each crosscoder's hparams.json by load_crosscoder()
# and threaded through as cc.layer; no module-level constant.


def cell_path(out_dir: Path, k_pct: int, alpha: int) -> Path:
    """File path for a (k%, α) cell. Two-digit zero-padded k_pct (3-digit when 100)."""
    out_dir = Path(out_dir)
    kp = "100" if k_pct == 100 else f"{k_pct:02d}"
    return out_dir / f"k{kp}_a{alpha:02d}.jsonl"


def is_cell_complete(p: Path, expected_n: int) -> bool:
    if not p.exists():
        return False
    try:
        with open(p) as f:
            n = sum(1 for line in f if line.strip())
    except OSError:
        return False
    return n >= expected_n


def _load_prompts(n: int, seed: int) -> list[str]:
    """Same sampling as sweep_eval.run_full_eval (seeded ToolRL test split)."""
    try:
        ds = load_dataset(DATASET_ID, split="test")
    except Exception:
        ds = load_dataset(DATASET_ID, split="train")
    examples = list(ds)
    random.seed(seed)
    sampled = random.sample(examples, min(n * 3, len(examples)))
    prompts: list[str] = []
    for ex in sampled:
        p = extract_prompt(ex)
        if p:
            prompts.append(p)
        if len(prompts) >= n:
            break
    return prompts[:n]


@torch.no_grad()
def _eval_prompt_baselines(
    prompt: str,
    crosscoder,
    tokenizer,
    model_a,
    model_b,
    device: str,
    layer: int,
) -> dict:
    """Compute clean + full-recon Model A scores; return cached state for steered eval."""
    ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
    ).input_ids
    h_a = get_last_token_activation(model_a, ids, device, layer=layer)
    h_b = get_last_token_activation(model_b, ids, device, layer=layer)
    x = torch.stack([h_a, h_b], dim=0).unsqueeze(0).to(device)
    features = crosscoder.encode(x)        # (1, dict_size)
    recon = crosscoder.decode(features)    # (1, 2, d)
    recon_a = recon[0, 0, :].float()

    resp_clean = generate_clean(model_a, ids, tokenizer, device)
    resp_recon = generate_with_patch(
        model_a, ids, recon_a, tokenizer, device, layer=layer,
    )
    return dict(
        ids=ids,
        h_a=h_a,
        features=features,
        clean_score=score_response(resp_clean, prompt),
        recon_score=score_response(resp_recon, prompt),
    )


@torch.no_grad()
def _eval_steered(
    prompt: str,
    baseline: dict,
    subset_indices: list[int],
    alpha: float,
    crosscoder,
    tokenizer,
    model_a,
    device: str,
    layer: int,
) -> dict:
    delta = compute_steering_delta(
        features=baseline["features"],
        subset_indices=subset_indices,
        alpha=alpha,
        dfc=crosscoder,
        model_idx=0,
    )
    patched_a = baseline["h_a"] + delta.to(baseline["h_a"].dtype)
    resp = generate_with_patch(
        model_a, baseline["ids"], patched_a, tokenizer, device, layer=layer,
    )
    return score_response(resp, prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted steering single-model sweep")
    parser.add_argument("--crosscoder", required=True)
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models-jsonl", default="models.jsonl")
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--k-list", type=str, default=None,
        help="Override SWEEP_KS with a comma-separated list of k% values "
             "(e.g. '1,3,7,11,15'). Lets you sweep finer grids for "
             "absolute-|S| comparisons.",
    )
    parser.add_argument(
        "--partition",
        choices=["a-excl", "shared", "b-excl", "all", "a-excl+shared"],
        default="a-excl",
        help=(
            "Which partition's top-k% to steer with. 'a-excl' = current "
            "behavior (DFC A-exclusive feats). 'shared'/'b-excl' = DFC "
            "ablations. 'all' = full-dictionary (CrossCoder). For CC "
            "variants (a_end == 0) this is forced to 'all'."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_ks = SWEEP_KS
    if args.k_list:
        sweep_ks = [int(x) for x in args.k_list.split(",") if x.strip()]
        if not sweep_ks:
            raise SystemExit(f"--k-list parsed empty from {args.k_list!r}")
        print(f"[k-list] overriding SWEEP_KS → {sweep_ks}")

    with open(args.models_jsonl) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    matched = [e for e in entries if e["name"] == args.crosscoder]
    if not matched:
        raise SystemExit(
            f"crosscoder {args.crosscoder} not found in {args.models_jsonl}"
        )
    entry = matched[0]

    rankings_df = pd.read_csv(args.rankings)

    # Determine which cells still need work.
    todo: list[tuple[int, int]] = []
    for k_pct in sweep_ks:
        for alpha in SWEEP_ALPHAS:
            cp = cell_path(out_dir, k_pct, alpha)
            if not is_cell_complete(cp, args.n_prompts):
                todo.append((k_pct, alpha))
    if not todo:
        print(f"All 48 cells already complete in {out_dir}")
        return
    print(f"Pending cells: {len(todo)} / 48")

    print(f"Loading tokenizer + Model A + Model B (device={args.device})…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A_ID, torch_dtype=torch.float16, device_map=args.device,
    ).eval()
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B_ID, torch_dtype=torch.float16, device_map=args.device,
    ).eval()
    print(f"Loading crosscoder {args.crosscoder}…")
    cc = load_crosscoder(entry, device=args.device)
    if cc is None:
        raise SystemExit("crosscoder load failed")
    layer = int(getattr(cc, "layer"))
    a_end = int(getattr(cc, "a_end", 0))
    b_end = int(getattr(cc, "b_end", 0))
    n_a = a_end
    partition = args.partition
    if a_end == 0:
        print("  (CrossCoder; using full-dictionary ranking — no a_end partition)")
        n_a = cc.dict_size
        partition = "all"
    print(f"  a_end={a_end}  b_end={b_end}  dict_size={cc.dict_size}  "
          f"layer={layer}  partition={partition}")

    subsets_by_k = {
        k_pct: select_top_subset(
            rankings_df,
            k_pct=float(k_pct),
            n_a=n_a,
            partition=partition,
            a_end=a_end if a_end > 0 else None,
            b_end=b_end if b_end > 0 else None,
        )
        for k_pct in sweep_ks
    }
    for k_pct, s in subsets_by_k.items():
        print(f"  k%={k_pct:>3} → |S|={len(s)}")

    prompts = _load_prompts(args.n_prompts, args.seed)
    print(f"\n{len(prompts)} prompts sampled")

    n_done = 0
    for k_pct, alpha in todo:
        cp = cell_path(out_dir, k_pct, alpha)
        existing_indices: set[int] = set()
        if cp.exists():
            with open(cp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    existing_indices.add(rec.get("prompt_index", -1))
        n_existing = len(existing_indices)
        if n_existing >= args.n_prompts:
            n_done += 1
            continue

        subset = subsets_by_k[k_pct]
        print(
            f"\n[{n_done+1}/{len(todo)}] k%={k_pct:>3} α={alpha:>3}  "
            f"|S|={len(subset)}  resuming@{n_existing}"
        )
        t0 = time.time()
        with open(cp, "a") as f:
            for i, prompt in enumerate(prompts):
                if i in existing_indices:
                    continue
                try:
                    baseline = _eval_prompt_baselines(
                        prompt, cc, tokenizer, model_a, model_b, args.device,
                        layer=layer,
                    )
                    steered_score = _eval_steered(
                        prompt, baseline, subset, float(alpha),
                        cc, tokenizer, model_a, args.device,
                        layer=layer,
                    )
                    rec = dict(
                        prompt_index=i,
                        prompt=prompt[:1000],
                        subset_indices=list(subset),
                        subset_size=len(subset),
                        k_pct=float(k_pct),
                        alpha=float(alpha),
                        clean_score=baseline["clean_score"],
                        recon_score=baseline["recon_score"],
                        steered_score=steered_score,
                    )
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                except Exception as e:
                    err = dict(
                        prompt_index=i, error=str(e),
                        k_pct=float(k_pct), alpha=float(alpha),
                    )
                    f.write(json.dumps(err) + "\n")
                    f.flush()
        n_done += 1
        elapsed = time.time() - t0
        print(f"   cell done in {elapsed:.0f}s")
    print(f"\nAll done. Cells written under {out_dir}")


if __name__ == "__main__":
    main()
