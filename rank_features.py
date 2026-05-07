"""
rank_features.py
────────────────
Rank crosscoder features by tool-vs-nontool discriminative score.

Reproduces the ``tool_neurons_A_full.csv`` schema produced by the zip's
neuron_identification pipeline. Output schema (one row per feature):

    rank, feature_idx, cohens_d, auroc, fire_rate_tool, fire_rate_nontool,
    diff, mean_tool, mean_nontool, sign

Usage:
    python rank_features.py \
        --crosscoder antebe1/cc-D8k-nol1-k45 \
        --out data/rankings/cc-D8k-nol1-k45.csv \
        [--n-tool 1000] [--n-nontool 1000] [--seed 42] [--device cuda]

For CrossCoder variants (n_a == 0), all features are ranked and ``sign``
is assigned as 'tool' if ``cohens_d > 0`` else 'nontool'. For DFC variants
(n_a > 0), only features in ``[0, n_a)`` are written (matches the zip's
A-exclusive-only file).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sweep_eval import (
    DATASET_ID,
    MAX_LENGTH,
    MODEL_A_ID,
    MODEL_B_ID,
    extract_prompt,
    get_last_token_activation,
    load_crosscoder,
)

FINEWEB_ID = "HuggingFaceFW/fineweb"
FINEWEB_NAME = "sample-10BT"


# ─── Metric helpers (covered by tests/test_rank_features.py) ───────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two 1-D samples. Returns 0 when both stds are zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    va = a.var(ddof=1) if a.size > 1 else 0.0
    vb = b.var(ddof=1) if b.size > 1 else 0.0
    pooled = math.sqrt((va + vb) / 2.0) if (va + vb) > 0 else 0.0
    if pooled == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """One-vs-one AUROC via Mann–Whitney U with tie correction."""
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    if pos.size == 0 or neg.size == 0:
        return 0.5
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    pos_ranks = ranks[labels == 1].sum()
    return float(
        (pos_ranks - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    )


def fire_rate(acts: np.ndarray, threshold: float = 0.0) -> float:
    """Fraction of samples where activation > threshold."""
    acts = np.asarray(acts)
    if acts.size == 0:
        return 0.0
    return float((acts > threshold).mean())


def rank_features(
    tool_acts: np.ndarray,
    nontool_acts: np.ndarray,
    feature_indices: list[int] | None = None,
) -> pd.DataFrame:
    """Rank features and return DataFrame with the standard schema.

    Args:
        tool_acts:    (n_tool, n_features) post-top-k activation matrix.
        nontool_acts: (n_nontool, n_features) same shape on second axis.
        feature_indices: explicit feature_idx values to write (default: range(n_features)).
    """
    n_features = tool_acts.shape[1]
    if feature_indices is None:
        feature_indices = list(range(n_features))
    rows = []
    for col, fidx in enumerate(feature_indices):
        t = tool_acts[:, col]
        n = nontool_acts[:, col]
        d = cohens_d(t, n)
        rows.append(
            dict(
                feature_idx=int(fidx),
                cohens_d=float(d),
                auroc=float(auroc(t, n)),
                fire_rate_tool=float(fire_rate(t)),
                fire_rate_nontool=float(fire_rate(n)),
                diff=float(t.mean() - n.mean()),
                mean_tool=float(t.mean()),
                mean_nontool=float(n.mean()),
                sign="tool" if d > 0 else "nontool",
            )
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("cohens_d", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index)
    return df[
        [
            "rank", "feature_idx", "cohens_d", "auroc",
            "fire_rate_tool", "fire_rate_nontool",
            "diff", "mean_tool", "mean_nontool", "sign",
        ]
    ]


# ─── Activation collection ─────────────────────────────────────────────────

def _sample_prompts(
    dataset_id: str, name: str | None, n: int, seed: int,
) -> list[str]:
    if name is not None:
        ds = load_dataset(dataset_id, name=name, split="train", streaming=True)
    else:
        try:
            ds = load_dataset(dataset_id, split="test")
        except Exception:
            ds = load_dataset(dataset_id, split="train")
    rng = random.Random(seed)
    if hasattr(ds, "shuffle"):
        try:
            ds = ds.shuffle(seed=seed)
        except Exception:
            pass
    prompts: list[str] = []
    for ex in ds:
        if not isinstance(ex, dict):
            continue
        # extract_prompt handles both `messages`-style and instruction/input/query schemas;
        # FineWeb rows fall back to the `text` field.
        p = extract_prompt(ex) or ex.get("text")
        if p:
            prompts.append(p)
        if len(prompts) >= n * 5:
            break
    rng.shuffle(prompts)
    return prompts[:n]


@torch.no_grad()
def collect_features(
    crosscoder,
    prompts: list[str],
    tokenizer,
    model_a,
    model_b,
    device: str,
    layer: int,
    desc: str = "Encoding",
) -> np.ndarray:
    """Encode each prompt's last-token (h_a, h_b) through the crosscoder; stack
    sparse top-k feature rows. Returns (len(prompts), dict_size) float32 numpy.

    ``layer`` is the transformer layer to read activations from — must match
    the layer the crosscoder was trained on (typically ``crosscoder.layer``).
    """
    rows = []
    for p in tqdm(prompts, desc=desc, unit="prompt"):
        ids = tokenizer(
            p, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
        ).input_ids
        h_a = get_last_token_activation(model_a, ids, device, layer=layer)
        h_b = get_last_token_activation(model_b, ids, device, layer=layer)
        x = torch.stack([h_a, h_b], dim=0).unsqueeze(0).to(device)
        feats = crosscoder.encode(x)  # (1, dict_size)
        rows.append(feats.float().cpu().numpy()[0])
    return np.stack(rows, axis=0)


# ─── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank crosscoder features by Cohen's d."
    )
    parser.add_argument("--crosscoder", required=True,
                        help="HF model id (e.g. antebe1/cc-D8k-nol1-k45)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--models-jsonl", default="models.jsonl")
    parser.add_argument("--n-tool", type=int, default=1000)
    parser.add_argument("--n-nontool", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rank-all-features",
        action="store_true",
        help=(
            "For DFC variants, rank ALL dict features (A-excl ∪ B-excl ∪ "
            "shared) instead of just A-excl. Required for partition "
            "ablations downstream of run_steering_eval --partition."
        ),
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.models_jsonl) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    matched = [e for e in entries if e["name"] == args.crosscoder]
    if not matched:
        raise SystemExit(
            f"crosscoder {args.crosscoder} not found in {args.models_jsonl}"
        )
    entry = matched[0]

    print(f"Loading tokenizer + LLMs (device={args.device})…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A_ID, torch_dtype=torch.float16, device_map=args.device,
    ).eval()
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B_ID, torch_dtype=torch.float16, device_map=args.device,
    ).eval()

    print(f"Loading crosscoder {args.crosscoder} …")
    cc = load_crosscoder(entry, device=args.device)
    if cc is None:
        raise SystemExit(f"failed to load crosscoder {args.crosscoder}")
    layer = int(getattr(cc, "layer"))
    print(f"  layer={layer}")

    print(f"Sampling {args.n_tool} tool prompts …")
    tool_prompts = _sample_prompts(DATASET_ID, None, args.n_tool, args.seed)
    print(f"  got {len(tool_prompts)}")
    print(f"Sampling {args.n_nontool} nontool prompts (FineWeb) …")
    nontool_prompts = _sample_prompts(
        FINEWEB_ID, FINEWEB_NAME, args.n_nontool, args.seed
    )
    print(f"  got {len(nontool_prompts)}")

    t0 = time.time()
    tool_feats = collect_features(
        cc, tool_prompts, tokenizer, model_a, model_b, args.device, layer=layer,
        desc="Encoding tool prompts",
    )
    print(f"  shape={tool_feats.shape}  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    nontool_feats = collect_features(
        cc, nontool_prompts, tokenizer, model_a, model_b, args.device, layer=layer,
        desc="Encoding nontool prompts",
    )
    print(f"  shape={nontool_feats.shape}  ({time.time()-t0:.0f}s)")

    n_a = int(getattr(cc, "a_end", 0))
    if n_a > 0 and not args.rank_all_features:
        feat_indices = list(range(n_a))
        tool_feats = tool_feats[:, :n_a]
        nontool_feats = nontool_feats[:, :n_a]
    else:
        feat_indices = list(range(cc.dict_size))
    df = rank_features(tool_feats, nontool_feats, feat_indices)

    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
