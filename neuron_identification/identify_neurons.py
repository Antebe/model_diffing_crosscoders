"""
identify_neurons.py

Identify tool-vs-non-tool important neurons using a trained DFC crosscoder.
Output is a ranked list of exclusive features by how strongly they
discriminate tool-use text from general text — candidates for steering.

All configuration is read from environment variables. Set them in
neuron_identification/identify_neurons.sh rather than editing this file.

Pipeline:
  1. Load crosscoder variant (by name from models.jsonl) + Model A + Model B.
  2. Collect A-exclusive and B-exclusive feature activations over:
       - tool corpus    (default: emrecanacikgoz/ToolRL)
       - non-tool corpus (default: HuggingFaceFW/fineweb, streamed)
  3. Rank each exclusive feature by mean(tool) - mean(non-tool).
  4. Write CSV rankings + discrimination bar plot + co-activation heatmaps
     + exclusive-magnitude boxplots.
"""

import json
import os
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _env(name, default, cast=str):
    raw = os.environ.get(name, default)
    return cast(raw) if cast is not str else raw


CROSSCODER_NAME = _env("CROSSCODER_NAME", "antebe1/dfc-D8k-excl10-freeexcl-k160")
MODELS_JSONL = _env("MODELS_JSONL", str(REPO_ROOT / "models.jsonl"))
MODEL_A_ID = _env("MODEL_A_ID", "chengq9/ToolRL-Qwen2.5-3B")
MODEL_B_ID = _env("MODEL_B_ID", "Qwen/Qwen2.5-3B")
HIDDEN_STATES_IDX = _env("HIDDEN_STATES_IDX", "14", int)
DEVICE = _env("DEVICE", "cuda")
N_PER_DATASET = _env("N_PER_DATASET", "1000", int)
TOOL_DATASET = _env("TOOL_DATASET", "emrecanacikgoz/ToolRL")
TOOL_DATASET_SPLIT = _env("TOOL_DATASET_SPLIT", "train")
NONTOOL_DATASET = _env("NONTOOL_DATASET", "HuggingFaceFW/fineweb")
NONTOOL_DATASET_CONFIG = _env("NONTOOL_DATASET_CONFIG", "default")
MAX_LENGTH = _env("MAX_LENGTH", "512", int)
TOP_K = _env("TOP_K", "15", int)
CORR_VLIM = _env("CORR_VLIM", "0.5", float)
OUTPUT_DIR = Path(_env("OUTPUT_DIR", str(REPO_ROOT / "neuron_identification/output")))
SEED = _env("SEED", "42", int)
CACHE_ACTIVATIONS = _env("CACHE_ACTIVATIONS", "0") == "1"

# Reuse repo utilities. Monkey-patch HIDDEN_STATES_IDX so the repo's
# get_last_token_activation picks up the env-configured value at call time.
import sweep_eval  # noqa: E402

sweep_eval.HIDDEN_STATES_IDX = HIDDEN_STATES_IDX
from sweep_eval import (  # noqa: E402
    extract_prompt,
    get_last_token_activation,
    load_crosscoder,
)


def resolve_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable, falling back to CPU")
        return "cpu"
    return device


def load_model_entry(models_jsonl, name):
    with open(models_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("name") == name:
                return entry
    raise ValueError(f"Crosscoder '{name}' not found in {models_jsonl}")


def load_llm(model_id, device):
    print(f"  Loading {model_id} on {device} ...")
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    return m.to(device).eval()


def get_nontool_text(record):
    for key in ("text", "content", "passage"):
        if key in record and record[key]:
            s = str(record[key]).strip()
            if len(s) > 50:
                return s
    return None


def collect_texts_tool(n):
    print(f"Loading tool dataset: {TOOL_DATASET} (split={TOOL_DATASET_SPLIT})")
    ds = None
    for split in (TOOL_DATASET_SPLIT, "train"):
        try:
            ds = load_dataset(TOOL_DATASET, split=split)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"Could not load split '{TOOL_DATASET_SPLIT}' or 'train' from {TOOL_DATASET}")
    ds = ds.shuffle(seed=SEED)
    texts = []
    for rec in ds:
        if len(texts) >= n:
            break
        t = extract_prompt(rec)
        if t:
            texts.append(t)
    print(f"  collected {len(texts)} tool texts (shuffled, seed={SEED})")
    return texts


def collect_texts_nontool(n):
    print(
        f"Streaming non-tool dataset: {NONTOOL_DATASET} (config={NONTOOL_DATASET_CONFIG})"
    )
    try:
        stream = load_dataset(
            NONTOOL_DATASET, NONTOOL_DATASET_CONFIG, split="train", streaming=True
        )
    except Exception:
        stream = load_dataset(NONTOOL_DATASET, split="train", streaming=True)
    # Reservoir-style shuffle over a buffer of upstream records.
    stream = stream.shuffle(seed=SEED, buffer_size=max(10_000, n * 10))
    texts = []
    for rec in stream:
        if len(texts) >= n:
            break
        t = get_nontool_text(rec)
        if t:
            texts.append(t)
    print(f"  collected {len(texts)} non-tool texts (shuffled, seed={SEED})")
    return texts


def write_prompt_activations_jsonl(
    path,
    tool_texts,
    nontool_texts,
    A_tool,
    A_nontool,
    B_tool,
    B_nontool,
    diff_a,
    diff_b,
    a_offset,
    b_offset,
):
    """One JSON object per line; each object = one prompt + its active neurons.

    Schema per line:
      {
        "dataset": "tool" | "nontool",
        "prompt_index": int,
        "prompt": str,
        "n_active_A": int,
        "n_active_B": int,
        "activations": [
          {"partition": "A"|"B", "feature_idx": int,
           "activation": float, "overall_rank": int},
          ...  # sorted by overall_rank ascending (most discriminative first)
        ]
      }

    overall_rank is the feature's rank in the |diff| ranking within its
    partition (0 = most discriminative), aligned with tool_neurons_{A,B}_full.csv.
    """
    rank_a = np.empty_like(diff_a, dtype=np.int64)
    rank_a[np.argsort(-np.abs(diff_a))] = np.arange(len(diff_a))
    rank_b = np.empty_like(diff_b, dtype=np.int64)
    rank_b[np.argsort(-np.abs(diff_b))] = np.arange(len(diff_b))

    corpora = [
        ("tool", tool_texts, A_tool, B_tool),
        ("nontool", nontool_texts, A_nontool, B_nontool),
    ]
    with open(path, "w") as f:
        for dataset, texts, A, B in corpora:
            for i, text in enumerate(texts):
                a_active = np.nonzero(A[i])[0]
                b_active = np.nonzero(B[i])[0]
                entries = []
                for local_idx in a_active:
                    entries.append(
                        {
                            "partition": "A",
                            "feature_idx": int(local_idx) + a_offset,
                            "activation": round(float(A[i, local_idx]), 6),
                            "overall_rank": int(rank_a[local_idx]),
                        }
                    )
                for local_idx in b_active:
                    entries.append(
                        {
                            "partition": "B",
                            "feature_idx": int(local_idx) + b_offset,
                            "activation": round(float(B[i, local_idx]), 6),
                            "overall_rank": int(rank_b[local_idx]),
                        }
                    )
                entries.sort(key=lambda e: e["overall_rank"])
                record = {
                    "dataset": dataset,
                    "prompt_index": i,
                    "prompt": text,
                    "n_active_A": int(len(a_active)),
                    "n_active_B": int(len(b_active)),
                    "activations": entries,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_exclusive_activations(
    texts, tokenizer, model_a, model_b, crosscoder, device
):
    a_list, b_list = [], []
    n = len(texts)
    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  {i + 1}/{n}")
        try:
            input_ids = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            ).input_ids
            h_a = get_last_token_activation(model_a, input_ids, device)
            h_b = get_last_token_activation(model_b, input_ids, device)
            x = torch.stack([h_a, h_b], dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = crosscoder.encode(x)
            a_list.append(feats[0, : crosscoder.a_end].cpu().numpy())
            b_list.append(feats[0, crosscoder.a_end : crosscoder.b_end].cpu().numpy())
        except Exception as e:
            print(f"  warning: text {i} failed: {e}")
            continue
    if not a_list:
        raise RuntimeError("No activations collected — every text failed.")
    return np.stack(a_list), np.stack(b_list)


def write_ranking_csv(path, diff, mean_tool, mean_nontool, offset, k=None):
    order = np.argsort(-np.abs(diff))
    if k is not None:
        order = order[:k]
    with open(path, "w") as f:
        f.write("rank,feature_idx,diff,mean_tool,mean_nontool,sign\n")
        for rank, idx in enumerate(order):
            sign = "tool" if diff[idx] > 0 else ("nontool" if diff[idx] < 0 else "tie")
            f.write(
                f"{rank},{int(idx) + offset},{diff[idx]:.6f},"
                f"{mean_tool[idx]:.6f},{mean_nontool[idx]:.6f},{sign}\n"
            )


def plot_discrimination_bars(diff_a, diff_b, a_offset, b_offset, k, out_path):
    idx_a = np.argsort(-np.abs(diff_a))[:k]
    idx_b = np.argsort(-np.abs(diff_b))[:k]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].barh(range(k), diff_a[idx_a], color="steelblue", alpha=0.9)
    axes[0].set_yticks(range(k))
    axes[0].set_yticklabels([f"A-{int(i) + a_offset}" for i in idx_a], fontsize=9)
    axes[0].set_xlabel("Mean(tool) − Mean(non-tool)")
    axes[0].set_title("A-exclusive (ToolRL-specific)")
    axes[0].axvline(0, color="gray", linestyle="--")
    axes[1].barh(range(k), diff_b[idx_b], color="coral", alpha=0.9)
    axes[1].set_yticks(range(k))
    axes[1].set_yticklabels([f"B-{int(i) + b_offset}" for i in idx_b], fontsize=9)
    axes[1].set_xlabel("Mean(tool) − Mean(non-tool)")
    axes[1].set_title("B-exclusive (Base-specific)")
    axes[1].axvline(0, color="gray", linestyle="--")
    fig.text(
        0.5,
        -0.04,
        "Positive = tool-specific | Negative = non-tool-specific",
        ha="center",
        fontsize=9,
    )
    plt.suptitle("Top discriminative exclusive neurons", y=1.02)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_coactivation(arr_tool, arr_nontool, diff, offset, k, vlim, label, out_path):
    idx = np.argsort(-np.abs(diff))[:k]
    stacked = np.vstack([arr_nontool, arr_tool])
    sub = stacked[:, idx]
    corr = np.corrcoef(sub.T)
    np.fill_diagonal(corr, 0)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ticklabels = [f"{label}-{int(idx[i]) + offset}" for i in range(k)]
    ax.set_xticks(range(k))
    ax.set_xticklabels(ticklabels, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(ticklabels)
    ax.set_title(f"Co-activation of top-{k} {label}-exclusive features")
    plt.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_magnitude_boxplot(a_tool, a_nontool, b_tool, b_nontool, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].boxplot(
        [a_nontool.sum(1), a_tool.sum(1)],
        labels=["non-tool", "tool"],
        patch_artist=True,
    )
    axes[0].set_ylabel("Sum of activations")
    axes[0].set_title("A-exclusive magnitude per sample")
    axes[1].boxplot(
        [b_nontool.sum(1), b_tool.sum(1)],
        labels=["non-tool", "tool"],
        patch_artist=True,
    )
    axes[1].set_ylabel("Sum of activations")
    axes[1].set_title("B-exclusive magnitude per sample")
    plt.suptitle("Exclusive-partition magnitude by dataset", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = resolve_device(DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "CROSSCODER_NAME": CROSSCODER_NAME,
        "MODELS_JSONL": MODELS_JSONL,
        "MODEL_A_ID": MODEL_A_ID,
        "MODEL_B_ID": MODEL_B_ID,
        "HIDDEN_STATES_IDX": HIDDEN_STATES_IDX,
        "DEVICE": device,
        "N_PER_DATASET": N_PER_DATASET,
        "TOOL_DATASET": TOOL_DATASET,
        "TOOL_DATASET_SPLIT": TOOL_DATASET_SPLIT,
        "NONTOOL_DATASET": NONTOOL_DATASET,
        "NONTOOL_DATASET_CONFIG": NONTOOL_DATASET_CONFIG,
        "MAX_LENGTH": MAX_LENGTH,
        "TOP_K": TOP_K,
        "CORR_VLIM": CORR_VLIM,
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "SEED": SEED,
        "CACHE_ACTIVATIONS": CACHE_ACTIVATIONS,
    }
    with open(OUTPUT_DIR / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print(f"\nLoading crosscoder: {CROSSCODER_NAME}")
    entry = load_model_entry(MODELS_JSONL, CROSSCODER_NAME)
    crosscoder = load_crosscoder(entry, device=device)
    if crosscoder is None:
        raise SystemExit("Failed to load crosscoder.")
    if crosscoder.a_end == 0:
        raise SystemExit(
            f"Crosscoder {CROSSCODER_NAME} has no exclusive partitions "
            f"(a_end=0). Neuron identification requires a DFC variant."
        )
    print(
        f"  a_end={crosscoder.a_end}  b_end={crosscoder.b_end}  "
        f"dict_size={crosscoder.dict_size}"
    )

    print(f"\nLoading tokenizer: {MODEL_B_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("\nLoading LLMs")
    model_a = load_llm(MODEL_A_ID, device)
    model_b = load_llm(MODEL_B_ID, device)

    print()
    tool_texts = collect_texts_tool(N_PER_DATASET)
    nontool_texts = collect_texts_nontool(N_PER_DATASET)

    print("\nEncoding tool texts through crosscoder")
    A_tool, B_tool = collect_exclusive_activations(
        tool_texts, tokenizer, model_a, model_b, crosscoder, device
    )
    print("\nEncoding non-tool texts through crosscoder")
    A_nontool, B_nontool = collect_exclusive_activations(
        nontool_texts, tokenizer, model_a, model_b, crosscoder, device
    )

    print(
        f"\nShapes: A_tool={A_tool.shape}  A_nontool={A_nontool.shape}  "
        f"B_tool={B_tool.shape}  B_nontool={B_nontool.shape}"
    )

    diff_a = A_tool.mean(axis=0) - A_nontool.mean(axis=0)
    diff_b = B_tool.mean(axis=0) - B_nontool.mean(axis=0)
    mean_a_tool = A_tool.mean(axis=0)
    mean_a_nontool = A_nontool.mean(axis=0)
    mean_b_tool = B_tool.mean(axis=0)
    mean_b_nontool = B_nontool.mean(axis=0)

    a_offset = 0
    b_offset = crosscoder.a_end

    print(f"\nWriting outputs to {OUTPUT_DIR}")
    write_prompt_activations_jsonl(
        OUTPUT_DIR / "prompt_activations.jsonl",
        tool_texts,
        nontool_texts,
        A_tool,
        A_nontool,
        B_tool,
        B_nontool,
        diff_a,
        diff_b,
        a_offset,
        b_offset,
    )
    write_ranking_csv(
        OUTPUT_DIR / "tool_neurons_A_full.csv",
        diff_a,
        mean_a_tool,
        mean_a_nontool,
        a_offset,
    )
    write_ranking_csv(
        OUTPUT_DIR / "tool_neurons_B_full.csv",
        diff_b,
        mean_b_tool,
        mean_b_nontool,
        b_offset,
    )
    write_ranking_csv(
        OUTPUT_DIR / f"tool_neurons_A_top{TOP_K}.csv",
        diff_a,
        mean_a_tool,
        mean_a_nontool,
        a_offset,
        k=TOP_K,
    )
    write_ranking_csv(
        OUTPUT_DIR / f"tool_neurons_B_top{TOP_K}.csv",
        diff_b,
        mean_b_tool,
        mean_b_nontool,
        b_offset,
        k=TOP_K,
    )

    print(f"Writing plots to {OUTPUT_DIR}")
    plot_discrimination_bars(
        diff_a,
        diff_b,
        a_offset,
        b_offset,
        TOP_K,
        OUTPUT_DIR / "discrimination_bars.svg",
    )
    plot_coactivation(
        A_tool,
        A_nontool,
        diff_a,
        a_offset,
        TOP_K,
        CORR_VLIM,
        "A",
        OUTPUT_DIR / "coactivation_A.svg",
    )
    plot_coactivation(
        B_tool,
        B_nontool,
        diff_b,
        b_offset,
        TOP_K,
        CORR_VLIM,
        "B",
        OUTPUT_DIR / "coactivation_B.svg",
    )
    plot_magnitude_boxplot(
        A_tool, A_nontool, B_tool, B_nontool, OUTPUT_DIR / "exclusive_magnitude_box.svg"
    )

    if CACHE_ACTIVATIONS:
        np.savez_compressed(
            OUTPUT_DIR / "activations_cache.npz",
            A_tool=A_tool,
            A_nontool=A_nontool,
            B_tool=B_tool,
            B_nontool=B_nontool,
            diff_a=diff_a,
            diff_b=diff_b,
        )
        print(f"Wrote raw activations to {OUTPUT_DIR / 'activations_cache.npz'}")

    print("\nTop-10 A-exclusive (positive diff = tool-specific):")
    for r, i in enumerate(np.argsort(-np.abs(diff_a))[:10]):
        print(
            f"  #{r} A-{int(i) + a_offset}: diff={diff_a[i]:+.4f}  "
            f"tool={mean_a_tool[i]:.4f}  nontool={mean_a_nontool[i]:.4f}"
        )

    print("\nTop-10 B-exclusive:")
    for r, i in enumerate(np.argsort(-np.abs(diff_b))[:10]):
        print(
            f"  #{r} B-{int(i) + b_offset}: diff={diff_b[i]:+.4f}  "
            f"tool={mean_b_tool[i]:.4f}  nontool={mean_b_nontool[i]:.4f}"
        )

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
