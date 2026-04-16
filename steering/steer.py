"""
steer.py

Steer specific crosscoder neurons during generation and compare against
unsteered (clean) completions. Two modes:

  scale  — multiply selected feature activations by SCALE_FACTOR
  clamp  — set selected feature activations to CLAMP_VALUE

All configuration is read from environment variables.
Set them in steering/steer.sh rather than editing this file.

Output: a JSONL file where each line contains the prompt, clean completion,
steered completion, and scoring metadata.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _env(name, default, cast=str):
    raw = os.environ.get(name, default)
    return cast(raw) if cast is not str else raw


# ─── Config from env ─────────────────────────────────────────────────────────

CROSSCODER_NAME = _env("CROSSCODER_NAME", "antebe1/dfc-D8k-excl10-freeexcl-k160")
MODELS_JSONL = _env("MODELS_JSONL", str(REPO_ROOT / "models.jsonl"))
MODEL_A_ID = _env("MODEL_A_ID", "chengq9/ToolRL-Qwen2.5-3B")
MODEL_B_ID = _env("MODEL_B_ID", "Qwen/Qwen2.5-3B")
HIDDEN_STATES_IDX = _env("HIDDEN_STATES_IDX", "14", int)
MAX_LENGTH = _env("MAX_LENGTH", "512", int)
DEVICE = _env("DEVICE", "mps")
MAX_NEW_TOKENS = _env("MAX_NEW_TOKENS", "200", int)
LAYER = _env("LAYER", "13", int)

# Steering config
STEER_MODE = _env("STEER_MODE", "scale")  # "scale" or "clamp"
NEURON_INDICES = _env("NEURON_INDICES", "")  # comma-separated global feature indices
SCALE_FACTOR = _env("SCALE_FACTOR", "2.0", float)
CLAMP_VALUE = _env("CLAMP_VALUE", "1.0", float)

# Custom prompt (overrides dataset when non-empty)
CUSTOM_PROMPT = _env("CUSTOM_PROMPT", "")

# Dataset (ignored when CUSTOM_PROMPT is set)
DATASET = _env("DATASET", "emrecanacikgoz/ToolRL")
DATASET_SPLIT = _env("DATASET_SPLIT", "train")
DATASET_CONFIG = _env("DATASET_CONFIG", "")
N_PROMPTS = _env("N_PROMPTS", "50", int)

OUTPUT_DIR = Path(_env("OUTPUT_DIR", str(REPO_ROOT / "steering" / "runs" / "default")))
SEED = _env("SEED", "42", int)

# Reuse repo utilities
import sweep_eval  # noqa: E402

sweep_eval.HIDDEN_STATES_IDX = HIDDEN_STATES_IDX
from sweep_eval import (  # noqa: E402
    extract_prompt,
    get_last_token_activation,
    load_crosscoder,
    score_response,
)

import transformers.pytorch_utils as _pu  # noqa: E402
import transformers.generation.utils as _gu  # noqa: E402

_orig_isin_mps = _pu.isin_mps_friendly

def _patched_isin_mps(elements, test_elements):
    if elements.dim() == 0:
        elements = elements.unsqueeze(0)
    if test_elements.dim() == 0:
        test_elements = test_elements.unsqueeze(0)
    return _orig_isin_mps(elements, test_elements)

_pu.isin_mps_friendly = _patched_isin_mps
_gu.isin_mps_friendly = _patched_isin_mps


def generate_clean(model, input_ids, tokenizer, device, max_new_tokens=200):
    with torch.no_grad():
        out = model.generate(
            input_ids.to(device),
            attention_mask=torch.ones_like(input_ids).to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def generate_with_patch(model, input_ids, patch_vector, tokenizer, device,
                        layer=13, max_new_tokens=200):
    patch = patch_vector.to(device).half()
    patched_once = [False]

    def patch_hook(module, inp, out):
        if not patched_once[0]:
            hidden = out[0] if isinstance(out, tuple) else out
            patched = hidden.clone()
            patched[0, -1, :] = patch
            patched_once[0] = True
            if isinstance(out, tuple):
                return (patched,) + out[1:]
            return patched

    handle = model.model.layers[layer].register_forward_hook(patch_hook)
    with torch.no_grad():
        out = model.generate(
            input_ids.to(device),
            attention_mask=torch.ones_like(input_ids).to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    handle.remove()
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def parse_neuron_indices(raw):
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def collect_prompts(n):
    print(f"Loading dataset: {DATASET} (split={DATASET_SPLIT})")
    kwargs = {}
    if DATASET_CONFIG:
        kwargs["name"] = DATASET_CONFIG

    ds = None
    for split in (DATASET_SPLIT, "train"):
        try:
            ds = load_dataset(DATASET, split=split, **kwargs)
            break
        except Exception:
            continue
    if ds is None:
        try:
            stream = load_dataset(DATASET, split="train", streaming=True, **kwargs)
            stream = stream.shuffle(seed=SEED, buffer_size=max(10_000, n * 10))
            texts = []
            for rec in stream:
                if len(texts) >= n:
                    break
                t = extract_prompt(rec)
                if not t:
                    for key in ("text", "content", "passage"):
                        if key in rec and rec[key]:
                            s = str(rec[key]).strip()
                            if len(s) > 50:
                                t = s
                                break
                if t:
                    texts.append(t)
            print(f"  collected {len(texts)} prompts (streamed)")
            return texts
        except Exception as e:
            raise RuntimeError(f"Could not load dataset {DATASET}: {e}")

    ds = ds.shuffle(seed=SEED)
    texts = []
    for rec in ds:
        if len(texts) >= n:
            break
        t = extract_prompt(rec)
        if not t:
            for key in ("text", "content", "passage"):
                if key in rec and rec[key]:
                    s = str(rec[key]).strip()
                    if len(s) > 50:
                        t = s
                        break
        if t:
            texts.append(t)
    print(f"  collected {len(texts)} prompts (shuffled, seed={SEED})")
    return texts


def compute_steering_delta(h_b, features, crosscoder, neuron_indices, mode,
                           scale_factor, clamp_value, device):
    """Compute the steering perturbation to add to the original h_b.

    Instead of full decode (which introduces reconstruction error from all
    features), we compute only the delta contributed by the steered neurons:

      scale: delta_i = (scale_factor - 1) * a_i * W_dec[i, 1, :]
      clamp: delta_i = (clamp_value  - a_i) * W_dec[i, 1, :]

    Returns (h_b + sum(delta_i), original_values, steered_values).
    """
    W_dec = crosscoder.W_dec.data  # (dict_size, 2, activation_dim)
    if hasattr(crosscoder, "dec_mask"):
        W_dec = W_dec * crosscoder.dec_mask.unsqueeze(-1)

    delta = torch.zeros_like(h_b)
    original_values = {}
    steered_values = {}

    for idx in neuron_indices:
        if idx < 0 or idx >= crosscoder.dict_size:
            continue
        a_i = features[0, idx]
        original_values[idx] = float(a_i.item())

        if mode == "scale":
            coeff = (scale_factor - 1.0) * a_i
            steered_values[idx] = float((a_i * scale_factor).item())
        elif mode == "clamp":
            coeff = clamp_value - a_i
            steered_values[idx] = float(clamp_value)

        # W_dec[idx, 1, :] is the decoder direction for Model B
        delta += coeff * W_dec[idx, 1, :]

    return h_b + delta, original_values, steered_values


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    neuron_indices = parse_neuron_indices(NEURON_INDICES)
    if not neuron_indices:
        raise SystemExit(
            "NEURON_INDICES is empty. Provide comma-separated global feature "
            "indices (e.g. NEURON_INDICES=\"5,12,100\")."
        )

    if STEER_MODE not in ("scale", "clamp"):
        raise SystemExit(f"STEER_MODE must be 'scale' or 'clamp', got '{STEER_MODE}'")

    device = resolve_device(DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "CROSSCODER_NAME": CROSSCODER_NAME,
        "MODELS_JSONL": MODELS_JSONL,
        "MODEL_A_ID": MODEL_A_ID,
        "MODEL_B_ID": MODEL_B_ID,
        "HIDDEN_STATES_IDX": HIDDEN_STATES_IDX,
        "MAX_LENGTH": MAX_LENGTH,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "LAYER": LAYER,
        "DEVICE": device,
        "STEER_MODE": STEER_MODE,
        "NEURON_INDICES": neuron_indices,
        "SCALE_FACTOR": SCALE_FACTOR if STEER_MODE == "scale" else None,
        "CLAMP_VALUE": CLAMP_VALUE if STEER_MODE == "clamp" else None,
        "CUSTOM_PROMPT": CUSTOM_PROMPT if CUSTOM_PROMPT else None,
        "DATASET": DATASET,
        "DATASET_SPLIT": DATASET_SPLIT,
        "DATASET_CONFIG": DATASET_CONFIG,
        "N_PROMPTS": N_PROMPTS,
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "SEED": SEED,
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
    print(
        f"  a_end={crosscoder.a_end}  b_end={crosscoder.b_end}  "
        f"dict_size={crosscoder.dict_size}"
    )

    for idx in neuron_indices:
        if idx < 0 or idx >= crosscoder.dict_size:
            print(f"  WARNING: neuron index {idx} is out of range [0, {crosscoder.dict_size})")
        elif idx < crosscoder.a_end:
            print(f"  neuron {idx}: A-exclusive")
        elif idx < crosscoder.b_end:
            print(f"  neuron {idx}: B-exclusive")
        else:
            print(f"  neuron {idx}: shared")

    print(f"\nLoading tokenizer: {MODEL_B_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("\nLoading LLMs")
    model_a = load_llm(MODEL_A_ID, device)
    model_b = load_llm(MODEL_B_ID, device)

    if CUSTOM_PROMPT:
        prompts = [CUSTOM_PROMPT]
        print(f"\nUsing custom prompt ({len(CUSTOM_PROMPT)} chars)")
    else:
        prompts = collect_prompts(N_PROMPTS)

    out_path = OUTPUT_DIR / "steering_results.jsonl"
    print(f"\nRunning steering ({STEER_MODE}) on {len(prompts)} prompts ...")
    print(f"  writing to {out_path}")

    with open(out_path, "w") as f:
        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0 or i == len(prompts) - 1:
                print(f"  {i + 1}/{len(prompts)}")

            try:
                input_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                ).input_ids

                # Clean completion from Model B (no steering)
                clean_text = generate_clean(
                    model_b, input_ids, tokenizer, device,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                # Get activations from both models
                h_a = get_last_token_activation(model_a, input_ids, device)
                h_b = get_last_token_activation(model_b, input_ids, device)
                x = torch.stack([h_a, h_b], dim=0).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = crosscoder.encode(x)

                    # Compute steering delta and add to original h_b.
                    # This avoids reconstruction error: we never fully decode,
                    # only add the perturbation from the steered neurons.
                    patched_b, original_values, steered_values = compute_steering_delta(
                        h_b, features, crosscoder, neuron_indices,
                        STEER_MODE, SCALE_FACTOR, CLAMP_VALUE, device,
                    )

                # Patch the steered h_b into Model B and generate
                steered_text = generate_with_patch(
                    model_b, input_ids, patched_b, tokenizer, device,
                    layer=LAYER, max_new_tokens=MAX_NEW_TOKENS,
                )

                # Score both completions (works for ToolRL prompts; harmless on others)
                clean_score = score_response(clean_text, prompt)
                steered_score = score_response(steered_text, prompt)

                record = {
                    "prompt_index": i,
                    "prompt": prompt,
                    "clean_completion": clean_text,
                    "steered_completion": steered_text,
                    "clean_score": clean_score,
                    "steered_score": steered_score,
                    "neuron_activations": {
                        "original": {str(k): v for k, v in original_values.items()},
                        "steered": {str(k): v for k, v in steered_values.items()},
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"  warning: prompt {i} failed: {e}")
                record = {
                    "prompt_index": i,
                    "prompt": prompt,
                    "error": str(e),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)

    n_ok = len(results)
    n_err = len(prompts) - n_ok
    print(f"  completed: {n_ok}   errors: {n_err}")

    if results:
        clean_scores = [r["clean_score"]["overall_score"] for r in results]
        steered_scores = [r["steered_score"]["overall_score"] for r in results]
        clean_fmt = sum(1 for r in results if r["clean_score"]["format_accuracy"])
        steered_fmt = sum(1 for r in results if r["steered_score"]["format_accuracy"])
        clean_tool = sum(1 for r in results if r["clean_score"]["tool_correctness"])
        steered_tool = sum(1 for r in results if r["steered_score"]["tool_correctness"])

        print(f"\n  format_accuracy:   clean={clean_fmt}/{n_ok}  steered={steered_fmt}/{n_ok}")
        print(f"  tool_correctness:  clean={clean_tool}/{n_ok}  steered={steered_tool}/{n_ok}")
        print(f"  overall_score:     clean={sum(clean_scores)/n_ok:.2f}  steered={sum(steered_scores)/n_ok:.2f}")

    print(f"\nDone. Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
