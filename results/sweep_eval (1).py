"""
sweep_eval.py
=============
Evaluation pipeline for 48 crosscoder/DFC model variants.

Loads two LLMs once (ToolRL + Base Qwen2.5-3B) and runs behavioral
reconstruction evaluation across all model variants from a JSONL file.

Design decisions:
  - hidden_states[14]: output of transformer layer 13 (index 0 = embeddings,
    so layer N output is at index N+1). Empirically verified to give better
    reconstruction than hidden_states[13].
  - LLMs loaded in float16 to save VRAM; activations cast to float32 for
    crosscoder arithmetic. Pre/post delta metrics are unaffected by this.
  - encode() does NOT apply enc_mask — matches original dfc.py training code.
  - decode() applies dec_mask to zero out forbidden decoder weights.
  - score_response uses <tool_call> + JSON name matching (ToolRL dataset format).
  - weights_only=False for torch.load — safer across PyTorch versions.
  - Crash-safe: saves results to JSONL after every model variant completes.

Usage:
    python sweep_eval.py --mode test    # 5 examples per model, sanity check
    python sweep_eval.py --mode full    # 100 examples per model, full eval
    python sweep_eval.py --mode steer   # steering experiment on DFC models
    python sweep_eval.py --mode all     # all three sequentially
"""

import os
import json
import re
import random
import argparse
import time
from pathlib import Path
from copy import deepcopy

# Prevent hf_hub_download from hanging on the initial metadata HEAD request.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODELS_JSONL       = "models.jsonl"
TEST_OUTPUT_JSONL  = "results_test.jsonl"
FULL_OUTPUT_JSONL  = "results_full.jsonl"
STEER_OUTPUT_JSONL = "results_steer.jsonl"

MODEL_A_ID     = "chengq9/ToolRL-Qwen2.5-3B"
MODEL_B_ID     = "Qwen/Qwen2.5-3B"
DATASET_ID     = "emrecanacikgoz/ToolRL"

# hidden_states[14] = output of transformer layer 13.
# Index 0 is embeddings, so layer N output is at index N+1.
# Empirically verified: hidden_states[14] gives cosine ~0.43 vs ~0.33 for [13].
HIDDEN_STATES_IDX = 14

MAX_NEW_TOKENS  = 200
MAX_LENGTH      = 2048
SEED            = 42
TEST_N_EXAMPLES = 5
FULL_N_EXAMPLES = 100

# Steering: scale factors to apply to A-exclusive feature activations.
STEER_SCALE_FACTORS  = [2.0, 5.0, 10.0]
# None = steer all A-exclusive features.
# Set to e.g. [20, 665] to steer only the two dominant tool-use neurons
# identified in the autointerp analysis.
STEER_NEURON_INDICES = None


# ──────────────────────────────────────────────────────────────────────────────
# CROSSCODER CLASSES
# Both classes use unmasked encode() to match dfc.py training code.
# decode() applies dec_mask to enforce partition constraints.
# ──────────────────────────────────────────────────────────────────────────────

class CrossCoder(nn.Module):
    """
    Standard crosscoder — no exclusive feature partitions.
    Used for model variants where model_a_exclusive_pct == 0.
    n_a, n_b, a_end, b_end are all 0 for steering compatibility.
    """
    def __init__(self, activation_dim, dict_size, k):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k
        self.n_a   = 0
        self.n_b   = 0
        self.a_end = 0
        self.b_end = 0

        self.W_enc = nn.Parameter(
            torch.randn(2, activation_dim, dict_size) / (activation_dim ** 0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(
            torch.randn(dict_size, 2, activation_dim) / (dict_size ** 0.5)
        )
        self.b_dec = nn.Parameter(torch.zeros(2, activation_dim))

    def encode(self, x):
        """x: (B, 2, d) -> sparse features (B, dict_size). No mask applied."""
        pre = torch.einsum("bmd,mdf->bf", x, self.W_enc) + self.b_enc
        pre = F.relu(pre)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        features = torch.zeros_like(pre)
        features.scatter_(-1, topk_idx, topk_vals)
        return features

    def decode(self, features):
        """features: (B, dict_size) -> (B, 2, d)."""
        return torch.einsum("bf,fmd->bmd", features, self.W_dec) + self.b_dec

    def forward(self, x):
        features = self.encode(x)
        return self.decode(features), features


class DFCCrossCoder(nn.Module):
    """
    Dedicated Feature Crosscoder with A-exclusive and B-exclusive partitions.
    Used for model variants where model_a_exclusive_pct > 0.

    Feature layout:
      [0 ... a_end)         = A-exclusive (ToolRL only)
      [a_end ... b_end)     = B-exclusive (Base only)
      [b_end ... dict_size) = Shared (both models)

    encode(): unmasked — matches original dfc.py training code.
      enc_mask is stored as buffer for steering/analysis but NOT applied
      during the forward encode pass.

    decode(): dec_mask IS applied to zero out forbidden decoder weights,
      ensuring A-exclusive features cannot influence Model B's reconstruction
      and vice versa.
    """
    def __init__(self, activation_dim, dict_size, k,
                 model_a_exclusive_pct=0.05, model_b_exclusive_pct=0.05):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k
        self.n_a      = int(dict_size * model_a_exclusive_pct)
        self.n_b      = int(dict_size * model_b_exclusive_pct)
        self.n_shared = dict_size - self.n_a - self.n_b
        self.a_end    = self.n_a
        self.b_end    = self.n_a + self.n_b

        self.W_enc = nn.Parameter(
            torch.randn(2, activation_dim, dict_size) / (activation_dim ** 0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(
            torch.randn(dict_size, 2, activation_dim) / (dict_size ** 0.5)
        )
        self.b_dec = nn.Parameter(torch.zeros(2, activation_dim))

        # enc_mask: stored for reference and steering analysis.
        # NOT applied during encode() — matches training behaviour.
        enc_mask = torch.ones(2, dict_size)
        enc_mask[1, :self.a_end] = 0
        enc_mask[0, self.a_end:self.b_end] = 0
        self.register_buffer("enc_mask", enc_mask)

        # dec_mask: applied during decode() to enforce partition constraints.
        dec_mask = torch.ones(dict_size, 2)
        dec_mask[:self.a_end, 1] = 0
        dec_mask[self.a_end:self.b_end, 0] = 0
        self.register_buffer("dec_mask", dec_mask)
        self._apply_masks()

    @torch.no_grad()
    def _apply_masks(self):
        """Zero forbidden weights in W_enc and W_dec. Called once at init."""
        for m in range(2):
            self.W_enc.data[m] *= self.enc_mask[m].unsqueeze(0)
        for m in range(2):
            self.W_dec.data[:, m, :] *= self.dec_mask[:, m].unsqueeze(1)

    def encode(self, x):
        """
        x: (B, 2, d) -> sparse features (B, dict_size).
        No enc_mask applied — matches original dfc.py training code.
        """
        pre = torch.einsum("bmd,mdf->bf", x, self.W_enc) + self.b_enc
        pre = F.relu(pre)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        features = torch.zeros_like(pre)
        features.scatter_(-1, topk_idx, topk_vals)
        return features

    def decode(self, features):
        """features: (B, dict_size) -> (B, 2, d). dec_mask enforced."""
        W = self.W_dec * self.dec_mask.unsqueeze(-1)
        return torch.einsum("bf,fmd->bmd", features, W) + self.b_dec

    def forward(self, x):
        features = self.encode(x)
        return self.decode(features), features


# ──────────────────────────────────────────────────────────────────────────────
# CROSSCODER LOADER
# ──────────────────────────────────────────────────────────────────────────────

def load_crosscoder(model_entry, device="cuda"):
    """
    Download and instantiate a crosscoder from HuggingFace.
    Reads config.json to get architecture params, falling back to
    hyperparameters from the JSONL entry if config fields are missing.
    Returns None if download or instantiation fails.
    """
    name = model_entry["name"]
    hp   = model_entry["hyperparameters"]

    print(f"  ⟳ Loading: {name}")

    try:
        # Try local cache first — avoids the network HEAD request that can hang.
        try:
            model_pt    = hf_hub_download(repo_id=name, filename="model.pt",    local_files_only=True)
            config_json = hf_hub_download(repo_id=name, filename="config.json", local_files_only=True)
        except Exception:
            model_pt    = hf_hub_download(repo_id=name, filename="model.pt")
            config_json = hf_hub_download(repo_id=name, filename="config.json")
    except Exception as e:
        print(f"  ⚠️  Download failed: {e}")
        return None

    cc_dir = str(Path(model_pt).parent)

    with open(config_json) as f:
        cfg = json.load(f)

    activation_dim = cfg.get("activation_dim", hp["activation_dim"])
    dict_size      = cfg.get("dict_size",      hp["dict_size"])
    k              = cfg.get("k",              hp["k"])
    n_a = cfg.get("n_a", int(dict_size * hp["model_a_exclusive_pct"]))
    n_b = cfg.get("n_b", int(dict_size * hp["model_b_exclusive_pct"]))
    arch = hp.get("architecture", "CrossCoder")

    # weights_only=False for compatibility across PyTorch versions
    state_dict = torch.load(
        f"{cc_dir}/model.pt", map_location=device, weights_only=False
    )

    # Use the state dict itself to determine class — more reliable than config.
    if "enc_mask" in state_dict:
        # Derive n_a, n_b from dec_mask rather than trusting config values.
        dec_mask = state_dict["dec_mask"]  # (dict_size, 2)
        n_a = int((dec_mask[:, 1] == 0).sum().item())
        n_b = int((dec_mask[n_a:, 0] == 0).sum().item())
        cc = DFCCrossCoder(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            model_a_exclusive_pct=n_a / dict_size,
            model_b_exclusive_pct=n_b / dict_size,
        )
    else:
        cc = CrossCoder(activation_dim=activation_dim, dict_size=dict_size, k=k)

    cc.load_state_dict(state_dict)
    cc = cc.to(device).eval()

    print(f"     arch={arch}  dict={dict_size}  k={k}  n_a={n_a}  n_b={n_b}")
    return cc


# ──────────────────────────────────────────────────────────────────────────────
# LLM HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_last_token_activation(model, input_ids, device):
    """
    Extract last-token residual stream activation at layer 13.

    Uses hidden_states[14]: index 0 is embedding output, so layer N
    output is at index N+1. Layer 13 -> index 14.

    LLM runs in float16; activation cast to float32 for crosscoder.
    The float16->float32 cast is consistent across clean and patched
    conditions so delta metrics are unaffected.

    Returns: (2048,) float32
    """
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(device),
            output_hidden_states=True
        )
    hidden = out.hidden_states[HIDDEN_STATES_IDX]  # (1, T, 2048)
    return hidden[0, -1, :].float()                 # last token, fp32


def generate_clean(model, input_ids, tokenizer, device,
                   max_new_tokens=MAX_NEW_TOKENS):
    """Generate text without any activation patching."""
    with torch.no_grad():
        out = model.generate(
            input_ids.to(device),
            attention_mask=torch.ones_like(input_ids).to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def generate_with_patch(model, input_ids, patch_vector, tokenizer, device,
                        layer=13, max_new_tokens=MAX_NEW_TOKENS):
    """
    Generate text with the last-token activation at `layer` replaced by
    patch_vector during the prefill pass only.

    The patched_once flag ensures the hook fires exactly once on the first
    forward pass (prefill) and not during autoregressive decoding, which
    would corrupt generation by repeatedly overwriting activations.

    patch_vector is cast to float16 to match the model dtype.
    """
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
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id
        )
    handle.remove()
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def generate_with_steer(model, input_ids, crosscoder, h_a, h_b,
                        tokenizer, device, scale_factor=2.0,
                        neuron_indices=None, layer=13,
                        max_new_tokens=MAX_NEW_TOKENS):
    """
    Steering experiment: amplify A-exclusive features before decoding into
    Model B. Tests whether ToolRL-specific representations can be transferred
    to the Base model by artificially boosting their activations.

    Steps:
      1. Encode both models' activations to sparse features
      2. Scale up A-exclusive features (indices 0 -> crosscoder.a_end)
         either all or a specific subset (neuron_indices)
      3. Decode steered features
      4. Patch decoded Model B activation into Model B and generate

    For plain CrossCoders (a_end == 0), no steering is applied and
    the function falls back to standard patched generation.
    """
    x = torch.stack([h_a, h_b], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        features = crosscoder.encode(x)   # (1, dict_size)

        if crosscoder.a_end > 0:
            steered = features.clone()
            if neuron_indices is not None:
                # Steer only specified neurons (e.g. [20, 665] from autointerp)
                for idx in neuron_indices:
                    if idx < crosscoder.a_end:
                        steered[0, idx] *= scale_factor
            else:
                # Steer all A-exclusive features
                steered[0, :crosscoder.a_end] *= scale_factor
            recon = crosscoder.decode(steered)
        else:
            # Plain CrossCoder — no exclusive features to steer
            recon = crosscoder.decode(features)

    recon_b = recon[0, 1, :].float()
    return generate_with_patch(
        model, input_ids, recon_b, tokenizer, device,
        layer=layer, max_new_tokens=max_new_tokens
    )


# ──────────────────────────────────────────────────────────────────────────────
# SCORING
# Uses ToolRL dataset format: <tool_call> tag + JSON "name" field.
# ──────────────────────────────────────────────────────────────────────────────

def score_response(response, prompt):
    """
    Score a generated response on 3 metrics.

    format_accuracy:
      True if response contains <tool_call> AND a JSON "name" field.
      Matches the exact format used in ToolRL training data.

    tool_correctness:
      True if the called tool name fuzzy-matches one of the tools listed
      in the prompt's numbered tool list.

    overall_score (-1 to +2):
      +2 = format correct AND tool correct
       0 = format correct, wrong tool
      +1 = tool correct, wrong format
      -1 = neither
    """
    response_lower = response.lower()

    has_tool_call = "<tool_call>" in response_lower
    has_json_name = bool(re.search(r'"name"\s*:\s*"[^"]+"', response))
    format_ok     = has_tool_call and has_json_name

    tool_names_in_prompt = re.findall(
        r'(?:^|\n)\d+\.\s*Name:\s*(.+?)(?:\n|$)', prompt
    )
    tool_names_in_prompt = [t.strip().lower() for t in tool_names_in_prompt]

    called = re.search(r'"name"\s*:\s*"([^"]+)"', response)
    called_name = called.group(1).lower() if called else ""

    tool_correct = any(
        called_name == t or called_name in t or t in called_name
        for t in tool_names_in_prompt
    ) if tool_names_in_prompt and called_name else False

    if format_ok and tool_correct:
        overall = 2
    elif format_ok:
        overall = 0
    elif tool_correct:
        overall = 1
    else:
        overall = -1

    return dict(
        format_accuracy=format_ok,
        tool_correctness=tool_correct,
        overall_score=overall
    )


def extract_prompt(example):
    """
    Extract user-facing prompt from a ToolRL dataset example.
    Concatenates system and user turns; excludes final assistant turn
    (which is the ground truth label).
    """
    if "messages" in example and example["messages"]:
        msgs  = example["messages"]
        parts = []
        for msg in msgs:
            if isinstance(msg, dict):
                role    = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "system"):
                    parts.append(content)
                elif role == "assistant" and msg != msgs[-1]:
                    parts.append(content)
        return "\n".join(parts) if parts else None

    for field in ["prompt", "instruction", "input", "query"]:
        if field in example and example[field]:
            return str(example[field])
    return None


# ──────────────────────────────────────────────────────────────────────────────
# CORE EVAL FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def eval_crosscoder(crosscoder, examples, tokenizer, model_a, model_b, device,
                    mode="reconstruction", steer_scale=None, steer_neurons=None):
    """
    Evaluate a single crosscoder on a list of examples.

    mode="reconstruction":
      Generates clean and reconstructed responses for both models.
      Returns pre/post metrics for Model A and Model B.

    mode="steering":
      Generates clean Model B response and a steered version where
      A-exclusive features are amplified by steer_scale before decoding
      into Model B. Tests capability transfer from ToolRL to Base.

    Returns: dict of aggregated metrics ready to merge into JSONL entry.
    """
    results_a_clean   = []
    results_a_patched = []
    results_b_clean   = []
    results_b_patched = []
    skipped = 0

    for i, example in enumerate(examples):
        prompt = extract_prompt(example)
        if not prompt:
            skipped += 1
            continue

        try:
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            ).input_ids

            h_a = get_last_token_activation(model_a, input_ids, device)
            h_b = get_last_token_activation(model_b, input_ids, device)
            x   = torch.stack([h_a, h_b], dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                recon, _ = crosscoder(x)
            recon_a = recon[0, 0, :].float()
            recon_b = recon[0, 1, :].float()

            if mode == "reconstruction":
                resp_a_clean   = generate_clean(model_a, input_ids, tokenizer, device)
                resp_a_patched = generate_with_patch(
                    model_a, input_ids, recon_a, tokenizer, device
                )
                resp_b_clean   = generate_clean(model_b, input_ids, tokenizer, device)
                resp_b_patched = generate_with_patch(
                    model_b, input_ids, recon_b, tokenizer, device
                )
                results_a_clean.append(score_response(resp_a_clean,   prompt))
                results_a_patched.append(score_response(resp_a_patched, prompt))
                results_b_clean.append(score_response(resp_b_clean,   prompt))
                results_b_patched.append(score_response(resp_b_patched, prompt))

            elif mode == "steering":
                resp_b_clean   = generate_clean(model_b, input_ids, tokenizer, device)
                resp_b_steered = generate_with_steer(
                    model_b, input_ids, crosscoder, h_a, h_b,
                    tokenizer, device,
                    scale_factor=steer_scale,
                    neuron_indices=steer_neurons
                )
                results_b_clean.append(score_response(resp_b_clean,   prompt))
                results_b_patched.append(score_response(resp_b_steered, prompt))

        except Exception as e:
            print(f"    ⚠️  Example {i} failed: {e}")
            skipped += 1
            continue

    def agg(results):
        if not results:
            return dict(
                overall_score_mean=None, overall_score_std=None,
                format_accuracy=None, tool_correctness=None, n=0
            )
        overall = [r["overall_score"]    for r in results]
        fmt     = [r["format_accuracy"]  for r in results]
        tool    = [r["tool_correctness"] for r in results]
        return dict(
            overall_score_mean = round(float(np.mean(overall)), 4),
            overall_score_std  = round(float(np.std(overall)),  4),
            format_accuracy    = round(float(np.mean(fmt))  * 100, 1),
            tool_correctness   = round(float(np.mean(tool)) * 100, 1),
            n = len(results)
        )

    def delta(post, pre):
        if post is None or pre is None:
            return None
        return round(post - pre, 4)

    out = dict(skipped=skipped)

    if mode == "reconstruction":
        ac = agg(results_a_clean);   ap = agg(results_a_patched)
        bc = agg(results_b_clean);   bp = agg(results_b_patched)
        out.update({
            "overall_score_A_pre":       ac["overall_score_mean"],
            "overall_score_A_pre_std":   ac["overall_score_std"],
            "overall_score_A_post":      ap["overall_score_mean"],
            "overall_score_A_post_std":  ap["overall_score_std"],
            "overall_score_A_delta":     delta(ap["overall_score_mean"], ac["overall_score_mean"]),
            "format_accuracy_A_pre":     ac["format_accuracy"],
            "format_accuracy_A_post":    ap["format_accuracy"],
            "format_accuracy_A_delta":   delta(ap["format_accuracy"], ac["format_accuracy"]),
            "tool_correctness_A_pre":    ac["tool_correctness"],
            "tool_correctness_A_post":   ap["tool_correctness"],
            "tool_correctness_A_delta":  delta(ap["tool_correctness"], ac["tool_correctness"]),
            "overall_score_B_pre":       bc["overall_score_mean"],
            "overall_score_B_pre_std":   bc["overall_score_std"],
            "overall_score_B_post":      bp["overall_score_mean"],
            "overall_score_B_post_std":  bp["overall_score_std"],
            "overall_score_B_delta":     delta(bp["overall_score_mean"], bc["overall_score_mean"]),
            "format_accuracy_B_pre":     bc["format_accuracy"],
            "format_accuracy_B_post":    bp["format_accuracy"],
            "format_accuracy_B_delta":   delta(bp["format_accuracy"], bc["format_accuracy"]),
            "tool_correctness_B_pre":    bc["tool_correctness"],
            "tool_correctness_B_post":   bp["tool_correctness"],
            "tool_correctness_B_delta":  delta(bp["tool_correctness"], bc["tool_correctness"]),
            "n_evaluated": ac["n"],
        })

    elif mode == "steering":
        bc = agg(results_b_clean)
        bs = agg(results_b_patched)
        out.update({
            "steer_scale_factor":             steer_scale,
            "steer_neuron_indices":           steer_neurons,
            "overall_score_B_baseline":       bc["overall_score_mean"],
            "overall_score_B_baseline_std":   bc["overall_score_std"],
            "overall_score_B_steered":        bs["overall_score_mean"],
            "overall_score_B_steered_std":    bs["overall_score_std"],
            "overall_score_B_steer_delta":    delta(bs["overall_score_mean"], bc["overall_score_mean"]),
            "format_accuracy_B_baseline":     bc["format_accuracy"],
            "format_accuracy_B_steered":      bs["format_accuracy"],
            "format_accuracy_B_steer_delta":  delta(bs["format_accuracy"], bc["format_accuracy"]),
            "tool_correctness_B_baseline":    bc["tool_correctness"],
            "tool_correctness_B_steered":     bs["tool_correctness"],
            "tool_correctness_B_steer_delta": delta(bs["tool_correctness"], bc["tool_correctness"]),
            "n_evaluated": bc["n"],
        })

    return out


# ──────────────────────────────────────────────────────────────────────────────
# TEST EVAL — 5 examples, all models
# ──────────────────────────────────────────────────────────────────────────────

def run_test_eval(model_entries, all_examples, tokenizer, model_a, model_b, device):
    """
    Quick sanity check: TEST_N_EXAMPLES examples through each model.
    Uses first N examples deterministically (no random sampling) so
    failures are reproducible. Saves after every model for crash safety.
    """
    print("\n" + "=" * 60)
    print(f"TEST EVAL — {TEST_N_EXAMPLES} examples x {len(model_entries)} models")
    print("=" * 60)

    # Resume: load existing results and skip already-evaluated models.
    existing = []
    if Path(TEST_OUTPUT_JSONL).exists():
        with open(TEST_OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
    done_names    = {e["name"] for e in existing}
    output_entries = existing[:]
    if done_names:
        print(f"  Resuming — {len(done_names)} already done, skipping.")

    test_examples  = all_examples[:TEST_N_EXAMPLES]

    for i, entry in enumerate(model_entries):
        print(f"\n[{i+1}/{len(model_entries)}] {entry['name']}")
        if entry["name"] in done_names:
            print("   skipped (already done)")
            continue
        result_entry = deepcopy(entry)

        try:
            cc = load_crosscoder(entry, device=device)
            if cc is None:
                result_entry["eval_error"] = "failed to load"
                output_entries.append(result_entry)
                continue

            metrics = eval_crosscoder(
                cc, test_examples, tokenizer, model_a, model_b, device,
                mode="reconstruction"
            )
            result_entry["test_eval"] = metrics
            print(f"   A: {metrics.get('overall_score_A_pre')} -> "
                  f"{metrics.get('overall_score_A_post')} "
                  f"(d{metrics.get('overall_score_A_delta')})")
            print(f"   B: {metrics.get('overall_score_B_pre')} -> "
                  f"{metrics.get('overall_score_B_post')} "
                  f"(d{metrics.get('overall_score_B_delta')})")

            del cc
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ERROR: {e}")
            result_entry["eval_error"] = str(e)

        output_entries.append(result_entry)
        with open(TEST_OUTPUT_JSONL, "w") as f:
            for e in output_entries:
                f.write(json.dumps(e) + "\n")

    print(f"\nTest eval complete -> {TEST_OUTPUT_JSONL}")
    return output_entries


# ──────────────────────────────────────────────────────────────────────────────
# FULL EVAL — 100 examples, all models
# ──────────────────────────────────────────────────────────────────────────────

def run_full_eval(model_entries, all_examples, tokenizer, model_a, model_b, device):
    """
    Full evaluation: FULL_N_EXAMPLES examples x all model variants.

    Output JSONL has original fields + all eval metric fields merged at
    top level, matching format request:
      { "name": ..., "hyperparameters": ..., "overall_score_A_pre": ..., ... }

    Random sample seeded for reproducibility. Saves after every model.
    """
    print("\n" + "=" * 60)
    print(f"FULL EVAL — {FULL_N_EXAMPLES} examples x {len(model_entries)} models")
    print("=" * 60)

    # Resume: load existing results and skip already-evaluated models.
    existing = []
    if Path(FULL_OUTPUT_JSONL).exists():
        with open(FULL_OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
    done_names     = {e["name"] for e in existing}
    output_entries = existing[:]
    if done_names:
        print(f"  Resuming — {len(done_names)} already done, skipping.")

    random.seed(SEED)
    eval_examples  = random.sample(all_examples, min(FULL_N_EXAMPLES, len(all_examples)))

    for i, entry in enumerate(model_entries):
        print(f"\n[{i+1}/{len(model_entries)}] {entry['name']}")
        if entry["name"] in done_names:
            print("   skipped (already done)")
            continue
        t0 = time.time()
        result_entry = deepcopy(entry)

        try:
            cc = load_crosscoder(entry, device=device)
            if cc is None:
                result_entry["eval_error"] = "failed to load"
                output_entries.append(result_entry)
                continue

            metrics = eval_crosscoder(
                cc, eval_examples, tokenizer, model_a, model_b, device,
                mode="reconstruction"
            )
            result_entry.update(metrics)

            elapsed = time.time() - t0
            print(f"   Done in {elapsed:.0f}s")
            print(f"   A: {metrics.get('overall_score_A_pre')}+/-"
                  f"{metrics.get('overall_score_A_pre_std')} -> "
                  f"{metrics.get('overall_score_A_post')}+/-"
                  f"{metrics.get('overall_score_A_post_std')} "
                  f"(d{metrics.get('overall_score_A_delta')})")
            print(f"   B: {metrics.get('overall_score_B_pre')}+/-"
                  f"{metrics.get('overall_score_B_pre_std')} -> "
                  f"{metrics.get('overall_score_B_post')}+/-"
                  f"{metrics.get('overall_score_B_post_std')} "
                  f"(d{metrics.get('overall_score_B_delta')})")

            del cc
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ERROR: {e}")
            result_entry["eval_error"] = str(e)

        output_entries.append(result_entry)
        with open(FULL_OUTPUT_JSONL, "w") as f:
            for e in output_entries:
                f.write(json.dumps(e) + "\n")

    print(f"\nFull eval complete -> {FULL_OUTPUT_JSONL}")
    return output_entries


# ──────────────────────────────────────────────────────────────────────────────
# STEERING EVAL
# ──────────────────────────────────────────────────────────────────────────────

def run_steer_eval(model_entries, all_examples, tokenizer, model_a, model_b, device):
    """
    Steering experiment: for each DFC crosscoder, amplify A-exclusive features
    at multiple scale factors and measure whether Base model improves on
    tool-calling metrics (overstimulation hypothesis).

    Only runs on DFC models (model_a_exclusive_pct > 0).
    Plain CrossCoders are skipped — no exclusive features to steer.

    One JSONL entry per (model, scale_factor) combination.
    """
    print("\n" + "=" * 60)
    print(f"STEER EVAL — {len(STEER_SCALE_FACTORS)} scale factors x DFC models only")
    print(f"Scale factors  : {STEER_SCALE_FACTORS}")
    print(f"Neuron indices : {STEER_NEURON_INDICES if STEER_NEURON_INDICES else 'all A-exclusive'}")
    print("=" * 60)

    # Resume: load existing results and skip already-evaluated (model, scale) pairs.
    existing = []
    if Path(STEER_OUTPUT_JSONL).exists():
        with open(STEER_OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
    done_pairs     = {(e["name"], e.get("scale_factor")) for e in existing}
    output_entries = existing[:]
    if done_pairs:
        print(f"  Resuming — {len(done_pairs)} (model, scale) pairs already done, skipping.")

    random.seed(SEED)
    steer_examples = random.sample(all_examples, min(FULL_N_EXAMPLES, len(all_examples)))

    for i, entry in enumerate(model_entries):
        if entry["hyperparameters"]["model_a_exclusive_pct"] == 0.0:
            print(f"\n[{i+1}/{len(model_entries)}] {entry['name']} — skipped (no exclusive features)")
            continue

        pending_scales = [s for s in STEER_SCALE_FACTORS
                          if (entry["name"], s) not in done_pairs]
        if not pending_scales:
            print(f"\n[{i+1}/{len(model_entries)}] {entry['name']} — skipped (all scales done)")
            continue

        print(f"\n[{i+1}/{len(model_entries)}] {entry['name']}")

        try:
            cc = load_crosscoder(entry, device=device)
            if cc is None:
                continue

            for scale in pending_scales:
                print(f"   Scale {scale}x ...")
                steer_entry = deepcopy(entry)

                metrics = eval_crosscoder(
                    cc, steer_examples, tokenizer, model_a, model_b, device,
                    mode="steering",
                    steer_scale=scale,
                    steer_neurons=STEER_NEURON_INDICES
                )
                steer_entry["scale_factor"] = scale
                steer_entry.update(metrics)
                output_entries.append(steer_entry)

                print(f"   B: {metrics.get('overall_score_B_baseline')} -> "
                      f"{metrics.get('overall_score_B_steered')} "
                      f"(d{metrics.get('overall_score_B_steer_delta')})  "
                      f"format: {metrics.get('format_accuracy_B_baseline')}% -> "
                      f"{metrics.get('format_accuracy_B_steered')}%  "
                      f"tool: {metrics.get('tool_correctness_B_baseline')}% -> "
                      f"{metrics.get('tool_correctness_B_steered')}%")

                with open(STEER_OUTPUT_JSONL, "w") as f:
                    for e in output_entries:
                        f.write(json.dumps(e) + "\n")

            del cc
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ERROR: {e}")

    print(f"\nSteering eval complete -> {STEER_OUTPUT_JSONL}")
    return output_entries


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crosscoder sweep evaluation")
    parser.add_argument(
        "--mode",
        choices=["test", "full", "steer", "all"],
        default="test",
        help="test=5 examples sanity check | full=100 examples | steer=steering | all=all three"
    )
    parser.add_argument("--models", default=MODELS_JSONL, help="Path to models JSONL")
    parser.add_argument("--device", default="cuda",       help="cuda or cpu")
    args = parser.parse_args()

    device = args.device
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model entries from JSONL
    print(f"\nReading model list from {args.models} ...")
    model_entries = []
    with open(args.models) as f:
        for line in f:
            line = line.strip()
            if line:
                model_entries.append(json.loads(line))
    print(f"  {len(model_entries)} model variants")

    # Load dataset
    print(f"\nLoading dataset {DATASET_ID} ...")
    try:
        dataset = load_dataset(DATASET_ID, split="test")
    except Exception:
        dataset = load_dataset(DATASET_ID, split="train")
    all_examples = list(dataset)
    print(f"  {len(all_examples)} examples")

    # Load tokenizer
    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_B_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load both LLMs once — kept in memory for the entire sweep.
    # Each crosscoder is loaded, evaluated, and deleted per variant.
    print(f"\nLoading Model A: {MODEL_A_ID} ...")
    t0 = time.time()
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A_ID, torch_dtype=torch.float16, device_map=device
    ).eval()
    print(f"  Done ({time.time()-t0:.1f}s)")

    print(f"\nLoading Model B: {MODEL_B_ID} ...")
    t0 = time.time()
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B_ID, torch_dtype=torch.float16, device_map=device
    ).eval()
    print(f"  Done ({time.time()-t0:.1f}s)")

    if device == "cuda":
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  VRAM after LLM load: {used:.1f} / {total:.1f} GB")

    # Dispatch
    if args.mode in ("test", "all"):
        run_test_eval(model_entries, all_examples, tokenizer, model_a, model_b, device)

    if args.mode in ("full", "all"):
        run_full_eval(model_entries, all_examples, tokenizer, model_a, model_b, device)

    if args.mode in ("steer", "all"):
        run_steer_eval(model_entries, all_examples, tokenizer, model_a, model_b, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
