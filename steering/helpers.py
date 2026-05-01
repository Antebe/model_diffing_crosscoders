"""Steering helpers for the DFC CrossCoder.

Provides functions to:
- Extract layer activations from both models
- Generate text with activation-space steering hooks
- Build steering vectors from DFC decoder weights
- Classify features by partition
"""

from __future__ import annotations

import sys
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing from parent package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dfc import DFCCrossCoder
from config import Config


def load_all(
    checkpoint: str = "../checkpoints/dfc2",
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM, DFCCrossCoder, Config]:
    """Load tokenizer, both models, and the trained DFC.

    Returns:
        (tokenizer, model_a, model_b, dfc, cfg)
    """
    cfg = Config()

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Model A (ToolRL)...")
    model_a = AutoModelForCausalLM.from_pretrained(
        cfg.model_a_name, torch_dtype=dtype
    ).to(device).eval()

    print("Loading Model B (Base)...")
    model_b = AutoModelForCausalLM.from_pretrained(
        cfg.model_b_name, torch_dtype=dtype
    ).to(device).eval()

    print("Loading DFC CrossCoder...")
    ckpt_path = str(Path(__file__).resolve().parent / checkpoint)
    dfc = DFCCrossCoder.load(ckpt_path, device=device)

    print(
        f"DFC layout: A-exclusive [0:{dfc.a_end}], "
        f"B-exclusive [{dfc.a_end}:{dfc.b_end}], "
        f"Shared [{dfc.b_end}:{dfc.dict_size}]"
    )
    return tokenizer, model_a, model_b, dfc, cfg


def feature_partition(idx: int, dfc: DFCCrossCoder) -> str:
    """Return which partition a feature index belongs to."""
    if idx < dfc.a_end:
        return "A-exclusive (ToolRL)"
    elif idx < dfc.b_end:
        return "B-exclusive (Base)"
    return "Shared"


@torch.no_grad()
def get_activations(
    text: str,
    tokenizer: AutoTokenizer,
    model_a: AutoModelForCausalLM,
    model_b: AutoModelForCausalLM,
    layer_idx: int,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Extract last-token hidden states at *layer_idx* from both models.

    Returns:
        Tensor of shape ``(1, 2, hidden_dim)`` on *device*.
    """
    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    out_a = model_a(**tokens, output_hidden_states=True)
    out_b = model_b(**tokens, output_hidden_states=True)

    h_a = out_a.hidden_states[layer_idx + 1][0, -1]  # (d,)
    h_b = out_b.hidden_states[layer_idx + 1][0, -1]  # (d,)
    return torch.stack([h_a, h_b], dim=0).unsqueeze(0)  # (1, 2, d)


@torch.no_grad()
def generate_with_hook(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layer_idx: int,
    steering_delta: torch.Tensor | None = None,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text, optionally adding *steering_delta* to the target layer.

    The Qwen2 decoder layer returns a plain ``torch.Tensor``, so the hook
    simply adds the delta.  Tuple and dataclass outputs are handled as
    fallbacks for other architectures.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        text: Input prompt.
        layer_idx: Which transformer layer to hook.
        steering_delta: ``(hidden_dim,)`` vector added to the layer output.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling cutoff.

    Returns:
        Decoded string of newly generated tokens.
    """
    device = next(model.parameters()).device
    tokens = tokenizer(text, return_tensors="pt").to(device)

    hook_handle = None
    if steering_delta is not None:
        delta = steering_delta.to(device)
        target_layer = model.model.layers[layer_idx]

        def _hook(module, input, output):
            # Qwen2DecoderLayer.forward → plain Tensor
            if isinstance(output, torch.Tensor):
                return output + delta
            # Older transformers → tuple (hidden, attn_weights, ...)
            if isinstance(output, tuple):
                return (output[0] + delta,) + tuple(output[1:])
            # Dataclass output
            out = copy.copy(output)
            out[0] = output[0] + delta
            return out

        hook_handle = target_layer.register_forward_hook(_hook)

    output_ids = model.generate(
        **tokens,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    if hook_handle is not None:
        hook_handle.remove()

    new_tokens = output_ids[0, tokens["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def generate_with_clamp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layer_idx: int,
    feature_idx: int,
    clamp_value: float,
    dfc: DFCCrossCoder,
    model_idx: int = 0,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text while clamping a DFC feature to a fixed value.

    At each forward pass through *layer_idx*, the hidden state's component
    along the feature's decoder direction is replaced so that the feature
    contribution equals ``clamp_value * decoder_dir``.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        text: Input prompt.
        layer_idx: Which transformer layer to hook.
        feature_idx: DFC feature index to clamp.
        clamp_value: Value to force the feature projection to.
        dfc: Trained DFCCrossCoder.
        model_idx: 0 for model A, 1 for model B decoder.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling cutoff.

    Returns:
        Decoded string of newly generated tokens.
    """
    device = next(model.parameters()).device
    tokens = tokenizer(text, return_tensors="pt").to(device)

    decoder_dir = dfc.W_dec[feature_idx, model_idx].to(device)  # (d,)
    dir_norm_sq = (decoder_dir @ decoder_dir).clamp(min=1e-10)
    target_layer = model.model.layers[layer_idx]

    def _hook(module, input, output):
        if isinstance(output, torch.Tensor):
            h = output
        elif isinstance(output, tuple):
            h = output[0]
        else:
            h = output[0]

        # Replace the component along decoder_dir with clamp_value * decoder_dir
        proj = (h @ decoder_dir).unsqueeze(-1) / dir_norm_sq  # (..., 1)
        h_new = h + (clamp_value - proj) * decoder_dir  # clamp projection

        if isinstance(output, torch.Tensor):
            return h_new
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        out = copy.copy(output)
        out[0] = h_new
        return out

    hook_handle = target_layer.register_forward_hook(_hook)

    output_ids = model.generate(
        **tokens,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    hook_handle.remove()
    new_tokens = output_ids[0, tokens["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_steering_vector(
    feature_scales: dict[int, float],
    dfc: DFCCrossCoder,
    model_idx: int = 0,
    verbose: bool = True,
) -> torch.Tensor:
    """Combine DFC decoder directions into a single steering vector.

    Args:
        feature_scales: ``{feature_index: scale}``.
            Positive = amplify, negative = suppress.
        dfc: Trained DFCCrossCoder.
        model_idx: 0 for model A, 1 for model B decoder.
        verbose: Print per-feature info.

    Returns:
        ``(hidden_dim,)`` steering vector on the same device as *dfc*.
    """
    device = dfc.W_dec.device
    steering = torch.zeros(dfc.activation_dim, device=device)
    for feat_idx, scale in feature_scales.items():
        decoder_dir = dfc.W_dec[feat_idx, model_idx]  # (d,)
        steering += scale * decoder_dir
        if verbose:
            part = feature_partition(feat_idx, dfc)
            print(
                f"  Feature {feat_idx} ({part}): "
                f"scale={scale:+.1f}, ||dec||={decoder_dir.norm():.4f}"
            )
    if verbose:
        print(f"\n  Combined steering vector norm: {steering.norm():.4f}")
    return steering


def build_clamped_vector(
    feature_clamps: dict[int, float],
    dfc: DFCCrossCoder,
    model_idx: int = 0,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build decoder directions and clamp targets for multi-feature clamping.

    Unlike ``build_steering_vector`` (which returns a static additive vector),
    clamping requires adjusting the hidden state based on its current projection
    onto each decoder direction.  This function returns the components needed
    to apply that adjustment efficiently inside a forward hook.

    The clamping correction for a hidden state *h* is::

        for each feature i:
            proj_i = (h @ dirs[i]) / (dirs[i] @ dirs[i])
            h = h + (targets[i] - proj_i) * dirs[i]

    Args:
        feature_clamps: ``{feature_index: clamp_value}``.
            Each feature's projection onto its decoder direction will be set
            to *clamp_value*.
        dfc: Trained DFCCrossCoder.
        model_idx: 0 for model A, 1 for model B decoder.
        verbose: Print per-feature info.

    Returns:
        ``(dirs, targets)`` where *dirs* is ``(n_features, hidden_dim)`` and
        *targets* is ``(n_features,)``, both on the same device as *dfc*.
    """
    device = dfc.W_dec.device
    dirs_list: list[torch.Tensor] = []
    targets_list: list[float] = []
    for feat_idx, clamp_value in feature_clamps.items():
        decoder_dir = dfc.W_dec[feat_idx, model_idx]  # (d,)
        dirs_list.append(decoder_dir)
        targets_list.append(clamp_value)
        if verbose:
            part = feature_partition(feat_idx, dfc)
            print(
                f"  Feature {feat_idx} ({part}): "
                f"clamp={clamp_value:+.1f}, ||dec||={decoder_dir.norm():.4f}"
            )
    dirs = torch.stack(dirs_list)  # (n_features, d)
    targets = torch.tensor(targets_list, device=device)  # (n_features,)
    if verbose:
        print(f"\n  Clamping {len(dirs_list)} features")
    return dirs, targets


def apply_clamped_vector(
    h: torch.Tensor,
    dirs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Apply clamping corrections to a hidden state.

    Args:
        h: Hidden state tensor ``(..., hidden_dim)``.
        dirs: Decoder directions ``(n_features, hidden_dim)`` from
            :func:`build_clamped_vector`.
        targets: Clamp targets ``(n_features,)`` from
            :func:`build_clamped_vector`.

    Returns:
        Adjusted hidden state with each feature projection set to its target.
    """
    dir_norm_sq = (dirs * dirs).sum(dim=-1).clamp(min=1e-10)  # (n_features,)
    projs = (h @ dirs.T) / dir_norm_sq  # (..., n_features)
    corrections = targets - projs  # (..., n_features)
    h = h + corrections @ dirs  # (..., d)
    return h


@torch.no_grad()
def generate_with_multi_clamp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layer_idx: int,
    feature_clamps: dict[int, float],
    dfc: DFCCrossCoder,
    model_idx: int = 0,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text while clamping multiple DFC features simultaneously.

    Combines :func:`build_clamped_vector` and :func:`apply_clamped_vector`
    into a single generation call, mirroring :func:`generate_with_hook` for
    steering but with per-feature clamping semantics.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        text: Input prompt.
        layer_idx: Which transformer layer to hook.
        feature_clamps: ``{feature_index: clamp_value}``.
        dfc: Trained DFCCrossCoder.
        model_idx: 0 for model A, 1 for model B decoder.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling cutoff.

    Returns:
        Decoded string of newly generated tokens.
    """
    device = next(model.parameters()).device
    tokens = tokenizer(text, return_tensors="pt").to(device)

    dirs, targets = build_clamped_vector(
        feature_clamps, dfc, model_idx=model_idx, verbose=False,
    )
    dirs = dirs.to(device)
    targets = targets.to(device)
    target_layer = model.model.layers[layer_idx]

    def _hook(module, input, output):
        if isinstance(output, torch.Tensor):
            h = output
        elif isinstance(output, tuple):
            h = output[0]
        else:
            h = output[0]

        h_new = apply_clamped_vector(h, dirs, targets)

        if isinstance(output, torch.Tensor):
            return h_new
        if isinstance(output, tuple):
            return (h_new,) + tuple(output[1:])
        out = copy.copy(output)
        out[0] = h_new
        return out

    hook_handle = target_layer.register_forward_hook(_hook)

    output_ids = model.generate(
        **tokens,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    hook_handle.remove()
    new_tokens = output_ids[0, tokens["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def top_features(
    features: torch.Tensor,
    dfc: DFCCrossCoder,
    k: int = 15,
) -> None:
    """Print the top-*k* active features with partition labels.

    Args:
        features: ``(dict_size,)`` activation vector (single sample).
        dfc: Trained DFCCrossCoder (for partition boundaries).
        k: How many to show.
    """
    top_vals, top_idx = torch.topk(features, k)
    print(f"{'Rank':>4}  {'Feat':>6}  {'Value':>8}  Partition")
    print("-" * 45)
    for rank, (v, i) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), 1):
        if v <= 0:
            break
        print(f"{rank:>4}  {i:>6}  {v:>8.4f}  {feature_partition(i, dfc)}")


def partition_counts(
    features: torch.Tensor,
    dfc: DFCCrossCoder,
) -> dict[str, int]:
    """Count active features per partition.

    Args:
        features: ``(dict_size,)`` activation vector.
        dfc: Trained DFCCrossCoder.

    Returns:
        Dict with keys ``"a_exclusive"``, ``"b_exclusive"``, ``"shared"``.
    """
    return {
        "a_exclusive": (features[:dfc.a_end] > 0).sum().item(),
        "b_exclusive": (features[dfc.a_end:dfc.b_end] > 0).sum().item(),
        "shared": (features[dfc.b_end:] > 0).sum().item(),
    }
