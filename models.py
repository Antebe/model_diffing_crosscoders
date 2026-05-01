"""
models.py — Load and unload Hugging Face models for DFC CrossCoder.

Handles Model A (ToolRL fine-tuned), Model B (base), and the tokenizer.
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config


def load_both_models(cfg: Config):
    """
    Load HF models and tokenizer. Returns (model_a, model_b, tokenizer).
    
    Args:
        cfg: Config object with model names and device settings
        
    Returns:
        Tuple of (model_a, model_b, tokenizer)
    """
    print(f"Loading tokenizer from {cfg.tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for decoder-only models

    print(f"Loading Model A (ToolRL): {cfg.model_a_name} → {cfg.device_a}")
    model_a = AutoModelForCausalLM.from_pretrained(
        cfg.model_a_name, 
        torch_dtype=torch.float32,
        device_map=None
    ).to(cfg.device_a).eval()

    print(f"Loading Model B (Base): {cfg.model_b_name} → {cfg.device_b}")
    model_b = AutoModelForCausalLM.from_pretrained(
        cfg.model_b_name, 
        torch_dtype=torch.float32,
        device_map=None
    ).to(cfg.device_b).eval()

    # Auto-detect activation dim from model config
    activation_dim = model_a.config.hidden_size
    if cfg.activation_dim != activation_dim:
        print(f"[Config] Updating activation_dim: {cfg.activation_dim} → {activation_dim}")
        cfg.activation_dim = activation_dim

    print(f"[Models] Loaded successfully. Activation dim: {activation_dim}")
    return model_a, model_b, tokenizer


def unload_models(model_a, model_b):
    """
    Free GPU memory by moving models to CPU and clearing cache.
    
    Args:
        model_a: First model to unload
        model_b: Second model to unload
    """
    print("\n[Models] Unloading models and freeing GPU memory ...")
    
    # Move to CPU
    model_a.cpu()
    model_b.cpu()
    
    # Delete references
    del model_a, model_b
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    print("[Models] Memory cleanup complete.")


def get_device_info():
    """
    Print GPU information for debugging.
    """
    if not torch.cuda.is_available():
        print("[GPU] CUDA not available")
        return
        
    print(f"[GPU] Found {torch.cuda.device_count()} CUDA device(s):")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  cuda:{i} - {props.name} ({mem_gb:.1f} GB)")
        
        # Memory usage
        if torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    Allocated: {allocated:.2f} GB | Cached: {cached:.2f} GB")