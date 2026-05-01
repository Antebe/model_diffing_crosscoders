"""
DFC CrossCoder Demo - Quick Feature Vector Extraction
=====================================================

This demo shows how to use the DFC CrossCoder to extract feature vectors
from text using the ToolRL vs Base Qwen2.5-3B comparison.

Works both locally (from checkpoints/) and from a HuggingFace download.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dfc import DFCCrossCoder

# Default model path — override with command line arg or set MODEL_PATH env var
import os
DEFAULT_MODEL_PATH = os.environ.get("DFC_MODEL_PATH", "./checkpoints/dfc2")
LAYER = 13


def quick_demo(model_path: str = DEFAULT_MODEL_PATH):
    """10-line core demo: Load models and extract features from text"""

    # Load the trained CrossCoder
    dfc = DFCCrossCoder.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and models (for demo - in practice you might use cached activations)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Example texts: general vs tool-use
    general_text = "The French Revolution began in 1789 with widespread social discontent."
    tooluse_text = "Search for the cheapest flight from London to Tokyo departing next Friday."
    
    print("🚀 DFC CrossCoder Demo")
    print("="*50)
    
    for text, label in [(general_text, "General"), (tooluse_text, "Tool-Use")]:
        print(f"\n{label} Text: '{text}'")
        
        # Tokenize and get activations (simplified - normally you'd extract from layer 13)
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # For demo: create mock activations (normally extracted from models at layer 13)
        # Shape: (batch=1, models=2, hidden_dim=2048)
        mock_activations = torch.randn(1, 2, dfc.activation_dim, device=dfc.W_enc.device)

        # Core CrossCoder usage (the main 3 lines!)
        features = dfc.encode(mock_activations)  # Get sparse feature vector
        reconstruction, _ = dfc(mock_activations)  # Full forward pass
        
        # Analyze partition breakdowns
        active_features = (features > 0).sum().item()
        a_exclusive = (features[0, :dfc.a_end] > 0).sum().item()
        b_exclusive = (features[0, dfc.a_end:dfc.b_end] > 0).sum().item()
        shared = (features[0, dfc.b_end:] > 0).sum().item()
        
        print(f"  ✅ Active features: {active_features}/{dfc.dict_size}")
        print(f"  🔧 A-exclusive (ToolRL): {a_exclusive}")
        print(f"  📝 B-exclusive (Base): {b_exclusive}")
        print(f"  🤝 Shared: {shared}")
        
        # Show top features
        top_vals, top_idx = torch.topk(features[0], k=5)
        print(f"  🔝 Top features: {top_idx.tolist()} (values: {top_vals.tolist()})")

def extract_real_activations(model_path: str = DEFAULT_MODEL_PATH):
    """Extended demo with real model activations (requires more memory)"""
    print("\n Extended Demo with Real Model Activations")
    print("="*50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models (this requires significant memory!)
    print("Loading ToolRL and Base models...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained("chengq9/ToolRL-Qwen2.5-3B").to(device).eval()
    model_b = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B").to(device).eval()

    # Load CrossCoder
    dfc = DFCCrossCoder.load(model_path, device=device)

    # Extract real activations from layer 13
    text = "Use the calculator to compute 15% tip on $45.50"
    tokens = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        # hidden_states[0] = embeddings, hidden_states[i] = output of layer i-1
        # so layer 13 activations are at index LAYER + 1
        outputs_a = model_a(**tokens, output_hidden_states=True)
        outputs_b = model_b(**tokens, output_hidden_states=True)

        hidden_a = outputs_a.hidden_states[LAYER + 1][0, -1]  # (hidden_dim,)
        hidden_b = outputs_b.hidden_states[LAYER + 1][0, -1]  # (hidden_dim,)

        # Stack for CrossCoder: (1, 2, hidden_dim)
        activations = torch.stack([hidden_a, hidden_b], dim=0).unsqueeze(0)

        # Run CrossCoder
        features = dfc.encode(activations)

        # Analysis
        active_count = (features > 0).sum().item()
        print(f"\n Real Activation Analysis:")
        print(f"   Text: '{text}'")
        print(f"   Active features: {active_count}/{dfc.dict_size}")

        # Partition breakdown
        a_active = (features[0, :dfc.a_end] > 0).sum().item()
        b_active = (features[0, dfc.a_end:dfc.b_end] > 0).sum().item()
        s_active = (features[0, dfc.b_end:] > 0).sum().item()

        print(f"   ToolRL-specific: {a_active}")
        print(f"   Base-specific: {b_active}")
        print(f"   Shared: {s_active}")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH

    # Run quick demo with mock data
    quick_demo(model_path)

    # Uncomment to run with real models (requires significant GPU memory)
    # extract_real_activations(model_path)