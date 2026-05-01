#!/usr/bin/env python3
"""
10-Line DFC CrossCoder Usage Example
===================================
Essential code to load and use the DFC CrossCoder for feature extraction.
"""

import torch
from dfc import DFCCrossCoder

# Load the trained CrossCoder (10 lines of core functionality)
dfc = DFCCrossCoder.load("./checkpoints/dfc2", device="cuda")                    # 1. Load model
activations = torch.randn(1, 2, 2048).to("cuda")                                # 2. Mock activations (batch=1, models=2, dim=2048)
features = dfc.encode(activations)                                              # 3. Extract sparse feature vector
active_count = (features > 0).sum().item()                                      # 4. Count active features
a_exclusive = (features[0, :dfc.a_end] > 0).sum().item()                       # 5. ToolRL-specific features
b_exclusive = (features[0, dfc.a_end:dfc.b_end] > 0).sum().item()              # 6. Base model-specific features  
shared = (features[0, dfc.b_end:] > 0).sum().item()                            # 7. Shared features
top_vals, top_idx = torch.topk(features[0], k=5)                              # 8. Get top 5 features
reconstruction, _ = dfc(activations)                                            # 9. Full reconstruction
print(f"Active: {active_count}, A-excl: {a_exclusive}, B-excl: {b_exclusive}, Shared: {shared}")  # 10. Results

# That's it! The CrossCoder is now extracting interpretable features from your text representations.