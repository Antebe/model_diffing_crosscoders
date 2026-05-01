# DFC CrossCoder: ToolRL vs Base Qwen2.5-3B

A Dedicated Feature CrossCoder (DFC) trained to compare activations between:
- **Model A**: ToolRL fine-tuned Qwen2.5-3B (`chengq9/ToolRL-Qwen2.5-3B`)
- **Model B**: Base Qwen2.5-3B (`Qwen/Qwen2.5-3B`)

## Model Overview

The DFC CrossCoder learns sparse feature representations with three partitions:
- **A-exclusive features**: Only accessible to ToolRL model (captures tool-use specific patterns)
- **B-exclusive features**: Only accessible to Base model (captures general text patterns)
- **Shared features**: Accessible to both models (captures common representations)

## Model Architecture

- **Dictionary Size**: 16,384 features
- **Top-K Active**: 90 features per sample
- **Layer**: 13 (middle layer of Qwen2.5-3B)
- **Activation Dim**: 2048
- **Partitioning**: 5% A-exclusive, 5% B-exclusive, 90% shared

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from dfc import DFCCrossCoder

# Download and load the model
model_path = hf_hub_download(repo_id="your-username/dfc-crosscoder-qwen-ToolRL", 
                            filename="model.pt")
config_path = hf_hub_download(repo_id="your-username/dfc-crosscoder-qwen-ToolRL", 
                             filename="config.json")

# Load the crosscoder
dfc = DFCCrossCoder.load("./", device="cuda")

# Example: Extract features from activations
# activations should be shape (batch_size, 2, 2048) where dim 1 is [model_a, model_b]
activations = torch.randn(1, 2, 2048)  # Replace with real activations
features = dfc.encode(activations)  # Returns sparse feature vector

print(f"Active features: {(features > 0).sum().item()}/{features.shape[-1]}")
```

## Quick Demo

See `demo.py` for a complete example that shows how to:
1. Load both models (ToolRL and base)
2. Extract activations from sample text
3. Run the CrossCoder to get feature vectors
4. Analyze partition-specific activations

## Training Data

- **FineWeb**: 15,000 general text samples
- **ToolRL Conversations**: 15,000 tool-use conversation samples
- **Layer**: 13 (middle layer representations)

## Files

- `model.pt`: PyTorch model weights
- `config.json`: Model configuration
- `dfc.py`: CrossCoder implementation
- `demo.py`: Usage example

## Citation

If you use this model, please cite:

```bibtex
@misc{dfc-crosscoder,
  title={DFC CrossCoder: Analyzing Tool-Use vs General Text Features},
  author={[Your Name]},
  year={2026},
  url={https://huggingface.co/your-username/dfc-crosscoder-qwen-ToolRL}
}
```