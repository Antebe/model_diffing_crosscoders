# DFC CrossCoder - HuggingFace Upload Instructions

## 🎉 Your DFC CrossCoder is Ready!

I've prepared your DFC CrossCoder project for HuggingFace upload with the following structure:

### 📁 Files Created/Updated:

1. **Core Model Files:**
   - `requirements.txt` - Dependencies for the model
   - `model_README.md` - HuggingFace model card 
   - `dfc.py` - Your CrossCoder implementation (already existed)

2. **Demo & Usage:**
   - `demo.py` - Complete interactive demo
   - `quick_start.py` - **10-line minimal example** ⭐
   - `README.md` - Updated with HF instructions

3. **Upload Tools:**
   - `upload_to_hf.py` - Automated upload script
   - `setup_for_upload.sh` - Pre-upload validation
   - `.gitignore` - Avoid uploading unnecessary files

### 🚀 Upload to HuggingFace:

```bash
# 1. Install HuggingFace Hub
pip install huggingface_hub

# 2. Run the upload script
python upload_to_hf.py
```

The script will:
- Prompt you to login to HuggingFace (need a token from https://huggingface.co/settings/tokens)
- Create a repository named `dfc-crosscoder-qwen-ToolRL`
- Upload your model, config, demo, and documentation

### 🔥 10-Line Demo Usage:

```python
# From quick_start.py - copy and run this anywhere!
import torch
from dfc import DFCCrossCoder

dfc = DFCCrossCoder.load("./checkpoints/dfc2", device="cuda")                    
activations = torch.randn(1, 2, 2048).to("cuda")                                
features = dfc.encode(activations)                                              
active_count = (features > 0).sum().item()                                      
a_exclusive = (features[0, :dfc.a_end] > 0).sum().item()                       
b_exclusive = (features[0, dfc.a_end:dfc.b_end] > 0).sum().item()              
shared = (features[0, dfc.b_end:] > 0).sum().item()                            
top_vals, top_idx = torch.topk(features[0], k=5)                              
reconstruction, _ = dfc(activations)                                            

print(f"Active: {active_count}, A-excl: {a_exclusive}, B-excl: {b_exclusive}, Shared: {shared}")
```

### 🧠 Your Model Stats:
- **Dictionary Size**: 16,384 features  
- **Active Features**: 90 per sample (top-k)
- **Partitioning**: 819 A-exclusive + 819 B-exclusive + 14,746 shared
- **Models**: ToolRL vs Base Qwen2.5-3B comparison
- **Layer**: 13 (middle layer representations)

### 📊 What Users Get:

After upload, users can:
1. **Install**: `pip install -r requirements.txt` 
2. **Download**: Model auto-downloads from HuggingFace
3. **Use**: Run `demo.py` or copy the 10-liner from `quick_start.py`
4. **Extract Features**: Get sparse interpretable feature vectors from any text!

### 🔗 Next Steps:

1. Run `python upload_to_hf.py` to upload
2. Share your HuggingFace model link 
3. Users can immediately start extracting crosscoder features!

Your crosscoder is now ready to help researchers analyze the differences between tool-use and general text representations! 🎯