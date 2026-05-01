#!/bin/bash
# setup_for_upload.sh - Prepare DFC CrossCoder for HuggingFace upload

echo "🔧 Setting up DFC CrossCoder for HuggingFace upload"
echo "================================================="

# Check if model is trained
if [ ! -f "./checkpoints/dfc2/model.pt" ]; then
    echo "❌ Model not found. Please train first:"
    echo "   python run_train.py"
    exit 1
fi

# Install required dependencies
echo "📦 Installing dependencies..."
pip install huggingface_hub>=0.16.0

# Check if all required files exist
echo "📋 Checking required files..."
required_files=(
    "dfc.py"
    "demo.py" 
    "quick_start.py"
    "requirements.txt"
    "model_README.md"
    "upload_to_hf.py"
    "checkpoints/dfc2/model.pt"
    "checkpoints/dfc2/config.json"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ Missing required files. Upload aborted."
    exit 1
fi

echo "✅ All files ready for upload!"
echo ""
echo "🚀 To upload to HuggingFace:"
echo "   python upload_to_hf.py"
echo ""
echo "📖 To test locally:"
echo "   python demo.py"
echo "   python quick_start.py"