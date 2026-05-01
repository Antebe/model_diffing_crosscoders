"""
Upload DFC CrossCoder to HuggingFace Hub
=======================================

This script uploads the trained DFC CrossCoder model to HuggingFace.
"""

from huggingface_hub import HfApi, login, create_repo
import shutil
import os
from pathlib import Path

def upload_to_huggingface(
    repo_name: str = "dfc-crosscoder-qwen-ToolRL",
    username: str = None,
    private: bool = False
):
    """Upload the DFC CrossCoder to HuggingFace Hub"""
    
    print("🚀 Uploading DFC CrossCoder to HuggingFace")
    print("="*50)
    
    # Login to HuggingFace (requires token)
    print("Please login to HuggingFace (you'll need a token from https://huggingface.co/settings/tokens)")
    login()
    
    # Get username if not provided
    api = HfApi()
    if username is None:
        username = api.whoami()["name"]
        print(f"Using username: {username}")
    
    repo_id = f"{username}/{repo_name}"
    
    # Create repository
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"✅ Repository created/found: {repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return
    
    # Prepare files for upload
    upload_dir = Path("./hf_upload_temp")
    upload_dir.mkdir(exist_ok=True)
    
    # Copy model files
    checkpoints_dir = Path("./checkpoints/dfc2")
    if not checkpoints_dir.exists():
        print(f"❌ Checkpoints directory not found: {checkpoints_dir}")
        print("Please train the model first with: python run_train.py")
        return
    
    # Copy core files
    files_to_copy = {
        "checkpoints/dfc2/model.pt": "model.pt",
        "checkpoints/dfc2/config.json": "config.json",
        "dfc.py": "dfc.py",
        "demo.py": "demo.py",
        "requirements.txt": "requirements.txt",
        "model_README.md": "README.md"
    }
    
    print("\n📁 Preparing files for upload...")
    for src, dest in files_to_copy.items():
        src_path = Path(src)
        dest_path = upload_dir / dest
        
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            print(f"   ✅ {src} → {dest}")
        else:
            print(f"   ⚠️  Missing: {src}")
    
    # Upload to HuggingFace
    print(f"\n⬆️  Uploading to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            commit_message="Upload DFC CrossCoder model",
        )
        print(f"🎉 Successfully uploaded to: https://huggingface.co/{repo_id}")
        
        # Clean up
        shutil.rmtree(upload_dir)
        print("🧹 Cleaned up temporary files")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("Please check your internet connection and HuggingFace token")
    
    return repo_id

if __name__ == "__main__":
    # Configuration
    REPO_NAME = "dfc-crosscoder-qwen-ToolRL"
    USERNAME = None  # Will auto-detect from login
    PRIVATE = False   # Set to True for private repo
    
    # Upload
    repo_id = upload_to_huggingface(
        repo_name=REPO_NAME,
        username=USERNAME,
        private=PRIVATE
    )
    
    if repo_id:
        print(f"\n🔗 Your model is now available at:")
        print(f"   https://huggingface.co/{repo_id}")
        print(f"\n📖 To use in code:")
        print(f'   from huggingface_hub import hf_hub_download')
        print(f'   model_path = hf_hub_download(repo_id="{repo_id}", filename="model.pt")')
        print(f'   config_path = hf_hub_download(repo_id="{repo_id}", filename="config.json")')