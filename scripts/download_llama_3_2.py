#!/usr/bin/env python3
"""
Download Llama 3.2 3B Instruct model from Hugging Face

This script downloads the Llama 3.2 3B Instruct model which is optimized for:
- Instruction following
- JSON generation
- Multilingual support
- Lower VRAM usage (~4-6GB vs ~10GB for 8B)
- Faster inference

Model: meta-llama/Llama-3.2-3B-Instruct
Size: ~6.5GB
"""

import os
import sys
from pathlib import Path

def download_llama_3_2():
    """Download Llama 3.2 3B Instruct model"""
    
    # Check if huggingface_hub is installed
    try:
        from huggingface_hub import snapshot_download
        print("✅ huggingface_hub is installed")
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("Installing huggingface_hub...")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import snapshot_download
    
    # Set download directory
    model_dir = Path(__file__).parent.parent / "models" / "llama-3.2-3b"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("📥 Downloading Llama 3.2 3B Instruct")
    print("="*70)
    print(f"Model: meta-llama/Llama-3.2-3B-Instruct")
    print(f"Destination: {model_dir}")
    print(f"Size: ~6.5GB")
    print("="*70 + "\n")
    
    # Check if already downloaded
    if (model_dir / "config.json").exists():
        print(f"⚠️  Model already exists at {model_dir}")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("✅ Using existing model")
            return True
    
    try:
        print("🚀 Starting download...")
        print("Note: This may take 10-20 minutes depending on your internet speed")
        print()
        
        # Download with progress
        snapshot_download(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("\n" + "="*70)
        print("✅ Download complete!")
        print("="*70)
        
        # Verify files
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        missing_files = [f for f in required_files if not (model_dir / f).exists()]
        
        if missing_files:
            print(f"⚠️  Warning: Missing files: {missing_files}")
            return False
        
        # Count safetensors files
        safetensor_files = list(model_dir.glob("*.safetensors"))
        print(f"\n📦 Model files:")
        print(f"   - Config files: ✅")
        print(f"   - Tokenizer files: ✅")
        print(f"   - Model weights: {len(safetensor_files)} safetensors files")
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        total_size_gb = total_size / (1024**3)
        print(f"   - Total size: {total_size_gb:.2f} GB")
        
        print("\n✅ Llama 3.2 3B Instruct is ready to use!")
        print(f"   Path: {model_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure you have enough disk space (~7GB)")
        print("3. You may need to accept Llama license on HuggingFace:")
        print("   https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
        return False

if __name__ == "__main__":
    success = download_llama_3_2()
    sys.exit(0 if success else 1)
