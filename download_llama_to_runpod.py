#!/usr/bin/env python3
"""
Download Llama 3.1 8B 4-bit quantized model from Hugging Face to RunPod

This script downloads the 4-bit quantized version of Meta's Llama 3.1 8B model,
which is optimized for inference with reduced memory requirements.

Requirements:
- huggingface_hub library
- Hugging Face token (for gated models)
- Sufficient disk space (~4-5GB for 4-bit quantized model)

Usage:
    python download_llama_to_runpod.py

Author: AI Istanbul Team
Date: December 2024
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import huggingface_hub
        logger.info("✅ huggingface_hub is installed")
        return True
    except ImportError:
        logger.error("❌ huggingface_hub is not installed")
        logger.info("Install with: pip install huggingface_hub")
        return False


def get_hf_token():
    """Get Hugging Face token from environment or user input."""
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    if not token:
        logger.warning("⚠️  No HF token found in environment variables")
        logger.info("You can set it with: export HUGGINGFACE_TOKEN='your_token_here'")
        token = input("Enter your Hugging Face token (or press Enter to skip): ").strip()
    
    return token if token else None


def download_llama_model(model_id: str, output_dir: str, token: str = None):
    """
    Download Llama model from Hugging Face.
    
    Args:
        model_id: Hugging Face model ID (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
        output_dir: Local directory to save the model
        token: Hugging Face API token (required for gated models)
    """
    try:
        # Login to Hugging Face if token provided
        if token:
            logger.info("🔐 Logging in to Hugging Face...")
            login(token=token)
            logger.info("✅ Successfully authenticated with Hugging Face")
        else:
            logger.warning("⚠️  No token provided - proceeding without authentication")
            logger.warning("   This may fail for gated models like Llama")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_path.absolute()}")
        
        # Download the model
        logger.info(f"📥 Downloading model: {model_id}")
        logger.info("   This may take several minutes depending on your connection...")
        
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token
        )
        
        logger.info(f"✅ Model downloaded successfully to: {downloaded_path}")
        
        # Show directory size
        total_size = sum(f.stat().st_size for f in Path(downloaded_path).rglob('*') if f.is_file())
        size_gb = total_size / (1024 ** 3)
        logger.info(f"📊 Total model size: {size_gb:.2f} GB")
        
        return downloaded_path
        
    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        
        if "401" in str(e) or "403" in str(e):
            logger.error("   This looks like an authentication error.")
            logger.error("   Make sure you:")
            logger.error("   1. Have a Hugging Face account")
            logger.error("   2. Have accepted the Llama license at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
            logger.error("   3. Have created a valid access token at: https://huggingface.co/settings/tokens")
        
        raise


def main():
    """Main function to download Llama 3.1 8B model."""
    logger.info("=" * 80)
    logger.info("🦙 Llama 3.1 8B Model Downloader for RunPod")
    logger.info("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Please install required dependencies first:")
        logger.error("pip install huggingface_hub")
        sys.exit(1)
    
    # Popular 4-bit quantized versions of Llama 3.1 8B
    models = {
        "1": {
            "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "name": "Official Meta Llama 3.1 8B Instruct (Full precision - ~16GB)",
            "size": "~16GB"
        },
        "2": {
            "id": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "name": "Unsloth 4-bit quantized (Recommended - ~4.5GB)",
            "size": "~4.5GB"
        },
        "3": {
            "id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "name": "GGUF format 4-bit (for llama.cpp - ~4.5GB)",
            "size": "~4.5GB"
        },
        "4": {
            "id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
            "name": "NousResearch version (Full precision - ~16GB)",
            "size": "~16GB"
        }
    }
    
    # Display options
    logger.info("\n📋 Available Llama 3.1 8B models:")
    for key, model in models.items():
        logger.info(f"   {key}. {model['name']}")
        logger.info(f"      Model ID: {model['id']}")
        logger.info(f"      Size: {model['size']}\n")
    
    # Get user choice
    choice = input("Select model (1-4) [default: 2 for 4-bit]: ").strip() or "2"
    
    if choice not in models:
        logger.error(f"Invalid choice: {choice}")
        sys.exit(1)
    
    selected_model = models[choice]
    model_id = selected_model["id"]
    
    logger.info(f"\n✅ Selected: {selected_model['name']}")
    logger.info(f"   Model ID: {model_id}")
    logger.info(f"   Expected size: {selected_model['size']}")
    
    # Get output directory
    default_dir = "/workspace/models/llama-3.1-8b"  # RunPod standard path
    output_dir = input(f"\nEnter output directory [default: {default_dir}]: ").strip() or default_dir
    
    # Get HF token
    token = get_hf_token()
    
    # Confirm download
    logger.info("\n" + "=" * 80)
    logger.info("📋 Download Summary:")
    logger.info(f"   Model: {model_id}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Size: {selected_model['size']}")
    logger.info(f"   Authentication: {'✅ Token provided' if token else '❌ No token'}")
    logger.info("=" * 80)
    
    confirm = input("\nProceed with download? (y/n) [default: y]: ").strip().lower() or "y"
    
    if confirm != 'y':
        logger.info("❌ Download cancelled")
        sys.exit(0)
    
    # Download the model
    try:
        downloaded_path = download_llama_model(model_id, output_dir, token)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Download completed successfully!")
        logger.info("=" * 80)
        logger.info(f"📁 Model location: {downloaded_path}")
        logger.info("\n💡 Next steps:")
        logger.info("   1. Load the model in your inference code")
        logger.info("   2. For transformers: model = AutoModelForCausalLM.from_pretrained('" + downloaded_path + "')")
        logger.info("   3. For GGUF: Use with llama.cpp or similar tools")
        logger.info("\n📚 Example usage code:")
        logger.info("""
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{}",
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{}")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
        """.format(downloaded_path, downloaded_path))
        
    except Exception as e:
        logger.error(f"\n❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
