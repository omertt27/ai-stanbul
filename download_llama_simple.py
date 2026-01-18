#!/usr/bin/env python3
"""
Simple Llama 3.1 8B downloader for RunPod (30GB)
Avoids metadata issues by using cache_dir instead of local_dir
"""

from huggingface_hub import HfFolder, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

print("🚀 Llama 3.1 8B Download (Simple Method)")
print("=" * 50)

# Get token
token = HfFolder.get_token()
if not token:
    print("❌ No HF token found. Run: huggingface-cli login")
    sys.exit(1)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR = "/workspace/models"

print(f"📥 Downloading {MODEL_ID}...")
print(f"📁 Cache directory: {CACHE_DIR}")
print("⏳ This takes 5-10 minutes...")
print("")

try:
    # Method 1: Use transformers to download (handles everything)
    print("Step 1: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        token=token
    )
    print("✅ Tokenizer downloaded!")
    
    print("")
    print("Step 2: Downloading model (~16GB)...")
    print("This is the large download, please wait...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        token=token,
        torch_dtype="auto",
        device_map=None,  # Don't load to GPU yet
        low_cpu_mem_usage=True
    )
    print("✅ Model downloaded!")
    
    print("")
    print("=" * 50)
    print("✅ Download Complete!")
    print("=" * 50)
    print(f"📁 Model cached in: {CACHE_DIR}")
    print("")
    print("Model location:")
    print(f"  {MODEL_ID}")
    print("")
    
except Exception as e:
    print(f"❌ Download failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
