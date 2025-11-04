#!/usr/bin/env python3
"""
Download LLaMA 3.1 8B for macOS Metal (without quantization)
Optimized for M2 Pro - Full model will use ~16GB disk, ~8GB RAM during inference
"""

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch

def download_llama_for_macos():
    """Download LLaMA 3.1 8B optimized for macOS Metal"""
    
    print("\n" + "="*80)
    print("🦙 LLaMA 3.1 8B Download for macOS Metal")
    print("="*80)
    print("\n📊 Model Info:")
    print("   • Model: LLaMA 3.1 8B Instruct")
    print("   • Size: ~16GB (full precision)")
    print("   • Device: Metal M2 Pro (MPS)")
    print("   • RAM Usage: ~8-10GB during inference")
    print("   • Speed: 5-8 seconds per response")
    print("\n" + "="*80)
    
    # Get HuggingFace token
    token = input("\n🔑 Enter your HuggingFace token: ").strip()
    
    if not token:
        print("❌ Token required!")
        return 1
    
    try:
        # Login
        print("\n🔐 Logging into HuggingFace...")
        login(token=token)
        print("✅ Authentication successful")
        
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        save_path = "./models/llama-3.1-8b"
        
        print(f"\n📥 Downloading {model_name}")
        print("   This will take 10-30 minutes depending on internet speed...")
        print(f"   Saving to: {save_path}")
        print("\n" + "="*80)
        
        # Create directory
        os.makedirs(save_path, exist_ok=True)
        
        # Download tokenizer
        print("\n1️⃣ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("   ✓ Tokenizer downloaded")
        
        # Download model (full precision, no quantization)
        print("\n2️⃣ Downloading model...")
        print("   📦 Size: ~16GB")
        print("   ⏱️  This may take a while...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 to save memory
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Save model
        print("\n3️⃣ Saving model...")
        model.save_pretrained(save_path)
        print("   ✓ Model saved")
        
        print("\n" + "="*80)
        print("✅ LLaMA 3.1 8B downloaded successfully!")
        print("="*80)
        
        print(f"\n📁 Model location: {save_path}")
        print("\n🚀 Next steps:")
        print("   1. Test the model:")
        print("      python3 scripts/test_llm_metal.py")
        print("\n   2. Start ML service:")
        print("      python3 ml_api_service.py")
        print("\n   3. Start backend:")
        print("      cd backend && python3 main.py")
        
        print("\n💡 The model will automatically use Metal (MPS) acceleration!")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you have HuggingFace access to LLaMA 3.1 8B")
        print("   2. Check your internet connection")
        print("   3. Ensure you have ~20GB free disk space")
        return 1


if __name__ == "__main__":
    sys.exit(download_llama_for_macos())
