#!/usr/bin/env python3
"""
Download LLAMA models for KAM system
Supports LLaMA 3 8B with Q4 quantization - optimized for Metal M2 Pro and T4 GPU
"""

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import torch

def download_llama_model(model_name, save_path, hf_token=None, use_quantization=False):
    """Download a LLAMA model from HuggingFace with optional quantization"""
    print(f"\n{'='*80}")
    print(f"📥 Downloading {model_name}")
    if use_quantization:
        print("   Using 4-bit quantization (Q4) for memory efficiency")
    print(f"{'='*80}")
    
    try:
        # Login if token provided
        if hf_token:
            print("🔐 Logging into HuggingFace...")
            login(token=hf_token)
            print("✅ Authentication successful")
        
        # Download tokenizer
        print("\n1️⃣ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure quantization if requested
        quantization_config = None
        if use_quantization:
            print("\n2️⃣ Configuring 4-bit quantization (Q4)...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("   ✓ Q4 quantization configured (reduces 16GB → ~4-5GB)")
        
        # Download model
        print(f"\n{'2️⃣' if not use_quantization else '3️⃣'} Downloading model (this may take a while)...")
        print("   Size: ~16GB full model, ~4-5GB with Q4 quantization")
        
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "device_map": "auto" if use_quantization else None
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Save to disk
        print(f"\n{'3️⃣' if not use_quantization else '4️⃣'} Saving to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"\n✅ {model_name} downloaded successfully!")
        print(f"   Location: {save_path}")
        if use_quantization:
            print(f"   Quantization: 4-bit (Q4)")
            print(f"   Memory: ~4-5GB (vs 16GB full precision)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {model_name}: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Make sure you have HuggingFace token with LLAMA access")
        print(f"  2. Install required packages: pip install bitsandbytes accelerate")
        print(f"  3. For quantization, ensure compatible GPU or use CPU fallback")
        return False


def main():
    """Main download process"""
    print("\n" + "="*80)
    print("🦙 LLAMA MODEL DOWNLOADER FOR KAM")
    print("="*80)
    print("\n🎯 RECOMMENDED: LLaMA 3 8B with Q4 Quantization")
    print("   • Original size: 16GB → Quantized: 4-5GB")
    print("   • Quality: Excellent (minimal loss from quantization)")
    print("   • Speed: Fast on Metal M2 Pro (5-8s) and T4 GPU (3-5s)")
    print("   • Memory: Fits in 8GB VRAM/RAM")
    
    print("\n" + "="*80)
    print("📦 AVAILABLE MODELS")
    print("="*80)
    
    print("\n1. ⭐ LLaMA 3.1 8B (Q4 Quantized) — RECOMMENDED")
    print("   • Size: ~4-5GB (quantized from 16GB)")
    print("   • Best for: Production use on Metal M2 Pro AND T4 GPU")
    print("   • Quality: Excellent for Istanbul tourism guidance")
    print("   • Speed: 3-8s per response")
    print("   • Memory: 8GB RAM/VRAM sufficient")
    
    print("\n2. LLaMA 3.2 3B-Instruct")
    print("   • Size: ~6GB")
    print("   • Best for: Fast development testing")
    print("   • Quality: Very good")
    print("   • Speed: 3-6s per response")
    
    print("\n3. LLaMA 3.2 1B-Instruct")
    print("   • Size: ~2.5GB")
    print("   • Best for: Rapid prototyping")
    print("   • Quality: Good")
    print("   • Speed: 2-4s per response")
    
    print("\n4. TinyLlama 1.1B (Fallback)")
    print("   • Size: ~2GB")
    print("   • Best for: Testing infrastructure")
    print("   • Quality: Basic")
    print("   • Speed: 2-3s per response")
    
    print("\n" + "="*80)
    
    # Get HuggingFace token
    print("\n🔑 HuggingFace Access Token Required")
    print("   Get your token from: https://huggingface.co/settings/tokens")
    print("   You need access to Meta LLAMA models")
    print("\n   To request access:")
    print("   1. Go to: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("   2. Click 'Request Access' and agree to terms")
    print("   3. Wait for approval (usually instant)")
    
    hf_token = input("\nEnter your HuggingFace token (or press Enter to skip): ").strip()
    
    if not hf_token:
        print("\n⚠️  No token provided.")
        print("   → Will use TinyLlama fallback model (no token required)")
        use_fallback = input("\n   Download TinyLlama instead? (y/n): ").strip().lower()
        if use_fallback == 'y':
            print("\n📥 Downloading TinyLlama (fallback model)...")
            if download_llama_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "./models/tinyllama", None, False):
                print("\n✅ TinyLlama downloaded successfully!")
                print("\n⚠️  Note: For production, use LLaMA 3 8B Q4 for better quality")
                return 0
        return 1
    
    # Ask which model to download
    print("\n" + "="*80)
    print("💡 RECOMMENDATION: Option 1 (LLaMA 3.1 8B Q4) is best for production")
    choice = input("\nWhich model to download? (1/2/3/4 or 'all'): ").strip()
    
    models = {
        "1": ("meta-llama/Llama-3.1-8B-Instruct", "./models/llama-3.1-8b-q4", True),   # Q4 quantized
        "2": ("meta-llama/Llama-3.2-3B-Instruct", "./models/llama-3.2-3b", False),
        "3": ("meta-llama/Llama-3.2-1B-Instruct", "./models/llama-3.2-1b", False),
        "4": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "./models/tinyllama", False)
    }
    
    downloads = []
    if choice.lower() == "all":
        downloads = list(models.values())
    elif choice in models:
        downloads = [models[choice]]
    else:
        print("❌ Invalid choice")
        return 1
    
    # Install required packages for quantization
    if any(use_quant for _, _, use_quant in downloads):
        print("\n" + "="*80)
        print("📦 Installing quantization dependencies...")
        print("="*80)
        os.system("pip install -q bitsandbytes accelerate")
        print("✅ Dependencies installed")
    
    # Download selected models
    print("\n" + "="*80)
    print(f"📦 Starting download of {len(downloads)} model(s)...")
    print("="*80)
    
    success_count = 0
    for model_name, save_path, use_quant in downloads:
        if download_llama_model(model_name, save_path, hf_token, use_quant):
            success_count += 1
    
    # Summary
    print("\n" + "="*80)
    print("📊 DOWNLOAD SUMMARY")
    print("="*80)
    print(f"✅ Successfully downloaded: {success_count}/{len(downloads)} models")
    
    if success_count > 0:
        print("\n📁 Downloaded models:")
        for _, save_path, use_quant in downloads:
            if os.path.exists(save_path):
                quant_label = " (Q4 Quantized)" if use_quant else ""
                print(f"   • {save_path}{quant_label}")
        
        print("\n🚀 Next steps:")
        print("   1. Model will auto-detect best available LLAMA")
        print("   2. Restart ml_api_service.py")
        print("   3. Test with: python3 scripts/test_llm_metal.py")
        print("\n💡 For production:")
        print("   • Metal M2 Pro: LLaMA 3.1 8B Q4 will use ~6GB RAM")
        print("   • T4 GPU: LLaMA 3.1 8B Q4 will use ~5GB VRAM")
    
    print("\n" + "="*80)
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
