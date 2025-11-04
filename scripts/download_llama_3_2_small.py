"""
Download LLaMA 3.2 1B or 3B models - optimized for Apple Silicon M2 Pro
These models are small enough to run on Metal with 20GB unified memory
"""

import os
import torch
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import getpass

def download_and_test_model(model_id, save_path):
    """Download and test a LLaMA model"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"Save path: {save_path}")
    print(f"{'='*60}\n")
    
    try:
        # Download model
        print("Downloading model files...")
        snapshot_download(
            repo_id=model_id,
            local_dir=save_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Model downloaded to {save_path}")
        
        # Test loading
        print("\nTesting model loading on Metal...")
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        model = AutoModelForCausalLM.from_pretrained(
            save_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Quick test
        test_prompt = "Istanbul is known for"
        inputs = tokenizer(test_prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest inference:")
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response}")
        print(f"\n✅ Model works on Metal!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

def main():
    print("="*60)
    print("LLaMA 3.2 Small Model Downloader for Apple Silicon")
    print("="*60)
    
    # Check Metal availability
    if not torch.backends.mps.is_available():
        print("⚠️  Metal (MPS) not available. Models will run on CPU.")
    else:
        print("✅ Metal (MPS) is available")
    
    # Login to Hugging Face
    print("\nPlease enter your Hugging Face token:")
    print("(Get it from: https://huggingface.co/settings/tokens)")
    token = getpass.getpass("Token: ")
    
    try:
        login(token=token)
        print("✅ Logged in to Hugging Face")
    except Exception as e:
        print(f"❌ Login failed: {str(e)}")
        return
    
    # Model options
    models = {
        "1": {
            "id": "meta-llama/Llama-3.2-1B",
            "name": "LLaMA 3.2 1B",
            "size": "~2GB",
            "path": "./models/llama-3.2-1b"
        },
        "2": {
            "id": "meta-llama/Llama-3.2-3B",
            "name": "LLaMA 3.2 3B",
            "size": "~6GB",
            "path": "./models/llama-3.2-3b"
        }
    }
    
    print("\n" + "="*60)
    print("Available models:")
    print("="*60)
    for key, model in models.items():
        print(f"{key}. {model['name']} ({model['size']})")
    
    choice = input("\nSelect model (1 or 2, or 'both'): ").strip().lower()
    
    if choice == "both":
        # Download both models
        for key in ["1", "2"]:
            model = models[key]
            print(f"\n{'='*60}")
            print(f"Downloading {model['name']}")
            print(f"{'='*60}")
            success = download_and_test_model(model["id"], model["path"])
            if not success:
                print(f"⚠️  Failed to download {model['name']}, continuing...")
    elif choice in models:
        model = models[choice]
        download_and_test_model(model["id"], model["path"])
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*60)
    print("✅ Download complete!")
    print("="*60)
    print("\nRecommendations:")
    print("• LLaMA 3.2 1B: Fastest, fits easily in 20GB Metal memory")
    print("• LLaMA 3.2 3B: Better quality, still fits in 20GB Metal memory")
    print("\nNext steps:")
    print("1. Test the models with: python scripts/test_llm_with_fallback.py")
    print("2. Update ml_api_service.py to use the new model")

if __name__ == "__main__":
    main()
