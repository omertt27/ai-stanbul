"""
Download and cache Llama 3.1 8B model in 4-bit quantization
This script should be run on your RunPod instance ONCE to download the model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

def download_llama_model():
    """
    Download Llama 3.1 8B model with 4-bit quantization
    """
    print("🦙 Downloading Llama 3.1 8B model with 4-bit quantization...")
    print("=" * 60)
    
    # Model ID on Hugging Face
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Cache directory
    cache_dir = "/workspace/models"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"📁 Cache directory: {cache_dir}")
    print(f"🔑 Model: {model_id}")
    print()
    
    # Check if HF token is set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️  WARNING: HF_TOKEN environment variable not set!")
        print("   You need a Hugging Face token to download Llama models.")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        print("   Then set it: export HF_TOKEN='your_token_here'")
        print()
        print("   Or you can use a public model instead.")
        return False
    
    # Configure 4-bit quantization
    print("⚙️  Configuring 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    print("✅ Quantization config:")
    print(f"   - 4-bit: {quantization_config.load_in_4bit}")
    print(f"   - Compute dtype: {quantization_config.bnb_4bit_compute_dtype}")
    print(f"   - Quant type: {quantization_config.bnb_4bit_quant_type}")
    print(f"   - Double quant: {quantization_config.bnb_4bit_use_double_quant}")
    print()
    
    # Download tokenizer
    print("📥 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=hf_token
    )
    print("✅ Tokenizer downloaded!")
    print()
    
    # Download model with 4-bit quantization
    print("📥 Downloading model (this may take a while)...")
    print("   Note: 4-bit model will be ~4.5GB instead of ~16GB")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=cache_dir,
        token=hf_token,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    print("✅ Model downloaded and quantized!")
    print()
    
    # Test the model
    print("🧪 Testing model...")
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test input: {test_prompt}")
    print(f"Test output: {response}")
    print()
    
    # Print model info
    print("📊 Model Information:")
    print(f"   - Model size: ~4.5GB (4-bit quantized)")
    print(f"   - Device: {model.device}")
    print(f"   - Dtype: {model.dtype}")
    print(f"   - Parameters: 8B (quantized to 4-bit)")
    print()
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 GPU Memory:")
        print(f"   - Allocated: {memory_allocated:.2f} GB")
        print(f"   - Reserved: {memory_reserved:.2f} GB")
        print()
    
    print("=" * 60)
    print("🎉 Download complete!")
    print()
    print("Model cached at:", cache_dir)
    print()
    print("Next step: Start the server with:")
    print("  python3 runpod_server.py")
    print()
    
    return True


if __name__ == "__main__":
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA is not available!")
        print("   This script requires a GPU with CUDA support.")
        exit(1)
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Download model
    success = download_llama_model()
    
    if success:
        exit(0)
    else:
        exit(1)
