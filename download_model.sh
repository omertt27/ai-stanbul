#!/bin/bash
# Download Llama 3.1 8B Model
# Run this after: pip3 install huggingface-hub transformers torch bitsandbytes accelerate

echo "=========================================="
echo "📥 Downloading Llama 3.1 8B Model"
echo "=========================================="
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli not found. Installing..."
    pip3 install huggingface-hub
fi

# Check if logged in
echo "🔐 Checking Hugging Face authentication..."
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "✅ Already logged in as: $(huggingface-cli whoami)"
else
    echo "❌ Not logged in to Hugging Face"
    echo ""
    echo "Please login first:"
    echo "1. Get your token from: https://huggingface.co/settings/tokens"
    echo "2. Request access to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo ""
    read -p "Enter your Hugging Face token: " HF_TOKEN
    echo "$HF_TOKEN" | huggingface-cli login --token
    
    if huggingface-cli whoami > /dev/null 2>&1; then
        echo "✅ Login successful!"
    else
        echo "❌ Login failed. Exiting."
        exit 1
    fi
fi

echo ""
echo "📥 Starting model download..."
echo "⏳ This will take 5-10 minutes"
echo ""

cd /workspace/models

# Option 1: Download with Python (includes 4-bit quantization)
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print("🔧 Configuring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("📥 Downloading Llama 3.1 8B model...")
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/workspace/models"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/workspace/models")
    
    print("\n✅ Model downloaded successfully!")
    print(f"📊 Model size in memory: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    # Quick test
    print("\n🧪 Testing model...")
    inputs = tokenizer("The capital of Turkey is", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=10)
    test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Test: {test_output}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import sys
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Download Complete!"
    echo "=========================================="
    echo ""
    echo "Model is cached at: /workspace/models"
    echo "You can now create and start the server!"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Download Failed"
    echo "=========================================="
    echo ""
    echo "Common issues:"
    echo "  1. No Hugging Face access - Request at:"
    echo "     https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "  2. Not enough GPU memory (need 8GB+ VRAM)"
    echo "  3. Network issues - Try again"
    echo ""
    exit 1
fi
