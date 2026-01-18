#!/bin/bash
# RunPod Llama 3.1 8B Setup - Step by Step
# Copy and paste each section into your RunPod terminal

echo "=========================================="
echo "🚀 RunPod Llama 3.1 8B Setup Script"
echo "=========================================="
echo ""

# STEP 1: System Updates
echo "📦 Step 1/6: Updating system packages..."
apt-get update -qq
apt-get install -y python3-pip git wget curl screen htop tmux -qq
echo "✅ System packages updated"
echo ""

# STEP 2: Install Hugging Face CLI
echo "🤗 Step 2/6: Installing Hugging Face CLI..."
pip3 install --upgrade pip -q
pip3 install huggingface-hub -q
echo "✅ Hugging Face CLI installed"
echo ""

# STEP 3: Authenticate with Hugging Face
echo "🔐 Step 3/6: Hugging Face Authentication"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "You need a Hugging Face token to download Llama models."
echo ""
echo "📝 Instructions:"
echo "1. Go to: https://huggingface.co/settings/tokens"
echo "2. Create a new token (read access is enough)"
echo "3. Request access to Llama 3.1 at:"
echo "   https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "4. Wait for approval (usually instant)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Prompt for token
read -p "Enter your Hugging Face token: " HF_TOKEN
echo ""

# Login with token
echo "$HF_TOKEN" | huggingface-cli login --token

# Verify login
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "✅ Successfully logged in to Hugging Face!"
    echo "   User: $(huggingface-cli whoami)"
else
    echo "❌ Login failed. Please check your token and try again."
    exit 1
fi
echo ""

# STEP 4: Install Python Dependencies
echo "🐍 Step 4/6: Installing Python packages (this may take 2-3 minutes)..."
pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -q transformers accelerate bitsandbytes scipy sentencepiece
pip3 install -q fastapi uvicorn python-multipart pydantic requests aiohttp python-dotenv
echo "✅ Python packages installed"
echo ""

# STEP 5: Create Directories
echo "📁 Step 5/6: Creating workspace directories..."
mkdir -p /workspace/models
mkdir -p /workspace/llm_server
mkdir -p /workspace/logs
echo "✅ Directories created"
echo ""

# STEP 6: Download Model
echo "📥 Step 6/6: Downloading Llama 3.1 8B model..."
echo "⏳ This will take 5-10 minutes depending on your connection"
echo ""

cd /workspace/models

python3 << 'PYTHON_EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print("🔧 Configuring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("📥 Downloading model from Hugging Face...")
print("   This may take several minutes...")
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
    
    print("✅ Model downloaded successfully!")
    print(f"📊 Model size in memory: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    # Test generation
    print("\n🧪 Testing model with quick generation...")
    inputs = tokenizer("Hello, I am", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=20)
    test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Test output: {test_output}")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print("\nPossible issues:")
    print("  1. Token doesn't have access to Llama 3.1")
    print("  2. Not enough GPU memory")
    print("  3. Network connection issues")
    exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model download complete!"
else
    echo ""
    echo "❌ Model download failed. Please check the error above."
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📊 Summary:"
echo "  • System packages: ✅ Installed"
echo "  • Hugging Face CLI: ✅ Installed"
echo "  • HF Authentication: ✅ Logged in"
echo "  • Python packages: ✅ Installed"
echo "  • Model download: ✅ Complete"
echo ""
echo "🎯 Next Steps:"
echo "  1. Run: ./create_server.sh"
echo "  2. Or manually create the server (see guide)"
echo ""
