#!/bin/bash
# 🚀 Complete RunPod Setup Script
# This script will:
# 1. Install all necessary dependencies
# 2. Authenticate with HuggingFace
# 3. Download Llama 3.1 8B model (4-bit quantized)
# 4. Start vLLM server

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 RunPod Complete Setup - Llama 3.1 8B (4-bit)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Connect to RunPod and run setup
ssh root@194.68.245.173 -p 22186 -i ~/.ssh/id_ed25519 << 'ENDSSH'

echo "📦 Step 1: Installing Dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Update pip
pip install --upgrade pip --break-system-packages

# Install HuggingFace Hub and CLI
echo "Installing HuggingFace Hub..."
pip install -U "huggingface_hub[cli]" --break-system-packages

# Install vLLM with quantization support
echo "Installing vLLM..."
pip install vllm --break-system-packages

# Install bitsandbytes for 4-bit quantization
echo "Installing bitsandbytes for 4-bit quantization..."
pip install bitsandbytes --break-system-packages

# Install additional dependencies
echo "Installing additional dependencies..."
pip install transformers accelerate --break-system-packages

echo ""
echo "✅ Dependencies installed!"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 Step 2: HuggingFace Authentication"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "You need to authenticate with HuggingFace to download Llama 3.1 8B."
echo ""
echo "📋 To get your token:"
echo "   1. Go to: https://huggingface.co/settings/tokens"
echo "   2. Create a token with 'Read access to contents of all public gated repos'"
echo "   3. Copy the token (starts with 'hf_...')"
echo ""
echo "📋 To accept Llama 3.1 license:"
echo "   1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "   2. Click 'Request Access' and accept the license"
echo "   3. Wait for approval (usually instant)"
echo ""
echo "Now running HuggingFace login..."
echo ""

# Run HuggingFace login
huggingface-cli login

echo ""
echo "✅ Authentication complete!"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 Step 3: Downloading Llama 3.1 8B Model (4-bit)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will download ~4-5GB of model files..."
echo "Estimated time: 5-10 minutes depending on connection speed"
echo ""

# Create download script
cat > /workspace/download_model.py << 'ENDPYTHON'
from huggingface_hub import snapshot_download
import os

print("📥 Starting model download...")
print("Model: meta-llama/Meta-Llama-3.1-8B-Instruct")
print("Cache: /workspace/.cache/huggingface/hub")
print("")

try:
    model_path = snapshot_download(
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        cache_dir="/workspace/.cache/huggingface/hub",
        resume_download=True,
        local_files_only=False
    )
    print("")
    print("✅ Model downloaded successfully!")
    print(f"📁 Model path: {model_path}")
    print("")
    
    # List downloaded files
    print("📋 Downloaded files:")
    for root, dirs, files in os.walk(model_path):
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath) / (1024**3)  # GB
            print(f"   {file}: {size:.2f} GB")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print("")
    print("Troubleshooting:")
    print("1. Make sure you've accepted the Llama 3.1 license on HuggingFace")
    print("2. Verify your HuggingFace token has the correct permissions")
    print("3. Check your internet connection")
    exit(1)
ENDPYTHON

# Run download script
python /workspace/download_model.py

echo ""
echo "✅ Model download complete!"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Step 4: Verifying Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check GPU
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Check model cache
echo "📦 Model Cache:"
ls -lh /workspace/.cache/huggingface/hub/ | grep llama
echo ""

# Check installed packages
echo "📚 Installed Packages:"
pip list | grep -E "(vllm|transformers|bitsandbytes|huggingface)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Step 5: Starting vLLM Server (4-bit quantized)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Kill any existing vLLM processes
pkill -f vllm 2>/dev/null || true

# Start vLLM with 4-bit quantization
echo "Starting vLLM server..."
echo "Configuration:"
echo "  - Model: meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "  - Quantization: 4-bit (bitsandbytes)"
echo "  - Max Length: 2048 tokens"
echo "  - GPU Memory: 85%"
echo "  - Port: 8000"
echo ""

cd /workspace

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  > /workspace/vllm.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"
echo ""

echo "⏳ Waiting for vLLM to initialize (this takes 2-3 minutes)..."
echo "   Tailing log file..."
echo ""

# Wait and show log
sleep 10
tail -f /workspace/vllm.log &
TAIL_PID=$!

# Wait for vLLM to be ready (check for "Uvicorn running")
for i in {1..60}; do
    if grep -q "Uvicorn running" /workspace/vllm.log 2>/dev/null; then
        echo ""
        echo "✅ vLLM server is ready!"
        kill $TAIL_PID 2>/dev/null || true
        break
    fi
    sleep 5
    if [ $i -eq 60 ]; then
        echo ""
        echo "⚠️  vLLM is taking longer than expected. Check the log:"
        echo "   tail -f /workspace/vllm.log"
        kill $TAIL_PID 2>/dev/null || true
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. Test the vLLM endpoint (from RunPod):"
echo "   curl http://localhost:8000/v1/models"
echo ""
echo "2. Test chat completion (from RunPod):"
echo "   curl http://localhost:8000/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""
echo "3. On your Mac, start the SSH tunnel:"
echo "   ./start_runpod_tunnel.sh"
echo ""
echo "4. Test from your Mac:"
echo "   curl http://localhost:8000/v1/models"
echo ""
echo "5. Start the backend:"
echo "   cd backend && python main_pure_llm.py"
echo ""
echo "6. Run multi-language tests:"
echo "   python test_multilanguage.py"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Useful Commands:"
echo "   - Check vLLM log: tail -f /workspace/vllm.log"
echo "   - Check vLLM process: ps aux | grep vllm"
echo "   - Restart vLLM: pkill -f vllm && ./start_vllm.sh"
echo "   - Check GPU: nvidia-smi"
echo ""

ENDSSH

echo ""
echo "✅ Setup script completed!"
echo ""
echo "The vLLM server should now be running on RunPod."
echo "Check the output above for any errors."
