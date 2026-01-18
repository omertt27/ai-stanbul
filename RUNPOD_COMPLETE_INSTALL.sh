#!/bin/bash

# 🚀 COMPLETE RUNPOD SETUP FROM SCRATCH
# Copy and paste this entire script into RunPod terminal

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 RunPod Complete Installation"
echo "   Installing: HuggingFace + vLLM + Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Update system
echo "📦 Step 1/6: Updating system..."
apt-get update -qq
apt-get install -y git curl wget nano htop

# Step 2: Upgrade pip
echo ""
echo "📦 Step 2/6: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Step 3: Install HuggingFace libraries
echo ""
echo "📦 Step 3/6: Installing HuggingFace libraries..."
pip install --upgrade \
  huggingface-hub \
  transformers \
  accelerate \
  tokenizers

# Step 4: Install vLLM
echo ""
echo "📦 Step 4/6: Installing vLLM (this may take a few minutes)..."
pip install vllm

# Step 5: Login to HuggingFace (optional but recommended)
echo ""
echo "📦 Step 5/6: HuggingFace setup..."
echo ""
echo "⚠️  IMPORTANT: Llama models require HuggingFace token!"
echo ""
echo "Do you have a HuggingFace token? (y/n)"
echo "If you don't have one:"
echo "  1. Go to https://huggingface.co/settings/tokens"
echo "  2. Create a token with 'read' access"
echo "  3. Accept Llama 3.1 license at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
echo ""
read -p "Enter your token (or press Enter to skip): " HF_TOKEN

if [ ! -z "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
    echo "✅ HuggingFace login successful!"
else
    echo "⚠️  Skipping HuggingFace login - model download may fail!"
    echo "If vLLM fails to start, run: huggingface-cli login"
fi

# Step 6: Download model (optional pre-download)
echo ""
echo "📦 Step 6/6: Pre-downloading model (optional, can skip)..."
echo ""
echo "Download model now? This will take ~16GB and 5-10 minutes."
echo "If you skip, vLLM will download it on first start."
read -p "Download now? (y/n): " DOWNLOAD_NOW

if [ "$DOWNLOAD_NOW" = "y" ] || [ "$DOWNLOAD_NOW" = "Y" ]; then
    echo "Downloading Meta-Llama-3.1-8B-Instruct..."
    python3 << 'PYCODE'
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        local_dir="/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct",
        local_dir_use_symlinks=False
    )
    print("✅ Model downloaded successfully!")
except Exception as e:
    print(f"⚠️  Download failed: {e}")
    print("Model will be downloaded when vLLM starts.")
PYCODE
else
    echo "Skipping pre-download. Model will download on first start."
fi

# Step 7: Check GPU
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎮 GPU Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi

# Step 8: Clean up old processes
echo ""
echo "🧹 Cleaning up old processes..."
pkill -9 -f vllm 2>/dev/null || true
sleep 2

# Step 9: Start vLLM
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting vLLM Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  • Model: Meta-Llama-3.1-8B-Instruct"
echo "  • Port: 8000"
echo "  • Host: 0.0.0.0 (accessible via SSH tunnel)"
echo "  • Max tokens: 2048"
echo "  • GPU memory: 50%"
echo ""

nohup python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.5 \
  --trust-remote-code \
  > /root/vllm.log 2>&1 &

VLLM_PID=$!
echo $VLLM_PID > /root/vllm.pid

echo "✅ vLLM started with PID: $VLLM_PID"
echo ""
echo "⏳ Waiting for model to load..."
echo "   This takes 60-90 seconds (or longer if downloading model)"
echo ""

# Step 10: Wait for ready
for i in {1..40}; do
    sleep 3
    ELAPSED=$((i * 3))
    
    # Show progress every 15 seconds
    if [ $((i % 5)) -eq 0 ]; then
        echo "   ${ELAPSED}s elapsed... (check logs: tail -20 /root/vllm.log)"
    fi
    
    # Test every 9 seconds
    if [ $((i % 3)) -eq 0 ]; then
        RESPONSE=$(curl -s http://localhost:8000/v1/models 2>&1)
        if echo "$RESPONSE" | grep -q "Meta-Llama"; then
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✅ SUCCESS! vLLM is ready!"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "📊 Model info:"
            curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>/dev/null | head -30
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🎉 INSTALLATION COMPLETE!"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "📌 Server Info:"
            echo "   • PID: $VLLM_PID (saved in /root/vllm.pid)"
            echo "   • Port: 8000"
            echo "   • Logs: /root/vllm.log"
            echo "   • Model: Meta-Llama-3.1-8B-Instruct"
            echo ""
            echo "🔧 Useful Commands:"
            echo "   • Test: curl http://localhost:8000/v1/models"
            echo "   • Logs: tail -f /root/vllm.log"
            echo "   • Stop: kill \$(cat /root/vllm.pid)"
            echo "   • Status: ps aux | grep vllm | grep -v grep"
            echo ""
            echo "✅ You can now CLOSE this terminal - vLLM will keep running!"
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "Next step: Go to your Mac and run './setup_fresh_tunnel.sh'"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            exit 0
        fi
    fi
done

# If we get here, check logs
echo ""
echo "⚠️  vLLM is taking longer than expected..."
echo ""
echo "Check if it's still loading:"
echo "  ps aux | grep vllm | grep -v grep"
echo ""
echo "Check logs for errors:"
echo "  tail -50 /root/vllm.log"
echo ""
echo "Common issues:"
echo "  1. Model still downloading - wait a few more minutes"
echo "  2. Need HuggingFace token - run: huggingface-cli login"
echo "  3. Out of memory - restart pod and try again"
echo ""
echo "Test manually after a few minutes:"
echo "  curl http://localhost:8000/v1/models"
echo ""
