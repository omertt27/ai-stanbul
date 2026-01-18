#!/bin/bash

# 🚀 COMPLETE RUNPOD SETUP - Run this in RunPod Terminal
# This will install and start vLLM from scratch

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 RunPod vLLM Complete Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Update system
echo "📦 Step 1: Updating system packages..."
apt-get update -qq

# Step 2: Install vLLM
echo ""
echo "📦 Step 2: Installing vLLM..."
pip install --upgrade pip
pip install vllm

# Step 3: Check GPU
echo ""
echo "🎮 Step 3: Checking GPU..."
nvidia-smi
echo ""

# Step 4: Kill any existing vLLM
echo "🧹 Step 4: Cleaning up old processes..."
pkill -9 -f vllm 2>/dev/null || true
sleep 2

# Step 5: Start vLLM with conservative settings
echo ""
echo "🚀 Step 5: Starting vLLM server..."
echo "   Model: Meta-Llama-3.1-8B-Instruct"
echo "   Port: 8000"
echo "   Max context: 2048 tokens"
echo "   GPU memory: 50%"
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

# Step 6: Wait for model to load
echo "⏳ Step 6: Waiting for model to load (this takes 60-90 seconds)..."
echo ""

for i in {1..30}; do
    sleep 3
    ELAPSED=$((i * 3))
    
    # Show progress every 15 seconds
    if [ $((i % 5)) -eq 0 ]; then
        echo "   ${ELAPSED}s elapsed..."
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
            echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🎉 INSTALLATION COMPLETE!"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "📌 Important Info:"
            echo "   • PID: $VLLM_PID (saved in /root/vllm.pid)"
            echo "   • Port: 8000"
            echo "   • Logs: /root/vllm.log"
            echo ""
            echo "🔧 Useful Commands:"
            echo "   • Test: curl http://localhost:8000/v1/models"
            echo "   • Logs: tail -f /root/vllm.log"
            echo "   • Stop: kill \$(cat /root/vllm.pid)"
            echo "   • Status: ps aux | grep vllm"
            echo ""
            echo "✅ You can now close this terminal - vLLM will keep running!"
            echo ""
            exit 0
        fi
    fi
done

# If we get here, it's taking longer than expected
echo ""
echo "⚠️  vLLM is still loading (taking longer than usual)..."
echo ""
echo "Check status:"
echo "  ps aux | grep vllm | grep -v grep"
echo ""
echo "Check logs:"
echo "  tail -50 /root/vllm.log"
echo ""
echo "Test manually:"
echo "  curl http://localhost:8000/v1/models"
echo ""
echo "The process is running, but may need more time to load the model."
echo "Check back in a minute or two!"
echo ""
