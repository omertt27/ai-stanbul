#!/bin/bash
# Quick Start Script for vLLM on RunPod
# Copy-paste this entire script into Jupyter Lab terminal

echo "============================================"
echo "🚀 Starting vLLM Server on RunPod"
echo "============================================"

# Kill any existing vLLM processes
echo "📋 Checking for existing vLLM processes..."
if ps aux | grep -v grep | grep vllm > /dev/null; then
    echo "⚠️  Found existing vLLM process, killing it..."
    pkill -9 -f vllm
    sleep 3
fi

# Check if model exists locally
echo "📋 Checking for model..."
if [ -d "/workspace/Meta-Llama-3.1-8B-Instruct" ]; then
    MODEL_PATH="/workspace/Meta-Llama-3.1-8B-Instruct"
    echo "✅ Found local model: $MODEL_PATH"
elif [ -d "/root/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct" ]; then
    MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "✅ Found cached model"
else
    MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "⚠️  Model not found locally, will download from HuggingFace"
    echo "   This may take a while on first run..."
fi

# Start vLLM server
echo ""
echo "🚀 Starting vLLM server..."
echo "   Model: $MODEL_PATH"
echo "   Port: 8000"
echo "   Host: 0.0.0.0"
echo ""

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  > /root/vllm.log 2>&1 &

VLLM_PID=$!

echo "✅ vLLM started with PID: $VLLM_PID"
echo ""
echo "⏳ Waiting for vLLM to initialize (this takes 30-60 seconds)..."

# Wait for vLLM to start
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "============================================"
        echo "✅ vLLM is UP and RUNNING!"
        echo "============================================"
        echo ""
        echo "📊 Server Status:"
        curl -s http://localhost:8000/health
        echo ""
        echo ""
        echo "📋 Available Models:"
        curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/v1/models
        echo ""
        echo ""
        echo "🌐 External URL:"
        echo "   https://pvj233wwhiu6j3-8000.proxy.runpod.net"
        echo ""
        echo "🔗 Endpoints:"
        echo "   Health:      https://pvj233wwhiu6j3-8000.proxy.runpod.net/health"
        echo "   Models:      https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/models"
        echo "   Completions: https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/completions"
        echo ""
        echo "📝 Useful Commands:"
        echo "   View logs:    tail -f /root/vllm.log"
        echo "   Check status: ps aux | grep vllm"
        echo "   Stop server:  pkill -9 -f vllm"
        echo ""
        echo "✅ Ready for Istanbul AI backend!"
        exit 0
    fi
    echo -n "."
    sleep 1
done

# If we get here, vLLM didn't start
echo ""
echo "============================================"
echo "❌ vLLM failed to start"
echo "============================================"
echo ""
echo "📋 Last 50 lines of log:"
tail -50 /root/vllm.log
echo ""
echo "💡 Troubleshooting:"
echo "   1. Check full logs: tail -f /root/vllm.log"
echo "   2. Check GPU: nvidia-smi"
echo "   3. Try reducing memory: --max-model-len 2048 --gpu-memory-utilization 0.8"
echo "   4. See: START_VLLM_ON_RUNPOD.md for more help"
exit 1
