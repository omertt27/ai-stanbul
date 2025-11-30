#!/bin/bash
# Quick vLLM Setup Script for RunPod
# Run this inside your RunPod terminal

echo "🚀 Quick vLLM Setup for RunPod"
echo "================================"
echo ""

# Step 1: Download the model
echo "📥 Step 1: Downloading Llama 3.1 8B AWQ model..."
echo "This will take 5-10 minutes depending on connection speed"
echo ""
cd /workspace
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4

echo ""
echo "✅ Model downloaded!"
echo ""

# Step 2: Start vLLM
echo "🔧 Step 2: Starting vLLM server on port 8888..."
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

echo "✅ vLLM starting in background..."
echo ""
echo "📋 To monitor logs:"
echo "   tail -f /workspace/vllm.log"
echo ""
echo "⏰ Wait 2-3 minutes for vLLM to start, then test:"
echo "   curl http://localhost:8888/health"
echo ""
echo "🎯 Your RunPod endpoint will be:"
echo "   https://YOUR-POD-ID-8888.proxy.runpod.net/v1"
echo ""
