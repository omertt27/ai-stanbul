#!/bin/bash
# RunPod vLLM Memory Fix and Startup Script
# This script clears GPU memory and starts vLLM with optimized settings

echo "🔍 Checking GPU memory usage..."
nvidia-smi

echo ""
echo "🧹 Killing any existing Python/vLLM processes..."
pkill -f vllm || true
pkill -f "python.*llama" || true
sleep 2

echo ""
echo "🔄 Clearing GPU memory..."
nvidia-smi --gpu-reset || true

echo ""
echo "📊 GPU memory after cleanup:"
nvidia-smi

echo ""
echo "🚀 Starting vLLM with optimized memory settings..."
echo "   - GPU Memory Utilization: 0.85 (85% instead of 90%)"
echo "   - Max Model Length: 2048 (reduced from 4096)"
echo "   - Using 4-bit quantization"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000
