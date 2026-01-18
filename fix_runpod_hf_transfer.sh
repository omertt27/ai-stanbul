#!/bin/bash
# Quick Fix for hf_transfer Error on RunPod
# Run this on your RunPod instance

echo "🔧 Installing missing hf_transfer package..."
pip install hf_transfer

echo ""
echo "✅ Installation complete!"
echo ""
echo "Now you can start vLLM with:"
echo ""
echo "export HF_TOKEN='your_token_here'"
echo "python -m vllm.entrypoints.openai.api_server \\"
echo "  --model meta-llama/Llama-3.1-8B-Instruct \\"
echo "  --quantization awq --dtype float16 \\"
echo "  --max-model-len 4096 --gpu-memory-utilization 0.9 \\"
echo "  --host 0.0.0.0 --port 8000"
