#!/bin/bash

# Commands to run INSIDE the RunPod terminal
# Copy-paste these one by one into your RunPod Web Terminal

echo "======================================"
echo "1. Check if vLLM process is running"
echo "======================================"
echo "Run: ps aux | grep vllm | grep -v grep"
echo ""

echo "======================================"
echo "2. Check vLLM logs (last 50 lines)"
echo "======================================"
echo "Run: tail -50 /root/vllm.log"
echo ""

echo "======================================"
echo "3. Watch logs in real-time"
echo "======================================"
echo "Run: tail -f /root/vllm.log"
echo "(Press Ctrl+C to stop)"
echo ""

echo "======================================"
echo "4. Test health endpoint (local)"
echo "======================================"
echo "Run: curl http://localhost:8000/health"
echo ""

echo "======================================"
echo "5. Test models endpoint (local)"
echo "======================================"
echo "Run: curl http://localhost:8000/v1/models"
echo ""

echo "======================================"
echo "What to look for in logs:"
echo "======================================"
echo "✅ 'Application startup complete' - vLLM is ready"
echo "✅ 'Uvicorn running on' - Server started"
echo "❌ 'CUDA out of memory' - Reduce --gpu-memory-utilization to 0.7"
echo "❌ 'Model not found' - Model still downloading"
echo ""
