#!/bin/bash

# START vLLM ON RUNPOD - Copy these commands into RunPod Web Terminal

echo "======================================"
echo "STEP 1: Kill any existing vLLM"
echo "======================================"
pkill -9 -f vllm

echo ""
echo "======================================"
echo "STEP 2: Start vLLM Server"
echo "======================================"
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  > /root/vllm.log 2>&1 &

echo ""
echo "Waiting 30 seconds for vLLM to start..."
sleep 30

echo ""
echo "======================================"
echo "STEP 3: Test Health Endpoint"
echo "======================================"
curl http://localhost:8000/health

echo ""
echo ""
echo "======================================"
echo "STEP 4: Test Models Endpoint"
echo "======================================"
curl http://localhost:8000/v1/models

echo ""
echo ""
echo "======================================"
echo "STEP 5: Check Process"
echo "======================================"
ps aux | grep vllm | grep -v grep

echo ""
echo ""
echo "======================================"
echo "DONE! vLLM should be running on port 8000"
echo "======================================"
echo ""
echo "Now go back to RunPod UI and expose port 8000!"
