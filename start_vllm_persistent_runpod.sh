#!/bin/bash

# 🚀 Start vLLM on RunPod - Persistent (survives terminal close)
# Run this script in RunPod terminal

echo "🚀 Starting vLLM (persistent mode)..."
echo ""

# Kill any existing vLLM process
echo "Stopping any existing vLLM..."
pkill -9 -f vllm 2>/dev/null || true
sleep 2

# Start vLLM with nohup (keeps running after terminal closes)
echo "Starting vLLM server..."
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  > /root/vllm.log 2>&1 &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"
echo ""

# Save PID for later
echo $VLLM_PID > /root/vllm.pid

echo "✅ vLLM is now running in the background"
echo ""
echo "Waiting for vLLM to load model (60-90 seconds)..."
echo ""

# Wait and check
for i in {1..18}; do
    sleep 5
    ELAPSED=$((i * 5))
    echo "⏳ ${ELAPSED}s elapsed..."
    
    RESPONSE=$(curl -s http://localhost:8000/v1/models 2>&1)
    if echo "$RESPONSE" | grep -q "Meta-Llama"; then
        echo ""
        echo "✅ SUCCESS! vLLM is ready!"
        echo ""
        echo "Model loaded:"
        echo "$RESPONSE" | grep -o '"id":"[^"]*"' | head -1
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ vLLM is running persistently!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "📌 Important:"
        echo "  • You can now CLOSE this terminal safely"
        echo "  • vLLM will keep running in the background"
        echo "  • Check status anytime: ps aux | grep vllm"
        echo "  • View logs: tail -f /root/vllm.log"
        echo "  • Stop it: kill \$(cat /root/vllm.pid)"
        echo ""
        exit 0
    fi
done

echo ""
echo "⚠️  vLLM is starting but not ready yet"
echo ""
echo "Check status with:"
echo "  ps aux | grep vllm"
echo ""
echo "Check logs:"
echo "  tail -f /root/vllm.log"
echo ""
echo "Test when ready:"
echo "  curl http://localhost:8000/v1/models"
echo ""
