#!/bin/bash

# Start Llama 3.1 8B 4-bit Model Server on RunPod
# Model: meta-llama/Meta-Llama-3.1-8B-Instruct (4-bit quantized)

echo "=============================================="
echo "🚀 Starting Llama 3.1 8B Instruct (4-bit)"
echo "=============================================="
echo ""

# Check if vLLM is installed
echo "Checking vLLM installation..."
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM not found. Installing..."
    pip install vllm
else
    echo "✅ vLLM already installed"
fi
echo ""

# Start the server
echo "Starting LLM server on port 8888..."
echo "Model: meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "Quantization: 4-bit (automatic by vLLM)"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --quantization awq \
  --dtype auto \
  --max-model-len 4096 \
  --served-model-name "Meta-Llama-3.1-8B-Instruct" &

SERVER_PID=$!
echo "Server starting with PID: $SERVER_PID"
echo ""

# Wait for server to start
echo "Waiting 45 seconds for model to load..."
for i in {45..1}; do
    echo -ne "⏳ $i seconds remaining...\r"
    sleep 1
done
echo ""
echo ""

# Test the server
echo "Testing server health..."
HEALTH_CHECK=$(curl -s http://localhost:8888/health)
if [ $? -eq 0 ]; then
    echo "✅ Server is responding!"
    echo "$HEALTH_CHECK"
else
    echo "⚠️ Server not responding yet, checking logs..."
    sleep 15
    curl -s http://localhost:8888/health || echo "❌ Still not responding"
fi
echo ""

# Test models endpoint
echo "Testing models endpoint..."
curl -s http://localhost:8888/v1/models | python3 -m json.tool 2>/dev/null || echo "Models endpoint not ready yet"
echo ""

echo "=============================================="
echo "Server Status:"
echo "  PID: $SERVER_PID"
echo "  Port: 8888"
echo "  Model: Meta-Llama-3.1-8B-Instruct (4-bit)"
echo "  Endpoint: http://localhost:8888"
echo ""
echo "To check logs:"
echo "  tail -f /tmp/vllm-*.log"
echo ""
echo "To stop server:"
echo "  kill $SERVER_PID"
echo "=============================================="
