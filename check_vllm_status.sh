#!/bin/bash

echo "======================================"
echo "🔍 vLLM Status Diagnostic"
echo "======================================"
echo ""

echo "1️⃣  Testing external health endpoint..."
response=$(curl -s -w "\n%{http_code}" https://pvj233wwhiu6j3-8000.proxy.runpod.net/health 2>&1)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "200" ]; then
    echo "✅ Port 8000 is exposed and vLLM is responding!"
    echo "   Response: $body"
else
    echo "❌ Port 8000 not responding (HTTP $http_code)"
    echo "   This means vLLM is either:"
    echo "   - Not started yet"
    echo "   - Crashed"
    echo "   - Starting up (wait 30s and try again)"
fi

echo ""
echo "2️⃣  Testing external models endpoint..."
models_response=$(curl -s -w "\n%{http_code}" https://pvj233wwhiu6j3-8000.proxy.runpod.net/v1/models 2>&1)
models_code=$(echo "$models_response" | tail -n1)

if [ "$models_code" = "200" ]; then
    echo "✅ Models endpoint working!"
else
    echo "❌ Models endpoint not responding (HTTP $models_code)"
fi

echo ""
echo "3️⃣  Testing Jupyter Lab..."
jupyter_response=$(curl -s -o /dev/null -w "%{http_code}" https://pvj233wwhiu6j3-8888.proxy.runpod.net 2>&1)

if [ "$jupyter_response" = "200" ]; then
    echo "✅ Jupyter Lab is accessible"
    echo "   URL: https://pvj233wwhiu6j3-8888.proxy.runpod.net"
else
    echo "⚠️  Jupyter Lab not accessible"
fi

echo ""
echo "======================================"
echo "📋 Summary"
echo "======================================"

if [ "$http_code" = "200" ]; then
    echo "✅ vLLM IS RUNNING - Your LLM should work!"
    echo ""
    echo "Next steps:"
    echo "1. Make sure backend is running: cd backend && uvicorn main:app --reload"
    echo "2. Make sure frontend is running: cd frontend && npm run dev"
    echo "3. Test chat at: http://localhost:5173"
else
    echo "❌ vLLM IS NOT RUNNING - You need to start it!"
    echo ""
    echo "📝 TO FIX:"
    echo ""
    echo "Option 1: Open Web Terminal in RunPod"
    echo "  1. Go to: https://www.runpod.io/console/pods"
    echo "  2. Find your pod (pvj233wwhiu6j3)"
    echo "  3. Click 'Open Web Terminal'"
    echo "  4. Run:"
    echo ""
    echo "     pkill -9 -f vllm"
    echo "     python3 -m vllm.entrypoints.openai.api_server \\"
    echo "       --model meta-llama/Meta-Llama-3.1-8B-Instruct \\"
    echo "       --port 8000 \\"
    echo "       --host 0.0.0.0 \\"
    echo "       --dtype auto \\"
    echo "       --max-model-len 4096 \\"
    echo "       --gpu-memory-utilization 0.9 \\"
    echo "       > /root/vllm.log 2>&1 &"
    echo ""
    echo "  5. Wait 30 seconds"
    echo "  6. Test: curl http://localhost:8000/health"
    echo ""
    echo "Option 2: Via Jupyter Lab"
    echo "  1. Open: https://pvj233wwhiu6j3-8888.proxy.runpod.net"
    echo "  2. File → New → Terminal"
    echo "  3. Run the same commands as above"
    echo ""
    echo "📚 See: FIX_LLM_ERROR_NOW.md for detailed instructions"
fi

echo ""
echo "======================================"
