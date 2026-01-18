#!/bin/bash

# RunPod LLM Server Startup Script
# Starts the FastAPI LLM server for Llama 3.1 8B Instruct

echo "🚀 Starting RunPod LLM Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Change to workspace
cd /workspace

# Check if model exists
if [ ! -d "/workspace/models" ]; then
    echo "❌ Model directory not found!"
    echo "Please run download_model.sh first"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q fastapi uvicorn transformers torch accelerate bitsandbytes 2>&1 | grep -v "already satisfied" || true

# Create logs directory
mkdir -p /workspace/logs

# Copy llm_server.py if not exists
if [ ! -f "/workspace/llm_server.py" ]; then
    echo "⚠️  llm_server.py not found in /workspace"
    echo "Please upload llm_server.py to /workspace first"
    exit 1
fi

# Kill any existing server
echo "🔍 Checking for existing servers..."
pkill -f "llm_server.py" || true
sleep 2

# Start server in background with nohup (survives SSH disconnect)
echo "🔄 Starting FastAPI server with nohup..."
nohup python /workspace/llm_server.py > /workspace/logs/llm_server.log 2>&1 &

# Get PID
SERVER_PID=$!
echo "✅ Server started with PID: $SERVER_PID"
echo $SERVER_PID > /workspace/llm_server.pid

# Disown the process to ensure it persists after shell exit
disown

# Wait for startup
echo "⏳ Waiting for server to initialize (this may take 30-60 seconds)..."
sleep 15

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server process is running!"
    
    # Wait a bit more for model loading
    sleep 15
    
    # Test health endpoint
    echo "🧪 Testing health endpoint..."
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Server is responding!"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ LLM Server is ready!"
        echo ""
        echo "📊 Server Information:"
        echo "   • PID: $SERVER_PID"
        echo "   • Port: 8000"
        echo "   • Health: http://localhost:8000/health"
        echo "   • Docs: http://localhost:8000/docs"
        echo ""
        echo "🔍 View logs:"
        echo "   tail -f /workspace/logs/llm_server.log"
        echo ""
        echo "🧪 Test health:"
        echo "   curl http://localhost:8000/health | jq"
        echo ""
        echo "🧪 Test completion:"
        echo '   curl -X POST http://localhost:8000/v1/completions \'
        echo '     -H "Content-Type: application/json" \'
        echo '     -d '"'"'{"prompt": "What are the top 3 tourist attractions in Istanbul?", "max_tokens": 100}'"'"' | jq'
        echo ""
        echo "🛑 Stop server:"
        echo "   kill $SERVER_PID"
        echo "   # or: pkill -f llm_server.py"
        echo ""
    else
        echo "⚠️  Server started but not responding yet"
        echo "Model may still be loading. Check logs:"
        echo "   tail -f /workspace/logs/llm_server.log"
    fi
else
    echo "❌ Server failed to start!"
    echo "Check logs: /workspace/logs/llm_server.log"
    echo ""
    echo "Last 20 lines of log:"
    tail -n 20 /workspace/logs/llm_server.log
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
