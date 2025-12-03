#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 🚀 START LLM SERVER WITH NOHUP - Persistent Background Process
# ═══════════════════════════════════════════════════════════════
# This ensures the server keeps running even after SSH disconnect

set -e  # Exit on error

echo "🚀 Starting LLM Server with nohup (persistent mode)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Change to workspace
cd /workspace

# Check if model exists
if [ ! -d "/workspace/models" ]; then
    echo "❌ Model directory not found!"
    echo "Run: ./download_model.sh"
    exit 1
fi

# Check if llm_server.py exists
if [ ! -f "/workspace/llm_server.py" ]; then
    echo "❌ llm_server.py not found!"
    echo "Upload files first"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -q fastapi uvicorn[standard] transformers torch accelerate bitsandbytes pydantic requests 2>&1 | grep -v "already satisfied" || true
echo "✅ Dependencies installed!"
echo ""

# Create logs directory
mkdir -p /workspace/logs

# Kill any existing server
echo "🔍 Stopping any existing servers..."
if [ -f "/workspace/llm_server.pid" ]; then
    OLD_PID=$(cat /workspace/llm_server.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "   Killing old server (PID: $OLD_PID)"
        kill $OLD_PID 2>/dev/null || true
        sleep 2
    fi
fi

# Also kill by process name as backup
pkill -f "llm_server.py" 2>/dev/null || true
sleep 2

# Start server with nohup (survives SSH disconnect)
echo "🔄 Starting server with nohup..."
echo "   Command: nohup python /workspace/llm_server.py > /workspace/logs/llm_server.log 2>&1 &"
echo ""

nohup python /workspace/llm_server.py > /workspace/logs/llm_server.log 2>&1 &

# Get PID
SERVER_PID=$!
echo $SERVER_PID > /workspace/llm_server.pid

# Disown to ensure persistence
disown 2>/dev/null || true

echo "✅ Server started!"
echo "   PID: $SERVER_PID"
echo "   Log: /workspace/logs/llm_server.log"
echo ""

# Wait for server to initialize
echo "⏳ Waiting for model to load (30 seconds)..."
for i in {1..6}; do
    echo -n "."
    sleep 5
done
echo ""
echo ""

# Check if process is still running
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✅ Server process is running!"
else
    echo "❌ Server process stopped unexpectedly!"
    echo ""
    echo "Last 30 lines of log:"
    tail -n 30 /workspace/logs/llm_server.log
    exit 1
fi

# Test health endpoint
echo "🧪 Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo "")

if [ -n "$HEALTH_RESPONSE" ]; then
    echo "✅ Server is responding!"
    echo ""
    echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎉 LLM SERVER IS READY!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📊 Server Info:"
    echo "   • PID: $SERVER_PID"
    echo "   • Port: 8000"
    echo "   • Logs: /workspace/logs/llm_server.log"
    echo ""
    echo "🔍 Check status:"
    echo "   ps aux | grep llm_server.py"
    echo ""
    echo "📜 View logs:"
    echo "   tail -f /workspace/logs/llm_server.log"
    echo ""
    echo "🧪 Test completion:"
    echo '   curl -X POST http://localhost:8000/v1/completions \'
    echo '     -H "Content-Type: application/json" \'
    echo '     -d '"'"'{"prompt": "Istanbul is", "max_tokens": 30}'"'"' | python3 -m json.tool'
    echo ""
    echo "🛑 Stop server:"
    echo "   kill $(cat /workspace/llm_server.pid)"
    echo ""
    echo "✅ Server will keep running even after you disconnect from SSH!"
    echo ""
else
    echo "⚠️  Server started but not responding yet"
    echo "   Model may still be loading..."
    echo ""
    echo "Check logs:"
    echo "   tail -f /workspace/logs/llm_server.log"
    echo ""
    echo "Wait 1-2 minutes and test again:"
    echo "   curl http://localhost:8000/health | python3 -m json.tool"
    echo ""
fi
