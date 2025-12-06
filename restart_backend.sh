#!/bin/bash
# Quick restart script for AI Istanbul backend

echo "🔄 Restarting AI Istanbul Backend..."
echo "======================================"

# Find and kill existing backend process
echo "1️⃣ Stopping existing backend..."
PID=$(ps aux | grep "uvicorn main:app" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "   Found backend process: $PID"
    kill $PID
    sleep 2
    echo "   ✅ Backend stopped"
else
    echo "   ℹ️  No existing backend found"
fi

# Navigate to backend directory
cd "$(dirname "$0")/backend" || exit 1

# Start backend with auto-reload
echo ""
echo "2️⃣ Starting backend with auto-reload..."
echo "   URL: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================"

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
