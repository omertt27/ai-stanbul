#!/bin/bash

# 🚀 Start AI Istanbul - Pure LLM System
# This script starts all components of the Pure LLM architecture

echo "🚀 Starting AI Istanbul - Pure LLM System"
echo "========================================"
echo ""

# Check if backend is already running
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Backend already running on port 8001"
    echo "   To restart: lsof -ti:8001 | xargs kill -9"
else
    echo "🔧 Starting Backend (Pure LLM)..."
    nohup python3 backend/main_pure_llm.py > backend_startup.log 2>&1 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
    sleep 3
    
    # Check if backend started successfully
    if curl -s http://localhost:8001/health | grep -q "healthy"; then
        echo "   ✅ Backend started successfully!"
    else
        echo "   ❌ Backend failed to start. Check backend_startup.log"
        exit 1
    fi
fi

echo ""

# Check if frontend is already running
if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Frontend already running on port 5173"
else
    echo "🎨 Starting Frontend (React)..."
    cd frontend
    npm run dev > /dev/null 2>&1 &
    FRONTEND_PID=$!
    cd ..
    echo "   Frontend PID: $FRONTEND_PID"
    sleep 3
    echo "   ✅ Frontend starting..."
fi

echo ""
echo "========================================"
echo "✅ AI Istanbul is starting!"
echo "========================================"
echo ""
echo "📍 URLs:"
echo "   Frontend:  http://localhost:5173"
echo "   Backend:   http://localhost:8001"
echo "   API Docs:  http://localhost:8001/docs"
echo "   Health:    http://localhost:8001/health"
echo ""
echo "🧪 Test it:"
echo "   curl http://localhost:8001/health"
echo ""
echo "📊 View logs:"
echo "   Backend:  tail -f backend_startup.log"
echo "   Frontend: cd frontend && npm run dev"
echo ""
echo "🛑 Stop all:"
echo "   lsof -ti:8001 | xargs kill -9  # Backend"
echo "   lsof -ti:5173 | xargs kill -9  # Frontend"
echo ""
echo "🎉 Open http://localhost:5173 and start chatting!"
echo ""
