#!/bin/bash

# Start AI Istanbul Services

echo "🚀 Starting AI Istanbul Backend..."
cd /Users/omer/Desktop/ai-stanbul/backend
/Users/omer/Desktop/ai-stanbul/.venv/bin/python main.py &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

echo "🎨 Starting Frontend..."
cd /Users/omer/Desktop/ai-stanbul/frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

echo "✅ Services started!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "To stop services, run:"
echo "kill $BACKEND_PID $FRONTEND_PID"

# Wait for both processes
wait
