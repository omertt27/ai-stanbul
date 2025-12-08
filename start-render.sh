#!/bin/bash
# Render startup script for AI Istanbul

set -e  # Exit on any error

echo "🚀 Starting AI Istanbul Backend on Render..."

# Navigate to backend directory
cd backend

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Print debug information
echo "📍 Current working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "📦 Python path: $PYTHONPATH"

# Check if critical files exist
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in $(pwd)"
    exit 1
fi

if [ ! -f "database.py" ]; then
    echo "❌ database.py not found in $(pwd)"
    exit 1
fi

echo "✅ Critical files verified"

# Test imports before starting
echo "🔍 Testing imports..."
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())
try:
    from database import engine, SessionLocal
    print('✅ Database import successful')
    from models import Base, Restaurant, Museum, Place, FeedbackEvent, IntentFeedback
    print('✅ Real-time learning models imported successfully')
    from models import GPSRoute, RouteWaypoint
    print('✅ GPS Navigation models imported successfully')
    print('✅ Models import successful')
    print('🎯 All critical imports successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

echo "🏃 Starting FastAPI server..."

# Start the server with proper error handling
if [ "$PORT" ]; then
    echo "🌐 Using PORT from environment: $PORT"
    exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
else
    echo "🌐 Using default port 8000"
    exec uvicorn main:app --host 0.0.0.0 --port 8000
fi
