#!/bin/bash

# Production startup script for AI Istanbul Backend
# This script ensures all dependencies are installed and the server starts properly

echo "🚀 Starting AIstanbul backend production deployment..."

# Set environment variables
export PYTHONPATH="/opt/render/project/src/backend:$PYTHONPATH"
export PORT=${PORT:-10000}

echo "📦 Installing any missing dependencies..."

# Install missing dependencies individually to handle failures gracefully
pip install slowapi || echo "⚠️ slowapi install failed, continuing without rate limiting"
pip install structlog || echo "⚠️ structlog install failed, using fallback logging"
pip install aiohttp || echo "⚠️ aiohttp install failed, advanced features disabled"
pip install google-analytics-data || echo "⚠️ google-analytics-data install failed, analytics disabled"
pip install redis==4.6.0 || echo "⚠️ redis install failed, using memory cache"

echo "🔧 Checking Python environment..."
python --version
echo "📍 Current directory: $(pwd)"
echo "🐍 Python path: $PYTHONPATH"

echo "🌐 Starting server on 0.0.0.0:$PORT"

# Start the server with proper error handling
cd /opt/render/project/src/backend
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
