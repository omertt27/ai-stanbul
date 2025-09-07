#!/bin/bash

# Render startup script with proper port handling and Python version check
# This ensures the PORT environment variable is properly expanded

echo "🚀 Starting AIstanbul backend..."
echo "Python version: $(python --version)"
echo "PORT environment variable: $PORT"

# Use the PORT environment variable, fallback to 8000 if not set
FINAL_PORT=${PORT:-8000}

echo "Starting server on 0.0.0.0:$FINAL_PORT"

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "❌ uvicorn not found, installing..."
    pip install uvicorn[standard]==0.24.0
fi

# Execute uvicorn with the resolved port
exec uvicorn backend.main:app --host 0.0.0.0 --port $FINAL_PORT --log-level info --access-log
