#!/bin/bash

# Render startup script with proper port handling
# This ensures the PORT environment variable is properly expanded

echo "🚀 Starting AIstanbul backend..."
echo "PORT environment variable: $PORT"

# Use the PORT environment variable, fallback to 8000 if not set
FINAL_PORT=${PORT:-8000}

echo "Starting server on 0.0.0.0:$FINAL_PORT"

# Execute uvicorn with the resolved port
exec uvicorn backend.main:app --host 0.0.0.0 --port $FINAL_PORT --log-level info --access-log
