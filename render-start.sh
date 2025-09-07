#!/bin/bash
# Render deployment script - handles PORT environment variable properly
echo "🚀 Starting AIstanbul backend for Render deployment..."
echo "PORT: ${PORT:-8000}"
echo "HOST: 0.0.0.0"

# Use uvicorn directly with proper environment variable handling
exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
