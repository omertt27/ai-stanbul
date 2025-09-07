#!/bin/bash

# Pre-deploy script for Render
echo "🔧 Running pre-deploy setup..."

# Check Python version
echo "Python version: $(python --version)"

# Ensure we have the required directories
mkdir -p images uploads

# Set proper permissions
chmod +x start-render.sh

# Check if main dependencies are available
python -c "import fastapi; print('✅ FastAPI available')" || echo "❌ FastAPI not available"
python -c "import uvicorn; print('✅ Uvicorn available')" || echo "❌ Uvicorn not available"

echo "✅ Pre-deploy setup complete"
