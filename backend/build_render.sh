#!/bin/bash

# AI-stanbul Render Build Script
echo "🚀 Building AI-stanbul Backend for Render..."

# Navigate to backend directory
cd backend

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install python-multipart specifically first (common issue)
echo "📦 Installing python-multipart..."
pip install python-multipart==0.0.6

# Install all dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Verify critical dependencies
echo "🔍 Verifying installations..."
python -c "
import fastapi
import uvicorn
import multipart
import sqlalchemy
print('✅ Core dependencies verified')
"

echo "✅ Build completed successfully!"
echo "🚀 Ready to start with: python start.py"
