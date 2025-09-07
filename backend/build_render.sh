#!/bin/bash

# AI-stanbul Render Build Script
echo "🚀 Building AI-stanbul Backend for Render..."

# Navigate to backend directory
cd backend

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build completed successfully!"
echo "🚀 Ready to start with: python start.py"
