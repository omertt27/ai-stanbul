#!/bin/bash
# Quick Start Script for ML Answering System
# This script starts the ML API service and runs tests

set -e

echo "🚀 Istanbul AI - ML Answering System Quick Start"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "ml_api_service.py" ]; then
    echo "❌ Error: ml_api_service.py not found"
    echo "   Please run this script from the project root directory"
    exit 1
fi

# Check if semantic index exists
if [ ! -f "data/semantic_index.bin" ]; then
    echo "⚠️  Warning: Semantic index not found"
    echo "   You may need to run: python scripts/index_database.py"
fi

echo "1️⃣ Checking dependencies..."
python -c "import fastapi, torch, transformers, sentence_transformers" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements_ml.txt
}

echo "✅ Dependencies OK"
echo ""

echo "2️⃣ Starting ML API Service on port 8001..."
echo "   (Press Ctrl+C to stop)"
echo ""

# Start the service
python ml_api_service.py
