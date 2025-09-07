#!/bin/bash

# AI-stanbul Production Deployment with Fallback
echo "🚀 Starting AI-stanbul Deployment with Fallback Support..."

cd backend

# Try to install all dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test if all modules can be imported
echo "🔍 Testing imports..."
python -c "
import sys
import importlib

modules_to_test = [
    'fastapi',
    'uvicorn', 
    'sqlalchemy',
    'openai',
    'fuzzywuzzy',
    'database',
    'models',
    'enhanced_chatbot'
]

missing_modules = []
for module in modules_to_test:
    try:
        importlib.import_module(module)
        print(f'✅ {module}')
    except ImportError as e:
        print(f'❌ {module}: {e}')
        missing_modules.append(module)

if missing_modules:
    print(f'⚠️  Missing modules: {missing_modules}')
    print('🔄 Will use minimal mode')
    sys.exit(1)
else:
    print('✅ All modules available - using full mode')
    sys.exit(0)
"

# Check the exit code
if [ $? -eq 0 ]; then
    echo "🚀 Starting full-featured backend..."
    MAIN_FILE="main.py"
else
    echo "🔄 Starting minimal backend..."
    MAIN_FILE="main_minimal.py"
fi

# Start the appropriate server
echo "🎯 Using $MAIN_FILE"

if [ "$NODE_ENV" = "production" ]; then
    gunicorn ${MAIN_FILE%.*}:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
else
    uvicorn ${MAIN_FILE%.*}:app --host 0.0.0.0 --port 8001 --reload
fi
