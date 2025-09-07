#!/bin/bash

# Simple and reliable Render build script
echo "🚀 AI-stanbul Backend Build for Render"

# Ensure we're in the right directory
cd backend || cd .

# Upgrade pip
python -m pip install --upgrade pip

# Install critical dependencies first (addresses common deployment errors)
echo "📦 Installing critical dependencies..."
python -m pip install python-multipart==0.0.6
python -m pip install fuzzywuzzy==0.18.0
python -m pip install python-levenshtein==0.20.9

# Install other core dependencies individually to catch errors
echo "📦 Installing core FastAPI dependencies..."
python -m pip install fastapi==0.104.1
python -m pip install uvicorn[standard]==0.24.0
python -m pip install sqlalchemy==2.0.23
python -m pip install python-dotenv==1.0.0
python -m pip install pydantic==2.5.0

# Install OpenAI dependencies
echo "📦 Installing AI dependencies..."
python -m pip install openai==1.3.0

# Install remaining dependencies
echo "📦 Installing remaining dependencies..."
python -m pip install -r requirements.txt

# Run dependency checker
echo "🔍 Running dependency verification..."
python check_dependencies.py

# Verify installation
echo "🔍 Final verification..."
python -c "
try:
    import fastapi
    import uvicorn
    import multipart
    import sqlalchemy
    import fuzzywuzzy
    import openai
    print('✅ All core dependencies verified')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo "✅ Build completed successfully!"
echo "🔧 All dependencies installed and verified"
