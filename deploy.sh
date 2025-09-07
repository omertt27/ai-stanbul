#!/bin/bash

# AI-stanbul Production Deployment Script
# This script handles the complete deployment process with error handling

echo "🚀 Starting AI-stanbul Production Deployment..."

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Backend deployment
echo "📦 Setting up Backend..."
cd backend

# Install Python dependencies with error handling
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt || {
    echo "❌ Failed to install Python dependencies"
    exit 1
}

# Check if critical dependencies are installed
echo "🔍 Verifying critical dependencies..."
python -c "
import sys
try:
    from fuzzywuzzy import fuzz
    print('✅ fuzzywuzzy installed successfully')
except ImportError:
    print('⚠️  fuzzywuzzy not found, installing...')
    import subprocess
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fuzzywuzzy', 'python-levenshtein'])
        print('✅ fuzzywuzzy installed successfully')
    except subprocess.CalledProcessError:
        print('⚠️  Could not install fuzzywuzzy, using fallback mode')

try:
    import fastapi
    print('✅ FastAPI installed successfully')
except ImportError:
    print('❌ FastAPI not found - this is required!')
    sys.exit(1)

try:
    import openai
    print('✅ OpenAI client installed successfully')
except ImportError:
    print('⚠️  OpenAI client not found - some features may be limited')
"

# Initialize database
echo "🗄️  Initializing database..."
python init_db.py || {
    echo "⚠️  Database initialization had issues, but continuing..."
}
pip install -r requirements.txt

# Add production dependencies
pip install gunicorn psycopg2-binary python-dotenv

# Initialize database
python init_db.py

# Populate with seed data
sqlite3 app.db < db/seed.sql

echo "✅ Backend setup complete"

# Frontend Setup
echo "🎨 Building frontend..."
cd ../frontend

# Install dependencies
npm install

# Build for production
npm run build

echo "✅ Frontend build complete"

# Final checks
echo "🔍 Running final checks..."

# Check if build directory exists
if [ -d "dist" ]; then
    echo "✅ Frontend dist directory exists"
else
    echo "❌ Frontend build failed - no dist directory"
    exit 1
fi

# Check if backend dependencies are installed
cd ../backend
if pip show fastapi uvicorn gunicorn > /dev/null 2>&1; then
    echo "✅ Backend dependencies installed"
else
    echo "❌ Backend dependencies missing"
    exit 1
fi

echo "🎉 Deployment preparation complete!"
echo "📋 Next steps:"
echo "   1. Set up your production environment variables"
echo "   2. Deploy to your hosting service"
echo "   3. Configure domain and SSL"
echo "   4. Run database migrations"
