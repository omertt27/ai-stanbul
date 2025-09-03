#!/bin/bash
# Production Deployment Script

echo "🚀 Starting production deployment..."

# Backend Setup
echo "📦 Setting up backend..."
cd backend

# Install dependencies
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
