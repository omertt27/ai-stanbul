#!/bin/bash

# 🚀 Istanbul AI Development Setup Script

set -e  # Exit on any error

echo "🌍 Setting up Istanbul AI development environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.docker .env
    echo "⚠️  Please edit .env file with your API keys before continuing!"
    read -p "Press Enter after you've updated the .env file..."
fi

# Build and start containers
echo "🐳 Building Docker containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d postgres redis

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Run database migrations
echo "📊 Running database migrations..."
docker-compose run --rm backend alembic upgrade head

# Start all services
echo "🎉 Starting all services..."
docker-compose up -d

echo "✅ Development environment is ready!"
echo ""
echo "🌐 Services:"
echo "  Frontend:  http://localhost:3000"
echo "  Backend:   http://localhost:8000"
echo "  Database:  postgresql://localhost:5432/istanbul_ai"
echo "  Redis:     redis://localhost:6379"
echo ""
echo "📋 Useful commands:"
echo "  docker-compose logs -f          # View logs"
echo "  docker-compose down             # Stop all services"
echo "  docker-compose restart backend  # Restart backend only"
echo "  docker-compose exec backend bash # Access backend container"
echo ""
echo "🧪 Run tests:"
echo "  docker-compose run --rm backend pytest"
echo "  docker-compose run --rm frontend npm test"
