#!/bin/bash

# 🚀 Production Deployment Script

set -e

echo "🚀 Deploying Istanbul AI to production..."

# Check if required environment variables are set
required_vars=("OPENAI_API_KEY" "GOOGLE_PLACES_API_KEY" "WEATHER_API_KEY" "DATABASE_URL")

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set"
        exit 1
    fi
done

# Build production image
echo "🏗️  Building production Docker image..."
docker build -t istanbul-ai:latest .

# Tag for registry
if [ ! -z "$DOCKER_REGISTRY" ]; then
    echo "🏷️  Tagging image for registry..."
    docker tag istanbul-ai:latest $DOCKER_REGISTRY/istanbul-ai:latest
    
    echo "📤 Pushing to registry..."
    docker push $DOCKER_REGISTRY/istanbul-ai:latest
fi

# Deploy based on platform
if [ "$DEPLOY_PLATFORM" = "render" ]; then
    echo "🚀 Deploying to Render..."
    curl -X POST "$RENDER_DEPLOY_HOOK"
    
elif [ "$DEPLOY_PLATFORM" = "railway" ]; then
    echo "🚀 Deploying to Railway..."
    railway deploy
    
elif [ "$DEPLOY_PLATFORM" = "fly" ]; then
    echo "🚀 Deploying to Fly.io..."
    flyctl deploy
    
else
    echo "⚠️  No deployment platform specified. Image built successfully."
fi

# Run database migrations
if [ ! -z "$DATABASE_URL" ]; then
    echo "📊 Running database migrations..."
    docker run --rm -e DATABASE_URL="$DATABASE_URL" istanbul-ai:latest alembic upgrade head
fi

# Health check
echo "🏥 Running health check..."
sleep 30  # Wait for deployment

if [ ! -z "$PRODUCTION_URL" ]; then
    health_status=$(curl -s -o /dev/null -w "%{http_code}" "$PRODUCTION_URL/health")
    
    if [ "$health_status" = "200" ]; then
        echo "✅ Deployment successful! Health check passed."
    else
        echo "❌ Health check failed with status: $health_status"
        exit 1
    fi
fi

echo "🎉 Production deployment completed successfully!"
