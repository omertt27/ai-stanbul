#!/bin/bash
# Quick fix: Rebuild Docker image without cache

echo "🔧 Rebuilding Docker image without cache..."
echo ""

# Build without cache to use the fixed Dockerfile
docker build --no-cache -f Dockerfile.4bit -t ai-istanbul-llm-4bit:latest .

echo ""
echo "✅ Docker image rebuilt successfully!"
echo ""
echo "Now you can continue with the deployment:"
echo "1. The image is ready locally"
echo "2. Continue with: ./deploy_to_ecs.sh"
echo "   (It will skip steps 1-3 and push the new image)"
