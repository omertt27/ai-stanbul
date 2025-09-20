#!/bin/bash

# AI Istanbul Production Deployment Test
echo "🧪 Testing Production Deployment Setup"
echo "======================================"

# Test critical files exist
FILES=("docker-compose.prod.yml" "Dockerfile.prod" "nginx/nginx-production.conf" "deploy-production.sh")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

echo "✅ All production deployment files are ready!"
echo "🚀 Run ./deploy-production.sh to deploy to production"
