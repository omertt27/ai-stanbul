#!/bin/bash

# AI Istanbul Production Deployment Script
# Optimized for production with Nginx reverse proxy

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AI Istanbul Production Deployment${NC}"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Check for required files
REQUIRED_FILES=(
    ".env.prod"
    "docker-compose.prod.yml"
    "nginx/nginx-production.conf"
    "Dockerfile.prod"
    "frontend/Dockerfile.prod"
)

echo -e "${YELLOW}🔍 Checking required files...${NC}"
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ Missing required file: $file${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Found: $file${NC}"
done

# Load environment variables
if [ -f ".env.prod" ]; then
    echo -e "${YELLOW}📋 Loading production environment variables...${NC}"
    export $(cat .env.prod | grep -v '^#' | xargs)
else
    echo -e "${RED}❌ Missing .env.prod file${NC}"
    exit 1
fi

# Validate critical environment variables
REQUIRED_VARS=(
    "OPENAI_API_KEY"
    "POSTGRES_PASSWORD"
    "SECRET_KEY"
    "DOMAIN_NAME"
    "ADMIN_EMAIL"
)

echo -e "${YELLOW}🔐 Validating environment variables...${NC}"
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}❌ Missing required environment variable: $var${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ $var is set${NC}"
done

# Create necessary directories
echo -e "${YELLOW}📁 Creating necessary directories...${NC}"
mkdir -p logs/nginx
mkdir -p nginx/cache
mkdir -p nginx/ssl
mkdir -p nginx/certbot-webroot
mkdir -p database/backup
mkdir -p data
echo -e "${GREEN}✅ Directories created${NC}"

# Stop existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose -f docker-compose.prod.yml down --remove-orphans || true

# Pull latest images
echo -e "${YELLOW}📥 Pulling latest base images...${NC}"
docker-compose -f docker-compose.prod.yml pull postgres redis nginx

# Build application images
echo -e "${YELLOW}🏗️ Building application images...${NC}"
docker-compose -f docker-compose.prod.yml build --no-cache

# Pre-deployment health checks
echo -e "${YELLOW}🏥 Running pre-deployment health checks...${NC}"

# Check disk space (require at least 2GB free)
DISK_SPACE=$(df . | tail -1 | awk '{print $4}')
if [ "$DISK_SPACE" -lt 2097152 ]; then
    echo -e "${RED}❌ Insufficient disk space. Need at least 2GB free${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Sufficient disk space available${NC}"

# Check if ports are available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}❌ Port $port is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $port is available${NC}"
        return 0
    fi
}

check_port 80 || exit 1
check_port 443 || exit 1

# Start the database first
echo -e "${YELLOW}🗄️ Starting database services...${NC}"
docker-compose -f docker-compose.prod.yml up -d postgres redis

# Wait for database to be ready
echo -e "${YELLOW}⏳ Waiting for database to be ready...${NC}"
timeout=60
counter=0
while ! docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U istanbul_user -d istanbul_ai; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo -e "${RED}❌ Database failed to start within $timeout seconds${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✅ Database is ready${NC}"

# Run database migrations/setup
echo -e "${YELLOW}🔧 Running database setup...${NC}"
docker-compose -f docker-compose.prod.yml exec -T postgres psql -U istanbul_user -d istanbul_ai -c "SELECT version();"

# Start backend
echo -e "${YELLOW}🖥️ Starting backend service...${NC}"
docker-compose -f docker-compose.prod.yml up -d backend

# Wait for backend to be ready
echo -e "${YELLOW}⏳ Waiting for backend to be ready...${NC}"
timeout=120
counter=0
while ! curl -f http://localhost:8000/health >/dev/null 2>&1; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo -e "${RED}❌ Backend failed to start within $timeout seconds${NC}"
        docker-compose -f docker-compose.prod.yml logs backend
        exit 1
    fi
done
echo -e "${GREEN}✅ Backend is ready${NC}"

# Build and start frontend
echo -e "${YELLOW}🎨 Starting frontend service...${NC}"
docker-compose -f docker-compose.prod.yml up -d frontend

# Start Nginx
echo -e "${YELLOW}🔗 Starting Nginx reverse proxy...${NC}"
docker-compose -f docker-compose.prod.yml up -d nginx

# Wait for Nginx to be ready
echo -e "${YELLOW}⏳ Waiting for Nginx to be ready...${NC}"
timeout=60
counter=0
while ! curl -f http://localhost/health >/dev/null 2>&1; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo -e "${RED}❌ Nginx failed to start within $timeout seconds${NC}"
        docker-compose -f docker-compose.prod.yml logs nginx
        exit 1
    fi
done
echo -e "${GREEN}✅ Nginx is ready${NC}"

# SSL Certificate setup (if domain is configured)
if [ "$DOMAIN_NAME" != "localhost" ] && [ -n "$DOMAIN_NAME" ]; then
    echo -e "${YELLOW}🔒 Setting up SSL certificate for $DOMAIN_NAME...${NC}"
    
    # Check if certificate already exists
    if [ ! -f "nginx/ssl/fullchain.pem" ]; then
        echo -e "${YELLOW}📜 Obtaining SSL certificate...${NC}"
        docker-compose -f docker-compose.prod.yml run --rm certbot || {
            echo -e "${YELLOW}⚠️ SSL certificate creation failed, continuing with HTTP${NC}"
        }
    else
        echo -e "${GREEN}✅ SSL certificate already exists${NC}"
    fi
fi

# Final health check
echo -e "${YELLOW}🏥 Running final health checks...${NC}"

# Check all services are running
SERVICES=("postgres" "redis" "backend" "frontend" "nginx")
for service in "${SERVICES[@]}"; do
    if docker-compose -f docker-compose.prod.yml ps -q $service | xargs docker inspect -f '{{.State.Running}}' | grep -q true; then
        echo -e "${GREEN}✅ $service is running${NC}"
    else
        echo -e "${RED}❌ $service is not running${NC}"
        exit 1
    fi
done

# Test critical endpoints
echo -e "${YELLOW}🧪 Testing critical endpoints...${NC}"

test_endpoint() {
    local url=$1
    local description=$2
    
    if curl -f -s "$url" >/dev/null; then
        echo -e "${GREEN}✅ $description: $url${NC}"
    else
        echo -e "${RED}❌ $description failed: $url${NC}"
        return 1
    fi
}

test_endpoint "http://localhost/health" "Health check"
test_endpoint "http://localhost/" "Frontend"

# Performance optimization
echo -e "${YELLOW}⚡ Running performance optimizations...${NC}"

# Warm up caches
echo -e "${YELLOW}🔥 Warming up application caches...${NC}"
curl -s "http://localhost/" >/dev/null || true
curl -s "http://localhost/health" >/dev/null || true

# Show deployment status
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo "================================================"
echo -e "${BLUE}📊 Deployment Summary:${NC}"
echo "- Frontend: http://localhost/"
echo "- API Health: http://localhost/health"
echo "- Nginx Status: http://localhost/nginx_status (internal)"
echo "- Database: PostgreSQL on port 5432"
echo "- Cache: Redis on port 6379"

if [ "$DOMAIN_NAME" != "localhost" ]; then
    echo "- Domain: https://$DOMAIN_NAME"
fi

echo ""
echo -e "${BLUE}🔧 Management Commands:${NC}"
echo "- View logs: docker-compose -f docker-compose.prod.yml logs [service]"
echo "- Stop services: docker-compose -f docker-compose.prod.yml down"
echo "- Update services: docker-compose -f docker-compose.prod.yml pull && docker-compose -f docker-compose.prod.yml up -d"
echo "- Monitor: docker-compose -f docker-compose.prod.yml ps"

echo ""
echo -e "${GREEN}🎉 AI Istanbul is now running in production mode!${NC}"

# Optional: Start monitoring services
read -p "Start monitoring services (Prometheus)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}📊 Starting monitoring services...${NC}"
    docker-compose -f docker-compose.prod.yml --profile monitoring up -d
    echo -e "${GREEN}✅ Monitoring available at http://localhost:9090${NC}"
fi
