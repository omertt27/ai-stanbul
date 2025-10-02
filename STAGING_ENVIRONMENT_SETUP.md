# 🏗️ AI Istanbul Staging Environment Setup Guide

## 📋 **Overview**

This guide sets up a complete staging environment for the AI Istanbul system that mirrors production while allowing safe testing and validation.

---

## 🌐 **Staging Environment Architecture**

```yaml
# Staging Environment Components:
Frontend: React + Vite (staging.aistanbul.local)
Backend: FastAPI + PostgreSQL (api-staging.aistanbul.local)  
Database: PostgreSQL (staging database)
Cache: Redis (staging instance)
Monitoring: Prometheus + Grafana (staging metrics)
Load Balancer: Nginx (staging configuration)
```

---

## 🔧 **1. Environment Configuration**

### **Staging Environment Variables**
```bash
# Create staging environment file
cp .env.example .env.staging

# Staging-specific configuration
ENVIRONMENT=staging
DEBUG=false
STAGING=true

# Database
DATABASE_URL=postgresql://istanbul_user:secure_pass@localhost:5432/istanbul_ai_staging
REDIS_URL=redis://localhost:6380/1

# API Keys (staging/sandbox versions)
OPENAI_API_KEY=sk-staging-...
GOOGLE_PLACES_API_KEY=AIza...staging
GOOGLE_WEATHER_API_KEY=AIza...staging

# Staging domain configuration
FRONTEND_URL=https://staging.aistanbul.local
BACKEND_URL=https://api-staging.aistanbul.local
CORS_ORIGINS=https://staging.aistanbul.local

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO
ANALYTICS_ENABLED=true

# Security (staging certificates)
SSL_CERT_PATH=/etc/ssl/staging/cert.pem
SSL_KEY_PATH=/etc/ssl/staging/key.pem
```

---

## 🐳 **2. Docker Staging Setup**

### **Docker Compose - Staging**
```yaml
# docker-compose.staging.yml
version: '3.8'
services:
  # PostgreSQL Database (Staging)
  postgres-staging:
    image: postgres:15
    container_name: istanbul-postgres-staging
    environment:
      POSTGRES_DB: istanbul_ai_staging
      POSTGRES_USER: istanbul_user
      POSTGRES_PASSWORD: secure_staging_password
    ports:
      - "5433:5432"
    volumes:
      - postgres_staging_data:/var/lib/postgresql/data
      - ./backend/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - istanbul-staging-network

  # Redis Cache (Staging)
  redis-staging:
    image: redis:7-alpine
    container_name: istanbul-redis-staging
    ports:
      - "6380:6379"
    volumes:
      - redis_staging_data:/data
    networks:
      - istanbul-staging-network

  # Backend API (Staging)
  backend-staging:
    build:
      context: ./backend
      dockerfile: Dockerfile.staging
    container_name: istanbul-backend-staging
    ports:
      - "8001:8000"
    environment:
      - DATABASE_URL=postgresql://istanbul_user:secure_staging_password@postgres-staging:5432/istanbul_ai_staging
      - REDIS_URL=redis://redis-staging:6379/1
      - ENVIRONMENT=staging
    depends_on:
      - postgres-staging
      - redis-staging
    volumes:
      - ./backend:/app
    networks:
      - istanbul-staging-network

  # Frontend (Staging)
  frontend-staging:
    build:
      context: ./frontend
      dockerfile: Dockerfile.staging
    container_name: istanbul-frontend-staging
    ports:
      - "3001:3000"
    environment:
      - VITE_API_URL=http://localhost:8001
      - VITE_ENVIRONMENT=staging
    depends_on:
      - backend-staging
    networks:
      - istanbul-staging-network

  # Nginx Load Balancer (Staging)
  nginx-staging:
    image: nginx:alpine
    container_name: istanbul-nginx-staging
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/staging.conf:/etc/nginx/nginx.conf
      - ./ssl/staging:/etc/ssl/staging
    depends_on:
      - frontend-staging
      - backend-staging
    networks:
      - istanbul-staging-network

  # Monitoring (Staging)
  prometheus-staging:
    image: prom/prometheus
    container_name: istanbul-prometheus-staging
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.staging.yml:/etc/prometheus/prometheus.yml
    networks:
      - istanbul-staging-network

volumes:
  postgres_staging_data:
  redis_staging_data:

networks:
  istanbul-staging-network:
    driver: bridge
```

---

## 🔧 **3. Backend Staging Configuration**

### **Dockerfile.staging**
```dockerfile
# backend/Dockerfile.staging
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set staging environment
ENV ENVIRONMENT=staging
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

---

## 🎯 **4. Frontend Staging Configuration**

### **Dockerfile.staging**
```dockerfile
# frontend/Dockerfile.staging
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build for staging
ENV VITE_ENVIRONMENT=staging
ENV VITE_API_URL=https://api-staging.aistanbul.local
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.staging.conf /etc/nginx/nginx.conf

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
```

---

## 🌐 **5. Nginx Staging Configuration**

```nginx
# nginx/staging.conf
events {
    worker_connections 1024;
}

http {
    upstream backend-staging {
        server backend-staging:8000;
    }

    upstream frontend-staging {
        server frontend-staging:3000;
    }

    # Staging Frontend
    server {
        listen 80;
        server_name staging.aistanbul.local;
        
        location / {
            proxy_pass http://frontend-staging;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

    # Staging API
    server {
        listen 80;
        server_name api-staging.aistanbul.local;
        
        location / {
            proxy_pass http://backend-staging;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # CORS headers for staging
            add_header Access-Control-Allow-Origin "https://staging.aistanbul.local" always;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
            add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
        }
    }
}
```

---

## 📊 **6. Database Migration for Staging**

```python
# scripts/setup_staging_db.py
import os
import subprocess
from sqlalchemy import create_engine
from backend.database import Base
from backend.models import *

def setup_staging_database():
    """Set up staging database with test data"""
    
    # Database connection
    staging_db_url = "postgresql://istanbul_user:secure_staging_password@localhost:5433/istanbul_ai_staging"
    engine = create_engine(staging_db_url)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("✅ Staging database tables created")
    
    # Load staging data
    load_staging_data(engine)
    print("✅ Staging data loaded")

def load_staging_data(engine):
    """Load staging-specific test data"""
    from sqlalchemy.orm import sessionmaker
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Load places data
        staging_places = [
            {"name": "Staging Test Palace", "district": "Test District", "category": "museum"},
            {"name": "Staging Test Restaurant", "district": "Test District", "category": "restaurant"},
        ]
        
        for place_data in staging_places:
            place = Place(**place_data)
            session.add(place)
        
        session.commit()
        print("✅ Staging test data loaded successfully")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error loading staging data: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    setup_staging_database()
```

---

## 🚀 **7. Staging Deployment Scripts**

### **Deploy to Staging**
```bash
#!/bin/bash
# scripts/deploy_staging.sh

set -e

echo "🚀 Deploying AI Istanbul to Staging Environment..."

# Build and deploy with Docker Compose
echo "📦 Building staging containers..."
docker-compose -f docker-compose.staging.yml build

echo "🗄️ Setting up staging database..."
python scripts/setup_staging_db.py

echo "🚀 Starting staging services..."
docker-compose -f docker-compose.staging.yml up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

echo "🔍 Running health checks..."
curl -f http://localhost:8001/api/health || exit 1
curl -f http://localhost:3001 || exit 1

echo "✅ Staging deployment completed successfully!"
echo "🌐 Frontend: http://staging.aistanbul.local"
echo "🔧 API: http://api-staging.aistanbul.local"
echo "📊 Metrics: http://localhost:9091"
```

### **Staging Environment Health Check**
```bash
#!/bin/bash
# scripts/staging_health_check.sh

echo "🔍 Running Staging Environment Health Checks..."

# Check services
services=("postgres-staging:5433" "redis-staging:6380" "backend-staging:8001" "frontend-staging:3001")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if nc -z localhost $port; then
        echo "✅ $name is running on port $port"
    else
        echo "❌ $name is NOT running on port $port"
        exit 1
    fi
done

# API Health Check
if curl -f http://localhost:8001/api/health > /dev/null 2>&1; then
    echo "✅ Backend API health check passed"
else
    echo "❌ Backend API health check failed"
    exit 1
fi

# Frontend Check
if curl -f http://localhost:3001 > /dev/null 2>&1; then
    echo "✅ Frontend accessibility check passed"
else
    echo "❌ Frontend accessibility check failed"
    exit 1
fi

echo "🎉 All staging environment health checks passed!"
```

---

## 🎯 **8. Usage Instructions**

### **Setup Staging Environment**
```bash
# 1. Clone and setup
git clone <repository>
cd ai-stanbul

# 2. Create staging environment file
cp .env.example .env.staging
# Edit .env.staging with staging-specific values

# 3. Deploy to staging
chmod +x scripts/deploy_staging.sh
./scripts/deploy_staging.sh

# 4. Add local DNS entries (for local testing)
echo "127.0.0.1 staging.aistanbul.local" | sudo tee -a /etc/hosts
echo "127.0.0.1 api-staging.aistanbul.local" | sudo tee -a /etc/hosts
```

### **Staging Environment Management**
```bash
# Start staging environment
docker-compose -f docker-compose.staging.yml up -d

# Stop staging environment  
docker-compose -f docker-compose.staging.yml down

# View logs
docker-compose -f docker-compose.staging.yml logs -f

# Run health checks
./scripts/staging_health_check.sh

# Reset staging data
docker-compose -f docker-compose.staging.yml down -v
./scripts/deploy_staging.sh
```

---

## 📊 **9. Monitoring & Observability**

### **Staging Metrics Dashboard**
- **Service Health:** All containers status and health
- **API Performance:** Response times, error rates, throughput
- **Database Metrics:** Connection pool, query performance
- **Cache Performance:** Hit rates, memory usage
- **User Journey:** End-to-end transaction tracking

### **Staging Alerts**
- Service downtime detection
- High error rates (>5%)
- Slow response times (>3s)
- Database connection issues
- High memory/CPU usage

---

## ✅ **10. Staging Environment Checklist**

- [ ] Docker containers built and running
- [ ] Database schema migrated
- [ ] Test data loaded
- [ ] API endpoints responding
- [ ] Frontend accessible
- [ ] CORS configured correctly
- [ ] SSL/TLS certificates (if using HTTPS)
- [ ] Monitoring dashboards active  
- [ ] Health checks passing
- [ ] Log aggregation working
- [ ] Backup procedures tested

---

**🎯 Staging Environment Status: READY FOR DEPLOYMENT**

This staging environment provides a complete, isolated replica of production for safe testing and validation of all changes before they go live.
