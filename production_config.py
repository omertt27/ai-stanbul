#!/usr/bin/env python3
"""
Production Deployment Configuration for Advanced Istanbul AI
Complete production-ready setup with monitoring, scaling, and optimization
"""

import os
import json
import yaml
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ProductionConfig:
    """Production deployment configuration"""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    max_requests: int = 1000
    timeout: int = 30
    
    # Database Configuration
    database_url: str = "postgresql://istanbul_ai:password@localhost:5432/istanbul_ai"
    redis_url: str = "redis://localhost:6379/0"
    
    # AI Model Configuration  
    model_cache_size: int = 100
    embeddings_path: str = "./models/istanbul_embeddings.pkl"
    neural_model_path: str = "./models/intent_classifier.pt"
    
    # API Keys (set via environment variables)
    openweather_api_key: str = os.getenv("OPENWEATHER_API_KEY", "")
    google_maps_api_key: str = os.getenv("GOOGLE_MAPS_API_KEY", "")
    istanbul_transport_api_key: str = os.getenv("ISTANBUL_TRANSPORT_API_KEY", "")
    
    # Monitoring & Logging
    log_level: str = "INFO"
    enable_metrics: bool = True
    sentry_dsn: str = os.getenv("SENTRY_DSN", "")
    
    # Performance
    enable_caching: bool = True
    cache_ttl: int = 300
    rate_limit: int = 100  # requests per minute per user
    
    # Security
    enable_cors: bool = True
    allowed_origins: list = None
    jwt_secret: str = os.getenv("JWT_SECRET", "your-secret-key")
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:3000", "https://istanbul-ai.com"]

def generate_docker_compose():
    """Generate Docker Compose configuration"""
    
    docker_compose = {
        'version': '3.8',
        'services': {
            'istanbul-ai': {
                'build': {
                    'context': '.',
                    'dockerfile': 'Dockerfile'
                },
                'ports': ['8080:8080'],
                'environment': [
                    'DATABASE_URL=postgresql://istanbul_ai:password@postgres:5432/istanbul_ai',
                    'REDIS_URL=redis://redis:6379/0',
                    'OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}',
                    'GOOGLE_MAPS_API_KEY=${GOOGLE_MAPS_API_KEY}',
                    'ISTANBUL_TRANSPORT_API_KEY=${ISTANBUL_TRANSPORT_API_KEY}',
                    'SENTRY_DSN=${SENTRY_DSN}',
                    'JWT_SECRET=${JWT_SECRET}'
                ],
                'depends_on': ['postgres', 'redis'],
                'restart': 'unless-stopped',
                'volumes': [
                    './models:/app/models:ro',
                    './logs:/app/logs'
                ],
                'healthcheck': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '40s'
                }
            },
            'postgres': {
                'image': 'postgres:15-alpine',
                'environment': [
                    'POSTGRES_DB=istanbul_ai',
                    'POSTGRES_USER=istanbul_ai',
                    'POSTGRES_PASSWORD=password'
                ],
                'volumes': [
                    'postgres_data:/var/lib/postgresql/data',
                    './sql/init.sql:/docker-entrypoint-initdb.d/init.sql'
                ],
                'restart': 'unless-stopped',
                'healthcheck': {
                    'test': ['CMD-SHELL', 'pg_isready -U istanbul_ai'],
                    'interval': '10s',
                    'timeout': '5s',
                    'retries': 5
                }
            },
            'redis': {
                'image': 'redis:7-alpine',
                'restart': 'unless-stopped',
                'volumes': ['redis_data:/data'],
                'healthcheck': {
                    'test': ['CMD', 'redis-cli', 'ping'],
                    'interval': '10s',
                    'timeout': '5s',
                    'retries': 5
                }
            },
            'nginx': {
                'image': 'nginx:alpine',
                'ports': ['80:80', '443:443'],
                'volumes': [
                    './nginx/nginx.conf:/etc/nginx/nginx.conf:ro',
                    './nginx/ssl:/etc/nginx/ssl:ro'
                ],
                'depends_on': ['istanbul-ai'],
                'restart': 'unless-stopped'
            },
            'prometheus': {
                'image': 'prom/prometheus:latest',
                'ports': ['9090:9090'],
                'volumes': ['./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro'],
                'restart': 'unless-stopped'
            },
            'grafana': {
                'image': 'grafana/grafana:latest',
                'ports': ['3000:3000'],
                'environment': [
                    'GF_SECURITY_ADMIN_PASSWORD=admin123'
                ],
                'volumes': [
                    'grafana_data:/var/lib/grafana',
                    './monitoring/grafana-dashboards:/etc/grafana/provisioning/dashboards',
                    './monitoring/grafana-datasources:/etc/grafana/provisioning/datasources'
                ],
                'restart': 'unless-stopped'
            }
        },
        'volumes': {
            'postgres_data': {},
            'redis_data': {},
            'grafana_data': {}
        },
        'networks': {
            'default': {
                'driver': 'bridge'
            }
        }
    }
    
    return docker_compose

def generate_dockerfile():
    """Generate optimized Dockerfile"""
    
    dockerfile = """# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libpq5 \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && useradd --create-home --shell /bin/bash istanbul

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=istanbul:istanbul . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models && \\
    chown -R istanbul:istanbul /app

# Switch to non-root user
USER istanbul

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
"""
    
    return dockerfile

def generate_requirements():
    """Generate comprehensive requirements.txt"""
    
    requirements = """# Core AI and ML
fastapi==0.104.1
uvicorn[standard]==0.24.0
gunicorn==21.2.0
pydantic==2.5.0
numpy==1.24.3
scikit-learn==1.3.0
torch==2.1.0
transformers==4.35.0
sentence-transformers==2.2.2

# Database and Caching
asyncpg==0.29.0
sqlalchemy==2.0.23
alembic==1.12.1
redis==5.0.1
psycopg2-binary==2.9.9

# Web and API
aiohttp==3.9.0
requests==2.31.0
httpx==0.25.0
websockets==12.0

# Monitoring and Logging
prometheus-client==0.19.0
sentry-sdk[fastapi]==1.38.0
structlog==23.2.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Data Processing
pandas==2.1.3
geopy==2.4.0
python-dateutil==2.8.2

# Testing and Development
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
flake8==6.1.0

# Production Utilities
python-dotenv==1.0.0
click==8.1.7
rich==13.7.0
typer==0.9.0
"""
    
    return requirements

def generate_nginx_config():
    """Generate Nginx configuration"""
    
    nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream istanbul_ai {
        least_conn;
        server istanbul-ai:8080 max_fails=3 fail_timeout=30s;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml;
    
    server {
        listen 80;
        server_name localhost istanbul-ai.com;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://istanbul_ai;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # Enable WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        # Health check
        location /health {
            proxy_pass http://istanbul_ai/health;
            access_log off;
        }
        
        # Metrics (restricted access)
        location /metrics {
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            deny all;
            proxy_pass http://istanbul_ai/metrics;
        }
        
        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # Frontend (if serving SPA)
        location / {
            try_files $uri $uri/ /index.html;
            root /usr/share/nginx/html;
        }
    }
}
"""
    
    return nginx_config

def generate_monitoring_config():
    """Generate Prometheus monitoring configuration"""
    
    prometheus_config = {
        'global': {
            'scrape_interval': '15s',
            'evaluation_interval': '15s'
        },
        'scrape_configs': [
            {
                'job_name': 'istanbul-ai',
                'static_configs': [
                    {
                        'targets': ['istanbul-ai:8080']
                    }
                ],
                'metrics_path': '/metrics',
                'scrape_interval': '10s'
            },
            {
                'job_name': 'prometheus',
                'static_configs': [
                    {
                        'targets': ['localhost:9090']
                    }
                ]
            }
        ],
        'rule_files': ['alert_rules.yml']
    }
    
    return prometheus_config

def generate_database_schema():
    """Generate database initialization SQL"""
    
    schema_sql = """-- Istanbul AI Database Schema
-- Production-ready with indexes and constraints

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- User profiles table
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    profile_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Conversation history table
CREATE TABLE conversation_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    turn_number INTEGER NOT NULL,
    user_input TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    intents JSONB,
    entities JSONB,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analytics and metrics
CREATE TABLE interaction_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    intent_type VARCHAR(100),
    response_time_ms INTEGER,
    user_satisfaction FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System performance metrics
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    tags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_conversation_history_session ON conversation_history(session_id);
CREATE INDEX idx_conversation_history_user ON conversation_history(user_id);
CREATE INDEX idx_conversation_history_created ON conversation_history(created_at);
CREATE INDEX idx_interaction_metrics_user ON interaction_metrics(user_id);
CREATE INDEX idx_interaction_metrics_created ON interaction_metrics(created_at);
CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_created ON system_metrics(created_at);

-- GIN indexes for JSONB columns
CREATE INDEX idx_user_profiles_data ON user_profiles USING GIN(profile_data);
CREATE INDEX idx_conversation_intents ON conversation_history USING GIN(intents);
CREATE INDEX idx_conversation_entities ON conversation_history USING GIN(entities);

-- Update trigger for user_profiles
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE
    ON user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for analytics
CREATE VIEW user_interaction_summary AS
SELECT 
    user_id,
    COUNT(*) as total_interactions,
    AVG(response_time_ms) as avg_response_time,
    AVG(user_satisfaction) as avg_satisfaction,
    MIN(created_at) as first_interaction,
    MAX(created_at) as last_interaction
FROM interaction_metrics
GROUP BY user_id;

CREATE VIEW popular_intents AS
SELECT 
    intent_type,
    COUNT(*) as frequency,
    AVG(response_time_ms) as avg_response_time
FROM interaction_metrics
WHERE intent_type IS NOT NULL
GROUP BY intent_type
ORDER BY frequency DESC;
"""
    
    return schema_sql

def generate_production_files():
    """Generate all production configuration files"""
    
    files_to_create = {
        'docker-compose.yml': yaml.dump(generate_docker_compose(), default_flow_style=False),
        'Dockerfile': generate_dockerfile(),
        'requirements.txt': generate_requirements(),
        'nginx/nginx.conf': generate_nginx_config(),
        'monitoring/prometheus.yml': yaml.dump(generate_monitoring_config(), default_flow_style=False),
        'sql/init.sql': generate_database_schema(),
        '.env.example': """# Environment Variables for Istanbul AI
OPENWEATHER_API_KEY=your_openweather_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key  
ISTANBUL_TRANSPORT_API_KEY=your_istanbul_transport_api_key
SENTRY_DSN=your_sentry_dsn
JWT_SECRET=your_jwt_secret_key
DATABASE_URL=postgresql://istanbul_ai:password@localhost:5432/istanbul_ai
REDIS_URL=redis://localhost:6379/0
""",
        'gunicorn.conf.py': """# Gunicorn configuration
bind = "0.0.0.0:8080"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 5
preload_app = True
user = "istanbul"
group = "istanbul"
tmp_upload_dir = None
logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "access": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"],
    },
    "loggers": {
        "gunicorn.error": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
        },
        "gunicorn.access": {
            "level": "INFO",
            "handlers": ["access"],
            "propagate": False,
        },
    },
}
"""
    }
    
    return files_to_create

if __name__ == "__main__":
    print("🚀 Generating Production Configuration Files...")
    
    files = generate_production_files()
    
    for filepath, content in files.items():
        # Create directory if it doesn't exist
        dirname = os.path.dirname(filepath)
        if dirname:  # Only create directory if there is one
            os.makedirs(dirname, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ Created: {filepath}")
    
    print("\n🎯 Production Configuration Complete!")
    print("\nNext Steps:")
    print("1. Copy .env.example to .env and fill in your API keys")
    print("2. Run: docker-compose up -d")
    print("3. Access the API at http://localhost:8080")
    print("4. Monitor with Grafana at http://localhost:3000")
    print("5. Check metrics with Prometheus at http://localhost:9090")
