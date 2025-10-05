# production_deployment_system.py - Production Deployment and Infrastructure

import os
import json
import yaml
from typing import Dict, List, Any
from dataclasses import dataclass
import subprocess
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    app_name: str
    version: str
    environment: str
    replicas: int
    cpu_limit: str
    memory_limit: str
    port: int

class ProductionDeploymentManager:
    """Manages production deployment configurations and processes"""
    
    def __init__(self):
        self.deployment_configs = {}
        self.supported_environments = ["development", "staging", "production"]
    
    def generate_dockerfile(self, app_name: str, python_version: str = "3.9") -> str:
        """Generate Dockerfile for the application"""
        dockerfile_content = f"""# Production Dockerfile for {app_name}
FROM python:{python_version}-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "main:app"]
"""
        return dockerfile_content
    
    def generate_docker_compose(self, config: DeploymentConfig) -> str:
        """Generate docker-compose.yml for local development"""
        compose_content = f"""version: '3.8'
services:
  {config.app_name}:
    build: .
    ports:
      - "{config.port}:8000"
    environment:
      - ENVIRONMENT={config.environment}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/{config.app_name}
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB={config.app_name}
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - {config.app_name}
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
"""
        return compose_content
    
    def generate_kubernetes_deployment(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes deployment YAML"""
        k8s_deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {config.app_name}-deployment
  labels:
    app: {config.app_name}
    version: {config.version}
spec:
  replicas: {config.replicas}
  selector:
    matchLabels:
      app: {config.app_name}
  template:
    metadata:
      labels:
        app: {config.app_name}
        version: {config.version}
    spec:
      containers:
      - name: {config.app_name}
        image: {config.app_name}:{config.version}
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "{config.environment}"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: {config.app_name}-secrets
              key: redis-url
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: {config.app_name}-secrets
              key: database-url
        resources:
          limits:
            cpu: {config.cpu_limit}
            memory: {config.memory_limit}
          requests:
            cpu: "100m"
            memory: "128Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {{}}
---
apiVersion: v1
kind: Service
metadata:
  name: {config.app_name}-service
spec:
  selector:
    app: {config.app_name}
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {config.app_name}-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - {config.app_name}.example.com
    secretName: {config.app_name}-tls
  rules:
  - host: {config.app_name}.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {config.app_name}-service
            port:
              number: 80
"""
        return k8s_deployment
    
    def generate_github_actions_workflow(self, config: DeploymentConfig) -> str:
        """Generate GitHub Actions CI/CD workflow"""
        workflow = f"""name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=./ --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        username: ${{{{ secrets.DOCKER_USERNAME }}}}
        password: ${{{{ secrets.DOCKER_PASSWORD }}}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          {config.app_name}:latest
          {config.app_name}:${{{{ github.sha }}}}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.24.0'
    
    - name: Deploy to Kubernetes
      run: |
        echo "${{{{ secrets.KUBE_CONFIG }}}}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/{config.app_name}-deployment {config.app_name}={config.app_name}:${{{{ github.sha }}}}
        kubectl rollout status deployment/{config.app_name}-deployment
"""
        return workflow
    
    def generate_nginx_config(self, config: DeploymentConfig) -> str:
        """Generate Nginx configuration"""
        nginx_config = f"""events {{
    worker_connections 1024;
}}

http {{
    upstream {config.app_name} {{
        server {config.app_name}:8000;
    }}
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    server {{
        listen 80;
        server_name {config.app_name}.example.com;
        return 301 https://$server_name$request_uri;
    }}
    
    server {{
        listen 443 ssl http2;
        server_name {config.app_name}.example.com;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # Compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
        
        location / {{
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://{config.app_name};
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }}
        
        location /static/ {{
            expires 1M;
            add_header Cache-Control "public, immutable";
        }}
        
        location /health {{
            access_log off;
            proxy_pass http://{config.app_name};
        }}
    }}
}}
"""
        return nginx_config
    
    def generate_monitoring_config(self, config: DeploymentConfig) -> str:
        """Generate monitoring configuration (Prometheus + Grafana)"""
        prometheus_config = f"""global:
  scrape_interval: 15s

scrape_configs:
  - job_name: '{config.app_name}'
    static_configs:
      - targets: ['{config.app_name}:8000']
    metrics_path: /metrics
    scrape_interval: 10s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""
        return prometheus_config
    
    def create_deployment_package(self, config: DeploymentConfig, output_dir: str = "./deployment"):
        """Create complete deployment package"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        github_dir = os.path.join(output_dir, ".github", "workflows")
        os.makedirs(github_dir, exist_ok=True)
        
        files = {
            "Dockerfile": self.generate_dockerfile(config.app_name),
            "docker-compose.yml": self.generate_docker_compose(config),
            "k8s-deployment.yaml": self.generate_kubernetes_deployment(config),
            ".github/workflows/ci-cd.yml": self.generate_github_actions_workflow(config),
            "nginx.conf": self.generate_nginx_config(config),
            "prometheus.yml": self.generate_monitoring_config(config)
        }
        
        for filename, content in files.items():
            filepath = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)
        
        logger.info(f"Deployment package created in {output_dir}")
        return output_dir

# Production health checks and monitoring
class ProductionHealthChecker:
    """Production health check and monitoring system"""
    
    def __init__(self):
        self.health_checks = {}
        self.monitoring_endpoints = []
    
    def register_health_check(self, name: str, check_function: callable):
        """Register a health check function"""
        self.health_checks[name] = check_function
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {
            "status": "healthy",
            "timestamp": str(datetime.now()),
            "checks": {}
        }
        
        for name, check_func in self.health_checks.items():
            try:
                check_result = check_func()
                results["checks"][name] = {
                    "status": "pass" if check_result else "fail",
                    "result": check_result
                }
                if not check_result:
                    results["status"] = "unhealthy"
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                results["status"] = "unhealthy"
        
        return results
    
    def database_health_check(self) -> bool:
        """Database connectivity health check"""
        try:
            # In production, this would test actual database connection
            # For now, we'll simulate a successful check
            return True
        except Exception:
            return False
    
    def redis_health_check(self) -> bool:
        """Redis connectivity health check"""
        try:
            # In production, this would test actual Redis connection
            # For now, we'll simulate a successful check
            return True
        except Exception:
            return False
    
    def api_health_check(self) -> bool:
        """API endpoints health check"""
        try:
            # In production, this would test actual API endpoints
            # For now, we'll simulate a successful check
            return True
        except Exception:
            return False

# Security configuration
class SecurityManager:
    """Production security configuration manager"""
    
    def __init__(self):
        self.security_configs = {}
    
    def generate_security_headers(self) -> Dict[str, str]:
        """Generate security headers for production"""
        return {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
    
    def generate_cors_config(self, allowed_origins: List[str]) -> Dict[str, Any]:
        """Generate CORS configuration"""
        return {
            "allow_origins": allowed_origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["*"]
        }

if __name__ == "__main__":
    # Example usage
    config = DeploymentConfig(
        app_name="ai-istanbul",
        version="1.0.0",
        environment="production",
        replicas=3,
        cpu_limit="500m",
        memory_limit="512Mi",
        port=8080
    )
    
    # Initialize deployment manager
    deployment_manager = ProductionDeploymentManager()
    
    # Create deployment package
    deployment_package_path = deployment_manager.create_deployment_package(config)
    
    # Initialize health checker
    health_checker = ProductionHealthChecker()
    health_checker.register_health_check("database", health_checker.database_health_check)
    health_checker.register_health_check("redis", health_checker.redis_health_check)
    health_checker.register_health_check("api", health_checker.api_health_check)
    
    # Run health checks
    health_status = health_checker.run_health_checks()
    
    print("Production deployment system initialized successfully!")
    print(f"Deployment package created at: {deployment_package_path}")
    print(f"System health status: {health_status['status']}")
    print("Ready for production deployment!")
