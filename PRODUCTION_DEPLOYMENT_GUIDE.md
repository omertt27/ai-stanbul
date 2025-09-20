# Production Deployment Guide - AI Istanbul

This guide covers deploying AI Istanbul with Nginx reverse proxy for optimal production performance, security, and reliability.

## Overview

The production deployment includes:
- **Nginx** as reverse proxy with SSL termination, caching, and rate limiting
- **Backend API** running with Gunicorn for high performance
- **Frontend** optimized build served by Nginx
- **PostgreSQL** database with connection pooling
- **Redis** for caching and rate limiting
- **SSL/TLS** with automatic certificate management
- **Monitoring** with health checks and logging
- **Security** headers and GDPR compliance

## Architecture

```
Internet → Nginx (Port 80/443) → Backend API (Port 8000)
                ↓
            Frontend Build → Static Files
                ↓
            PostgreSQL Database
                ↓
            Redis Cache
```

## Prerequisites

1. **Docker & Docker Compose** installed
2. **Domain name** configured (for SSL)
3. **Environment variables** configured
4. **Minimum 2GB RAM** and 10GB disk space

## Quick Start

### 1. Environment Setup

Copy and configure the production environment:

```bash
cp .env.prod.template .env.prod
```

Edit `.env.prod` with your actual values:

```bash
# Required API Keys
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_PLACES_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Security
POSTGRES_PASSWORD=MyS3cur3P@ssw0rd!2024
SECRET_KEY=my-super-secret-key-that-is-at-least-32-characters-long

# Domain Configuration
DOMAIN_NAME=aistanbul.app
ADMIN_EMAIL=admin@aistanbul.app
CORS_ORIGINS=https://aistanbul.app,https://www.aistanbul.app
```

### 2. Deploy to Production

Run the automated deployment script:

```bash
./deploy-production.sh
```

This script will:
- ✅ Validate all requirements and environment variables
- 🏗️ Build optimized Docker images
- 🗄️ Start database services and run migrations
- 🖥️ Deploy backend with Gunicorn
- 🎨 Build and deploy frontend
- 🔗 Configure Nginx reverse proxy
- 🔒 Set up SSL certificates (if domain configured)
- 🏥 Run comprehensive health checks

### 3. Verify Deployment

Check that all services are running:

```bash
docker-compose -f docker-compose.prod.yml ps
```

Test critical endpoints:
- Frontend: `https://yourdomain.com/`
- Health Check: `https://yourdomain.com/health`
- API: `https://yourdomain.com/api/`

## Configuration Details

### Nginx Reverse Proxy

**Features:**
- SSL termination with HTTP/2
- Static asset caching (1 year for assets, 1 hour for HTML)
- API response caching (5 minutes for cacheable responses)
- Rate limiting (10 req/s for API, 5 req/m for AI)
- Security headers (HSTS, XSS protection, CSP)
- Gzip compression
- Load balancing ready

**Key Locations:**
- `/` → Frontend static files
- `/api/` → Backend API (prefix removed)
- `/ai` → AI endpoint (no caching, extended timeout)
- `/health` → Health check endpoint

### Backend API

**Production Optimizations:**
- Gunicorn WSGI server with 4 workers
- Uvicorn async workers for FastAPI
- Connection pooling for database
- Redis caching with fallback
- Structured logging
- Health check endpoint
- GDPR compliance endpoints

### Database

**PostgreSQL Configuration:**
- Performance indexes applied
- Connection pooling
- Automated backups
- Health checks
- Data retention policies

### Caching

**Redis Configuration:**
- Memory-optimized (256MB limit)
- LRU eviction policy
- Persistence enabled
- Health monitoring

## SSL/TLS Setup

### Automatic (Let's Encrypt)

For domains, SSL is automatically configured:

```bash
# Included in deploy-production.sh
docker-compose -f docker-compose.prod.yml run --rm certbot
```

### Manual Certificate

Place your certificates in:
- `nginx/ssl/fullchain.pem`
- `nginx/ssl/privkey.pem`

## Monitoring & Maintenance

### Health Monitoring

Access health status:
```bash
curl https://yourdomain.com/health
```

Response includes:
- Database connectivity
- Redis status
- Service availability
- System metrics

### Logs

View application logs:
```bash
# All services
docker-compose -f docker-compose.prod.yml logs

# Specific service
docker-compose -f docker-compose.prod.yml logs backend
docker-compose -f docker-compose.prod.yml logs nginx
```

### Monitoring Dashboard (Optional)

Enable Prometheus monitoring:
```bash
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

Access at: `http://yourdomain.com:9090`

## Performance Optimizations

### Nginx Caching
- Static assets: 1 year cache
- API responses: 5 minutes cache
- HTML files: 1 hour cache

### Database
- Connection pooling
- Query optimization with indexes
- Read replicas support ready

### Frontend
- Minified and compressed assets
- Service worker caching
- CDN-ready configuration

### Backend
- Async request handling
- Response caching
- Rate limiting
- Connection reuse

## Security Features

### Network Security
- HTTPS enforcement
- Security headers (HSTS, CSP, XSS)
- Rate limiting
- CORS configuration

### Application Security
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- GDPR compliance

### Infrastructure Security
- Non-root container users
- Resource limits
- Network isolation
- Secret management

## Scaling & High Availability

### Horizontal Scaling
```bash
# Scale backend workers
docker-compose -f docker-compose.prod.yml up -d --scale backend=3

# Add load balancer
# Update nginx upstream configuration
```

### Database Scaling
- Read replicas support
- Connection pooling
- Query optimization

### Cache Scaling
- Redis cluster support
- Multi-level caching
- CDN integration

## Backup & Recovery

### Database Backups
```bash
# Manual backup
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U istanbul_user istanbul_ai > backup.sql

# Restore
docker-compose -f docker-compose.prod.yml exec -T postgres psql -U istanbul_user istanbul_ai < backup.sql
```

### Application Data
- Configuration backups
- Log retention
- SSL certificate backup

## Troubleshooting

### Common Issues

**503 Service Unavailable**
```bash
# Check backend health
docker-compose -f docker-compose.prod.yml logs backend

# Restart backend
docker-compose -f docker-compose.prod.yml restart backend
```

**SSL Certificate Issues**
```bash
# Renew certificates
docker-compose -f docker-compose.prod.yml run --rm certbot renew

# Check certificate expiry
openssl x509 -in nginx/ssl/fullchain.pem -text -noout | grep "Not After"
```

**Database Connection Issues**
```bash
# Check database status
docker-compose -f docker-compose.prod.yml exec postgres pg_isready

# View database logs
docker-compose -f docker-compose.prod.yml logs postgres
```

**High Memory Usage**
```bash
# Check container resource usage
docker stats

# Restart specific service
docker-compose -f docker-compose.prod.yml restart [service-name]
```

### Performance Issues

**Slow Response Times**
- Check Redis cache hit rates
- Review database query performance
- Monitor Nginx access logs
- Verify rate limiting settings

**High CPU Usage**
- Scale backend workers
- Optimize database queries
- Review caching strategy

## Maintenance Tasks

### Regular Tasks
- Monitor SSL certificate expiry (auto-renewed)
- Review application logs
- Database maintenance and optimization
- Security updates for base images

### Weekly Tasks
- Review performance metrics
- Check backup integrity
- Update dependencies (if needed)

### Monthly Tasks
- Security audit
- Performance optimization review
- Capacity planning

## Cost Optimization

### Resource Efficiency
- Right-sized containers
- Efficient caching strategy
- Database query optimization
- Image optimization

### Monitoring Costs
- Resource usage tracking
- Performance metrics
- Capacity planning

## Support

For issues or questions:
1. Check logs with `docker-compose -f docker-compose.prod.yml logs`
2. Review health endpoint: `/health`
3. Check this documentation
4. Review application metrics

## Version History

- **v1.0.0** - Initial production deployment
  - Nginx reverse proxy
  - SSL/TLS support
  - Performance optimizations
  - Monitoring and health checks
  - GDPR compliance
  - Security enhancements
