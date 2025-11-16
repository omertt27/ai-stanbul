# 🚀 AI Istanbul - Deployment Guide

> **Complete Staging & Production Deployment Checklist**  
> **Generated:** November 16, 2025  
> **Status:** Ready for Deployment

---

## 📊 Deployment Overview

### Current Status: ✅ **READY FOR STAGING**

All systems are integrated and tested. This guide covers:
1. **Staging Deployment** (Next Step)
2. **Production Deployment** (After staging validation)
3. **Monitoring & Maintenance**
4. **Rollback Procedures**
5. **Admin Dashboard Deployment** ⭐ NEW

---

## 🎯 Phase 1: Staging Deployment (HIGH PRIORITY)

### Estimated Time: 1-2 days

### Pre-Deployment Checklist

#### Code Readiness
- [x] All tests passing (55 tests, 100%)
- [x] Documentation complete
- [x] Environment variables documented
- [x] Database migrations ready
- [x] API endpoints tested
- [x] Error handling implemented
- [x] Logging configured
- [x] Security measures in place

#### Infrastructure Requirements
- [ ] Staging server provisioned
- [ ] PostgreSQL database setup
- [ ] Redis instance (optional, for caching)
- [ ] SSL certificate installed
- [ ] Domain/subdomain configured (staging.ai-istanbul.com)
- [ ] Firewall rules configured
- [ ] Backup system configured

#### Environment Variables
- [ ] OpenAI API key configured
- [ ] Database credentials set
- [ ] Redis URL (if using)
- [ ] CORS origins configured
- [ ] Secret keys generated
- [ ] OSRM server URL (or use public)

---

## 📝 Step-by-Step Staging Deployment

### Step 1: Server Setup (30 minutes)

```bash
# SSH into staging server
ssh user@staging.ai-istanbul.com

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.10 python3-pip postgresql redis-server nginx

# Install Node.js (for frontend)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Create application directory
sudo mkdir -p /var/www/ai-istanbul
sudo chown $USER:$USER /var/www/ai-istanbul
cd /var/www/ai-istanbul
```

### Step 2: Clone Repository (5 minutes)

```bash
# Clone the repository
git clone https://github.com/your-org/ai-istanbul.git .

# Create staging branch if needed
git checkout -b staging

# Or pull latest changes
git pull origin main
```

### Step 3: Backend Setup (20 minutes)

```bash
# Navigate to backend
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
# Database
DATABASE_URL=postgresql://ai_istanbul_user:secure_password@localhost:5432/ai_istanbul_staging

# OpenAI
OPENAI_API_KEY=sk-your-openai-key-here

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# App Config
API_HOST=0.0.0.0
API_PORT=8001
ENVIRONMENT=staging
DEBUG=false

# CORS
CORS_ORIGINS=["https://staging.ai-istanbul.com","http://localhost:3000"]

# OSRM
OSRM_SERVER=http://router.project-osrm.org

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
EOF

# Set proper permissions
chmod 600 .env
```

### Step 4: Database Setup (15 minutes)

```bash
# Create PostgreSQL user and database
sudo -u postgres psql << EOF
CREATE USER ai_istanbul_user WITH PASSWORD 'secure_password';
CREATE DATABASE ai_istanbul_staging OWNER ai_istanbul_user;
GRANT ALL PRIVILEGES ON DATABASE ai_istanbul_staging TO ai_istanbul_user;
\q
EOF

# Run database migrations
cd /var/www/ai-istanbul/backend
source venv/bin/activate
python scripts/create_db.py

# Import POI data
python scripts/import_pois.py pois_raw.json

# Verify database
python scripts/verify_db.py
```

**Expected Output:**
```
✅ Database connected
✅ Tables created (pois, users, feedback, sessions)
✅ POI data imported: 51 locations
✅ Indexes created
✅ Database ready
```

### Step 5: Frontend Setup (15 minutes)

```bash
# Navigate to frontend
cd /var/www/ai-istanbul/frontend

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
VITE_API_URL=https://staging.ai-istanbul.com
EOF

# Build production assets
npm run build

# Output will be in dist/ folder
ls -la dist/
```

### Step 6: Nginx Configuration (10 minutes)

```bash
# Create Nginx config
sudo cat > /etc/nginx/sites-available/ai-istanbul-staging << 'EOF'
# API Backend
upstream backend {
    server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name staging.ai-istanbul.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name staging.ai-istanbul.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/staging.ai-istanbul.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/staging.ai-istanbul.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Frontend (React)
    root /var/www/ai-istanbul/frontend/dist;
    index index.html;
    
    # API Proxy
    location /api/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Frontend routes (React Router)
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/ai-istanbul-staging /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

### Step 7: SSL Certificate (10 minutes)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d staging.ai-istanbul.com

# Auto-renewal is configured by default
sudo certbot renew --dry-run
```

### Step 8: Backend Service (10 minutes)

```bash
# Create systemd service
sudo cat > /etc/systemd/system/ai-istanbul-backend.service << EOF
[Unit]
Description=AI Istanbul Backend Service
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/var/www/ai-istanbul/backend
Environment="PATH=/var/www/ai-istanbul/backend/venv/bin"
ExecStart=/var/www/ai-istanbul/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/var/log/ai-istanbul/app.log
StandardError=append:/var/log/ai-istanbul/error.log

[Install]
WantedBy=multi-user.target
EOF

# Create log directory
sudo mkdir -p /var/log/ai-istanbul
sudo chown $USER:$USER /var/log/ai-istanbul

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ai-istanbul-backend
sudo systemctl start ai-istanbul-backend

# Check status
sudo systemctl status ai-istanbul-backend
```

**Expected Output:**
```
● ai-istanbul-backend.service - AI Istanbul Backend Service
   Loaded: loaded (/etc/systemd/system/ai-istanbul-backend.service)
   Active: active (running) since ...
   ...
   ✅ Application startup complete
   ✅ Uvicorn running on http://0.0.0.0:8001
```

---

## 🧪 Step 9: Health Checks & Validation (30 minutes)

### Basic Health Check

```bash
# 1. API Health
curl https://staging.ai-istanbul.com/api/health

# Expected: {"status":"healthy","timestamp":"..."}

# 2. Detailed Health
curl https://staging.ai-istanbul.com/api/health/detailed

# Expected:
# {
#   "status": "healthy",
#   "services": {
#     "database": "healthy",
#     "redis": "healthy",
#     "llm": "healthy"
#   },
#   "version": "1.0"
# }
```

### LLM System Test

```bash
# Test Pure LLM endpoint
curl -X POST https://staging.ai-istanbul.com/api/chat/pure-llm \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Hagia Sophia",
    "session_id": "test-session-1"
  }'

# Expected: JSON response with:
# - response: (LLM answer about Hagia Sophia)
# - session_id: "test-session-1"
# - intent: "attraction"
# - confidence: 0.8+
```

### Route Planning Test

```bash
# Test route planning with map
curl -X POST https://staging.ai-istanbul.com/api/chat/pure-llm \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Plan a route from Hagia Sophia to Galata Tower",
    "session_id": "test-session-2"
  }'

# Expected: JSON response with:
# - response: (Route description)
# - map_data: { coordinates, markers, route_data }
# - signals: { needs_gps_routing: true, needs_map: true }
```

### Signal Detection Test

```bash
# Test all 13 signals
curl -X POST https://staging.ai-istanbul.com/api/chat/pure-llm \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Best restaurants in Beyoğlu",
    "session_id": "test-session-3"
  }'

# Expected signals:
# - needs_restaurant: true
# - needs_map: false (no route)
```

### Feedback System Test

```bash
# Submit feedback
curl -X POST https://staging.ai-istanbul.com/api/llm/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user",
    "query": "Best kebab restaurants",
    "response": "Here are the best...",
    "feedback_type": "positive",
    "rating": 5,
    "detected_signals": ["needs_restaurant"],
    "signal_scores": {"needs_restaurant": 0.92}
  }'

# Expected: {"status":"success","message":"Feedback recorded"}
```

### Load Test (Optional)

```bash
# Install Apache Bench
sudo apt install -y apache2-utils

# Test 1000 requests, 10 concurrent
ab -n 1000 -c 10 -H "Content-Type: application/json" \
  -p test_request.json \
  https://staging.ai-istanbul.com/api/chat/pure-llm

# Expected:
# - Requests per second: 20-50
# - Time per request: 20-50ms
# - Failed requests: 0
```

---

## 📊 Step 10: Monitoring Setup (20 minutes)

### Application Logs

```bash
# View real-time logs
tail -f /var/log/ai-istanbul/app.log

# Search for errors
grep ERROR /var/log/ai-istanbul/error.log

# Monitor specific signals
tail -f /var/log/ai-istanbul/app.log | grep "Signal Detection"
```

### System Monitoring Script

```bash
# Create monitoring script
cat > /var/www/ai-istanbul/scripts/monitor.sh << 'EOF'
#!/bin/bash

echo "=== AI Istanbul Staging Monitor ==="
echo "Timestamp: $(date)"
echo ""

# Service status
echo "Backend Service:"
sudo systemctl is-active ai-istanbul-backend
echo ""

# API Health
echo "API Health:"
curl -s https://staging.ai-istanbul.com/api/health | jq .
echo ""

# Database connections
echo "Database Connections:"
sudo -u postgres psql -d ai_istanbul_staging -c "SELECT count(*) FROM pg_stat_activity WHERE datname='ai_istanbul_staging';"
echo ""

# Disk usage
echo "Disk Usage:"
df -h /var/www/ai-istanbul
echo ""

# Memory usage
echo "Memory Usage:"
free -h
echo ""

# Recent errors
echo "Recent Errors (last 10):"
tail -10 /var/log/ai-istanbul/error.log
EOF

chmod +x /var/www/ai-istanbul/scripts/monitor.sh

# Run monitoring
./scripts/monitor.sh
```

### Set Up Cron Job for Daily Health Checks

```bash
# Add daily health check
crontab -e

# Add line:
0 9 * * * /var/www/ai-istanbul/scripts/monitor.sh > /var/log/ai-istanbul/daily-health-$(date +\%Y\%m\%d).log 2>&1
```

---

## ✅ Staging Deployment Complete Checklist

### Infrastructure
- [ ] Server provisioned and configured
- [ ] PostgreSQL database created and populated
- [ ] Redis installed (optional)
- [ ] Nginx configured and running
- [ ] SSL certificate installed and auto-renewal enabled
- [ ] Backend service running (systemd)
- [ ] Frontend built and deployed
- [ ] Firewall rules configured

### Functionality
- [ ] `/api/health` endpoint returns healthy
- [ ] `/api/health/detailed` shows all services healthy
- [ ] Pure LLM chat working (`/api/chat/pure-llm`)
- [ ] Signal detection working (13 signals)
- [ ] Route planning with OSRM working
- [ ] Map visualization data generated
- [ ] POI queries returning results (<10ms)
- [ ] Feedback system recording data
- [ ] Session management working
- [ ] User personalization tracking

### Performance
- [ ] Response time <3s for LLM queries
- [ ] POI queries <10ms
- [ ] Route planning <500ms
- [ ] No errors under 100 concurrent requests
- [ ] Memory usage stable
- [ ] CPU usage reasonable (<80%)

### Monitoring
- [ ] Application logs accessible
- [ ] Error logs accessible
- [ ] Monitoring script working
- [ ] Daily health checks scheduled
- [ ] Alerts configured (optional)

---

## 🚀 Phase 2: Production Deployment (AFTER STAGING)

### Estimated Time: 1 day (after 1-2 weeks of staging validation)

### Pre-Production Checklist

#### Staging Validation (1-2 weeks)
- [ ] No critical bugs found in staging
- [ ] Performance metrics acceptable
- [ ] User acceptance testing complete
- [ ] Load testing passed
- [ ] Security audit passed
- [ ] Documentation reviewed and updated

#### Production Infrastructure
- [ ] Production server provisioned (higher specs than staging)
- [ ] Load balancer configured (optional, for scaling)
- [ ] CDN configured (optional, for frontend assets)
- [ ] Backup system tested and verified
- [ ] Disaster recovery plan documented
- [ ] Monitoring & alerting configured

### Production Deployment Steps

**Same as staging, with these differences:**

1. **Environment:**
   - `ENVIRONMENT=production`
   - `DEBUG=false`
   - Stronger `SECRET_KEY` and `JWT_SECRET`

2. **Domain:**
   - `ai-istanbul.com` (instead of `staging.ai-istanbul.com`)

3. **Database:**
   - Production database with regular backups
   - Database replication (optional)

4. **Resources:**
   - More workers: `--workers 8` (instead of 4)
   - Larger server (4-8 CPU cores, 8-16 GB RAM)

5. **Monitoring:**
   - Production-grade monitoring (Datadog, New Relic, etc.)
   - PagerDuty or similar for alerts
   - Uptime monitoring (UptimeRobot, Pingdom)

---

## 🎯 Admin Dashboard - Production Ready ⭐

The **Admin Dashboard** is fully implemented and ready for deployment. See `ADMIN_DASHBOARD_STATUS.md` for complete details.

### Quick Access
- **Development:** http://localhost:5173/admin
- **Production:** https://ai-istanbul.vercel.app/admin (after deployment)

### Required Environment Variables

#### Backend (Render):
```env
ADMIN_USERNAME=your_admin_username
ADMIN_PASSWORD_HASH=$2b$12$your_bcrypt_hash_here
JWT_SECRET_KEY=your_secure_random_key_here
JWT_ALGORITHM=HS256
```

#### Frontend (Vercel):
```env
VITE_API_URL=https://ai-istanbul-backend.render.com
```

### Generate Admin Credentials

```bash
# Install bcrypt
pip install bcrypt

# Generate password hash
python3 -c "import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode())"
```

### Dashboard Features
✅ Secure JWT authentication  
✅ Real-time analytics with WebSocket  
✅ LLM performance metrics  
✅ User behavior insights  
✅ Blog analytics  
✅ Feedback management  
✅ System health monitoring  
✅ Data export (JSON/CSV)  
✅ Dark mode support  
✅ Responsive design  

---

## 🔄 Deployment Commands Quick Reference

### Start/Stop Services

```bash
# Backend
sudo systemctl start ai-istanbul-backend
sudo systemctl stop ai-istanbul-backend
sudo systemctl restart ai-istanbul-backend
sudo systemctl status ai-istanbul-backend

# Nginx
sudo systemctl reload nginx
sudo systemctl restart nginx
```

### View Logs

```bash
# Real-time application logs
tail -f /var/log/ai-istanbul/app.log

# Real-time error logs
tail -f /var/log/ai-istanbul/error.log

# Nginx access logs
tail -f /var/log/nginx/access.log

# Nginx error logs
tail -f /var/log/nginx/error.log

# Systemd service logs
sudo journalctl -u ai-istanbul-backend -f
```

### Update Code

```bash
# Pull latest changes
cd /var/www/ai-istanbul
git pull origin main

# Backend: restart service
sudo systemctl restart ai-istanbul-backend

# Frontend: rebuild
cd frontend
npm run build
```

### Database Backup

```bash
# Create backup
sudo -u postgres pg_dump ai_istanbul_staging > backup_$(date +%Y%m%d).sql

# Restore backup
sudo -u postgres psql ai_istanbul_staging < backup_20251116.sql
```

---

## 🛡️ Rollback Procedure

### If deployment fails:

```bash
# 1. Stop services
sudo systemctl stop ai-istanbul-backend

# 2. Revert code
cd /var/www/ai-istanbul
git reset --hard HEAD~1

# 3. Restore database (if needed)
sudo -u postgres psql ai_istanbul_staging < backup_last_good.sql

# 4. Rebuild frontend
cd frontend
npm run build

# 5. Restart services
sudo systemctl start ai-istanbul-backend
sudo systemctl reload nginx

# 6. Verify
curl https://staging.ai-istanbul.com/api/health
```

---

## 📞 Troubleshooting

### Backend not starting

```bash
# Check logs
sudo journalctl -u ai-istanbul-backend -n 50

# Common issues:
# - Database connection failed → Check DATABASE_URL in .env
# - OpenAI API key invalid → Check OPENAI_API_KEY
# - Port already in use → Check if another process is using 8001
```

### 502 Bad Gateway (Nginx)

```bash
# Check if backend is running
sudo systemctl status ai-istanbul-backend

# Check Nginx error logs
tail -f /var/log/nginx/error.log

# Test backend directly
curl http://localhost:8001/api/health
```

### Database connection errors

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U ai_istanbul_user -d ai_istanbul_staging -h localhost

# Check firewall
sudo ufw status
```

---

## 📊 Success Metrics

### After 1 week in staging:

- [ ] 0 critical bugs
- [ ] >95% uptime
- [ ] <3s average response time
- [ ] 0 data loss incidents
- [ ] Positive user feedback

### Production goals:

- [ ] 99.9% uptime
- [ ] <2s average response time
- [ ] Handle 1,000+ concurrent users
- [ ] <0.1% error rate

---

## ✅ Final Deployment Checklist

- [ ] Staging deployment complete
- [ ] All health checks passing
- [ ] Load testing passed
- [ ] Security audit completed
- [ ] Documentation updated
- [ ] Team trained on deployment process
- [ ] Monitoring configured
- [ ] Backup system tested
- [ ] Rollback procedure documented and tested
- [ ] Production deployment plan approved

---

## 🎉 Next Steps After Deployment

1. **Monitor for 24 hours** - Watch for any issues
2. **Collect user feedback** - Use feedback system
3. **Analyze metrics** - Response times, error rates
4. **Optimize performance** - Based on real usage
5. **Plan next features** - Based on feedback

---

**Generated:** November 16, 2025  
**Status:** ✅ Ready for Staging Deployment  
**Estimated Time:** 1-2 days (staging), 1 day (production)  
**Risk Level:** 🟢 Low (all systems tested)
