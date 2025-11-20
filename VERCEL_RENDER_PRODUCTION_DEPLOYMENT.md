# 🚀 Vercel + Render Production Deployment Guide

**Date:** January 2025  
**Status:** Phase 4-8 Implementation Complete - Ready for Production  
**Stack:** Vercel (Frontend) + Render (Backend + PostgreSQL + Redis)

---

## 📋 Overview

This guide walks you through deploying the Istanbul AI (KAM) platform to production with all advanced features:

✅ **Core Features:**
- Multi-language LLM (Turkish, English, Russian, Arabic)
- Context-aware recommendations
- Real-time chat with WebSocket
- Map integration (OSRM routing, Leaflet UI)

✅ **Advanced Features (Phase 4-8):**
- CI/CD Pipeline (GitHub Actions)
- 3-Tier Caching (Redis L1 + Semantic L2 + Persistent L3)
- A/B Testing Framework
- User Feedback Collection & Analysis
- Prometheus + Grafana Monitoring
- Admin Dashboard

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          PRODUCTION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  VERCEL (Frontend - CDN + Edge)                                  │
│  ├── Next.js/React App                                           │
│  ├── Static Assets (CDN)                                         │
│  ├── SSR/ISR Pages                                               │
│  └── Environment: NEXT_PUBLIC_BACKEND_URL                        │
│       ↓ HTTPS                                                    │
│  RENDER (Backend - Auto-scaling)                                 │
│  ├── FastAPI Application                                         │
│  │   ├── /api/chat (LLM endpoints)                               │
│  │   ├── /api/recommendations (personalization)                  │
│  │   ├── /api/feedback (user feedback)                           │
│  │   ├── /api/ab-tests (experiments)                             │
│  │   ├── /api/monitoring (metrics)                               │
│  │   └── /metrics (Prometheus)                                   │
│  ├── PostgreSQL (Managed)                                        │
│  │   ├── User data                                               │
│  │   ├── Chat history                                            │
│  │   ├── Feedback records                                        │
│  │   └── A/B test results                                        │
│  └── Redis (Managed)                                             │
│      ├── L1 Cache (hot data)                                     │
│      ├── Session storage                                         │
│      └── Rate limiting                                           │
│                                                                  │
│  MONITORING (Optional - External)                                │
│  ├── Grafana Cloud (metrics visualization)                       │
│  ├── Prometheus (metrics collection)                             │
│  └── Loki (log aggregation)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Prerequisites

### Required Accounts (All Free Tier Available)

- ✅ **GitHub Account** (for code hosting + CI/CD)
- ✅ **Vercel Account** (for frontend hosting)
  - Sign up: https://vercel.com/signup
  - Free tier: Unlimited personal projects
- ✅ **Render Account** (for backend + databases)
  - Sign up: https://render.com/register
  - Free tier: 750 hours/month
- ✅ **Groq API Key** (for LLM)
  - Get key: https://console.groq.com/
  - Free tier: 100 requests/day

### Optional (For Full Monitoring)

- ⭐ **Grafana Cloud** (for dashboards)
  - Sign up: https://grafana.com/auth/sign-up/create-user
  - Free tier: 10k series metrics

---

## 📦 Step 1: Backend Deployment to Render (30 mins)

### 1.1 Create Render Account & Services

1. **Go to Render Dashboard:**
   ```
   https://dashboard.render.com/
   ```

2. **Create PostgreSQL Database:**
   - Click "New +" → "PostgreSQL"
   - Name: `istanbul-ai-db`
   - Instance Type: `Free`
   - Region: `Frankfurt` (EU - closest to Istanbul)
   - Click "Create Database"
   - **Copy the Internal Database URL** (starts with `postgresql://`)

3. **Create Redis Instance:**
   - Click "New +" → "Redis"
   - Name: `istanbul-ai-redis`
   - Instance Type: `Free` (25MB)
   - Region: `Frankfurt`
   - Click "Create Redis"
   - **Copy the Internal Redis URL** (starts with `redis://`)

### 1.2 Deploy Backend Web Service

1. **Create Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Name: `istanbul-ai-backend`
   - Region: `Frankfurt`
   - Branch: `main`
   - Root Directory: `.` (or leave blank)
   - Runtime: `Python 3.11`
   - Build Command:
     ```bash
     pip install -r backend/requirements.txt
     ```
   - Start Command:
     ```bash
     cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - Instance Type: `Free`

2. **Set Environment Variables:**

   Click "Environment" tab and add:

   ```env
   # Core Configuration
   ENVIRONMENT=production
   DEBUG=false
   PORT=10000
   
   # Database URLs (from Step 1.1)
   DATABASE_URL=postgresql://user:pass@hostname/dbname
   REDIS_URL=redis://hostname:port
   
   # LLM Configuration
   GROQ_API_KEY=your_groq_api_key_here
   LLM_MODEL=llama-3.1-70b-versatile
   
   # CORS (will update after Vercel deployment)
   CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
   
   # Advanced Features
   ENABLE_AB_TESTING=true
   ENABLE_MONITORING=true
   ENABLE_CACHING=true
   ENABLE_FEEDBACK=true
   
   # Cache Configuration
   REDIS_CACHE_TTL=3600
   SEMANTIC_CACHE_THRESHOLD=0.85
   
   # A/B Testing
   AB_TEST_SAMPLE_RATE=0.1
   
   # Rate Limiting
   RATE_LIMIT_PER_MINUTE=60
   
   # Session
   SESSION_SECRET_KEY=generate_a_secure_random_string_here
   ```

3. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy
   - Wait for deployment to complete (~5-10 mins)
   - **Copy your backend URL:** `https://istanbul-ai-backend.onrender.com`

### 1.3 Verify Backend Deployment

```bash
# Health check
curl https://istanbul-ai-backend.onrender.com/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}

# Test chat endpoint
curl -X POST https://istanbul-ai-backend.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Merhaba, İstanbul'\''da ne yapabilirim?",
    "language": "tr"
  }'

# Test recommendations
curl -X POST https://istanbul-ai-backend.onrender.com/api/recommendations/personalized \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "limit": 5}'

# Test metrics endpoint (Prometheus)
curl https://istanbul-ai-backend.onrender.com/metrics
```

---

## 🎨 Step 2: Frontend Deployment to Vercel (15 mins)

### 2.1 Prepare Frontend for Deployment

1. **Update Environment Variables:**

   Create `.env.production` in your frontend directory:

   ```env
   # Backend API URL (from Step 1.3)
   NEXT_PUBLIC_API_URL=https://istanbul-ai-backend.onrender.com
   NEXT_PUBLIC_BACKEND_URL=https://istanbul-ai-backend.onrender.com
   
   # WebSocket URL (same as backend)
   NEXT_PUBLIC_WEBSOCKET_URL=wss://istanbul-ai-backend.onrender.com
   
   # Features
   NEXT_PUBLIC_ENABLE_AB_TESTING=true
   NEXT_PUBLIC_ENABLE_FEEDBACK=true
   NEXT_PUBLIC_ENABLE_ANALYTICS=true
   
   # Google Analytics (optional)
   NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX
   ```

2. **Verify Build Locally:**

   ```bash
   cd frontend
   npm install
   npm run build
   npm run start
   
   # Open http://localhost:3000
   # Test chat, recommendations, maps
   ```

### 2.2 Deploy to Vercel

#### Option A: Vercel CLI (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy to production
cd frontend
vercel --prod

# Follow prompts:
# - Project name: istanbul-ai
# - Framework: Next.js (auto-detected)
# - Root directory: ./frontend (or current)
# - Build command: npm run build (auto-detected)
```

#### Option B: Vercel Dashboard (GUI)

1. **Go to Vercel Dashboard:**
   ```
   https://vercel.com/dashboard
   ```

2. **Import Project:**
   - Click "Add New" → "Project"
   - Import your GitHub repository
   - Select repository: `your-username/ai-stanbul`

3. **Configure Project:**
   - Framework Preset: `Next.js`
   - Root Directory: `frontend` (if your Next.js app is in a subdirectory)
   - Build Command: `npm run build` (auto-detected)
   - Output Directory: `.next` (auto-detected)

4. **Set Environment Variables:**
   - Click "Environment Variables"
   - Add all variables from `.env.production` above
   - Make sure to use the correct backend URL from Step 1.3

5. **Deploy:**
   - Click "Deploy"
   - Wait for build to complete (~3-5 mins)
   - **Copy your frontend URL:** `https://istanbul-ai.vercel.app`

### 2.3 Update CORS in Backend

Now that you have the Vercel URL, update the CORS configuration:

1. **Go to Render Dashboard:**
   - Open your backend service: `istanbul-ai-backend`
   - Go to "Environment" tab

2. **Update CORS_ORIGINS:**
   ```env
   CORS_ORIGINS=https://istanbul-ai.vercel.app,https://www.istanbul-ai.com
   ```

3. **Redeploy:**
   - Click "Manual Deploy" → "Deploy latest commit"
   - Wait for redeployment (~2 mins)

### 2.4 Verify Frontend Deployment

```bash
# Open your production site
open https://istanbul-ai.vercel.app

# Test features:
# ✅ Chat interface loads
# ✅ Send a message in Turkish: "Merhaba!"
# ✅ Get recommendations
# ✅ View map (if applicable)
# ✅ Submit feedback
# ✅ Check A/B test assignment
```

---

## 🔍 Step 3: Monitoring Setup (Optional - 30 mins)

### Option A: Grafana Cloud (Recommended)

1. **Create Grafana Cloud Account:**
   ```
   https://grafana.com/auth/sign-up/create-user
   ```

2. **Set Up Prometheus Remote Write:**

   In your backend environment variables (Render), add:

   ```env
   PROMETHEUS_REMOTE_WRITE_URL=https://prometheus-prod-01-eu-west-0.grafana.net/api/prom/push
   PROMETHEUS_REMOTE_WRITE_USERNAME=your_grafana_username
   PROMETHEUS_REMOTE_WRITE_PASSWORD=your_grafana_api_key
   ```

3. **Import Dashboards:**

   - Go to Grafana Cloud → Dashboards → Import
   - Use dashboard IDs from `monitoring/grafana_dashboards.py`
   - Dashboard 1: LLM Performance (custom)
   - Dashboard 2: A/B Test Results (custom)
   - Dashboard 3: User Engagement (custom)

### Option B: Self-Hosted Monitoring (Docker)

If you prefer self-hosted monitoring (requires separate server/VPS):

```bash
# Clone monitoring setup
git clone your-repo
cd ai-stanbul

# Run monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards
open http://your-server:3000  # Grafana
open http://your-server:9090  # Prometheus
```

---

## 🧪 Step 4: Post-Deployment Testing (15 mins)

### 4.1 Automated Health Checks

```bash
# Run the validation script
./validate_week3-4_integration.sh

# Or manually test:
curl https://istanbul-ai-backend.onrender.com/health
curl https://istanbul-ai-backend.onrender.com/metrics
```

### 4.2 Feature Testing Checklist

#### Basic Features
- [ ] **Homepage loads** (Vercel)
- [ ] **Chat interface** renders
- [ ] **Send message** in Turkish: "Merhaba!"
- [ ] **Get response** from LLM
- [ ] **Multi-language** (test English, Russian, Arabic)

#### Advanced Features
- [ ] **Recommendations API:**
  ```bash
  curl -X POST https://istanbul-ai-backend.onrender.com/api/recommendations/personalized \
    -H "Content-Type: application/json" \
    -d '{"user_id": "test", "limit": 5}'
  ```

- [ ] **A/B Testing:**
  ```bash
  curl https://istanbul-ai-backend.onrender.com/api/ab-tests/active
  ```

- [ ] **Feedback Submission:**
  ```bash
  curl -X POST https://istanbul-ai-backend.onrender.com/api/feedback \
    -H "Content-Type: application/json" \
    -d '{
      "user_id": "test",
      "rating": 5,
      "comment": "Great app!",
      "category": "general"
    }'
  ```

- [ ] **Metrics Endpoint:**
  ```bash
  curl https://istanbul-ai-backend.onrender.com/metrics
  ```

- [ ] **Cache Performance:**
  - Send same query twice
  - Second response should be faster (<50ms)

### 4.3 Performance Benchmarks

**Expected Performance (Free Tier):**
- Cold start: <5s (first request after idle)
- Warm requests: <200ms
- LLM response: 1-3s
- Cache hit: <50ms
- Database query: <100ms

```bash
# Benchmark script
time curl -X POST https://istanbul-ai-backend.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Merhaba", "language": "tr"}'
```

---

## 🎛️ Step 5: Admin Dashboard Access

### 5.1 Set Admin Credentials

In Render environment variables:

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=create_secure_password_here
ADMIN_EMAIL=your_email@example.com
```

### 5.2 Access Dashboard

```
https://istanbul-ai.vercel.app/admin
```

**Login with:**
- Username: `admin`
- Password: (from environment variable)

### 5.3 Dashboard Features

- 📊 **Analytics:** User engagement, chat volume, language distribution
- 🧪 **A/B Tests:** Active experiments, results, confidence intervals
- 📝 **Feedback:** User ratings, comments, sentiment analysis
- 🔍 **Monitoring:** Response times, error rates, cache hit ratios
- 👥 **Users:** Active users, retention, top queries

---

## 🚨 Step 6: CI/CD Pipeline (GitHub Actions)

### 6.1 Verify CI/CD Workflow

The CI/CD pipeline is already configured in `.github/workflows/staging-deploy.yml`.

**Automatic Triggers:**
- ✅ **On Push to `main`:** Full test suite + deploy to Render
- ✅ **On Pull Request:** Run tests only
- ✅ **On Schedule:** Daily health checks

### 6.2 GitHub Secrets

Add these secrets to your GitHub repository:

**Settings → Secrets and variables → Actions → New repository secret:**

```
RENDER_API_KEY=your_render_api_key
RENDER_SERVICE_ID=your_service_id
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=your_postgres_url
REDIS_URL=your_redis_url
VERCEL_TOKEN=your_vercel_token
VERCEL_ORG_ID=your_org_id
VERCEL_PROJECT_ID=your_project_id
```

**To get Render API Key:**
1. Go to Render Dashboard → Account Settings
2. Click "API Keys" → "Create API Key"
3. Copy the key

**To get Vercel tokens:**
```bash
vercel link  # Links your project
cat .vercel/project.json  # Get project ID and org ID
```

### 6.3 Test CI/CD

```bash
# Make a small change
echo "# Production ready!" >> README.md

# Commit and push
git add .
git commit -m "Test CI/CD pipeline"
git push origin main

# Watch the workflow
# Go to: https://github.com/your-username/ai-stanbul/actions
```

---

## 📈 Step 7: Production Optimization

### 7.1 Performance Tuning

**Backend (Render):**

```env
# Increase workers for better concurrency
WORKERS=2
WORKER_CLASS=uvicorn.workers.UvicornWorker

# Enable production optimizations
PYTHON_ENV=production
PYTHONOPTIMIZE=1
```

**Frontend (Vercel):**

- Enable ISR (Incremental Static Regeneration):
  ```typescript
  // pages/index.tsx
  export const revalidate = 60; // Revalidate every 60 seconds
  ```

- Use Next.js Image Optimization:
  ```typescript
  import Image from 'next/image';
  <Image src="/logo.png" width={200} height={50} alt="Logo" />
  ```

### 7.2 Database Optimization

**Create Indexes:**

```bash
# Connect to Render PostgreSQL
# Go to Render Dashboard → Database → Connect → Copy PSQL command

# Run indexes
CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX idx_feedback_created_at ON user_feedback(created_at);
CREATE INDEX idx_ab_tests_user_id ON ab_test_assignments(user_id);
```

### 7.3 Redis Optimization

**Connection Pooling:**

```python
# backend/services/caching/advanced_cache.py
from redis.connection import ConnectionPool

pool = ConnectionPool.from_url(
    os.getenv("REDIS_URL"),
    max_connections=10,
    socket_connect_timeout=5
)
```

---

## 🔒 Step 8: Security Hardening

### 8.1 Environment Variables

**Never commit:**
- ❌ API keys
- ❌ Database URLs
- ❌ Admin passwords
- ❌ Session secrets

**Use `.env` and `.env.production`:**
```bash
# Add to .gitignore
.env
.env.local
.env.production
```

### 8.2 Rate Limiting

Already configured in `backend/services/feedback/feedback_collector.py`:

```python
# Default rate limits:
# - 60 requests/minute per IP
# - 100 requests/hour per user
```

### 8.3 CORS Configuration

**Restrict origins in production:**

```env
# Only allow your domains
CORS_ORIGINS=https://istanbul-ai.vercel.app,https://www.istanbul-ai.com
```

### 8.4 SSL/HTTPS

✅ **Automatically handled by Vercel and Render**
- Free SSL certificates (Let's Encrypt)
- HTTPS enforced by default
- No configuration needed

---

## 📊 Step 9: Monitoring & Alerts

### 9.1 Key Metrics to Monitor

**Application Metrics:**
- Request rate (requests/min)
- Response time (p50, p95, p99)
- Error rate (%)
- Cache hit rate (%)

**Business Metrics:**
- Daily Active Users (DAU)
- Chat sessions per user
- Recommendation CTR
- Feedback submission rate

**Infrastructure Metrics:**
- CPU usage (%)
- Memory usage (%)
- Database connections
- Redis memory usage

### 9.2 Set Up Alerts (Grafana Cloud)

**Create alerts for:**
- Error rate > 5%
- Response time p95 > 3s
- Database connection pool > 80%
- Redis memory > 90%

```yaml
# Example alert rule (Grafana)
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value }}%"
```

### 9.3 Uptime Monitoring

**Use UptimeRobot (Free):**

1. Sign up: https://uptimerobot.com
2. Add monitor:
   - Type: HTTP(S)
   - URL: `https://istanbul-ai-backend.onrender.com/health`
   - Interval: 5 minutes
3. Set up email alerts

---

## 🎯 Step 10: Go Live Checklist

### Pre-Launch
- [ ] All tests passing in CI/CD
- [ ] Environment variables configured
- [ ] Database indexes created
- [ ] Redis cache configured
- [ ] CORS origins restricted
- [ ] Rate limiting enabled
- [ ] Monitoring dashboards set up
- [ ] Admin dashboard accessible
- [ ] SSL certificates valid

### Launch
- [ ] Deploy backend to Render
- [ ] Deploy frontend to Vercel
- [ ] Verify all endpoints
- [ ] Test critical user flows
- [ ] Check monitoring dashboards
- [ ] Set up uptime monitoring
- [ ] Configure alerts

### Post-Launch
- [ ] Monitor error rates (first 24h)
- [ ] Check performance metrics
- [ ] Review user feedback
- [ ] Analyze A/B test results
- [ ] Optimize slow queries
- [ ] Scale resources if needed

---

## 🆘 Troubleshooting

### Issue: Backend Won't Start on Render

**Symptoms:**
- Service shows "Deploy failed"
- Build logs show dependency errors

**Solutions:**
```bash
# Check Python version
python --version  # Should be 3.11+

# Verify requirements.txt
pip install -r backend/requirements.txt

# Check start command
cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Issue: Frontend Can't Connect to Backend

**Symptoms:**
- Network errors in browser console
- "Failed to fetch" errors

**Solutions:**

1. **Check CORS:**
   ```env
   # In Render backend
   CORS_ORIGINS=https://your-app.vercel.app
   ```

2. **Verify Backend URL:**
   ```env
   # In Vercel frontend
   NEXT_PUBLIC_API_URL=https://istanbul-ai-backend.onrender.com
   ```

3. **Test backend directly:**
   ```bash
   curl https://istanbul-ai-backend.onrender.com/health
   ```

### Issue: Slow Response Times

**Symptoms:**
- Requests take >5s
- Timeouts

**Solutions:**

1. **Check Render free tier limits:**
   - Free tier spins down after 15 mins of inactivity
   - First request after idle = cold start (5-10s)
   - Consider upgrading to paid tier for always-on

2. **Enable caching:**
   ```env
   ENABLE_CACHING=true
   REDIS_CACHE_TTL=3600
   ```

3. **Optimize database queries:**
   ```sql
   -- Add indexes
   CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);
   ```

### Issue: Redis Connection Errors

**Symptoms:**
- "Connection refused" errors
- Cache misses

**Solutions:**

1. **Verify Redis URL:**
   ```bash
   echo $REDIS_URL
   # Should be: redis://hostname:port
   ```

2. **Check Redis health:**
   - Go to Render Dashboard → Redis service
   - Check "Status" (should be "Available")

3. **Test connection:**
   ```python
   import redis
   r = redis.from_url(os.getenv("REDIS_URL"))
   r.ping()  # Should return True
   ```

### Issue: A/B Tests Not Working

**Symptoms:**
- All users get same variant
- No metrics recorded

**Solutions:**

1. **Verify A/B testing enabled:**
   ```env
   ENABLE_AB_TESTING=true
   AB_TEST_SAMPLE_RATE=0.1
   ```

2. **Check database tables:**
   ```sql
   SELECT * FROM ab_test_experiments;
   SELECT * FROM ab_test_assignments LIMIT 10;
   ```

3. **Test API:**
   ```bash
   curl https://your-backend.onrender.com/api/ab-tests/active
   ```

---

## 📚 Additional Resources

### Documentation
- 📖 **Implementation Tracker:** `IMPLEMENTATION_TRACKER.md`
- 📖 **Quick Integration Steps:** `QUICK_INTEGRATION_STEPS.md`
- 📖 **Phase 4-8 Complete:** `PHASE_4_8_IMPLEMENTATION_COMPLETE.md`
- 📖 **LLaMA Enhancement Plan:** `LLAMA_ENHANCEMENT_PLAN.md`

### API Documentation
- 🔗 **Backend API:** `https://istanbul-ai-backend.onrender.com/docs`
- 🔗 **Redoc:** `https://istanbul-ai-backend.onrender.com/redoc`

### Platform Documentation
- 🔗 **Vercel Docs:** https://vercel.com/docs
- 🔗 **Render Docs:** https://render.com/docs
- 🔗 **FastAPI Docs:** https://fastapi.tiangolo.com
- 🔗 **Next.js Docs:** https://nextjs.org/docs

### Support
- 💬 **Issues:** GitHub Issues
- 📧 **Email:** your_email@example.com
- 🐛 **Bug Reports:** GitHub Issues with `bug` label

---

## 🎉 Congratulations!

Your Istanbul AI (KAM) platform is now live in production! 🚀

**Next Steps:**
1. Monitor metrics for first 24 hours
2. Collect user feedback
3. Analyze A/B test results
4. Optimize based on data
5. Plan next features (Phase 9-10)

**Celebrate your launch! 🎊**

---

*Last Updated: January 2025*  
*Version: 1.0.0*  
*Status: Production Ready ✅*
