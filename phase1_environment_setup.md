# 🔧 Phase 1: Environment Setup Guide

## Overview
This guide walks you through setting up environment variables for production deployment across Vercel (frontend) and Render (backend).

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- ✅ RunPod LLM server running (accessible via proxy URL)
- ✅ Backend code deployed to Render
- ✅ Frontend code deployed to Vercel
- ✅ Domain names configured (frontend and API)
- ✅ SSL certificates active (automatic on Vercel/Render)
- ✅ Database (PostgreSQL) provisioned
- ✅ Redis instance (optional but recommended)

---

## 🎯 Step 1: Get Your RunPod Proxy URL

### 1.1 Access RunPod Console
1. Go to: https://www.runpod.io/console/pods
2. Log in to your account
3. Locate your active pod (should show "Running")

### 1.2 Get HTTP Service URL
1. Click on your pod
2. Click **"Connect"** button
3. Select **"HTTP Service [Port 8888]"** (or whatever port your LLM server uses)
4. Copy the proxy URL (example: `https://abc123def456-8888.proxy.runpod.net`)

### 1.3 Format the URL
- Your LLM server expects `/v1` endpoint
- Final URL: `https://abc123def456-8888.proxy.runpod.net/v1`
- Save this URL - you'll need it for backend configuration

**Example:**
```
RunPod Proxy: https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm
LLM API URL:  https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
```

---

## 🚀 Step 2: Configure Vercel (Frontend)

### 2.1 Access Vercel Dashboard
1. Go to: https://vercel.com/dashboard
2. Find your project (e.g., `ai-stanbul`)
3. Click on the project

### 2.2 Navigate to Environment Variables
1. Click **"Settings"** tab
2. Click **"Environment Variables"** in sidebar

### 2.3 Add/Update Variables

Add these environment variables (one by one):

#### Required Variables:
```bash
# API Endpoints
VITE_API_BASE_URL=https://api.aistanbul.net
VITE_PURE_LLM_API_URL=https://api.aistanbul.net

# WebSocket (if using real-time features)
VITE_WEBSOCKET_URL=wss://api.aistanbul.net

# Map Configuration (Free - No API Keys)
VITE_MAP_PROVIDER=openstreetmap
VITE_OSM_TILE_URL=https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
VITE_DEFAULT_MAP_CENTER_LAT=41.0082
VITE_DEFAULT_MAP_CENTER_LNG=28.9784
VITE_DEFAULT_MAP_ZOOM=13

# Geocoding & Routing (Free - No API Keys)
VITE_GEOCODING_PROVIDER=nominatim
VITE_NOMINATIM_URL=https://nominatim.openstreetmap.org
VITE_ROUTING_PROVIDER=osrm
VITE_OSRM_URL=https://router.project-osrm.org

# Feature Flags
VITE_ENABLE_BLOG=true
VITE_ENABLE_CHAT=true
VITE_ENABLE_MAPS=true
VITE_ENABLE_LOCATION_TRACKING=true

# Production Settings
VITE_ENVIRONMENT=production
```

### 2.4 Select Environment Scope
For each variable:
- Check: **Production** ✅
- Check: **Preview** (optional)
- Check: **Development** (optional for local testing)

### 2.5 Redeploy Frontend
1. Go to **"Deployments"** tab
2. Find the latest deployment
3. Click **"⋯"** (three dots) → **"Redeploy"**
4. Wait 2-3 minutes for build to complete
5. Check deployment logs for errors

### 2.6 Verify Frontend
```bash
# Open in browser
https://aistanbul.net

# Check browser console (F12)
# Should see: "Chatbot component loaded with Pure LLM backend"
```

---

## ⚙️ Step 3: Configure Render (Backend)

### 3.1 Access Render Dashboard
1. Go to: https://dashboard.render.com
2. Find your backend service (e.g., `ai-stanbul-backend`)
3. Click on the service

### 3.2 Navigate to Environment Variables
1. Click **"Environment"** tab in sidebar

### 3.3 Add/Update Variables

#### Critical Variables:
```bash
# LLM Configuration (CRITICAL!)
LLM_API_URL=https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1

# CORS Configuration (CRITICAL!)
ALLOWED_ORIGINS=["https://aistanbul.net","https://www.aistanbul.net","https://api.aistanbul.net","http://localhost:3000","http://localhost:5173"]

# Database
DATABASE_URL=postgresql://username:password@host:5432/dbname

# Redis (optional but recommended for caching)
REDIS_URL=redis://host:6379

# Environment
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# Security
SECRET_KEY=your_super_secret_key_here_change_this
JWT_SECRET_KEY=your_jwt_secret_key_here_change_this

# Rate Limiting
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Feature Flags
USE_NEURAL_RANKING=True
ADVANCED_UNDERSTANDING_ENABLED=True
```

#### Optional Variables (for advanced features):
```bash
# Error Tracking (Optional - Sign up at sentry.io)
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1

# Weather API (Optional)
OPENWEATHER_API_KEY=your_api_key_here

# Monitoring
ENABLE_PROMETHEUS=True
METRICS_PORT=9090
```

### 3.4 Save and Auto-Deploy
1. Click **"Save Changes"** button
2. Render will automatically trigger a redeploy
3. Wait 3-5 minutes for build and deployment
4. Check deployment logs

### 3.5 Verify Backend Logs
Look for these success messages in logs:
```
✅ RunPod LLM Client loaded
✅ LLM Connection: Success
🚀 Server started on 0.0.0.0:8000
📊 Health check endpoint: /health
```

### 3.6 Test Backend Health
```bash
# Test health endpoint
curl https://api.aistanbul.net/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "llm_status": "connected"
}
```

---

## 🧪 Step 4: Run Health Checks

### 4.1 Install Dependencies
```bash
pip install requests colorama python-dotenv
```

### 4.2 Set Environment Variables
```bash
export BACKEND_URL=https://api.aistanbul.net
export FRONTEND_URL=https://aistanbul.net
export LLM_API_URL=https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
```

### 4.3 Run Health Check Script
```bash
python3 phase1_health_check.py
```

### 4.4 Expected Output
```
🏥 PHASE 1 HEALTH CHECK - PRODUCTION READINESS

Configuration:
  Backend URL: https://api.aistanbul.net
  Frontend URL: https://aistanbul.net
  LLM API URL: https://pbvs3agzznvsgn-8888.proxy.runpod.net/v1

1️⃣ Backend Health Check
✅ Backend Health: Backend is healthy

2️⃣ LLM Server Health Check
✅ LLM Health: LLM server is healthy

3️⃣ Frontend Loading Check
✅ Frontend Loading: Frontend loads successfully

4️⃣ CORS Configuration Check
✅ CORS Configuration: CORS configured correctly

5️⃣ Multi-Language Chat Tests
✅ Chat (en): Chat works in en
✅ Chat (tr): Chat works in tr
✅ Chat (ar): Chat works in ar
✅ Chat (de): Chat works in de
✅ Chat (fr): Chat works in fr
✅ Chat (es): Chat works in es

📊 SUMMARY
Total Tests: 10
Passed: 10
Failed: 0
Success Rate: 100.0%

🎉 ALL TESTS PASSED! System is production ready!
```

---

## 🐛 Troubleshooting

### Issue 1: Backend can't connect to LLM
**Symptoms:**
- Health check shows "LLM_API_URL not configured"
- Chat responses timeout or fail

**Solutions:**
1. Verify `LLM_API_URL` is set in Render environment
2. Check RunPod pod is running (not stopped)
3. Test URL directly:
   ```bash
   curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health
   ```
4. Check Render logs for connection errors

### Issue 2: CORS Errors in Browser
**Symptoms:**
- Browser console shows "CORS policy blocked"
- Frontend can't reach backend

**Solutions:**
1. Check `ALLOWED_ORIGINS` includes your frontend URL
2. Ensure URLs include protocol (`https://`)
3. Check for trailing slashes
4. Verify backend redeployed after changes

### Issue 3: Frontend Not Using Pure LLM
**Symptoms:**
- Chat works but doesn't use advanced features
- Responses are generic

**Solutions:**
1. Check browser console for API URLs
2. Verify `VITE_PURE_LLM_API_URL` is set correctly
3. Clear browser cache and hard refresh (Cmd+Shift+R)
4. Check Network tab - requests should go to correct endpoint

### Issue 4: Multi-Language Not Working
**Symptoms:**
- All languages return English responses
- Language selector doesn't affect output

**Solutions:**
1. Check `language` parameter is sent in API requests
2. Verify backend prompt templates include language
3. Test directly with curl:
   ```bash
   curl -X POST https://api.aistanbul.net/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"Merhaba","language":"tr"}'
   ```

---

## ✅ Success Criteria

Your Phase 1 setup is complete when:

- ✅ Backend health check returns 200 OK
- ✅ LLM health check returns 200 OK
- ✅ Frontend loads without errors
- ✅ CORS allows frontend → backend communication
- ✅ All 6 languages return appropriate responses
- ✅ Chat responses are contextual and intelligent
- ✅ No console errors in browser
- ✅ API response time < 5 seconds
- ✅ Health check script passes 100%

---

## 📝 Next Steps

Once all checks pass:

1. **Document Your URLs** - Save all production URLs
2. **Run Extended Tests** - Test edge cases and error handling
3. **Monitor Performance** - Watch response times and error rates
4. **Gather Feedback** - Have team test in real scenarios
5. **Move to Phase 2** - Begin modular handler implementation

---

## 📞 Support

If you encounter issues not covered here:

1. Check deployment logs in Vercel/Render
2. Review browser console for errors
3. Test API endpoints directly with curl
4. Verify RunPod pod status and logs
5. Check DATABASE_URL and REDIS_URL connectivity

**Log Locations:**
- Vercel: https://vercel.com/dashboard → Deployments → View Logs
- Render: https://dashboard.render.com → Service → Logs
- RunPod: https://www.runpod.io/console/pods → Pod → Logs

---

## 🎯 Checklist

- [ ] RunPod proxy URL obtained and formatted
- [ ] Vercel environment variables configured
- [ ] Vercel redeployed successfully
- [ ] Render environment variables configured
- [ ] Render redeployed successfully
- [ ] Backend health check passes
- [ ] LLM health check passes
- [ ] Frontend loads without errors
- [ ] CORS configured correctly
- [ ] All 6 languages tested and working
- [ ] Health check script passes 100%
- [ ] Production URLs documented
- [ ] Team notified of deployment

**Estimated Time:** 1-2 hours  
**Difficulty:** Beginner-Intermediate  
**Prerequisites:** Access to Vercel, Render, and RunPod dashboards
