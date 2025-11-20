# 🚨 API PATH ISSUE - Quick Fix Required

**Issue:** Frontend API calls are using wrong paths causing 404 errors  
**Cause:** Double `/ai/ai/` in URL paths  
**Fix Time:** 5 minutes  
**Status:** CRITICAL - Blocking chat functionality

---

## 🔍 Problem Analysis

### Current Errors:
```javascript
❌ https://ai-stanbul.onrender.com/ai/ai/stream → 404
❌ https://ai-stanbul.onrender.com/ai/api/chat-sessions → 404
❌ /ai/ai/stream → Should be /ai/stream
❌ /ai/api/chat-sessions → Should be /api/chat-sessions
```

### Root Cause:
Your Vercel environment variable `VITE_API_URL` is set to:
```
VITE_API_URL=https://ai-stanbul.onrender.com/ai
```

But your frontend code is appending `/ai/stream` again, resulting in:
```
https://ai-stanbul.onrender.com/ai + /ai/stream = /ai/ai/stream ❌
```

---

## ✅ Solution: Fix Environment Variables

### Current (Wrong):
```env
VITE_API_URL=https://ai-stanbul.onrender.com/ai
VITE_API_BASE_URL=https://ai-stanbul.onrender.com/ai
```

### Should Be:
```env
VITE_API_URL=https://ai-stanbul.onrender.com
VITE_API_BASE_URL=https://ai-stanbul.onrender.com
```

**OR** if your backend expects `/ai` prefix:
```env
VITE_API_URL=https://ai-stanbul.onrender.com/ai
VITE_API_BASE_URL=https://ai-stanbul.onrender.com
```

---

## 🚀 Quick Fix (5 minutes)

### Step 1: Check Backend Routes

Let's verify what paths your backend actually accepts:

```bash
# Test the root
curl https://ai-stanbul.onrender.com/health

# Test with /ai prefix
curl https://ai-stanbul.onrender.com/ai/health

# Test API docs
curl https://ai-stanbul.onrender.com/docs
curl https://ai-stanbul.onrender.com/ai/docs
```

**Run these and tell me which ones return 200 OK!**

---

### Step 2: Fix Vercel Environment Variables

Based on backend responses, update Vercel:

#### Option A: If backend responds at root (`/`):

1. Go to https://vercel.com/dashboard
2. Your project → Settings → Environment Variables
3. Update these:

```env
VITE_API_URL=https://ai-stanbul.onrender.com
VITE_API_BASE_URL=https://ai-stanbul.onrender.com
VITE_LOCATION_API_URL=https://ai-stanbul.onrender.com/api
VITE_WEBSOCKET_URL=wss://ai-stanbul.onrender.com/ws
```

#### Option B: If backend responds at `/ai`:

```env
VITE_API_URL=https://ai-stanbul.onrender.com/ai
VITE_API_BASE_URL=https://ai-stanbul.onrender.com
VITE_LOCATION_API_URL=https://ai-stanbul.onrender.com/api
VITE_WEBSOCKET_URL=wss://ai-stanbul.onrender.com/ws
```

4. **Important:** Enable for all environments:
   - ✅ Production
   - ✅ Preview  
   - ✅ Development

5. Click "Save"

---

### Step 3: Redeploy Frontend

After updating env vars:

1. In Vercel dashboard
2. Go to "Deployments" tab
3. Click "..." menu on latest deployment
4. Click "Redeploy"
5. Wait 3-5 minutes for build

---

## 🧪 Test Backend Endpoints First

Before fixing frontend, let's verify backend paths:

```bash
# Test health endpoints
curl https://ai-stanbul.onrender.com/health
curl https://ai-stanbul.onrender.com/api/health

# Test chat endpoint
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "language": "en"}'

# Test streaming endpoint
curl https://ai-stanbul.onrender.com/ai/stream
```

**Expected:** One of these should return 200, others 404

---

## 📊 Understanding the Path Structure

### Your Frontend Code Probably Does:
```javascript
const apiUrl = import.meta.env.VITE_API_URL; // https://ai-stanbul.onrender.com/ai
const streamPath = '/ai/stream';
const fullUrl = apiUrl + streamPath; // ❌ Results in /ai/ai/stream
```

### Should Be:
```javascript
const apiUrl = import.meta.env.VITE_API_URL; // https://ai-stanbul.onrender.com
const streamPath = '/ai/stream';
const fullUrl = apiUrl + streamPath; // ✅ Results in /ai/stream
```

**OR:**
```javascript
const apiUrl = import.meta.env.VITE_API_URL; // https://ai-stanbul.onrender.com/ai
const streamPath = '/stream'; // No /ai prefix
const fullUrl = apiUrl + streamPath; // ✅ Results in /ai/stream
```

---

## 🔧 Quick Diagnostic Commands

Run these to understand your backend structure:

```bash
# 1. Check main endpoint
curl -I https://ai-stanbul.onrender.com/

# 2. Check /api prefix
curl -I https://ai-stanbul.onrender.com/api/

# 3. Check /ai prefix
curl -I https://ai-stanbul.onrender.com/ai/

# 4. Check health with different paths
curl https://ai-stanbul.onrender.com/health
curl https://ai-stanbul.onrender.com/api/health
curl https://ai-stanbul.onrender.com/ai/health

# 5. Check API docs
curl https://ai-stanbul.onrender.com/docs
```

**Copy the output and I'll tell you exactly what to set!**

---

## 🎯 Most Likely Solution

Based on FastAPI conventions, your backend probably has:

```
Root: https://ai-stanbul.onrender.com/
├── /health           → Health check
├── /docs            → API documentation
├── /api/
│   ├── /chat        → Chat endpoint
│   └── /chat-sessions → Session management
├── /ai/
│   └── /stream      → Streaming endpoint
└── /blog/
    └── /recommendations → Blog API
```

**So the correct Vercel env vars should be:**

```env
VITE_API_BASE_URL=https://ai-stanbul.onrender.com
VITE_API_URL=https://ai-stanbul.onrender.com
```

**Then your frontend should construct paths as:**
- Chat: `${VITE_API_URL}/api/chat`
- Stream: `${VITE_API_URL}/ai/stream`
- Sessions: `${VITE_API_URL}/api/chat-sessions`

---

## ✅ Quick Checklist

- [ ] Run diagnostic curl commands above
- [ ] Identify which path structure backend uses
- [ ] Update VITE_API_URL in Vercel (remove `/ai` suffix)
- [ ] Update VITE_API_BASE_URL in Vercel
- [ ] Save changes
- [ ] Redeploy frontend
- [ ] Wait 3-5 minutes
- [ ] Test chat functionality
- [ ] Verify no 404 errors in console

---

## 🆘 If Still Not Working

### Check Frontend Code:

Look for how API URLs are constructed in your frontend code:

```javascript
// frontend/src/services/api.ts (or similar)
// Look for lines like:
const baseUrl = import.meta.env.VITE_API_URL;
const endpoint = '/ai/stream'; // or '/stream'
```

**Key question:** Does your frontend code already include `/ai/` prefix in the path?
- If YES → Remove `/ai` from VITE_API_URL
- If NO → Keep `/ai` in VITE_API_URL but fix the paths

---

## 🔗 After Fix, Update CORS

Once paths are working, update CORS to allow your domain:

```json
["http://localhost:3000","http://localhost:5173","https://aistanbul.net","https://www.aistanbul.net"]
```

---

**ACTION REQUIRED NOW:**

1. **Run these commands and share output:**
   ```bash
   curl https://ai-stanbul.onrender.com/health
   curl https://ai-stanbul.onrender.com/api/health
   curl https://ai-stanbul.onrender.com/ai/health
   curl https://ai-stanbul.onrender.com/docs
   ```

2. **Then I'll tell you exact env var values to use!**

OR

**Quick Fix (Most Likely):**
- Go to Vercel → Environment Variables
- Change `VITE_API_URL` from `https://ai-stanbul.onrender.com/ai` to `https://ai-stanbul.onrender.com`
- Redeploy
- Test!

---

**This is the last blocker before your app is fully functional!** 🚀
