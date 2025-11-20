# ✅ DEFINITIVE FIX - API Path Configuration

**Issue:** Frontend using wrong API paths → 404 errors  
**Root Cause:** `VITE_API_URL` has `/ai` suffix it shouldn't have  
**Solution:** Remove `/ai` from environment variable  
**Fix Time:** 5 minutes

---

## 🎯 The Problem (Confirmed!)

### Your Backend Structure (Verified):
```
✅ Root:        https://ai-stanbul.onrender.com/
✅ Health:      https://ai-stanbul.onrender.com/api/health
✅ Docs:        https://ai-stanbul.onrender.com/docs
✅ Chat:        https://ai-stanbul.onrender.com/api/chat
✅ Stream:      https://ai-stanbul.onrender.com/ai/stream
❌ /health:     404 (doesn't exist at root)
❌ /ai/health:  404 (doesn't exist)
```

### Current Vercel Config (WRONG):
```env
VITE_API_URL=https://ai-stanbul.onrender.com/ai  ❌
```

### When Frontend Constructs URL:
```javascript
baseUrl = "https://ai-stanbul.onrender.com/ai"
path = "/ai/stream"
result = "https://ai-stanbul.onrender.com/ai/ai/stream"  ❌ 404!
```

---

## ✅ THE FIX (Do This Now!)

### Step 1: Update Vercel Environment Variables (3 min)

1. **Go to Vercel Dashboard:**
   ```
   https://vercel.com/dashboard
   ```

2. **Navigate to your project:**
   - Click `ai-stanbul` (or your project name)
   - Click "Settings" tab
   - Click "Environment Variables"

3. **Find and Update These Variables:**

   **Change FROM:**
   ```env
   VITE_API_URL=https://ai-stanbul.onrender.com/ai
   VITE_API_BASE_URL=https://ai-stanbul.onrender.com/ai
   ```

   **Change TO:**
   ```env
   VITE_API_URL=https://ai-stanbul.onrender.com
   VITE_API_BASE_URL=https://ai-stanbul.onrender.com
   ```

   **Keep These As-Is:**
   ```env
   VITE_LOCATION_API_URL=https://ai-stanbul.onrender.com/api
   VITE_WEBSOCKET_URL=wss://ai-stanbul.onrender.com/ws
   VITE_BLOG_API_URL=https://ai-stanbul.onrender.com/blog/
   ```

4. **Ensure all are enabled for:**
   - ✅ Production
   - ✅ Preview
   - ✅ Development

5. **Click "Save"**

---

### Step 2: Redeploy Frontend (2 min)

1. **In Vercel Dashboard:**
   - Go to "Deployments" tab
   - Find your latest deployment
   - Click "..." (three dots menu)
   - Click "Redeploy"

2. **Wait for Build:**
   - Build time: ~3-5 minutes
   - Watch build logs
   - Wait for "Deployment Complete"

---

### Step 3: Test the Fix (1 min)

1. **Open your site:**
   ```
   https://aistanbul.net
   ```

2. **Open browser console (F12)**

3. **Go to Chat page**

4. **Send a test message**

5. **Check console:**
   ```javascript
   ✅ Should see: https://ai-stanbul.onrender.com/ai/stream
   ❌ Should NOT see: https://ai-stanbul.onrender.com/ai/ai/stream
   ```

---

## 📊 Before vs After

### Before (BROKEN):
```
Frontend sends:
❌ https://ai-stanbul.onrender.com/ai/ai/stream → 404
❌ https://ai-stanbul.onrender.com/ai/api/chat → 404
❌ https://ai-stanbul.onrender.com/ai/api/chat-sessions → 404
```

### After (WORKING):
```
Frontend sends:
✅ https://ai-stanbul.onrender.com/ai/stream → 200
✅ https://ai-stanbul.onrender.com/api/chat → 200
✅ https://ai-stanbul.onrender.com/api/chat-sessions → 200
```

---

## 🧪 Verification Commands

After redeployment, test with these:

```bash
# Test what frontend will now call
curl https://ai-stanbul.onrender.com/api/health
# Expected: {"status":"healthy",...}

curl https://ai-stanbul.onrender.com/ai/stream
# Expected: Some response (not 404)

curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"test","language":"en"}'
# Expected: Chat response (or error about LLM config, but not 404)
```

---

## ✅ Complete Environment Variable Reference

**After the fix, your Vercel env vars should be:**

```env
# Core API URLs (FIXED - removed /ai suffix)
VITE_API_URL=https://ai-stanbul.onrender.com
VITE_API_BASE_URL=https://ai-stanbul.onrender.com

# API Endpoints (keep as-is)
VITE_LOCATION_API_URL=https://ai-stanbul.onrender.com/api
VITE_LOCATION_API_TIMEOUT=10000

# WebSocket
VITE_WEBSOCKET_URL=wss://ai-stanbul.onrender.com/ws

# Blog API
VITE_BLOG_API_URL=https://ai-stanbul.onrender.com/blog/

# Map Configuration (keep as-is)
VITE_MAP_PROVIDER=openstreetmap
VITE_OSM_TILE_URL=https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
VITE_DEFAULT_MAP_CENTER_LAT=41.0082
VITE_DEFAULT_MAP_CENTER_LNG=28.9784
VITE_DEFAULT_MAP_ZOOM=12
VITE_ENABLE_GOOGLE_MAPS=false

# Geocoding (keep as-is)
VITE_GEOCODING_PROVIDER=nominatim
VITE_NOMINATIM_URL=https://nominatim.openstreetmap.org

# Routing (keep as-is)
VITE_ROUTING_PROVIDER=osrm
VITE_OSRM_URL=https://router.project-osrm.org

# Feature Flags (keep as-is)
VITE_ENABLE_LOCATION_TRACKING=true
VITE_ENABLE_AB_TESTING=true
VITE_ENABLE_FEEDBACK=true
VITE_ENABLE_ANALYTICS=true

# Additional Config (keep as-is)
VITE_CACHE_DURATION=300000
VITE_MAX_RETRIES=2
VITE_RETRY_DELAY=1000
VITE_ENABLE_DEBUG_MODE=false
```

---

## 🎯 Success Criteria

After the fix, you should see:

### In Browser Console:
```javascript
✅ API Configuration loaded
✅ Starting streaming request to: https://ai-stanbul.onrender.com/ai/stream
✅ Request to /api/chat-sessions
✅ No 404 errors on API calls
⚠️ May still see LLM errors (need API key - separate issue)
```

### In Network Tab (F12 → Network):
```
✅ /api/health → 200 OK
✅ /ai/stream → 200 or 500 (not 404)
✅ /api/chat → 200 or 500 (not 404)
✅ /api/chat-sessions → 200 or 404 is OK if feature not used
```

---

## 🚨 Common Mistakes to Avoid

1. **Don't add `/ai` back to VITE_API_URL**
   - ❌ `VITE_API_URL=https://ai-stanbul.onrender.com/ai`
   - ✅ `VITE_API_URL=https://ai-stanbul.onrender.com`

2. **Don't forget to redeploy**
   - Changing env vars doesn't auto-redeploy
   - You MUST manually redeploy for changes to take effect

3. **Don't edit only one environment**
   - Make sure Production, Preview, and Development all have the fix

---

## 🔄 After This is Working

Once 404s are fixed, you may see other errors. That's GOOD! It means the path is correct:

### Expected Next Issues (All OK):
```
⚠️ "LLM not configured" → Need to add GROQ_API_KEY or OPENAI_API_KEY
⚠️ "CORS error" → Need to add aistanbul.net to ALLOWED_ORIGINS
⚠️ "Timeout" → Backend may be slow/cold-starting
```

### NOT Expected (Would be problems):
```
❌ Still getting 404 on /ai/ai/stream → Env vars not updated correctly
❌ Still getting 404 on /api/chat → Wrong VITE_API_URL
```

---

## 📋 Quick Checklist

- [ ] Go to Vercel Dashboard
- [ ] Settings → Environment Variables
- [ ] Change `VITE_API_URL` to `https://ai-stanbul.onrender.com` (remove `/ai`)
- [ ] Change `VITE_API_BASE_URL` to `https://ai-stanbul.onrender.com` (remove `/ai`)
- [ ] Verify enabled for Production, Preview, Development
- [ ] Save changes
- [ ] Go to Deployments tab
- [ ] Redeploy latest deployment
- [ ] Wait 3-5 minutes for build
- [ ] Test on https://aistanbul.net
- [ ] Open console, try chat
- [ ] Verify no 404 errors
- [ ] ✅ FIXED!

---

**TIME ESTIMATE:**
- Update env vars: 2 minutes
- Redeploy: 3-5 minutes
- Test: 1 minute
- **Total: ~8 minutes to fully working chat!**

---

**ACTION: Go to Vercel NOW and make these changes!**

🎯 **After this fix, your chat will work (or show different, solvable errors)!**
