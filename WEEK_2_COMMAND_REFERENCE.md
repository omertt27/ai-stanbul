# 🚀 Week 2 Quick Command Reference

**Purpose:** Essential commands and URLs for Week 2 frontend deployment  
**Time:** 75 minutes total  
**Current Status:** Backend ✅ | Frontend ⏳

---

## 📋 Quick Info

### Current Status
- ✅ **Backend URL:** https://ai-stanbul.onrender.com/
- ⏳ **Frontend URL:** TBD after Day 6 deployment
- ✅ **Database:** PostgreSQL on Render (connected)
- ✅ **Cache:** Redis on Render (connected)

### Your Credentials
- **Vercel:** Sign up at https://vercel.com (use GitHub)
- **GitHub:** Your existing account
- **Render:** Already set up ✅

---

## 🔗 Essential URLs

### Day 4: Setup
```
Vercel Dashboard:     https://vercel.com/dashboard
GitHub Repository:    https://github.com/[your-username]/ai-stanbul
Project Root:         frontend/
```

### Day 5: No URLs needed (configuration only)

### Day 6: Your New URLs
```
Production:  https://ai-stanbul.vercel.app (or auto-generated)
Preview:     https://ai-stanbul-git-[branch].vercel.app
```

### Day 7: Update Backend
```
Render Dashboard:     https://dashboard.render.com
Backend Service:      ai-stanbul (your service name)
Environment Tab:      Click to edit ALLOWED_ORIGINS
```

---

## ⚡ Quick Test Commands

### Test Backend Health (Anytime)
```bash
curl https://ai-stanbul.onrender.com/health
```

**Expected:** `{"status":"healthy","version":"2.1.0",...}`

---

### Test Backend Chat (Anytime)
```bash
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Istanbul!", "language": "en"}'
```

**Expected:** JSON response with message (may be fallback)

---

### Test Frontend After Deployment (Day 6)
```bash
# Replace with your actual URL
curl https://ai-stanbul.vercel.app
```

**Expected:** HTML content (homepage)

---

### Test CORS After Day 7
```bash
# Replace with your actual Vercel URL
curl -H "Origin: https://ai-stanbul.vercel.app" \
  -H "Access-Control-Request-Method: POST" \
  -X OPTIONS \
  https://ai-stanbul.onrender.com/api/chat
```

**Expected:** Headers including `Access-Control-Allow-Origin`

---

## 🔧 Environment Variables Copy-Paste

### All 23 Variables for Vercel (Day 5)

**Copy these, add ONE BY ONE in Vercel dashboard:**

```env
VITE_API_BASE_URL=https://ai-stanbul.onrender.com
VITE_API_URL=https://ai-stanbul.onrender.com
VITE_WEBSOCKET_URL=wss://ai-stanbul.onrender.com
VITE_LOCATION_API_URL=https://ai-stanbul.onrender.com
VITE_LOCATION_API_TIMEOUT=30000
VITE_MAP_PROVIDER=openstreetmap
VITE_OSM_TILE_URL=https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
VITE_DEFAULT_MAP_CENTER_LAT=41.0082
VITE_DEFAULT_MAP_CENTER_LNG=28.9784
VITE_DEFAULT_MAP_ZOOM=13
VITE_ENABLE_GOOGLE_MAPS=false
VITE_GEOCODING_PROVIDER=nominatim
VITE_NOMINATIM_URL=https://nominatim.openstreetmap.org
VITE_ROUTING_PROVIDER=osrm
VITE_OSRM_URL=https://router.project-osrm.org
VITE_ENABLE_LOCATION_TRACKING=true
VITE_ENABLE_AB_TESTING=true
VITE_ENABLE_FEEDBACK=true
VITE_ENABLE_ANALYTICS=true
```

**For each variable:** Select ✅ Production, ✅ Preview, ✅ Development

---

## 🎯 Day-by-Day Checklist

### Day 4 (30 min) - Vercel Setup
```
✅ Go to https://vercel.com
✅ Sign up with GitHub
✅ Click "Add New" → "Project"
✅ Import "ai-stanbul"
✅ Set root directory: frontend
✅ Framework: Vite (auto-detected)
✅ Build command: npm run build
✅ Output: dist
✅ STOP - Don't deploy yet!
```

---

### Day 5 (15 min) - Environment Variables
```
✅ Open Vercel project settings
✅ Go to Environment Variables
✅ Add all 23 variables (see above)
✅ Apply to: Production, Preview, Development
✅ Verify all variables added
```

---

### Day 6 (20 min) - Deploy
```
✅ Click "Deploy" button
✅ Wait 5-10 minutes
✅ Watch build logs
✅ Get deployment URL
✅ Test homepage: https://[your-url].vercel.app
✅ Save URL for Day 7
✅ Note: Chat may not work yet (expected)
```

---

### Day 7 (10 min) - Connect
```
✅ Go to https://dashboard.render.com
✅ Click backend service
✅ Environment tab
✅ Edit ALLOWED_ORIGINS
✅ Add: "https://[your-vercel-url].vercel.app"
✅ Format: ["http://localhost:3000","https://your-url.vercel.app"]
✅ Save (auto-redeploys)
✅ Wait 2-3 minutes
✅ Test chat on Vercel site
✅ Verify no CORS errors
✅ WEEK 2 DONE! 🎉
```

---

## 🆘 Troubleshooting Quick Fixes

### Build Fails
```bash
# Test locally first
cd frontend
npm install
npm run build
# If successful locally, issue is with Vercel config
```

---

### CORS Error
```bash
# Verify format in Render
# Must be JSON array with strings
# Each URL must be exact (HTTPS, no trailing slash)
# Example:
["http://localhost:3000","https://ai-stanbul.vercel.app"]
```

---

### Environment Variables Not Working
```bash
# In Vercel:
# 1. Settings → Environment Variables
# 2. Verify all 23 are present
# 3. Check "Production" is enabled
# 4. Redeploy: Deployments → Redeploy
```

---

### Chat Not Responding
```bash
# Check each:
1. Backend health: curl https://ai-stanbul.onrender.com/health
2. Frontend console: F12 → Console tab (check errors)
3. CORS configured: See above
4. Variables set: See above
```

---

## 📊 Success Indicators

### After Day 4
- ✅ Vercel project exists
- ✅ Connected to GitHub
- ✅ Build settings configured
- ✅ Ready to deploy

### After Day 5
- ✅ 23 environment variables added
- ✅ All applied to 3 environments
- ✅ Ready to deploy

### After Day 6
- ✅ Build successful (green checkmark)
- ✅ Deployment URL available
- ✅ Homepage loads
- ✅ Styling correct

### After Day 7
- ✅ Chat works (end-to-end)
- ✅ No CORS errors
- ✅ Multi-language works
- ✅ Map loads
- ✅ Full-stack integration complete! 🎉

---

## 📱 Browser Tests (Day 7)

### Test in Frontend
```
1. Open: https://[your-url].vercel.app
2. Open console: F12
3. Type message: "Hello"
4. Send message
5. Check: Response received
6. Check: No errors in console
7. Switch language: Select "Türkçe"
8. Type: "Merhaba"
9. Check: Response in Turkish
10. Success! ✅
```

---

## 🔍 Verification Checklist

Before marking Week 2 complete:

```
✅ Backend live: https://ai-stanbul.onrender.com/health returns 200
✅ Frontend live: https://[your-url].vercel.app loads
✅ Chat works: Can send and receive messages
✅ No CORS errors: Browser console clean
✅ Multi-language: Can switch languages
✅ Map loads: OpenStreetMap visible
✅ SSL working: Both URLs use HTTPS
✅ Both URLs saved: Documented for future use
```

---

## 📚 Full Documentation

For detailed explanations, see:
- **WEEK_2_DEPLOYMENT_WALKTHROUGH.md** - Complete step-by-step guide
- **IMPLEMENTATION_TRACKER.md** - Overall progress tracker
- **DAY_2_DEPLOYMENT_VERIFICATION.md** - Backend verification
- **NEXT_STEPS_GUIDE.md** - What to do after Week 2

---

## ⏱️ Time Estimate

```
Day 4: Vercel Setup           → 30 minutes
Day 5: Environment Variables  → 15 minutes
Day 6: Deployment             → 20 minutes
Day 7: CORS & Integration     → 10 minutes
───────────────────────────────────────────
Total:                          75 minutes
```

---

## 🎯 Your Next Action

**START HERE:**

1. **Go to:** https://vercel.com
2. **Sign up** with GitHub
3. **Follow:** WEEK_2_DEPLOYMENT_WALKTHROUGH.md
4. **Reference:** This file for quick commands

**Ready? Let's deploy! 🚀**

---

**Last Updated:** January 2025  
**Status:** Ready for Week 2 execution  
**Backend:** ✅ Live and healthy  
**Frontend:** ⏳ Ready to deploy
