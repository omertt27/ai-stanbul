# 🚀 Week 2: Frontend Deployment Walkthrough
## Vercel Deployment - Step by Step Guide

**Date:** January 2025  
**Status:** Backend Deployed ✅ → Frontend Ready to Deploy  
**Time Required:** ~75 minutes total  
**Backend URL:** https://ai-stanbul.onrender.com/

---

## 📋 Pre-Deployment Checklist

### ✅ Already Complete
- [x] Backend deployed to Render
- [x] Backend health check passing
- [x] PostgreSQL database connected
- [x] Redis cache operational
- [x] API endpoints accessible
- [x] Backend URL: https://ai-stanbul.onrender.com/

### 🎯 Week 2 Goals
- [ ] Deploy frontend to Vercel
- [ ] Configure environment variables
- [ ] Connect frontend to backend
- [ ] Update CORS settings
- [ ] Verify full-stack integration

---

## 📅 Day 4: Vercel Account & Project Setup
**Time:** 30 minutes  
**Status:** ⏳ Pending

### Step 1: Create Vercel Account (5 min)

1. **Go to:** https://vercel.com
2. **Click:** "Sign Up"
3. **Choose:** "Continue with GitHub"
4. **Authorize:** Vercel to access your GitHub account
5. **Complete:** Profile setup

✅ **Success:** You're logged into Vercel dashboard

---

### Step 2: Import Your Project (10 min)

1. **Click:** "Add New..." → "Project" (top right)
2. **Search:** "ai-stanbul" in your repositories
3. **Click:** "Import" next to ai-stanbul repository

**Important:** If the repository is not visible:
- Click "Adjust GitHub App Permissions"
- Grant Vercel access to the repository
- Return and refresh

✅ **Success:** Project import screen opens

---

### Step 3: Configure Project Settings (15 min)

**Framework Preset:** Vite (should auto-detect)

**Root Directory:** 
```
frontend
```
**Important:** Click "Edit" and set to `frontend` folder

**Build Command:**
```bash
npm run build
```

**Output Directory:**
```
dist
```

**Install Command:**
```bash
npm install
```

**Node Version:** 18.x (default is fine)

---

### Step 4: Environment Variables Setup

**🚨 STOP! Do NOT click Deploy yet!**

Click "Environment Variables" section (expand if collapsed)

**We'll add these on Day 5** - Just note what's needed:
- VITE_API_BASE_URL
- VITE_API_URL
- VITE_LOCATION_API_URL
- VITE_MAP_PROVIDER
- Feature flags

✅ **Checkpoint:** Project configured but NOT deployed yet

---

## 📅 Day 5: Environment Variables Configuration
**Time:** 15 minutes  
**Status:** ⏳ Pending

### Environment Variables to Add

In the Vercel project settings (Environment Variables section), add each of these:

#### 1. Core API Configuration

**Variable:** `VITE_API_BASE_URL`  
**Value:** `https://ai-stanbul.onrender.com`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_API_URL`  
**Value:** `https://ai-stanbul.onrender.com`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_WEBSOCKET_URL`  
**Value:** `wss://ai-stanbul.onrender.com`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_LOCATION_API_URL`  
**Value:** `https://ai-stanbul.onrender.com`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_LOCATION_API_TIMEOUT`  
**Value:** `30000`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

---

#### 2. Map Configuration (100% Free - OpenStreetMap)

**Variable:** `VITE_MAP_PROVIDER`  
**Value:** `openstreetmap`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_OSM_TILE_URL`  
**Value:** `https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_DEFAULT_MAP_CENTER_LAT`  
**Value:** `41.0082`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_DEFAULT_MAP_CENTER_LNG`  
**Value:** `28.9784`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_DEFAULT_MAP_ZOOM`  
**Value:** `13`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_ENABLE_GOOGLE_MAPS`  
**Value:** `false`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

---

#### 3. Free Geocoding & Routing (No API Keys)

**Variable:** `VITE_GEOCODING_PROVIDER`  
**Value:** `nominatim`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_NOMINATIM_URL`  
**Value:** `https://nominatim.openstreetmap.org`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_ROUTING_PROVIDER`  
**Value:** `osrm`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_OSRM_URL`  
**Value:** `https://router.project-osrm.org`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

---

#### 4. Feature Flags

**Variable:** `VITE_ENABLE_LOCATION_TRACKING`  
**Value:** `true`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_ENABLE_AB_TESTING`  
**Value:** `true`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_ENABLE_FEEDBACK`  
**Value:** `true`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

**Variable:** `VITE_ENABLE_ANALYTICS`  
**Value:** `true`  
**Environments:** ✅ Production, ✅ Preview, ✅ Development

---

### Quick Copy-Paste Format (for Vercel UI)

```env
# Core API
VITE_API_BASE_URL=https://ai-stanbul.onrender.com
VITE_API_URL=https://ai-stanbul.onrender.com
VITE_WEBSOCKET_URL=wss://ai-stanbul.onrender.com
VITE_LOCATION_API_URL=https://ai-stanbul.onrender.com
VITE_LOCATION_API_TIMEOUT=30000

# Maps (100% Free)
VITE_MAP_PROVIDER=openstreetmap
VITE_OSM_TILE_URL=https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
VITE_DEFAULT_MAP_CENTER_LAT=41.0082
VITE_DEFAULT_MAP_CENTER_LNG=28.9784
VITE_DEFAULT_MAP_ZOOM=13
VITE_ENABLE_GOOGLE_MAPS=false

# Geocoding & Routing (Free)
VITE_GEOCODING_PROVIDER=nominatim
VITE_NOMINATIM_URL=https://nominatim.openstreetmap.org
VITE_ROUTING_PROVIDER=osrm
VITE_OSRM_URL=https://router.project-osrm.org

# Features
VITE_ENABLE_LOCATION_TRACKING=true
VITE_ENABLE_AB_TESTING=true
VITE_ENABLE_FEEDBACK=true
VITE_ENABLE_ANALYTICS=true
```

**Note:** Add each variable separately in Vercel UI, selecting all 3 environments for each

✅ **Checkpoint:** All 23 environment variables added

---

## 📅 Day 6: Deploy Frontend
**Time:** 20 minutes  
**Status:** ⏳ Pending

### Step 1: Trigger Deployment (2 min)

1. **Go to:** Vercel project page
2. **Click:** "Deploy" button (top right)
3. **Wait:** Build starts automatically

✅ **Success:** Build log appears

---

### Step 2: Monitor Build (5-10 min)

**Watch for these steps:**
1. ✅ Cloning repository
2. ✅ Installing dependencies (npm install)
3. ✅ Building application (npm run build)
4. ✅ Optimizing output
5. ✅ Deploying to Vercel Edge Network
6. ✅ Assigning domains

**Expected Duration:** 5-10 minutes

**Build Success Indicators:**
```
✓ Build completed successfully
✓ Deployment ready
✓ Assigned domains
```

---

### Step 3: Get Your Deployment URL (1 min)

Once deployed, you'll see:

**Production URL:** `https://ai-stanbul.vercel.app`  
(Or similar - Vercel auto-generates)

**Also available:**
- `https://ai-stanbul-[username].vercel.app`
- `https://[project-id].vercel.app`

**Copy this URL** - You'll need it for Day 7!

---

### Step 4: Initial Testing (5 min)

1. **Click** your deployment URL
2. **Check:** Homepage loads
3. **Check:** No console errors (F12)
4. **Check:** Styling looks correct
5. **Check:** Language selector visible

**Expected Issues at this stage:**
- ⚠️ Chat might not work yet (CORS not configured)
- ⚠️ API calls may fail (backend doesn't allow your domain yet)

**This is normal! We'll fix it on Day 7.**

✅ **Checkpoint:** Frontend deployed, URL copied

---

## 📅 Day 7: Connect Backend & Frontend
**Time:** 10 minutes  
**Status:** ⏳ Pending

### Step 1: Update Backend CORS (5 min)

1. **Go to:** https://dashboard.render.com
2. **Click:** Your backend service (ai-stanbul)
3. **Click:** "Environment" tab
4. **Find:** `ALLOWED_ORIGINS` variable
5. **Click:** Edit (pencil icon)

**Current Value:**
```json
["http://localhost:3000","http://localhost:5173"]
```

**New Value (add your Vercel URL):**
```json
["http://localhost:3000","http://localhost:5173","https://ai-stanbul.vercel.app","https://ai-stanbul-[username].vercel.app"]
```

**Important:** 
- Include the exact URL from Day 6
- Use HTTPS (not HTTP)
- Keep localhost URLs for local development
- No trailing slashes

6. **Click:** "Save Changes"

**Result:** Backend automatically redeploys (2-3 minutes)

---

### Step 2: Wait for Backend Redeploy (3 min)

Watch the "Events" tab on Render:
1. ✅ Build started
2. ✅ Build succeeded
3. ✅ Deploy live

**When you see:** "Deploy live" → Backend is ready!

---

### Step 3: Test Full-Stack Integration (2 min)

**Go to your Vercel URL:** `https://ai-stanbul.vercel.app`

**Open browser console:** F12 → Console tab

**Test chat:**
1. Type: "Hello, what's the weather in Istanbul?"
2. Click: Send
3. **Check:** Message sends successfully
4. **Check:** Bot responds (may be fallback response if LLM not configured)
5. **Check:** No CORS errors in console

**Expected Success:**
```
✅ Request sent to backend
✅ Response received
✅ No CORS errors
✅ Chat interface working
```

**If you see fallback responses:**
- This is expected if LLM is not yet configured
- The integration is working correctly
- You can configure LLM separately (see NEXT_STEPS_GUIDE.md)

---

### Step 4: Test All Features (5 min)

#### Language Switching
1. **Click:** Language dropdown
2. **Select:** Turkish (Türkçe)
3. **Type:** "Merhaba"
4. **Verify:** Response in Turkish

#### Map Integration
1. **Click:** "Show Map" or map icon
2. **Verify:** Map loads (OpenStreetMap)
3. **Verify:** Istanbul is centered

#### Multiple Messages
1. Send 3-4 different messages
2. Verify all responses come through
3. Check response times

---

## ✅ Week 2 Complete Checklist

Mark each when done:

### Day 4: Vercel Setup
- [ ] Vercel account created
- [ ] GitHub connected
- [ ] Project imported
- [ ] Build settings configured
- [ ] Root directory set to `frontend`

### Day 5: Environment Variables
- [ ] All 23 environment variables added
- [ ] All variables applied to Production
- [ ] All variables applied to Preview
- [ ] All variables applied to Development

### Day 6: Deployment
- [ ] First deployment triggered
- [ ] Build completed successfully
- [ ] Deployment URL received
- [ ] Homepage loads correctly
- [ ] No build errors

### Day 7: Integration
- [ ] CORS updated on backend
- [ ] Backend redeployed
- [ ] Chat functionality tested
- [ ] Multi-language tested
- [ ] Map functionality tested
- [ ] No console errors

---

## 🎉 Success Criteria

When Week 2 is complete, you should have:

1. ✅ **Backend Live:** https://ai-stanbul.onrender.com/
2. ✅ **Frontend Live:** https://ai-stanbul.vercel.app (your URL)
3. ✅ **Full Integration:** Frontend → Backend communication working
4. ✅ **No CORS Errors:** Console clean
5. ✅ **Features Working:** Chat, maps, multi-language
6. ✅ **Professional URLs:** Both HTTPS with SSL

---

## 🆘 Troubleshooting

### Build Fails on Vercel

**Error:** "Module not found"  
**Fix:** Check `package.json` has all dependencies
```bash
cd frontend
npm install
npm run build  # Test locally first
```

**Error:** "Out of memory"  
**Fix:** Vite config may need optimization
- Check `vite.config.js`
- Reduce bundle size
- Split large components

---

### CORS Errors

**Error:** "Access-Control-Allow-Origin"  
**Fix Checklist:**
1. Verify exact Vercel URL in CORS settings
2. Use HTTPS (not HTTP)
3. No trailing slash
4. Backend redeployed after CORS change
5. Clear browser cache (Ctrl+Shift+R)

**Test CORS:**
```bash
curl -H "Origin: https://ai-stanbul.vercel.app" \
  -H "Access-Control-Request-Method: POST" \
  -X OPTIONS \
  https://ai-stanbul.onrender.com/api/chat
```

Should return: `Access-Control-Allow-Origin: https://ai-stanbul.vercel.app`

---

### Environment Variables Not Working

**Symptoms:**
- Map not loading
- API calls failing
- Features disabled

**Fix:**
1. Go to Vercel project → Settings → Environment Variables
2. Verify all 23 variables are there
3. Check they're enabled for Production
4. Redeploy: Deployments → Latest → Redeploy

**Important:** Vite env vars must start with `VITE_`

---

### Chat Not Responding

**Check:**
1. ✅ Backend is live: https://ai-stanbul.onrender.com/health
2. ✅ CORS configured correctly
3. ✅ Environment variables set
4. ✅ Browser console for errors

**Test backend directly:**
```bash
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "language": "en"}'
```

Should return a response (may be fallback if LLM not configured)

---

## 📊 Verification Commands

### Test Backend Health
```bash
curl https://ai-stanbul.onrender.com/health
```

### Test Frontend Live
```bash
curl https://ai-stanbul.vercel.app
# Should return HTML
```

### Test CORS
```bash
curl -I -H "Origin: https://ai-stanbul.vercel.app" \
  https://ai-stanbul.onrender.com/api/chat
```

### Check Environment Variables (Local)
```bash
cd frontend
npm run build  # Should use VITE_ variables
```

---

## 🚀 Next Steps After Week 2

Once Week 2 is complete, proceed to:

### Week 3: Monitoring & Testing
- Set up monitoring (Grafana Cloud or self-hosted)
- Comprehensive testing (all 10 use cases)
- Load testing
- Security audit

### Optional: Configure LLM
- Add GROQ_API_KEY or OPENAI_API_KEY to Render
- Set PURE_LLM_MODE=true
- Redeploy backend
- Test AI-generated responses

See `NEXT_STEPS_GUIDE.md` for detailed instructions.

---

## 📚 Reference Documents

- **Backend Deployment:** VERCEL_RENDER_PRODUCTION_DEPLOYMENT.md
- **Day 2 Verification:** DAY_2_DEPLOYMENT_VERIFICATION.md
- **Day 3 Testing:** DAY_3_TESTING_REPORT.md
- **Environment Variables:** RENDER_ENV_VARS.md (backend)
- **Quick Reference:** WEEK_2_QUICK_GUIDE.md
- **Main Tracker:** IMPLEMENTATION_TRACKER.md

---

## ✅ Ready to Start?

1. **Start with Day 4:** Go to https://vercel.com
2. **Follow each step** in order
3. **Check off items** as you complete them
4. **Test thoroughly** at each stage
5. **Ask for help** if you get stuck

**Estimated Total Time:** 75 minutes  
**Result:** Production-ready full-stack application! 🚀

---

**Last Updated:** January 2025  
**Status:** Ready for execution  
**Backend:** ✅ Live  
**Frontend:** ⏳ Ready to deploy
