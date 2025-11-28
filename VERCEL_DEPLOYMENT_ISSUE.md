# Vercel Deployment Issue - Troubleshooting Guide

**Issue**: Vercel shows old version (2.0.0) instead of latest (2.3.0)  
**Date**: November 28, 2024  
**Status**: 🔴 Needs manual intervention in Vercel Dashboard

## 🔍 Problem Analysis

### What We Found
- ✅ Local code has version **2.3.0** (just updated)
- ✅ Git commits are pushed to `main` branch
- ❌ Production shows version **2.0.0** (old)
- ❌ Blog API returns HTML instead of JSON (wrong routing)

### Root Cause
Vercel is likely:
1. Deploying from a cached build
2. Deploying from wrong branch/commit
3. Using an old Vercel project configuration
4. Not detecting the frontend directory changes

## 🛠️ Step-by-Step Fix

### Step 1: Check Vercel Dashboard
1. Go to https://vercel.com/dashboard
2. Find your project (likely named "ai-stanbul" or "aistanbul")
3. Click on **"Deployments"** tab
4. Check:
   - What branch is it deploying from?
   - What commit SHA is it using?
   - When was the last deployment?
   - Are there any failed builds?

### Step 2: Identify the Correct Project
You might have **multiple Vercel projects**:
- Frontend project (should deploy from `/frontend` directory)
- Backend project (should deploy from root with `production_server.py`)

**Action**: Check if you have 2 separate projects or 1 monorepo project

### Step 3: Force Redeploy from Vercel Dashboard

#### Option A: Manual Redeploy
1. Go to your project → **Deployments**
2. Find the latest commit (`5d680d8` - "Bump version to 2.3.0")
3. Click the **three dots (...)** menu
4. Select **"Redeploy"**
5. Check **"Use existing Build Cache"** is **UNCHECKED**
6. Click **"Redeploy"**

#### Option B: Delete and Recreate Deployment
1. Go to **Settings** → **Git**
2. Note your Git repository connection
3. Click **"Disconnect"**
4. Wait 10 seconds
5. Click **"Connect Git Repository"** again
6. Select your repo and branch (`main`)
7. Configure root directory:
   - For frontend: Set root directory to `frontend`
   - For backend: Leave empty (root)

### Step 4: Verify Build Settings

#### Frontend Project Settings
```json
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
Install Command: npm install
Framework Preset: Vite
Node Version: 18.x or 20.x
```

#### Backend Project Settings
```json
Root Directory: (leave empty or /)
Build Command: (automatic)
Framework Preset: Other
Python Version: 3.11
```

### Step 5: Check Environment Variables
Go to **Settings** → **Environment Variables**

Make sure these are set (if applicable):
- `VITE_API_BASE_URL` (for frontend)
- Any API keys or secrets

## 🎯 Quick Commands to Run NOW

### 1. Verify Your Commit is Pushed
```bash
cd /Users/omer/Desktop/ai-stanbul
git log --oneline -3
```
Expected output:
```
5d680d8 (HEAD -> main, origin/main) chore: Bump version to 2.3.0
8806537 Force Vercel redeploy - update all latest changes
a807580 ...
```

### 2. Check Remote Status
```bash
git status
git remote -v
```

### 3. Force Another Push (if needed)
```bash
# Add a comment to trigger rebuild
echo "# Vercel rebuild trigger" >> DEPLOYMENT_STATUS.md
git add DEPLOYMENT_STATUS.md
git commit -m "Force Vercel rebuild - manual trigger"
git push origin main --force-with-lease
```

## 🔴 Critical Checks in Vercel Dashboard

### Check 1: Project Root Directory
- **Frontend should point to**: `frontend/`
- **Backend should point to**: `/` (root)

If your Vercel project is set to deploy from **root** but expects **frontend**, that's the issue!

### Check 2: Build Logs
1. Click on the latest deployment
2. Click **"Building"** or **"Deployment"**
3. Check the logs for:
   - What files it's copying
   - What directory it's building from
   - Any cache hits
   - Build errors

### Check 3: Domain Configuration
Check which domain is connected to which project:
- `aistanbul.net` should point to → **Frontend project**
- API subdomain (if any) should point to → **Backend project**

## 📋 Expected Behavior After Fix

### Test 1: Service Worker Version
```bash
curl https://aistanbul.net/sw-enhanced.js | grep "version"
```
**Expected**: `@version 2.3.0` and `CACHE_VERSION = 'ai-istanbul-v2.3.0'`

### Test 2: Blog API
```bash
curl -I "https://aistanbul.net/api/blog/posts"
```
**Expected**: 
- Status: `200 OK` or `307 Redirect` (not 404)
- Content-Type: `application/json` (not HTML)

### Test 3: Full Blog API Response
```bash
curl "https://aistanbul.net/api/blog/posts?per_page=5" | head -50
```
**Expected**: JSON with `{"posts": [...], "total": X, "page": 1, ...}`

## 🚨 If Still Not Working

### Nuclear Option: Create New Vercel Project
1. Go to Vercel Dashboard
2. Click **"Add New..."** → **"Project"**
3. Import your Git repository
4. Configure:
   - **Project Name**: `aistanbul-frontend`
   - **Root Directory**: `frontend`
   - **Framework**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Click **"Deploy"**
6. Once deployed, update your domain settings

### Alternative: Check Vercel CLI
```bash
# Install Vercel CLI (if not installed)
npm install -g vercel

# Login
vercel login

# Check which project you're linked to
cd /Users/omer/Desktop/ai-stanbul/frontend
vercel ls

# Deploy manually
vercel --prod
```

## 📊 Current Commit Status

```
Latest commit: 5d680d8
Time: Just now
Branch: main
Message: "chore: Bump version to 2.3.0 and force fresh deployment"

Previous commit: 8806537
Time: ~10 minutes ago
Message: "Force Vercel redeploy - update all latest changes"

Commit with fixes: a807580
Time: 21 hours ago
Message: Contains all blog API and GPS fixes
```

## 🎯 Action Items (Priority Order)

1. **[NOW]** Check Vercel Dashboard → Deployments
2. **[NOW]** Verify which commit is being deployed
3. **[NOW]** Check if root directory is set correctly
4. **[NOW]** Force redeploy with cache disabled
5. **[WAIT]** Wait 3-5 minutes for deployment
6. **[TEST]** Run curl commands to verify version
7. **[IF FAILED]** Delete and recreate Vercel project

## 📞 Contact Points

- **Vercel Support**: https://vercel.com/support
- **Vercel Docs**: https://vercel.com/docs
- **GitHub Repo**: https://github.com/omertt27/ai-stanbul

---

**Next Step**: Go to Vercel Dashboard RIGHT NOW and check the deployment status!
