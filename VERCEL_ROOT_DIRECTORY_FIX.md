# 🔧 VERCEL FRONTEND CONFIGURATION FIX

## ❌ Current Problem

Vercel build is still failing with peer dependency error even though we added `.npmrc`.

**Root Cause:** Vercel doesn't know where the frontend code is located (it's in `/frontend` subdirectory).

---

## ✅ SOLUTION: Configure Vercel Project Settings

### Step 1: Update Vercel Project Settings (2 minutes)

1. **Go to Vercel Dashboard:**
   - URL: https://vercel.com/dashboard
   - Click on **ai-stanbul** project

2. **Go to Settings:**
   - Click **Settings** tab in top navigation

3. **Update Root Directory:**
   - Find **"Build & Development Settings"** section
   - Look for **"Root Directory"** setting
   - Click **Edit**
   - Enter: `frontend`
   - Click **Save**

4. **Verify Install Command:**
   - In same section, find **"Install Command"**
   - Should be: `npm install --legacy-peer-deps` (if not, click Edit and set it)
   - Click **Save**

5. **Verify Build Command:**
   - Find **"Build Command"**
   - Should be: `npm run build`
   - Click **Save**

6. **Verify Output Directory:**
   - Find **"Output Directory"**
   - Should be: `dist`
   - Click **Save**

---

### Step 2: Redeploy (1 minute)

After saving settings:

1. **Go to Deployments tab**
2. Click **⋯** on latest deployment
3. Click **Redeploy**
4. Wait for build (~2 minutes)

---

## 📋 Expected Configuration

Your Vercel project should have these settings:

```
Framework Preset: Vite
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
Install Command: npm install --legacy-peer-deps
Node Version: 18.x (or 20.x)
```

---

## 🎯 Alternative: Use vercel.json at Root

If Vercel dashboard settings don't work, you can also configure via `vercel.json`:

**Note:** You already have a `vercel.json` for backend. We need to replace it or update Vercel settings manually.

---

## ✅ After Configuration

Once properly configured:

1. **Vercel will:**
   - Change directory to `frontend/`
   - Run `npm install --legacy-peer-deps`
   - Build the Vite app
   - Deploy the `dist` folder

2. **Build will succeed** because `.npmrc` will be used

3. **Frontend will be live** with correct API endpoints

---

## 🔍 Verify It Worked

After successful deployment:

1. **Check Build Logs:**
   - Should see: "Root Directory: frontend"
   - Should see: "npm install --legacy-peer-deps"
   - Should see: "Build succeeded"

2. **Test Frontend:**
   - Open: https://ai-stanbul.vercel.app
   - Hard refresh (Cmd+Shift+R)
   - Open DevTools console
   - Should see correct API URL
   - Chat should work with no 404 errors

---

## 🆘 If Still Failing

**Option: Deploy frontend from /frontend directory directly**

1. In Vercel dashboard, disconnect current project
2. Import new project
3. Select **frontend** folder as root
4. Deploy

This ensures Vercel treats the frontend folder as the project root.

---

## 📊 Summary

**The key issue:** Vercel is trying to build from project root (`/Users/omer/Desktop/ai-stanbul`) instead of `/Users/omer/Desktop/ai-stanbul/frontend`.

**The fix:** Tell Vercel the Root Directory is `frontend` in project settings.

**GO TO VERCEL SETTINGS NOW AND UPDATE ROOT DIRECTORY! 🚀**
