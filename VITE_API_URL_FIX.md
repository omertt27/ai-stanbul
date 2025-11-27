# 🔍 VERCEL ENVIRONMENT VARIABLE VERIFICATION

## ❌ Current Issue

Frontend is still trying to call `/chat` on Vercel (404) instead of the backend.

This means `VITE_API_URL` wasn't available during the build.

---

## ✅ SOLUTION: Verify and Re-add Environment Variable

### Step 1: Double-Check Environment Variable (1 minute)

1. **Go to Vercel Dashboard:** https://vercel.com/dashboard
2. **Click:** ai-stanbul project
3. **Settings → Environment Variables**
4. **Verify you see:**
   ```
   VITE_API_URL = https://ai-stanbul.onrender.com
   ✓ Production
   ✓ Preview  
   ✓ Development
   ```

**IF IT'S NOT THERE** or **IF ONLY ONE ENVIRONMENT IS CHECKED:**
- Click **Add New Variable**
- Key: `VITE_API_URL`
- Value: `https://ai-stanbul.onrender.com`
- Check **ALL THREE**: Production, Preview, Development
- Click **Save**

---

### Step 2: Trigger New Deployment (1 minute)

After adding/verifying the variable:

**Option A - Redeploy from Dashboard:**
1. Go to **Deployments** tab
2. Click **⋯** on the latest deployment
3. Click **Redeploy**
4. **IMPORTANT:** Make sure you see "Building..." status

**Option B - Force new deployment via Git:**
```bash
cd /Users/omer/Desktop/ai-stanbul
git commit --allow-empty -m "Force rebuild with VITE_API_URL"
git push origin main
```

---

### Step 3: Verify Build Logs (Important!)

While the build is running:

1. **Click on the building deployment**
2. **Open "Build Logs"** tab
3. **Search for:** `VITE_API_URL`
4. **You should see:**
   ```
   vite v5.x.x building for production...
   ✓ VITE_API_URL is set
   ```

**If you DON'T see the variable mentioned:**
- The environment variable wasn't available during build
- Go back and re-add it, making sure ALL environments are checked

---

### Step 4: Test After Successful Build (30 seconds)

Once build shows "Ready":

1. **Wait 30 seconds** for CDN to update
2. **Open in INCOGNITO window:** https://ai-stanbul.vercel.app
3. **Open DevTools Console** (F12)
4. **Look for the API Configuration log:**
   ```javascript
   API Configuration: {
     BASE_URL: "https://ai-stanbul.onrender.com",  // Should be Render, NOT localhost!
     API_URL: "https://ai-stanbul.onrender.com/api/chat"
   }
   ```

5. **If you see `localhost:8001` instead:**
   - Environment variable still not working
   - Try Option B (force new deployment via Git)

---

## 🎯 Quick Diagnostic

**Run this in your terminal to check if env var is in the built files:**

```bash
# This will show you what API URL is baked into the production build
curl -s https://ai-stanbul.vercel.app/assets/index-*.js | grep -o "localhost\|ai-stanbul.onrender.com" | head -1
```

**Expected output:** `ai-stanbul.onrender.com`  
**If you see:** `localhost` → Environment variable wasn't used in build

---

## 🔧 Alternative Solution: Hardcode for Now

If environment variables keep failing, we can temporarily hardcode the API URL:

**Edit `/Users/omer/Desktop/ai-stanbul/frontend/src/api/api.js`:**

Change line 12 from:
```javascript
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
```

To:
```javascript
const BASE_URL = import.meta.env.VITE_API_URL || 'https://ai-stanbul.onrender.com';
```

Then commit and push:
```bash
cd /Users/omer/Desktop/ai-stanbul
git add frontend/src/api/api.js
git commit -m "Fix: Hardcode production API URL as fallback"
git push origin main
```

This way, even if VITE_API_URL isn't set, it will use the production backend.

---

## 📋 Verification Checklist

- [ ] VITE_API_URL exists in Vercel environment variables
- [ ] All three environments (Production, Preview, Development) are checked
- [ ] New deployment triggered after adding variable
- [ ] Build logs show environment variable
- [ ] Deployment status shows "Ready"
- [ ] Browser console shows correct API URL (not localhost)
- [ ] Network tab shows requests going to Render backend
- [ ] No 404 errors on /chat

---

## 🎯 What Should Happen

**Correct Flow:**
```
User types message
  ↓
Frontend: https://ai-stanbul.vercel.app
  ↓
Sends POST request to: https://ai-stanbul.onrender.com/api/chat
  ↓
Backend processes request
  ↓
Returns restaurant recommendations
  ↓
Frontend displays response
```

**Current (Wrong) Flow:**
```
User types message
  ↓
Frontend: https://ai-stanbul.vercel.app
  ↓
Sends POST request to: /chat (relative URL on Vercel)
  ↓
Vercel returns 404 (no such route exists)
```

---

## ⚡ QUICK FIX NOW

**Do this right now:**

1. Verify `VITE_API_URL` is in Vercel Settings → Environment Variables
2. If it's there, trigger a new deployment
3. If it's NOT there or only one environment is checked, add it with ALL THREE checked
4. Wait for "Ready" status
5. Test in incognito window

**Let me know what the browser console shows after the new deployment!**
