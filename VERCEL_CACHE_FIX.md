# Vercel Build Cache Issue - Fix Guide

## Problem
Vercel is serving an old cached build that still uses `/api/blog/posts/` instead of the new `/blog/` endpoint.

## Solution Options

### Option 1: Force Redeploy via Vercel Dashboard (Recommended)
1. Go to: https://vercel.com/dashboard
2. Find your project: `ai-stanbul`
3. Click on the latest deployment
4. Click **"Redeploy"** button
5. **IMPORTANT**: Check the box "Use existing Build Cache" and **UNCHECK** it
6. Click "Redeploy" to start a fresh build

### Option 2: Trigger New Deployment with Empty Commit
```bash
cd /Users/omer/Desktop/ai-stanbul
git commit --allow-empty -m "Force Vercel rebuild"
git push origin main
```

### Option 3: Add Vercel Build Configuration
Create or update `vercel.json` to disable build cache:

```json
{
  "buildCommand": "cd frontend && npm run build",
  "outputDirectory": "frontend/dist",
  "build": {
    "env": {
      "ENABLE_EXPERIMENTAL_COREPACK": "1"
    }
  },
  "cleanUrls": true,
  "trailingSlash": false
}
```

## Verify the Fix
After redeployment, check the console in production:
```
BLOG_API_URL: https://ai-stanbul.onrender.com/blog/
```

Should be `/blog/` NOT `/api/blog/posts/`

## Test Endpoints
- Blog list: https://ai-stanbul.onrender.com/blog/
- Blog post: https://ai-stanbul.onrender.com/blog/{post_id}
- Featured: https://ai-stanbul.onrender.com/blog/featured

## Current Status
- ✅ Code is correct in repository
- ✅ Changes committed and pushed
- ✅ Empty commit pushed to force rebuild
- 🔄 New Vercel deployment in progress
- ⏱️ Estimated completion: 2-3 minutes

## Deployment Timeline
- **49 minutes ago**: First deployment (cached old code)
- **Just now**: Empty commit pushed to trigger fresh build
- **In progress**: New deployment building...

---
**Next Steps**: 
1. Wait 2-3 minutes for deployment to complete
2. Hard refresh browser (Cmd+Shift+R)
3. Check console for correct API URL: `/blog/`
4. Test blog page functionality
