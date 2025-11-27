# CSP Errors Fix - Complete Guide

## Problem
Browser is showing CSP (Content Security Policy) errors even after updating `vercel.json`. The errors indicate that certain domains are blocked:
- `https://www.googletagmanager.com` 
- `https://vercel.live`
- Vercel feedback widget scripts

## Root Cause
1. **Browser cache**: Old CSP headers cached by browser
2. **Service worker cache**: Old service worker with cached responses
3. **CDN cache**: Vercel edge cache serving old headers

## Solutions Applied

### 1. Updated CSP in `vercel.json` ✅
Added all required domains to Content-Security-Policy:
- Google Tag Manager: `https://www.googletagmanager.com`
- Google Analytics domains
- Vercel Live domains: `https://vercel.live`, `https://*.vercel.live`
- Font domains: `https://fonts.googleapis.com`, `https://fonts.gstatic.com`

### 2. Bumped Service Worker Version ✅
Changed version from `2.0.0` to `2.1.0` to force cache refresh:
```javascript
const CACHE_VERSION = 'ai-istanbul-v2.1.0';
const MAP_TILES_CACHE = 'map-tiles-v2';
```

### 3. Clear Browser Cache (Manual Step)
Users need to:
1. **Hard refresh**: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows)
2. **Clear service worker**: 
   - Open DevTools (F12)
   - Go to Application tab
   - Click "Service Workers"
   - Click "Unregister"
   - Refresh page
3. **Clear site data**:
   - DevTools > Application > Storage
   - Click "Clear site data"
   - Refresh page

## How to Test After Fix

### 1. Check Console Logs
Open browser console and verify:
```
✅ No CSP errors
✅ Service Worker version: 2.1.0
✅ Google Analytics loading
✅ All scripts loading
```

### 2. Verify CSP Headers
In DevTools Network tab:
1. Click on main document request
2. Check Response Headers
3. Verify CSP includes all domains

### 3. Test Functionality
- [ ] Google Analytics tracking works
- [ ] Maps load correctly
- [ ] Fonts display properly
- [ ] No console errors

## Expected CSP Policy
```
default-src 'self';
script-src 'self' 'unsafe-inline' 'unsafe-eval' https://www.googletagmanager.com https://www.google-analytics.com https://maps.googleapis.com https://vercel.live https://*.vercel.live;
style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;
font-src 'self' https://fonts.gstatic.com data:;
img-src 'self' data: https: blob:;
connect-src 'self' https://ai-stanbul.onrender.com https://www.google-analytics.com https://www.googletagmanager.com https://maps.googleapis.com https://fonts.googleapis.com https://vercel.live https://*.vercel.live wss://vercel.live;
```

## Deployment Checklist
- [x] Updated `vercel.json` with complete CSP
- [x] Bumped service worker version to 2.1.0
- [x] Updated cache versions
- [ ] Commit and push changes
- [ ] Wait for Vercel deployment (2-3 min)
- [ ] Hard refresh browser
- [ ] Clear service worker cache
- [ ] Verify no CSP errors

## If Errors Persist

### Option 1: Disable Service Worker Temporarily
Add to your code:
```javascript
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(registrations => {
    registrations.forEach(registration => registration.unregister());
  });
}
```

### Option 2: Open in Incognito Mode
Test in private/incognito window to bypass all caches

### Option 3: Use Different Browser
Test in a browser where you haven't visited the site before

## Production URLs
- **Frontend**: https://ai-stanbul.vercel.app
- **Backend**: https://ai-stanbul.onrender.com

---
**Status**: ✅ Changes ready - needs deployment and cache clear
**Date**: November 27, 2025
