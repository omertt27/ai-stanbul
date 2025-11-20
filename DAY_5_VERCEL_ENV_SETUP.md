# 🔐 Day 5: Vercel Environment Variables Setup

**Date:** January 2025  
**Time Required:** 15 minutes  
**Status:** ⏳ Ready to Start

---

## 📋 Before You Begin

### Prerequisites Checklist
- [x] Vercel account created
- [x] ai-stanbul project imported to Vercel
- [x] Project settings configured (root: `frontend`)
- [ ] Environment variables added ← **YOU ARE HERE**

---

## 🎯 Goal

Add **23 environment variables** to your Vercel project so the frontend can:
- ✅ Connect to your backend API
- ✅ Use OpenStreetMap (100% free maps)
- ✅ Enable location tracking & routing
- ✅ Configure feature flags

---

## 📍 Step-by-Step Instructions

### Step 1: Access Environment Variables Settings

1. **Go to:** https://vercel.com/dashboard
2. **Click:** Your `ai-stanbul` project
3. **Click:** "Settings" tab (top navigation)
4. **Click:** "Environment Variables" (left sidebar)

✅ **You should see:** Empty list or "Add New" button

---

### Step 2: Add Environment Variables (One by One)

For **EACH** variable below:

1. **Click:** "Add New" or "Add Variable"
2. **Enter:** Variable Name (exactly as shown)
3. **Enter:** Value (exactly as shown)
4. **Select:** ✅ Production, ✅ Preview, ✅ Development (all 3)
5. **Click:** "Save"
6. **Repeat** for next variable

---

## 🔑 Environment Variables to Add

### Group 1: Core API Configuration (5 variables)

#### Variable 1
```
Name:  VITE_API_BASE_URL
Value: https://ai-stanbul.onrender.com
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 2
```
Name:  VITE_API_URL
Value: https://ai-stanbul.onrender.com
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 3
```
Name:  VITE_WEBSOCKET_URL
Value: wss://ai-stanbul.onrender.com
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 4
```
Name:  VITE_LOCATION_API_URL
Value: https://ai-stanbul.onrender.com
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 5
```
Name:  VITE_LOCATION_API_TIMEOUT
Value: 30000
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

---

### Group 2: Map Configuration - OpenStreetMap (6 variables)

#### Variable 6
```
Name:  VITE_MAP_PROVIDER
Value: openstreetmap
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 7
```
Name:  VITE_OSM_TILE_URL
Value: https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 8
```
Name:  VITE_DEFAULT_MAP_CENTER_LAT
Value: 41.0082
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 9
```
Name:  VITE_DEFAULT_MAP_CENTER_LNG
Value: 28.9784
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 10
```
Name:  VITE_DEFAULT_MAP_ZOOM
Value: 13
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 11
```
Name:  VITE_ENABLE_GOOGLE_MAPS
Value: false
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

---

### Group 3: Free Geocoding & Routing (4 variables)

#### Variable 12
```
Name:  VITE_GEOCODING_PROVIDER
Value: nominatim
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 13
```
Name:  VITE_NOMINATIM_URL
Value: https://nominatim.openstreetmap.org
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 14
```
Name:  VITE_ROUTING_PROVIDER
Value: osrm
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 15
```
Name:  VITE_OSRM_URL
Value: https://router.project-osrm.org
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

---

### Group 4: Feature Flags (4 variables)

#### Variable 16
```
Name:  VITE_ENABLE_LOCATION_TRACKING
Value: true
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 17
```
Name:  VITE_ENABLE_AB_TESTING
Value: true
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 18
```
Name:  VITE_ENABLE_FEEDBACK
Value: true
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 19
```
Name:  VITE_ENABLE_ANALYTICS
Value: true
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

---

### Group 5: Additional Configuration (4 variables)

#### Variable 20
```
Name:  VITE_CACHE_DURATION
Value: 3600
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 21
```
Name:  VITE_MAX_RETRIES
Value: 3
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 22
```
Name:  VITE_RETRY_DELAY
Value: 1000
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

#### Variable 23
```
Name:  VITE_ENABLE_DEBUG_MODE
Value: false
Envs:  ✅ Production  ✅ Preview  ✅ Development
```

---

## ✅ Verification Checklist

After adding all variables:

1. **Count:** You should see **23 variables** listed
2. **Check:** Each shows "Production, Preview, Development"
3. **Verify:** No typos in variable names (case-sensitive!)
4. **Confirm:** All URLs use `https://` (except WSS which uses `wss://`)

---

## 🎉 Completion Checklist

- [ ] All 23 variables added
- [ ] All variables show 3 environments
- [ ] No error messages in Vercel
- [ ] Variables saved successfully

**When complete:**
✅ **Move to Day 6:** Deploy Frontend to Vercel

---

## 📋 Quick Copy-Paste Reference

If Vercel allows bulk import (some plans do), use this:

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
VITE_CACHE_DURATION=3600
VITE_MAX_RETRIES=3
VITE_RETRY_DELAY=1000
VITE_ENABLE_DEBUG_MODE=false
```

**Note:** Most Vercel free plans require adding one-by-one through the UI.

---

## 🆘 Troubleshooting

### Issue: "Variable name already exists"
**Solution:** Skip that variable, it's already added

### Issue: "Invalid environment selection"
**Solution:** Make sure all 3 checkboxes are checked before saving

### Issue: "Value too long"
**Solution:** Check for extra spaces or line breaks in the value field

### Issue: Can't find Environment Variables section
**Solution:** Make sure you're in Settings → Environment Variables (left sidebar)

---

## 📊 Why These Variables?

| Variable Group | Purpose | Cost |
|---------------|---------|------|
| Core API | Connect frontend to backend | Free ✅ |
| OpenStreetMap | Display interactive maps | Free ✅ |
| Nominatim | Geocoding (address → coordinates) | Free ✅ |
| OSRM | Route calculation & navigation | Free ✅ |
| Feature Flags | Enable/disable features | Free ✅ |

**Total Cost:** $0/month 🎉

---

## ⏭️ Next Steps

After completing Day 5:

1. ✅ Verify all 23 variables are added
2. ⏭️ Proceed to **Day 6: Deploy Frontend**
3. 📝 Save your Vercel deployment URL for Day 7

**Estimated Time to Next Step:** 20 minutes

---

## 📚 Related Documentation

- [WEEK_2_DEPLOYMENT_WALKTHROUGH.md](./WEEK_2_DEPLOYMENT_WALKTHROUGH.md) - Full Week 2 guide
- [WEEK_2_PROGRESS_TRACKER.md](./WEEK_2_PROGRESS_TRACKER.md) - Track your progress
- [WEEK_2_COMMAND_REFERENCE.md](./WEEK_2_COMMAND_REFERENCE.md) - CLI commands
- [DAY_3_TESTING_REPORT.md](./DAY_3_TESTING_REPORT.md) - Backend verification

---

**Last Updated:** January 2025  
**Status:** Ready for implementation ✅
