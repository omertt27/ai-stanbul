# OpenStreetMap (OSM) Integration Fix

## 🎯 Issue
OpenStreetMap tiles were being blocked by Content Security Policy (CSP), causing map rendering failures.

## 🐛 Error Messages
```
Connecting to 'https://a.tile.openstreetmap.org/12/2377/1535.png' violates the following Content Security Policy directive: "connect-src 'self' ...". The action has been blocked.

Fetch API cannot load https://a.tile.openstreetmap.org/12/2377/1535.png. Refused to connect because it violates the document's Content Security Policy.
```

## ✅ Solution
Updated Content Security Policy in `backend/core/middleware.py` to explicitly allow:

### 1. **OpenStreetMap Tile Servers**
Added to `connect-src` and `img-src` directives:
- `https://*.tile.openstreetmap.org` (wildcard for all subdomains)
- `https://tile.openstreetmap.org` (main domain)
- `https://a.tile.openstreetmap.org` (tile server A)
- `https://b.tile.openstreetmap.org` (tile server B)
- `https://c.tile.openstreetmap.org` (tile server C)

### 2. **Amplitude Analytics**
Added to `script-src` and `connect-src`:
- `https://cdn.amplitude.com` (analytics script CDN)

## 📝 Changes Made

**File**: `backend/core/middleware.py` (Lines 90-112)

### Before:
```python
csp_directives = [
    "default-src 'self'",
    "connect-src 'self' ... wss://*.vercel.live",  # OSM missing from connect-src
    "img-src 'self' https://images.unsplash.com ...",
    # Single long line - hard to read and maintain
]
```

### After:
```python
csp_directives = [
    "default-src 'self'",
    # Connect-src: APIs, WebSockets, OSM tiles, Analytics
    "connect-src 'self' https://ai-stanbul.onrender.com ... "
    "https://*.tile.openstreetmap.org https://tile.openstreetmap.org "
    "https://a.tile.openstreetmap.org https://b.tile.openstreetmap.org https://c.tile.openstreetmap.org "
    "... https://cdn.amplitude.com ...",
    # Img-src: Images and map tiles
    "img-src 'self' ... "
    "https://*.tile.openstreetmap.org https://tile.openstreetmap.org "
    "https://a.tile.openstreetmap.org https://b.tile.openstreetmap.org https://c.tile.openstreetmap.org "
    "data: blob:",
    # Script-src: Allow analytics and tracking scripts
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' ... "
    "https://cdn.amplitude.com ...",
]
```

## 🎯 Benefits

### 1. **Map Tiles Now Load**
- ✅ OpenStreetMap tiles can be fetched
- ✅ No more CSP violations
- ✅ Maps render correctly

### 2. **Analytics Working**
- ✅ Amplitude analytics script loads
- ✅ No console errors for tracking

### 3. **Better Code Organization**
- ✅ Multi-line CSP directives (easier to read)
- ✅ Comments explaining each section
- ✅ Grouped by purpose (APIs, tiles, analytics)

## 🔧 Technical Details

### CSP Directives Explained

#### `connect-src`
Controls which URLs can be loaded using:
- `fetch()`
- `XMLHttpRequest`
- WebSocket connections
- EventSource
- Service Worker fetch events

**Why OSM needs this**: Map tiles are loaded via `fetch()` API.

#### `img-src`
Controls which URLs can be used as image sources:
- `<img>` tags
- CSS `background-image`
- `<picture>` elements
- Favicon

**Why OSM needs this**: Tile images are rendered as `<img>` elements.

#### `script-src`
Controls which scripts can be executed:
- `<script>` tags
- `eval()`
- Inline event handlers

**Why Amplitude needs this**: Analytics script from CDN.

## 🚀 Deployment

### Backend Restart Required
The middleware changes require a backend restart to take effect:

```bash
# Development
cd backend
python main.py

# Production (Render)
# Auto-deploys on git push
git add backend/core/middleware.py
git commit -m "Fix OSM map tiles CSP"
git push origin main
```

### Verification

1. **Check Console**: No CSP errors
2. **Check Maps**: Tiles load correctly
3. **Check Network Tab**: OSM requests succeed (200 status)

## 📊 Expected Behavior

### Before Fix:
```
❌ CSP violation: tile.openstreetmap.org
❌ Map tiles don't load
❌ Console errors
❌ White/blank map tiles
```

### After Fix:
```
✅ No CSP violations
✅ Map tiles load from OSM
✅ Clean console
✅ Full map rendering
```

## 🔍 Testing

### Test OSM Integration:
1. Navigate to chat page
2. Ask: "Show me a map of Sultanahmet"
3. Verify:
   - Map renders
   - Tiles load
   - No console errors
   - Zoom/pan works

### Test Transportation RAG:
1. Ask: "How do I get from Kadıköy to Taksim?"
2. Verify:
   - Route shown on map
   - Markers for stations
   - No CSP errors
   - Interactive map

## 📚 Related Files

### Modified:
- ✅ `backend/core/middleware.py` (CSP configuration)

### Related (No Changes):
- `frontend/src/components/Map.jsx` (Map component)
- `frontend/src/services/mapService.js` (Map rendering)
- `backend/services/llm/context.py` (Transportation RAG)

## 🎓 Best Practices

### CSP Security Tips:

1. **Be Specific**: Use exact domains instead of wildcards when possible
2. **Avoid `unsafe-inline`**: We use it for analytics, but minimize usage
3. **Test Thoroughly**: Check all features after CSP changes
4. **Monitor Console**: Watch for new violations
5. **Document Changes**: Comment why each domain is allowed

### OSM Usage Guidelines:

1. **Tile Usage Policy**: OpenStreetMap is free but has usage limits
2. **Attribution**: Always credit OpenStreetMap
3. **Caching**: Cache tiles to reduce requests
4. **Fallback**: Have fallback for when OSM is down

## ✅ Status

**Fixed**: ✅ December 10, 2025  
**Tested**: ✅ CSP validation passed  
**Deployed**: 🚧 Requires backend restart  
**Monitoring**: 🔍 Check production logs  

---

## 📝 Additional Notes

### Why This Matters:
- Maps are core to transportation queries
- CSP violations break user experience
- Google Maps-level quality requires reliable maps
- Professional appearance needs working tiles

### Impact on Transportation RAG:
- ✅ Routes can be visualized
- ✅ Station markers show correctly
- ✅ Transfer points visible on map
- ✅ User location tracking works

### Future Improvements:
- [ ] Add tile caching in service worker
- [ ] Implement offline map support
- [ ] Add custom map styling
- [ ] Integrate with transportation routes
- [ ] Add real-time vehicle positions

---

**Last Updated**: December 10, 2025  
**Fix Type**: Security (CSP)  
**Priority**: High (Breaks maps)  
**Complexity**: Low (Configuration only)  
**Author**: AI Istanbul Team
