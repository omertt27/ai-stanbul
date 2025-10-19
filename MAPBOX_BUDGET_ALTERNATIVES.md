# Mapbox Free Tier Optimization Guide

## 🎓 Student Budget Strategy for 10k+ Users

### Option 1A: Use Mapbox Free Tier Wisely

**Free Tier Limits:**
- 50,000 map loads/month
- 100,000 geocoding requests/month
- Unlimited static maps

**Optimization Strategies:**

1. **Lazy Loading** - Only load map when user requests navigation
2. **Static Maps** - Use static images for previews (unlimited!)
3. **Caching** - Cache map tiles aggressively
4. **Session Reuse** - Don't reload map on every navigation

**Implementation:**

```jsx
// Lazy load map only when needed
const [mapEnabled, setMapEnabled] = useState(false);

// User clicks "Show on Map" button
<button onClick={() => setMapEnabled(true)}>
  Show on Map
</button>

{mapEnabled && (
  <MapboxNavigationMap {...props} />
)}
```

**Expected Usage:**
- 10k users/month
- 30% use map feature = 3k users
- Each user loads map 3 times = 9k loads/month
- **Well within 50k free limit!** ✅

---

## Option 1B: Mapbox Education Program

**Apply for Student/Education Discount:**

- Mapbox offers educational discounts
- Up to 100% discount for student projects
- Application: https://www.mapbox.com/community/education

**Requirements:**
- Valid student email (.edu)
- Academic project description
- Non-commercial use

**Apply here:**
```
https://www.mapbox.com/community/education
```

---

## Option 2: Open-Source Alternative - MapLibre GL JS

**FREE forever, no limits!**

MapLibre is a fork of Mapbox GL JS (before it went proprietary).

**Benefits:**
- 100% free and open-source
- Same API as Mapbox GL JS
- No token required
- Unlimited users
- Self-hosted tiles

**Implementation:**

```bash
# Replace Mapbox with MapLibre
npm uninstall mapbox-gl
npm install maplibre-gl
```

**Code changes (minimal):**

```jsx
// Before (Mapbox)
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
mapboxgl.accessToken = 'your_token';

// After (MapLibre) - NO TOKEN NEEDED!
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json', // Free style
  center: [28.9784, 41.0082],
  zoom: 12
});
```

**Free Tile Sources:**
- OpenMapTiles: https://openmaptiles.org/
- Maptiler (free tier): https://www.maptiler.com/
- OpenStreetMap: https://www.openstreetmap.org/

**Cost:** $0/month for unlimited users! 🎉

---

## Option 3: Leaflet.js (Simplest, Free Forever)

**Benefits:**
- Completely free
- No 3D buildings (but simpler)
- Lightweight
- You already have it installed!

**You already have this working:**
- `frontend/src/components/MapView.jsx` uses Leaflet
- Just enhance it with routing visualization

**Update existing Leaflet component:**

```jsx
// Add route line to existing MapView.jsx
import { Polyline } from 'react-leaflet';

function MapView({ locations, route }) {
  return (
    <MapContainer>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      
      {/* Add route line */}
      {route && (
        <Polyline
          positions={route.coordinates}
          color="blue"
          weight={5}
        />
      )}
      
      {/* Existing markers */}
      {locations.map((loc) => (
        <Marker position={[loc.lat, loc.lng]} />
      ))}
    </MapContainer>
  );
}
```

**Cost:** $0/month ✅

---

## Option 4: Hybrid Approach (Best of Both Worlds)

**Strategy:**
1. Use **Leaflet** for basic map display (free)
2. Use **Mapbox** only for premium features (3D buildings)
3. Let users choose view mode

**Implementation:**

```jsx
function NavigationMap({ route, pois, userLocation }) {
  const [use3D, setUse3D] = useState(false);
  
  return (
    <div>
      <button onClick={() => setUse3D(!use3D)}>
        {use3D ? '2D View' : '3D View'}
      </button>
      
      {use3D ? (
        <MapboxNavigationMap {...props} />  // Counts toward limit
      ) : (
        <LeafletMapView {...props} />       // FREE!
      )}
    </div>
  );
}
```

**Expected Usage:**
- 90% of users use free Leaflet (0 cost)
- 10% enable 3D Mapbox view
- Well within free tier!

---

## Option 5: Self-Hosted Map Tiles

**Ultimate Free Solution:**

1. **Download Istanbul OSM data** (~200MB)
2. **Host your own tile server** (Docker)
3. **No external dependencies**
4. **Unlimited users**

**Setup:**

```bash
# 1. Download Istanbul area
wget https://download.geofabrik.de/europe/turkey-latest.osm.pbf

# 2. Run tile server
docker run -d -p 8080:80 \
  -v $(pwd)/data:/data \
  overv/openstreetmap-tile-server \
  import
```

**Use in app:**

```jsx
<MapContainer>
  <TileLayer
    url="http://localhost:8080/tile/{z}/{x}/{y}.png"
    attribution="© OpenStreetMap contributors"
  />
</MapContainer>
```

**Pros:**
- $0/month
- Unlimited users
- Full control
- Fast (local)

**Cons:**
- Requires server (~$5/month VPS)
- Initial setup complexity
- No 3D buildings

**Net Cost:** ~$5/month VPS (DigitalOcean, Linode)

---

## Recommended Solution for Your Case

### 🎯 Best Strategy: Progressive Enhancement

**Phase 1: MVP (Free)**
```
Use Leaflet.js (already installed)
↓
Add OSRM routing (already have backend)
↓
Display routes on 2D map
↓
Cost: $0/month
```

**Phase 2: Polish (Still Free)**
```
Apply for Mapbox Education discount
↓
Add 3D view as optional feature
↓
Keep Leaflet as default
↓
Cost: $0/month (education discount)
```

**Phase 3: Scale (Minimal Cost)**
```
If >50k map loads/month
↓
Switch to MapLibre GL JS (free fork)
↓
Or: Self-host tiles ($5/month VPS)
↓
Cost: $0-5/month
```

---

## Implementation Plan

### Step 1: Update to Use Leaflet (Already Have It!)

```bash
# You already have Leaflet installed!
# Just enhance the existing MapView.jsx
```

### Step 2: Add Route Rendering to Leaflet

```jsx
// Update frontend/src/components/MapView.jsx
import { Polyline, Marker, Popup } from 'react-leaflet';

function MapView({ locations, routeData }) {
  return (
    <MapContainer center={[41.0082, 28.9784]} zoom={13}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; OpenStreetMap contributors'
      />
      
      {/* Route line */}
      {routeData?.geometry && (
        <Polyline
          positions={routeData.geometry.coordinates.map(c => [c[1], c[0]])}
          color="blue"
          weight={5}
          opacity={0.7}
        />
      )}
      
      {/* POI markers */}
      {locations?.map((loc, idx) => (
        <Marker key={idx} position={[loc.lat, loc.lng]}>
          <Popup>{loc.name}</Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
```

**Cost: $0** ✅

### Step 3: Apply for Mapbox Education (Optional)

1. Go to: https://www.mapbox.com/community/education
2. Fill out form with:
   - Student email
   - Project: "AI Istanbul Travel Assistant - Educational Project"
   - Description: "Navigation system for tourists, part of university project"
3. Get approved (usually 1-2 weeks)
4. Get free tier upgrade or 100% discount

---

## Cost Comparison Table

| Solution | Initial Cost | Monthly Cost (10k users) | 3D Support | Setup Time |
|----------|--------------|-------------------------|------------|------------|
| **Leaflet (Current)** | $0 | $0 | ❌ No | 0 min ✅ |
| **Mapbox Free Tier** | $0 | $0 (if <50k loads) | ✅ Yes | 5 min |
| **Mapbox Education** | $0 | $0 | ✅ Yes | 1-2 weeks |
| **MapLibre GL JS** | $0 | $0 | ✅ Yes | 30 min |
| **Self-Hosted Tiles** | $0 | $5 (VPS) | ❌ No | 2 hours |
| **Paid Mapbox** | $0 | $250+ | ✅ Yes | 5 min |

---

## My Recommendation

### For Student Budget + 10k Users:

**Immediate (Today):**
1. ✅ Use **Leaflet** (you already have it!)
2. ✅ Enhance existing MapView.jsx with route rendering
3. ✅ Keep all navigation functionality
4. ✅ **Cost: $0/month**

**This Week:**
1. Apply for **Mapbox Education Program**
2. While waiting, use Leaflet
3. If approved, add Mapbox as optional "3D View"

**If Education Denied:**
1. Switch to **MapLibre GL JS** (free Mapbox alternative)
2. Or keep Leaflet (works great!)
3. **Still $0/month**

---

## Updated Implementation (Using Leaflet)

I can update your code to use the **existing Leaflet** component you already have, which means:
- ✅ No Mapbox token needed
- ✅ $0 cost for unlimited users
- ✅ Works right now
- ✅ All navigation features still work

Would you like me to:
1. **Update NavigationChatbot to use Leaflet** (free, works now)
2. **Keep Mapbox integration** but as optional (for later)
3. **Apply for Mapbox Education** discount in parallel

Let me know your preference!
