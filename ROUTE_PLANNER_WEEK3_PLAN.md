# Route Planner Week 3 - Frontend Map Visualization

**Date:** November 6, 2025  
**Status:** 🚀 **IN PROGRESS** - Building Interactive Map Interface

---

## 🎯 Week 3 Objectives

### 1. Leaflet.js Map Integration
- [ ] Set up Leaflet.js in frontend
- [ ] Display interactive map of Istanbul
- [ ] Configure map tiles and styling
- [ ] Add zoom and pan controls

### 2. Route Visualization
- [ ] Display polyline routes on map
- [ ] Color-coded route segments
- [ ] Animated route drawing
- [ ] Route distance/duration overlay

### 3. Custom Markers
- [ ] Numbered location markers
- [ ] Different icons for types (museum, cafe, restaurant)
- [ ] Marker clustering for nearby locations
- [ ] Popup info cards on marker click

### 4. Interactive Features
- [ ] Drag-and-drop waypoint reordering
- [ ] Click to add/remove locations
- [ ] Real-time route recalculation
- [ ] Alternative route display

### 5. Mobile Responsiveness
- [ ] Touch-friendly controls
- [ ] Responsive layout
- [ ] Swipe gestures
- [ ] Bottom sheet for mobile

### 6. Route Management
- [ ] Save routes to database
- [ ] Share route via URL
- [ ] Export to GPX/KML
- [ ] Print-friendly view

---

## 📁 Files to Create/Modify

### New Frontend Files

1. **`/frontend/components/RouteMap.tsx`** (NEW)
   - Main map component with Leaflet
   - Route polyline rendering
   - Marker management

2. **`/frontend/components/RouteMarker.tsx`** (NEW)
   - Custom marker component
   - Different icons for location types
   - Popup info cards

3. **`/frontend/components/RouteSidebar.tsx`** (NEW)
   - Itinerary list view
   - Drag-and-drop reordering
   - Location details

4. **`/frontend/components/RouteControls.tsx`** (NEW)
   - Save/share buttons
   - Transport mode selector
   - Route options

5. **`/frontend/styles/map.css`** (NEW)
   - Map styling
   - Marker animations
   - Responsive layout

### Updated Backend Files

6. **`/backend/api/route_planner_routes.py`** (UPDATE)
   - Add save route endpoint
   - Add share route endpoint
   - Add route export endpoints

7. **`/backend/models/saved_routes.py`** (NEW)
   - Database model for saved routes
   - User-route relationships

---

## 🗺️ Implementation Plan

### Phase 1: Basic Map Setup (2-3 hours)

**Step 1: Install Dependencies**
```bash
npm install leaflet react-leaflet
npm install @types/leaflet --save-dev
```

**Step 2: Create Map Component**
```tsx
// frontend/components/RouteMap.tsx
import { MapContainer, TileLayer, Polyline, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

interface Location {
  id: string;
  name: string;
  position: [number, number];
  type: string;
}

interface RouteMapProps {
  locations: Location[];
  polyline: [number, number][];
  center: [number, number];
}

export default function RouteMap({ locations, polyline, center }: RouteMapProps) {
  return (
    <MapContainer
      center={center}
      zoom={13}
      style={{ height: '600px', width: '100%' }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; OpenStreetMap contributors'
      />
      
      {/* Route polyline */}
      <Polyline 
        positions={polyline}
        color="#2563eb"
        weight={4}
        opacity={0.7}
      />
      
      {/* Location markers */}
      {locations.map((loc, index) => (
        <Marker key={loc.id} position={loc.position}>
          <Popup>
            <div>
              <h3>{index + 1}. {loc.name}</h3>
              <p>{loc.type}</p>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
```

### Phase 2: Custom Markers & Icons (1-2 hours)

**Step 3: Create Custom Marker Icons**
```tsx
// frontend/components/RouteMarker.tsx
import L from 'leaflet';
import { Marker, Popup } from 'react-leaflet';

const getMarkerIcon = (type: string, number: number) => {
  const iconUrl = type === 'museum' 
    ? '/icons/museum.svg'
    : type === 'cafe'
    ? '/icons/cafe.svg'
    : '/icons/default.svg';
    
  return L.divIcon({
    html: `
      <div class="custom-marker ${type}">
        <div class="marker-number">${number}</div>
        <img src="${iconUrl}" alt="${type}" />
      </div>
    `,
    className: 'custom-marker-container',
    iconSize: [40, 40],
    iconAnchor: [20, 40],
  });
};

export function RouteMarker({ location, index }: Props) {
  const icon = getMarkerIcon(location.type, index + 1);
  
  return (
    <Marker position={location.position} icon={icon}>
      <Popup>
        <LocationInfoCard location={location} />
      </Popup>
    </Marker>
  );
}
```

### Phase 3: Interactive Sidebar (2-3 hours)

**Step 4: Create Draggable Itinerary List**
```tsx
// frontend/components/RouteSidebar.tsx
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

export function RouteSidebar({ locations, onReorder }: Props) {
  const handleDragEnd = (result) => {
    if (!result.destination) return;
    
    const items = Array.from(locations);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);
    
    onReorder(items);
  };
  
  return (
    <DragDropContext onDragEnd={handleDragEnd}>
      <Droppable droppableId="locations">
        {(provided) => (
          <div {...provided.droppableProps} ref={provided.innerRef}>
            {locations.map((loc, index) => (
              <Draggable key={loc.id} draggableId={loc.id} index={index}>
                {(provided) => (
                  <LocationCard
                    ref={provided.innerRef}
                    {...provided.draggableProps}
                    {...provided.dragHandleProps}
                    location={loc}
                    index={index}
                  />
                )}
              </Draggable>
            ))}
            {provided.placeholder}
          </div>
        )}
      </Droppable>
    </DragDropContext>
  );
}
```

### Phase 4: Route Management (2-3 hours)

**Step 5: Add Save/Share Functionality**
```tsx
// frontend/components/RouteControls.tsx
export function RouteControls({ route, onSave, onShare }: Props) {
  const handleSave = async () => {
    const response = await fetch('/api/routes/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(route)
    });
    const data = await response.json();
    return data.routeId;
  };
  
  const handleShare = async () => {
    const routeId = await handleSave();
    const shareUrl = `${window.location.origin}/routes/${routeId}`;
    navigator.clipboard.writeText(shareUrl);
    toast.success('Route link copied to clipboard!');
  };
  
  return (
    <div className="route-controls">
      <button onClick={handleSave}>
        💾 Save Route
      </button>
      <button onClick={handleShare}>
        🔗 Share Route
      </button>
      <button onClick={() => window.print()}>
        🖨️ Print
      </button>
    </div>
  );
}
```

### Phase 5: Mobile Optimization (1-2 hours)

**Step 6: Responsive Design**
```css
/* frontend/styles/map.css */
@media (max-width: 768px) {
  .route-container {
    flex-direction: column;
  }
  
  .route-map {
    height: 50vh;
  }
  
  .route-sidebar {
    height: 50vh;
    overflow-y: auto;
  }
  
  .custom-marker {
    transform: scale(0.8);
  }
}

/* Touch-friendly controls */
.map-controls {
  padding: 16px;
  touch-action: manipulation;
}

.draggable-item {
  cursor: grab;
  user-select: none;
}

.draggable-item:active {
  cursor: grabbing;
}
```

---

## 🎨 Design Mockup

### Desktop Layout
```
┌─────────────────────────────────────────────────────────┐
│  AI Istanbul Route Planner              🔍 💾 🔗 🖨️    │
├──────────────────────┬──────────────────────────────────┤
│                      │                                  │
│  Itinerary Sidebar   │         Interactive Map          │
│                      │                                  │
│  1. 🏛️ Hagia Sophia  │    ┌──────────────────────┐    │
│     90 min           │    │  ①  →  ②  →  ③       │    │
│                      │    │                      │    │
│  ─────────────       │    │    [Route Line]      │    │
│  Walk • 1.2 km       │    │                      │    │
│  15 min              │    │  ④  →  ⑤             │    │
│  ─────────────       │    └──────────────────────┘    │
│                      │                                  │
│  2. ☕ Cafe Break     │  Legend:                        │
│     30 min           │  ① = Museum  ☕ = Cafe           │
│                      │  ─ = Walking Route              │
│  ─────────────       │                                  │
│  Walk • 0.5 km       │  Total: 3.5 km • 3h 15min      │
│  7 min               │                                  │
│  ─────────────       │                                  │
│                      │                                  │
│  3. 🏛️ Blue Mosque   │                                  │
│     60 min           │                                  │
│                      │                                  │
└──────────────────────┴──────────────────────────────────┘
```

### Mobile Layout
```
┌─────────────────────┐
│  AI Istanbul Route  │
│  🔍 💾 🔗           │
├─────────────────────┤
│                     │
│  [Interactive Map]  │
│                     │
│   ①  →  ②  →  ③    │
│                     │
│   ④  →  ⑤          │
│                     │
├─────────────────────┤
│ ▼ Itinerary (5)     │
├─────────────────────┤
│ 1. 🏛️ Hagia Sophia │
│    90 min           │
│ ─────── 1.2km ──────│
│ 2. ☕ Cafe Break    │
│    30 min           │
│ ─────── 0.5km ──────│
│ 3. 🏛️ Blue Mosque  │
│                     │
└─────────────────────┘
```

---

## 🔧 Backend API Updates

### New Endpoints

**1. Save Route**
```http
POST /api/routes/save
Content-Type: application/json

{
  "route": { /* itinerary data */ },
  "name": "My Istanbul Tour",
  "user_id": "optional"
}

Response: { "route_id": "abc123", "share_url": "..." }
```

**2. Get Saved Route**
```http
GET /api/routes/{route_id}

Response: { /* full itinerary data */ }
```

**3. Export Route**
```http
GET /api/routes/{route_id}/export?format=gpx
GET /api/routes/{route_id}/export?format=kml

Response: GPX/KML file download
```

---

## 📊 Success Metrics

| Feature | Target | Measurement |
|---------|--------|-------------|
| Map Load Time | <2s | First paint |
| Route Render | <500ms | Polyline draw |
| Marker Click Response | <100ms | Popup open |
| Mobile Usability | >90% | Touch accuracy |
| Save Success Rate | >95% | API success |

---

## 🚀 Quick Start (After Implementation)

### Test the Map Interface

```bash
# Start frontend dev server
cd frontend
npm run dev

# Visit http://localhost:3000/routes
```

### Create a Route

1. Enter query: "Show me museums in Sultanahmet"
2. Map displays with route polyline
3. Click markers to see details
4. Drag to reorder stops
5. Click "Save Route" to persist
6. Click "Share" to get URL

---

## 📱 Mobile Features

1. **Touch Gestures**
   - Pinch to zoom
   - Swipe to pan
   - Tap markers for info
   - Long press for options

2. **Bottom Sheet**
   - Expandable itinerary list
   - Swipe up for full view
   - Swipe down to minimize

3. **Offline Support** (Future)
   - Cache map tiles
   - Save routes locally
   - Sync when online

---

## 🎯 Week 3 Timeline

| Day | Tasks | Duration |
|-----|-------|----------|
| Day 1 | Leaflet setup, basic map | 3-4 hours |
| Day 2 | Custom markers, polylines | 3-4 hours |
| Day 3 | Sidebar, drag-and-drop | 3-4 hours |
| Day 4 | Save/share functionality | 2-3 hours |
| Day 5 | Mobile optimization | 2-3 hours |
| Day 6 | Testing & polish | 2-3 hours |

**Total Estimated Time:** 15-21 hours

---

## 🔍 Technical Decisions

### Map Library: Leaflet.js ✅
**Why Leaflet?**
- Lightweight (38KB)
- Open source
- Mobile-friendly
- Large ecosystem
- No API keys needed

**Alternatives Considered:**
- ❌ Google Maps (requires API key, costs money)
- ❌ Mapbox (requires API key, pricing)
- ❌ OpenLayers (heavier, more complex)

### State Management: React Hooks ✅
**Why Hooks?**
- Simple for this use case
- No external dependencies
- Easy to understand

**For Larger App:**
- Consider Redux Toolkit
- Or Zustand for simplicity

### Styling: Tailwind CSS ✅
**Why Tailwind?**
- Rapid development
- Mobile-first
- Consistent design system

---

## 📚 Resources

### Leaflet Documentation
- [Leaflet Quick Start](https://leafletjs.com/examples/quick-start/)
- [React Leaflet Docs](https://react-leaflet.js.org/)
- [Custom Markers Guide](https://leafletjs.com/examples/custom-icons/)

### Design Inspiration
- Google Maps route planning
- Citymapper app
- Komoot route builder
- AllTrails map interface

### Icon Sets
- [Heroicons](https://heroicons.com/) - Free SVG icons
- [Lucide](https://lucide.dev/) - Beautiful icons
- [Font Awesome](https://fontawesome.com/) - Classic icons

---

## ✅ Definition of Done

Week 3 will be complete when:

- [ ] Interactive map displays Istanbul
- [ ] Route polylines render correctly
- [ ] Custom numbered markers work
- [ ] Sidebar shows draggable itinerary
- [ ] Save/share functionality works
- [ ] Mobile layout is responsive
- [ ] All features tested on mobile
- [ ] Documentation updated
- [ ] Screenshots added to docs

---

**Week 3 Status:** 🚀 **STARTING**  
**Estimated Completion:** 15-21 hours  
**Next Review:** After Phase 1 (Basic Map)

---

*Started: November 6, 2025*  
*Target Completion: November 8-9, 2025*
