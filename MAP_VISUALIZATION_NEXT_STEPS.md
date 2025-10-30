# 🗺️ Map Visualization - Implementation Roadmap

**Date:** October 30, 2025  
**Status:** 🟡 Backend Complete, Frontend Pending  
**Priority:** HIGH - Core feature for transportation and route planning

---

## 📋 Overview

The backend now returns map visualization data for transportation and route planning queries. This document outlines the steps to complete the frontend integration and enable real-time GPS tracking.

---

## ✅ Completed (Backend)

- [x] MapVisualization Pydantic model created
- [x] GPS location parameter added to process_message()
- [x] Structured response format with map_data
- [x] Intent-based map data generation for 6 intent types
- [x] Route coordinates, markers, and transport lines support
- [x] Comprehensive documentation (MAP_VISUALIZATION_INTEGRATION.md)

---

## 🎯 Next Steps

### Step 1: Frontend Map Component Implementation (4-6 hours)

#### 1.1 Install Required Dependencies

```bash
# Using npm
npm install leaflet react-leaflet
npm install --save-dev @types/leaflet

# Or using yarn
yarn add leaflet react-leaflet
yarn add -D @types/leaflet
```

#### 1.2 Create MapVisualization Component

**File:** `frontend/src/components/MapVisualization.tsx`

```tsx
import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default marker icon issue in React-Leaflet
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

interface MapData {
  type?: string;
  coordinates?: [number, number][];
  markers?: Array<{
    lat: number;
    lon: number;
    label: string;
    type?: string;
  }>;
  center?: { lat: number; lon: number };
  zoom?: number;
  route_data?: {
    distance_km?: number;
    duration_min?: number;
    transport_mode?: string;
    lines?: string[];
  };
  transport_lines?: Array<{
    line: string;
    color: string;
    name: string;
  }>;
}

interface MapVisualizationProps {
  mapData: MapData;
  className?: string;
}

// Custom marker icons
const markerIcons = {
  start: L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
  }),
  destination: L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
  }),
  restaurant: L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-orange.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
  }),
  museum: L.icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-violet.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
  }),
  default: DefaultIcon
};

const MapVisualization: React.FC<MapVisualizationProps> = ({ mapData, className = '' }) => {
  if (!mapData || !mapData.center) {
    return null;
  }

  const center: [number, number] = [mapData.center.lat, mapData.center.lon];
  const zoom = mapData.zoom || 13;

  return (
    <div className={`map-container ${className}`}>
      <MapContainer 
        center={center} 
        zoom={zoom} 
        style={{ height: '400px', width: '100%', borderRadius: '8px' }}
        scrollWheelZoom={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {/* Draw route polyline */}
        {mapData.type === 'route' && mapData.coordinates && mapData.coordinates.length > 0 && (
          <Polyline 
            positions={mapData.coordinates as [number, number][]} 
            color="#0066CC"
            weight={4}
            opacity={0.7}
          />
        )}
        
        {/* Draw markers */}
        {mapData.markers && mapData.markers.map((marker, index) => {
          const markerType = marker.type || 'default';
          const icon = markerIcons[markerType as keyof typeof markerIcons] || markerIcons.default;
          
          return (
            <Marker 
              key={index} 
              position={[marker.lat, marker.lon]}
              icon={icon}
            >
              <Popup>
                <div className="marker-popup">
                  <strong>{marker.label}</strong>
                  {marker.type && <div className="marker-type">{marker.type}</div>}
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
      
      {/* Route information panel */}
      {mapData.route_data && (
        <div className="route-info-panel">
          <div className="route-info-item">
            <span className="label">Distance:</span>
            <span className="value">{mapData.route_data.distance_km} km</span>
          </div>
          <div className="route-info-item">
            <span className="label">Duration:</span>
            <span className="value">~{mapData.route_data.duration_min} min</span>
          </div>
          {mapData.route_data.transport_mode && (
            <div className="route-info-item">
              <span className="label">Mode:</span>
              <span className="value">{mapData.route_data.transport_mode}</span>
            </div>
          )}
          {mapData.route_data.lines && mapData.route_data.lines.length > 0 && (
            <div className="route-info-item">
              <span className="label">Lines:</span>
              <span className="value">{mapData.route_data.lines.join(', ')}</span>
            </div>
          )}
        </div>
      )}
      
      {/* Transport lines legend */}
      {mapData.transport_lines && mapData.transport_lines.length > 0 && (
        <div className="transport-lines-legend">
          <h4>Transport Lines:</h4>
          {mapData.transport_lines.map((line, index) => (
            <div key={index} className="transport-line-item">
              <span 
                className="line-color-indicator" 
                style={{ backgroundColor: line.color }}
              />
              <span className="line-name">{line.line} - {line.name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MapVisualization;
```

#### 1.3 Add CSS Styling

**File:** `frontend/src/components/MapVisualization.css`

```css
.map-container {
  margin: 16px 0;
  position: relative;
}

.route-info-panel {
  position: absolute;
  top: 10px;
  right: 10px;
  background: white;
  padding: 12px 16px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  min-width: 200px;
}

.route-info-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
}

.route-info-item:last-child {
  margin-bottom: 0;
}

.route-info-item .label {
  font-weight: 600;
  color: #555;
  margin-right: 12px;
}

.route-info-item .value {
  color: #0066CC;
  font-weight: 500;
}

.transport-lines-legend {
  position: absolute;
  bottom: 10px;
  left: 10px;
  background: white;
  padding: 12px 16px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  max-width: 300px;
}

.transport-lines-legend h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.transport-line-item {
  display: flex;
  align-items: center;
  margin-bottom: 6px;
  font-size: 13px;
}

.transport-line-item:last-child {
  margin-bottom: 0;
}

.line-color-indicator {
  width: 16px;
  height: 16px;
  border-radius: 3px;
  margin-right: 8px;
  flex-shrink: 0;
}

.line-name {
  color: #555;
}

.marker-popup {
  font-size: 14px;
}

.marker-popup strong {
  display: block;
  margin-bottom: 4px;
  color: #333;
}

.marker-type {
  font-size: 12px;
  color: #777;
  text-transform: capitalize;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
  .route-info-panel {
    top: auto;
    bottom: 60px;
    right: 10px;
    font-size: 12px;
    padding: 10px 12px;
    min-width: 150px;
  }
  
  .transport-lines-legend {
    bottom: 60px;
    left: 10px;
    font-size: 12px;
    max-width: 200px;
  }
}
```

#### 1.4 Integrate with Chat Component

**File:** `frontend/src/components/Chat.tsx`

```tsx
import React, { useState } from 'react';
import MapVisualization from './MapVisualization';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  mapData?: any;
  intent?: string;
  confidence?: number;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Get user's GPS location if available
      let gpsLocation = null;
      if (navigator.geolocation) {
        try {
          const position = await new Promise<GeolocationPosition>((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject, {
              timeout: 5000,
              maximumAge: 60000
            });
          });
          
          gpsLocation = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
          };
        } catch (error) {
          console.log('GPS not available or permission denied');
        }
      }

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          session_id: sessionStorage.getItem('session_id'),
          user_id: localStorage.getItem('user_id') || 'anonymous',
          gps_location: gpsLocation
        }),
      });

      const data = await response.json();

      // Store session ID for continuity
      if (data.session_id) {
        sessionStorage.setItem('session_id', data.session_id);
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.response,
        mapData: data.map_data,
        intent: data.intent,
        confidence: data.confidence
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-content">
              {message.content}
            </div>
            
            {/* Show map if map data is available */}
            {message.mapData && (
              <MapVisualization mapData={message.mapData} />
            )}
            
            {/* Show metadata */}
            {message.intent && (
              <div className="message-metadata">
                <span className="intent-badge">{message.intent}</span>
                {message.confidence && (
                  <span className="confidence-badge">
                    {(message.confidence * 100).toFixed(0)}% confidence
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant loading">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={sendMessage} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about Istanbul..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default Chat;
```

---

### Step 2: Test with Real Transportation Queries (2 hours)

#### 2.1 Test Query Categories

Create a test suite to verify map visualization works correctly:

**Test Queries:**

1. **Transportation Routes:**
   - "How do I get to Taksim Square?"
   - "Best way from Sultanahmet to Kadıköy"
   - "Metro route to the airport"

2. **Multiple Locations:**
   - "Show me restaurants in Beyoğlu"
   - "Museums near Sultanahmet"
   - "Best cafes in Karaköy"

3. **GPS-based Queries:**
   - "Restaurants near me"
   - "How do I get to Blue Mosque from here?"
   - "Nearest metro station"

4. **Route Planning:**
   - "Plan a day trip to Sultanahmet"
   - "Visit 3 museums in one day"
   - "Walking tour of Galata"

#### 2.2 Test Checklist

- [ ] Map renders correctly
- [ ] Markers appear in correct locations
- [ ] Route polylines are drawn
- [ ] Popup information displays
- [ ] Route info panel shows data
- [ ] Transport lines legend appears
- [ ] Map centers on correct location
- [ ] Zoom level is appropriate
- [ ] Mobile responsive
- [ ] GPS permission handling works

#### 2.3 Debug Tools

```typescript
// Add console logging for debugging
useEffect(() => {
  if (mapData) {
    console.log('Map Data Received:', {
      type: mapData.type,
      markerCount: mapData.markers?.length,
      coordinateCount: mapData.coordinates?.length,
      hasRouteData: !!mapData.route_data
    });
  }
}, [mapData]);
```

---

### Step 3: Add Route Visualization Enhancements (2 hours)

#### 3.1 Animated Route Drawing

```typescript
import { useEffect, useState } from 'react';

const AnimatedRoute: React.FC<{ coordinates: [number, number][] }> = ({ coordinates }) => {
  const [visibleCoordinates, setVisibleCoordinates] = useState<[number, number][]>([]);

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      if (index < coordinates.length) {
        setVisibleCoordinates(coordinates.slice(0, index + 1));
        index++;
      } else {
        clearInterval(interval);
      }
    }, 50); // Draw at 50ms intervals

    return () => clearInterval(interval);
  }, [coordinates]);

  return (
    <Polyline 
      positions={visibleCoordinates} 
      color="#0066CC"
      weight={4}
      opacity={0.7}
    />
  );
};
```

#### 3.2 Interactive Markers with Details

```typescript
const EnhancedMarker: React.FC<{ marker: MarkerData }> = ({ marker }) => {
  return (
    <Marker position={[marker.lat, marker.lon]} icon={getMarkerIcon(marker.type)}>
      <Popup>
        <div className="enhanced-popup">
          <h3>{marker.label}</h3>
          {marker.description && <p>{marker.description}</p>}
          {marker.rating && (
            <div className="rating">
              ⭐ {marker.rating} / 5
            </div>
          )}
          {marker.distance && (
            <div className="distance">
              📍 {marker.distance} km away
            </div>
          )}
        </div>
      </Popup>
    </Marker>
  );
};
```

#### 3.3 Current Location Tracking

```typescript
const [userLocation, setUserLocation] = useState<[number, number] | null>(null);

useEffect(() => {
  if (navigator.geolocation) {
    const watchId = navigator.geolocation.watchPosition(
      (position) => {
        setUserLocation([
          position.coords.latitude,
          position.coords.longitude
        ]);
      },
      (error) => console.error('Location error:', error),
      {
        enableHighAccuracy: true,
        timeout: 5000,
        maximumAge: 0
      }
    );

    return () => navigator.geolocation.clearWatch(watchId);
  }
}, []);

// In MapContainer:
{userLocation && (
  <Marker 
    position={userLocation}
    icon={currentLocationIcon}
  >
    <Popup>Your Current Location</Popup>
  </Marker>
)}
```

---

### Step 4: Enable Real-time GPS Tracking (2 hours)

#### 4.1 GPS Permission Component

```typescript
const GPSPermissionRequest: React.FC<{ onPermissionGranted: () => void }> = ({ onPermissionGranted }) => {
  const [permissionStatus, setPermissionStatus] = useState<'prompt' | 'granted' | 'denied'>('prompt');

  const requestPermission = async () => {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      () => {
        setPermissionStatus('granted');
        onPermissionGranted();
      },
      (error) => {
        if (error.code === error.PERMISSION_DENIED) {
          setPermissionStatus('denied');
        }
      }
    );
  };

  if (permissionStatus === 'granted') return null;

  return (
    <div className="gps-permission-banner">
      <p>📍 Enable location services for better route recommendations</p>
      <button onClick={requestPermission}>Enable GPS</button>
      {permissionStatus === 'denied' && (
        <p className="error">Location access denied. Please enable in browser settings.</p>
      )}
    </div>
  );
};
```

#### 4.2 Real-time Location Updates

```typescript
const useRealTimeLocation = () => {
  const [location, setLocation] = useState<GeolocationPosition | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!navigator.geolocation) {
      setError('Geolocation not supported');
      return;
    }

    const watchId = navigator.geolocation.watchPosition(
      (position) => {
        setLocation(position);
        setError(null);
      },
      (err) => {
        setError(err.message);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );

    return () => navigator.geolocation.clearWatch(watchId);
  }, []);

  return { location, error };
};
```

#### 4.3 Auto-refresh Routes on Location Change

```typescript
const Chat: React.FC = () => {
  const { location } = useRealTimeLocation();
  const [lastQueryWithGPS, setLastQueryWithGPS] = useState<string | null>(null);

  useEffect(() => {
    // If user moved significantly and had a GPS-based query, refresh
    if (location && lastQueryWithGPS) {
      const distanceMoved = calculateDistance(
        previousLocation,
        location.coords
      );

      if (distanceMoved > 0.1) { // 100 meters
        // Re-send query with new location
        refreshQueryWithNewLocation(lastQueryWithGPS);
      }
    }
  }, [location]);

  const calculateDistance = (
    prev: GeolocationCoordinates,
    current: GeolocationCoordinates
  ): number => {
    // Haversine formula implementation
    const R = 6371; // Earth's radius in km
    const dLat = (current.latitude - prev.latitude) * Math.PI / 180;
    const dLon = (current.longitude - prev.longitude) * Math.PI / 180;
    const a = 
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(prev.latitude * Math.PI / 180) *
      Math.cos(current.latitude * Math.PI / 180) *
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };
};
```

---

## 📊 Testing Plan

### Unit Tests

```typescript
describe('MapVisualization', () => {
  it('renders without crashing', () => {
    const mapData = {
      center: { lat: 41.0082, lon: 28.9784 },
      zoom: 13
    };
    render(<MapVisualization mapData={mapData} />);
  });

  it('displays markers correctly', () => {
    const mapData = {
      center: { lat: 41.0082, lon: 28.9784 },
      markers: [
        { lat: 41.0082, lon: 28.9784, label: 'Test', type: 'start' }
      ]
    };
    const { container } = render(<MapVisualization mapData={mapData} />);
    expect(container.querySelector('.leaflet-marker-icon')).toBeInTheDocument();
  });

  it('draws route polyline', () => {
    const mapData = {
      center: { lat: 41.0082, lon: 28.9784 },
      type: 'route',
      coordinates: [[41.0082, 28.9784], [41.0100, 28.9800]]
    };
    const { container } = render(<MapVisualization mapData={mapData} />);
    expect(container.querySelector('.leaflet-interactive')).toBeInTheDocument();
  });
});
```

### Integration Tests

```typescript
describe('Chat with Map Integration', () => {
  it('shows map for transportation queries', async () => {
    render(<Chat />);
    
    const input = screen.getByPlaceholderText('Ask about Istanbul...');
    const submitButton = screen.getByText('Send');
    
    fireEvent.change(input, { target: { value: 'How do I get to Taksim?' } });
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(screen.getByClassName('map-container')).toBeInTheDocument();
    });
  });

  it('requests GPS permission when needed', async () => {
    const mockGeolocation = {
      getCurrentPosition: jest.fn()
    };
    global.navigator.geolocation = mockGeolocation;
    
    render(<Chat />);
    
    // Test GPS request logic
  });
});
```

---

## 🚀 Deployment Checklist

- [ ] Install dependencies (leaflet, react-leaflet)
- [ ] Create MapVisualization component
- [ ] Add CSS styling
- [ ] Integrate with Chat component
- [ ] Add GPS permission handling
- [ ] Test on desktop browsers
- [ ] Test on mobile devices
- [ ] Test with various query types
- [ ] Verify map performance
- [ ] Check accessibility
- [ ] Add error boundaries
- [ ] Document component API
- [ ] Create usage examples
- [ ] Deploy to staging
- [ ] User acceptance testing
- [ ] Deploy to production

---

## 📈 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Map load time | <2s | Performance API |
| GPS accuracy | <100m | Geolocation API |
| Mobile usability | >80% | User testing |
| Error rate | <1% | Error logging |
| User engagement | +50% | Analytics |

---

## 🎯 Timeline

### Week 1 (Nov 1-7, 2025)
- **Day 1-2:** Frontend component implementation
- **Day 3:** CSS styling and responsive design
- **Day 4:** Chat integration
- **Day 5:** GPS tracking implementation
- **Day 6-7:** Testing and bug fixes

### Week 2 (Nov 8-14, 2025)
- **Day 1-2:** Advanced features (animations, enhanced markers)
- **Day 3-4:** Mobile optimization
- **Day 5:** Performance optimization
- **Day 6-7:** Staging deployment and UAT

### Week 3 (Nov 15-21, 2025)
- **Day 1-3:** Production deployment
- **Day 4-7:** Monitoring and iteration

---

## 🐛 Known Issues & Solutions

### Issue 1: Leaflet CSS Not Loading
**Solution:** Import CSS in main App.tsx:
```typescript
import 'leaflet/dist/leaflet.css';
```

### Issue 2: Marker Icons Not Displaying
**Solution:** Use webpack configuration to handle images:
```javascript
// webpack.config.js
module: {
  rules: [
    {
      test: /\.(png|jpe?g|gif|svg)$/i,
      type: 'asset/resource',
    },
  ],
}
```

### Issue 3: GPS Permission Blocked
**Solution:** Provide clear instructions to users:
```typescript
const showGPSInstructions = () => {
  alert(`
    To enable GPS:
    1. Click the lock icon in your browser's address bar
    2. Change location permission to "Allow"
    3. Refresh the page
  `);
};
```

---

## 📚 Resources

### Documentation
- [Leaflet Documentation](https://leafletjs.com/reference.html)
- [React-Leaflet Guide](https://react-leaflet.js.org/)
- [Geolocation API](https://developer.mozilla.org/en-US/docs/Web/API/Geolocation_API)
- [OpenStreetMap Tiles](https://wiki.openstreetmap.org/wiki/Tile_servers)

### Tutorials
- [React-Leaflet Tutorial](https://react-leaflet.js.org/docs/start-introduction/)
- [GPS Tracking with React](https://www.digitalocean.com/community/tutorials/react-geolocation)

### Tools
- [Leaflet Marker Generator](https://github.com/pointhi/leaflet-color-markers)
- [Map Testing Tool](https://geojson.io/)
- [GPS Simulator](https://chrome.google.com/webstore/detail/gps-simulator)

---

**Created:** October 30, 2025  
**Status:** 🟡 In Progress  
**Priority:** HIGH  
**Estimated Completion:** November 21, 2025

**Next Action:** Begin Step 1.1 - Install Dependencies
