import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default markers for React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom marker icons
const createCustomIcon = (color, symbol) => {
  return L.divIcon({
    className: 'custom-icon',
    html: `
      <div style="
        background-color: ${color};
        width: 30px;
        height: 30px;
        border-radius: 50%;
        border: 3px solid white;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        font-size: 16px;
      ">
        ${symbol}
      </div>
    `,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
    popupAnchor: [0, -15]
  });
};

const startIcon = createCustomIcon('#4CAF50', '🚩');
const endIcon = createCustomIcon('#F44336', '🎯');
const userIcon = createCustomIcon('#2196F3', '📍');
const poiIcon = createCustomIcon('#FF9800', '📌');

// Component to auto-fit bounds when route changes
function AutoFitBounds({ bounds }) {
  const map = useMap();
  
  useEffect(() => {
    if (bounds && bounds.length > 0) {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
    }
  }, [bounds, map]);
  
  return null;
}

/**
 * LeafletNavigationMap - Free, open-source navigation map
 * 
 * Features:
 * - Route rendering with polyline
 * - Start/end markers
 * - POI markers
 * - GPS location tracking
 * - Turn-by-turn popup
 * - Auto-zoom to fit route
 * - No API token required
 * - Unlimited users - $0 cost!
 */
function LeafletNavigationMap({
  routeData = null,
  pois = [],
  userLocation = null,
  height = '500px',
  className = '',
  showControls = true
}) {
  const mapRef = useRef(null);
  
  // Calculate center and bounds
  const getCenter = () => {
    if (userLocation) {
      return [userLocation.latitude, userLocation.longitude];
    }
    if (routeData?.geometry?.coordinates && routeData.geometry.coordinates.length > 0) {
      const firstCoord = routeData.geometry.coordinates[0];
      return [firstCoord[1], firstCoord[0]]; // [lat, lng]
    }
    if (pois.length > 0) {
      return [pois[0].lat, pois[0].lng];
    }
    return [41.0082, 28.9784]; // Istanbul center
  };

  // Calculate bounds for auto-fit
  const getBounds = () => {
    const bounds = [];
    
    // Add route coordinates
    if (routeData?.geometry?.coordinates) {
      routeData.geometry.coordinates.forEach(coord => {
        bounds.push([coord[1], coord[0]]); // [lat, lng]
      });
    }
    
    // Add POI coordinates
    if (pois && pois.length > 0) {
      pois.forEach(poi => {
        bounds.push([poi.lat, poi.lng]);
      });
    }
    
    // Add user location
    if (userLocation) {
      bounds.push([userLocation.latitude, userLocation.longitude]);
    }
    
    return bounds.length > 0 ? bounds : null;
  };

  // Convert route geometry to Leaflet polyline format
  const getRouteCoordinates = () => {
    if (!routeData?.geometry?.coordinates) return [];
    
    // Convert from [lng, lat] to [lat, lng]
    return routeData.geometry.coordinates.map(coord => [coord[1], coord[0]]);
  };

  // Get route color based on mode
  const getRouteColor = () => {
    const mode = routeData?.mode || 'driving';
    const colors = {
      walking: '#4CAF50',
      driving: '#2196F3',
      cycling: '#FF9800',
      transit: '#9C27B0'
    };
    return colors[mode] || '#2196F3';
  };

  const center = getCenter();
  const bounds = getBounds();
  const routeCoordinates = getRouteCoordinates();
  const routeColor = getRouteColor();

  const tileUrl = import.meta.env.VITE_OSM_TILE_URL || 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';

  return (
    <div className={`leaflet-navigation-container ${className}`} style={{ position: 'relative' }}>
      <MapContainer
        ref={mapRef}
        center={center}
        zoom={13}
        style={{ height, width: '100%', zIndex: 0 }}
        scrollWheelZoom={true}
        zoomControl={showControls}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url={tileUrl}
        />

        {/* Auto-fit bounds when route changes */}
        <AutoFitBounds bounds={bounds} />

        {/* Route polyline */}
        {routeCoordinates.length > 0 && (
          <Polyline
            positions={routeCoordinates}
            color={routeColor}
            weight={6}
            opacity={0.8}
            smoothFactor={1}
          >
            <Popup>
              <div style={{ minWidth: '150px' }}>
                <strong>Route Info</strong>
                <br />
                Distance: {(routeData.distance / 1000).toFixed(2)} km
                <br />
                Duration: {Math.round(routeData.duration / 60)} min
                <br />
                Mode: {routeData.mode || 'driving'}
              </div>
            </Popup>
          </Polyline>
        )}

        {/* Start marker */}
        {routeCoordinates.length > 0 && (
          <Marker position={routeCoordinates[0]} icon={startIcon}>
            <Popup>
              <strong>Start</strong>
              <br />
              Starting point of your route
            </Popup>
          </Marker>
        )}

        {/* End marker */}
        {routeCoordinates.length > 0 && (
          <Marker 
            position={routeCoordinates[routeCoordinates.length - 1]} 
            icon={endIcon}
          >
            <Popup>
              <strong>Destination</strong>
              <br />
              End point of your route
            </Popup>
          </Marker>
        )}

        {/* User location marker */}
        {userLocation && (
          <Marker
            position={[userLocation.latitude, userLocation.longitude]}
            icon={userIcon}
          >
            <Popup>
              <strong>Your Location</strong>
              <br />
              Current GPS position
            </Popup>
          </Marker>
        )}

        {/* POI markers */}
        {pois && pois.map((poi, index) => (
          <Marker
            key={index}
            position={[poi.lat || poi.latitude, poi.lng || poi.longitude]}
            icon={poiIcon}
          >
            <Popup>
              <div style={{ minWidth: '150px' }}>
                <strong>{poi.name}</strong>
                {poi.category && (
                  <>
                    <br />
                    <span style={{ fontSize: '12px', color: '#666' }}>
                      {poi.category}
                    </span>
                  </>
                )}
                {poi.description && (
                  <>
                    <br />
                    <span style={{ fontSize: '12px' }}>{poi.description}</span>
                  </>
                )}
                {poi.distance && (
                  <>
                    <br />
                    <span style={{ fontSize: '12px', color: '#2196F3' }}>
                      {(poi.distance / 1000).toFixed(2)} km away
                    </span>
                  </>
                )}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>

      {/* Route info overlay */}
      {routeData && (
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          background: 'white',
          padding: '12px 16px',
          borderRadius: '8px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          zIndex: 1000,
          fontSize: '14px',
          minWidth: '180px'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '8px', fontSize: '15px' }}>
            📍 Route Summary
          </div>
          {routeData.distance && (
            <div style={{ marginBottom: '4px' }}>
              <strong>Distance:</strong> {(routeData.distance / 1000).toFixed(2)} km
            </div>
          )}
          {routeData.duration && (
            <div style={{ marginBottom: '4px' }}>
              <strong>Duration:</strong> {Math.round(routeData.duration / 60)} min
            </div>
          )}
          {routeData.mode && (
            <div>
              <strong>Mode:</strong> {routeData.mode}
            </div>
          )}
        </div>
      )}

      <style>{`
        .leaflet-navigation-container .custom-icon {
          animation: markerPulse 2s infinite;
        }
        
        @keyframes markerPulse {
          0%, 100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.1);
          }
        }

        .leaflet-container {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }

        .leaflet-popup-content-wrapper {
          border-radius: 8px;
        }

        .leaflet-popup-content {
          margin: 12px;
          line-height: 1.5;
        }
      `}</style>
    </div>
  );
}

export default React.memo(LeafletNavigationMap);
