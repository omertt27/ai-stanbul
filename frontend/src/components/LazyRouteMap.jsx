import React from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

/**
 * LazyRouteMap - Lazy-loaded map component for better performance
 * This component is loaded on-demand to reduce initial bundle size
 */
const LazyRouteMap = ({ 
  center, 
  routes, 
  markers, 
  userLocation,
  isDarkMode,
  onMapReady,
  MapControls,
  AnimatedPolyline,
  PulsingMarker,
  transferSteps,
  enableAnimations
}) => {
  return (
    <MapContainer 
      center={center} 
      zoom={13} 
      style={{ height: '100%', width: '100%' }}
      scrollWheelZoom={false}
      whenReady={() => {
        console.log('🗺️ Lazy map ready');
        if (onMapReady) onMapReady();
      }}
    >
      <TileLayer
        attribution='© CARTO © OpenStreetMap contributors'
        url='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
        maxZoom={19}
        keepBuffer={4}
        updateWhenIdle={true}
        updateWhenZooming={false}
      />
      
      {/* Interactive Map Controls */}
      {MapControls && <MapControls center={center} />}
      
      {/* Route Lines */}
      {routes.map((route, idx) => {
        const positions = route.coordinates?.map(coord => [
          coord.lat, 
          coord.lon || coord.lng
        ]).filter(pos => pos[0] !== undefined && pos[1] !== undefined) || [];
        
        if (positions.length === 0) return null;
        
        const getLineStyle = (mode) => {
          const styles = {
            walk: { weight: 3, color: '#9CA3AF', opacity: 0.6 },
            metro: { weight: 5, color: '#DC2626', opacity: 0.8 },
            bus: { weight: 4, color: '#2563EB', opacity: 0.8 },
            tram: { weight: 4, color: '#16A34A', opacity: 0.8 },
            ferry: { weight: 4, color: '#0891B2', opacity: 0.8 },
            default: { weight: 4, color: '#4F46E5', opacity: 0.7 }
          };
          return styles[mode?.toLowerCase()] || styles.default;
        };
        
        const lineStyle = getLineStyle(route.mode);
        
        return AnimatedPolyline ? (
          <AnimatedPolyline
            key={`route-${idx}`}
            positions={positions}
            color={route.color || lineStyle.color}
            weight={lineStyle.weight}
            opacity={lineStyle.opacity}
            speed={enableAnimations ? 30 : 0}
          />
        ) : (
          <Polyline
            key={`route-${idx}`}
            positions={positions}
            color={route.color || lineStyle.color}
            weight={lineStyle.weight}
            opacity={lineStyle.opacity}
          />
        );
      })}
      
      {/* Transfer Point Markers - Pulsing */}
      {PulsingMarker && transferSteps?.map((step, idx) => {
        if (!step.lat || !step.lon) return null;
        return (
          <PulsingMarker
            key={`transfer-${idx}`}
            position={[step.lat, step.lon]}
            label={`Transfer: ${step.instruction || step.description}`}
            isPulse={true}
          />
        );
      })}
      
      {/* Markers */}
      {markers.map((marker, idx) => {
        const markerLat = marker.lat || marker.latitude;
        const markerLon = marker.lon || marker.lng || marker.longitude;
        
        if (!markerLat || !markerLon) {
          console.warn(`⚠️ Marker ${idx} missing coordinates:`, marker);
          return null;
        }
        
        return (
          <Marker 
            key={`marker-${idx}`} 
            position={[markerLat, markerLon]}
          >
            <Popup>
              <div>
                <strong>{marker.label || marker.name}</strong>
                {marker.type && <div className="text-sm text-gray-600">{marker.type}</div>}
              </div>
            </Popup>
          </Marker>
        );
      })}
      
      {/* User Location Marker */}
      {userLocation && (
        <Marker 
          position={userLocation}
          icon={L.divIcon({
            className: 'user-location-marker',
            html: '<div style="background: #4F46E5; width: 16px; height: 16px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px rgba(0,0,0,0.3);"></div>',
            iconSize: [16, 16],
            iconAnchor: [8, 8]
          })}
        >
          <Popup>
            <div>
              <strong>Your Location</strong>
            </div>
          </Popup>
        </Marker>
      )}
    </MapContainer>
  );
};

export default LazyRouteMap;
