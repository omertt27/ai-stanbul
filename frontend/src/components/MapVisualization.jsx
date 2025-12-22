/**
 * MapVisualization Component
 * ==========================
 * Displays interactive maps with routes, markers, and transport lines
 * for transportation and route planning queries.
 * 
 * Features:
 * - Route visualization with polylines
 * - Multiple markers with labels
 * - Transport line indicators
 * - Auto-fit bounds
 * - Custom marker colors
 * - Route information display
 * 
 * Props:
 * - mapData: Object containing map visualization data from backend
 * - height: Map container height (default: '400px')
 * - className: Additional CSS classes
 */

import React, { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default markers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom marker icons with colors
const createMarkerIcon = (type = 'default', label = '') => {
  const colors = {
    start: '#22c55e',      // Green
    end: '#ef4444',        // Red
    destination: '#ef4444', // Red
    origin: '#22c55e',     // Green
    stop: '#3b82f6',       // Blue
    attraction: '#8b5cf6', // Purple
    default: '#6b7280'     // Gray
  };
  
  const color = colors[type] || colors.default;
  const iconHtml = `
    <div style="
      background-color: ${color};
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 3px solid white;
      box-shadow: 0 2px 6px rgba(0,0,0,0.4);
      font-size: 16px;
      color: white;
      font-weight: bold;
    ">
      ${label || '📍'}
    </div>
  `;
  
  return L.divIcon({
    html: iconHtml,
    className: 'custom-marker-icon',
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16]
  });
};

// Auto-fit bounds component
const AutoFitBounds = ({ coordinates, markers }) => {
  const map = useMap();

  useEffect(() => {
    if (!map) return;

    const bounds = [];
    
    // Add route coordinates
    if (coordinates && coordinates.length > 0) {
      coordinates.forEach(coord => {
        if (Array.isArray(coord) && coord.length === 2) {
          bounds.push([coord[0], coord[1]]);
        }
      });
    }
    
    // Add marker positions
    if (markers && markers.length > 0) {
      markers.forEach(marker => {
        if (marker.lat !== undefined && marker.lon !== undefined) {
          bounds.push([marker.lat, marker.lon]);
        }
      });
    }
    
    // Fit map to bounds
    if (bounds.length > 0) {
      const leafletBounds = L.latLngBounds(bounds);
      map.fitBounds(leafletBounds, { padding: [50, 50] });
    }
  }, [map, coordinates, markers]);

  return null;
};

const MapVisualization = ({ 
  mapData, 
  height = '400px',
  className = '' 
}) => {
  const [error, setError] = useState(null);

  if (!mapData) {
    return null;
  }

  // Extract data from mapData
  const {
    type = 'route',
    coordinates = [],
    markers = [],
    center = { lat: 41.0082, lon: 28.9784 }, // Istanbul default
    zoom = 13,
    route_data = null,
    transport_lines = []
  } = mapData;

  // Extract coordinates - check both top-level and nested in route_data
  let extractedCoordinates = coordinates;
  if ((!coordinates || coordinates.length === 0) && route_data?.coordinates) {
    extractedCoordinates = route_data.coordinates;
  }

  // Default center if not provided
  const defaultCenter = [center.lat || 41.0082, center.lon || 28.9784];

  // Route line color based on transport mode
  const getRouteColor = () => {
    if (route_data?.transport_mode) {
      const mode = route_data.transport_mode.toLowerCase();
      if (mode.includes('metro')) return '#e74c3c';
      if (mode.includes('tram')) return '#3498db';
      if (mode.includes('bus')) return '#2ecc71';
      if (mode.includes('ferry')) return '#1abc9c';
      if (mode.includes('walk')) return '#95a5a6';
    }
    return '#3b82f6'; // Default blue
  };

  // Convert coordinates to Leaflet format [[lat, lon], ...]
  const routeCoordinates = extractedCoordinates.map(coord => {
    if (Array.isArray(coord) && coord.length === 2) {
      return [coord[0], coord[1]];
    }
    return null;
  }).filter(Boolean);

  return (
    <div className={`map-visualization-container ${className}`}>
      {/* Route Information */}
      {route_data && (
        <div className="route-info-panel" style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          padding: '12px 16px',
          borderRadius: '8px 8px 0 0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '20px' }}>🗺️</span>
            <span style={{ fontWeight: 'bold' }}>Route Details</span>
          </div>
          <div style={{ display: 'flex', gap: '16px', fontSize: '14px' }}>
            {route_data.distance_km && (
              <span>📏 {route_data.distance_km.toFixed(1)} km</span>
            )}
            {route_data.duration_min && (
              <span>⏱️ {Math.round(route_data.duration_min)} min</span>
            )}
            {route_data.transport_mode && (
              <span>🚇 {route_data.transport_mode}</span>
            )}
          </div>
          {route_data.lines && route_data.lines.length > 0 && (
            <div style={{ width: '100%', marginTop: '8px', fontSize: '13px' }}>
              <strong>Lines:</strong> {route_data.lines.join(', ')}
            </div>
          )}
        </div>
      )}

      {/* Transport Lines Info */}
      {transport_lines && transport_lines.length > 0 && (
        <div className="transport-lines-panel" style={{
          background: '#f8f9fa',
          padding: '8px 12px',
          display: 'flex',
          gap: '12px',
          flexWrap: 'wrap',
          fontSize: '12px',
          borderBottom: '1px solid #e0e0e0'
        }}>
          {transport_lines.map((line, idx) => (
            <div key={idx} style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              padding: '4px 8px',
              borderRadius: '4px',
              background: line.color || '#3b82f6',
              color: 'white',
              fontWeight: 'bold'
            }}>
              {line.line}: {line.name}
            </div>
          ))}
        </div>
      )}

      {/* Leaflet Map */}
      <div style={{ height, width: '100%', borderRadius: '0 0 8px 8px', overflow: 'hidden' }}>
        <MapContainer
          center={defaultCenter}
          zoom={zoom}
          style={{ height: '100%', width: '100%' }}
          zoomControl={true}
          attributionControl={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            maxZoom={19}
          />

          {/* Auto-fit bounds */}
          <AutoFitBounds coordinates={routeCoordinates} markers={markers} />

          {/* Draw route polyline */}
          {routeCoordinates.length > 0 && (
            <Polyline
              positions={routeCoordinates}
              color={getRouteColor()}
              weight={5}
              opacity={0.7}
              smoothFactor={1}
            />
          )}

          {/* Draw markers */}
          {markers && markers.map((marker, idx) => {
            if (!marker.lat || !marker.lon) return null;
            
            const position = [marker.lat, marker.lon];
            const markerType = marker.type || (idx === 0 ? 'start' : idx === markers.length - 1 ? 'end' : 'stop');
            const icon = createMarkerIcon(markerType, idx === 0 ? '🚩' : idx === markers.length - 1 ? '🏁' : '');

            return (
              <Marker key={idx} position={position} icon={icon}>
                <Popup>
                  <div style={{ minWidth: '150px' }}>
                    <strong style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>
                      {marker.label || `Stop ${idx + 1}`}
                    </strong>
                    {marker.type && (
                      <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                        Type: {marker.type}
                      </div>
                    )}
                    <div style={{ fontSize: '11px', color: '#999', marginTop: '4px' }}>
                      {position[0].toFixed(5)}, {position[1].toFixed(5)}
                    </div>
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>

      <style jsx>{`
        .map-visualization-container {
          margin: 16px 0;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          overflow: hidden;
        }

        .route-info-panel,
        .transport-lines-panel {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }

        @media (max-width: 768px) {
          .route-info-panel {
            padding: 10px 12px;
            font-size: 13px;
          }

          .route-info-panel > div {
            font-size: 12px;
          }

          .transport-lines-panel {
            font-size: 11px;
          }
        }
      `}</style>
    </div>
  );
};

export default MapVisualization;
