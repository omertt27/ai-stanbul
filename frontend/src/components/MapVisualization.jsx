/**
 * MapVisualization Component
 * ==========================
 * Google Maps-style interactive route display with:
 * - Color-coded transit line segments
 * - Origin/transfer/destination markers
 * - Auto-fit bounds
 * - Route info panel
 * - Alternative routes display
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
const createMarkerIcon = (type = 'default', color = null, emoji = '') => {
  const colors = {
    start: '#22c55e',      // Green
    end: '#ef4444',        // Red
    destination: '#ef4444', // Red
    origin: '#22c55e',     // Green
    transfer: '#f59e0b',   // Orange/Amber
    stop: '#3b82f6',       // Blue
    attraction: '#8b5cf6', // Purple
    default: '#6b7280'     // Gray
  };
  
  const bgColor = color || colors[type] || colors.default;
  const displayEmoji = emoji || (type === 'origin' ? '🚩' : type === 'destination' ? '🏁' : type === 'transfer' ? '🔄' : '📍');
  
  const iconHtml = `
    <div style="
      background-color: ${bgColor};
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 3px solid white;
      box-shadow: 0 3px 8px rgba(0,0,0,0.4);
      font-size: 18px;
    ">
      ${displayEmoji}
    </div>
  `;
  
  return L.divIcon({
    html: iconHtml,
    className: 'custom-marker-icon',
    iconSize: [36, 36],
    iconAnchor: [18, 18],
    popupAnchor: [0, -18]
  });
};

// Helper function to normalize coordinates to [lat, lng] array format
const normalizeCoord = (coord) => {
  if (Array.isArray(coord) && coord.length === 2) {
    return [coord[0], coord[1]];
  }
  if (coord && typeof coord === 'object') {
    const lat = coord.lat ?? coord.latitude;
    const lng = coord.lng ?? coord.lon ?? coord.longitude;
    if (lat !== undefined && lng !== undefined) {
      return [lat, lng];
    }
  }
  return null;
};

// Auto-fit bounds component
const AutoFitBounds = ({ coordinates, markers, routes }) => {
  const map = useMap();

  useEffect(() => {
    if (!map) return;

    const bounds = [];
    
    // Add route coordinates from all segments
    if (routes && routes.length > 0) {
      routes.forEach(route => {
        if (route.coordinates) {
          route.coordinates.forEach(coord => {
            const normalized = normalizeCoord(coord);
            if (normalized) {
              bounds.push(normalized);
            }
          });
        }
      });
    }
    
    // Add flat coordinates
    if (coordinates && coordinates.length > 0) {
      coordinates.forEach(coord => {
        const normalized = normalizeCoord(coord);
        if (normalized) {
          bounds.push(normalized);
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
  }, [map, coordinates, markers, routes]);

  return null;
};

const MapVisualization = ({ 
  mapData, 
  height = '400px',
  className = '',
  selectedRouteIndex = null,
  onRouteHover = null
}) => {
  const [error, setError] = useState(null);
  const [activeRouteIndex, setActiveRouteIndex] = useState(selectedRouteIndex ?? 0);

  useEffect(() => {
    if (selectedRouteIndex !== null) {
      setActiveRouteIndex(selectedRouteIndex);
    }
  }, [selectedRouteIndex]);

  if (!mapData) {
    return null;
  }

  // Extract data from mapData
  const {
    type = 'route',
    coordinates = [],
    markers = [],
    routes = [],
    center = { lat: 41.0082, lon: 28.9784 }, // Istanbul default
    zoom = 13,
    route_data = null,
    transport_lines = [],
    alternatives = [],
    // Multi-route specific fields
    multi_routes = [],  // Array of complete route objects with comfort scores
    primary_route = null,
    route_comparison = {}
  } = mapData;

  // Check if this is a multi-route map
  const isMultiRoute = type === 'multi_route' || multi_routes.length > 0;
  
  // Use multi_routes if available, otherwise fall back to alternatives
  const allRouteOptions = multi_routes.length > 0 ? multi_routes : alternatives;
  
  // Determine which routes to display on map
  const displayRoutes = isMultiRoute 
    ? (allRouteOptions[activeRouteIndex]?.routes || routes)
    : routes;

  // Comfort score calculation (example: average of all segments' weights)
  const comfortScores = isMultiRoute 
    ? allRouteOptions.map(option => {
        const totalWeight = (option.routes || []).reduce((sum, route) => sum + (route.weight || 0), 0);
        const segmentCount = (option.routes || []).length;
        return segmentCount > 0 ? totalWeight / segmentCount : 0;
      })
    : [];

  // Find the primary route index in the multi-route options (if available)
  const primaryRouteIndex = isMultiRoute 
    ? allRouteOptions.findIndex(option => option.routes?.some(route => route.is_primary))
    : -1;

  return (
    <div className={`map-visualization-container ${className}`}>
      {/* Route Information Header */}
      {route_data && (
        <div className="route-info-panel" style={{
          background: 'linear-gradient(135deg, #1a73e8 0%, #4285f4 100%)',
          color: 'white',
          padding: '14px 18px',
          borderRadius: '8px 8px 0 0',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '24px' }}>🗺️</span>
              <div>
                <div style={{ fontWeight: 'bold', fontSize: '16px' }}>
                  {route_data.origin} → {route_data.destination}
                </div>
                <div style={{ fontSize: '13px', opacity: 0.9 }}>
                  {isMultiRoute && allRouteOptions.length > 0
                    ? `${allRouteOptions.length} route options available`
                    : route_data.transfers === 0 
                      ? 'Direct' 
                      : `${route_data.transfers} transfer${route_data.transfers > 1 ? 's' : ''}`
                  }
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '20px', fontSize: '15px', fontWeight: '500' }}>
              {route_data.duration_min && (
                <span>⏱️ {Math.round(route_data.duration_min)} min</span>
              )}
              {route_data.distance_km && (
                <span>📍 {route_data.distance_km.toFixed(1)} km</span>
              )}
            </div>
          </div>
          
          {/* Multi-route selector */}
          {isMultiRoute && allRouteOptions.length > 0 && (
            <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {allRouteOptions.map((routeOpt, idx) => {
                const isActive = idx === activeRouteIndex;
                const comfortScore = routeOpt.comfort_score?.overall_comfort || 0;
                
                return (
                  <button
                    key={idx}
                    onClick={() => setActiveRouteIndex(idx)}
                    onMouseEnter={() => onRouteHover && onRouteHover(idx)}
                    onMouseLeave={() => onRouteHover && onRouteHover(null)}
                    style={{
                      padding: '8px 12px',
                      borderRadius: '6px',
                      border: isActive ? '2px solid white' : '2px solid transparent',
                      background: isActive ? 'rgba(255, 255, 255, 0.2)' : 'rgba(255, 255, 255, 0.1)',
                      color: 'white',
                      fontSize: '13px',
                      fontWeight: isActive ? 'bold' : 'normal',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      minWidth: '120px'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                      <span>{idx === 0 ? '⚡' : idx === 1 ? '🔄' : idx === 2 ? '🛋️' : '⚖️'}</span>
                      <span style={{ textTransform: 'capitalize' }}>
                        {routeOpt.preference || `Option ${idx + 1}`}
                      </span>
                    </div>
                    <div style={{ fontSize: '11px', opacity: 0.9 }}>
                      {routeOpt.duration_minutes}'  • {routeOpt.num_transfers} transfers
                    </div>
                    <div style={{ fontSize: '11px', opacity: 0.9 }}>
                      Comfort: {Math.round(comfortScore)}/100
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Transport Lines Legend */}
      {transport_lines && transport_lines.length > 0 && (
        <div className="transport-lines-panel" style={{
          background: '#f8f9fa',
          padding: '10px 14px',
          display: 'flex',
          gap: '10px',
          flexWrap: 'wrap',
          fontSize: '13px',
          borderBottom: '1px solid #e0e0e0'
        }}>
          {transport_lines.map((line, idx) => (
            <div key={idx} style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '5px 10px',
              borderRadius: '16px',
              background: line.color || '#3b82f6',
              color: 'white',
              fontWeight: 'bold',
              fontSize: '12px'
            }}>
              <span>{line.type === 'metro' ? '🚇' : line.type === 'tram' ? '🚊' : line.type === 'ferry' ? '⛴️' : '🚆'}</span>
              <span>{line.line}</span>
            </div>
          ))}
        </div>
      )}

      {/* Leaflet Map */}
      <div style={{ height, width: '100%', borderRadius: routes?.length > 0 || transport_lines?.length > 0 ? '0' : '8px 8px 0 0', overflow: 'hidden' }}>
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
          <AutoFitBounds coordinates={coordinates} markers={markers} routes={routes} />

          {/* Draw color-coded route segments */}
          {displayRoutes && displayRoutes.map((segment, idx) => {
            if (!segment.coordinates || segment.coordinates.length < 2) return null;
            
            // Normalize coordinates to [lat, lng] array format
            const normalizedPositions = segment.coordinates
              .map(normalizeCoord)
              .filter(Boolean);
            
            if (normalizedPositions.length < 2) return null;
            
            // Adjust opacity for multi-route display
            const opacity = isMultiRoute ? 0.9 : (segment.opacity || 0.85);
            const weight = isMultiRoute ? 6 : (segment.weight || 5);
            
            return (
              <Polyline
                key={`segment-${idx}`}
                positions={normalizedPositions}
                color={segment.color || '#4285F4'}
                weight={weight}
                opacity={opacity}
                smoothFactor={1}
              />
            );
          })}

          {/* Fallback: Draw single polyline from flat coordinates */}
          {(!routes || routes.length === 0) && coordinates.length > 0 && (
            <Polyline
              positions={coordinates}
              color="#4285F4"
              weight={5}
              opacity={0.7}
              smoothFactor={1}
            />
          )}

          {/* Draw markers */}
          {markers && markers.map((marker, idx) => {
            if (!marker.lat || !marker.lon) return null;
            
            const position = [marker.lat, marker.lon];
            const markerType = marker.type || 'default';
            const icon = createMarkerIcon(markerType, marker.color);

            return (
              <Marker key={`marker-${idx}`} position={position} icon={icon}>
                <Popup>
                  <div style={{ minWidth: '180px' }}>
                    <strong style={{ fontSize: '15px', display: 'block', marginBottom: '6px' }}>
                      {marker.title || marker.label || `Stop ${idx + 1}`}
                    </strong>
                    {marker.description && (
                      <div style={{ fontSize: '13px', color: '#555', marginTop: '4px' }}>
                        {marker.description}
                      </div>
                    )}
                    {marker.line && (
                      <div style={{ 
                        fontSize: '12px', 
                        marginTop: '8px',
                        padding: '4px 8px',
                        borderRadius: '12px',
                        background: marker.color || '#4285F4',
                        color: 'white',
                        display: 'inline-block'
                      }}>
                        {marker.line}
                      </div>
                    )}
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>

      {/* Alternative Routes */}
      {alternatives && alternatives.length > 0 && (
        <div style={{
          background: '#fff',
          padding: '12px 16px',
          borderTop: '1px solid #e0e0e0',
          borderRadius: '0 0 8px 8px'
        }}>
          <div style={{ fontSize: '13px', fontWeight: '600', marginBottom: '8px', color: '#555' }}>
            Alternative Routes:
          </div>
          {alternatives.map((alt, idx) => (
            <div key={idx} style={{
              fontSize: '12px',
              padding: '6px 10px',
              background: '#f5f5f5',
              borderRadius: '6px',
              marginBottom: '4px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span>{alt.summary || `${alt.lines_used?.join(' → ')} (${alt.total_time} min)`}</span>
              <span style={{ color: '#666' }}>+{alt.total_time - (route_data?.duration_min || 0)} min</span>
            </div>
          ))}
        </div>
      )}

      <style jsx>{`
        .map-visualization-container {
          margin: 16px 0;
          border-radius: 8px;
          box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
          overflow: hidden;
        }

        .route-info-panel,
        .transport-lines-panel {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }

        @media (max-width: 768px) {
          .route-info-panel {
            padding: 10px 12px;
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
