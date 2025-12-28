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

// Custom marker icons with colors - Moovit-style minimal design
const createMarkerIcon = (type = 'default', color = null, emoji = '') => {
  const colors = {
    start: '#00C853',      // Vibrant green
    end: '#FF1744',        // Vibrant red
    destination: '#FF1744', // Red
    origin: '#00C853',     // Green
    transfer: '#FF9100',   // Vibrant amber
    stop: '#2979FF',       // Vibrant blue
    attraction: '#D500F9', // Vibrant purple
    default: '#757575'     // Gray
  };
  
  const bgColor = color || colors[type] || colors.default;
  const displayEmoji = emoji || (type === 'origin' ? '🚩' : type === 'destination' ? '🏁' : type === 'transfer' ? '🔄' : '📍');
  
  // Moovit-style: very small, minimal markers with clean shadow
  const iconHtml = `
    <div style="
      background-color: ${bgColor};
      width: 28px;
      height: 28px;
      border-radius: 50% 50% 50% 0;
      transform: rotate(-45deg);
      display: flex;
      align-items: center;
      justify-content: center;
      border: 3px solid white;
      box-shadow: 0 3px 8px rgba(0,0,0,0.25);
    ">
      <div style="
        transform: rotate(45deg);
        font-size: 14px;
        margin-bottom: 3px;
      ">
        ${displayEmoji}
      </div>
    </div>
  `;
  
  return L.divIcon({
    html: iconHtml,
    className: 'custom-marker-icon',
    iconSize: [28, 28],
    iconAnchor: [14, 28],
    popupAnchor: [0, -28]
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
      {/* Route Information Header - Moovit-style minimal design */}
      {route_data && (
        <div className="route-info-panel" style={{
          background: 'white',
          color: '#1a1a1a',
          padding: '16px 20px',
          borderBottom: '1px solid #e5e5e5',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '12px',
                background: 'linear-gradient(135deg, #2979FF 0%, #1E88E5 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '20px',
                boxShadow: '0 2px 8px rgba(41, 121, 255, 0.3)'
              }}>
                🗺️
              </div>
              <div>
                <div style={{ fontWeight: '600', fontSize: '15px', color: '#1a1a1a', marginBottom: '4px' }}>
                  {route_data.origin} → {route_data.destination}
                </div>
                <div style={{ fontSize: '13px', color: '#666' }}>
                  {isMultiRoute && allRouteOptions.length > 0
                    ? `${allRouteOptions.length} route options`
                    : route_data.transfers === 0 
                      ? 'Direct route' 
                      : `${route_data.transfers} transfer${route_data.transfers > 1 ? 's' : ''}`
                  }
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '16px', fontSize: '14px', color: '#1a1a1a' }}>
              {route_data.duration_min && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '16px' }}>⏱️</span>
                  <span style={{ fontWeight: '600' }}>{Math.round(route_data.duration_min)} min</span>
                </div>
              )}
              {route_data.distance_km && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '16px' }}>📍</span>
                  <span style={{ fontWeight: '600' }}>{route_data.distance_km.toFixed(1)} km</span>
                </div>
              )}
            </div>
          </div>
          
          {/* Multi-route selector - Moovit-style pills */}
          {isMultiRoute && allRouteOptions.length > 0 && (
            <div style={{ marginTop: '16px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
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
                      padding: '10px 14px',
                      borderRadius: '12px',
                      border: isActive ? '2px solid #2979FF' : '2px solid #e5e5e5',
                      background: isActive ? '#F0F7FF' : 'white',
                      color: isActive ? '#2979FF' : '#1a1a1a',
                      fontSize: '13px',
                      fontWeight: isActive ? '600' : '500',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      minWidth: '120px',
                      boxShadow: isActive ? '0 2px 8px rgba(41, 121, 255, 0.15)' : 'none'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                      <span style={{ fontSize: '16px' }}>
                        {idx === 0 ? '⚡' : idx === 1 ? '🔄' : idx === 2 ? '🛋️' : '⚖️'}
                      </span>
                      <span style={{ textTransform: 'capitalize' }}>
                        {routeOpt.preference || `Option ${idx + 1}`}
                      </span>
                    </div>
                    <div style={{ fontSize: '12px', color: isActive ? '#2979FF' : '#666', marginBottom: '4px' }}>
                      {routeOpt.duration_minutes}' • {routeOpt.num_transfers} transfers
                    </div>
                    <div style={{ 
                      fontSize: '11px',
                      padding: '2px 8px',
                      borderRadius: '8px',
                      background: isActive ? '#2979FF' : '#f5f5f5',
                      color: isActive ? 'white' : '#666',
                      fontWeight: '600'
                    }}>
                      Comfort {Math.round(comfortScore)}/100
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Transport Lines Legend - Moovit-style minimal */}
      {transport_lines && transport_lines.length > 0 && (
        <div className="transport-lines-panel" style={{
          background: 'white',
          padding: '12px 20px',
          display: 'flex',
          gap: '8px',
          flexWrap: 'wrap',
          fontSize: '13px',
          borderBottom: '1px solid #e5e5e5'
        }}>
          <span style={{ fontSize: '12px', color: '#666', fontWeight: '500', marginRight: '4px' }}>
            Lines:
          </span>
          {transport_lines.map((line, idx) => (
            <div key={idx} style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '12px',
              background: line.color || '#2979FF',
              color: 'white',
              fontWeight: '600',
              fontSize: '12px',
              boxShadow: '0 1px 4px rgba(0,0,0,0.1)'
            }}>
              <span style={{ fontSize: '14px' }}>
                {line.type === 'metro' ? '🚇' : line.type === 'tram' ? '🚊' : line.type === 'ferry' ? '⛴️' : '🚆'}
              </span>
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
          attributionControl={false}
        >
          {/* Use ultra-light, minimal map style (Moovit/Apple Maps-like) */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
            maxZoom={19}
          />

          {/* Auto-fit bounds */}
          <AutoFitBounds coordinates={coordinates} markers={markers} routes={routes} />

          {/* Draw color-coded route segments - Moovit-style clean lines */}
          {displayRoutes && displayRoutes.map((segment, idx) => {
            if (!segment.coordinates || segment.coordinates.length < 2) return null;
            
            // Normalize coordinates to [lat, lng] array format
            const normalizedPositions = segment.coordinates
              .map(normalizeCoord)
              .filter(Boolean);
            
            if (normalizedPositions.length < 2) return null;
            
            // Moovit-style: bold, clean route lines with subtle shadows
            const opacity = isMultiRoute ? 0.9 : (segment.opacity || 0.85);
            const weight = isMultiRoute ? 8 : (segment.weight || 7);
            
            return (
              <React.Fragment key={`segment-${idx}`}>
                {/* Shadow/outline for depth */}
                <Polyline
                  positions={normalizedPositions}
                  color="#000000"
                  weight={weight + 2}
                  opacity={0.15}
                  smoothFactor={1.5}
                  lineCap="round"
                  lineJoin="round"
                />
                {/* Main route line */}
                <Polyline
                  positions={normalizedPositions}
                  color={segment.color || '#2979FF'}
                  weight={weight}
                  opacity={opacity}
                  smoothFactor={1.5}
                  lineCap="round"
                  lineJoin="round"
                />
              </React.Fragment>
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

          {/* Draw markers - Moovit-style minimal design */}
          {markers && markers.map((marker, idx) => {
            if (!marker.lat || !marker.lon) return null;
            
            const position = [marker.lat, marker.lon];
            const markerType = marker.type || 'default';
            const icon = createMarkerIcon(markerType, marker.color);

            return (
              <Marker key={`marker-${idx}`} position={position} icon={icon}>
                <Popup className="moovit-style-popup" minWidth={200} maxWidth={300}>
                  <div style={{ 
                    padding: '8px 4px',
                    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                  }}>
                    <div style={{ 
                      fontSize: '14px', 
                      fontWeight: '600', 
                      marginBottom: '6px',
                      color: '#1a1a1a'
                    }}>
                      {marker.title || marker.label || `Stop ${idx + 1}`}
                    </div>
                    {marker.description && (
                      <div style={{ 
                        fontSize: '12px', 
                        color: '#666', 
                        lineHeight: '1.4',
                        marginTop: '4px'
                      }}>
                        {marker.description}
                      </div>
                    )}
                    {marker.line && (
                      <div style={{ 
                        fontSize: '11px', 
                        marginTop: '8px',
                        padding: '3px 8px',
                        borderRadius: '10px',
                        background: marker.color || '#2979FF',
                        color: 'white',
                        display: 'inline-block',
                        fontWeight: '600',
                        letterSpacing: '0.3px'
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

      {/* Alternative Routes - Moovit-style minimal */}
      {alternatives && alternatives.length > 0 && (
        <div style={{
          background: '#fafafa',
          padding: '14px 20px',
          borderTop: '1px solid #e5e5e5',
          borderRadius: '0 0 12px 12px'
        }}>
          <div style={{ fontSize: '12px', fontWeight: '600', marginBottom: '10px', color: '#666' }}>
            Alternative Routes
          </div>
          {alternatives.map((alt, idx) => (
            <div key={idx} style={{
              fontSize: '13px',
              padding: '8px 12px',
              background: 'white',
              borderRadius: '10px',
              marginBottom: '6px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              border: '1px solid #e5e5e5',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
            }}>
              <span style={{ color: '#1a1a1a', fontWeight: '500' }}>
                {alt.summary || `${alt.lines_used?.join(' → ')} (${alt.total_time} min)`}
              </span>
              <span style={{ 
                color: '#666',
                fontSize: '12px',
                padding: '2px 8px',
                borderRadius: '8px',
                background: '#f5f5f5'
              }}>
                +{alt.total_time - (route_data?.duration_min || 0)} min
              </span>
            </div>
          ))}
        </div>
      )}

      <style jsx>{`
        .map-visualization-container {
          margin: 16px 0;
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
          overflow: hidden;
          background: white;
        }

        .route-info-panel,
        .transport-lines-panel {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }

        /* Moovit-style minimal popup styling */
        :global(.moovit-style-popup .leaflet-popup-content-wrapper) {
          border-radius: 12px;
          box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
          padding: 0;
        }

        :global(.moovit-style-popup .leaflet-popup-content) {
          margin: 0;
        }

        :global(.moovit-style-popup .leaflet-popup-tip) {
          border-radius: 2px;
        }

        /* Clean marker icons */
        :global(.custom-marker-icon) {
          border: none;
          background: transparent;
        }

        /* Improve zoom controls */
        :global(.leaflet-control-zoom) {
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
          border: none;
        }

        :global(.leaflet-control-zoom a) {
          width: 36px;
          height: 36px;
          line-height: 36px;
          font-size: 20px;
          border: none;
          color: #333;
        }

        :global(.leaflet-control-zoom a:hover) {
          background: #f0f0f0;
        }

        @media (max-width: 768px) {
          .route-info-panel {
            padding: 10px 12px;
          }

          .transport-lines-panel {
            font-size: 11px;
          }

          :global(.leaflet-control-zoom a) {
            width: 32px;
            height: 32px;
            line-height: 32px;
          }
        }
      `}</style>
    </div>
  );
};

export default MapVisualization;
