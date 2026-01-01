/**
 * ChatMapView Component - Displays map data from AI chat responses
 * Renders locations (restaurants, attractions, neighborhoods) on an interactive map
 * Phase 2: Includes user location, route visualization, and filters
 */

import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import { useLocation } from '../contexts/LocationContext';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default markers for React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom icons for different location types
const createCustomIcon = (type, color) => {
  const icons = {
    restaurant: '🍽️',
    attraction: '🏛️',
    neighborhood: '🏘️',
    poi: '📍',
    hotel: '🏨',
    cafe: '☕',
    bar: '🍺',
    museum: '🏛️',
    park: '🌳',
    shopping: '🛍️',
    origin: '🟢',
    destination: '🏁',
    default: '📍'
  };

  const icon = icons[type] || icons.default;
  const bgColor = color || (type === 'origin' ? '#10b981' : type === 'destination' ? '#ef4444' : '#3B82F6');
  
  return L.divIcon({
    html: `
      <div style="
        background-color: ${bgColor}; 
        width: 36px; 
        height: 36px; 
        border-radius: 50%; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        border: 3px solid white; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.3); 
        font-size: 18px;
        cursor: pointer;
      ">${icon}</div>
    `,
    className: 'custom-chat-map-icon',
    iconSize: [36, 36],
    iconAnchor: [18, 18],
    popupAnchor: [0, -18]
  });
};

// User location icon with pulse animation
const userLocationIcon = L.divIcon({
  html: `
    <div class="user-location-marker">
      <div class="user-location-pulse"></div>
      <div class="user-location-dot">
        <span style="font-size: 14px;">📍</span>
      </div>
    </div>
  `,
  className: 'user-location-icon-container',
  iconSize: [24, 24],
  iconAnchor: [12, 12],
  popupAnchor: [0, -12]
});

// Auto-fit bounds component (updated to include user location and routes)
const AutoFitBounds = ({ locations, userLocation, routePolyline, routes }) => {
  const map = useMap();

  useEffect(() => {
    const allPoints = [];
    
    // Add location markers
    if (locations && locations.length > 0) {
      locations.forEach(loc => allPoints.push([loc.lat, loc.lon]));
    }
    
    // Add user location if available
    if (userLocation) {
      allPoints.push([userLocation.lat, userLocation.lng]);
    }
    
    // Add route polyline points if available (legacy single route)
    if (routePolyline && routePolyline.length > 0) {
      routePolyline.forEach(point => allPoints.push([point.lat, point.lng]));
    }
    
    // Add all routes if available (new multi-route support)
    if (routes && routes.length > 0) {
      routes.forEach(route => {
        if (route.coordinates && route.coordinates.length > 0) {
          route.coordinates.forEach(coord => {
            allPoints.push([coord.lat, coord.lng]);
          });
        }
      });
    }
    
    if (allPoints.length > 0) {
      const leafletBounds = L.latLngBounds(allPoints);
      map.fitBounds(leafletBounds, { padding: [50, 50], maxZoom: 15 });
    }
  }, [locations, userLocation, routePolyline, routes, map]);

  return null;
};

const ChatMapView = ({ mapData, darkMode = false, showRoute = true, showUserLocation = true }) => {
  const [activeFilters, setActiveFilters] = useState(['restaurant', 'attraction', 'cafe', 'bar', 'museum', 'park', 'shopping', 'hotel', 'neighborhood', 'poi', 'origin', 'destination']);
  const { currentLocation, hasLocation } = useLocation();

  // Extract data from map_data
  const locations = mapData?.locations || [];
  const markers = mapData?.markers || []; // NEW: Support for route markers from backend
  const routePolyline = mapData?.route_polyline || null; // Legacy single route support
  const routes = mapData?.routes || []; // NEW: Support for multiple routes
  const center = mapData?.center || { lat: 41.0082, lon: 28.9784 }; // Istanbul center as fallback
  
  // Combine locations and markers for display
  const allMarkers = [...locations, ...markers.map(m => ({
    lat: m.lat,
    lon: m.lon,
    type: m.type || 'poi',
    title: m.title || m.label,
    description: m.description,
    metadata: { color: m.color }
  }))];
  
  // Determine tile layer based on theme
  const tileUrl = darkMode
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
  
  const tileAttribution = darkMode
    ? '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>'
    : '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

  // Filter toggle handler
  const toggleFilter = (type) => {
    setActiveFilters(prev => 
      prev.includes(type) 
        ? prev.filter(t => t !== type)
        : [...prev, type]
    );
  };

  // Get unique location types from current locations and markers
  const availableTypes = [...new Set(allMarkers.map(loc => loc.type))];
  
  // Filter type configurations
  const filterConfig = {
    restaurant: { icon: '🍽️', label: 'Restaurants', color: '#f59e0b' },
    attraction: { icon: '🏛️', label: 'Attractions', color: '#3b82f6' },
    cafe: { icon: '☕', label: 'Cafes', color: '#a855f7' },
    bar: { icon: '🍺', label: 'Bars', color: '#ef4444' },
    museum: { icon: '🏛️', label: 'Museums', color: '#06b6d4' },
    park: { icon: '🌳', label: 'Parks', color: '#10b981' },
    shopping: { icon: '🛍️', label: 'Shopping', color: '#ec4899' },
    hotel: { icon: '🏨', label: 'Hotels', color: '#8b5cf6' },
    neighborhood: { icon: '🏘️', label: 'Areas', color: '#6366f1' },
    poi: { icon: '📍', label: 'POIs', color: '#64748b' },
    origin: { icon: '🟢', label: 'Start', color: '#10b981' },
    destination: { icon: '🏁', label: 'End', color: '#ef4444' }
  };

  if (!allMarkers || allMarkers.length === 0) {
    return (
      <div style={{
        padding: '20px',
        textAlign: 'center',
        backgroundColor: darkMode ? '#1e293b' : '#f3f4f6',
        borderRadius: '8px',
        color: darkMode ? '#e2e8f0' : '#374151'
      }}>
        <p>No location data available to display on map.</p>
      </div>
    );
  }

  return (
    <div style={{ 
      width: '100%', 
      height: '400px', 
      marginTop: '12px',
      borderRadius: '12px',
      overflow: 'hidden',
      boxShadow: darkMode 
        ? '0 4px 6px rgba(0, 0, 0, 0.3)' 
        : '0 4px 6px rgba(0, 0, 0, 0.1)'
    }}>
      <MapContainer 
        center={[center.lat, center.lon]} 
        zoom={13} 
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution={tileAttribution}
          url={tileUrl}
        />
        
        <AutoFitBounds 
          locations={allMarkers} 
          userLocation={showUserLocation && hasLocation ? currentLocation : null}
          routePolyline={routePolyline}
          routes={routes}
        />
        
        {/* Multiple Routes - NEW: Support for multiple transportation routes */}
        {routes && routes.length > 0 && routes.map((route, index) => (
          <Polyline
            key={`route-${index}`}
            positions={route.coordinates.map(coord => [coord.lat, coord.lng])}
            pathOptions={{
              color: route.color || '#3b82f6',
              weight: route.weight || 4,
              opacity: route.opacity || 0.8,
              dashArray: route.isMain ? undefined : '10, 5',
              lineJoin: 'round',
              lineCap: 'round'
            }}
          >
            {route.description && (
              <Popup>
                <div style={{ padding: '8px', textAlign: 'center' }}>
                  <h4 style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: '600' }}>
                    {route.description}
                  </h4>
                </div>
              </Popup>
            )}
          </Polyline>
        ))}
        
        {/* Route Polyline - Legacy single route support */}
        {routePolyline && routePolyline.length > 0 && routes.length === 0 && (
          <Polyline
            positions={routePolyline.map(point => [point.lat, point.lng])}
            pathOptions={{
              color: '#3b82f6',
              weight: 4,
              opacity: 0.8,
              dashArray: '10, 5',
              lineJoin: 'round',
              lineCap: 'round'
            }}
          />
        )}
        
        {/* Route Visualization - Simple path between multiple locations (not for transportation) */}
        {showRoute && !routePolyline && routes.length === 0 && allMarkers.length > 1 && (
          <Polyline
            positions={allMarkers.map(loc => [loc.lat, loc.lon])}
            pathOptions={{
              color: '#8a2be2',
              weight: 3,
              opacity: 0.6,
              dashArray: '10, 10',
              lineJoin: 'round'
            }}
          />
        )}
        
        {/* User Location Marker */}
        {showUserLocation && hasLocation && currentLocation && (
          <Marker
            position={[currentLocation.lat, currentLocation.lng]}
            icon={userLocationIcon}
            zIndexOffset={1000}
          >
            <Popup>
              <div style={{ padding: '8px', textAlign: 'center' }}>
                <h4 style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: '600' }}>
                  📍 You are here
                </h4>
                <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>
                  Current location
                </p>
              </div>
            </Popup>
          </Marker>
        )}
        
        {/* Location Markers - Filtered */}
        {allMarkers
          .filter(loc => activeFilters.includes(loc.type))
          .map((location, idx) => (
          <Marker 
            key={idx} 
            position={[location.lat, location.lon]}
            icon={createCustomIcon(location.type, location.metadata?.color)}
          >
            <Popup maxWidth={300}>
              <div style={{ 
                padding: '8px', 
                maxWidth: '280px',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
              }}>
                <h3 style={{ 
                  margin: '0 0 8px 0', 
                  fontSize: '16px', 
                  fontWeight: '600',
                  color: '#1f2937'
                }}>
                  {location.name}
                </h3>
                
                {location.metadata?.description && (
                  <p style={{ 
                    margin: '4px 0', 
                    fontSize: '14px', 
                    color: '#4b5563',
                    lineHeight: '1.4'
                  }}>
                    {location.metadata.description}
                  </p>
                )}
                
                {location.metadata?.address && (
                  <p style={{ 
                    margin: '4px 0', 
                    fontSize: '13px', 
                    color: '#6b7280',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    <span>📍</span>
                    {location.metadata.address}
                  </p>
                )}
                
                <div style={{ 
                  display: 'flex', 
                  gap: '12px', 
                  marginTop: '8px',
                  flexWrap: 'wrap'
                }}>
                  {location.metadata?.rating && (
                    <span style={{ 
                      fontSize: '13px', 
                      color: '#059669',
                      fontWeight: '500'
                    }}>
                      ⭐ {location.metadata.rating}
                    </span>
                  )}
                  
                  {location.metadata?.price && (
                    <span style={{ 
                      fontSize: '13px', 
                      color: '#7c3aed',
                      fontWeight: '500'
                    }}>
                      {location.metadata.price}
                    </span>
                  )}
                </div>
                
                {location.metadata?.hours && (
                  <p style={{ 
                    margin: '6px 0 0 0', 
                    fontSize: '12px', 
                    color: '#6b7280'
                  }}>
                    🕒 {location.metadata.hours}
                  </p>
                )}
                
                <a
                  href={`https://www.google.com/maps/search/?api=1&query=${location.lat},${location.lon}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    display: 'inline-block',
                    marginTop: '8px',
                    padding: '6px 12px',
                    backgroundColor: '#3B82F6',
                    color: 'white',
                    textDecoration: 'none',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: '500',
                    transition: 'background-color 0.2s'
                  }}
                  onMouseEnter={(e) => e.target.style.backgroundColor = '#2563EB'}
                  onMouseLeave={(e) => e.target.style.backgroundColor = '#3B82F6'}
                >
                  Open in Google Maps →
                </a>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      
      {/* Filter Controls - Only show if there are multiple types */}
      {availableTypes.length > 1 && (
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          background: darkMode ? 'rgba(30, 41, 59, 0.95)' : 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          borderRadius: '12px',
          padding: '8px',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          zIndex: 1000,
          maxWidth: '200px'
        }}>
          <div style={{
            fontSize: '11px',
            fontWeight: '600',
            color: darkMode ? '#e2e8f0' : '#374151',
            marginBottom: '6px',
            paddingLeft: '4px'
          }}>
            Filter Locations
          </div>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '4px'
          }}>
            {availableTypes.map(type => {
              const config = filterConfig[type] || filterConfig.poi;
              const isActive = activeFilters.includes(type);
              
              return (
                <button
                  key={type}
                  onClick={() => toggleFilter(type)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 8px',
                    background: isActive 
                      ? (darkMode ? 'rgba(138, 43, 226, 0.3)' : 'rgba(138, 43, 226, 0.1)') 
                      : 'transparent',
                    border: isActive 
                      ? `1px solid ${config.color}` 
                      : `1px solid ${darkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                    borderRadius: '8px',
                    color: darkMode ? '#e2e8f0' : '#374151',
                    fontSize: '12px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    opacity: isActive ? 1 : 0.6
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.opacity = '1';
                    e.target.style.transform = 'translateX(-2px)';
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.opacity = isActive ? '1' : '0.6';
                    e.target.style.transform = 'translateX(0)';
                  }}
                >
                  <span>{config.icon}</span>
                  <span>{config.label}</span>
                  {isActive && <span style={{ marginLeft: 'auto', fontSize: '10px' }}>✓</span>}
                </button>
              );
            })}
          </div>
          
          {/* Stats */}
          <div style={{
            marginTop: '8px',
            paddingTop: '8px',
            borderTop: `1px solid ${darkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
            fontSize: '10px',
            color: darkMode ? '#94a3b8' : '#6b7280',
            textAlign: 'center'
          }}>
            Showing {locations.filter(loc => activeFilters.includes(loc.type)).length} of {locations.length} locations
          </div>
        </div>
      )}
    </div>
  );
};

export default React.memo(ChatMapView);
