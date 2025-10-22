/**
 * ChatMapView Component - Displays map data from AI chat responses
 * Renders locations (restaurants, attractions, neighborhoods) on an interactive map
 */

import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
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
    default: '📍'
  };

  const icon = icons[type] || icons.default;
  const bgColor = color || '#3B82F6';
  
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

// Auto-fit bounds component
const AutoFitBounds = ({ locations }) => {
  const map = useMap();

  useEffect(() => {
    if (locations && locations.length > 0) {
      const bounds = locations.map(loc => [loc.lat, loc.lon]);
      const leafletBounds = L.latLngBounds(bounds);
      map.fitBounds(leafletBounds, { padding: [50, 50], maxZoom: 15 });
    }
  }, [locations, map]);

  return null;
};

const ChatMapView = ({ mapData, darkMode = false }) => {
  const [expandedMarker, setExpandedMarker] = useState(null);

  // Extract data from map_data
  const locations = mapData?.locations || [];
  const center = mapData?.center || { lat: 41.0082, lon: 28.9784 }; // Istanbul center as fallback
  
  // Determine tile layer based on theme
  const tileUrl = darkMode
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
  
  const tileAttribution = darkMode
    ? '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>'
    : '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

  if (!locations || locations.length === 0) {
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
        
        <AutoFitBounds locations={locations} />
        
        {locations.map((location, idx) => (
          <Marker 
            key={idx} 
            position={[location.lat, location.lon]}
            icon={createCustomIcon(location.type, location.metadata?.color)}
            eventHandlers={{
              click: () => setExpandedMarker(idx),
            }}
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
    </div>
  );
};

export default React.memo(ChatMapView);
