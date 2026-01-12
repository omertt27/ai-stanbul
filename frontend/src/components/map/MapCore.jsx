/**
 * MapCore Component
 * =================
 * 
 * Reusable Leaflet map engine extracted from RouteCard, MapVisualization, and TransportationRouteCard.
 * Provides a unified map interface with all advanced features.
 * 
 * Features:
 * - Dark mode tile support (auto-detects system preference)
 * - Custom marker icons with colors and emojis
 * - Animated polylines for routes
 * - Interactive controls (zoom, recenter, fullscreen, geolocation)
 * - Auto-fit bounds
 * - Pulsing markers for transfer points
 * - Loading states
 * - Error handling
 * 
 * Props:
 * - markers: Array of marker objects { lat, lng, type, label, emoji, color }
 * - routes: Array of route polylines { coordinates, color, mode, animated }
 * - center: [lat, lng] - Map center point
 * - zoom: Number - Initial zoom level (default: 13)
 * - darkMode: Boolean - Use dark map tiles
 * - showControls: Boolean - Show interactive controls (default: true)
 * - height: String - Map height (default: '400px')
 * - onMapReady: Function - Callback when map is ready
 * - className: String - Additional CSS classes
 * 
 * Author: AI Istanbul Team
 * Date: January 12, 2026
 */

import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default markers for React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

/**
 * Custom marker icon creator
 */
const createMarkerIcon = (type = 'default', color = null, emoji = '') => {
  const colors = {
    start: '#00C853',
    origin: '#00C853',
    end: '#FF1744',
    destination: '#FF1744',
    transfer: '#FF9100',
    stop: '#2979FF',
    attraction: '#D500F9',
    restaurant: '#FF6F00',
    poi: '#795548',
    default: '#757575'
  };
  
  const emojis = {
    origin: '🚩',
    start: '🚩',
    destination: '🏁',
    end: '🏁',
    transfer: '🔄',
    stop: '📍',
    attraction: '🎭',
    restaurant: '🍽️',
    poi: '📌',
    default: '📍'
  };
  
  const bgColor = color || colors[type] || colors.default;
  const displayEmoji = emoji || emojis[type] || emojis.default;
  
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

/**
 * Pulsing marker icon for transfer points
 */
const createPulsingMarker = (color = '#4F46E5', emoji = '🔄') => {
  const iconHtml = `
    <div style="position: relative;">
      <div style="
        width: 20px; 
        height: 20px; 
        background: ${color}; 
        border-radius: 50%; 
        border: 3px solid white; 
        box-shadow: 0 0 10px rgba(79, 70, 229, 0.5);
        animation: pulse 2s ease-in-out infinite;
      "></div>
      <div style="
        position: absolute;
        top: 0;
        left: 0;
        width: 20px;
        height: 20px;
        background: ${color};
        border-radius: 50%;
        opacity: 0;
        animation: pulse-ring 2s ease-in-out infinite;
      "></div>
    </div>
    <style>
      @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
      }
      @keyframes pulse-ring {
        0% { transform: scale(1); opacity: 0.5; }
        100% { transform: scale(2); opacity: 0; }
      }
    </style>
  `;
  
  return L.divIcon({
    html: iconHtml,
    className: 'pulsing-marker-icon',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
    popupAnchor: [0, -10]
  });
};

/**
 * Animated Polyline Component
 */
const AnimatedPolyline = ({ positions, color, weight = 5, opacity = 0.8, speed = 50, animated = true }) => {
  const [visiblePoints, setVisiblePoints] = useState(animated ? 1 : positions.length);
  
  useEffect(() => {
    if (!animated || positions.length <= 1) {
      setVisiblePoints(positions.length);
      return;
    }
    
    const interval = setInterval(() => {
      setVisiblePoints(prev => {
        if (prev >= positions.length) {
          clearInterval(interval);
          return positions.length;
        }
        return prev + 1;
      });
    }, speed);
    
    return () => clearInterval(interval);
  }, [positions.length, speed, animated]);
  
  if (positions.length === 0) return null;
  
  return (
    <Polyline 
      positions={positions.slice(0, visiblePoints)}
      color={color}
      weight={weight}
      opacity={opacity}
    />
  );
};

/**
 * Auto-fit Bounds Component
 */
const AutoFitBounds = ({ markers, routes }) => {
  const map = useMap();

  useEffect(() => {
    if (!map) return;

    const allPoints = [];
    
    // Add marker points
    if (markers && markers.length > 0) {
      markers.forEach(marker => {
        const lat = marker.lat || marker.latitude;
        const lng = marker.lng || marker.lon || marker.longitude;
        if (lat && lng) {
          allPoints.push([lat, lng]);
        }
      });
    }
    
    // Add route points
    if (routes && routes.length > 0) {
      routes.forEach(route => {
        if (route.coordinates && Array.isArray(route.coordinates)) {
          route.coordinates.forEach(coord => {
            if (Array.isArray(coord) && coord.length === 2) {
              allPoints.push(coord);
            }
          });
        }
      });
    }
    
    if (allPoints.length > 0) {
      const bounds = L.latLngBounds(allPoints);
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
    }
  }, [map, markers, routes]);

  return null;
};

/**
 * Map Controls Component
 */
const MapControls = ({ onZoomIn, onZoomOut, onRecenter, onFullscreen, onGeolocation, darkMode }) => {
  return (
    <div className="absolute top-2 right-2 z-[1000] flex flex-col gap-2">
      <button
        onClick={onZoomIn}
        className={`p-2 rounded-lg shadow-lg transition-all hover:scale-105 ${
          darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-700'
        }`}
        title="Zoom in"
      >
        <span className="text-xl">🔍+</span>
      </button>
      <button
        onClick={onZoomOut}
        className={`p-2 rounded-lg shadow-lg transition-all hover:scale-105 ${
          darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-700'
        }`}
        title="Zoom out"
      >
        <span className="text-xl">🔍−</span>
      </button>
      <button
        onClick={onRecenter}
        className={`p-2 rounded-lg shadow-lg transition-all hover:scale-105 ${
          darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-700'
        }`}
        title="Recenter map"
      >
        <span className="text-xl">🎯</span>
      </button>
      <button
        onClick={onFullscreen}
        className={`p-2 rounded-lg shadow-lg transition-all hover:scale-105 ${
          darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-700'
        }`}
        title="Fullscreen"
      >
        <span className="text-xl">⛶</span>
      </button>
      <button
        onClick={onGeolocation}
        className={`p-2 rounded-lg shadow-lg transition-all hover:scale-105 ${
          darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-700'
        }`}
        title="My location"
      >
        <span className="text-xl">📍</span>
      </button>
    </div>
  );
};

/**
 * Map Controller - Handle map interactions
 */
const MapController = ({ center, zoom, onMapReady }) => {
  const map = useMap();
  const mapRef = useRef(map);

  useEffect(() => {
    mapRef.current = map;
    if (onMapReady) {
      onMapReady(map);
    }
  }, [map, onMapReady]);

  return null;
};

/**
 * Main MapCore Component
 */
const MapCore = ({
  markers = [],
  routes = [],
  center = [41.0082, 28.9784], // Istanbul center
  zoom = 13,
  darkMode = false,
  showControls = true,
  height = '400px',
  onMapReady,
  className = '',
  autoFitBounds = true
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [mapInstance, setMapInstance] = useState(null);
  const [userLocation, setUserLocation] = useState(null);

  const tileUrl = darkMode
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';

  const attribution = darkMode
    ? '© CARTO © OpenStreetMap contributors'
    : '© OpenStreetMap contributors';

  // Map control handlers
  const handleZoomIn = () => {
    if (mapInstance) mapInstance.zoomIn();
  };

  const handleZoomOut = () => {
    if (mapInstance) mapInstance.zoomOut();
  };

  const handleRecenter = () => {
    if (mapInstance) {
      if (autoFitBounds && (markers.length > 0 || routes.length > 0)) {
        // Refit bounds
        const allPoints = [];
        markers.forEach(m => {
          const lat = m.lat || m.latitude;
          const lng = m.lng || m.lon || m.longitude;
          if (lat && lng) allPoints.push([lat, lng]);
        });
        routes.forEach(r => {
          if (r.coordinates) {
            r.coordinates.forEach(coord => {
              if (Array.isArray(coord)) allPoints.push(coord);
            });
          }
        });
        if (allPoints.length > 0) {
          const bounds = L.latLngBounds(allPoints);
          mapInstance.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
        }
      } else {
        mapInstance.setView(center, zoom);
      }
    }
  };

  const handleFullscreen = () => {
    const container = mapInstance?.getContainer();
    if (container) {
      if (!document.fullscreenElement) {
        container.requestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
    }
  };

  const handleGeolocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setUserLocation([latitude, longitude]);
          if (mapInstance) {
            mapInstance.setView([latitude, longitude], 15);
          }
        },
        (error) => {
          console.error('Geolocation error:', error);
          alert('Could not get your location. Please enable location services.');
        }
      );
    } else {
      alert('Geolocation is not supported by your browser.');
    }
  };

  return (
    <div className={`map-core-container relative ${className}`} style={{ height }}>
      {/* Loading Skeleton */}
      {isLoading && (
        <div className="absolute inset-0 z-[999] bg-gray-100 animate-pulse flex items-center justify-center">
          <div className="text-center">
            <div className="inline-block w-12 h-12 border-4 border-gray-300 border-t-indigo-600 rounded-full animate-spin"></div>
            <div className="mt-3 text-gray-500 font-medium">Loading map...</div>
          </div>
        </div>
      )}

      {/* Map Container */}
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={false}
        whenReady={() => {
          setTimeout(() => setIsLoading(false), 300);
        }}
      >
        <TileLayer
          attribution={attribution}
          url={tileUrl}
          maxZoom={19}
          keepBuffer={4}
          updateWhenIdle={true}
        />

        {/* Map Controller */}
        <MapController
          center={center}
          zoom={zoom}
          onMapReady={(map) => {
            setMapInstance(map);
            if (onMapReady) onMapReady(map);
          }}
        />

        {/* Auto-fit Bounds */}
        {autoFitBounds && <AutoFitBounds markers={markers} routes={routes} />}

        {/* Render Markers */}
        {markers.map((marker, index) => {
          const lat = marker.lat || marker.latitude;
          const lng = marker.lng || marker.lon || marker.longitude;
          
          if (!lat || !lng) return null;

          const icon = marker.pulse
            ? createPulsingMarker(marker.color, marker.emoji)
            : createMarkerIcon(marker.type, marker.color, marker.emoji);

          return (
            <Marker
              key={`marker-${index}`}
              position={[lat, lng]}
              icon={icon}
            >
              {marker.label && (
                <Popup>
                  <div className="text-sm">
                    <strong>{marker.label}</strong>
                    {marker.description && <p className="text-xs mt-1">{marker.description}</p>}
                  </div>
                </Popup>
              )}
            </Marker>
          );
        })}

        {/* Render User Location */}
        {userLocation && (
          <Marker
            position={userLocation}
            icon={createMarkerIcon('default', '#4F46E5', '📍')}
          >
            <Popup>
              <div className="text-sm">
                <strong>You are here</strong>
              </div>
            </Popup>
          </Marker>
        )}

        {/* Render Routes */}
        {routes.map((route, index) => {
          if (!route.coordinates || route.coordinates.length === 0) return null;

          return (
            <AnimatedPolyline
              key={`route-${index}`}
              positions={route.coordinates}
              color={route.color || '#4F46E5'}
              weight={route.weight || 5}
              opacity={route.opacity || 0.8}
              speed={route.speed || 50}
              animated={route.animated !== false}
            />
          );
        })}
      </MapContainer>

      {/* Map Controls */}
      {showControls && !isLoading && (
        <MapControls
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onRecenter={handleRecenter}
          onFullscreen={handleFullscreen}
          onGeolocation={handleGeolocation}
          darkMode={darkMode}
        />
      )}
    </div>
  );
};

export default MapCore;
export { createMarkerIcon, createPulsingMarker, AnimatedPolyline };
