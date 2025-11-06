/**
 * Intelligent Route Map Component
 * ================================
 * Week 3: Interactive map for intelligent route planning with OSRM integration
 * 
 * Features:
 * - Drag-and-drop waypoint reordering
 * - Real-time route recalculation with OSRM
 * - Multiple transport modes (walk, drive, bike)
 * - Custom markers for different location types
 * - Turn-by-turn directions display
 * - Route optimization and visualization
 * - Save/share routes
 * - Mobile-responsive design
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './IntelligentRouteMap.css';

// Fix Leaflet default markers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom marker icons with enhanced styling
const createMarkerIcon = (type = 'default', index = null, isDragging = false) => {
  const iconConfig = {
    start: { emoji: '🏁', color: '#22c55e', label: 'Start' },
    end: { emoji: '🎯', color: '#ef4444', label: 'End' },
    museum: { emoji: '🏛️', color: '#8b5cf6', label: 'Museum' },
    restaurant: { emoji: '🍽️', color: '#f59e0b', label: 'Restaurant' },
    cafe: { emoji: '☕', color: '#06b6d4', label: 'Cafe' },
    attraction: { emoji: '⭐', color: '#ec4899', label: 'Attraction' },
    default: { emoji: '📍', color: '#6b7280', label: 'Location' }
  };
  
  const config = iconConfig[type] || iconConfig.default;
  const displayText = index !== null ? index + 1 : config.emoji;
  
  const iconHtml = `
    <div class="intelligent-route-marker ${isDragging ? 'dragging' : ''}" style="
      background-color: ${config.color};
      width: 40px;
      height: 40px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 3px solid white;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      font-size: 18px;
      color: white;
      font-weight: bold;
      cursor: ${isDragging ? 'grabbing' : 'grab'};
      transition: transform 0.2s;
      ${isDragging ? 'transform: scale(1.3); z-index: 1000;' : ''}
    ">
      ${displayText}
    </div>
  `;
  
  return L.divIcon({
    html: iconHtml,
    className: 'custom-intelligent-marker',
    iconSize: [40, 40],
    iconAnchor: [20, 40],
    popupAnchor: [0, -40]
  });
};

// Auto-fit bounds component
const AutoFitBounds = ({ coordinates, markers }) => {
  const map = useMap();

  useEffect(() => {
    if (!map) return;
    const bounds = [];
    
    if (coordinates && coordinates.length > 0) {
      coordinates.forEach(coord => {
        if (Array.isArray(coord) && coord.length === 2) {
          bounds.push([coord[0], coord[1]]);
        }
      });
    }
    
    if (markers && markers.length > 0) {
      markers.forEach(marker => {
        if (marker.lat !== undefined && marker.lon !== undefined) {
          bounds.push([marker.lat, marker.lon]);
        }
      });
    }
    
    if (bounds.length > 0) {
      const leafletBounds = L.latLngBounds(bounds);
      map.fitBounds(leafletBounds, { padding: [50, 50], maxZoom: 15 });
    }
  }, [map, coordinates, markers]);

  return null;
};

const IntelligentRouteMap = ({
  waypoints = [],
  route = null,
  onWaypointReorder,
  onWaypointRemove,
  onWaypointAdd,
  onRecalculateRoute,
  transportMode = 'walk',
  height = '600px',
  enableDragDrop = true,
  showDirections = true,
  showStats = true,
  className = ''
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [draggedMarkerIndex, setDraggedMarkerIndex] = useState(null);
  const [isRecalculating, setIsRecalculating] = useState(false);
  const mapRef = useRef(null);

  // Istanbul center as default
  const defaultCenter = [41.0082, 28.9784];
  const mapCenter = waypoints.length > 0 
    ? [waypoints[0].lat, waypoints[0].lon]
    : defaultCenter;

  // Handle marker drag events
  const handleDragStart = useCallback((index) => {
    setIsDragging(true);
    setDraggedMarkerIndex(index);
  }, []);

  const handleDragEnd = useCallback(async (index, event) => {
    setIsDragging(false);
    setDraggedMarkerIndex(null);
    
    const newPosition = event.target.getLatLng();
    const updatedWaypoint = {
      ...waypoints[index],
      lat: newPosition.lat,
      lon: newPosition.lng
    };
    
    const newWaypoints = [...waypoints];
    newWaypoints[index] = updatedWaypoint;
    
    if (onWaypointReorder) {
      onWaypointReorder(newWaypoints);
    }
    
    // Recalculate route if callback provided
    if (onRecalculateRoute) {
      setIsRecalculating(true);
      try {
        await onRecalculateRoute(newWaypoints);
      } finally {
        setIsRecalculating(false);
      }
    }
  }, [waypoints, onWaypointReorder, onRecalculateRoute]);

  // Determine marker type based on position and category
  const getMarkerType = (waypoint, index) => {
    if (index === 0) return 'start';
    if (index === waypoints.length - 1) return 'end';
    return waypoint.category || 'default';
  };

  // Calculate route statistics
  const routeStats = route ? {
    distance: (route.total_distance / 1000).toFixed(2), // Convert to km
    duration: Math.round(route.total_duration / 60), // Convert to minutes
    segments: route.segments?.length || 0
  } : null;

  // Get route color based on transport mode
  const getRouteColor = (mode) => {
    const colors = {
      walk: '#3b82f6',    // Blue
      drive: '#f59e0b',   // Orange
      bike: '#10b981',    // Green
      transit: '#8b5cf6'  // Purple
    };
    return colors[mode] || colors.walk;
  };

  return (
    <div className={`intelligent-route-map-container ${className}`}>
      <MapContainer
        center={mapCenter}
        zoom={13}
        style={{ height, width: '100%' }}
        ref={mapRef}
        className="intelligent-route-map"
      >
        {/* OpenStreetMap Tiles */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Auto-fit bounds */}
        <AutoFitBounds
          coordinates={route?.polyline || []}
          markers={waypoints}
        />

        {/* Route polyline */}
        {route && route.polyline && route.polyline.length > 0 && (
          <Polyline
            positions={route.polyline}
            color={getRouteColor(transportMode)}
            weight={5}
            opacity={0.8}
            className="route-polyline"
          />
        )}

        {/* Waypoint markers */}
        {waypoints.map((waypoint, index) => {
          const markerType = getMarkerType(waypoint, index);
          const position = [waypoint.lat, waypoint.lon];
          
          return (
            <Marker
              key={`waypoint-${index}-${waypoint.name}`}
              position={position}
              icon={createMarkerIcon(markerType, index, draggedMarkerIndex === index)}
              draggable={enableDragDrop}
              eventHandlers={{
                dragstart: () => handleDragStart(index),
                dragend: (e) => handleDragEnd(index, e)
              }}
            >
              <Popup>
                <div className="route-popup">
                  <h3 className="popup-title">{waypoint.name}</h3>
                  {waypoint.category && (
                    <p className="popup-category">{waypoint.category}</p>
                  )}
                  {waypoint.address && (
                    <p className="popup-address">📍 {waypoint.address}</p>
                  )}
                  {waypoint.description && (
                    <p className="popup-description">{waypoint.description}</p>
                  )}
                  {waypoint.visit_duration && (
                    <p className="popup-duration">⏱️ {waypoint.visit_duration} min visit</p>
                  )}
                  <div className="popup-actions">
                    {onWaypointRemove && waypoints.length > 2 && (
                      <button 
                        className="btn-remove"
                        onClick={() => onWaypointRemove(index)}
                      >
                        ❌ Remove
                      </button>
                    )}
                    {index > 0 && index < waypoints.length - 1 && onWaypointReorder && (
                      <>
                        <button 
                          className="btn-move"
                          onClick={() => {
                            const newWaypoints = [...waypoints];
                            [newWaypoints[index], newWaypoints[index - 1]] = 
                              [newWaypoints[index - 1], newWaypoints[index]];
                            onWaypointReorder(newWaypoints);
                          }}
                        >
                          ⬆️ Up
                        </button>
                        <button 
                          className="btn-move"
                          onClick={() => {
                            const newWaypoints = [...waypoints];
                            [newWaypoints[index], newWaypoints[index + 1]] = 
                              [newWaypoints[index + 1], newWaypoints[index]];
                            onWaypointReorder(newWaypoints);
                          }}
                        >
                          ⬇️ Down
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>

      {/* Route statistics overlay */}
      {showStats && routeStats && (
        <div className="route-stats-overlay">
          <div className="stat-item">
            <span className="stat-icon">📏</span>
            <span className="stat-value">{routeStats.distance} km</span>
          </div>
          <div className="stat-item">
            <span className="stat-icon">⏱️</span>
            <span className="stat-value">{routeStats.duration} min</span>
          </div>
          <div className="stat-item">
            <span className="stat-icon">📍</span>
            <span className="stat-value">{waypoints.length} stops</span>
          </div>
          <div className="stat-item">
            <span className="stat-icon">
              {transportMode === 'walk' ? '🚶' : 
               transportMode === 'drive' ? '🚗' : 
               transportMode === 'bike' ? '🚴' : '🚇'}
            </span>
            <span className="stat-value">{transportMode}</span>
          </div>
        </div>
      )}

      {/* Recalculating indicator */}
      {isRecalculating && (
        <div className="recalculating-overlay">
          <div className="spinner"></div>
          <p>Recalculating route...</p>
        </div>
      )}
    </div>
  );
};

export default IntelligentRouteMap;
