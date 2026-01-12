/**
 * MapOnlyView Component
 * =====================
 * 
 * Standalone map visualization without route information.
 * Wrapper around MapCore for simple map displays.
 * 
 * Features:
 * - Clean map display
 * - Custom markers
 * - Route polylines
 * - Dark mode support
 * - Interactive controls
 * 
 * Author: AI Istanbul Team
 * Date: January 12, 2026
 */

import React from 'react';
import MapCore from '../map/MapCore';

const MapOnlyView = ({ data, darkMode, className = '', ...props }) => {
  const mapData = data.map_data || data.mapData || data;
  const markers = mapData.markers || [];
  const routes = mapData.routes || [];
  const title = mapData.title || null;
  const description = mapData.description || null;

  // Transform markers to MapCore format
  const transformedMarkers = markers.map(marker => ({
    lat: marker.lat || marker.latitude,
    lng: marker.lng || marker.lon || marker.longitude,
    type: marker.type || 'default',
    label: marker.name || marker.label,
    description: marker.description,
    color: marker.color,
    emoji: marker.emoji,
    pulse: marker.pulse || false
  }));

  // Transform routes to MapCore format
  const transformedRoutes = routes.map(route => ({
    coordinates: route.coordinates || route.path || [],
    color: route.color || '#4F46E5',
    weight: route.weight || 4,
    opacity: route.opacity || 0.8,
    animated: route.animated !== false
  }));

  // Calculate center
  const center = markers.length > 0
    ? [
        markers[0].lat || markers[0].latitude,
        markers[0].lng || markers[0].lon || markers[0].longitude
      ]
    : [41.0082, 28.9784]; // Istanbul center

  return (
    <div className={`map-only-view ${className}`}>
      {/* Header */}
      {(title || description) && (
        <div className={`p-4 border-b ${
          darkMode 
            ? 'bg-gray-800 border-gray-700 text-white' 
            : 'bg-white border-gray-200 text-gray-900'
        }`}>
          {title && <h3 className="text-lg font-semibold">{title}</h3>}
          {description && (
            <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {description}
            </p>
          )}
        </div>
      )}

      {/* Map */}
      <MapCore
        markers={transformedMarkers}
        routes={transformedRoutes}
        center={center}
        zoom={13}
        darkMode={darkMode}
        showControls={true}
        height="400px"
        autoFitBounds={true}
      />

      {/* Footer Info */}
      {markers.length > 0 && (
        <div className={`p-4 text-sm ${
          darkMode ? 'bg-gray-800 text-gray-400' : 'bg-white text-gray-600'
        }`}>
          <div className="flex items-center justify-between">
            <span>📍 {markers.length} location{markers.length !== 1 ? 's' : ''} shown</span>
            {routes.length > 0 && (
              <span>🛣️ {routes.length} route{routes.length !== 1 ? 's' : ''}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default MapOnlyView;
