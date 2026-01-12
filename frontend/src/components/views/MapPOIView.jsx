/**
 * MapPOIView Component
 * ====================
 * 
 * Map view for Points of Interest (attractions, restaurants, etc.)
 * Uses MapCore for visualization with custom markers.
 * 
 * Features:
 * - Custom marker icons for different POI types
 * - Interactive popups with details
 * - Auto-fit bounds to show all POIs
 * - Dark mode support
 * - Share locations
 * 
 * Author: AI Istanbul Team
 * Date: January 12, 2026
 */

import React from 'react';
import MapCore from '../map/MapCore';

const MapPOIView = ({ data, darkMode, className = '', ...props }) => {
  const mapData = data.map_data || data.mapData || data;
  const markers = mapData.markers || [];
  const routes = mapData.routes || [];
  
  // Transform markers to MapCore format
  const transformedMarkers = markers.map(marker => ({
    lat: marker.lat || marker.latitude,
    lng: marker.lng || marker.lon || marker.longitude,
    type: marker.type || 'poi',
    label: marker.name || marker.label || marker.title,
    description: marker.description || marker.address,
    color: marker.color,
    emoji: marker.emoji || getEmojiForType(marker.type),
    pulse: false
  }));

  // Transform routes to MapCore format
  const transformedRoutes = routes.map(route => ({
    coordinates: route.coordinates || route.path || [],
    color: route.color || '#4F46E5',
    weight: route.weight || 4,
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
    <div className={`map-poi-view ${className}`}>
      {/* POI Header */}
      {mapData.title && (
        <div className={`p-4 border-b ${
          darkMode 
            ? 'bg-gray-800 border-gray-700 text-white' 
            : 'bg-white border-gray-200 text-gray-900'
        }`}>
          <h3 className="text-lg font-semibold">{mapData.title}</h3>
          {mapData.description && (
            <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {mapData.description}
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

      {/* POI List */}
      {markers.length > 0 && (
        <div className={`p-4 ${
          darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'
        }`}>
          <h4 className="font-semibold mb-3">📍 Locations ({markers.length})</h4>
          <div className="space-y-2">
            {markers.slice(0, 5).map((marker, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg border ${
                  darkMode 
                    ? 'bg-gray-700 border-gray-600' 
                    : 'bg-gray-50 border-gray-200'
                }`}
              >
                <div className="flex items-start space-x-2">
                  <span className="text-xl">{marker.emoji || getEmojiForType(marker.type)}</span>
                  <div className="flex-1">
                    <div className="font-medium">{marker.name || marker.label}</div>
                    {marker.description && (
                      <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        {marker.description}
                      </div>
                    )}
                    {marker.address && (
                      <div className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                        📍 {marker.address}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {markers.length > 5 && (
              <div className={`text-sm text-center ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                +{markers.length - 5} more locations
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Helper function to get emoji for POI type
const getEmojiForType = (type) => {
  const emojiMap = {
    attraction: '🎭',
    restaurant: '🍽️',
    cafe: '☕',
    hotel: '🏨',
    museum: '🏛️',
    park: '🌳',
    shopping: '🛍️',
    transport: '🚇',
    poi: '📌',
    default: '📍'
  };
  return emojiMap[type] || emojiMap.default;
};

export default MapPOIView;
