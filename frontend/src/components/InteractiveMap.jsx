/**
 * InteractiveMap Component - Displays map with POIs, route, and user location
 */

import React, { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, useMapEvents } from 'react-leaflet';
import { useLocation } from '../contexts/LocationContext';
import L from 'leaflet';

// Fix default markers for React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom icons for different POI types
const createCustomIcon = (category, color = '#3B82F6') => {
  const icons = {
    restaurant: '🍽️',
    museum: '🏛️',
    landmark: '🏗️',
    shopping: '🛍️',
    entertainment: '🎭',
    religious: '🕌',
    park: '🌳',
    viewpoint: '🔭',
    transport: '🚌',
    hotel: '🏨'
  };

  const icon = icons[category] || '📍';
  
  return L.divIcon({
    html: `<div style="background-color: ${color}; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3); font-size: 14px;">${icon}</div>`,
    className: 'custom-div-icon',
    iconSize: [30, 30],
    iconAnchor: [15, 15],
    popupAnchor: [0, -15]
  });
};

// User location icon
const userLocationIcon = L.divIcon({
  html: '<div style="background-color: #EF4444; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3); animation: pulse 2s infinite;"><div style="width: 8px; height: 8px; background-color: white; border-radius: 50%;"></div></div>',
  className: 'user-location-icon',
  iconSize: [20, 20],
  iconAnchor: [10, 10]
});

// Map event handler component
const MapEventHandler = ({ onMapClick, onLocationFound }) => {
  const map = useMapEvents({
    click: (e) => {
      if (onMapClick) {
        onMapClick(e.latlng);
      }
    },
    locationfound: (e) => {
      if (onLocationFound) {
        onLocationFound(e.latlng);
      }
    }
  });

  return null;
};

// Auto-fit bounds component
const AutoFitBounds = ({ bounds, padding = [20, 20] }) => {
  const map = useMap();

  useEffect(() => {
    if (bounds && bounds.length > 0) {
      const leafletBounds = L.latLngBounds(bounds);
      map.fitBounds(leafletBounds, { padding });
    }
  }, [bounds, map, padding]);

  return null;
};

const InteractiveMap = ({ 
  height = '400px', 
  showUserLocation = true, 
  showPOIs = true, 
  showRoute = true,
  allowLocationSelection = false,
  onLocationSelect,
  zoom = 13 
}) => {
  const {
    currentLocation,
    recommendations,
    currentRoute,
    nearbyPOIs,
    preferences
  } = useLocation();

  const [mapCenter, setMapCenter] = useState([41.0082, 28.9784]); // Default to Sultanahmet
  const [selectedPOI, setSelectedPOI] = useState(null);
  const [mapBounds, setMapBounds] = useState([]);

  // Update map center when user location changes
  useEffect(() => {
    if (currentLocation) {
      setMapCenter([currentLocation.lat, currentLocation.lng]);
    }
  }, [currentLocation]);

  // Calculate bounds for auto-fitting
  useEffect(() => {
    const bounds = [];
    
    if (currentLocation && showUserLocation) {
      bounds.push([currentLocation.lat, currentLocation.lng]);
    }
    
    if (showPOIs) {
      recommendations.forEach(rec => {
        if (rec.poi && rec.poi.coordinates) {
          bounds.push([rec.poi.coordinates.latitude, rec.poi.coordinates.longitude]);
        }
      });
      
      nearbyPOIs.forEach(poi => {
        if (poi.coordinates) {
          bounds.push([poi.coordinates.latitude, poi.coordinates.longitude]);
        }
      });
    }
    
    if (showRoute && currentRoute && currentRoute.route && currentRoute.route.segments) {
      // Add route waypoints to bounds
      currentRoute.poi_details?.forEach(poi => {
        if (poi.coordinates) {
          bounds.push([poi.coordinates.latitude, poi.coordinates.longitude]);
        }
      });
    }
    
    setMapBounds(bounds);
  }, [currentLocation, recommendations, nearbyPOIs, currentRoute, showUserLocation, showPOIs, showRoute]);

  const handleMapClick = (latlng) => {
    if (allowLocationSelection && onLocationSelect) {
      onLocationSelect({
        lat: latlng.lat,
        lng: latlng.lng
      });
    }
  };

  const getRouteCoordinates = () => {
    if (!currentRoute || !currentRoute.route || !currentRoute.route.segments) {
      return [];
    }

    const coordinates = [];
    
    // Add starting location if available
    if (currentLocation) {
      coordinates.push([currentLocation.lat, currentLocation.lng]);
    }

    // Add POI coordinates based on route order
    if (currentRoute.poi_details) {
      currentRoute.poi_details.forEach(poi => {
        if (poi.coordinates) {
          coordinates.push([poi.coordinates.latitude, poi.coordinates.longitude]);
        }
      });
    }

    return coordinates;
  };

  const formatPOIPopup = (poi, distance = null) => {
    return (
      <div className="poi-popup max-w-xs">
        <h4 className="font-semibold text-gray-800 mb-2">{poi.name}</h4>
        
        <div className="space-y-1 text-sm">
          <div className="flex items-center space-x-2">
            <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
              {poi.category}
            </span>
            {poi.rating && (
              <span className="text-yellow-500">
                {'★'.repeat(Math.floor(poi.rating))} {poi.rating.toFixed(1)}
              </span>
            )}
          </div>
          
          {distance && (
            <p className="text-gray-600">
              Distance: {distance < 1 ? `${Math.round(distance * 1000)}m` : `${distance.toFixed(1)}km`}
            </p>
          )}
          
          {poi.estimated_visit_duration && (
            <p className="text-gray-600">
              Visit time: {poi.estimated_visit_duration} minutes
            </p>
          )}
          
          {poi.description && (
            <p className="text-gray-700 mt-2">{poi.description}</p>
          )}
          
          <div className="mt-2 pt-2 border-t border-gray-200">
            <button
              onClick={() => setSelectedPOI(poi)}
              className="w-full px-3 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
            >
              View Details
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="interactive-map">
      <div style={{ height }} className="rounded-lg overflow-hidden shadow-md">
        <MapContainer
          center={mapCenter}
          zoom={zoom}
          style={{ height: '100%', width: '100%' }}
          className="z-0"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {/* Map event handlers */}
          <MapEventHandler 
            onMapClick={handleMapClick}
          />
          
          {/* Auto-fit bounds */}
          <AutoFitBounds bounds={mapBounds} />
          
          {/* User location marker */}
          {showUserLocation && currentLocation && (
            <Marker
              position={[currentLocation.lat, currentLocation.lng]}
              icon={userLocationIcon}
            >
              <Popup>
                <div className="text-center">
                  <h4 className="font-semibold text-gray-800">Your Location</h4>
                  <p className="text-sm text-gray-600">
                    {currentLocation.lat.toFixed(4)}, {currentLocation.lng.toFixed(4)}
                  </p>
                  {currentLocation.accuracy && (
                    <p className="text-xs text-gray-500">
                      Accuracy: ±{Math.round(currentLocation.accuracy)}m
                    </p>
                  )}
                </div>
              </Popup>
            </Marker>
          )}
          
          {/* POI markers from recommendations */}
          {showPOIs && recommendations.map((rec, index) => {
            const poi = rec.poi;
            if (!poi || !poi.coordinates) return null;
            
            return (
              <Marker
                key={`rec-${poi.id}-${index}`}
                position={[poi.coordinates.latitude, poi.coordinates.longitude]}
                icon={createCustomIcon(poi.category, '#10B981')}
              >
                <Popup>
                  {formatPOIPopup(poi, rec.distance_km)}
                </Popup>
              </Marker>
            );
          })}
          
          {/* POI markers from nearby search */}
          {showPOIs && nearbyPOIs.map((poi, index) => {
            if (!poi.coordinates) return null;
            
            return (
              <Marker
                key={`nearby-${poi.id}-${index}`}
                position={[poi.coordinates.latitude, poi.coordinates.longitude]}
                icon={createCustomIcon(poi.category, '#8B5CF6')}
              >
                <Popup>
                  {formatPOIPopup(poi)}
                </Popup>
              </Marker>
            );
          })}
          
          {/* Route visualization */}
          {showRoute && (() => {
            const routeCoords = getRouteCoordinates();
            if (routeCoords.length < 2) return null;
            
            return (
              <Polyline
                positions={routeCoords}
                pathOptions={{
                  color: '#3B82F6',
                  weight: 4,
                  opacity: 0.8,
                  dashArray: '5, 10'
                }}
              >
                <Popup>
                  <div className="text-center">
                    <h4 className="font-semibold text-gray-800">Planned Route</h4>
                    {currentRoute && currentRoute.route && (
                      <div className="text-sm text-gray-600 mt-1">
                        <p>Distance: {currentRoute.route.total_distance_km?.toFixed(1)}km</p>
                        <p>Time: {Math.round(currentRoute.route.total_time_minutes)} minutes</p>
                        <p>Stops: {currentRoute.poi_details?.length || 0}</p>
                      </div>
                    )}
                  </div>
                </Popup>
              </Polyline>
            );
          })()}
          
          {/* Route waypoint markers */}
          {showRoute && currentRoute && currentRoute.poi_details && 
            currentRoute.poi_details.map((poi, index) => {
              if (!poi.coordinates) return null;
              
              return (
                <Marker
                  key={`route-${poi.id}-${index}`}
                  position={[poi.coordinates.latitude, poi.coordinates.longitude]}
                  icon={createCustomIcon(poi.category, '#EF4444')}
                >
                  <Popup>
                    <div className="route-poi-popup">
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="w-6 h-6 bg-red-600 text-white rounded-full flex items-center justify-center text-sm font-medium">
                          {index + 1}
                        </span>
                        <h4 className="font-semibold text-gray-800">{poi.name}</h4>
                      </div>
                      {formatPOIPopup(poi)}
                    </div>
                  </Popup>
                </Marker>
              );
            })
          }
        </MapContainer>
      </div>
      
      {/* Map legend */}
      <div className="mt-2 flex flex-wrap items-center gap-4 text-sm text-gray-600">
        {showUserLocation && (
          <div className="flex items-center space-x-1">
            <div className="w-4 h-4 bg-red-500 rounded-full border border-white"></div>
            <span>Your Location</span>
          </div>
        )}
        {showPOIs && recommendations.length > 0 && (
          <div className="flex items-center space-x-1">
            <div className="w-4 h-4 bg-green-500 rounded-full border border-white"></div>
            <span>Recommendations</span>
          </div>
        )}
        {showPOIs && nearbyPOIs.length > 0 && (
          <div className="flex items-center space-x-1">
            <div className="w-4 h-4 bg-purple-500 rounded-full border border-white"></div>
            <span>Nearby POIs</span>
          </div>
        )}
        {showRoute && currentRoute && (
          <div className="flex items-center space-x-1">
            <div className="w-4 h-1 bg-blue-500 border-dashed border border-blue-300"></div>
            <span>Planned Route</span>
          </div>
        )}
        {allowLocationSelection && (
          <div className="text-xs text-gray-500">
            Click map to select location
          </div>
        )}
      </div>
      
      {/* POI Detail Modal */}
      {selectedPOI && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-md w-full max-h-90vh overflow-y-auto">
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-800">{selectedPOI.name}</h3>
                <button
                  onClick={() => setSelectedPOI(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="p-4 space-y-4">
              <div className="flex items-center space-x-2">
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  {selectedPOI.category}
                </span>
                {selectedPOI.rating && (
                  <span className="text-yellow-500">
                    {'★'.repeat(Math.floor(selectedPOI.rating))} {selectedPOI.rating.toFixed(1)}
                  </span>
                )}
              </div>
              
              {selectedPOI.description && (
                <p className="text-gray-700">{selectedPOI.description}</p>
              )}
              
              {selectedPOI.address && (
                <div>
                  <h4 className="font-medium text-gray-800 mb-1">Address</h4>
                  <p className="text-sm text-gray-600">{selectedPOI.address}</p>
                </div>
              )}
              
              {selectedPOI.estimated_visit_duration && (
                <div>
                  <h4 className="font-medium text-gray-800 mb-1">Visit Duration</h4>
                  <p className="text-sm text-gray-600">{selectedPOI.estimated_visit_duration} minutes</p>
                </div>
              )}
              
              {selectedPOI.features && selectedPOI.features.length > 0 && (
                <div>
                  <h4 className="font-medium text-gray-800 mb-1">Features</h4>
                  <div className="flex flex-wrap gap-1">
                    {selectedPOI.features.map((feature, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default InteractiveMap;
