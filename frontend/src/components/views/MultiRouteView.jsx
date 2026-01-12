/**
 * MultiRouteView Component
 * =========================
 * 
 * Multi-route comparison view using MultiRouteComparison component.
 * Displays alternative routes side-by-side with comparison metrics.
 * 
 * Features:
 * - Side-by-side route comparison
 * - Fastest/cheapest/least transfers badges
 * - Comfort scoring
 * - Interactive route selection
 * - Dark mode support
 * 
 * Author: AI Istanbul Team
 * Date: January 12, 2026
 */

import React, { useState } from 'react';
import MultiRouteComparison from '../MultiRouteComparison';
import MapCore from '../map/MapCore';

const MultiRouteView = ({ data, darkMode, onRouteSelect, className = '', ...props }) => {
  const [selectedRouteIndex, setSelectedRouteIndex] = useState(0);
  
  const alternatives = data.alternatives || data.alternative_routes || [];
  const primaryRoute = data.route_info ? {
    ...data.route_info,
    map_data: data.map_data
  } : null;

  // Generate route comparison summary
  const generateRouteComparison = (routes) => {
    if (!routes || routes.length === 0) return {};
    
    let fastestIdx = 0, fewestTransfersIdx = 0, leastWalkingIdx = 0, mostComfortableIdx = 0;
    
    routes.forEach((route, idx) => {
      const duration = route.duration_minutes || route.duration || 0;
      const transfers = route.num_transfers || route.transfers || 0;
      const walking = route.walking_meters || 0;
      const comfort = route.comfort_score?.overall_comfort || 0;
      
      if (duration < (routes[fastestIdx].duration_minutes || routes[fastestIdx].duration || 0)) {
        fastestIdx = idx;
      }
      if (transfers < (routes[fewestTransfersIdx].num_transfers || routes[fewestTransfersIdx].transfers || 0)) {
        fewestTransfersIdx = idx;
      }
      if (walking < (routes[leastWalkingIdx].walking_meters || 0)) {
        leastWalkingIdx = idx;
      }
      if (comfort > (routes[mostComfortableIdx].comfort_score?.overall_comfort || 0)) {
        mostComfortableIdx = idx;
      }
    });
    
    return {
      fastest_route: fastestIdx,
      fewest_transfers: fewestTransfersIdx,
      least_walking: leastWalkingIdx,
      most_comfortable: mostComfortableIdx
    };
  };

  const allRoutes = primaryRoute ? [primaryRoute, ...alternatives] : alternatives;
  const routeComparison = generateRouteComparison(allRoutes);
  const selectedRoute = allRoutes[selectedRouteIndex];

  // Extract map data from selected route
  const selectedMapData = selectedRoute?.map_data || {};
  const markers = selectedMapData.markers || [];
  const routes = selectedMapData.routes || [];

  return (
    <div className={`multi-route-view ${className}`}>
      {/* Multi-Route Comparison */}
      <div className={darkMode ? 'bg-gray-900' : 'bg-gray-50'}>
        <MultiRouteComparison
          routes={allRoutes}
          routeComparison={routeComparison}
          onRouteSelect={(route, index) => {
            setSelectedRouteIndex(index);
            if (onRouteSelect) onRouteSelect(route, index);
          }}
          darkMode={darkMode}
        />
      </div>

      {/* Selected Route Map Preview */}
      {selectedRoute && markers.length > 0 && (
        <div className={`mt-4 ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <div className={`p-4 border-b ${
            darkMode ? 'border-gray-700' : 'border-gray-200'
          }`}>
            <h4 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              📍 Route Preview
            </h4>
            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Option {selectedRouteIndex + 1} - Click on map controls to interact
            </p>
          </div>
          <MapCore
            markers={markers.map(m => ({
              lat: m.lat || m.latitude,
              lng: m.lng || m.lon || m.longitude,
              type: m.type,
              label: m.name || m.label,
              emoji: m.emoji,
              color: m.color
            }))}
            routes={routes.map(r => ({
              coordinates: r.coordinates || r.path || [],
              color: r.color || '#4F46E5',
              animated: true
            }))}
            darkMode={darkMode}
            showControls={true}
            height="300px"
            autoFitBounds={true}
          />
        </div>
      )}
    </div>
  );
};

export default MultiRouteView;
