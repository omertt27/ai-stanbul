/**
 * RouteDisplay Component - Shows planned routes and allows route management
 */

import React, { useState, useEffect } from 'react';
import { useLocation } from '../contexts/LocationContext';

const RouteDisplay = ({ showOptimization = true, showExport = true }) => {
  const {
    currentRoute,
    routeLoading,
    routeError,
    currentLocation,
    preferences,
    planRoute,
    updatePreferences
  } = useLocation();

  const [routeOptions, setRouteOptions] = useState({
    algorithm: 'tsp_nearest',
    transport: 'walking',
    optimizeFor: 'time'
  });

  const [expandedSegment, setExpandedSegment] = useState(null);

  // Sync route options with preferences
  useEffect(() => {
    setRouteOptions(prevOptions => ({
      ...prevOptions,
      transport: preferences.transportMode || 'walking'
    }));
  }, [preferences.transportMode]);

  const algorithms = [
    { value: 'tsp_nearest', label: 'Nearest Neighbor (Fast)', description: 'Quick route optimization' },
    { value: 'tsp_greedy', label: 'Greedy Algorithm', description: 'Balance speed and optimization' },
    { value: 'dijkstra', label: 'Dijkstra (Accurate)', description: 'Most accurate shortest path' },
    { value: 'a_star', label: 'A* Search', description: 'Heuristic-based optimization' }
  ];

  const transportModes = [
    { value: 'walking', label: '🚶 Walking', speed: '5 km/h' },
    { value: 'public', label: '🚌 Public Transport', speed: '15 km/h' },
    { value: 'driving', label: '🚗 Driving', speed: '30 km/h' },
    { value: 'mixed', label: '🔄 Mixed', speed: 'Variable' }
  ];

  const handleOptionChange = (key, value) => {
    const newOptions = { ...routeOptions, [key]: value };
    setRouteOptions(newOptions);
    
    // Update global preferences
    if (key === 'transport') {
      updatePreferences({ transportMode: value });
    }
  };

  const handleReoptimizeRoute = async () => {
    if (!currentRoute || !currentRoute.poi_details) return;
    
    try {
      const poiIds = currentRoute.poi_details.map(poi => poi.id);
      await planRoute(poiIds, routeOptions);
    } catch (error) {
      console.error('Failed to reoptimize route:', error);
    }
  };

  const formatTime = (minutes) => {
    if (minutes < 60) {
      return `${Math.round(minutes)}min`;
    }
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}min`;
  };

  const formatDistance = (km) => {
    if (km < 1) {
      return `${Math.round(km * 1000)}m`;
    }
    return `${km.toFixed(1)}km`;
  };

  const getTransportIcon = (mode) => {
    const modeInfo = transportModes.find(t => t.value === mode);
    return modeInfo ? modeInfo.label.split(' ')[0] : '🚶';
  };

  const exportRoute = () => {
    if (!currentRoute) return;
    
    const routeData = {
      route: currentRoute,
      exportedAt: new Date().toISOString(),
      location: currentLocation
    };
    
    const blob = new Blob([JSON.stringify(routeData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `istanbul-route-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const shareRoute = async () => {
    if (!currentRoute || !navigator.share) {
      // Fallback: copy to clipboard
      const routeText = `Istanbul Route Plan:\n${currentRoute.poi_details?.map(poi => `• ${poi.name}`).join('\n')}\nTotal: ${formatDistance(currentRoute.route?.total_distance_km)} / ${formatTime(currentRoute.route?.total_time_minutes)}`;
      
      try {
        await navigator.clipboard.writeText(routeText);
        alert('Route copied to clipboard!');
      } catch (error) {
        console.error('Failed to copy route:', error);
      }
      return;
    }
    
    try {
      await navigator.share({
        title: 'My Istanbul Route',
        text: `Check out my planned route in Istanbul! Total distance: ${formatDistance(currentRoute.route?.total_distance_km)}, estimated time: ${formatTime(currentRoute.route?.total_time_minutes)}`,
        url: window.location.href
      });
    } catch (error) {
      console.error('Failed to share route:', error);
    }
  };

  return (
    <div className="route-display bg-white rounded-lg shadow-md p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Route Plan</h3>
        
        {currentRoute && (
          <div className="flex space-x-2">
            {showExport && (
              <>
                <button
                  onClick={exportRoute}
                  className="px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-700"
                >
                  Export
                </button>
                <button
                  onClick={shareRoute}
                  className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Share
                </button>
              </>
            )}
          </div>
        )}
      </div>

      {/* Route Optimization Options */}
      {showOptimization && (
        <div className="mb-6 p-4 bg-gray-50 rounded-md space-y-4">
          <h4 className="font-medium text-gray-700">Route Options</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Algorithm Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-2">Algorithm</label>
              <select
                value={routeOptions.algorithm}
                onChange={(e) => handleOptionChange('algorithm', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {algorithms.map(algo => (
                  <option key={algo.value} value={algo.value}>
                    {algo.label}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {algorithms.find(a => a.value === routeOptions.algorithm)?.description}
              </p>
            </div>

            {/* Transport Mode */}
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-2">Transport Mode</label>
              <select
                value={routeOptions.transport}
                onChange={(e) => handleOptionChange('transport', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {transportModes.map(mode => (
                  <option key={mode.value} value={mode.value}>
                    {mode.label} ({mode.speed})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {currentRoute && (
            <button
              onClick={handleReoptimizeRoute}
              disabled={routeLoading}
              className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              {routeLoading ? 'Optimizing...' : 'Reoptimize Route'}
            </button>
          )}
        </div>
      )}

      {/* Loading State */}
      {routeLoading && (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Planning your route...</p>
        </div>
      )}

      {/* Error State */}
      {routeError && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
          <p className="text-red-800">
            <strong>Error:</strong> {routeError}
          </p>
        </div>
      )}

      {/* Route Summary */}
      {currentRoute && currentRoute.route && (
        <div className="mb-6 p-4 bg-blue-50 rounded-md">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {formatDistance(currentRoute.route.total_distance_km)}
              </div>
              <div className="text-sm text-gray-600">Total Distance</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">
                {formatTime(currentRoute.route.total_time_minutes)}
              </div>
              <div className="text-sm text-gray-600">Estimated Time</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-600">
                {currentRoute.poi_details?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Stops</div>
            </div>
            <div>
              <div className="text-2xl">
                {getTransportIcon(currentRoute.route.route_type)}
              </div>
              <div className="text-sm text-gray-600">{currentRoute.route.route_type}</div>
            </div>
          </div>
          
          {currentRoute.route.confidence_score && (
            <div className="mt-3 text-center">
              <span className="text-sm text-gray-600">
                Confidence: {Math.round(currentRoute.route.confidence_score * 100)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* Route Steps */}
      {currentRoute && currentRoute.route && currentRoute.route.segments && (
        <div className="space-y-3">
          <h4 className="font-medium text-gray-700">Route Steps</h4>
          
          {currentRoute.route.segments.map((segment, index) => (
            <div
              key={index}
              className="border border-gray-200 rounded-md overflow-hidden"
            >
              <div
                className="p-3 bg-gray-50 cursor-pointer hover:bg-gray-100 flex items-center justify-between"
                onClick={() => setExpandedSegment(expandedSegment === index ? null : index)}
              >
                <div className="flex items-center space-x-3">
                  <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-medium">
                    {index + 1}
                  </div>
                  <div>
                    <span className="font-medium text-gray-800">
                      {segment.from} → {segment.to}
                    </span>
                    <div className="text-sm text-gray-600">
                      {formatDistance(segment.distance_km)} • {formatTime(segment.time_minutes)}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{getTransportIcon(segment.transport_mode)}</span>
                  <span className="text-gray-400">
                    {expandedSegment === index ? '−' : '+'}
                  </span>
                </div>
              </div>
              
              {expandedSegment === index && segment.instructions && (
                <div className="p-3 border-t border-gray-200 bg-white">
                  <h5 className="font-medium text-gray-700 mb-2">Instructions:</h5>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                    {segment.instructions.map((instruction, instrIndex) => (
                      <li key={instrIndex}>{instruction}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* POI Details */}
      {currentRoute && currentRoute.poi_details && (
        <div className="mt-6 space-y-3">
          <h4 className="font-medium text-gray-700">Destination Details</h4>
          
          {currentRoute.poi_details.map((poi, index) => (
            <div key={poi.id} className="p-3 border border-gray-200 rounded-md">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <span className="w-6 h-6 bg-green-600 text-white rounded-full flex items-center justify-center text-sm font-medium">
                      {index + 1}
                    </span>
                    <h5 className="font-medium text-gray-800">{poi.name}</h5>
                    <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                      {poi.category}
                    </span>
                  </div>
                  
                  {poi.estimated_visit_duration && (
                    <p className="text-sm text-gray-600 mt-1 ml-8">
                      Suggested visit time: {poi.estimated_visit_duration} minutes
                    </p>
                  )}
                </div>
                
                <div className="text-right">
                  <span className={`px-2 py-1 rounded text-xs ${
                    poi.is_open ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {poi.is_open ? 'Open' : 'Closed'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!routeLoading && !routeError && !currentRoute && (
        <div className="text-center py-8 text-gray-500">
          <div className="text-4xl mb-2">🗺️</div>
          <p>No route planned yet</p>
          <p className="text-sm mt-1">Select POIs from recommendations to plan a route</p>
        </div>
      )}
    </div>
  );
};

export default RouteDisplay;
