/**
 * TransportationInterface Component
 * ==================================
 * Main interface for Istanbul AI Transportation System
 * 
 * Features:
 * - Chat-based route planning
 * - Interactive map with live routes
 * - Step-by-step instructions
 * - GPS location integration
 * - Multi-modal transport search
 * - Real-time updates
 * - Mobile-responsive design
 * 
 * Integration:
 * - Connects chatbot with transportation API
 * - Renders routes on map
 * - Displays instructions panel
 * - Supports voice input (future)
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import TransportationMap from './TransportationMap';
import RouteInstructions from './RouteInstructions';
import { useLocation } from '../contexts/LocationContext';
import transportationApi from '../api/transportationApi';

const TransportationInterface = ({
  initialOrigin = null,
  initialDestination = null,
  initialMode = 'transit',
  darkMode = false,
  className = ''
}) => {
  // State
  const [origin, setOrigin] = useState(initialOrigin);
  const [destination, setDestination] = useState(initialDestination);
  const [mode, setMode] = useState(initialMode);
  const [route, setRoute] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showInstructions, setShowInstructions] = useState(true);
  const [useGPS, setUseGPS] = useState(false);
  const [accessible, setAccessible] = useState(false);
  
  // Location context
  const { currentLocation, hasLocation, requestLocation } = useLocation();
  
  // Refs
  const originInputRef = useRef(null);
  const destinationInputRef = useRef(null);

  // Handle GPS toggle
  const handleGPSToggle = useCallback(async () => {
    if (!useGPS && !hasLocation) {
      try {
        await requestLocation();
        setUseGPS(true);
        if (currentLocation) {
          setOrigin({
            lat: currentLocation.latitude,
            lng: currentLocation.longitude,
            name: 'My Location'
          });
        }
      } catch (err) {
        alert('Unable to access your location. Please enable location permissions.');
      }
    } else {
      setUseGPS(!useGPS);
      if (!useGPS && currentLocation) {
        setOrigin({
          lat: currentLocation.latitude,
          lng: currentLocation.longitude,
          name: 'My Location'
        });
      }
    }
  }, [useGPS, hasLocation, currentLocation, requestLocation]);

  // Update origin when GPS is enabled and location changes
  useEffect(() => {
    if (useGPS && currentLocation) {
      setOrigin({
        lat: currentLocation.latitude,
        lng: currentLocation.longitude,
        name: 'My Location'
      });
    }
  }, [useGPS, currentLocation]);

  // Get route directions
  const handleGetDirections = useCallback(async () => {
    if (!origin || !destination) {
      setError('Please enter both origin and destination');
      return;
    }

    setLoading(true);
    setError(null);
    setRoute(null);

    try {
      let routeData;
      
      if (useGPS && currentLocation) {
        // Use GPS-based route
        routeData = await transportationApi.getGPSRoute({
          userLocation: currentLocation,
          destination: {
            lat: destination.lat,
            lng: destination.lng,
            name: destination.name
          },
          mode,
          accessible
        });
      } else {
        // Use standard route
        routeData = await transportationApi.getTransportationDirections({
          from: origin,
          to: destination,
          mode,
          accessible,
          alternatives: true
        });
      }

      setRoute(routeData);
      setShowInstructions(true);
      
    } catch (err) {
      console.error('Failed to get directions:', err);
      setError(err.message || 'Failed to get directions. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [origin, destination, mode, accessible, useGPS, currentLocation]);

  // Parse location input (can be name or coordinates)
  const parseLocationInput = (input) => {
    if (!input) return null;
    
    // Check if it's coordinates (lat,lng)
    const coordMatch = input.match(/(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)/);
    if (coordMatch) {
      return {
        lat: parseFloat(coordMatch[1]),
        lng: parseFloat(coordMatch[2]),
        name: `${coordMatch[1]}, ${coordMatch[2]}`
      };
    }
    
    // Otherwise, treat as location name
    // In production, this would geocode the name to coordinates
    // For now, we'll just store the name and let the backend handle it
    return {
      name: input,
      lat: null,
      lng: null
    };
  };

  // Handle origin input
  const handleOriginChange = (e) => {
    const location = parseLocationInput(e.target.value);
    setOrigin(location);
  };

  // Handle destination input
  const handleDestinationChange = (e) => {
    const location = parseLocationInput(e.target.value);
    setDestination(location);
  };

  // Handle alternative route selection
  const handleSelectAlternative = useCallback(async (alternative) => {
    console.log('Selected alternative:', alternative);
    // In production, this would fetch the full alternative route
    // For now, just show a message
    alert(`Alternative route selected: ${alternative.summary}`);
  }, []);

  // Swap origin and destination
  const handleSwap = () => {
    const temp = origin;
    setOrigin(destination);
    setDestination(temp);
  };

  return (
    <div className={`transportation-interface ${className}`}>
      {/* Search Panel */}
      <div className="search-panel bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 mb-4">
        <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4 flex items-center gap-2">
          <span>🚇</span>
          <span>Istanbul Transportation</span>
        </h2>

        {/* Origin Input */}
        <div className="input-group mb-3">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            From
          </label>
          <div className="flex gap-2">
            <input
              ref={originInputRef}
              type="text"
              value={origin?.name || ''}
              onChange={handleOriginChange}
              placeholder="Enter origin or coordinates (lat, lng)"
              disabled={useGPS}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            />
            <button
              onClick={handleGPSToggle}
              className={`px-4 py-2 rounded-lg transition ${
                useGPS 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              title="Use my location"
            >
              📍
            </button>
          </div>
          {useGPS && (
            <p className="text-xs text-blue-600 mt-1">Using your GPS location</p>
          )}
        </div>

        {/* Swap Button */}
        <div className="flex justify-center my-2">
          <button
            onClick={handleSwap}
            className="p-2 hover:bg-gray-100 rounded-full transition"
            title="Swap origin and destination"
          >
            🔄
          </button>
        </div>

        {/* Destination Input */}
        <div className="input-group mb-3">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            To
          </label>
          <input
            ref={destinationInputRef}
            type="text"
            value={destination?.name || ''}
            onChange={handleDestinationChange}
            placeholder="Enter destination or coordinates (lat, lng)"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Transport Mode */}
        <div className="mode-selector mb-3">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Transport Mode
          </label>
          <div className="flex gap-2 flex-wrap">
            {['transit', 'walking', 'driving'].map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-4 py-2 rounded-lg border-2 transition ${
                  mode === m
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-gray-700 border-gray-300 hover:border-blue-400'
                }`}
              >
                {m === 'transit' && '🚇 Transit'}
                {m === 'walking' && '🚶 Walking'}
                {m === 'driving' && '🚗 Driving'}
              </button>
            ))}
          </div>
        </div>

        {/* Accessibility Toggle */}
        <div className="accessibility-toggle mb-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={accessible}
              onChange={(e) => setAccessible(e.target.checked)}
              className="w-4 h-4 rounded"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">
              ♿ Prefer accessible routes
            </span>
          </label>
        </div>

        {/* Get Directions Button */}
        <button
          onClick={handleGetDirections}
          disabled={loading || !origin || !destination}
          className="w-full px-6 py-3 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              Finding Route...
            </span>
          ) : (
            '🗺️ Get Directions'
          )}
        </button>

        {/* Error Message */}
        {error && (
          <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-800 text-sm">
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Map and Instructions Layout */}
      <div className="content-layout grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Map - Takes 2 columns on large screens */}
        <div className="map-container lg:col-span-2">
          <TransportationMap
            route={route}
            origin={origin}
            destination={destination}
            showUserLocation={true}
            showNearbyStations={true}
            height="600px"
            darkMode={darkMode}
          />
        </div>

        {/* Instructions - Takes 1 column on large screens */}
        {route && showInstructions && (
          <div className="instructions-container lg:col-span-1">
            <div className="sticky top-4">
              <RouteInstructions
                route={route}
                origin={origin}
                destination={destination}
                showAlternatives={true}
                onSelectAlternative={handleSelectAlternative}
              />
            </div>
          </div>
        )}
      </div>

      {/* Quick Tips */}
      {!route && (
        <div className="tips-panel bg-blue-50 border border-blue-200 rounded-lg p-4 mt-4">
          <h3 className="font-bold text-blue-800 mb-2">💡 Quick Tips</h3>
          <ul className="text-sm text-blue-700 space-y-1">
            <li>• Use the 📍 button to get directions from your current location</li>
            <li>• Enter location names (e.g., "Taksim", "Sultanahmet") or coordinates</li>
            <li>• Enable accessibility mode for wheelchair-accessible routes</li>
            <li>• Try different transport modes for alternative routes</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default TransportationInterface;
