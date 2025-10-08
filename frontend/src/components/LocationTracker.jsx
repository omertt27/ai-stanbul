/**
 * LocationTracker Component - Handles user location tracking and session management
 */

import React, { useEffect, useState } from 'react';
import { useLocation } from '../contexts/LocationContext';

const LocationTracker = ({ onLocationUpdate, autoStart = true }) => {
  const {
    currentLocation,
    locationError,
    locationLoading,
    sessionActive,
    isTracking,
    getCurrentLocation,
    createSession,
    startLocationTracking,
    stopLocationTracking,
    updatePreferences
  } = useLocation();

  const [permissionStatus, setPermissionStatus] = useState('prompt');
  const [manualLocation, setManualLocation] = useState('');

  // Check location permission status
  useEffect(() => {
    const checkPermission = async () => {
      if ('permissions' in navigator) {
        try {
          const permission = await navigator.permissions.query({ name: 'geolocation' });
          setPermissionStatus(permission.state);
          
          permission.addEventListener('change', () => {
            setPermissionStatus(permission.state);
          });
        } catch (error) {
          console.error('Failed to check geolocation permission:', error);
        }
      }
    };

    checkPermission();
  }, []);

  // Auto-start location tracking if enabled
  useEffect(() => {
    if (autoStart && permissionStatus === 'granted' && !isTracking) {
      handleStartTracking();
    }
  }, [autoStart, permissionStatus, isTracking]);

  // Notify parent of location updates
  useEffect(() => {
    if (currentLocation && onLocationUpdate) {
      onLocationUpdate(currentLocation);
    }
  }, [currentLocation, onLocationUpdate]);

  const handleStartTracking = async () => {
    try {
      // Get initial location
      const location = await getCurrentLocation();
      
      // Create session if not active
      if (!sessionActive) {
        await createSession(location, {
          autoTracking: true,
          timestamp: new Date().toISOString()
        });
      }
      
      // Start continuous tracking
      startLocationTracking();
    } catch (error) {
      console.error('Failed to start location tracking:', error);
    }
  };

  const handleStopTracking = () => {
    stopLocationTracking();
  };

  const handleManualLocation = async () => {
    if (!manualLocation.trim()) return;
    
    try {
      // Parse manual location (expected format: "lat,lng" or "district name")
      const coords = manualLocation.split(',').map(s => s.trim());
      
      if (coords.length === 2) {
        const lat = parseFloat(coords[0]);
        const lng = parseFloat(coords[1]);
        
        if (!isNaN(lat) && !isNaN(lng)) {
          const location = { lat, lng, accuracy: null };
          
          if (!sessionActive) {
            await createSession(location, { manualEntry: true });
          }
          
          // Trigger location update in context (this would normally call updateLocation)
          if (onLocationUpdate) {
            onLocationUpdate(location);
          }
        }
      } else {
        // Handle district name or address (would need geocoding service)
        console.log('District/Address search not implemented yet:', manualLocation);
      }
    } catch (error) {
      console.error('Failed to set manual location:', error);
    }
  };

  const getLocationStatusText = () => {
    if (locationLoading) return 'Getting your location...';
    if (locationError) return `Location Error: ${locationError}`;
    if (currentLocation) {
      return `Location: ${currentLocation.lat.toFixed(4)}, ${currentLocation.lng.toFixed(4)}`;
    }
    return 'No location available';
  };

  const getPermissionStatusText = () => {
    switch (permissionStatus) {
      case 'granted':
        return '✅ Location access granted';
      case 'denied':
        return '❌ Location access denied';
      case 'prompt':
        return '⚠️ Location permission needed';
      default:
        return '❓ Unknown permission status';
    }
  };

  return (
    <div className="location-tracker bg-white rounded-lg shadow-md p-4 mb-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Location Tracking</h3>
        <div className="flex items-center space-x-2">
          {isTracking && (
            <div className="flex items-center text-green-600">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2"></div>
              <span className="text-sm">Tracking Active</span>
            </div>
          )}
        </div>
      </div>

      {/* Permission Status */}
      <div className="mb-3">
        <p className="text-sm text-gray-600">{getPermissionStatusText()}</p>
      </div>

      {/* Current Location Display */}
      <div className="mb-4 p-3 bg-gray-50 rounded-md">
        <p className="text-sm font-medium text-gray-700">Current Location:</p>
        <p className="text-sm text-gray-600">{getLocationStatusText()}</p>
        {currentLocation && (
          <p className="text-xs text-gray-500 mt-1">
            Accuracy: {currentLocation.accuracy ? `±${Math.round(currentLocation.accuracy)}m` : 'Unknown'}
          </p>
        )}
      </div>

      {/* Control Buttons */}
      <div className="flex flex-wrap gap-2 mb-4">
        {!isTracking ? (
          <button
            onClick={handleStartTracking}
            disabled={locationLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {locationLoading ? 'Starting...' : 'Start Tracking'}
          </button>
        ) : (
          <button
            onClick={handleStopTracking}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
          >
            Stop Tracking
          </button>
        )}

        <button
          onClick={getCurrentLocation}
          disabled={locationLoading}
          className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {locationLoading ? 'Getting...' : 'Get Location'}
        </button>
      </div>

      {/* Manual Location Entry */}
      <div className="border-t pt-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Manual Location Entry</h4>
        <div className="flex gap-2">
          <input
            type="text"
            value={manualLocation}
            onChange={(e) => setManualLocation(e.target.value)}
            placeholder="Enter coordinates (lat,lng) or district name"
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleManualLocation}
            className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
          >
            Set
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-1">
          Example: "41.0082, 28.9784" or "Sultanahmet"
        </p>
      </div>

      {/* Error Display */}
      {locationError && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-800">
            <strong>Error:</strong> {locationError}
          </p>
          <p className="text-xs text-red-600 mt-1">
            Try enabling location services or entering coordinates manually.
          </p>
        </div>
      )}

      {/* Session Status */}
      <div className="mt-4 pt-3 border-t">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">
            Session: {sessionActive ? '✅ Active' : '❌ Inactive'}
          </span>
          {sessionActive && (
            <span className="text-xs text-gray-500">
              Real-time updates enabled
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default LocationTracker;
