import React, { useState } from 'react';
import { MapPin, Navigation, AlertCircle, Settings, RefreshCw, X } from 'lucide-react';
import { useLocation } from '../contexts/LocationContext';
import LocationPermissionModal from './LocationPermissionModal';

const GPSLocationStatus = ({ className = '', showDetails = true }) => {
  const {
    currentLocation,
    locationSummary,
    gpsPermission,
    locationSource,
    gpsAccuracy,
    isLocationRecent,
    locationError,
    isLocationLoading,
    hasLocationPermission,
    isTracking,
    requestGPSLocation,
    setManualLocation,
    startGPSTracking,
    stopGPSTracking,
    clearLocation
  } = useLocation();

  const [showPermissionModal, setShowPermissionModal] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const getLocationStatusColor = () => {
    if (locationError) return 'text-red-600';
    if (currentLocation && isLocationRecent) return 'text-green-600';
    if (currentLocation) return 'text-yellow-600';
    return 'text-gray-400';
  };

  const getLocationStatusIcon = () => {
    if (isLocationLoading) {
      return <RefreshCw className="w-4 h-4 animate-spin" />;
    }
    if (locationError) {
      return <AlertCircle className="w-4 h-4" />;
    }
    if (locationSource === 'gps' && isLocationRecent) {
      return <Navigation className="w-4 h-4" />;
    }
    return <MapPin className="w-4 h-4" />;
  };

  const getLocationStatusText = () => {
    if (locationError) return 'Location Error';
    if (isLocationLoading) return 'Getting location...';
    if (!currentLocation) return 'No location';
    if (locationSource === 'gps' && isLocationRecent) return 'GPS Active';
    if (locationSource === 'gps') return 'GPS (Outdated)';
    if (locationSource === 'manual') return 'Manual Location';
    if (locationSource === 'cached') return 'Cached Location';
    return 'Location Set';
  };

  const handleLocationRequest = async () => {
    try {
      if (!hasLocationPermission) {
        setShowPermissionModal(true);
      } else {
        await requestGPSLocation();
      }
    } catch (error) {
      console.error('Location request failed:', error);
    }
  };

  const handlePermissionGranted = async (locationData) => {
    try {
      if (locationData.latitude && locationData.longitude) {
        // GPS location
        await requestGPSLocation();
      } else {
        // Manual location
        await setManualLocation(locationData.name || locationData.address);
      }
      setShowPermissionModal(false);
    } catch (error) {
      console.error('Failed to set location:', error);
    }
  };

  const handlePermissionDenied = () => {
    setShowPermissionModal(false);
  };

  const toggleTracking = () => {
    if (isTracking) {
      stopGPSTracking();
    } else {
      startGPSTracking();
    }
  };

  if (!showDetails) {
    // Compact mode - just icon and status
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <div className={getLocationStatusColor()}>
          {getLocationStatusIcon()}
        </div>
        <span className={`text-sm ${getLocationStatusColor()}`}>
          {locationSummary}
        </span>
      </div>
    );
  }

  return (
    <>
      <div className={`bg-white rounded-lg border shadow-sm p-4 ${className}`}>
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <div className={getLocationStatusColor()}>
              {getLocationStatusIcon()}
            </div>
            <span className="font-medium text-gray-900">
              {getLocationStatusText()}
            </span>
          </div>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>

        {/* Location Display */}
        <div className="space-y-2">
          <div className="text-sm text-gray-600">
            <strong>Location:</strong> {locationSummary}
          </div>
          
          {gpsAccuracy && (
            <div className="text-sm text-gray-600">
              <strong>Accuracy:</strong> ±{Math.round(gpsAccuracy)}m
            </div>
          )}
          
          {locationSource && (
            <div className="text-sm text-gray-600">
              <strong>Source:</strong> {locationSource.charAt(0).toUpperCase() + locationSource.slice(1)}
            </div>
          )}
        </div>

        {/* Error Display */}
        {locationError && (
          <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-800">
            {locationError}
          </div>
        )}

        {/* Settings Panel */}
        {showSettings && (
          <div className="mt-4 pt-4 border-t space-y-3">
            <div className="grid grid-cols-2 gap-2">
              {/* Request Location Button */}
              <button
                onClick={handleLocationRequest}
                disabled={isLocationLoading}
                className="flex items-center justify-center px-3 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
              >
                <Navigation className="w-4 h-4 mr-1" />
                {hasLocationPermission ? 'Update GPS' : 'Enable GPS'}
              </button>

              {/* Tracking Toggle */}
              {hasLocationPermission && (
                <button
                  onClick={toggleTracking}
                  className={`flex items-center justify-center px-3 py-2 rounded text-sm transition-colors ${
                    isTracking 
                      ? 'bg-green-600 text-white hover:bg-green-700' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  {isTracking ? 'Stop Tracking' : 'Start Tracking'}
                </button>
              )}
            </div>

            {/* Clear Location Button */}
            {currentLocation && (
              <button
                onClick={clearLocation}
                className="w-full flex items-center justify-center px-3 py-2 bg-red-100 text-red-700 rounded text-sm hover:bg-red-200 transition-colors"
              >
                <X className="w-4 h-4 mr-1" />
                Clear Location
              </button>
            )}

            {/* Permission Status */}
            <div className="text-xs text-gray-500">
              Permission: {gpsPermission === 'granted' ? '✅ Granted' : 
                          gpsPermission === 'denied' ? '❌ Denied' : 
                          '❓ Unknown'}
            </div>
          </div>
        )}

        {/* Quick Actions (when no settings shown) */}
        {!showSettings && !currentLocation && (
          <div className="mt-3">
            <button
              onClick={handleLocationRequest}
              disabled={isLocationLoading}
              className="w-full flex items-center justify-center px-3 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
            >
              <Navigation className="w-4 h-4 mr-1" />
              Get My Location
            </button>
          </div>
        )}
      </div>

      {/* Location Permission Modal */}
      <LocationPermissionModal
        isOpen={showPermissionModal}
        onClose={() => setShowPermissionModal(false)}
        onLocationSet={handlePermissionGranted}
      />
    </>
  );
};

export default GPSLocationStatus;
