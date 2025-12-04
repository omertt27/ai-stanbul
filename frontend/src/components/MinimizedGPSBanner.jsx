/**
 * Minimized GPS Banner Component
 * ================================
 * Compact GPS location indicator optimized for mobile chat interface
 * 
 * Features:
 * - Minimal vertical space (32-36px)
 * - Collapsible details
 * - Safe area inset support
 * - Touch-optimized controls
 * - Auto-hide option
 * 
 * Based on: MOBILE_ERGONOMIC_ENHANCEMENTS.md Phase 1
 */

import React, { useState, useEffect } from 'react';
import { MapPin, Navigation, ChevronDown, ChevronUp, X } from 'lucide-react';
import { useLocation } from '../contexts/LocationContext';

const MinimizedGPSBanner = ({ 
  autoHide = false,
  autoHideDelay = 3000,
  allowDismiss = true,
  className = '' 
}) => {
  const {
    currentLocation,
    locationSummary,
    gpsPermission,
    locationSource,
    isLocationRecent,
    locationError,
    isLocationLoading,
  } = useLocation();

  const [isExpanded, setIsExpanded] = useState(false);
  const [isDismissed, setIsDismissed] = useState(false);
  const [isVisible, setIsVisible] = useState(true);

  // Auto-hide after delay if location is set
  useEffect(() => {
    if (autoHide && currentLocation && !locationError && !isLocationLoading) {
      const timer = setTimeout(() => {
        setIsVisible(false);
      }, autoHideDelay);

      return () => clearTimeout(timer);
    }
  }, [autoHide, autoHideDelay, currentLocation, locationError, isLocationLoading]);

  // Don't render if dismissed or auto-hidden
  if (isDismissed || !isVisible) {
    return null;
  }

  const getStatusColor = () => {
    if (locationError) return 'text-red-500';
    if (currentLocation && isLocationRecent) return 'text-green-500';
    if (currentLocation) return 'text-yellow-500';
    return 'text-gray-400';
  };

  const getStatusIcon = () => {
    if (locationSource === 'gps' && isLocationRecent) {
      return <Navigation className="w-4 h-4" />;
    }
    return <MapPin className="w-4 h-4" />;
  };

  const getStatusText = () => {
    if (locationError) return 'Location Error';
    if (isLocationLoading) return 'Getting location...';
    if (!currentLocation) return 'No location set';
    
    // Show summary if available
    if (locationSummary) {
      // Truncate long summaries for mobile
      return locationSummary.length > 30 
        ? `${locationSummary.substring(0, 27)}...` 
        : locationSummary;
    }
    
    if (locationSource === 'gps' && isLocationRecent) return 'GPS Active';
    if (locationSource === 'manual') return 'Manual Location';
    return 'Location Set';
  };

  const handleToggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  const handleDismiss = () => {
    setIsDismissed(true);
  };

  return (
    <div 
      className={`
        minimized-gps-banner
        fixed top-0 left-0 right-0
        bg-gray-900/95
        backdrop-blur-xl
        border-b border-purple-500/20
        z-[999]
        transition-all duration-300 ease-in-out
        ${isExpanded ? 'h-auto' : 'h-9'}
        ${className}
      `}
      style={{
        paddingTop: 'env(safe-area-inset-top, 0px)',
      }}
    >
      {/* Compact Status Bar */}
      <div 
        className="flex items-center justify-between px-4 h-9 cursor-pointer"
        onClick={handleToggleExpand}
      >
        {/* Left: Icon + Status */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <div className={`flex-shrink-0 ${getStatusColor()}`}>
            {getStatusIcon()}
          </div>
          <span className="text-xs text-gray-200 truncate">
            {getStatusText()}
          </span>
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Expand/Collapse Button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleToggleExpand();
            }}
            className="p-1 hover:bg-gray-800/50 rounded-full transition-colors"
            aria-label={isExpanded ? 'Collapse' : 'Expand'}
            style={{ minWidth: '28px', minHeight: '28px' }}
          >
            {isExpanded ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>

          {/* Dismiss Button */}
          {allowDismiss && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDismiss();
              }}
              className="p-1 hover:bg-gray-800/50 rounded-full transition-colors"
              aria-label="Dismiss"
              style={{ minWidth: '28px', minHeight: '28px' }}
            >
              <X className="w-4 h-4 text-gray-400" />
            </button>
          )}
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-4 pb-3 pt-1 space-y-2 border-t border-gray-800/50">
          {/* Location Details */}
          {currentLocation && (
            <div className="space-y-1">
              {locationSummary && (
                <div className="text-xs text-gray-300">
                  📍 {locationSummary}
                </div>
              )}
              
              <div className="text-xs text-gray-400">
                Source: {locationSource === 'gps' ? 'GPS' : locationSource}
                {currentLocation.accuracy && (
                  <span> • Accuracy: ±{Math.round(currentLocation.accuracy)}m</span>
                )}
              </div>

              {currentLocation.lat && currentLocation.lng && (
                <div className="text-xs text-gray-500 font-mono">
                  {currentLocation.lat.toFixed(5)}, {currentLocation.lng.toFixed(5)}
                </div>
              )}
            </div>
          )}

          {/* Error Message with Troubleshooting */}
          {locationError && (
            <div className="space-y-1">
              <div className="text-xs text-red-400 bg-red-500/10 px-2 py-1.5 rounded">
                ⚠️ {locationError}
              </div>
              
              {/* Troubleshooting hints based on error type */}
              {locationError.includes('unavailable') && (
                <div className="text-xs text-gray-400 px-2">
                  💡 Try: Move outdoors • Enable Location Services • Check device settings
                </div>
              )}
              {locationError.includes('denied') && (
                <div className="text-xs text-gray-400 px-2">
                  💡 Enable location in browser settings, then reload the page
                </div>
              )}
              {locationError.includes('timeout') && (
                <div className="text-xs text-gray-400 px-2">
                  💡 Weak signal. Move to open area and try again
                </div>
              )}
            </div>
          )}

          {/* Loading State */}
          {isLocationLoading && (
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <div className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
              <span>Getting your location...</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MinimizedGPSBanner;
