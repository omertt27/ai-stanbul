/**
 * UnifiedRouteCard Component
 * ==========================
 * 
 * Single unified component for all map and route visualizations.
 * Replaces RouteCard, MapVisualization, and TransportationRouteCard.
 * 
 * Features:
 * - Auto-detects display mode based on data structure
 * - Full-featured route display with embedded map
 * - Multi-route comparison
 * - Simple route display (no map)
 * - POI/attraction map display
 * - Standalone map view
 * - All Priority 1-5.1 features included
 * 
 * Modes:
 * - 'full-route': Complete route with map, steps, and all features
 * - 'simple-route': Route steps without map (legacy fallback)
 * - 'map-poi': POI markers on map (attractions, restaurants)
 * - 'multi-route': Route comparison view
 * - 'map-only': Standalone map visualization
 * - 'auto': Auto-detect mode based on data (default)
 * 
 * Props:
 * - data: Object - Route/map data from backend
 * - mode: String - Display mode ('auto', 'full-route', 'simple-route', 'map-poi', 'multi-route', 'map-only')
 * - darkMode: Boolean - Dark mode flag
 * - onCopy: Function - Callback for copy action
 * - onShare: Function - Callback for share action
 * - onStartNavigation: Function - Callback for navigation start
 * - showQuickActions: Boolean - Show quick action buttons
 * - className: String - Additional CSS classes
 * 
 * Author: AI Istanbul Team
 * Date: January 12, 2026
 */

import React, { useState, useEffect, lazy, Suspense } from 'react';

// Import view components
const FullRouteView = lazy(() => import('./views/FullRouteView'));
const SimpleRouteView = lazy(() => import('./views/SimpleRouteView'));
const MapPOIView = lazy(() => import('./views/MapPOIView'));
const MultiRouteView = lazy(() => import('./views/MultiRouteView'));
const MapOnlyView = lazy(() => import('./views/MapOnlyView'));

/**
 * Mode Detection Logic
 * Analyzes data structure to determine best display mode
 */
const detectMode = (data) => {
  if (!data) return 'map-only';

  // Check for multi-route comparison data
  if (data.alternatives && data.alternatives.length > 1) {
    return 'multi-route';
  }

  if (data.alternative_routes && data.alternative_routes.length > 1) {
    return 'multi-route';
  }

  // Check for full route data (route_info + map_data)
  if ((data.route_info || data.routeInfo) && (data.map_data || data.mapData)) {
    return 'full-route';
  }

  // Check for simple route data (route_info without map)
  if ((data.route_info || data.routeInfo || data.routeData) && !(data.map_data || data.mapData)) {
    return 'simple-route';
  }

  // Check for POI/attraction map data
  if (data.map_data && data.map_data.markers && data.map_data.markers.length > 0) {
    const markers = data.map_data.markers;
    const hasPOI = markers.some(m => 
      m.type === 'attraction' || 
      m.type === 'restaurant' || 
      m.type === 'poi'
    );
    if (hasPOI) return 'map-poi';
  }

  // Check for standalone map
  if (data.map_data || data.mapData) {
    return 'map-only';
  }

  // Fallback
  return 'map-only';
};

/**
 * Loading Skeleton Component
 */
const LoadingSkeleton = ({ darkMode }) => {
  return (
    <div className={`animate-pulse space-y-3 p-4 rounded-lg ${
      darkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      <div className={`h-48 rounded-lg ${
        darkMode ? 'bg-gray-700' : 'bg-gray-200'
      }`} />
      <div className={`h-4 rounded w-3/4 ${
        darkMode ? 'bg-gray-700' : 'bg-gray-200'
      }`} />
      <div className={`h-4 rounded w-1/2 ${
        darkMode ? 'bg-gray-700' : 'bg-gray-200'
      }`} />
    </div>
  );
};

/**
 * Error Display Component
 */
const ErrorDisplay = ({ error, darkMode, onRetry }) => {
  return (
    <div className={`text-center p-6 rounded-lg ${
      darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'
    }`}>
      <span className="text-4xl mb-3 block">🗺️</span>
      <h3 className="text-lg font-semibold mb-2">Couldn't load map</h3>
      <p className={`text-sm mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
        {error?.message || 'An error occurred while loading the map'}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            darkMode
              ? 'bg-indigo-600 hover:bg-indigo-500 text-white'
              : 'bg-indigo-600 hover:bg-indigo-700 text-white'
          }`}
        >
          🔄 Try Again
        </button>
      )}
    </div>
  );
};

/**
 * Main UnifiedRouteCard Component
 */
const UnifiedRouteCard = ({
  data,
  mode = 'auto',
  darkMode = false,
  onCopy,
  onShare,
  onStartNavigation,
  showQuickActions = true,
  className = '',
  ...otherProps
}) => {
  const [displayMode, setDisplayMode] = useState(mode);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  // Detect mode on data change
  useEffect(() => {
    try {
      if (mode === 'auto') {
        const detected = detectMode(data);
        setDisplayMode(detected);
        console.log('🎯 UnifiedRouteCard detected mode:', detected);
      } else {
        setDisplayMode(mode);
        console.log('🎯 UnifiedRouteCard using explicit mode:', mode);
      }
      setError(null);
    } catch (err) {
      console.error('❌ Error detecting mode:', err);
      setError(err);
      setDisplayMode('map-only');
    }
  }, [data, mode, retryCount]);

  // Retry handler
  const handleRetry = () => {
    setRetryCount(prev => prev + 1);
    setError(null);
  };

  // Error boundary fallback
  if (error) {
    return <ErrorDisplay error={error} darkMode={darkMode} onRetry={handleRetry} />;
  }

  if (!data) {
    return (
      <div className={`text-center p-6 rounded-lg ${
        darkMode ? 'bg-gray-800 text-gray-400' : 'bg-white text-gray-600'
      }`}>
        <span className="text-4xl mb-3 block">🚫</span>
        <p className="text-sm">No map data available</p>
      </div>
    );
  }

  // Common props for all views
  const commonProps = {
    data,
    darkMode,
    onCopy,
    onShare,
    onStartNavigation,
    showQuickActions,
    ...otherProps
  };

  // Render based on detected mode
  return (
    <div className={`unified-route-card ${className}`} data-mode={displayMode}>
      <Suspense fallback={<LoadingSkeleton darkMode={darkMode} />}>
        {displayMode === 'full-route' && <FullRouteView {...commonProps} />}
        {displayMode === 'simple-route' && <SimpleRouteView {...commonProps} />}
        {displayMode === 'map-poi' && <MapPOIView {...commonProps} />}
        {displayMode === 'multi-route' && <MultiRouteView {...commonProps} />}
        {displayMode === 'map-only' && <MapOnlyView {...commonProps} />}
      </Suspense>
    </div>
  );
};

export default UnifiedRouteCard;
export { detectMode };
