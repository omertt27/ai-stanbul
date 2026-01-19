/**
 * Custom hook for location-based features
 * Provides easy access to location services for any component
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocation } from '../contexts/LocationContext';
import locationApi from '../services/locationApi';
import { Logger } from '../utils/logger';

const logger = new Logger('useLocationSystem');

/**
 * Hook for managing location tracking
 */
export const useLocationTracking = (options = {}) => {
  const {
    currentLocation,
    sessionActive,
    isTracking,
    getCurrentLocation,
    createSession,
    updateLocation,
    startLocationTracking,
    stopLocationTracking
  } = useLocation();

  const { autoStart = false, onLocationChange } = options;

  useEffect(() => {
    if (autoStart && !isTracking) {
      startLocationTracking();
    }
  }, [autoStart, isTracking, startLocationTracking]);

  useEffect(() => {
    if (currentLocation && onLocationChange) {
      onLocationChange(currentLocation);
    }
  }, [currentLocation, onLocationChange]);

  return {
    currentLocation,
    sessionActive,
    isTracking,
    getCurrentLocation,
    createSession,
    updateLocation,
    startLocationTracking,
    stopLocationTracking
  };
};

/**
 * Hook for POI recommendations
 */
export const usePOIRecommendations = (options = {}) => {
  const {
    recommendations,
    recommendationsLoading,
    recommendationsError,
    getRecommendations,
    currentLocation
  } = useLocation();

  const [filters, setFilters] = useState({
    categories: options.categories || [],
    radius: options.radius || 2.0,
    rating: options.minRating || 0,
    openNow: options.openNow || false
  });

  const loadRecommendations = useCallback(async (customFilters = {}) => {
    const finalFilters = { ...filters, ...customFilters };
    
    try {
      await getRecommendations({
        categories: finalFilters.categories,
        radius: finalFilters.radius,
        filters: {
          rating: finalFilters.rating > 0 ? finalFilters.rating : undefined,
          open_now: finalFilters.openNow
        }
      });
    } catch (error) {
      logger.error('Failed to load recommendations:', error);
      throw error;
    }
  }, [filters, getRecommendations]);

  // Auto-load recommendations when location changes
  useEffect(() => {
    if (currentLocation && options.autoLoad) {
      loadRecommendations();
    }
  }, [currentLocation, options.autoLoad, loadRecommendations]);

  return {
    recommendations,
    loading: recommendationsLoading,
    error: recommendationsError,
    filters,
    setFilters,
    loadRecommendations
  };
};

/**
 * Hook for route planning
 */
export const useRoutePlanning = () => {
  const {
    currentRoute,
    routeLoading,
    routeError,
    planRoute,
    preferences
  } = useLocation();

  const [selectedPOIs, setSelectedPOIs] = useState([]);

  const planOptimalRoute = useCallback(async (pois, options = {}) => {
    const routeOptions = {
      algorithm: options.algorithm || 'tsp_nearest',
      transport: options.transport || preferences.transportMode || 'walking',
      optimizeFor: options.optimizeFor || 'time'
    };

    try {
      const poiIds = Array.isArray(pois) ? pois : [pois];
      await planRoute(poiIds, routeOptions);
    } catch (error) {
      logger.error('Failed to plan route:', error);
      throw error;
    }
  }, [planRoute, preferences.transportMode]);

  const addPOIToRoute = useCallback((poi) => {
    setSelectedPOIs(prev => {
      const exists = prev.some(p => p.id === poi.id);
      if (exists) return prev;
      return [...prev, poi];
    });
  }, []);

  const removePOIFromRoute = useCallback((poiId) => {
    setSelectedPOIs(prev => prev.filter(p => p.id !== poiId));
  }, []);

  const clearSelectedPOIs = useCallback(() => {
    setSelectedPOIs([]);
  }, []);

  const planSelectedRoute = useCallback(async (options = {}) => {
    if (selectedPOIs.length === 0) {
      throw new Error('No POIs selected for route planning');
    }

    const poiIds = selectedPOIs.map(poi => poi.id);
    await planOptimalRoute(poiIds, options);
  }, [selectedPOIs, planOptimalRoute]);

  return {
    currentRoute,
    loading: routeLoading,
    error: routeError,
    selectedPOIs,
    addPOIToRoute,
    removePOIFromRoute,
    clearSelectedPOIs,
    planOptimalRoute,
    planSelectedRoute
  };
};

/**
 * Hook for nearby search
 */
export const useNearbySearch = (options = {}) => {
  const {
    nearbyPOIs,
    nearbyLoading,
    nearbyError,
    searchNearby,
    currentLocation
  } = useLocation();

  const [searchParams, setSearchParams] = useState({
    radius: options.radius || 1.0,
    category: options.category || null,
    filters: options.filters || {}
  });

  const search = useCallback(async (location = null, customParams = {}) => {
    const searchLocation = location || currentLocation;
    if (!searchLocation) {
      throw new Error('No location available for search');
    }

    const params = { ...searchParams, ...customParams };
    
    try {
      await searchNearby(searchLocation, params);
    } catch (error) {
      logger.error('Failed to search nearby:', error);
      throw error;
    }
  }, [searchNearby, currentLocation, searchParams]);

  // Auto-search when location changes
  useEffect(() => {
    if (currentLocation && options.autoSearch) {
      search();
    }
  }, [currentLocation, options.autoSearch, search]);

  return {
    nearbyPOIs,
    loading: nearbyLoading,
    error: nearbyError,
    searchParams,
    setSearchParams,
    search
  };
};

/**
 * Hook for geolocation utilities
 */
export const useGeolocation = () => {
  const [position, setPosition] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const getCurrentPosition = useCallback(() => {
    if (!navigator.geolocation) {
      setError(new Error('Geolocation is not supported'));
      return;
    }

    setLoading(true);
    setError(null);

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setPosition({
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
          timestamp: new Date(pos.timestamp)
        });
        setLoading(false);
      },
      (err) => {
        setError(err);
        setLoading(false);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 300000
      }
    );
  }, []);

  const watchPosition = useCallback((callback) => {
    if (!navigator.geolocation) {
      setError(new Error('Geolocation is not supported'));
      return null;
    }

    return navigator.geolocation.watchPosition(
      (pos) => {
        const position = {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
          timestamp: new Date(pos.timestamp)
        };
        setPosition(position);
        callback(position);
      },
      (err) => {
        setError(err);
      },
      {
        enableHighAccuracy: true,
        timeout: 30000,
        maximumAge: 60000
      }
    );
  }, []);

  const clearWatch = useCallback((watchId) => {
    if (watchId && navigator.geolocation) {
      navigator.geolocation.clearWatch(watchId);
    }
  }, []);

  return {
    position,
    error,
    loading,
    getCurrentPosition,
    watchPosition,
    clearWatch
  };
};

/**
 * Hook for distance calculations
 */
export const useDistanceCalculations = () => {
  const haversineDistance = useCallback((coord1, coord2) => {
    const R = 6371; // Earth's radius in kilometers
    
    const lat1 = coord1.lat * Math.PI / 180;
    const lat2 = coord2.lat * Math.PI / 180;
    const deltaLat = (coord2.lat - coord1.lat) * Math.PI / 180;
    const deltaLng = (coord2.lng - coord1.lng) * Math.PI / 180;

    const a = Math.sin(deltaLat/2) * Math.sin(deltaLat/2) +
              Math.cos(lat1) * Math.cos(lat2) *
              Math.sin(deltaLng/2) * Math.sin(deltaLng/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

    return R * c;
  }, []);

  const formatDistance = useCallback((km) => {
    if (km < 1) {
      return `${Math.round(km * 1000)}m`;
    }
    return `${km.toFixed(1)}km`;
  }, []);

  const estimateWalkingTime = useCallback((distanceKm, speedKmh = 5) => {
    return (distanceKm / speedKmh) * 60; // minutes
  }, []);

  const formatTime = useCallback((minutes) => {
    if (minutes < 60) {
      return `${Math.round(minutes)}min`;
    }
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}min`;
  }, []);

  return {
    haversineDistance,
    formatDistance,
    estimateWalkingTime,
    formatTime
  };
};

/**
 * Combined hook for full location functionality
 */
export const useLocationSystem = (options = {}) => {
  const locationTracking = useLocationTracking(options.tracking);
  const poiRecommendations = usePOIRecommendations(options.recommendations);
  const routePlanning = useRoutePlanning();
  const nearbySearch = useNearbySearch(options.nearby);
  const distanceCalculations = useDistanceCalculations();

  return {
    tracking: locationTracking,
    recommendations: poiRecommendations,
    routing: routePlanning,
    nearby: nearbySearch,
    utils: distanceCalculations
  };
};
