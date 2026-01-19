/**
 * Location Context - Provides location state management across the app
 * Enhanced with GPS Location Service integration
 */

import React, { createContext, useContext, useReducer, useEffect } from 'react';
import locationApi from '../services/locationApi';
import gpsLocationService from '../services/gpsLocationService';
import { Logger } from '../utils/logger.js';

const log = new Logger('LocationContext');

// Initial state
const initialState = {
  // Location tracking
  currentLocation: null,
  locationError: null,
  locationLoading: false,
  
  // Session management
  sessionId: null,
  sessionActive: false,
  
  // POI recommendations
  recommendations: [],
  recommendationsLoading: false,
  recommendationsError: null,
  
  // Route planning
  currentRoute: null,
  routeLoading: false,
  routeError: null,
  
  // Nearby search
  nearbyPOIs: [],
  nearbyLoading: false,
  nearbyError: null,
  
  // Districts
  districts: [],
  districtsLoading: false,
  
  // Real-time tracking
  isTracking: false,
  trackingWatchId: null,
  
  // GPS-specific state
  gpsPermission: 'unknown',
  gpsAccuracy: null,
  lastKnownPosition: null,
  locationTimestamp: null,
  neighborhood: null,
  locationSource: null, // 'gps', 'manual', 'cached'
  
  // User preferences
  preferences: {
    language: 'en',
    transportMode: 'walking',
    radius: 2.0,
    categories: [],
    filters: {}
  }
};

// Action types
const actionTypes = {
  // Location actions
  SET_LOCATION_LOADING: 'SET_LOCATION_LOADING',
  SET_CURRENT_LOCATION: 'SET_CURRENT_LOCATION',
  SET_LOCATION_ERROR: 'SET_LOCATION_ERROR',
  
  // GPS-specific actions
  SET_GPS_PERMISSION: 'SET_GPS_PERMISSION',
  SET_GPS_POSITION: 'SET_GPS_POSITION',
  SET_NEIGHBORHOOD: 'SET_NEIGHBORHOOD',
  SET_LOCATION_SOURCE: 'SET_LOCATION_SOURCE',
  SET_TRACKING: 'SET_TRACKING',
  CLEAR_LOCATION: 'CLEAR_LOCATION',
  
  // Session actions
  SET_SESSION_ACTIVE: 'SET_SESSION_ACTIVE',
  SET_SESSION_ID: 'SET_SESSION_ID',
  
  // Recommendations actions
  SET_RECOMMENDATIONS_LOADING: 'SET_RECOMMENDATIONS_LOADING',
  SET_RECOMMENDATIONS: 'SET_RECOMMENDATIONS',
  SET_RECOMMENDATIONS_ERROR: 'SET_RECOMMENDATIONS_ERROR',
  
  // Route actions
  SET_ROUTE_LOADING: 'SET_ROUTE_LOADING',
  SET_CURRENT_ROUTE: 'SET_CURRENT_ROUTE',
  SET_ROUTE_ERROR: 'SET_ROUTE_ERROR',
  
  // Nearby search actions
  SET_NEARBY_LOADING: 'SET_NEARBY_LOADING',
  SET_NEARBY_POIS: 'SET_NEARBY_POIS',
  SET_NEARBY_ERROR: 'SET_NEARBY_ERROR',
  
  // Districts actions
  SET_DISTRICTS_LOADING: 'SET_DISTRICTS_LOADING',
  SET_DISTRICTS: 'SET_DISTRICTS',
  
  // Tracking actions
  SET_TRACKING_WATCH_ID: 'SET_TRACKING_WATCH_ID',
  
  // Preferences actions
  UPDATE_PREFERENCES: 'UPDATE_PREFERENCES',
  
  // Reset actions
  RESET_STATE: 'RESET_STATE'
};

// Reducer function
function locationReducer(state, action) {
  switch (action.type) {
    case actionTypes.SET_LOCATION_LOADING:
      return { ...state, locationLoading: action.payload };
      
    case actionTypes.SET_CURRENT_LOCATION:
      return { 
        ...state, 
        currentLocation: action.payload, 
        locationError: null,
        locationLoading: false 
      };
      
    case actionTypes.SET_LOCATION_ERROR:
      return { 
        ...state, 
        locationError: action.payload, 
        locationLoading: false 
      };
      
    case actionTypes.SET_GPS_PERMISSION:
      return { ...state, gpsPermission: action.payload };
      
    case actionTypes.SET_GPS_POSITION:
      return { 
        ...state, 
        currentLocation: action.payload.location,
        gpsAccuracy: action.payload.accuracy,
        locationTimestamp: action.payload.timestamp,
        lastKnownPosition: action.payload.location,
        locationError: null,
        locationLoading: false
      };
      
    case actionTypes.SET_NEIGHBORHOOD:
      return { ...state, neighborhood: action.payload };
      
    case actionTypes.SET_LOCATION_SOURCE:
      return { ...state, locationSource: action.payload };
      
    case actionTypes.SET_TRACKING:
      return { 
        ...state, 
        isTracking: action.payload.isTracking,
        trackingWatchId: action.payload.watchId
      };
      
    case actionTypes.CLEAR_LOCATION:
      return { 
        ...state, 
        currentLocation: null,
        gpsAccuracy: null,
        locationTimestamp: null,
        neighborhood: null,
        locationSource: null,
        locationError: null
      };
      
    case actionTypes.SET_SESSION_ACTIVE:
      return { ...state, sessionActive: action.payload };
      
    case actionTypes.SET_SESSION_ID:
      return { ...state, sessionId: action.payload };
      
    case actionTypes.SET_RECOMMENDATIONS_LOADING:
      return { ...state, recommendationsLoading: action.payload };
      
    case actionTypes.SET_RECOMMENDATIONS:
      return { 
        ...state, 
        recommendations: action.payload, 
        recommendationsError: null,
        recommendationsLoading: false 
      };
      
    case actionTypes.SET_RECOMMENDATIONS_ERROR:
      return { 
        ...state, 
        recommendationsError: action.payload, 
        recommendationsLoading: false 
      };
      
    case actionTypes.SET_ROUTE_LOADING:
      return { ...state, routeLoading: action.payload };
      
    case actionTypes.SET_CURRENT_ROUTE:
      return { 
        ...state, 
        currentRoute: action.payload, 
        routeError: null,
        routeLoading: false 
      };
      
    case actionTypes.SET_ROUTE_ERROR:
      return { 
        ...state, 
        routeError: action.payload, 
        routeLoading: false 
      };
      
    case actionTypes.SET_NEARBY_LOADING:
      return { ...state, nearbyLoading: action.payload };
      
    case actionTypes.SET_NEARBY_POIS:
      return { 
        ...state, 
        nearbyPOIs: action.payload, 
        nearbyError: null,
        nearbyLoading: false 
      };
      
    case actionTypes.SET_NEARBY_ERROR:
      return { 
        ...state, 
        nearbyError: action.payload, 
        nearbyLoading: false 
      };
      
    case actionTypes.SET_DISTRICTS_LOADING:
      return { ...state, districtsLoading: action.payload };
      
    case actionTypes.SET_DISTRICTS:
      return { 
        ...state, 
        districts: action.payload, 
        districtsLoading: false 
      };
      
      
    case actionTypes.SET_TRACKING_WATCH_ID:
      return { ...state, trackingWatchId: action.payload };
      
    case actionTypes.UPDATE_PREFERENCES:
      return { 
        ...state, 
        preferences: { ...state.preferences, ...action.payload } 
      };
      
    case actionTypes.RESET_STATE:
      return { ...initialState };
      
    default:
      return state;
  }
}

// Create context
const LocationContext = createContext();

// Custom hook to use location context
export const useLocation = () => {
  const context = useContext(LocationContext);
  if (!context) {
    throw new Error('useLocation must be used within a LocationProvider');
  }
  return context;
};

// Provider component
export const LocationProvider = ({ children }) => {
  const [state, dispatch] = useReducer(locationReducer, initialState);

  // Initialize session from localStorage
  useEffect(() => {
    const savedSessionId = localStorage.getItem('ai_istanbul_session_id');
    const savedPreferences = localStorage.getItem('ai_istanbul_preferences');
    
    if (savedSessionId) {
      dispatch({ type: actionTypes.SET_SESSION_ID, payload: savedSessionId });
      dispatch({ type: actionTypes.SET_SESSION_ACTIVE, payload: true });
    }
    
    if (savedPreferences) {
      try {
        const preferences = JSON.parse(savedPreferences);
        dispatch({ type: actionTypes.UPDATE_PREFERENCES, payload: preferences });
      } catch (error) {
        log.error('Failed to parse saved preferences:', error);
      }
    }
  }, []);

  // Save preferences to localStorage when they change
  useEffect(() => {
    localStorage.setItem('ai_istanbul_preferences', JSON.stringify(state.preferences));
  }, [state.preferences]);

  // Initialize GPS service and set up location listeners
  useEffect(() => {
    const initializeGPS = async () => {
      try {
        // Check permission status
        const permission = await gpsLocationService.getLocationPermissionStatus();
        dispatch({ type: actionTypes.SET_GPS_PERMISSION, payload: permission });
        
        // Try to get last known position
        const lastKnown = gpsLocationService.getLastKnownPosition();
        if (lastKnown) {
          const neighborhood = await gpsLocationService.getNeighborhoodFromCoordinates(lastKnown);
          dispatch({ 
            type: actionTypes.SET_GPS_POSITION, 
            payload: {
              location: { lat: lastKnown.lat, lng: lastKnown.lng },
              accuracy: lastKnown.accuracy,
              timestamp: lastKnown.timestamp
            }
          });
          dispatch({ type: actionTypes.SET_NEIGHBORHOOD, payload: neighborhood });
          dispatch({ type: actionTypes.SET_LOCATION_SOURCE, payload: 'cached' });
        }
      } catch (error) {
        log.error('Error initializing GPS:', error);
      }
    };

    // Set up location update listeners
    const locationUnsubscribe = gpsLocationService.onLocationUpdate(async (position) => {
      try {
        const neighborhood = await gpsLocationService.getNeighborhoodFromCoordinates(position);
        dispatch({ 
          type: actionTypes.SET_GPS_POSITION, 
          payload: {
            location: { lat: position.lat, lng: position.lng },
            accuracy: position.accuracy,
            timestamp: position.timestamp
          }
        });
        dispatch({ type: actionTypes.SET_NEIGHBORHOOD, payload: neighborhood });
        dispatch({ type: actionTypes.SET_LOCATION_SOURCE, payload: 'gps' });
      } catch (error) {
        log.error('Error processing location update:', error);
      }
    });

    const errorUnsubscribe = gpsLocationService.onLocationError((error) => {
      // Enhanced error message with troubleshooting
      let userFriendlyMessage = error.message;
      
      // Check if enhanced error object with tips
      if (error.title) {
        userFriendlyMessage = `${error.title}: ${error.message}`;
      } else {
        // Format standard error messages
        if (error.message.includes('unavailable')) {
          userFriendlyMessage = 'GPS signal unavailable. Try moving to an area with better signal.';
        } else if (error.message.includes('denied')) {
          userFriendlyMessage = 'Location access denied. Please enable in settings.';
        } else if (error.message.includes('timeout')) {
          userFriendlyMessage = 'GPS request timeout. Signal may be weak.';
        }
      }
      
      dispatch({ type: actionTypes.SET_LOCATION_ERROR, payload: userFriendlyMessage });
    });

    initializeGPS();

    // Cleanup
    return () => {
      locationUnsubscribe();
      errorUnsubscribe();
      gpsLocationService.stopLocationTracking();
    };
  }, []);

  // Action creators
  const actions = {
    // Location actions
    async getCurrentLocation() {
      dispatch({ type: actionTypes.SET_LOCATION_LOADING, payload: true });
      
      try {
        const location = await locationApi.getCurrentLocation();
        dispatch({ type: actionTypes.SET_CURRENT_LOCATION, payload: location });
        return location;
      } catch (error) {
        dispatch({ type: actionTypes.SET_LOCATION_ERROR, payload: error.message });
        throw error;
      }
    },

    // GPS-specific actions
    async requestGPSLocation(options = {}) {
      dispatch({ type: actionTypes.SET_LOCATION_LOADING, payload: true });
      
      try {
        const position = await gpsLocationService.requestLocationPermission();
        const neighborhood = await gpsLocationService.getNeighborhoodFromCoordinates(position);
        
        dispatch({ 
          type: actionTypes.SET_GPS_POSITION, 
          payload: {
            location: { lat: position.lat, lng: position.lng },
            accuracy: position.accuracy,
            timestamp: position.timestamp
          }
        });
        
        dispatch({ type: actionTypes.SET_NEIGHBORHOOD, payload: neighborhood });
        dispatch({ type: actionTypes.SET_LOCATION_SOURCE, payload: 'gps' });
        
        // Update permission status
        const permission = await gpsLocationService.getLocationPermissionStatus();
        dispatch({ type: actionTypes.SET_GPS_PERMISSION, payload: permission });
        
        return { ...position, neighborhood };
      } catch (error) {
        dispatch({ type: actionTypes.SET_LOCATION_ERROR, payload: error.message });
        
        // Update permission status on error
        const permission = await gpsLocationService.getLocationPermissionStatus();
        dispatch({ type: actionTypes.SET_GPS_PERMISSION, payload: permission });
        
        throw error;
      }
    },

    async setManualLocation(locationData) {
      try {
        let processedLocation;
        
        if (typeof locationData === 'string') {
          processedLocation = {
            lat: null,
            lng: null,
            neighborhood: locationData
          };
        } else {
          const neighborhood = locationData.neighborhood || 
            (locationData.lat && locationData.lng ? 
              await gpsLocationService.getNeighborhoodFromCoordinates(locationData) : 
              'Unknown Area');
              
          processedLocation = {
            lat: locationData.lat || null,
            lng: locationData.lng || null,
            neighborhood
          };
        }
        
        dispatch({ type: actionTypes.SET_CURRENT_LOCATION, payload: processedLocation });
        dispatch({ type: actionTypes.SET_NEIGHBORHOOD, payload: processedLocation.neighborhood });
        dispatch({ type: actionTypes.SET_LOCATION_SOURCE, payload: 'manual' });
        
        return processedLocation;
      } catch (error) {
        dispatch({ type: actionTypes.SET_LOCATION_ERROR, payload: 'Failed to set manual location' });
        throw error;
      }
    },

    startGPSTracking() {
      try {
        const watchId = gpsLocationService.startLocationTracking();
        dispatch({ type: actionTypes.SET_TRACKING, payload: { isTracking: true, watchId } });
        return watchId;
      } catch (error) {
        dispatch({ type: actionTypes.SET_LOCATION_ERROR, payload: error.message });
        throw error;
      }
    },

    stopGPSTracking() {
      gpsLocationService.stopLocationTracking();
      dispatch({ type: actionTypes.SET_TRACKING, payload: { isTracking: false, watchId: null } });
    },

    clearLocation() {
      dispatch({ type: actionTypes.CLEAR_LOCATION });
      gpsLocationService.stopLocationTracking();
    },

    getDistanceToLocation(targetLocation) {
      if (!state.currentLocation || !targetLocation) return null;
      
      return gpsLocationService.calculateDistance(
        state.currentLocation,
        targetLocation
      );
    },

    async createSession(userLocation, preferences = {}) {
      try {
        const response = await locationApi.createSession(userLocation, preferences);
        dispatch({ type: actionTypes.SET_SESSION_ID, payload: response.session_id });
        dispatch({ type: actionTypes.SET_SESSION_ACTIVE, payload: true });
        return response;
      } catch (error) {
        log.error('Failed to create session:', error);
        throw error;
      }
    },

    async updateLocation(newLocation) {
      try {
        const response = await locationApi.updateLocation(newLocation);
        dispatch({ type: actionTypes.SET_CURRENT_LOCATION, payload: newLocation });
        return response;
      } catch (error) {
        dispatch({ type: actionTypes.SET_LOCATION_ERROR, payload: error.message });
        throw error;
      }
    },

    async getRecommendations(options = {}) {
      dispatch({ type: actionTypes.SET_RECOMMENDATIONS_LOADING, payload: true });
      
      try {
        const response = await locationApi.getRecommendations({
          ...options,
          categories: state.preferences.categories,
          radius: state.preferences.radius,
          filters: state.preferences.filters
        });
        
        dispatch({ type: actionTypes.SET_RECOMMENDATIONS, payload: response.recommendations || [] });
        return response;
      } catch (error) {
        dispatch({ type: actionTypes.SET_RECOMMENDATIONS_ERROR, payload: error.message });
        throw error;
      }
    },

    async planRoute(targetPois, options = {}) {
      dispatch({ type: actionTypes.SET_ROUTE_LOADING, payload: true });
      
      try {
        const response = await locationApi.planRoute(targetPois, {
          transport: state.preferences.transportMode,
          ...options
        });
        
        dispatch({ type: actionTypes.SET_CURRENT_ROUTE, payload: response });
        return response;
      } catch (error) {
        dispatch({ type: actionTypes.SET_ROUTE_ERROR, payload: error.message });
        throw error;
      }
    },

    async searchNearby(location, options = {}) {
      dispatch({ type: actionTypes.SET_NEARBY_LOADING, payload: true });
      
      try {
        const response = await locationApi.searchNearby(location || state.currentLocation, {
          radius: state.preferences.radius,
          ...options
        });
        
        dispatch({ type: actionTypes.SET_NEARBY_POIS, payload: response.pois || [] });
        return response;
      } catch (error) {
        dispatch({ type: actionTypes.SET_NEARBY_ERROR, payload: error.message });
        throw error;
      }
    },

    async loadDistricts() {
      dispatch({ type: actionTypes.SET_DISTRICTS_LOADING, payload: true });
      
      try {
        const response = await locationApi.getDistricts();
        dispatch({ type: actionTypes.SET_DISTRICTS, payload: response.districts || [] });
        return response;
      } catch (error) {
        log.error('Failed to load districts:', error);
        throw error;
      }
    },

    startLocationTracking() {
      if (state.isTracking) return;
      
      const watchId = locationApi.watchLocation(
        (location) => {
          dispatch({ type: actionTypes.SET_CURRENT_LOCATION, payload: location });
          // Auto-update session location if session is active
          if (state.sessionActive) {
            locationApi.updateLocation(location).catch(log.error);
          }
        },
        (error) => {
          dispatch({ type: actionTypes.SET_LOCATION_ERROR, payload: error.message });
        }
      );
      
      if (watchId) {
        dispatch({ type: actionTypes.SET_TRACKING, payload: true });
        dispatch({ type: actionTypes.SET_TRACKING_WATCH_ID, payload: watchId });
      }
    },

    stopLocationTracking() {
      if (state.trackingWatchId) {
        locationApi.clearLocationWatch(state.trackingWatchId);
        dispatch({ type: actionTypes.SET_TRACKING, payload: false });
        dispatch({ type: actionTypes.SET_TRACKING_WATCH_ID, payload: null });
      }
    },

    updatePreferences(newPreferences) {
      dispatch({ type: actionTypes.UPDATE_PREFERENCES, payload: newPreferences });
    },

    async cleanupSession() {
      try {
        await locationApi.cleanupSession();
        dispatch({ type: actionTypes.SET_SESSION_ACTIVE, payload: false });
        dispatch({ type: actionTypes.SET_SESSION_ID, payload: null });
      } catch (error) {
        log.error('Failed to cleanup session:', error);
      }
    },

    resetState() {
      dispatch({ type: actionTypes.RESET_STATE });
    }
  };

  const value = {
    // State
    ...state,
    
    // Actions
    ...actions,
    
    // Computed values
    hasLocation: !!state.currentLocation,
    hasRecommendations: state.recommendations.length > 0,
    hasRoute: !!state.currentRoute,
    hasNearbyPOIs: state.nearbyPOIs.length > 0,
    
    // GPS-specific computed values
    hasGPSLocation: !!(state.currentLocation && state.locationSource === 'gps'),
    hasLocationPermission: state.gpsPermission === 'granted',
    isLocationRecent: state.locationTimestamp ? 
      (Date.now() - new Date(state.locationTimestamp).getTime() < 300000) : false, // 5 minutes
    locationSummary: state.neighborhood || 
      (state.currentLocation ? 
        `${state.currentLocation.lat?.toFixed(4)}, ${state.currentLocation.lng?.toFixed(4)}` : 
        'No location'),
    
    // Utility methods
    isLocationAvailable: gpsLocationService.isLocationAvailable(),
    formatLocationForAI: () => ({
      lat: state.currentLocation?.lat || null,
      lng: state.currentLocation?.lng || null,
      accuracy: state.gpsAccuracy,
      timestamp: state.locationTimestamp,
      neighborhood: state.neighborhood,
      source: state.locationSource
    })
  };

  return (
    <LocationContext.Provider value={value}>
      {children}
    </LocationContext.Provider>
  );
};

export default LocationContext;
