/**
 * Location Context - Provides location state management across the app
 */

import React, { createContext, useContext, useReducer, useEffect } from 'react';
import locationApi from '../services/locationApi';

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
  SET_TRACKING: 'SET_TRACKING',
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
      
    case actionTypes.SET_TRACKING:
      return { ...state, isTracking: action.payload };
      
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
        console.error('Failed to parse saved preferences:', error);
      }
    }
  }, []);

  // Save preferences to localStorage when they change
  useEffect(() => {
    localStorage.setItem('ai_istanbul_preferences', JSON.stringify(state.preferences));
  }, [state.preferences]);

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

    async createSession(userLocation, preferences = {}) {
      try {
        const response = await locationApi.createSession(userLocation, preferences);
        dispatch({ type: actionTypes.SET_SESSION_ID, payload: response.session_id });
        dispatch({ type: actionTypes.SET_SESSION_ACTIVE, payload: true });
        return response;
      } catch (error) {
        console.error('Failed to create session:', error);
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
        console.error('Failed to load districts:', error);
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
            locationApi.updateLocation(location).catch(console.error);
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
        console.error('Failed to cleanup session:', error);
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
    hasNearbyPOIs: state.nearbyPOIs.length > 0
  };

  return (
    <LocationContext.Provider value={value}>
      {children}
    </LocationContext.Provider>
  );
};

export default LocationContext;
