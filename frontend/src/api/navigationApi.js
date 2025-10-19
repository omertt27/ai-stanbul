/**
 * Navigation API Service
 * Handles all navigation-related API calls including:
 * - AI conversational navigation
 * - GPS-based routing (OSRM)
 * - Multi-waypoint optimization
 * - ML-enhanced hybrid routing
 */

import { fetchWithRetry, handleApiError } from '../utils/errorHandler';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * AI Conversational Navigation
 * POST /api/chat/navigation
 */
export const chatNavigation = async (query, userContext = null) => {
  try {
    console.log('🗺️ AI Navigation Request:', query);
    
    const requestBody = {
      query,
      user_context: userContext || {}
    };

    const response = await fetchWithRetry(`${BASE_URL}/api/chat/navigation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(requestBody),
      timeout: 30000
    }, {
      maxAttempts: 2,
      baseDelay: 1000
    });

    const data = await response.json();
    console.log('✅ Navigation Response:', data);
    
    return {
      response: data.response,
      metadata: data.metadata || {},
      intent: data.intent,
      confidence: data.confidence,
      suggestions: data.suggestions || []
    };
    
  } catch (error) {
    console.error('Navigation API error:', error);
    throw handleApiError(error, null, 'AI Navigation');
  }
};

/**
 * GPS-Based Route Planning
 * POST /api/route/gps-plan
 */
export const planGPSRoute = async (routeRequest) => {
  try {
    console.log('🧭 GPS Route Planning:', routeRequest);
    
    const response = await fetchWithRetry(`${BASE_URL}/api/route/gps-plan`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(routeRequest),
      timeout: 30000
    }, {
      maxAttempts: 2,
      baseDelay: 1000
    });

    const data = await response.json();
    console.log('✅ GPS Route:', data);
    
    return {
      route: data.route || data,
      distance: data.distance,
      duration: data.duration,
      geometry: data.geometry || data.route_geometry,
      segments: data.segments || [],
      mode: data.mode || routeRequest.mode
    };
    
  } catch (error) {
    console.error('GPS routing error:', error);
    throw handleApiError(error, null, 'GPS Route Planning');
  }
};

/**
 * Multi-Waypoint Optimization
 * POST /api/route/gps-optimize
 */
export const optimizeRoute = async (waypoints, options = {}) => {
  try {
    console.log('🎯 Optimizing route for waypoints:', waypoints);
    
    const requestBody = {
      waypoints,
      ...options
    };

    const response = await fetchWithRetry(`${BASE_URL}/api/route/gps-optimize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(requestBody),
      timeout: 30000
    }, {
      maxAttempts: 2,
      baseDelay: 1000
    });

    const data = await response.json();
    console.log('✅ Optimized Route:', data);
    
    return {
      route: data.route || data,
      optimized_order: data.optimized_order || [],
      total_distance: data.total_distance,
      total_duration: data.total_duration,
      geometry: data.geometry
    };
    
  } catch (error) {
    console.error('Route optimization error:', error);
    throw handleApiError(error, null, 'Route Optimization');
  }
};

/**
 * ML-Enhanced Hybrid Routing
 * POST /api/route/hybrid-plan
 */
export const planHybridRoute = async (routeRequest) => {
  try {
    console.log('🤖 ML-Enhanced Hybrid Route:', routeRequest);
    
    const response = await fetchWithRetry(`${BASE_URL}/api/route/hybrid-plan`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(routeRequest),
      timeout: 30000
    }, {
      maxAttempts: 3,
      baseDelay: 1000
    });

    const data = await response.json();
    console.log('✅ Hybrid Route:', data);
    
    return {
      route: data.route || data,
      distance: data.distance,
      duration: data.duration,
      predicted_duration: data.predicted_duration,
      geometry: data.geometry,
      ml_enhanced: data.ml_enhanced || true,
      traffic_prediction: data.traffic_prediction,
      alternative_routes: data.alternative_routes || []
    };
    
  } catch (error) {
    console.error('Hybrid routing error:', error);
    throw handleApiError(error, null, 'ML-Enhanced Routing');
  }
};

/**
 * Get POI Recommendations
 * Can be extracted from navigation metadata
 */
export const getPOIRecommendations = (navigationResponse) => {
  if (!navigationResponse || !navigationResponse.metadata) {
    return [];
  }

  return navigationResponse.metadata.poi_recommendations || 
         navigationResponse.metadata.nearby_pois || 
         [];
};

/**
 * Get Turn-by-Turn Directions
 * Can be extracted from route data
 */
export const getTurnByTurnDirections = (routeData) => {
  if (!routeData) return [];

  // Extract from segments
  if (routeData.segments && Array.isArray(routeData.segments)) {
    return routeData.segments.map((segment, index) => ({
      step: index + 1,
      instruction: segment.instruction || segment.name || `Segment ${index + 1}`,
      distance: segment.distance,
      duration: segment.duration,
      maneuver: segment.maneuver
    }));
  }

  // Extract from steps
  if (routeData.steps && Array.isArray(routeData.steps)) {
    return routeData.steps.map((step, index) => ({
      step: index + 1,
      instruction: step.instruction || step.name || `Step ${index + 1}`,
      distance: step.distance,
      duration: step.duration,
      maneuver: step.maneuver
    }));
  }

  return [];
};

/**
 * Parse route geometry for map rendering
 */
export const parseRouteGeometry = (routeData) => {
  if (!routeData) return null;

  // Check various possible geometry fields
  const geometry = routeData.geometry || 
                   routeData.route_geometry || 
                   routeData.geojson;

  if (!geometry) return null;

  // If it's already a GeoJSON LineString
  if (geometry.type === 'LineString') {
    return geometry;
  }

  // If it's a full GeoJSON Feature
  if (geometry.type === 'Feature' && geometry.geometry) {
    return geometry.geometry;
  }

  // If it's a string (encoded polyline)
  if (typeof geometry === 'string') {
    return {
      type: 'LineString',
      coordinates: [], // Would need polyline decoder
      encoded: geometry
    };
  }

  return null;
};

/**
 * Format distance for display
 */
export const formatDistance = (meters) => {
  if (!meters) return 'N/A';
  
  if (meters < 1000) {
    return `${Math.round(meters)} m`;
  }
  
  return `${(meters / 1000).toFixed(2)} km`;
};

/**
 * Format duration for display
 */
export const formatDuration = (seconds) => {
  if (!seconds) return 'N/A';
  
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  
  if (hours > 0) {
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  }
  
  return `${minutes} min`;
};

export default {
  chatNavigation,
  planGPSRoute,
  optimizeRoute,
  planHybridRoute,
  getPOIRecommendations,
  getTurnByTurnDirections,
  parseRouteGeometry,
  formatDistance,
  formatDuration
};
