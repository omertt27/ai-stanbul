/**
 * Route API Service
 * Phase 3: API client for route generation and optimization
 */

import { fetchWithRetry, handleApiError } from '../utils/errorHandler';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const ROUTES_API_URL = `${BASE_URL}/api/routes`;

// Generate a new route
export const generateRoute = async (routeRequest) => {
  try {
    console.log('🚀 Generating route with request:', routeRequest);
    
    const response = await fetchWithRetry(`${ROUTES_API_URL}/generate`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(routeRequest),
      timeout: 30000 // 30 seconds for route generation
    }, {
      maxAttempts: 2,
      baseDelay: 1000
    });
    
    const route = await response.json();
    console.log('✅ Route generated:', route);
    return route;
    
  } catch (error) {
    throw handleApiError(error, null, 'Route Generation');
  }
};

// Fetch attractions near a location
export const fetchAttractions = async ({ lat, lng, radius_km = 5, categories = [], limit = 20 }) => {
  try {
    console.log('🔍 Fetching attractions near:', { lat, lng, radius_km, categories });
    
    const params = new URLSearchParams({
      lat: lat.toString(),
      lng: lng.toString(),
      radius_km: radius_km.toString(),
      limit: limit.toString()
    });
    
    if (categories.length > 0) {
      categories.forEach(cat => params.append('categories', cat));
    }
    
    const response = await fetchWithRetry(`${ROUTES_API_URL}/attractions/nearby?${params}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      timeout: 15000
    }, {
      maxAttempts: 2,
      baseDelay: 500
    });
    
    const data = await response.json();
    console.log('✅ Attractions fetched:', data.attractions?.length || 0);
    return data.attractions || [];
    
  } catch (error) {
    throw handleApiError(error, null, 'Attraction Search');
  }
};

// Analyze TSP optimization for selected attractions
export const analyzeTSP = async (analysisRequest) => {
  try {
    console.log('🧮 Running TSP analysis:', analysisRequest);
    
    const response = await fetchWithRetry(`${ROUTES_API_URL}/tsp/analyze`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(analysisRequest),
      timeout: 20000
    }, {
      maxAttempts: 2,
      baseDelay: 1000
    });
    
    const analysis = await response.json();
    console.log('✅ TSP analysis completed:', analysis);
    return analysis;
    
  } catch (error) {
    console.warn('TSP analysis failed:', error.message);
    throw handleApiError(error, null, 'TSP Analysis');
  }
};

// Get route maker service information
export const getRouteInfo = async () => {
  try {
    const response = await fetchWithRetry(ROUTES_API_URL, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      timeout: 10000
    });
    
    const info = await response.json();
    console.log('ℹ️ Route maker info:', info);
    return info;
    
  } catch (error) {
    throw handleApiError(error, null, 'Route Info');
  }
};

// Get available attraction categories
export const getAttractionCategories = async () => {
  try {
    const response = await fetchWithRetry(`${ROUTES_API_URL}/attractions/categories`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      timeout: 10000
    });
    
    const data = await response.json();
    return data.categories || [];
    
  } catch (error) {
    throw handleApiError(error, null, 'Categories Fetch');
  }
};

// Get district status and coverage
export const getDistrictStatus = async () => {
  try {
    const response = await fetchWithRetry(`${ROUTES_API_URL}/districts/status`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      timeout: 10000
    });
    
    const status = await response.json();
    console.log('🗺️ District status:', status);
    return status;
    
  } catch (error) {
    throw handleApiError(error, null, 'District Status');
  }
};

// Switch to a different district for routing
export const switchDistrict = async (districtName) => {
  try {
    console.log('🔄 Switching to district:', districtName);
    
    const response = await fetchWithRetry(`${ROUTES_API_URL}/districts/switch?district_name=${encodeURIComponent(districtName)}`, {
      method: 'POST',
      headers: { 'Accept': 'application/json' },
      timeout: 15000
    });
    
    const result = await response.json();
    console.log('✅ District switched:', result);
    return result;
    
  } catch (error) {
    throw handleApiError(error, null, 'District Switch');
  }
};

// Save a generated route
export const saveRoute = async (routeData) => {
  try {
    console.log('💾 Saving route:', routeData.name);
    
    const response = await fetchWithRetry(`${ROUTES_API_URL}/save`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(routeData),
      timeout: 15000
    });
    
    const savedRoute = await response.json();
    console.log('✅ Route saved with ID:', savedRoute.id);
    return savedRoute;
    
  } catch (error) {
    throw handleApiError(error, null, 'Route Save');
  }
};

// Load a saved route
export const loadRoute = async (routeId) => {
  try {
    console.log('📂 Loading route:', routeId);
    
    const response = await fetchWithRetry(`${ROUTES_API_URL}/${routeId}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      timeout: 10000
    });
    
    const route = await response.json();
    console.log('✅ Route loaded:', route.name);
    return route;
    
  } catch (error) {
    throw handleApiError(error, null, 'Route Load');
  }
};

// Get user's saved routes
export const getUserRoutes = async (limit = 20) => {
  try {
    const response = await fetchWithRetry(`${ROUTES_API_URL}/user?limit=${limit}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      timeout: 10000
    });
    
    const data = await response.json();
    return data.routes || [];
    
  } catch (error) {
    throw handleApiError(error, null, 'User Routes');
  }
};

// Delete a saved route
export const deleteRoute = async (routeId) => {
  try {
    console.log('🗑️ Deleting route:', routeId);
    
    const response = await fetchWithRetry(`${ROUTES_API_URL}/${routeId}`, {
      method: 'DELETE',
      headers: { 'Accept': 'application/json' },
      timeout: 10000
    });
    
    const result = await response.json();
    console.log('✅ Route deleted');
    return result;
    
  } catch (error) {
    throw handleApiError(error, null, 'Route Delete');
  }
};

// Get route analytics and performance stats
export const getRouteAnalytics = async (routeId) => {
  try {
    const response = await fetchWithRetry(`${ROUTES_API_URL}/${routeId}/analytics`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      timeout: 10000
    });
    
    const analytics = await response.json();
    return analytics;
    
  } catch (error) {
    throw handleApiError(error, null, 'Route Analytics');
  }
};

// Export route to different formats
export const exportRoute = async (routeId, format = 'json') => {
  try {
    console.log('📤 Exporting route:', routeId, 'as', format);
    
    const response = await fetchWithRetry(`${ROUTES_API_URL}/${routeId}/export?format=${format}`, {
      method: 'GET',
      headers: { 'Accept': format === 'json' ? 'application/json' : 'text/plain' },
      timeout: 15000
    });
    
    if (format === 'json') {
      return await response.json();
    } else {
      return await response.text();
    }
    
  } catch (error) {
    throw handleApiError(error, null, 'Route Export');
  }
};

// Cache management
export const getCacheStats = async () => {
  try {
    const response = await fetchWithRetry(`${ROUTES_API_URL}/cache/stats`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      timeout: 5000
    });
    
    const stats = await response.json();
    console.log('📊 Cache stats:', stats);
    return stats;
    
  } catch (error) {
    console.warn('Cache stats not available:', error.message);
    return null;
  }
};

export const clearCache = async () => {
  try {
    const response = await fetchWithRetry(`${ROUTES_API_URL}/cache/clear`, {
      method: 'POST',
      headers: { 'Accept': 'application/json' },
      timeout: 10000
    });
    
    const result = await response.json();
    console.log('🗑️ Cache cleared');
    return result;
    
  } catch (error) {
    throw handleApiError(error, null, 'Cache Clear');
  }
};

// Real-time route tracking (for future implementation)
export const trackRoute = async (routeId, currentLocation) => {
  try {
    const response = await fetchWithRetry(`${ROUTES_API_URL}/${routeId}/track`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        lat: currentLocation.lat,
        lng: currentLocation.lng,
        timestamp: new Date().toISOString()
      }),
      timeout: 5000
    });
    
    const tracking = await response.json();
    return tracking;
    
  } catch (error) {
    console.warn('Route tracking not available:', error.message);
    return null;
  }
};

export default {
  generateRoute,
  fetchAttractions,
  analyzeTSP,
  getRouteInfo,
  getAttractionCategories,
  getDistrictStatus,
  switchDistrict,
  saveRoute,
  loadRoute,
  getUserRoutes,
  deleteRoute,
  getRouteAnalytics,
  exportRoute,
  getCacheStats,
  clearCache,
  trackRoute
};
