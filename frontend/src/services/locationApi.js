/**
 * Location and Routing API Service
 * Connects React frontend to FastAPI backend location endpoints
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

class LocationApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
    this.sessionId = null;
  }

  /**
   * Generic API request handler
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  /**
   * Create a new location session
   */
  async createSession(userLocation, preferences = {}) {
    try {
      const response = await this.request('/location/session', {
        method: 'POST',
        body: JSON.stringify({
          user_id: `user_${Date.now()}`,
          latitude: userLocation.lat,
          longitude: userLocation.lng,
          accuracy: userLocation.accuracy || null,
          preferences: preferences
        }),
      });

      if (response.session_id) {
        this.sessionId = response.session_id;
        localStorage.setItem('ai_istanbul_session_id', this.sessionId);
      }

      return response;
    } catch (error) {
      console.error('Failed to create location session:', error);
      throw error;
    }
  }

  /**
   * Get current session details
   */
  async getSession() {
    if (!this.sessionId) {
      this.sessionId = localStorage.getItem('ai_istanbul_session_id');
    }

    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      return await this.request(`/location/session/${this.sessionId}`);
    } catch (error) {
      console.error('Failed to get session:', error);
      throw error;
    }
  }

  /**
   * Update user location
   */
  async updateLocation(newLocation) {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      return await this.request('/location/update', {
        method: 'POST',
        body: JSON.stringify({
          session_id: this.sessionId,
          latitude: newLocation.lat,
          longitude: newLocation.lng,
          accuracy: newLocation.accuracy || null,
          timestamp: new Date().toISOString()
        }),
      });
    } catch (error) {
      console.error('Failed to update location:', error);
      throw error;
    }
  }

  /**
   * Get POI recommendations
   */
  async getRecommendations(options = {}) {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      return await this.request('/location/recommendations', {
        method: 'POST',
        body: JSON.stringify({
          session_id: this.sessionId,
          categories: options.categories || [],
          radius_km: options.radius || 2.0,
          filters: options.filters || {},
          limit: options.limit || 20
        }),
      });
    } catch (error) {
      console.error('Failed to get recommendations:', error);
      throw error;
    }
  }

  /**
   * Plan optimal route
   */
  async planRoute(targetPois, options = {}) {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      return await this.request('/location/route/plan', {
        method: 'POST',
        body: JSON.stringify({
          session_id: this.sessionId,
          target_pois: targetPois,
          algorithm: options.algorithm || 'tsp_nearest',
          transport_mode: options.transport || 'walking',
          optimize_for: options.optimizeFor || 'time'
        }),
      });
    } catch (error) {
      console.error('Failed to plan route:', error);
      throw error;
    }
  }

  /**
   * Plan multi-stop route
   */
  async planMultiStopRoute(startLocation, stops, preferences = {}) {
    try {
      return await this.request('/location/route/multi-stop', {
        method: 'POST',
        body: JSON.stringify({
          start_location: startLocation,
          stops: stops,
          preferences: preferences
        }),
      });
    } catch (error) {
      console.error('Failed to plan multi-stop route:', error);
      throw error;
    }
  }

  /**
   * Update existing route with real-time conditions
   */
  async updateRoute(routeData, currentLocation, conditions = {}) {
    try {
      return await this.request('/location/route/update-route', {
        method: 'POST',
        body: JSON.stringify({
          route: routeData,
          current_location: currentLocation,
          conditions: conditions
        }),
      });
    } catch (error) {
      console.error('Failed to update route:', error);
      throw error;
    }
  }

  /**
   * Search nearby POIs
   */
  async searchNearby(location, options = {}) {
    try {
      return await this.request('/location/search/nearby', {
        method: 'POST',
        body: JSON.stringify({
          location: location,
          radius_km: options.radius || 1.0,
          category: options.category || null,
          filters: options.filters || {}
        }),
      });
    } catch (error) {
      console.error('Failed to search nearby:', error);
      throw error;
    }
  }

  /**
   * Get district information
   */
  async getDistricts() {
    try {
      return await this.request('/location/districts');
    } catch (error) {
      console.error('Failed to get districts:', error);
      throw error;
    }
  }

  /**
   * Get offline data for location
   */
  async getOfflineData(location, radiusKm = 2.0) {
    try {
      return await this.request('/location/offline', {
        method: 'POST',
        body: JSON.stringify({
          location: location,
          radius_km: radiusKm
        }),
      });
    } catch (error) {
      console.error('Failed to get offline data:', error);
      throw error;
    }
  }

  /**
   * Get all available POIs
   */
  async getAvailablePOIs() {
    try {
      return await this.request('/location/available-pois');
    } catch (error) {
      console.error('Failed to get available POIs:', error);
      throw error;
    }
  }

  /**
   * Cleanup session
   */
  async cleanupSession() {
    if (!this.sessionId) {
      return;
    }

    try {
      await this.request(`/location/cleanup/${this.sessionId}`, {
        method: 'DELETE',
      });
      
      this.sessionId = null;
      localStorage.removeItem('ai_istanbul_session_id');
    } catch (error) {
      console.error('Failed to cleanup session:', error);
      // Don't throw - cleanup should be non-blocking
    }
  }

  /**
   * Get user's current location using browser geolocation
   */
  async getCurrentLocation() {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation is not supported by this browser'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
            accuracy: position.coords.accuracy
          });
        },
        (error) => {
          let errorMessage;
          switch (error.code) {
            case error.PERMISSION_DENIED:
              errorMessage = 'Location access denied by user';
              break;
            case error.POSITION_UNAVAILABLE:
              errorMessage = 'Location information unavailable';
              break;
            case error.TIMEOUT:
              errorMessage = 'Location request timed out';
              break;
            default:
              errorMessage = 'An unknown error occurred while retrieving location';
              break;
          }
          reject(new Error(errorMessage));
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000 // 5 minutes
        }
      );
    });
  }

  /**
   * Watch user location changes
   */
  watchLocation(callback, errorCallback) {
    if (!navigator.geolocation) {
      errorCallback?.(new Error('Geolocation is not supported'));
      return null;
    }

    return navigator.geolocation.watchPosition(
      (position) => {
        callback({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
          accuracy: position.coords.accuracy,
          timestamp: new Date(position.timestamp)
        });
      },
      (error) => {
        errorCallback?.(error);
      },
      {
        enableHighAccuracy: true,
        timeout: 30000,
        maximumAge: 60000 // 1 minute
      }
    );
  }

  /**
   * Clear location watch
   */
  clearLocationWatch(watchId) {
    if (watchId && navigator.geolocation) {
      navigator.geolocation.clearWatch(watchId);
    }
  }
}

// Create and export a singleton instance
const locationApi = new LocationApiService();
export default locationApi;

// Also export the class for testing purposes
export { LocationApiService };
