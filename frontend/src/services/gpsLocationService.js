/**
 * GPS Location Service
 * Handles real-time GPS location access, permissions, and location tracking
 * Integrates with Istanbul Daily Talk AI System
 */

class GPSLocationService {
  constructor() {
    this.watchId = null;
    this.currentPosition = null;
    this.lastKnownPosition = null;
    this.locationCallbacks = new Set();
    this.errorCallbacks = new Set();
    this.options = {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 60000 // Cache location for 1 minute
    };
    
    // Load last known position from localStorage
    this.loadLastKnownPosition();
  }

  /**
   * Request location permission and get current position
   */
  async requestLocationPermission() {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation is not supported by this browser'));
        return;
      }

      try {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            this.currentPosition = {
              lat: position.coords.latitude,
              lng: position.coords.longitude,
              accuracy: position.coords.accuracy,
              timestamp: new Date()
            };
            
            this.lastKnownPosition = { ...this.currentPosition };
            this.saveLastKnownPosition();
            this.notifyLocationCallbacks(this.currentPosition);
            
            resolve(this.currentPosition);
          },
          (error) => {
            this.notifyErrorCallbacks(error);
            reject(this.handleGeolocationError(error));
          },
          this.options
        );
      } catch (error) {
        // Catch permissions policy errors that are thrown synchronously
        if (error.message && error.message.includes('permissions policy')) {
          reject(new Error('Geolocation has been disabled by permissions policy. Please enable location services or use manual location entry.'));
        } else {
          reject(error);
        }
      }
    });
  }

  /**
   * Start continuous location tracking
   */
  startLocationTracking() {
    if (!navigator.geolocation) {
      throw new Error('Geolocation is not supported by this browser');
    }

    if (this.watchId) {
      this.stopLocationTracking();
    }

    this.watchId = navigator.geolocation.watchPosition(
      (position) => {
        const newPosition = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
          accuracy: position.coords.accuracy,
          timestamp: new Date()
        };

        // Only update if position changed significantly (>10 meters)
        if (this.hasPositionChanged(newPosition, this.currentPosition)) {
          this.currentPosition = newPosition;
          this.lastKnownPosition = { ...newPosition };
          this.saveLastKnownPosition();
          this.notifyLocationCallbacks(newPosition);
        }
      },
      (error) => {
        this.notifyErrorCallbacks(error);
      },
      this.options
    );

    return this.watchId;
  }

  /**
   * Stop location tracking
   */
  stopLocationTracking() {
    if (this.watchId) {
      navigator.geolocation.clearWatch(this.watchId);
      this.watchId = null;
    }
  }

  /**
   * Get current position (from cache or request new)
   */
  async getCurrentPosition() {
    // Return cached position if recent enough
    if (this.currentPosition && this.isPositionRecent(this.currentPosition)) {
      return this.currentPosition;
    }

    // Otherwise request new position
    return this.requestLocationPermission();
  }

  /**
   * Get last known position (from cache/localStorage)
   */
  getLastKnownPosition() {
    return this.lastKnownPosition;
  }

  /**
   * Check if location services are available
   */
  isLocationAvailable() {
    return 'geolocation' in navigator;
  }

  /**
   * Check location permission status
   */
  async getLocationPermissionStatus() {
    if (!navigator.permissions) {
      return 'unknown';
    }

    try {
      const result = await navigator.permissions.query({name: 'geolocation'});
      return result.state; // 'granted', 'denied', or 'prompt'
    } catch (error) {
      return 'unknown';
    }
  }

  /**
   * Calculate distance between two GPS coordinates (in meters)
   */
  calculateDistance(pos1, pos2) {
    const R = 6371e3; // Earth's radius in meters
    const φ1 = pos1.lat * Math.PI/180;
    const φ2 = pos2.lat * Math.PI/180;
    const Δφ = (pos2.lat - pos1.lat) * Math.PI/180;
    const Δλ = (pos2.lng - pos1.lng) * Math.PI/180;

    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

    return R * c; // Distance in meters
  }

  /**
   * Get neighborhood from GPS coordinates
   */
  async getNeighborhoodFromCoordinates(position) {
    // Istanbul neighborhood boundaries (approximate)
    const neighborhoods = {
      'Sultanahmet': {
        bounds: { 
          north: 41.0150, south: 41.0050, 
          east: 28.9850, west: 28.9750 
        }
      },
      'Beyoğlu': {
        bounds: { 
          north: 41.0400, south: 41.0250, 
          east: 28.9850, west: 28.9700 
        }
      },
      'Kadıköy': {
        bounds: { 
          north: 40.9950, south: 40.9800, 
          east: 29.0350, west: 29.0150 
        }
      },
      'Beşiktaş': {
        bounds: { 
          north: 41.0500, south: 41.0350, 
          east: 29.0150, west: 28.9900 
        }
      },
      'Fatih': {
        bounds: { 
          north: 41.0250, south: 41.0050, 
          east: 28.9750, west: 28.9500 
        }
      },
      'Üsküdar': {
        bounds: { 
          north: 41.0350, south: 41.0150, 
          east: 29.0350, west: 29.0050 
        }
      }
    };

    for (const [name, data] of Object.entries(neighborhoods)) {
      const { bounds } = data;
      if (position.lat >= bounds.south && position.lat <= bounds.north &&
          position.lng >= bounds.west && position.lng <= bounds.east) {
        return name;
      }
    }

    return 'Unknown Area';
  }

  /**
   * Subscribe to location updates
   */
  onLocationUpdate(callback) {
    this.locationCallbacks.add(callback);
    return () => this.locationCallbacks.delete(callback);
  }

  /**
   * Subscribe to location errors
   */
  onLocationError(callback) {
    this.errorCallbacks.add(callback);
    return () => this.errorCallbacks.delete(callback);
  }

  /**
   * Private helper methods
   */
  hasPositionChanged(newPos, oldPos) {
    if (!oldPos) return true;
    const distance = this.calculateDistance(newPos, oldPos);
    return distance > 10; // 10 meters threshold
  }

  isPositionRecent(position) {
    if (!position?.timestamp) return false;
    const ageMs = Date.now() - position.timestamp.getTime();
    return ageMs < this.options.maximumAge;
  }

  notifyLocationCallbacks(position) {
    this.locationCallbacks.forEach(callback => {
      try {
        callback(position);
      } catch (error) {
        console.error('Error in location callback:', error);
      }
    });
  }

  notifyErrorCallbacks(error) {
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (error) {
        console.error('Error in error callback:', error);
      }
    });
  }

  handleGeolocationError(error) {
    const errorMessages = {
      1: 'Location access denied by user',
      2: 'Location information unavailable',
      3: 'Location request timeout'
    };

    // Check for permissions policy error
    if (error.message && error.message.includes('permissions policy')) {
      return new Error('Geolocation has been disabled by permissions policy. Please enable location services or use manual location entry.');
    }

    return new Error(errorMessages[error.code] || 'Unknown location error');
  }

  saveLastKnownPosition() {
    if (this.lastKnownPosition) {
      localStorage.setItem('lastKnownPosition', JSON.stringify({
        ...this.lastKnownPosition,
        timestamp: this.lastKnownPosition.timestamp.toISOString()
      }));
    }
  }

  loadLastKnownPosition() {
    try {
      const saved = localStorage.getItem('lastKnownPosition');
      if (saved) {
        const parsed = JSON.parse(saved);
        parsed.timestamp = new Date(parsed.timestamp);
        
        // Only use if less than 1 hour old
        if (Date.now() - parsed.timestamp.getTime() < 3600000) {
          this.lastKnownPosition = parsed;
        }
      }
    } catch (error) {
      console.error('Error loading last known position:', error);
    }
  }

  /**
   * Format position for AI system
   */
  formatPositionForAI(position) {
    if (!position) return null;
    
    return {
      lat: position.lat,
      lng: position.lng,
      accuracy: position.accuracy,
      timestamp: position.timestamp?.toISOString(),
      neighborhood: null // Will be filled by getNeighborhoodFromCoordinates
    };
  }

  /**
   * Clean up resources
   */
  destroy() {
    this.stopLocationTracking();
    this.locationCallbacks.clear();
    this.errorCallbacks.clear();
  }
}

// Create singleton instance
const gpsLocationService = new GPSLocationService();

export default gpsLocationService;
