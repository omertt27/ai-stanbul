/**
 * GPS Location Service
 * Handles real-time GPS location access, permissions, and location tracking
 * Integrates with Istanbul Daily Talk AI System
 */

import logger from '../utils/logger.js';
const log = logger.namespace('GPS');

class GPSLocationService {
  constructor() {
    this.watchId = null;
    this.currentPosition = null;
    this.lastKnownPosition = null;
    this.locationCallbacks = new Set();
    this.errorCallbacks = new Set();
    
    // Maximum accuracy options - same as Google Maps!
    this.options = {
      enableHighAccuracy: true,    // Use GPS chip for best accuracy (5-10 meters)
      timeout: 15000,               // Wait up to 15 seconds for accurate position
      maximumAge: 0                 // Don't use cached position, always get fresh GPS data
    };
    
    // Continuous tracking options (for watchPosition)
    this.watchOptions = {
      enableHighAccuracy: true,    // Always use GPS
      timeout: 10000,              // Faster timeout for continuous updates
      maximumAge: 5000             // Allow 5-second cache for smooth tracking
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

    log.debug('🎯 Starting high-accuracy GPS tracking (Google Maps quality)...');

    this.watchId = navigator.geolocation.watchPosition(
      (position) => {
        const newPosition = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
          accuracy: position.coords.accuracy,
          altitude: position.coords.altitude,
          heading: position.coords.heading,
          speed: position.coords.speed,
          timestamp: new Date()
        };

        log.debug(`📍 GPS Update: ${newPosition.lat.toFixed(6)}, ${newPosition.lng.toFixed(6)} (±${newPosition.accuracy.toFixed(1)}m)`);

        // Update on every position change for maximum accuracy
        // Only skip if position is identical (within 1 meter)
        if (this.hasPositionChanged(newPosition, this.currentPosition, 1)) {
          this.currentPosition = newPosition;
          this.lastKnownPosition = { ...newPosition };
          this.saveLastKnownPosition();
          this.notifyLocationCallbacks(newPosition);
        }
      },
      (error) => {
        log.error('❌ GPS Error:', error.message);
        this.notifyErrorCallbacks(error);
      },
      this.watchOptions  // Use continuous tracking options
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
      },
      'Taksim': {
        bounds: { 
          north: 41.0400, south: 41.0320, 
          east: 28.9900, west: 28.9800 
        }
      },
      'Ortaköy': {
        bounds: { 
          north: 41.0550, south: 41.0450, 
          east: 29.0300, west: 29.0150 
        }
      },
      'Karaköy': {
        bounds: { 
          north: 41.0280, south: 41.0200, 
          east: 28.9800, west: 28.9700 
        }
      },
      'Eminönü': {
        bounds: { 
          north: 41.0200, south: 41.0100, 
          east: 28.9750, west: 28.9650 
        }
      }
    };

    // Check if location is in a known neighborhood
    for (const [name, data] of Object.entries(neighborhoods)) {
      const { bounds } = data;
      if (position.lat >= bounds.south && position.lat <= bounds.north &&
          position.lng >= bounds.west && position.lng <= bounds.east) {
        return name;
      }
    }

    // Fallback: Show friendly location description instead of "Unknown Area"
    // Check if in Istanbul region (approximately)
    const istanbulBounds = {
      north: 41.3,
      south: 40.8,
      east: 29.5,
      west: 28.5
    };
    
    const isInIstanbul = position.lat >= istanbulBounds.south && 
                        position.lat <= istanbulBounds.north &&
                        position.lng >= istanbulBounds.west && 
                        position.lng <= istanbulBounds.east;
    
    if (isInIstanbul) {
      // In Istanbul but not in a mapped neighborhood
      return `Istanbul (${position.lat.toFixed(4)}°, ${position.lng.toFixed(4)}°)`;
    } else {
      // Outside Istanbul
      return `📍 ${position.lat.toFixed(4)}°N, ${position.lng.toFixed(4)}°E`;
    }
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
  hasPositionChanged(newPos, oldPos, thresholdMeters = 10) {
    if (!oldPos) return true;
    const distance = this.calculateDistance(newPos, oldPos);
    return distance > thresholdMeters;
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
        log.error('Error in location callback:', error);
      }
    });
  }

  notifyErrorCallbacks(error) {
    this.errorCallbacks.forEach(callback => {
      try {
        callback(error);
      } catch (error) {
        log.error('Error in error callback:', error);
      }
    });
  }

  handleGeolocationError(error) {
    log.error('🔴 GPS Error Details:', {
      code: error.code,
      message: error.message,
      PERMISSION_DENIED: error.PERMISSION_DENIED,
      POSITION_UNAVAILABLE: error.POSITION_UNAVAILABLE,
      TIMEOUT: error.TIMEOUT
    });

    // Check for permissions policy error first
    if (error.message && error.message.includes('permissions policy')) {
      return new Error('Location disabled in browser settings. Please enable location access in your browser or device settings.');
    }

    // Detailed error messages with troubleshooting tips
    const errorDetails = {
      1: { // PERMISSION_DENIED
        title: 'Location Access Denied',
        message: 'Please allow location access in your browser settings to use GPS features.',
        tips: [
          'Check browser location permissions',
          'Enable location services on your device',
          'Reload the page after granting permission'
        ]
      },
      2: { // POSITION_UNAVAILABLE
        title: 'GPS Location Unavailable',
        message: 'Unable to determine your location. This may be due to GPS signal issues.',
        tips: [
          'Make sure you\'re not in a GPS-blocked area (underground, indoors with thick walls)',
          'Check if location services are enabled on your device',
          'Try moving to an area with better signal',
          'Restart your browser or device',
          'On iOS: Settings → Privacy → Location Services → Safari → While Using',
          'On Android: Settings → Location → App permissions → Browser → Allow'
        ]
      },
      3: { // TIMEOUT
        title: 'GPS Timeout',
        message: 'Location request took too long. Your GPS signal may be weak.',
        tips: [
          'Move to an area with better GPS signal',
          'Try again in a few moments',
          'Check if location services are enabled'
        ]
      }
    };

    const details = errorDetails[error.code] || {
      title: 'Location Error',
      message: `GPS error: ${error.message}`,
      tips: ['Please try again or use manual location entry']
    };

    // Create a more informative error object
    const enhancedError = new Error(details.message);
    enhancedError.title = details.title;
    enhancedError.tips = details.tips;
    enhancedError.code = error.code;
    enhancedError.originalMessage = error.message;

    return enhancedError;
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
      log.error('Error loading last known position:', error);
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
