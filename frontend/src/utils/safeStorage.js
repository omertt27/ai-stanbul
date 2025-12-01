/**
 * Safe localStorage wrapper with error handling and validation
 * Prevents app crashes from corrupted localStorage data
 */

const STORAGE_VERSION = '1.0';
const STORAGE_VERSION_KEY = 'ai_istanbul_storage_version';

class SafeStorage {
  constructor() {
    this.checkAndMigrateStorage();
  }

  /**
   * Check storage version and clear if outdated
   */
  checkAndMigrateStorage() {
    try {
      const currentVersion = localStorage.getItem(STORAGE_VERSION_KEY);
      if (currentVersion !== STORAGE_VERSION) {
        console.warn('⚠️ Storage version mismatch, clearing old data');
        this.clearInvalidData();
        localStorage.setItem(STORAGE_VERSION_KEY, STORAGE_VERSION);
      }
    } catch (error) {
      console.error('Error checking storage version:', error);
      this.clearInvalidData();
    }
  }

  /**
   * Clear invalid or corrupted data from localStorage
   */
  clearInvalidData() {
    const keysToValidate = [
      'chat-messages',
      'ai-istanbul-cookie-consent',
      'ai-stanbul-feedbacks',
      'saved_routes',
      'chat_sessions',
      'userInteractions'
    ];

    keysToValidate.forEach(key => {
      try {
        const value = localStorage.getItem(key);
        if (value) {
          // Try to parse JSON to validate
          JSON.parse(value);
        }
      } catch (error) {
        console.warn(`Removing corrupted data for key: ${key}`);
        localStorage.removeItem(key);
      }
    });
  }

  /**
   * Safely get item from localStorage with JSON parsing
   * @param {string} key - Storage key
   * @param {any} defaultValue - Default value if key doesn't exist or is invalid
   * @returns {any} Parsed value or default value
   */
  getItem(key, defaultValue = null) {
    try {
      const item = localStorage.getItem(key);
      if (item === null) {
        return defaultValue;
      }
      return item;
    } catch (error) {
      console.error(`Error getting item ${key}:`, error);
      return defaultValue;
    }
  }

  /**
   * Safely get and parse JSON from localStorage
   * @param {string} key - Storage key
   * @param {any} defaultValue - Default value if key doesn't exist or is invalid
   * @returns {any} Parsed JSON or default value
   */
  getJSON(key, defaultValue = null) {
    try {
      const item = localStorage.getItem(key);
      if (item === null) {
        return defaultValue;
      }
      return JSON.parse(item);
    } catch (error) {
      console.error(`Error parsing JSON for ${key}:`, error);
      // Remove corrupted data
      localStorage.removeItem(key);
      return defaultValue;
    }
  }

  /**
   * Safely set item in localStorage
   * @param {string} key - Storage key
   * @param {any} value - Value to store
   * @returns {boolean} Success status
   */
  setItem(key, value) {
    try {
      localStorage.setItem(key, value);
      return true;
    } catch (error) {
      console.error(`Error setting item ${key}:`, error);
      // Handle quota exceeded error
      if (error.name === 'QuotaExceededError') {
        console.warn('localStorage quota exceeded, attempting cleanup');
        this.cleanup();
        try {
          localStorage.setItem(key, value);
          return true;
        } catch (retryError) {
          console.error('Failed to set item after cleanup:', retryError);
          return false;
        }
      }
      return false;
    }
  }

  /**
   * Safely set JSON in localStorage
   * @param {string} key - Storage key
   * @param {any} value - Value to stringify and store
   * @returns {boolean} Success status
   */
  setJSON(key, value) {
    try {
      const jsonString = JSON.stringify(value);
      return this.setItem(key, jsonString);
    } catch (error) {
      console.error(`Error stringifying JSON for ${key}:`, error);
      return false;
    }
  }

  /**
   * Remove item from localStorage
   * @param {string} key - Storage key
   */
  removeItem(key) {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error(`Error removing item ${key}:`, error);
    }
  }

  /**
   * Clean up old or large items from localStorage
   */
  cleanup() {
    try {
      // Remove old session data
      const now = Date.now();
      const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 days

      Object.keys(localStorage).forEach(key => {
        if (key.startsWith('chat-messages-')) {
          // Remove old session messages
          const timestamp = key.split('-').pop();
          if (now - parseInt(timestamp) > maxAge) {
            localStorage.removeItem(key);
          }
        }
      });
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  }

  /**
   * Clear all data from localStorage
   * Used by ErrorBoundary for recovery
   */
  clearAll() {
    try {
      console.warn('⚠️ Clearing all localStorage data');
      localStorage.clear();
      // Re-initialize storage version
      localStorage.setItem(STORAGE_VERSION_KEY, STORAGE_VERSION);
      console.log('✅ localStorage cleared successfully');
      return true;
    } catch (error) {
      console.error('Error clearing localStorage:', error);
      return false;
    }
  }

  /**
   * Get storage usage information
   * @returns {object} Storage stats
   */
  getStorageInfo() {
    try {
      let totalSize = 0;
      const items = {};

      Object.keys(localStorage).forEach(key => {
        const value = localStorage.getItem(key);
        const size = value ? value.length : 0;
        totalSize += size;
        items[key] = size;
      });

      return {
        totalSize,
        totalItems: Object.keys(localStorage).length,
        items,
        available: true
      };
    } catch (error) {
      console.error('Error getting storage info:', error);
      return {
        totalSize: 0,
        totalItems: 0,
        items: {},
        available: false
      };
    }
  }
}

// Create singleton instance
const safeStorage = new SafeStorage();

export default safeStorage;
