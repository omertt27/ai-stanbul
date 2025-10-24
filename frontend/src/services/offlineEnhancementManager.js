/**
 * Offline Enhancement Manager
 * Integrates all offline enhancement features
 * 
 * @version 1.0.0
 */

import offlineMapTileCache from './offlineMapTileCache.js';
import offlineIntentDetector from './offlineIntentDetector.js';
import offlineDatabase from './offlineDatabase.js';

class OfflineEnhancementManager {
  constructor() {
    this.isOnline = navigator.onLine;
    this.isInitialized = false;
    this.config = {
      autoSyncOnReconnect: true,
      cacheMapTilesOnInstall: false, // User opt-in for large download
      enablePeriodicSync: true,
      enableOfflineIntents: true
    };
  }

  /**
   * Initialize all offline enhancements
   * @param {Object} config - Configuration options
   * @returns {Promise<object>} Initialization status
   */
  async initialize(config = {}) {
    if (this.isInitialized) {
      console.log('⚠️ Already initialized');
      return { status: 'already_initialized' };
    }

    this.config = { ...this.config, ...config };
    console.log('🚀 Initializing offline enhancements...');

    try {
      // 1. Setup online/offline listeners
      this.setupNetworkListeners();

      // 2. Initialize IndexedDB
      await offlineDatabase.init();
      console.log('✅ IndexedDB initialized');

      // 3. Register enhanced service worker
      if ('serviceWorker' in navigator) {
        await this.registerServiceWorker();
      }

      // 4. Setup periodic sync (if supported)
      if (this.config.enablePeriodicSync) {
        await this.setupPeriodicSync();
      }

      // 5. Check cache status
      const stats = await this.getCacheStatus();
      console.log('📊 Cache status:', stats);

      // 6. Auto-sync if online
      if (this.isOnline && this.config.autoSyncOnReconnect) {
        this.syncAllData().catch(err => 
          console.warn('Initial sync failed:', err)
        );
      }

      this.isInitialized = true;
      console.log('✅ Offline enhancements initialized');

      return {
        status: 'success',
        features: {
          mapTiles: true,
          offlineIntents: this.config.enableOfflineIntents,
          indexedDB: true,
          periodicSync: this.config.enablePeriodicSync
        },
        stats
      };
    } catch (error) {
      console.error('❌ Initialization failed:', error);
      return {
        status: 'error',
        error: error.message
      };
    }
  }

  /**
   * Setup network status listeners
   */
  setupNetworkListeners() {
    window.addEventListener('online', () => {
      console.log('🌐 Back online');
      this.isOnline = true;
      this.handleReconnect();
    });

    window.addEventListener('offline', () => {
      console.log('📴 Offline mode activated');
      this.isOnline = false;
      this.handleOffline();
    });
  }

  /**
   * Register enhanced service worker
   */
  async registerServiceWorker() {
    try {
      const registration = await navigator.serviceWorker.register('/sw-enhanced.js');
      console.log('✅ Enhanced service worker registered');
      
      // Listen for updates
      registration.addEventListener('updatefound', () => {
        console.log('🔄 Service worker update found');
        const newWorker = registration.installing;
        
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            console.log('✅ New service worker installed, refresh to activate');
            this.notifyUpdate();
          }
        });
      });

      return registration;
    } catch (error) {
      console.error('❌ Service worker registration failed:', error);
      throw error;
    }
  }

  /**
   * Setup periodic background sync
   */
  async setupPeriodicSync() {
    try {
      const registration = await navigator.serviceWorker.ready;
      
      if ('periodicSync' in registration) {
        await registration.periodicSync.register('update-cache', {
          minInterval: 24 * 60 * 60 * 1000 // 24 hours
        });
        console.log('✅ Periodic sync registered');
      } else {
        console.log('⚠️ Periodic sync not supported');
      }
    } catch (error) {
      console.error('❌ Periodic sync setup failed:', error);
    }
  }

  /**
   * Handle reconnection - sync data
   */
  async handleReconnect() {
    if (!this.config.autoSyncOnReconnect) return;

    console.log('🔄 Syncing data after reconnection...');
    
    try {
      await this.syncAllData();
      this.notifyReconnect();
    } catch (error) {
      console.error('❌ Reconnect sync failed:', error);
    }
  }

  /**
   * Handle offline mode activation
   */
  handleOffline() {
    this.notifyOffline();
  }

  /**
   * Sync all offline data
   * @returns {Promise<object>} Sync results
   */
  async syncAllData() {
    console.log('🔄 Syncing all offline data...');
    
    const results = {
      restaurants: null,
      attractions: null,
      pois: null
    };

    try {
      // Sync restaurants
      results.restaurants = await offlineDatabase.syncFromServer(
        '/api/restaurants',
        'restaurants'
      );

      // Sync attractions
      results.attractions = await offlineDatabase.syncFromServer(
        '/api/attractions',
        'attractions'
      );

      // Sync POIs
      results.pois = await offlineDatabase.syncFromServer(
        '/api/pois',
        'pois'
      );

      console.log('✅ Data sync complete:', results);
      return results;
    } catch (error) {
      console.error('❌ Data sync failed:', error);
      throw error;
    }
  }

  /**
   * Cache Istanbul map tiles
   * @param {function} onProgress - Progress callback
   * @returns {Promise<object>} Cache results
   */
  async cacheMapTiles(onProgress = null) {
    console.log('📦 Starting map tile caching...');
    
    try {
      const result = await offlineMapTileCache.cacheIstanbulTiles(onProgress);
      
      if (result.status === 'complete') {
        await offlineDatabase.saveMetadata('map_tiles_cached', true);
        this.notifyMapTilesCached(result);
      }
      
      return result;
    } catch (error) {
      console.error('❌ Map tile caching failed:', error);
      throw error;
    }
  }

  /**
   * Process user query (online or offline)
   * @param {string} query - User query
   * @returns {Promise<object>} Query result
   */
  async processQuery(query) {
    if (!this.config.enableOfflineIntents) {
      return { 
        handled: false, 
        reason: 'offline_intents_disabled' 
      };
    }

    // Detect intent
    const result = offlineIntentDetector.process(query, this.isOnline);

    // If online, let backend handle it
    if (this.isOnline) {
      return {
        handled: false,
        intent: result.intent,
        confidence: result.confidence,
        shouldUseBackend: true
      };
    }

    // If offline and can handle, return offline response
    if (result.canHandleOffline) {
      return {
        handled: true,
        intent: result.intent,
        confidence: result.confidence,
        response: result.response,
        offline: true
      };
    }

    // If offline and cannot handle, queue for later
    return {
      handled: true,
      intent: result.intent,
      queued: true,
      response: result.response,
      offline: true
    };
  }

  /**
   * Search restaurants offline
   * @param {string} query - Search query
   * @param {Object} filters - Optional filters
   * @returns {Promise<Array>} Results
   */
  async searchRestaurants(query, filters = {}) {
    if (query) {
      return await offlineDatabase.search('restaurants', query);
    }
    return await offlineDatabase.filter('restaurants', filters);
  }

  /**
   * Search attractions offline
   * @param {string} query - Search query
   * @param {Object} filters - Optional filters
   * @returns {Promise<Array>} Results
   */
  async searchAttractions(query, filters = {}) {
    if (query) {
      return await offlineDatabase.search('attractions', query);
    }
    return await offlineDatabase.filter('attractions', filters);
  }

  /**
   * Get cache status for all features
   * @returns {Promise<object>} Status information
   */
  async getCacheStatus() {
    const [mapStats, dbStats, mapTilesReady] = await Promise.all([
      offlineMapTileCache.getCacheStats(),
      offlineDatabase.getStats(),
      offlineMapTileCache.isOfflineReady()
    ]);

    return {
      mapTiles: {
        cached: mapStats.cached,
        expected: mapStats.expected,
        percentage: mapStats.percentage,
        ready: mapTilesReady
      },
      database: dbStats,
      online: this.isOnline
    };
  }

  /**
   * Clear all offline caches
   * @returns {Promise<object>} Clear results
   */
  async clearAllCaches() {
    console.log('🗑️ Clearing all caches...');
    
    const results = {
      mapTiles: false,
      restaurants: false,
      attractions: false,
      pois: false
    };

    try {
      results.mapTiles = await offlineMapTileCache.clearCache();
      results.restaurants = await offlineDatabase.clear('restaurants');
      results.attractions = await offlineDatabase.clear('attractions');
      results.pois = await offlineDatabase.clear('pois');
      
      console.log('✅ Caches cleared');
      return results;
    } catch (error) {
      console.error('❌ Failed to clear caches:', error);
      throw error;
    }
  }

  /**
   * Notification helpers
   */
  notifyUpdate() {
    this.dispatchEvent('sw-update', { updateAvailable: true });
  }

  notifyReconnect() {
    this.dispatchEvent('reconnect', { online: true });
  }

  notifyOffline() {
    this.dispatchEvent('offline', { online: false });
  }

  notifyMapTilesCached(result) {
    this.dispatchEvent('map-tiles-cached', result);
  }

  dispatchEvent(name, detail) {
    window.dispatchEvent(new CustomEvent(`offline-enhancement-${name}`, { detail }));
  }
}

// Singleton instance
const offlineEnhancementManager = new OfflineEnhancementManager();

export default offlineEnhancementManager;
