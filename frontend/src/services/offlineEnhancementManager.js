/**
 * Offline Enhancement Manager
 * Integrates all offline enhancement features
 * 
 * @version 1.0.0
 */

// 🔧 TEMPORARY: Disable offline sync until backend endpoints are ready
const OFFLINE_SYNC_ENABLED = false;

import offlineMapTileCache from './offlineMapTileCache.js';
import offlineIntentDetector from './offlineIntentDetector.js';
import offlineDatabase from './offlineDatabase.js';
import { Logger } from '../utils/logger.js';

const log = new Logger('OfflineManager');

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
      log.warn('⚠️ Already initialized');
      return { status: 'already_initialized' };
    }

    this.config = { ...this.config, ...config };
    log.info('🚀 Initializing offline enhancements...');

    try {
      // 1. Setup online/offline listeners
      this.setupNetworkListeners();

      // 2. Initialize IndexedDB
      await offlineDatabase.init();
      log.info('✅ IndexedDB initialized');

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
      log.debug('📊 Cache status:', stats);

      // 6. Auto-sync if online
      if (this.isOnline && this.config.autoSyncOnReconnect) {
        this.syncAllData().catch(err => 
          log.warn('Initial sync failed:', err)
        );
      }

      this.isInitialized = true;
      log.debug('✅ Offline enhancements initialized');

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
      log.error('❌ Initialization failed:', error);
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
      log.debug('🌐 Back online');
      this.isOnline = true;
      this.handleReconnect();
    });

    window.addEventListener('offline', () => {
      log.debug('📴 Offline mode activated');
      this.isOnline = false;
      this.handleOffline();
    });
  }

  /**
   * Register enhanced service worker
   */
  async registerServiceWorker() {
    try {
      const registration = await navigator.serviceWorker.register('/sw-enhanced.js', {
        updateViaCache: 'none' // Force check for updates on every page load
      });
      log.debug('✅ Enhanced service worker registered');
      
      // Force update check on page load
      registration.update().catch(err => 
        log.warn('Update check failed:', err)
      );
      
      // Listen for updates
      registration.addEventListener('updatefound', () => {
        log.debug('🔄 Service worker update found');
        const newWorker = registration.installing;
        
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            log.debug('✅ New service worker installed, activating...');
            
            // Automatically skip waiting and reload
            newWorker.postMessage({ type: 'SKIP_WAITING' });
            
            // Notify user and offer to reload
            this.notifyUpdateAndReload();
          }
        });
      });

      // Handle controlling service worker changes
      let refreshing = false;
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        if (!refreshing) {
          refreshing = true;
          log.debug('🔄 New service worker activated, reloading page...');
          // Automatically reload to use new service worker
          window.location.reload();
        }
      });

      return registration;
    } catch (error) {
      log.error('❌ Service worker registration failed:', error);
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
        // Check if periodic sync permission is granted
        const status = await navigator.permissions.query({ name: 'periodic-background-sync' }).catch(() => null);
        
        if (status && status.state === 'granted') {
          await registration.periodicSync.register('update-cache', {
            minInterval: 24 * 60 * 60 * 1000 // 24 hours
          });
          log.debug('✅ Periodic sync registered');
        } else {
          log.debug('ℹ️ Periodic sync permission not granted (this is optional)');
        }
      } else {
        log.debug('ℹ️ Periodic sync not supported by this browser');
      }
    } catch (error) {
      // Periodic sync is optional - fail silently
      if (error.name === 'NotAllowedError') {
        log.debug('ℹ️ Periodic sync not allowed (this is optional and doesn\'t affect functionality)');
      } else {
        log.debug('ℹ️ Periodic sync unavailable:', error.message);
      }
    }
  }

  /**
   * Handle reconnection - sync data
   */
  async handleReconnect() {
    if (!this.config.autoSyncOnReconnect) return;

    log.debug('🔄 Syncing data after reconnection...');
    
    try {
      await this.syncAllData();
      this.notifyReconnect();
    } catch (error) {
      log.error('❌ Reconnect sync failed:', error);
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
    if (!OFFLINE_SYNC_ENABLED) {
      log.debug('ℹ️ Offline sync disabled - backend endpoints not ready yet');
      return {
        success: true,
        disabled: true,
        restaurants: { skipped: true },
        attractions: { skipped: true },
        pois: { skipped: true }
      };
    }
    
    log.debug('🔄 Syncing all offline data...');
    
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

      log.debug('✅ Data sync complete:', results);
      return results;
    } catch (error) {
      log.error('❌ Data sync failed:', error);
      throw error;
    }
  }

  /**
   * Cache Istanbul map tiles
   * @param {function} onProgress - Progress callback
   * @returns {Promise<object>} Cache results
   */
  async cacheMapTiles(onProgress = null) {
    log.debug('📦 Starting map tile caching...');
    
    try {
      const result = await offlineMapTileCache.cacheIstanbulTiles(onProgress);
      
      if (result.status === 'complete') {
        await offlineDatabase.saveMetadata('map_tiles_cached', true);
        this.notifyMapTilesCached(result);
      }
      
      return result;
    } catch (error) {
      log.error('❌ Map tile caching failed:', error);
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
    log.debug('🗑️ Clearing all caches...');
    
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
      
      log.debug('✅ Caches cleared');
      return results;
    } catch (error) {
      log.error('❌ Failed to clear caches:', error);
      throw error;
    }
  }

  /**
   * Notification helpers
   */
  notifyUpdate() {
    this.dispatchEvent('sw-update', { updateAvailable: true });
  }

  notifyUpdateAndReload() {
    log.debug('🔄 Notifying user of update...');
    this.dispatchEvent('sw-update-ready', { 
      updateAvailable: true,
      message: 'A new version is available and will load shortly...' 
    });
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
