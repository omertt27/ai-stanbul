/**
 * Offline Database Service
 * Uses IndexedDB for structured          poiStore.createIndex('type', 'type', { unique: false });
          log.debug('📦 Created POIs store');ffline storage of restaurants, attractions, and POIs
 * 
 * @version 1.0.0
 * @priority MEDIUM
 */

import { Logger } from '../utils/logger.js';
const log = new Logger('OfflineDB');

class OfflineDatabase {
  constructor() {
    this.dbName = 'istanbul-ai';
    this.version = 1;
    this.db = null;
  }

  /**
   * Initialize IndexedDB connection
   * @returns {Promise<IDBDatabase>} Database instance
   */
  async init() {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => {
        log.error('❌ Failed to open IndexedDB:', request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        log.info('✅ IndexedDB initialized:', this.dbName);
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Create object stores
        if (!db.objectStoreNames.contains('restaurants')) {
          const restaurantStore = db.createObjectStore('restaurants', { keyPath: 'id' });
          restaurantStore.createIndex('name', 'name', { unique: false });
          restaurantStore.createIndex('cuisine', 'cuisine', { unique: false });
          restaurantStore.createIndex('district', 'district', { unique: false });
          restaurantStore.createIndex('rating', 'rating', { unique: false });
          log.debug('📦 Created restaurants store');
        }

        if (!db.objectStoreNames.contains('attractions')) {
          const attractionStore = db.createObjectStore('attractions', { keyPath: 'id' });
          attractionStore.createIndex('name', 'name', { unique: false });
          attractionStore.createIndex('category', 'category', { unique: false });
          attractionStore.createIndex('district', 'district', { unique: false });
          attractionStore.createIndex('rating', 'rating', { unique: false          });
          log.debug('📦 Created attractions store');
        }

        if (!db.objectStoreNames.contains('pois')) {
          const poiStore = db.createObjectStore('pois', { keyPath: 'id' });
          poiStore.createIndex('type', 'type', { unique: false });
          poiStore.createIndex('name', 'name', { unique: false });
          log.debug('📦 Created POIs store');
        }

        if (!db.objectStoreNames.contains('metadata')) {
          db.createObjectStore('metadata', { keyPath: 'key' });
          log.debug('📦 Created metadata store');
        }
      };
    });
  }

  /**
   * Add or update items in a store
   * @param {string} storeName - Store name
   * @param {Array|Object} items - Items to store
   * @returns {Promise<number>} Number of items stored
   */
  async putItems(storeName, items) {
    await this.init();
    const itemsArray = Array.isArray(items) ? items : [items];

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      let count = 0;

      itemsArray.forEach(item => {
        store.put(item);
        count++;
      });

      transaction.oncomplete = () => {
        log.debug(`✅ Stored ${count} items in ${storeName}`);
        resolve(count);
      };

      transaction.onerror = () => {
        log.error(`❌ Error storing items in ${storeName}:`, transaction.error);
        reject(transaction.error);
      };
    });
  }

  /**
   * Get all items from a store
   * @param {string} storeName - Store name
   * @returns {Promise<Array>} All items
   */
  async getAll(storeName) {
    await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Get item by ID
   * @param {string} storeName - Store name
   * @param {string|number} id - Item ID
   * @returns {Promise<Object>} Item
   */
  async getById(storeName, id) {
    await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Search items by index
   * @param {string} storeName - Store name
   * @param {string} indexName - Index to search
   * @param {any} value - Value to search for
   * @returns {Promise<Array>} Matching items
   */
  async searchByIndex(storeName, indexName, value) {
    await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const index = store.index(indexName);
      const request = index.getAll(value);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Full-text search across items
   * @param {string} storeName - Store name
   * @param {string} query - Search query
   * @returns {Promise<Array>} Matching items
   */
  async search(storeName, query) {
    const items = await this.getAll(storeName);
    const normalizedQuery = query.toLowerCase();

    return items.filter(item => {
      const searchableText = JSON.stringify(Object.values(item)).toLowerCase();
      return searchableText.includes(normalizedQuery);
    });
  }

  /**
   * Filter items by multiple criteria
   * @param {string} storeName - Store name
   * @param {Object} filters - Filter criteria
   * @returns {Promise<Array>} Filtered items
   */
  async filter(storeName, filters) {
    const items = await this.getAll(storeName);

    return items.filter(item => {
      return Object.entries(filters).every(([key, value]) => {
        if (value === null || value === undefined) return true;
        
        if (Array.isArray(value)) {
          return value.includes(item[key]);
        }
        
        if (typeof value === 'object' && value.min !== undefined && value.max !== undefined) {
          return item[key] >= value.min && item[key] <= value.max;
        }
        
        return item[key] === value;
      });
    });
  }

  /**
   * Count items in store
   * @param {string} storeName - Store name
   * @returns {Promise<number>} Item count
   */
  async count(storeName) {
    await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.count();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Clear all data from a store
   * @param {string} storeName - Store name
   * @returns {Promise<void>}
   */
  async clear(storeName) {
    await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.clear();

      request.onsuccess = () => {
        log.debug(`✅ Cleared ${storeName}`);
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Save metadata (e.g., last sync time)
   * @param {string} key - Metadata key
   * @param {any} value - Metadata value
   * @returns {Promise<void>}
   */
  async saveMetadata(key, value) {
    await this.putItems('metadata', {
      key,
      value,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Get metadata
   * @param {string} key - Metadata key
   * @returns {Promise<Object>} Metadata
   */
  async getMetadata(key) {
    return await this.getById('metadata', key);
  }

  /**
   * Sync data from server
   * @param {string} apiEndpoint - API endpoint to fetch data
   * @param {string} storeName - Store to sync to
   * @returns {Promise<object>} Sync results
   */
  async syncFromServer(apiEndpoint, storeName) {
    try {
      log.debug(`🔄 Syncing ${storeName} from server...`);
      
      const response = await fetch(apiEndpoint);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      const items = Array.isArray(data) ? data : data.results || data.data || [];
      
      const count = await this.putItems(storeName, items);
      await this.saveMetadata(`${storeName}_last_sync`, new Date().toISOString());
      
      log.info(`✅ Synced ${count} ${storeName} from server`);
      
      return {
        success: true,
        count,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      log.error(`❌ Failed to sync ${storeName}:`, error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get database statistics
   * @returns {Promise<object>} Statistics
   */
  async getStats() {
    const stats = {};
    const stores = ['restaurants', 'attractions', 'pois'];

    for (const store of stores) {
      stats[store] = await this.count(store);
    }

    // Get last sync times
    for (const store of stores) {
      const metadata = await this.getMetadata(`${store}_last_sync`);
      stats[`${store}_last_sync`] = metadata?.value || 'Never';
    }

    return stats;
  }
}

// Singleton instance
const offlineDatabase = new OfflineDatabase();

export default offlineDatabase;

// Convenience exports for specific stores
export const restaurantDB = {
  getAll: () => offlineDatabase.getAll('restaurants'),
  search: (query) => offlineDatabase.search('restaurants', query),
  filter: (filters) => offlineDatabase.filter('restaurants', filters),
  syncFromServer: (endpoint) => offlineDatabase.syncFromServer(endpoint, 'restaurants')
};

export const attractionDB = {
  getAll: () => offlineDatabase.getAll('attractions'),
  search: (query) => offlineDatabase.search('attractions', query),
  filter: (filters) => offlineDatabase.filter('attractions', filters),
  syncFromServer: (endpoint) => offlineDatabase.syncFromServer(endpoint, 'attractions')
};

export const poiDB = {
  getAll: () => offlineDatabase.getAll('pois'),
  search: (query) => offlineDatabase.search('pois', query),
  filter: (filters) => offlineDatabase.filter('pois', filters),
  syncFromServer: (endpoint) => offlineDatabase.syncFromServer(endpoint, 'pois')
};
