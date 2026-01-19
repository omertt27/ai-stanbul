/**
 * Offline Map Tile Cache Service
 * Pre-caches map tiles for Istanbul area for true offline map support
 * 
 * @version 1.0.0
 * @priority HIGH
 */

// Istanbul bounds for tile caching
const ISTANBUL_BOUNDS = {
  north: 41.2,
  south: 40.85,
  east: 29.4,
  west: 28.6
};

// Zoom levels to cache (10-16 for optimal balance)
const ZOOM_LEVELS = [10, 11, 12, 13, 14, 15, 16];

// Tile server URL (OpenStreetMap)
const TILE_URL_TEMPLATE = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

class OfflineMapTileCache {
  constructor() {
    this.cacheName = 'map-tiles-v1';
    this.totalTiles = 0;
    this.cachedTiles = 0;
    this.isCaching = false;
  }

  /**
   * Calculate tile numbers for given lat/lon at specific zoom level
   * @param {number} lat - Latitude
   * @param {number} lon - Longitude
   * @param {number} zoom - Zoom level
   * @returns {object} Tile coordinates {x, y, z}
   */
  latLonToTile(lat, lon, zoom) {
    const n = Math.pow(2, zoom);
    const x = Math.floor((lon + 180) / 360 * n);
    const y = Math.floor((1 - Math.log(Math.tan(lat * Math.PI / 180) + 1 / Math.cos(lat * Math.PI / 180)) / Math.PI) / 2 * n);
    return { x, y, z: zoom };
  }

  /**
   * Generate all tile URLs for Istanbul bounds at specified zoom levels
   * @returns {Array<string>} Array of tile URLs
   */
  generateTileUrls() {
    const urls = [];
    
    ZOOM_LEVELS.forEach(zoom => {
      const topLeft = this.latLonToTile(ISTANBUL_BOUNDS.north, ISTANBUL_BOUNDS.west, zoom);
      const bottomRight = this.latLonToTile(ISTANBUL_BOUNDS.south, ISTANBUL_BOUNDS.east, zoom);
      
      for (let x = topLeft.x; x <= bottomRight.x; x++) {
        for (let y = topLeft.y; y <= bottomRight.y; y++) {
          const url = TILE_URL_TEMPLATE
            .replace('{z}', zoom)
            .replace('{x}', x)
            .replace('{y}', y);
          urls.push(url);
        }
      }
    });
    
    return urls;
  }

  /**
   * Pre-cache all Istanbul map tiles
   * @param {function} onProgress - Progress callback (current, total, percentage)
   * @returns {Promise<object>} Cache statistics
   */
  async cacheIstanbulTiles(onProgress = null) {
    if (this.isCaching) {
      log.warn('⚠️ Caching already in progress');
      return { status: 'in_progress' };
    }

    this.isCaching = true;
    this.cachedTiles = 0;
    
    const tileUrls = this.generateTileUrls();
    this.totalTiles = tileUrls.length;
    
    log.debug(`📦 Starting to cache ${this.totalTiles} tiles for Istanbul area`);
    
    try {
      const cache = await caches.open(this.cacheName);
      
      // Check which tiles are already cached
      const cachedRequests = await cache.keys();
      const cachedUrls = new Set(cachedRequests.map(req => req.url));
      
      // Filter out already cached tiles
      const tilesToCache = tileUrls.filter(url => !cachedUrls.has(url));
      
      if (tilesToCache.length === 0) {
        log.debug('✅ All tiles already cached');
        this.isCaching = false;
        return {
          status: 'complete',
          total: this.totalTiles,
          cached: this.totalTiles,
          skipped: 0
        };
      }
      
      log.debug(`🔄 Caching ${tilesToCache.length} new tiles (${cachedUrls.size} already cached)`);
      
      // Cache tiles in batches to avoid overwhelming the browser
      const BATCH_SIZE = 20;
      let completed = cachedUrls.size;
      
      for (let i = 0; i < tilesToCache.length; i += BATCH_SIZE) {
        const batch = tilesToCache.slice(i, i + BATCH_SIZE);
        
        await Promise.allSettled(
          batch.map(async url => {
            try {
              const response = await fetch(url, {
                mode: 'cors',
                credentials: 'omit'
              });
              
              if (response.ok) {
                await cache.put(url, response.clone());
                this.cachedTiles++;
                completed++;
                
                if (onProgress) {
                  const percentage = Math.round((completed / this.totalTiles) * 100);
                  onProgress(completed, this.totalTiles, percentage);
                }
              }
            } catch (error) {
              log.warn(`Failed to cache tile: ${url}`, error);
            }
          })
        );
        
        // Small delay between batches to prevent rate limiting
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      this.isCaching = false;
      
      const result = {
        status: 'complete',
        total: this.totalTiles,
        cached: completed,
        new: tilesToCache.length
      };
      
      log.debug('✅ Map tile caching complete:', result);
      return result;
      
    } catch (error) {
      this.isCaching = false;
      log.error('❌ Error caching map tiles:', error);
      throw error;
    }
  }

  /**
   * Get caching statistics
   * @returns {Promise<object>} Cache stats
   */
  async getCacheStats() {
    try {
      const cache = await caches.open(this.cacheName);
      const cachedRequests = await cache.keys();
      const expectedTiles = this.generateTileUrls().length;
      
      return {
        cached: cachedRequests.length,
        expected: expectedTiles,
        percentage: Math.round((cachedRequests.length / expectedTiles) * 100),
        isCaching: this.isCaching
      };
    } catch (error) {
      log.error('Error getting cache stats:', error);
      return { cached: 0, expected: 0, percentage: 0, isCaching: false };
    }
  }

  /**
   * Clear all cached map tiles
   * @returns {Promise<boolean>} Success status
   */
  async clearCache() {
    try {
      const deleted = await caches.delete(this.cacheName);
      log.debug(deleted ? '✅ Map tile cache cleared' : '⚠️ No cache to clear');
      return deleted;
    } catch (error) {
      log.error('Error clearing cache:', error);
      return false;
    }
  }

  /**
   * Check if map tiles are available offline
   * @returns {Promise<boolean>} True if substantial cache exists
   */
  async isOfflineReady() {
    const stats = await this.getCacheStats();
    return stats.percentage >= 80; // Consider ready if 80%+ cached
  }
}

// Singleton instance
const offlineMapTileCache = new OfflineMapTileCache();

export default offlineMapTileCache;
