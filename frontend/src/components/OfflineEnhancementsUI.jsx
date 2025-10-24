/**
 * Offline Enhancement UI Component
 * Example React component showing how to use the offline enhancements
 * 
 * @version 1.0.0
 */

import React, { useState, useEffect } from 'react';
import offlineEnhancementManager from '../services/offlineEnhancementManager';
import '../styles/offline-enhancements.css';

export default function OfflineEnhancementsUI() {
  const [status, setStatus] = useState({
    initialized: false,
    online: navigator.onLine,
    caching: false,
    cacheStats: null
  });
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Initialize enhancements
    initializeEnhancements();

    // Listen for status changes
    window.addEventListener('offline-enhancement-reconnect', handleReconnect);
    window.addEventListener('offline-enhancement-offline', handleOffline);
    window.addEventListener('offline-enhancement-map-tiles-cached', handleMapTilesCached);

    return () => {
      window.removeEventListener('offline-enhancement-reconnect', handleReconnect);
      window.removeEventListener('offline-enhancement-offline', handleOffline);
      window.removeEventListener('offline-enhancement-map-tiles-cached', handleMapTilesCached);
    };
  }, []);

  const initializeEnhancements = async () => {
    try {
      const result = await offlineEnhancementManager.initialize({
        autoSyncOnReconnect: true,
        enablePeriodicSync: true,
        enableOfflineIntents: true
      });

      setStatus(prev => ({
        ...prev,
        initialized: true,
        cacheStats: result.stats
      }));
    } catch (error) {
      console.error('Failed to initialize:', error);
    }
  };

  const handleCacheMapTiles = async () => {
    if (status.caching) return;

    setStatus(prev => ({ ...prev, caching: true }));
    setProgress(0);

    try {
      await offlineEnhancementManager.cacheMapTiles((current, total, percentage) => {
        setProgress(percentage);
      });

      // Update cache stats
      const stats = await offlineEnhancementManager.getCacheStatus();
      setStatus(prev => ({
        ...prev,
        caching: false,
        cacheStats: stats
      }));
    } catch (error) {
      console.error('Failed to cache tiles:', error);
      setStatus(prev => ({ ...prev, caching: false }));
    }
  };

  const handleSyncData = async () => {
    try {
      await offlineEnhancementManager.syncAllData();
      
      // Update cache stats
      const stats = await offlineEnhancementManager.getCacheStatus();
      setStatus(prev => ({ ...prev, cacheStats: stats }));
      
      alert('✅ Data synced successfully!');
    } catch (error) {
      console.error('Failed to sync:', error);
      alert('❌ Sync failed. Please try again.');
    }
  };

  const handleClearCache = async () => {
    if (!confirm('Clear all offline caches? This will remove all downloaded data.')) {
      return;
    }

    try {
      await offlineEnhancementManager.clearAllCaches();
      
      const stats = await offlineEnhancementManager.getCacheStatus();
      setStatus(prev => ({ ...prev, cacheStats: stats }));
      
      alert('✅ Caches cleared successfully!');
    } catch (error) {
      console.error('Failed to clear:', error);
      alert('❌ Clear failed. Please try again.');
    }
  };

  const handleReconnect = () => {
    setStatus(prev => ({ ...prev, online: true }));
  };

  const handleOffline = () => {
    setStatus(prev => ({ ...prev, online: false }));
  };

  const handleMapTilesCached = (event) => {
    console.log('Map tiles cached:', event.detail);
  };

  if (!status.initialized) {
    return (
      <div className="offline-enhancements loading">
        <p>🔄 Initializing offline enhancements...</p>
      </div>
    );
  }

  return (
    <div className="offline-enhancements">
      <div className="status-header">
        <h2>🚀 Offline Enhancements</h2>
        <div className={`status-indicator ${status.online ? 'online' : 'offline'}`}>
          {status.online ? '🌐 Online' : '📴 Offline'}
        </div>
      </div>

      {/* Map Tiles Section */}
      <section className="enhancement-section">
        <h3>📍 Map Tiles</h3>
        <p>Download Istanbul map tiles for complete offline map support</p>
        
        {status.cacheStats?.mapTiles && (
          <div className="cache-stats">
            <div className="stat-item">
              <span className="stat-label">Cached Tiles:</span>
              <span className="stat-value">
                {status.cacheStats.mapTiles.cached} / {status.cacheStats.mapTiles.expected}
              </span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${status.cacheStats.mapTiles.percentage}%` }}
              />
            </div>
            <div className="stat-percentage">
              {status.cacheStats.mapTiles.percentage}% complete
            </div>
          </div>
        )}

        {status.caching && (
          <div className="caching-progress">
            <p>⏳ Caching map tiles... {progress}%</p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        <button 
          onClick={handleCacheMapTiles}
          disabled={status.caching || !status.online}
          className="btn btn-primary"
        >
          {status.caching ? '⏳ Caching...' : '📥 Download Map Tiles'}
        </button>
        
        {!status.online && (
          <p className="warning">⚠️ Connect to internet to download map tiles</p>
        )}
      </section>

      {/* Database Section */}
      <section className="enhancement-section">
        <h3>💾 Offline Database</h3>
        <p>Sync restaurants, attractions, and POIs for offline access</p>
        
        {status.cacheStats?.database && (
          <div className="cache-stats">
            <div className="stat-item">
              <span className="stat-label">Restaurants:</span>
              <span className="stat-value">{status.cacheStats.database.restaurants}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Attractions:</span>
              <span className="stat-value">{status.cacheStats.database.attractions}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">POIs:</span>
              <span className="stat-value">{status.cacheStats.database.pois}</span>
            </div>
            <div className="stat-item small">
              <span className="stat-label">Last Sync:</span>
              <span className="stat-value">
                {status.cacheStats.database.restaurants_last_sync}
              </span>
            </div>
          </div>
        )}

        <button 
          onClick={handleSyncData}
          disabled={!status.online}
          className="btn btn-primary"
        >
          🔄 Sync Data
        </button>
        
        {!status.online && (
          <p className="warning">⚠️ Connect to internet to sync data</p>
        )}
      </section>

      {/* Actions Section */}
      <section className="enhancement-section">
        <h3>⚙️ Actions</h3>
        
        <button 
          onClick={handleClearCache}
          className="btn btn-danger"
        >
          🗑️ Clear All Caches
        </button>
      </section>

      {/* Info Section */}
      <section className="enhancement-section info">
        <h3>ℹ️ Offline Features</h3>
        <ul>
          <li>✅ Complete transit map (7 metro, 3 tram, 5+ ferry lines)</li>
          <li>✅ Station lookup and nearest station finder</li>
          <li>✅ Basic route planning with static schedules</li>
          <li>✅ Restaurant search (when synced)</li>
          <li>✅ Attraction discovery (when synced)</li>
          <li>✅ Offline intent detection for common queries</li>
          <li>✅ Background sync when reconnected</li>
          <li>✅ Periodic cache updates (every 24 hours)</li>
        </ul>
      </section>
    </div>
  );
}
