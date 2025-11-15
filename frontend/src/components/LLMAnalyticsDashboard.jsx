/**
 * LLM Analytics Dashboard
 * 
 * Comprehensive dashboard for monitoring Pure LLM system performance,
 * analytics, and real-time metrics.
 * 
 * Features:
 * - Real-time statistics
 * - Performance metrics visualization
 * - Signal detection analytics
 * - Cache performance monitoring
 * - User behavior insights
 * - Live WebSocket updates
 * 
 * Author: AI Istanbul Team
 * Date: November 15, 2025
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  getGeneralStats,
  getSignalStats,
  getPerformanceStats,
  getCacheStats,
  getUserStats,
  getHourlyTrends,
  exportStats,
  createStatsWebSocket
} from '../api/llmStatsApi';
import './LLMAnalyticsDashboard.css';

const LLMAnalyticsDashboard = () => {
  // State management
  const [stats, setStats] = useState({
    general: null,
    signals: null,
    performance: null,
    cache: null,
    users: null,
    hourly: null
  });
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [realTimeEnabled, setRealTimeEnabled] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  
  const wsRef = useRef(null);
  const refreshTimerRef = useRef(null);
  
  /**
   * Fetch all statistics
   */
  const fetchAllStats = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [general, signals, performance, cache, users, hourly] = await Promise.all([
        getGeneralStats(),
        getSignalStats(),
        getPerformanceStats(24),
        getCacheStats(),
        getUserStats(7),
        getHourlyTrends(24)
      ]);
      
      setStats({
        general,
        signals,
        performance,
        cache,
        users,
        hourly
      });
      
      setLastUpdate(new Date());
      setLoading(false);
    } catch (err) {
      console.error('Error fetching stats:', err);
      setError(err.message);
      setLoading(false);
    }
  }, []);
  
  /**
   * Handle WebSocket messages
   */
  const handleWebSocketMessage = useCallback((data) => {
    console.log('📊 Real-time stats update:', data);
    
    // Update relevant stats based on message type
    if (data.type === 'general') {
      setStats(prev => ({ ...prev, general: data.data }));
    } else if (data.type === 'performance') {
      setStats(prev => ({ ...prev, performance: data.data }));
    } else if (data.type === 'full') {
      // Full update
      setStats(data.data);
    }
    
    setLastUpdate(new Date());
  }, []);
  
  /**
   * Handle WebSocket errors
   */
  const handleWebSocketError = useCallback((error) => {
    console.error('WebSocket error:', error);
    setRealTimeEnabled(false);
  }, []);
  
  /**
   * Toggle real-time updates
   */
  const toggleRealTime = useCallback(() => {
    if (realTimeEnabled) {
      // Disconnect WebSocket
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setRealTimeEnabled(false);
    } else {
      // Connect WebSocket
      wsRef.current = createStatsWebSocket(
        handleWebSocketMessage,
        handleWebSocketError
      );
      setRealTimeEnabled(true);
    }
  }, [realTimeEnabled, handleWebSocketMessage, handleWebSocketError]);
  
  /**
   * Export statistics
   */
  const handleExport = async (format) => {
    try {
      const blob = await exportStats(format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `llm-stats-${new Date().toISOString()}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Error exporting stats:', err);
      alert('Failed to export statistics');
    }
  };
  
  /**
   * Setup auto-refresh
   */
  useEffect(() => {
    if (autoRefresh && !realTimeEnabled) {
      refreshTimerRef.current = setInterval(() => {
        fetchAllStats();
      }, refreshInterval * 1000);
      
      return () => {
        if (refreshTimerRef.current) {
          clearInterval(refreshTimerRef.current);
        }
      };
    }
  }, [autoRefresh, realTimeEnabled, refreshInterval, fetchAllStats]);
  
  /**
   * Initial data fetch
   */
  useEffect(() => {
    fetchAllStats();
    
    // Cleanup WebSocket on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [fetchAllStats]);
  
  /**
   * Format number with K/M suffixes
   */
  const formatNumber = (num) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };
  
  /**
   * Format percentage
   */
  const formatPercent = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };
  
  /**
   * Get status color based on value
   */
  const getStatusColor = (value, thresholds) => {
    if (value >= thresholds.good) return 'status-good';
    if (value >= thresholds.warning) return 'status-warning';
    return 'status-critical';
  };
  
  // Loading state
  if (loading && !stats.general) {
    return (
      <div className="llm-analytics-dashboard">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading analytics...</p>
        </div>
      </div>
    );
  }
  
  // Error state
  if (error && !stats.general) {
    return (
      <div className="llm-analytics-dashboard">
        <div className="error-container">
          <h2>⚠️ Error Loading Analytics</h2>
          <p>{error}</p>
          <button onClick={fetchAllStats} className="btn-primary">
            Retry
          </button>
        </div>
      </div>
    );
  }
  
  const { general, signals, performance, cache, users, hourly } = stats;
  
  return (
    <div className="llm-analytics-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-left">
          <h1>🤖 LLM Analytics Dashboard</h1>
          <p className="subtitle">Pure LLM System Monitoring & Analytics</p>
        </div>
        
        <div className="header-right">
          <div className="header-controls">
            {/* Real-time toggle */}
            <button
              className={`btn-toggle ${realTimeEnabled ? 'active' : ''}`}
              onClick={toggleRealTime}
              title={realTimeEnabled ? 'Disable real-time updates' : 'Enable real-time updates'}
            >
              <span className={`indicator ${realTimeEnabled ? 'live' : ''}`}></span>
              {realTimeEnabled ? 'Live' : 'Static'}
            </button>
            
            {/* Auto-refresh toggle */}
            {!realTimeEnabled && (
              <button
                className={`btn-toggle ${autoRefresh ? 'active' : ''}`}
                onClick={() => setAutoRefresh(!autoRefresh)}
                title="Toggle auto-refresh"
              >
                🔄 Auto-refresh {autoRefresh ? 'ON' : 'OFF'}
              </button>
            )}
            
            {/* Refresh button */}
            <button
              className="btn-icon"
              onClick={fetchAllStats}
              title="Refresh now"
              disabled={loading}
            >
              🔄
            </button>
            
            {/* Export buttons */}
            <button
              className="btn-secondary"
              onClick={() => handleExport('json')}
              title="Export as JSON"
            >
              📄 JSON
            </button>
            <button
              className="btn-secondary"
              onClick={() => handleExport('csv')}
              title="Export as CSV"
            >
              📊 CSV
            </button>
          </div>
          
          {lastUpdate && (
            <div className="last-update">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>
      
      {/* Key Metrics Overview */}
      <div className="metrics-overview">
        <div className="metric-card">
          <div className="metric-icon">📊</div>
          <div className="metric-content">
            <div className="metric-value">{formatNumber(general?.total_queries || 0)}</div>
            <div className="metric-label">Total Queries</div>
          </div>
        </div>
        
        <div className="metric-card">
          <div className="metric-icon">⚡</div>
          <div className="metric-content">
            <div className="metric-value">{general?.average_response_time_ms || 0}ms</div>
            <div className="metric-label">Avg Response Time</div>
            <div className={`metric-status ${getStatusColor(
              general?.average_response_time_ms || 0,
              { good: 500, warning: 1000 }
            )}`}>
              {general?.average_response_time_ms < 500 ? '✅ Excellent' : 
               general?.average_response_time_ms < 1000 ? '⚠️ Good' : '❌ Slow'}
            </div>
          </div>
        </div>
        
        <div className="metric-card">
          <div className="metric-icon">💾</div>
          <div className="metric-content">
            <div className="metric-value">{formatPercent(general?.cache_hit_rate || 0)}</div>
            <div className="metric-label">Cache Hit Rate</div>
            <div className={`metric-status ${getStatusColor(
              general?.cache_hit_rate || 0,
              { good: 0.7, warning: 0.5 }
            )}`}>
              {general?.cache_hit_rate >= 0.7 ? '✅ Excellent' : 
               general?.cache_hit_rate >= 0.5 ? '⚠️ Good' : '❌ Low'}
            </div>
          </div>
        </div>
        
        <div className="metric-card">
          <div className="metric-icon">❌</div>
          <div className="metric-content">
            <div className="metric-value">{formatPercent(general?.error_rate || 0)}</div>
            <div className="metric-label">Error Rate</div>
            <div className={`metric-status ${getStatusColor(
              1 - (general?.error_rate || 0),
              { good: 0.99, warning: 0.95 }
            )}`}>
              {general?.error_rate < 0.01 ? '✅ Excellent' : 
               general?.error_rate < 0.05 ? '⚠️ Good' : '❌ High'}
            </div>
          </div>
        </div>
        
        <div className="metric-card">
          <div className="metric-icon">👥</div>
          <div className="metric-content">
            <div className="metric-value">{formatNumber(general?.active_users || 0)}</div>
            <div className="metric-label">Active Users</div>
          </div>
        </div>
      </div>
      
      {/* Performance Metrics */}
      <div className="dashboard-section">
        <h2>⚡ Performance Metrics</h2>
        <div className="performance-grid">
          <div className="stat-card">
            <h3>Response Time Percentiles</h3>
            <div className="percentiles">
              <div className="percentile-item">
                <span className="percentile-label">P50 (Median)</span>
                <span className="percentile-value">{general?.performance?.p50_ms || 0}ms</span>
              </div>
              <div className="percentile-item">
                <span className="percentile-label">P95</span>
                <span className="percentile-value">{general?.performance?.p95_ms || 0}ms</span>
              </div>
              <div className="percentile-item">
                <span className="percentile-label">P99</span>
                <span className="percentile-value">{general?.performance?.p99_ms || 0}ms</span>
              </div>
            </div>
          </div>
          
          <div className="stat-card">
            <h3>Cache Performance</h3>
            <div className="cache-stats">
              <div className="cache-item">
                <span className="cache-label">Hits</span>
                <span className="cache-value">{formatNumber(cache?.hits || 0)}</span>
              </div>
              <div className="cache-item">
                <span className="cache-label">Misses</span>
                <span className="cache-value">{formatNumber(cache?.misses || 0)}</span>
              </div>
              <div className="cache-item">
                <span className="cache-label">Hit Rate</span>
                <span className="cache-value">{formatPercent(cache?.hit_rate || 0)}</span>
              </div>
              <div className="cache-item">
                <span className="cache-label">Size</span>
                <span className="cache-value">{cache?.cache_size || 0} entries</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Signal Detection */}
      <div className="dashboard-section">
        <h2>🎯 Top Detected Signals</h2>
        <div className="signals-list">
          {general?.top_signals?.map((signal, index) => (
            <div key={index} className="signal-item">
              <div className="signal-rank">#{index + 1}</div>
              <div className="signal-name">{signal.signal}</div>
              <div className="signal-count">{formatNumber(signal.count)} detections</div>
              <div className="signal-bar">
                <div
                  className="signal-bar-fill"
                  style={{
                    width: `${(signal.count / (general?.top_signals?.[0]?.count || 1)) * 100}%`
                  }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Language Distribution */}
      <div className="dashboard-section">
        <h2>🌍 Language Distribution</h2>
        <div className="languages-grid">
          {Object.entries(general?.languages || {}).map(([lang, count]) => (
            <div key={lang} className="language-card">
              <div className="language-flag">{lang === 'en' ? '🇬🇧' : lang === 'tr' ? '🇹🇷' : '🌐'}</div>
              <div className="language-name">{lang.toUpperCase()}</div>
              <div className="language-count">{formatNumber(count)}</div>
            </div>
          ))}
        </div>
      </div>
      
      {/* System Status */}
      <div className="dashboard-section">
        <h2>💚 System Status</h2>
        <div className="status-grid">
          <div className="status-item">
            <span className="status-indicator status-good"></span>
            <span className="status-label">Pure LLM Core</span>
            <span className="status-value">Operational</span>
          </div>
          <div className="status-item">
            <span className={`status-indicator ${cache?.enabled ? 'status-good' : 'status-warning'}`}></span>
            <span className="status-label">Cache System</span>
            <span className="status-value">{cache?.enabled ? 'Enabled' : 'Disabled'}</span>
          </div>
          <div className="status-item">
            <span className={`status-indicator ${realTimeEnabled ? 'status-good' : 'status-warning'}`}></span>
            <span className="status-label">Real-time Monitoring</span>
            <span className="status-value">{realTimeEnabled ? 'Active' : 'Inactive'}</span>
          </div>
          <div className="status-item">
            <span className="status-indicator status-good"></span>
            <span className="status-label">Analytics Tracking</span>
            <span className="status-value">Active</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LLMAnalyticsDashboard;
