/**
 * LLM Statistics API Client
 * 
 * Provides functions to interact with the Pure LLM statistics endpoints
 * for real-time analytics and monitoring.
 * 
 * Author: AI Istanbul Team
 * Date: November 15, 2025
 */

import { Logger } from '../utils/logger';

const logger = new Logger('LLMStatsAPI');
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const STATS_API_BASE = `${BASE_URL}/api/v1/llm`;

/**
 * Fetch general LLM statistics
 * @returns {Promise<Object>} General statistics overview
 */
export const getGeneralStats = async () => {
  try {
    // Fetch both LLM stats and feedback stats for real data
    const [llmResponse, feedbackResponse] = await Promise.all([
      fetch(`${STATS_API_BASE}/stats`).catch(() => null),
      fetch(`${BASE_URL}/api/feedback/stats`).catch(() => null)
    ]);

    let stats = {
      total_queries: 0,
      cache_hits: 0,
      cache_misses: 0,
      llm_calls: 0,
      cache_hit_rate: 0,
      unique_users: 0,
      error_rate: 0
    };

    // Get LLM stats if available
    if (llmResponse?.ok) {
      const llmData = await llmResponse.json();
      stats = { ...stats, ...llmData };
    }

    // Merge with real feedback stats from database
    if (feedbackResponse?.ok) {
      const feedbackData = await feedbackResponse.json();
      stats.total_queries = feedbackData.total_interactions || stats.total_queries;
      stats.languages = feedbackData.languages;
      stats.intents = feedbackData.intents;
      stats.positive_feedback = feedbackData.positive_feedback;
      stats.negative_feedback = feedbackData.negative_feedback;
      stats.feedback_rate = feedbackData.feedback_rate;
    }

    return stats;
  } catch (error) {
    logger.error('Error fetching general stats:', error);
    throw error;
  }
};

/**
 * Fetch signal detection statistics
 * @param {Object} filters - Optional filters (date_from, date_to, signal_type, language)
 * @returns {Promise<Object>} Signal detection analytics
 */
export const getSignalStats = async (filters = {}) => {
  try {
    const params = new URLSearchParams();
    if (filters.date_from) params.append('date_from', filters.date_from);
    if (filters.date_to) params.append('date_to', filters.date_to);
    if (filters.signal_type) params.append('signal_type', filters.signal_type);
    if (filters.language) params.append('language', filters.language);
    
    const url = `${STATS_API_BASE}/stats/signals${params.toString() ? '?' + params.toString() : ''}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    logger.error('Error fetching signal stats:', error);
    throw error;
  }
};

/**
 * Fetch performance statistics
 * @param {number} hours - Number of hours to look back (default: 24)
 * @returns {Promise<Object>} Performance metrics
 */
export const getPerformanceStats = async (hours = 24) => {
  try {
    const response = await fetch(`${STATS_API_BASE}/stats/performance?hours=${hours}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    logger.error('Error fetching performance stats:', error);
    throw error;
  }
};

/**
 * Fetch cache statistics
 * @returns {Promise<Object>} Cache performance metrics
 */
export const getCacheStats = async () => {
  try {
    const response = await fetch(`${STATS_API_BASE}/stats/cache`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    logger.error('Error fetching cache stats:', error);
    throw error;
  }
};

/**
 * Fetch user behavior statistics
 * @param {number} days - Number of days to analyze (default: 7)
 * @returns {Promise<Object>} User behavior analytics
 */
export const getUserStats = async (days = 7) => {
  try {
    const response = await fetch(`${STATS_API_BASE}/stats/users?days=${days}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    logger.error('Error fetching user stats:', error);
    throw error;
  }
};

/**
 * Fetch error statistics
 * @returns {Promise<Object>} Error tracking data
 */
export const getErrorStats = async () => {
  try {
    const response = await fetch(`${STATS_API_BASE}/stats/errors`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    logger.error('Error fetching error stats:', error);
    throw error;
  }
};

/**
 * Fetch hourly trends
 * @param {number} hours - Number of hours to analyze (default: 24)
 * @returns {Promise<Object>} Hourly trend data
 */
export const getHourlyTrends = async (hours = 24) => {
  try {
    const response = await fetch(`${STATS_API_BASE}/stats/hourly?hours=${hours}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    logger.error('Error fetching hourly trends:', error);
    throw error;
  }
};

/**
 * Export statistics to CSV
 * @param {string} format - Export format ('json' or 'csv')
 * @returns {Promise<Blob>} Export file blob
 */
export const exportStats = async (format = 'json') => {
  try {
    const response = await fetch(`${STATS_API_BASE}/stats/export?format=${format}`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.blob();
  } catch (error) {
    logger.error('Error exporting stats:', error);
    throw error;
  }
};

/**
 * Create WebSocket connection for real-time stats streaming
 * @param {Function} onMessage - Callback for incoming messages
 * @param {Function} onError - Callback for errors
 * @returns {WebSocket} WebSocket connection
 */
export const createStatsWebSocket = (onMessage, onError) => {
  const wsUrl = BASE_URL.replace('http', 'ws');
  const ws = new WebSocket(`${wsUrl}/api/v1/llm/stats/stream`);
  
  ws.onopen = () => {
    logger.info('📊 Connected to LLM stats stream');
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      logger.error('Error parsing stats message:', error);
      if (onError) onError(error);
    }
  };
  
  ws.onerror = (error) => {
    logger.error('WebSocket error:', error);
    if (onError) onError(error);
  };
  
  ws.onclose = () => {
    logger.info('📊 Disconnected from LLM stats stream');
  };
  
  return ws;
};

/**
 * Fetch all statistics in one call
 * @returns {Promise<Object>} Combined statistics object
 */
export const getAllStats = async () => {
  try {
    const [general, signals, performance, cache, users] = await Promise.all([
      getGeneralStats(),
      getSignalStats(),
      getPerformanceStats(),
      getCacheStats(),
      getUserStats()
    ]);
    
    return {
      general,
      signals,
      performance,
      cache,
      users,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    logger.error('Error fetching all stats:', error);
    throw error;
  }
};

export default {
  getGeneralStats,
  getSignalStats,
  getPerformanceStats,
  getCacheStats,
  getUserStats,
  getErrorStats,
  getHourlyTrends,
  exportStats,
  createStatsWebSocket,
  getAllStats
};
