/**
 * Chat Service for Pure LLM Backend (Llama 3.1 8B)
 * Handles communication with the FastAPI backend on port 8002
 */

// API Configuration
const PURE_LLM_BASE_URL = import.meta.env.VITE_PURE_LLM_API_URL || 'http://localhost:8002';
const CHAT_ENDPOINT = `${PURE_LLM_BASE_URL}/api/chat`;
const HEALTH_ENDPOINT = `${PURE_LLM_BASE_URL}/health`;

// Session Management
export const generateSessionId = () => {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
};

export const getSessionId = () => {
  let sessionId = sessionStorage.getItem('llm_chat_session_id');
  if (!sessionId) {
    sessionId = generateSessionId();
    sessionStorage.setItem('llm_chat_session_id', sessionId);
  }
  return sessionId;
};

export const clearSession = () => {
  sessionStorage.removeItem('llm_chat_session_id');
  sessionStorage.removeItem('llm_chat_history');
};

/**
 * Send a message to the Pure LLM backend
 * @param {string} message - User's message
 * @param {string} sessionId - Session ID (optional)
 * @param {string} language - Language code ('en' or 'tr')
 * @returns {Promise<Object>} Response from the LLM
 */
export async function sendMessage(message, sessionId = null, language = 'en') {
  try {
    if (!message || message.trim().length === 0) {
      throw new Error('Message cannot be empty');
    }

    const requestBody = {
      message: message.trim(),
      session_id: sessionId || getSessionId(),
      language: language
    };

    console.log('📤 Sending message to Pure LLM:', {
      endpoint: CHAT_ENDPOINT,
      messageLength: message.length,
      sessionId: requestBody.session_id,
      language
    });

    const startTime = Date.now();
    
    const response = await fetch(CHAT_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    const responseTime = Date.now() - startTime;

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    
    console.log('📥 Received response from Pure LLM:', {
      responseTime: `${responseTime}ms`,
      method: data.method,
      cached: data.cached,
      confidence: data.confidence,
      contextUsed: data.context_used?.length || 0
    });

    // Add response time to metadata
    if (data.metadata) {
      data.metadata.frontend_response_time = responseTime;
    }

    return {
      success: true,
      data,
      responseTime
    };

  } catch (error) {
    console.error('❌ Chat service error:', error);
    
    // Check if it's a network error
    if (!navigator.onLine) {
      return {
        success: false,
        error: 'No internet connection. Please check your network.',
        type: 'NETWORK_ERROR'
      };
    }

    // Check if backend is unreachable
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      return {
        success: false,
        error: 'Cannot connect to AI backend. Please ensure the server is running on port 8002.',
        type: 'CONNECTION_ERROR'
      };
    }

    return {
      success: false,
      error: error.message,
      type: 'API_ERROR'
    };
  }
}

/**
 * Check if the Pure LLM backend is healthy
 * @returns {Promise<Object>} Health status
 */
export async function checkBackendHealth() {
  try {
    const response = await fetch(HEALTH_ENDPOINT, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }

    const data = await response.json();
    
    console.log('✅ Backend health check:', data);

    return {
      success: true,
      data,
      isHealthy: data.status === 'healthy'
    };

  } catch (error) {
    console.error('❌ Health check failed:', error);
    return {
      success: false,
      isHealthy: false,
      error: error.message
    };
  }
}

/**
 * Save chat history to session storage
 * @param {Array} messages - Array of chat messages
 */
export function saveChatHistory(messages) {
  try {
    sessionStorage.setItem('llm_chat_history', JSON.stringify(messages));
  } catch (error) {
    console.error('Failed to save chat history:', error);
  }
}

/**
 * Load chat history from session storage
 * @returns {Array} Array of chat messages
 */
export function loadChatHistory() {
  try {
    const history = sessionStorage.getItem('llm_chat_history');
    return history ? JSON.parse(history) : [];
  } catch (error) {
    console.error('Failed to load chat history:', error);
    return [];
  }
}

/**
 * Clear chat history from session storage
 */
export function clearChatHistory() {
  try {
    sessionStorage.removeItem('llm_chat_history');
  } catch (error) {
    console.error('Failed to clear chat history:', error);
  }
}

/**
 * Format a message for display
 * @param {Object} message - Message object
 * @returns {Object} Formatted message
 */
export function formatMessage(message) {
  return {
    id: message.id || Date.now() + Math.random(),
    text: message.text || message.response || '',
    sender: message.sender || 'ai',
    timestamp: message.timestamp || new Date().toISOString(),
    metadata: message.metadata || {},
    suggestions: message.suggestions || [],
    cached: message.cached || false
  };
}

/**
 * Get cached response if available
 * @param {string} message - User's message
 * @returns {Object|null} Cached response or null
 */
export function getCachedResponse(message) {
  try {
    const cache = JSON.parse(sessionStorage.getItem('llm_response_cache') || '{}');
    const normalizedMessage = message.toLowerCase().trim();
    
    if (cache[normalizedMessage]) {
      const cachedItem = cache[normalizedMessage];
      // Check if cache is still valid (1 hour)
      const cacheAge = Date.now() - cachedItem.timestamp;
      if (cacheAge < 3600000) { // 1 hour in milliseconds
        console.log('✨ Using cached response');
        return cachedItem.response;
      }
    }
    return null;
  } catch (error) {
    console.error('Failed to get cached response:', error);
    return null;
  }
}

/**
 * Cache a response
 * @param {string} message - User's message
 * @param {Object} response - API response
 */
export function cacheResponse(message, response) {
  try {
    const cache = JSON.parse(sessionStorage.getItem('llm_response_cache') || '{}');
    const normalizedMessage = message.toLowerCase().trim();
    
    cache[normalizedMessage] = {
      response,
      timestamp: Date.now()
    };
    
    // Keep only last 50 cached responses
    const entries = Object.entries(cache);
    if (entries.length > 50) {
      const sorted = entries.sort((a, b) => b[1].timestamp - a[1].timestamp);
      const limited = Object.fromEntries(sorted.slice(0, 50));
      sessionStorage.setItem('llm_response_cache', JSON.stringify(limited));
    } else {
      sessionStorage.setItem('llm_response_cache', JSON.stringify(cache));
    }
  } catch (error) {
    console.error('Failed to cache response:', error);
  }
}

export { checkBackendHealth };

export default {
  sendMessage,
  checkBackendHealth,
  getSessionId,
  generateSessionId,
  clearSession,
  saveChatHistory,
  loadChatHistory,
  clearChatHistory,
  formatMessage,
  getCachedResponse,
  cacheResponse
};
