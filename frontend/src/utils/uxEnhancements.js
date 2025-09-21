/**
 * UX Enhancement Utilities
 * Helper functions for integrating UX enhancements with the existing API
 */

/**
 * Simulates typing for AI responses based on response source
 * @param {string} text - The response text
 * @param {string} source - Response source ('ai', 'cache', 'fallback')
 * @param {function} onChunk - Callback for each typed chunk
 * @param {function} onComplete - Callback when typing is complete
 */
export const simulateTyping = (text, source = 'ai', onChunk, onComplete) => {
  const speeds = {
    ai: { base: 50, variation: 30 },      // Slower, more human-like
    cache: { base: 20, variation: 10 },   // Fast, since it's cached
    fallback: { base: 30, variation: 15 } // Medium speed
  };

  const config = speeds[source] || speeds.ai;
  
  if (source === 'cache') {
    // Show cached responses immediately
    onChunk && onChunk(text);
    onComplete && onComplete();
    return;
  }

  // Word-by-word typing for better readability
  const words = text.split(' ');
  let currentIndex = 0;
  let displayText = '';

  const typeNextWord = () => {
    if (currentIndex >= words.length) {
      onComplete && onComplete();
      return;
    }

    displayText += (displayText ? ' ' : '') + words[currentIndex];
    onChunk && onChunk(displayText);
    currentIndex++;

    setTimeout(typeNextWord, config.base + Math.random() * config.variation);
  };

  typeNextWord();
};

/**
 * Determines appropriate loading skeleton based on query type
 * @param {string} query - User query
 * @returns {string} - Skeleton type ('restaurant', 'museum', 'blog', 'search')
 */
export const getSkeletonType = (query) => {
  const lowerQuery = query.toLowerCase();
  
  if (lowerQuery.includes('restaurant') || lowerQuery.includes('food') || lowerQuery.includes('eat')) {
    return 'restaurant';
  }
  
  if (lowerQuery.includes('museum') || lowerQuery.includes('history') || lowerQuery.includes('culture')) {
    return 'museum';
  }
  
  if (lowerQuery.includes('blog') || lowerQuery.includes('article') || lowerQuery.includes('post')) {
    return 'blog';
  }
  
  return 'search';
};

/**
 * Enhanced API wrapper with UX improvements
 */
export class EnhancedAPI {
  constructor(baseUrl = 'http://localhost:8001') {
    this.baseUrl = baseUrl;
  }

  /**
   * Enhanced chat request with typing simulation
   * @param {string} message - User message
   * @param {string} sessionId - Session ID
   * @param {function} onTypingChunk - Callback for typing chunks
   * @param {function} onComplete - Callback when response is complete
   */
  async sendMessage(message, sessionId, onTypingChunk, onComplete) {
    try {
      const response = await fetch(`${this.baseUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          session_id: sessionId
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Simulate typing based on response source
      simulateTyping(
        data.response,
        data.source || 'ai',
        onTypingChunk,
        () => onComplete && onComplete(data)
      );

      return data;
    } catch (error) {
      console.error('API Error:', error);
      
      // Fallback response with typing
      const fallbackResponse = {
        response: "I apologize, but I'm having trouble connecting right now. Please try again in a moment, or feel free to ask me about Istanbul's restaurants, museums, or attractions!",
        source: 'fallback',
        timestamp: new Date().toISOString(),
        session_id: sessionId
      };

      simulateTyping(
        fallbackResponse.response,
        'fallback',
        onTypingChunk,
        () => onComplete && onComplete(fallbackResponse)
      );

      return fallbackResponse;
    }
  }

  /**
   * Search with appropriate loading states
   * @param {string} query - Search query
   * @param {string} type - Search type ('restaurants', 'museums', 'places')
   */
  async search(query, type = 'places') {
    const endpoint = type === 'restaurants' ? '/restaurants/search' : `/places/${type}`;
    
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}?q=${encodeURIComponent(query)}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Search Error:', error);
      return { error: 'Search failed', results: [] };
    }
  }
}

/**
 * Message queue for managing typing animations
 */
export class MessageQueue {
  constructor() {
    this.queue = [];
    this.isProcessing = false;
  }

  /**
   * Add message to typing queue
   * @param {object} message - Message object
   * @param {function} onComplete - Completion callback
   */
  addMessage(message, onComplete) {
    this.queue.push({ message, onComplete });
    this.processQueue();
  }

  /**
   * Process the message queue sequentially
   */
  async processQueue() {
    if (this.isProcessing || this.queue.length === 0) {
      return;
    }

    this.isProcessing = true;

    while (this.queue.length > 0) {
      const { message, onComplete } = this.queue.shift();
      
      await new Promise((resolve) => {
        simulateTyping(
          message.content,
          message.source,
          onComplete,
          resolve
        );
      });
    }

    this.isProcessing = false;
  }

  /**
   * Clear the queue
   */
  clear() {
    this.queue = [];
    this.isProcessing = false;
  }
}

/**
 * Performance monitoring for UX features
 */
export class UXPerformanceMonitor {
  constructor() {
    this.metrics = {
      typingDuration: [],
      loadingDuration: [],
      responseDelay: []
    };
  }

  /**
   * Start timing an operation
   * @param {string} operation - Operation name
   */
  startTiming(operation) {
    this.startTimes = this.startTimes || {};
    this.startTimes[operation] = performance.now();
  }

  /**
   * End timing and record metric
   * @param {string} operation - Operation name
   */
  endTiming(operation) {
    if (!this.startTimes || !this.startTimes[operation]) {
      return;
    }

    const duration = performance.now() - this.startTimes[operation];
    
    if (this.metrics[operation]) {
      this.metrics[operation].push(duration);
    }

    delete this.startTimes[operation];
  }

  /**
   * Get performance summary
   */
  getSummary() {
    const summary = {};
    
    for (const [operation, times] of Object.entries(this.metrics)) {
      if (times.length > 0) {
        summary[operation] = {
          count: times.length,
          average: times.reduce((a, b) => a + b, 0) / times.length,
          min: Math.min(...times),
          max: Math.max(...times)
        };
      }
    }

    return summary;
  }
}

/**
 * Local storage utilities for UX preferences
 */
export const UXPreferences = {
  /**
   * Get user's typing animation preference
   */
  getTypingEnabled() {
    return localStorage.getItem('ux_typing_enabled') !== 'false';
  },

  /**
   * Set typing animation preference
   * @param {boolean} enabled - Whether typing is enabled
   */
  setTypingEnabled(enabled) {
    localStorage.setItem('ux_typing_enabled', enabled.toString());
  },

  /**
   * Get typing speed preference
   */
  getTypingSpeed() {
    return parseInt(localStorage.getItem('ux_typing_speed') || '50');
  },

  /**
   * Set typing speed preference
   * @param {number} speed - Typing speed in milliseconds
   */
  setTypingSpeed(speed) {
    localStorage.setItem('ux_typing_speed', speed.toString());
  },

  /**
   * Get skeleton animation preference
   */
  getSkeletonsEnabled() {
    return localStorage.getItem('ux_skeletons_enabled') !== 'false';
  },

  /**
   * Set skeleton animation preference
   * @param {boolean} enabled - Whether skeletons are enabled
   */
  setSkeletonsEnabled(enabled) {
    localStorage.setItem('ux_skeletons_enabled', enabled.toString());
  }
};

/**
 * Records user interaction for analytics
 * @param {string} event - The interaction event
 * @param {object} data - Additional data about the interaction
 */
export const recordUserInteraction = (event, data = {}) => {
  // In development, just log to console
  if (import.meta.env.DEV) {
    console.log('User Interaction:', event, data);
  }
  
  // Could integrate with analytics service later
  try {
    const interaction = {
      event,
      data,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };
    
    // Store in localStorage for now
    const interactions = JSON.parse(localStorage.getItem('userInteractions') || '[]');
    interactions.push(interaction);
    
    // Keep only last 100 interactions
    if (interactions.length > 100) {
      interactions.splice(0, interactions.length - 100);
    }
    
    localStorage.setItem('userInteractions', JSON.stringify(interactions));
  } catch (error) {
    console.warn('Failed to record user interaction:', error);
  }
};

/**
 * Measures API response time
 * @param {function} apiCall - The API call function
 * @returns {Promise} - Promise with timing data
 */
export const measureApiResponseTime = async (apiCall) => {
  const startTime = performance.now();
  
  try {
    const result = await apiCall();
    const endTime = performance.now();
    const responseTime = endTime - startTime;
    
    recordUserInteraction('api_response_time', {
      responseTime: Math.round(responseTime),
      success: true
    });
    
    return {
      result,
      responseTime,
      success: true
    };
  } catch (error) {
    const endTime = performance.now();
    const responseTime = endTime - startTime;
    
    recordUserInteraction('api_response_time', {
      responseTime: Math.round(responseTime),
      success: false,
      error: error.message
    });
    
    return {
      result: null,
      responseTime,
      success: false,
      error
    };
  }
};

export default {
  simulateTyping,
  getSkeletonType,
  EnhancedAPI,
  MessageQueue,
  UXPerformanceMonitor,
  UXPreferences,
  recordUserInteraction,
  measureApiResponseTime
};
