import { 
  fetchWithRetry, 
  getUserFriendlyMessage, 
  classifyError, 
  ErrorTypes,
  networkStatus,
  createCircuitBreaker,
  debounce
} from '../utils/errorHandler.js';

// API utility that works for both local and deployed environments
// Updated: Backend migrated from Render to GCP Cloud Run
const BASE_URL = import.meta.env.VITE_API_URL || (typeof window !== 'undefined' && window.location.hostname === 'localhost' ? 'http://localhost:8001' : 'https://ai-stanbul-509659445005.europe-west1.run.app');

// Clean up BASE_URL and construct proper endpoints
const cleanBaseUrl = BASE_URL.replace(/\/$/, ''); // Remove trailing slash

// Pure LLM API endpoints
const API_URL = `${cleanBaseUrl}/api/chat`;  // Pure LLM chat endpoint
// STREAMING NOT IMPLEMENTED IN BACKEND YET
// const STREAM_API_URL = `${cleanBaseUrl}/api/stream`;  // Streaming endpoint
const RESTAURANTS_API_URL = `${cleanBaseUrl}/api/v2/restaurants`; // ✅ Fixed: correct endpoint
const PLACES_API_URL = `${cleanBaseUrl}/api/places`;
// Chat history endpoints  
const CHAT_HISTORY_API_URL = `${cleanBaseUrl}/api/chat/history`;

// Session management
export const generateSessionId = () => {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
};

export const getSessionId = () => {
  let sessionId = localStorage.getItem('chat_session_id');
  if (!sessionId) {
    sessionId = generateSessionId();
    localStorage.setItem('chat_session_id', sessionId);
  }
  return sessionId;
};

export const clearSession = () => {
  localStorage.removeItem('chat_session_id');
  localStorage.removeItem('chat-messages');
};

console.log('API Configuration:', {
  BASE_URL: cleanBaseUrl,
  API_URL,
  // STREAM_API_URL,  // Not implemented in backend
  RESTAURANTS_API_URL,
  PLACES_API_URL,
  CHAT_HISTORY_API_URL,
  VITE_API_URL: import.meta.env.VITE_API_URL
});

// Circuit breakers for different API endpoints
const chatCircuitBreaker = createCircuitBreaker({
  failureThreshold: 3,
  resetTimeout: 30000 // 30 seconds
});

const restaurantsCircuitBreaker = createCircuitBreaker({
  failureThreshold: 3,
  resetTimeout: 30000
});

const placesCircuitBreaker = createCircuitBreaker({
  failureThreshold: 3,
  resetTimeout: 30000
});

// Enhanced error handling wrapper
const handleApiError = (error, response = null, context = '') => {
  const errorType = classifyError(error, response);
  const userMessage = getUserFriendlyMessage(error, response);
  
  console.error(`${context} error:`, {
    message: error.message,
    type: errorType,
    userMessage,
    status: response?.status,
    online: navigator.onLine
  });
  
  // Create enhanced error object
  const enhancedError = new Error(userMessage);
  enhancedError.originalError = error;
  enhancedError.type = errorType;
  enhancedError.response = response;
  enhancedError.isRetryable = [ErrorTypes.NETWORK, ErrorTypes.TIMEOUT, ErrorTypes.SERVER].includes(errorType);
  
  return enhancedError;
};

export const fetchResults = async (query, sessionId = null) => {
  return chatCircuitBreaker.call(async () => {
    try {
      console.log('🚀 Making chat API request to:', API_URL, 'with query:', query);
      
      const requestBody = { 
        message: query 
      };
      
      // Add session ID if provided
      if (sessionId) {
        requestBody.session_id = sessionId;
      }
      
      const response = await fetchWithRetry(API_URL, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestBody),
        timeout: 30000 // 30 second timeout
      }, {
        maxAttempts: 3,
        baseDelay: 1000
      });
      
      const data = await response.json();
      console.log('✅ Chat API response data:', data);
      return data;
      
    } catch (error) {
      throw handleApiError(error, null, 'Chat API');
    }
  });
};

/**
 * Chat endpoint with GPS support and map visualization
 * This endpoint provides:
 * - GPS location support for "my location" queries
 * - Map visualization data for transportation queries
 * - Session management
 */
export const fetchUnifiedChat = async (query, options = {}) => {
  return chatCircuitBreaker.call(async () => {
    try {
      const sessionId = options.sessionId || getSessionId();
      const userId = options.userId || 'anonymous';
      const gpsLocation = options.gpsLocation || null;
      
      console.log('🎯 Making chat API request:', {
        url: API_URL,
        query: query.substring(0, 50) + '...',
        sessionId,
        hasGPS: !!gpsLocation
      });
      
      const requestBody = { 
        message: query,
        session_id: sessionId,
        user_id: userId
      };
      
      // Add GPS location if available (backend expects user_location)
      if (gpsLocation) {
        requestBody.user_location = gpsLocation;
      }
      
      const response = await fetchWithRetry(API_URL, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestBody),
        timeout: 30000
      }, {
        maxAttempts: 3,
        baseDelay: 1000
      });
      
      const data = await response.json();
      
      console.log('✅ Chat response:', {
        hasResponse: !!data.response || !!data.message,
        hasMapData: !!data.map_data,
        sessionId: data.session_id
      });
      
      return data;
      
    } catch (error) {
      throw handleApiError(error, null, 'Chat API');
    }
  });
};

export const fetchStreamingResults = async (query, onChunk, onComplete = null, onError = null, locationContext = null) => {
  // STREAMING NOT IMPLEMENTED IN BACKEND YET
  // Fallback to regular chat API
  console.warn('⚠️ Streaming not available, using regular chat endpoint');
  
  try {
    const result = await fetchResults(query);
    if (onChunk) onChunk(result.response || result.message);
    if (onComplete) onComplete(result.response || result.message);
    return result;
  } catch (error) {
    if (onError) onError(error);
    throw error;
  }
};

/* ORIGINAL STREAMING CODE - COMMENTED OUT UNTIL BACKEND IMPLEMENTS IT
export const fetchStreamingResults_ORIGINAL = async (query, onChunk, onComplete = null, onError = null, locationContext = null) => {
  return chatCircuitBreaker.call(async () => {
    try {
      console.log('🌊 Starting streaming request to:', STREAM_API_URL);
      
      // Use provided sessionId or get current session
      const currentSessionId = getSessionId();
      
      const requestBody = { 
        message: query,
        session_id: currentSessionId,  // Always include session ID
        ...(locationContext && { location_context: locationContext }) // Add location context if provided
      };
      
      console.log('📋 Request body:', { ...requestBody, message: query.substring(0, 50) + '...' });
      
      const response = await fetchWithRetry(STREAM_API_URL, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestBody),
        timeout: 60000 // 60 second timeout for streaming
      }, {
        maxAttempts: 2, // Fewer retries for streaming
        baseDelay: 2000
      });
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let completeResponse = '';
      let metadata = null;
      
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') {
                // Call onComplete with full response and metadata if available
                if (onComplete) {
                  onComplete(completeResponse, metadata);
                }
                return;
              }
              
              try {
                const parsed = JSON.parse(data);
                
                // Handle error response
                if (parsed.error) {
                  console.error('Backend streaming error:', parsed.error);
                  onError?.(new Error(parsed.error));
                  return;
                }
                
                // Handle completion signal with metadata
                if (parsed.done) {
                  console.log('✅ Streaming completed');
                  // Extract metadata if present
                  if (parsed.metadata) {
                    metadata = parsed.metadata;
                    console.log('📊 Received metadata:', metadata);
                  }
                  // Call onComplete with full response and metadata
                  if (onComplete) {
                    onComplete(completeResponse, metadata);
                  }
                  return;
                }
                
                // Handle chunk data - backend sends {chunk: "text"}
                if (parsed.chunk) {
                  completeResponse += parsed.chunk;
                  onChunk(parsed.chunk);
                }
                
                // Legacy: Handle OpenAI-style format if present
                if (parsed.delta && parsed.delta.content) {
                  completeResponse += parsed.delta.content;
                  onChunk(parsed.delta.content);
                }
              } catch (e) {
                // Only warn for non-empty data that failed to parse
                if (data.trim()) {
                  console.warn('Failed to parse streaming chunk:', data, e);
                }
              }
            }
          }
        }
        
        // Fallback if stream ends without [DONE] signal
        if (onComplete) {
          onComplete(completeResponse, metadata);
        }
        
      } finally {
        reader.releaseLock();
      }
      
    } catch (error) {
      throw handleApiError(error, null, 'Streaming API');
    }
  });
};
*/

// New chat history management functions
export const fetchChatHistory = async (sessionId) => {
  try {
    console.log('📚 Fetching chat history for session:', sessionId);
    
    const response = await fetchWithRetry(`${CHAT_HISTORY_API_URL}/${sessionId}`, {
      method: 'GET',
      headers: { 
        'Accept': 'application/json'
      },
      timeout: 10000
    }, {
      maxAttempts: 2,
      baseDelay: 500
    });
    
    const data = await response.json();
    console.log('✅ Chat history fetched:', data);
    return data.messages || [];
    
  } catch (error) {
    console.warn('⚠️ Failed to fetch chat history:', error.message);
    return []; // Return empty array if history fetch fails
  }
};

export const clearChatHistory = async (sessionId) => {
  try {
    console.log('🗑️ Clearing chat history for session:', sessionId);
    
    const response = await fetchWithRetry(`${CHAT_HISTORY_API_URL}/${sessionId}`, {
      method: 'DELETE',
      headers: { 
        'Accept': 'application/json'
      },
      timeout: 10000
    }, {
      maxAttempts: 2,
      baseDelay: 500
    });
    
    const data = await response.json();
    console.log('✅ Chat history cleared:', data);
    return data;
    
  } catch (error) {
    throw handleApiError(error, null, 'Clear Chat History');
  }
};

// Helper function to extract location/district from user input
export const extractLocationFromQuery = (userInput) => {
  const input = userInput.toLowerCase();
  
  // Istanbul districts (both Turkish and English variants)
  const districts = {
    'beyoğlu': 'Beyoglu',
    'beyoglu': 'Beyoglu',
    'galata': 'Beyoglu',
    'taksim': 'Beyoglu',
    'sultanahmet': 'Sultanahmet',
    'fatih': 'Fatih',
    'kadıköy': 'Kadikoy',
    'kadikoy': 'Kadikoy',
    'beşiktaş': 'Besiktas',
    'besiktas': 'Besiktas',
    'şişli': 'Sisli',
    'sisli': 'Sisli',
    'üsküdar': 'Uskudar',
    'uskudar': 'Uskudar',
    'ortaköy': 'Besiktas',
    'ortakoy': 'Besiktas',
    'karaköy': 'Beyoglu',
    'karakoy': 'Beyoglu',
    'eminönü': 'Fatih',
    'eminonu': 'Fatih',
    'bakırköy': 'Bakirkoy',
    'bakirkoy': 'Bakirkoy'
  };
  
  // Check for district matches
  for (const [key, value] of Object.entries(districts)) {
    if (input.includes(key)) {
      return { district: value, keyword: null };
    }
  }
  
  // Check for cuisine keywords
  const cuisineKeywords = {
    'turkish': 'Turkish',
    'ottoman': 'Turkish',
    'kebab': 'kebab',
    'seafood': 'seafood',
    'fish': 'seafood',
    'italian': 'Italian',
    'pizza': 'Italian',
    'asian': 'Asian',
    'chinese': 'Chinese',
    'japanese': 'Japanese',
    'sushi': 'Japanese',
    'mediterranean': 'Mediterranean',
    'coffee': 'cafe',
    'cafe': 'cafe'
  };
  
  for (const [key, value] of Object.entries(cuisineKeywords)) {
    if (input.includes(key)) {
      return { district: null, keyword: value };
    }
  }
  
  return { district: null, keyword: null };
};

export const fetchRestaurantRecommendations = async (userInput = '', limit = 4) => {
  return restaurantsCircuitBreaker.call(async () => {
    try {
      console.log('🍽️ fetchRestaurantRecommendations called with userInput:', userInput);
      const { district, keyword } = extractLocationFromQuery(userInput);
      console.log('Extracted filters - District:', district, 'Keyword:', keyword);
      
      const params = new URLSearchParams();
      if (district) params.append('district', district);
      if (keyword) params.append('keyword', keyword);
      params.append('limit', limit.toString());

      const url = `${RESTAURANTS_API_URL}?${params}`;
      console.log('Making restaurant API request to:', url);
      
      const response = await fetchWithRetry(url, {
        method: 'GET',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        timeout: 20000 // 20 second timeout
      }, {
        maxAttempts: 3,
        baseDelay: 1000
      });
      
      const data = await response.json();
      console.log('✅ Restaurant API response data:', data);
      console.log('Number of restaurants returned:', data.restaurants?.length);
      return data;
      
    } catch (error) {
      throw handleApiError(error, null, 'Restaurant API');
    }
  });
};

export const fetchPlacesRecommendations = async (userInput = '', limit = 6) => {
  return placesCircuitBreaker.call(async () => {
    try {
      console.log('🏛️ fetchPlacesRecommendations called with userInput:', userInput);
      const { district, keyword } = extractLocationFromQuery(userInput);
      console.log('Extracted filters - District:', district, 'Keyword:', keyword);
      
      const params = new URLSearchParams();
      if (district) params.append('district', district);
      if (keyword) params.append('keyword', keyword);
      params.append('limit', limit.toString());

      const url = `${PLACES_API_URL}?${params}`;
      console.log('Making places API request to:', url);
      
      const response = await fetchWithRetry(url, {
        method: 'GET',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        timeout: 20000 // 20 second timeout
      }, {
        maxAttempts: 3,
        baseDelay: 1000
      });
      
      const data = await response.json();
      console.log('✅ Places API response data:', data);
      console.log('Number of places returned:', data.length);
      return { places: data }; // Wrap in places object to match expected format
      
    } catch (error) {
      throw handleApiError(error, null, 'Places API');
    }
  });
};

// Debounced versions for rapid successive calls
export const debouncedFetchRestaurants = debounce(fetchRestaurantRecommendations, 300);
export const debouncedFetchPlaces = debounce(fetchPlacesRecommendations, 300);

// GPS Journey Planning API
export const planJourneyFromGPS = async (gpsLat, gpsLng, destination, options = {}) => {
  try {
    const {
      maxWalkingM = 1000,
      minimizeTransfers = true
    } = options;

    const requestBody = {
      gps_lat: gpsLat,
      gps_lng: gpsLng,
      destination: destination,
      max_walking_m: maxWalkingM,
      minimize_transfers: minimizeTransfers
    };

    const url = `${cleanBaseUrl}/api/route/from-gps`;
    console.log('🚇 Making GPS journey planning request to:', url, 'with body:', requestBody);

    const response = await fetchWithRetry(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(requestBody),
      timeout: 20000
    }, {
      maxAttempts: 3,
      baseDelay: 1000
    });

    const data = await response.json();
    console.log('✅ GPS journey planning response:', data);

    if (data.success && data.journey) {
      return data.journey;
    } else {
      throw new Error(data.error || 'No route found');
    }
  } catch (error) {
    throw handleApiError(error, null, 'GPS Journey Planning API');
  }
};

// Health check utility
export const checkApiHealth = async () => {
  try {
    const healthUrl = `${cleanBaseUrl}/api/health`;
    const response = await fetchWithRetry(healthUrl, {
      method: 'GET',
      timeout: 5000
    }, {
      maxAttempts: 1 // No retries for health checks
    });
    
    return { 
      healthy: response.ok, 
      status: response.status,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    return { 
      healthy: false, 
      error: error.message,
      timestamp: new Date().toISOString()
    };
  }
};

// Network status monitoring
let networkStatusSubscription = null;

export const subscribeToNetworkStatus = (callback) => {
  networkStatusSubscription = callback;
  networkStatus.addListener(callback);
  
  // Return unsubscribe function
  return () => {
    if (networkStatusSubscription) {
      networkStatus.removeListener(networkStatusSubscription);
      networkStatusSubscription = null;
    }
  };
};

// ============================================
// Pure LLM Backend Integration (Llama 3.1 8B)
// ============================================

const PURE_LLM_BASE_URL = import.meta.env.VITE_PURE_LLM_API_URL || 'http://localhost:8002';
const PURE_LLM_CHAT_URL = `${PURE_LLM_BASE_URL}/api/chat`;
const PURE_LLM_HEALTH_URL = `${PURE_LLM_BASE_URL}/health`;

/**
 * Send message to Pure LLM backend (Llama 3.1 8B)
 * @param {string} message - User's message
 * @param {Object} options - Options object
 * @param {string} options.sessionId - Session ID
 * @param {string} options.language - Language code ('en', 'tr', 'fr', 'ru', 'de', 'ar')
 * @param {Object} options.gpsLocation - GPS location {lat, lon}
 * @returns {Promise<Object>} Response from LLM
 */
export const fetchPureLLMChat = async (message, options = {}) => {
  return chatCircuitBreaker.call(async () => {
    try {
      const sessionId = options.sessionId || getSessionId();
      const language = options.language || 'en';
      const gpsLocation = options.gpsLocation || null;
      
      console.log('🦙 Making Pure LLM request:', {
        url: PURE_LLM_CHAT_URL,
        messageLength: message.length,
        sessionId,
        language,
        hasGPS: !!gpsLocation
      });
      
      const requestBody = {
        message: message.trim(),
        session_id: sessionId,
        language: language
      };
      
      // Add GPS location if available
      if (gpsLocation) {
        requestBody.user_location = {
          lat: gpsLocation.lat || gpsLocation.latitude,
          lon: gpsLocation.lon || gpsLocation.lng || gpsLocation.longitude
        };
        console.log('📍 Including GPS location:', requestBody.user_location);
      }
      
      const startTime = Date.now();
      
      const response = await fetchWithRetry(PURE_LLM_CHAT_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestBody),
        timeout: 60000 // 60 second timeout for LLM
      }, {
        maxAttempts: 2,
        baseDelay: 2000
      });
      
      const data = await response.json();
      const responseTime = Date.now() - startTime;
      
      console.log('✅ Pure LLM response:', {
        responseTime: `${responseTime}ms`,
        method: data.method,
        cached: data.cached,
        confidence: data.confidence,
        contextCount: data.context_used?.length || 0,
        hasMapData: !!data.map_data
      });
      
      // Add frontend response time to metadata
      if (data.metadata) {
        data.metadata.frontend_response_time = responseTime;
      }
      
      return {
        success: true,
        data,
        responseTime
      };
      
    } catch (error) {
      console.error('❌ Pure LLM error:', error);
      throw handleApiError(error, null, 'Pure LLM Chat');
    }
  });
};

/**
 * Check Pure LLM backend health
 * @returns {Promise<Object>} Health status
 */
export const checkPureLLMHealth = async () => {
  try {
    const response = await fetchWithRetry(PURE_LLM_HEALTH_URL, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 5000
    }, {
      maxAttempts: 1
    });
    
    const data = await response.json();
    
    console.log('✅ Pure LLM health check:', data);
    
    return {
      healthy: response.ok && data.status === 'healthy',
      data,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('❌ Pure LLM health check failed:', error);
    return {
      healthy: false,
      error: error.message,
      timestamp: new Date().toISOString()
    };
  }
};

/**
 * Unified chat function that can switch between backends
 * Use this as a drop-in replacement for fetchUnifiedChat
 * @param {string} query - User's query
 * @param {Object} options - Options object
 * @param {boolean} options.usePureLLM - Use Pure LLM backend (default: false)
 * @returns {Promise<Object>} Chat response
 */
export const fetchUnifiedChatV2 = async (query, options = {}) => {
  const usePureLLM = options.usePureLLM || false;
  
  if (usePureLLM) {
    // Use Pure LLM backend
    const result = await fetchPureLLMChat(query, options);
    
    // Transform response to match expected format
    return {
      response: result.data.response,
      message: result.data.response,
      session_id: result.data.session_id,
      method: result.data.method,
      cached: result.data.cached,
      confidence: result.data.confidence,
      suggestions: result.data.suggestions || [],
      metadata: result.data.metadata,
      context_used: result.data.context_used || [],
      response_time: result.responseTime,
      map_data: result.data.map_data, // Include map data from backend
      route_data: result.data.route_data, // 🔥 WEEK 2: Include route_data for TransportationRouteCard
      llm_mode: result.data.llm_mode, // 🔥 WEEK 2: Include llm_mode for conditional rendering
      intent: result.data.intent // 🔥 WEEK 2: Include intent classification
    };
  } else {
    // Use original backend
    return fetchUnifiedChat(query, options);
  }
};

// ============================================
// Streaming Chat API (Real-time responses)
// ============================================

const STREAM_API_URL = `${cleanBaseUrl}/api/stream`;

/**
 * Stream chat response using Server-Sent Events (SSE)
 * Provides real-time word-by-word response like ChatGPT
 * 
 * @param {string} message - User's message
 * @param {Object} options - Options object
 * @param {Function} options.onToken - Callback for each token received
 * @param {Function} options.onComplete - Callback when streaming completes
 * @param {Function} options.onError - Callback on error
 * @param {string} options.sessionId - Session ID
 * @param {string} options.language - Language code
 * @param {Object} options.gpsLocation - GPS location
 * @returns {Promise<void>}
 */
export const fetchStreamingChat = async (message, options = {}) => {
  const {
    onToken,
    onComplete,
    onError,
    onStart,
    sessionId = getSessionId(),
    language = 'en',
    gpsLocation = null
  } = options;

  try {
    console.log('🌊 Starting streaming chat request');

    const requestBody = {
      message: message.trim(),
      session_id: sessionId,
      language: language,
      include_context: true
    };

    if (gpsLocation) {
      requestBody.user_location = {
        lat: gpsLocation.lat || gpsLocation.latitude,
        lon: gpsLocation.lon || gpsLocation.lng || gpsLocation.longitude
      };
    }

    const response = await fetch(`${STREAM_API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullResponse = '';
    let metadata = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          const eventType = line.slice(7).trim();
          continue;
        }

        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          
          try {
            const parsed = JSON.parse(data);

            if (parsed.timestamp && !parsed.content && onStart) {
              // Start event
              onStart(parsed);
            } else if (parsed.content !== undefined && !parsed.metadata) {
              // Token event
              fullResponse += parsed.content;
              if (onToken) {
                onToken(parsed.content, fullResponse);
              }
            } else if (parsed.metadata) {
              // Complete event
              metadata = parsed.metadata;
              if (onComplete) {
                onComplete(parsed.content || fullResponse, metadata);
              }
            } else if (parsed.error) {
              // Error event
              if (onError) {
                onError(new Error(parsed.error));
              }
            }
          } catch (e) {
            // Non-JSON data, treat as raw token
            if (data.trim()) {
              fullResponse += data;
              if (onToken) {
                onToken(data, fullResponse);
              }
            }
          }
        }
      }
    }

    // If no complete event was received, call onComplete with accumulated response
    if (!metadata && onComplete) {
      onComplete(fullResponse, { streaming: true });
    }

    return { response: fullResponse, metadata };

  } catch (error) {
    console.error('❌ Streaming error:', error);
    if (onError) {
      onError(error);
    }
    throw error;
  }
};

/**
 * Check if streaming is available
 * @returns {Promise<boolean>}
 */
export const checkStreamingAvailable = async () => {
  try {
    const response = await fetch(`${STREAM_API_URL}/analyze?text=test`, {
      method: 'GET',
      timeout: 5000
    });
    return response.ok;
  } catch (error) {
    console.warn('Streaming not available:', error.message);
    return false;
  }
};
