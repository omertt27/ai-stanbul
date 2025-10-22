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
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Clean up BASE_URL and construct proper endpoints
const cleanBaseUrl = BASE_URL.replace(/\/$/, ''); // Remove trailing slash

// Correct API endpoints - backend uses /ai/chat not /ai/ai/chat
const API_URL = `${cleanBaseUrl}/ai/chat`;  // Fixed: direct path to chat endpoint
const STREAM_API_URL = `${cleanBaseUrl}/ai/stream`;
const RESTAURANTS_API_URL = `${cleanBaseUrl}/restaurants/search`;
const PLACES_API_URL = `${cleanBaseUrl}/places/`;
// Chat history endpoints  
const CHAT_HISTORY_API_URL = `${cleanBaseUrl}/chat/history`;

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
  STREAM_API_URL,
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

export const fetchStreamingResults = async (query, onChunk, onComplete = null, onError = null, locationContext = null) => {
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

// Health check utility
export const checkApiHealth = async () => {
  try {
    const healthUrl = `${cleanBaseUrl}/health`;
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
