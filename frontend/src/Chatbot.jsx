import { useState, useEffect } from 'react';
import { 
  fetchStreamingResults, 
  fetchRestaurantRecommendations, 
  fetchPlacesRecommendations, 
  extractLocationFromQuery,
  subscribeToNetworkStatus,
  checkApiHealth,
  debouncedFetchRestaurants,
  debouncedFetchPlaces
} from './api/api';
import { 
  ErrorTypes, 
  classifyError, 
  getUserFriendlyMessage,
  networkStatus 
} from './utils/errorHandler';
import ErrorNotification, { NetworkStatusIndicator, RetryButton } from './components/ErrorNotification';
import TypingIndicator from './components/TypingIndicator';
import MessageActions from './components/MessageActions';
import ScrollToBottom from './components/ScrollToBottom';
import ChatHeader from './components/ChatHeader';
import POICard from './components/POICard';
import DistrictInfo from './components/DistrictInfo';
import ItineraryTimeline from './components/ItineraryTimeline';
import MLInsights from './components/MLInsights';

console.log('🔄 Chatbot component loaded with restaurant functionality and comprehensive error handling');

// Input security and normalization functions - ENHANCED SECURITY
const sanitizeInput = (input) => {
  if (!input || typeof input !== 'string') return '';
  
  // Remove potential SQL injection patterns - ENHANCED
  const sqlPatterns = [
    /[';]/g,                              // Remove semicolons
    /--/g,                                // Remove SQL comments  
    /\/\*/g,                              // Remove SQL block comments start
    /\*\//g,                              // Remove SQL block comments end
    /\b(UNION|SELECT|DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|DECLARE)\b/gi, // SQL keywords - expanded
    /\b(OR|AND)\s+['"]\d+['"]?\s*=\s*['"]\d+['"]/gi, // OR '1'='1' patterns
    /['"]\s*(OR|AND)\s+['"]/gi,          // Injection connector patterns
  ];
  
  // Remove XSS patterns - ENHANCED
  const xssPatterns = [
    /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, // <script> tags
    /<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, // <iframe> tags
    /<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>/gi, // <object> tags
    /<embed\b[^>]*>/gi,                   // <embed> tags
    /<link\b[^>]*>/gi,                    // <link> tags
    /<meta\b[^>]*>/gi,                    // <meta> tags
    /<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi,   // <style> tags
    /<[^>]*>/g,                           // Remove all HTML tags
    /javascript:/gi,                      // Remove javascript: protocol
    /on\w+\s*=/gi,                        // Remove event handlers (onclick, etc.)
    /expression\s*\(/gi,                  // Remove CSS expression() calls
    /url\s*\(/gi,                         // Remove CSS url() calls
  ];

  let sanitized = input;

  // Apply all SQL injection sanitization
  sqlPatterns.forEach(pattern => {
    sanitized = sanitized.replace(pattern, '');
  });

  // Apply all XSS sanitization
  xssPatterns.forEach(pattern => {
    sanitized = sanitized.replace(pattern, '');
  });

  // Additional character filtering - keep most international chars but remove dangerous ones
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, ''); // Control characters
  
  // Trim and limit length
  sanitized = sanitized.trim().substring(0, 1000);
  
  return sanitized;
};

const addFuzzyMatching = (text) => {
  if (!text || typeof text !== 'string') return '';
  
  const corrections = {
    'istambul': 'istanbul',
    'instanbul': 'istanbul', 
    'stanbul': 'istanbul',
    'galata': 'galata',
    'galata tower': 'galata tower',
    'sultanahmat': 'sultanahmet',
    'sultuanahmet': 'sultanahmet',
    'taksim': 'taksim',
    'kadikoy': 'kadıköy',
    'beyoglu': 'beyoğlu',
    'besiktas': 'beşiktaş',
    'uskudar': 'üsküdar',
    'ortakoy': 'ortaköy',
    'karakoy': 'karaköy'
  };
  
  let corrected = text.toLowerCase();
  
  Object.entries(corrections).forEach(([wrong, right]) => {
    const regex = new RegExp(`\\b${wrong}\\b`, 'gi');
    corrected = corrected.replace(regex, right);
  });
  
  return corrected;
};

const normalizeText = (text) => {
  if (!text || typeof text !== 'string') return '';
  
  const normalized = text
    .toLowerCase()
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/[^\w\s-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  
  return normalized;
};

// Enhanced preprocessing function
const preprocessInput = (userInput) => {
  if (!userInput || typeof userInput !== 'string') return '';
  
  console.log('🔒 Original input:', userInput);
  
  // Step 1: Security sanitization FIRST
  const sanitized = sanitizeInput(userInput);
  console.log('🛡️ Sanitized:', sanitized);
  
  // Step 2: Text normalization
  const normalized = normalizeText(sanitized);
  console.log('📏 Normalized:', normalized);
  
  // Step 3: Fix common typos
  const corrected = addFuzzyMatching(normalized);
  console.log('✏️ Typo-corrected:', corrected);
  
  return corrected;
};

// Helper function to render text with clickable links
const renderMessageContent = (content, darkMode) => {
  // Convert Markdown-style links [text](url) to clickable HTML links
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  
  console.log('Rendering content:', content.substring(0, 100) + '...');
  
  const parts = [];
  let lastIndex = 0;
  let match;
  
  while ((match = linkRegex.exec(content)) !== null) {
    const linkText = match[1];
    const linkUrl = match[2];
    
    console.log('Found link:', linkText, '->', linkUrl);
    
    // Add text before the link
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {content.substring(lastIndex, match.index)}
        </span>
      );
    }
    
    // Add the clickable link
    parts.push(
      <a
        key={`link-${match.index}`}
        href={linkUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={`underline transition-colors duration-200 hover:opacity-80 cursor-pointer ${
          darkMode 
            ? 'text-blue-400 hover:text-blue-300' 
            : 'text-blue-600 hover:text-blue-700'
        }`}
        onClick={(e) => {
          console.log('Link clicked:', linkUrl);
        }}
      >
        {linkText}
      </a>
    );
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < content.length) {
    parts.push(
      <span key={`text-${lastIndex}`}>
        {content.substring(lastIndex)}
      </span>
    );
  }
  
  return parts.length > 0 ? parts : content;
};

// Format restaurant recommendations
const formatRestaurantRecommendations = (restaurants) => {
  if (!restaurants || restaurants.length === 0) {
    return "I couldn't find any restaurant recommendations at this time. Please try again later or search for restaurants in a specific area like 'restaurants in Taksim'.";
  }

  console.log('Formatting restaurants:', restaurants);

  let response = `Here are some great restaurant recommendations for you:\n\n`;

  restaurants.slice(0, 4).forEach((restaurant, index) => {
    response += `**${index + 1}. ${restaurant.name}**\n`;
    
    if (restaurant.rating) {
      response += `⭐ Rating: ${restaurant.rating}/5`;
      if (restaurant.user_ratings_total) {
        response += ` (${restaurant.user_ratings_total} reviews)`;
      }
      response += '\n';
    }
    
    if (restaurant.price_level !== undefined) {
      const priceSymbols = ['💰', '💰💰', '💰💰💰', '💰💰💰💰'];
      response += `💸 Price: ${priceSymbols[restaurant.price_level - 1] || 'N/A'}\n`;
    }
    
    if (restaurant.vicinity || restaurant.formatted_address) {
      response += `📍 Location: ${restaurant.vicinity || restaurant.formatted_address}\n`;
    }
    
    if (restaurant.types && restaurant.types.length > 0) {
      const displayTypes = restaurant.types
        .filter(type => !type.includes('_') && type !== 'establishment' && type !== 'point_of_interest')
        .slice(0, 3)
        .map(type => type.charAt(0).toUpperCase() + type.slice(1))
        .join(', ');
      if (displayTypes) {
        response += `🍽️ Cuisine: ${displayTypes}\n`;
      }
    }
    
    if (restaurant.opening_hours && restaurant.opening_hours.open_now !== undefined) {
      response += `🕒 ${restaurant.opening_hours.open_now ? 'Open now' : 'Closed now'}\n`;
    }
    
    response += '\n';
  });

  response += "These restaurants are highly rated and popular with locals and visitors. Enjoy your dining experience! 🍽️✨";
  
  return response;
};

// Format places recommendations  
const formatPlacesRecommendations = (places) => {
  if (!places || places.length === 0) {
    return "I couldn't find any places or attractions at this time. Please try searching for specific areas like 'attractions in Sultanahmet' or 'things to do in Beyoğlu'.";
  }

  console.log('Formatting places:', places);

  let response = `Here are some amazing places and attractions in Istanbul:\n\n`;

  places.slice(0, 4).forEach((place, index) => {
    response += `**${index + 1}. ${place.name}**\n`;
    
    if (place.description) {
      response += `${place.description}\n`;
    }
    
    if (place.district) {
      response += `📍 District: ${place.district}\n`;
    }
    
    if (place.address) {
      response += `🗺️ Address: ${place.address}\n`;
    }
    
    if (place.opening_hours) {
      response += `🕒 Hours: ${place.opening_hours}\n`;
    }
    
    if (place.admission_fee) {
      response += `🎫 Admission: ${place.admission_fee}\n`;
    }
    
    if (place.website) {
      response += `🌐 Website: [${place.website}](${place.website})\n`;
    }
    
    response += '\n';
  });

  response += "These are some of Istanbul's most beloved attractions. Each offers a unique glimpse into the city's rich history and vibrant culture! 🏛️✨";
  
  return response;
};

// Simplified helper function - detect restaurant requests
const isExplicitRestaurantRequest = (userInput) => {
  console.log('🍽️ Checking for explicit restaurant request:', userInput);
  
  // CRITICAL SECURITY: Preprocess and sanitize input FIRST
  const processedInput = preprocessInput(userInput);
  
  // SECURITY CHECK: Reject if input is suspicious after sanitization
  if (!processedInput || processedInput.trim().length === 0) {
    console.log('🛡️ Input rejected: Empty after sanitization');
    return false;
  }
  
  // SECURITY CHECK: Reject overly long inputs (DoS protection)
  if (processedInput.length > 500) {
    console.log('🛡️ Input rejected: Too long after sanitization');
    return false;
  }
  
  // SECURITY CHECK: Double-check for remaining malicious patterns
  const suspiciousPatterns = [
    /<[^>]*>/,
    /javascript:/i,
    /on\w+\s*=/i,
    /[;'"`].*(--)|(\/\*)/,
    /\$\([^)]*\)/
  ];
  
  if (suspiciousPatterns.some(pattern => pattern.test(processedInput))) {
    console.log('🛡️ Input still contains suspicious patterns after sanitization');
    return false;
  }

  const input = processedInput.toLowerCase();
  
  // More explicit patterns for restaurant recommendations
  const explicitRestaurantRequests = [
    'restaurant advice',      // "give me restaurant advice"
    'restaurant recommendation', // "restaurant recommendation"
    'restaurant recommendations', // "restaurant recommendations"
    'recommend restaurants',   // "recommend restaurants"
    'recommend a restaurant',  // "recommend a restaurant"
    'recommend some restaurants', // "recommend some restaurants"
    'good restaurants',        // "what are some good restaurants"
    'best restaurants',        // "best restaurants"
    'restaurants in',          // "restaurants in taksim"
    'restaurant in',           // "restaurant in galata"
    'food recommendations',    // "food recommendations"
    'where to eat',           // "where to eat in istanbul"
    'good places to eat',     // "good places to eat"
    'dining recommendations', // "dining recommendations"
    'food advice',            // "food advice"
    'any good restaurants in', // "any good restaurants in taksim"
    'any food in',           // "any food in galata"
    'any place to eat in',   // "any place to eat in kadikoy"
    'any suggestions for food in', // "any suggestions for food in fatih"
    'any suggestions for restaurants in', // "any suggestions for restaurants in sultanahmet"
  ];
  
  // Only allow Istanbul or known districts
  const istanbulDistricts = [
    'istanbul', 'beyoglu', 'beyoğlu', 'galata', 'taksim', 'sultanahmet', 'fatih',
    'kadikoy', 'kadıköy', 'besiktas', 'beşiktaş', 'uskudar', 'üsküdar', 'ortakoy',
    'ortaköy', 'sisli', 'şişli', 'karakoy', 'karaköy', 'bebek', 'arnavutkoy',
    'arnavutköy', 'balat', 'fener', 'eminonu', 'eminönü', 'bakirkoy', 'bakırköy', 'maltepe'
  ];

  const isExplicit = explicitRestaurantRequests.some(keyword => input.includes(keyword));
  if (!isExplicit) return false;

  // Extract location and check if it's Istanbul or a known district
  const { district, location } = extractLocationFromQuery(processedInput);
  if (!district && !location) return false;

  // Use either district or location for matching
  const normalized = (district || location || '').trim().toLowerCase();

  // Only allow if normalized exactly matches a known Istanbul district (no partial matches)
  const isIstanbul = istanbulDistricts.includes(normalized);
  if (!isIstanbul) {
    console.log('❌ Location is not Istanbul or a known district:', normalized);
    return false;
  }

  return true;
};

// Simplified helper function - detect places/attractions requests
const isExplicitPlacesRequest = (userInput) => {
  console.log('🏛️ Checking for explicit places/attractions request:', userInput);
  
  // CRITICAL SECURITY: Preprocess and sanitize input FIRST
  const processedInput = preprocessInput(userInput);
  
  // SECURITY CHECK: Reject if input is suspicious after sanitization
  if (!processedInput || processedInput.trim().length === 0) {
    console.log('🛡️ Input rejected: Empty after sanitization');
    return false;
  }
  
  // SECURITY CHECK: Reject overly long inputs (DoS protection)
  if (processedInput.length > 500) {
    console.log('🛡️ Input rejected: Too long after sanitization');
    return false;
  }
  
  // SECURITY CHECK: Double-check for remaining malicious patterns
  const suspiciousPatterns = [
    /<[^>]*>/,
    /javascript:/i,
    /on\w+\s*=/i,
    /[;'"`].*(--)|(\/\*)/,
    /\$\([^)]*\)/
  ];
  
  if (suspiciousPatterns.some(pattern => pattern.test(processedInput))) {
    console.log('🛡️ Input still contains suspicious patterns after sanitization');
    return false;
  }

  const input = processedInput.toLowerCase();
  
  // Explicit patterns for places/attractions requests
  const explicitPlacesRequests = [
    'attractions in',
    'places to visit in',
    'tourist attractions',
    'sightseeing in',
    'landmarks in',
    'museums in',
    'historical sites',
    'things to do in',
    'places to see in',
    'tourist spots',
    'attractions and landmarks',
    'best places to visit',
    'top attractions',
  ];
  
  const istanbulDistricts = [
    'istanbul', 'beyoglu', 'beyoğlu', 'galata', 'taksim', 'sultanahmet', 'fatih',
    'kadikoy', 'kadıköy', 'besiktas', 'beşiktaş', 'uskudar', 'üsküdar', 'ortakoy',
    'ortaköy', 'sisli', 'şişli', 'karakoy', 'karaköy', 'bebek', 'arnavutkoy',
    'arnavutköy', 'balat', 'fener', 'eminonu', 'eminönü', 'bakirkoy', 'bakırköy', 'maltepe'
  ];

  const isExplicit = explicitPlacesRequests.some(keyword => input.includes(keyword));
  if (!isExplicit) return false;

  // Extract location and check if it's Istanbul or a known district
  const { district, location } = extractLocationFromQuery(processedInput);
  if (!district && !location) return false;

  // Use either district or location for matching
  const normalized = (district || location || '').trim().toLowerCase();

  // Only allow if normalized exactly matches a known Istanbul district
  const isIstanbul = istanbulDistricts.includes(normalized);
  if (!isIstanbul) {
    console.log('❌ Location is not Istanbul or a known district:', normalized);
    return false;
  }

  return true;
};

function Chatbot() {
  // Enhanced state management
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('chat-messages');
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error('Failed to load messages from localStorage:', error);
      return [];
    }
  });
  
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    try {
      const saved = localStorage.getItem('dark-mode');
      return saved ? JSON.parse(saved) : false;
    } catch (error) {
      console.error('Failed to load dark mode from localStorage:', error);
      return false;
    }
  });

  // Enhanced UI state
  const [isTyping, setIsTyping] = useState(false);
  const [typingMessage, setTypingMessage] = useState('AI is thinking...');
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  
  // Enhanced error handling state
  const [currentError, setCurrentError] = useState(null);
  const [retryAction, setRetryAction] = useState(null);
  const [lastFailedMessage, setLastFailedMessage] = useState(null);
  
  // Network and health monitoring
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [apiHealth, setApiHealth] = useState('unknown');

  // Enhanced message management
  const addMessage = (text, sender = 'assistant', metadata = {}) => {
    const newMessage = {
      id: Date.now() + Math.random(),
      text: typeof text === 'string' ? text : '',
      sender,
      timestamp: new Date().toISOString(),
      ...metadata
    };
    
    setMessages(prev => {
      const updated = [...prev, newMessage];
      // Persist to localStorage
      try {
        localStorage.setItem('chat-messages', JSON.stringify(updated));
      } catch (error) {
        console.error('Failed to save messages to localStorage:', error);
      }
      return updated;
    });
  };

  const clearChatHistory = () => {
    setMessages([]);
    try {
      localStorage.removeItem('chat-messages');
    } catch (error) {
      console.error('Failed to clear messages from localStorage:', error);
    }
    console.log('🗑️ Chat history cleared');
  };

  // Enhanced clipboard and sharing
  const copyMessageToClipboard = async (message) => {
    try {
      await navigator.clipboard.writeText(message.text);
      console.log('📋 Message copied to clipboard');
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  const shareMessage = async (message) => {
    const shareText = `KAM AI Assistant: ${message.text}`;
    
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'KAM AI Assistant Response',
          text: shareText,
        });
      } catch (error) {
        if (error.name !== 'AbortError') {
          console.error('Error sharing:', error);
          // Fallback to clipboard
          await copyMessageToClipboard(message);
        }
      }
    } else {
      // Fallback to clipboard
      await copyMessageToClipboard(message);
    }
  };

  // Enhanced scroll management
  const scrollToBottom = () => {
    const chatContainer = document.getElementById('chat-messages');
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  };

  // Enhanced effect hooks
  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    // Persist dark mode
    try {
      localStorage.setItem('dark-mode', JSON.stringify(darkMode));
    } catch (error) {
      console.error('Failed to save dark mode to localStorage:', error);
    }
  }, [darkMode]);

  useEffect(() => {
    // Monitor scroll position for scroll-to-bottom button
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = chatContainer;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollToBottom(!isNearBottom && messages.length > 0);
    };

    chatContainer.addEventListener('scroll', handleScroll);
    return () => chatContainer.removeEventListener('scroll', handleScroll);
  }, [messages.length]);

  useEffect(() => {
    // Network status monitoring
    const unsubscribe = subscribeToNetworkStatus((status) => {
      setIsOnline(status.isOnline);
      console.log('🌐 Network status changed:', status);
    });

    return unsubscribe;
  }, []);

  useEffect(() => {
    // Periodic API health checks
    const checkHealth = async () => {
      try {
        const isHealthy = await checkApiHealth();
        setApiHealth(isHealthy ? 'healthy' : 'unhealthy');
      } catch (error) {
        setApiHealth('error');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Enhanced error handling
  const handleError = (error, context = 'unknown', failedMessage = null) => {
    console.error(`Error in ${context}:`, error);
    
    const errorInfo = {
      type: classifyError(error),
      message: getUserFriendlyMessage(error),
      context,
      timestamp: Date.now(),
      failedMessage
    };
    
    setCurrentError(errorInfo);
    
    // Set retry action if we have a failed message
    if (failedMessage && failedMessage.input) {
      setRetryAction(() => () => {
        console.log('🔄 Retrying failed message:', failedMessage.input);
        handleSend(failedMessage.input);
      });
    }
    
    setLoading(false);
  };

  const handleRetry = () => {
    if (retryAction) {
      console.log('🔄 Retrying last action');
      setCurrentError(null);
      retryAction();
    }
  };

  const dismissError = () => {
    setCurrentError(null);
    setRetryAction(null);
    setLastFailedMessage(null);
  };

  const handleSend = async (customInput = null) => {
    const originalUserInput = customInput || input.trim();
    if (!originalUserInput) return;

    // Create retry action for this message
    const retryCurrentMessage = () => {
      console.log('🔄 Retrying message:', originalUserInput);
      handleSend(originalUserInput);
    };
    setRetryAction(() => retryCurrentMessage);

    // CRITICAL SECURITY: Sanitize input immediately
    const sanitizedInput = preprocessInput(originalUserInput);
    
    // SECURITY CHECK: Reject if sanitization removed everything
    if (!sanitizedInput || sanitizedInput.trim().length === 0) {
      console.log('🛡️ Input rejected: Empty after sanitization');
      addMessage('Sorry, your input contains invalid characters. Please try again with a different message.', 'assistant', {
        type: 'error'
      });
      setLoading(false);
      return;
    }
    
    // SECURITY CHECK: Verify sanitized input is safe
    const isSuspicious = [
      /<[^>]*>/,
      /javascript:/i,
      /on\w+\s*=/i,
      /[;'"`].*(--)|(\/\*)/,
      /\$\([^)]*\)/
    ].some(pattern => pattern.test(sanitizedInput));
    
    if (isSuspicious) {
      console.log('🛡️ Input still suspicious after sanitization, rejecting');
      addMessage('Sorry, your input appears to contain invalid content. Please try again with a different message.', 'assistant', {
        type: 'error'
      });
      setLoading(false);
      return;
    }

    console.log('✅ Using sanitized input:', sanitizedInput);
    
    // Add user message with enhanced metadata
    addMessage(originalUserInput, 'user', {
      sanitizedInput,
      originalLength: originalUserInput.length,
      sanitizedLength: sanitizedInput.length
    });
    
    setInput('');
    setLoading(true);
    setIsTyping(true);

    // Store failed message for retry purposes
    setLastFailedMessage({
      input: originalUserInput,
      sanitizedInput,
      timestamp: Date.now()
    });

    try {
      // Check if user is asking for restaurant recommendations - using SANITIZED input
      if (isExplicitRestaurantRequest(originalUserInput)) {
        setTypingMessage('Finding restaurants for you...');
        console.log('Detected restaurant advice request, fetching recommendations...');
        console.log('Original input:', originalUserInput);
        console.log('🛡️ Sending SANITIZED input to backend:', sanitizedInput);
        
        // CRITICAL: Use sanitized input for API call
        const restaurantData = await fetchRestaurantRecommendations(sanitizedInput);
        console.log('Restaurant API response:', restaurantData);
        const formattedResponse = formatRestaurantRecommendations(restaurantData.restaurants);
        console.log('Formatted response:', formattedResponse);
        
        addMessage(formattedResponse, 'assistant', {
          type: 'restaurant-recommendation',
          dataSource: 'google-places',
          resultCount: restaurantData.restaurants?.length || 0
        });
        
        // Clear failed message on success
        setLastFailedMessage(null);
        return;
      }

      // Check if user is asking for places/attractions recommendations - using SANITIZED input
      if (isExplicitPlacesRequest(originalUserInput)) {
        setTypingMessage('Searching for places and attractions...');
        console.log('Detected places/attractions request, fetching recommendations...');
        console.log('Original input:', originalUserInput);
        console.log('🛡️ Sending SANITIZED input to backend:', sanitizedInput);
        
        // CRITICAL: Use sanitized input for API call
        const placesData = await fetchPlacesRecommendations(sanitizedInput);
        console.log('Places API response:', placesData);
        const formattedResponse = formatPlacesRecommendations(placesData.places);
        console.log('Formatted response:', formattedResponse);
        
        addMessage(formattedResponse, 'assistant', {
          type: 'places-recommendation',
          dataSource: 'database',
          resultCount: placesData?.places?.length || 0
        });
        
        // Clear failed message on success
        setLastFailedMessage(null);
        return;
      }

      // Regular streaming response for non-restaurant/places queries - use SANITIZED input
      setTypingMessage('KAM is thinking...');
      let streamedContent = '';
      
      console.log('🛡️ Sending SANITIZED input to GPT:', sanitizedInput);
      await fetchStreamingResults(sanitizedInput, (chunk) => {
        streamedContent += chunk;
        // If assistant message already exists, update it; else, add it
        setMessages((prev) => {
          // If last message is assistant and was streaming, update it
          if (prev.length > 0 && prev[prev.length - 1].role === 'assistant' && prev[prev.length - 1].streaming) {
            return [
              ...prev.slice(0, -1),
              { role: 'assistant', content: streamedContent, streaming: true }
            ];
          } else {
            return [
              ...prev,
              { role: 'assistant', content: streamedContent, streaming: true }
            ];
          }
        });
      });
      
      // Clear failed message on success
      setLastFailedMessage(null);
      
    } catch (error) {
      handleError(error, 'message sending', lastFailedMessage);
      
      // Add error message with enhanced metadata
      const errorMessage = error.message.includes('fetch')
        ? 'Sorry, I encountered an error connecting to the server. Please check your connection and try again.'
        : `Sorry, there was an error: ${error.message}. Please try again.`;
      
      addMessage(errorMessage, 'assistant', {
        type: 'error',
        errorType: classifyError(error),
        canRetry: true,
        originalInput: originalUserInput
      });
    } finally {
      setLoading(false);
      setIsTyping(false);
      setTypingMessage('AI is thinking...');
      
      // Remove streaming flag on last assistant message
      setMessages((prev) => {
        if (prev.length > 0 && prev[prev.length - 1].role === 'assistant' && prev[prev.length - 1].streaming) {
          return [
            ...prev.slice(0, -1),
            { role: 'assistant', content: prev[prev.length - 1].content }
          ];
        }
        return prev;
      });
    }
  };

  const handleSampleClick = (question) => {
    // Automatically send the message
    handleSend(question);
  };

  return (
    <div className={`flex flex-col h-screen w-full pt-16 transition-colors duration-200 ${
      darkMode ? 'bg-gray-900' : 'bg-gray-100'
    }`}>
      
      {/* Enhanced Header with chat management */}
      <ChatHeader
        darkMode={darkMode}
        onDarkModeToggle={() => setDarkMode(!darkMode)}
        onClearHistory={clearChatHistory}
        messageCount={messages.length}
        isOnline={isOnline}
        apiHealth={apiHealth}
      />

      {/* Chat Messages Container - Full screen like ChatGPT */}
      <div className="flex-1 overflow-y-auto chat-messages" id="chat-messages">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center px-4">
            {/* KAM Definition Card - Always visible */}
            <div className={`max-w-2xl w-full mb-8 p-6 rounded-xl border transition-all duration-200 ${
              darkMode 
                ? 'bg-gray-800 border-gray-700 shadow-lg' 
                : 'bg-white border-gray-300 shadow-lg'
            }`}>
              <div className={`text-center mb-4`}>
                <h3 className={`text-xl font-bold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-blue-300' : 'text-blue-700'
                }`}>
                  KAM - Your AI Istanbul Guide
                </h3>
                <div className={`text-sm leading-relaxed transition-colors duration-200 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  <p className="mb-2">
                    <strong>Kam</strong>, in Turkish, Altaic, and Mongolian folk culture, is a shaman, a religious leader, wisdom person. Also referred to as "Gam" or Ham.
                  </p>
                  <p className={`italic ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    A religious leader believed to communicate with supernatural powers within communities.
                  </p>
                </div>
              </div>
              <div className={`text-center text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Just like the traditional Kam guides their community, I'm here to guide you through Istanbul's wonders.
              </div>
            </div>

            <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-6 transition-colors duration-200 ${
              darkMode ? 'bg-white' : 'bg-gradient-to-br from-blue-600 to-purple-600'
            }`}>
              <svg className={`w-8 h-8 transition-colors duration-200 ${
                darkMode ? 'text-black' : 'text-white'
              }`} fill="currentColor" viewBox="0 0 24 24">
                <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
              </svg>
            </div>
            <h2 className={`text-3xl font-bold mb-4 transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>How can I help you today?</h2>
            <p className={`text-center max-w-2xl text-lg leading-relaxed mb-8 transition-colors duration-200 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              I'm your AI assistant for exploring Istanbul. Ask me about restaurants, attractions, 
              neighborhoods, culture, history, or anything else about this amazing city!
            </p>
            
            {/* Enhanced Sample Cards with Better Light Mode Styling */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl w-full px-4">
              <div 
                onClick={() => handleSampleClick('Show me the best attractions and landmarks in Istanbul')}
                className={`p-5 rounded-xl border-2 transition-all duration-200 cursor-pointer hover:shadow-xl hover:scale-105 transform ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600' 
                    : 'bg-white border-blue-200 hover:bg-blue-50 hover:border-blue-400 shadow-md hover:shadow-lg'
                }`}
              >
                <div className={`font-bold text-lg mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🏛️ Top Attractions</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-700'
                }`}>Show me the best attractions and landmarks in Istanbul</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Give me restaurant advice - recommend 4 good restaurants')}
                className={`p-5 rounded-xl border-2 transition-all duration-200 cursor-pointer hover:shadow-xl hover:scale-105 transform ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600' 
                    : 'bg-white border-red-200 hover:bg-red-50 hover:border-red-400 shadow-md hover:shadow-lg'
                }`}
              >
                <div className={`font-bold text-lg mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🍽️ Restaurants</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-700'
                }`}>Give me restaurant advice - recommend 4 good restaurants</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Tell me about Istanbul neighborhoods and districts to visit')}
                className={`p-5 rounded-xl border-2 transition-all duration-200 cursor-pointer hover:shadow-xl hover:scale-105 transform ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600' 
                    : 'bg-white border-green-200 hover:bg-green-50 hover:border-green-400 shadow-md hover:shadow-lg'
                }`}
              >
                <div className={`font-bold text-lg mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🏘️ Neighborhoods</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-700'
                }`}>Tell me about Istanbul neighborhoods and districts to visit</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('What are the best cultural experiences and activities in Istanbul?')}
                className={`p-5 rounded-xl border-2 transition-all duration-200 cursor-pointer hover:shadow-xl hover:scale-105 transform ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600' 
                    : 'bg-white border-purple-200 hover:bg-purple-50 hover:border-purple-400 shadow-md hover:shadow-lg'
                }`}
              >
                <div className={`font-bold text-lg mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🎭 Culture & Activities</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-700'
                }`}>What are the best cultural experiences and activities in Istanbul?</div>
              </div>
            </div>
          </div>
        )}
            
        {/* Message Display Area */}
        <div className="max-w-full mx-auto px-4">
          {messages.map((msg, index) => (
            <div key={msg.id || index} className="group py-4">
              <div className="flex items-start space-x-3">
                {msg.sender === 'user' ? (
                  <>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500' 
                        : 'bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600'
                    }`}>
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                        darkMode ? 'text-gray-300' : 'text-gray-600'
                      }`}>You</div>
                      <div className={`text-sm whitespace-pre-wrap transition-colors duration-200 ${
                        darkMode ? 'text-white' : 'text-gray-900'
                      }`}>
                        {msg.text}
                      </div>
                      {msg.timestamp && (
                        <div className={`text-xs mt-1 transition-colors duration-200 ${
                          darkMode ? 'text-gray-500' : 'text-gray-500'
                        }`}>
                          {new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                        </div>
                      )}
                    </div>
                    <MessageActions 
                      message={msg}
                      onCopy={copyMessageToClipboard}
                      onShare={shareMessage}
                      darkMode={darkMode}
                    />
                  </>
                ) : (
                  <>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                        : 'bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600'
                    }`}>
                      <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                        darkMode ? 'text-gray-300' : 'text-gray-600'
                      }`}>KAM Assistant</div>
                      <div className={`text-sm whitespace-pre-wrap leading-relaxed transition-colors duration-200 ${
                        darkMode ? 'text-white' : 'text-gray-900'
                      }`}>
                        {renderMessageContent(msg.text || msg.content, darkMode)}
                      </div>
                      
                      {/* Metadata Components */}
                      {msg.sender === 'assistant' && (
                        <>
                          {/* ML Insights */}
                          {msg.metadata?.ml_predictions && (
                            <div className="mt-3">
                              <MLInsights predictions={msg.metadata.ml_predictions} darkMode={darkMode} />
                            </div>
                          )}
                          
                          {/* POI Cards */}
                          {msg.metadata?.pois?.map((poi, idx) => (
                            <div key={idx} className="mt-3">
                              <POICard poi={poi} darkMode={darkMode} />
                            </div>
                          ))}
                          
                          {/* District Info */}
                          {msg.metadata?.district_info && (
                            <div className="mt-3">
                              <DistrictInfo district={msg.metadata.district_info} darkMode={darkMode} />
                            </div>
                          )}
                          
                          {/* Itinerary */}
                          {msg.metadata?.total_itinerary && (
                            <div className="mt-3">
                              <ItineraryTimeline itinerary={msg.metadata.total_itinerary} darkMode={darkMode} />
                            </div>
                          )}
                        </>
                      )}
                      
                      <div className="hidden">
                      </div>
                      {msg.timestamp && (
                        <div className={`text-xs mt-1 flex items-center space-x-2 transition-colors duration-200 ${
                          darkMode ? 'text-gray-500' : 'text-gray-500'
                        }`}>
                          <span>{new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                          {msg.type && (
                            <span className={`px-2 py-1 rounded text-xs ${
                              darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-700'
                            }`}>
                              {msg.type}
                            </span>
                          )}
                          {msg.resultCount && (
                            <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                              {msg.resultCount} results
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    <MessageActions 
                      message={msg}
                      onCopy={copyMessageToClipboard}
                      onShare={shareMessage}
                      onRetry={msg.canRetry ? () => handleSend(msg.originalInput) : null}
                      darkMode={darkMode}
                    />
                  </>
                )}
              </div>
            </div>
          ))}
          
          <TypingIndicator 
            isTyping={isTyping} 
            message={typingMessage}
            darkMode={darkMode}
          />
        </div>
      </div>

      {/* Scroll to bottom button */}
      <ScrollToBottom 
        show={showScrollToBottom}
        onClick={scrollToBottom}
        darkMode={darkMode}
      />

      {/* Enhanced Input Area with Better Light Mode Styling */}
      <div className={`border-t p-4 transition-colors duration-200 ${
        darkMode 
          ? 'bg-gray-900 border-gray-700' 
          : 'bg-white border-gray-300'
      }`}>
        <div className="max-w-4xl mx-auto">
          <div className={`flex items-end space-x-3 p-4 rounded-xl border-2 transition-all duration-200 ${
            darkMode 
              ? 'bg-gray-800 border-gray-700 focus-within:border-gray-600' 
              : 'bg-gray-50 border-gray-300 focus-within:border-blue-400 shadow-sm'
          }`}>
            <div className="flex-1">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Ask about Istanbul..."
                className={`w-full bg-transparent border-0 outline-none focus:outline-none focus:ring-0 text-base resize-none transition-colors duration-200 ${
                  darkMode 
                    ? 'placeholder-gray-400 text-white' 
                    : 'placeholder-gray-500 text-gray-900'
                }`}
                disabled={loading}
                autoComplete="off"
              />
            </div>
            <button 
              onClick={handleSend} 
              disabled={loading || !input.trim()}
              className={`p-3 rounded-lg transition-all duration-200 ${
                darkMode 
                  ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-700 hover:via-indigo-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600' 
                  : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500 hover:from-blue-600 hover:via-indigo-600 hover:to-purple-600 disabled:from-gray-400 disabled:to-gray-400'
              } disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95`}
            >
              {loading ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                </svg>
              )}
            </button>
          </div>
          <div className={`text-xs text-center mt-2 transition-colors duration-200 ${
            darkMode ? 'text-gray-500' : 'text-gray-600'
          }`}>
            Your AI-powered Istanbul guide
          </div>
        </div>
      </div>

      {/* Error Notification */}
      {currentError && (
        <ErrorNotification
          error={currentError}
          onRetry={handleRetry}
          onDismiss={dismissError}
          autoHide={false}
          darkMode={darkMode}
        />
      )}
      
      {/* Network Status Indicator */}
      <NetworkStatusIndicator darkMode={darkMode} />
    </div>
  );
}

export default Chatbot;
