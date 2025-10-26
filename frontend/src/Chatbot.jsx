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
import ErrorNotification, { NetworkStatusIndicator } from './components/ErrorNotification';
import TypingIndicator from './components/TypingIndicator';
import MessageActions from './components/MessageActions';
import ScrollToBottom from './components/ScrollToBottom';
import ChatHeader from './components/ChatHeader';
import POICard from './components/POICard';
import DistrictInfo from './components/DistrictInfo';
import ItineraryTimeline from './components/ItineraryTimeline';
import MLInsights from './components/MLInsights';
import ChatMapView from './components/ChatMapView';
import TransportationInterface from './components/TransportationInterface';
import { useLocation } from './contexts/LocationContext';
import './components/Chatbot.css';

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
  // Location Context - GPS Integration
  const {
    currentLocation,
    hasLocation,
    hasGPSLocation,
    locationSummary,
    neighborhood,
    gpsPermission,
    requestGPSLocation,
    startGPSTracking,
    stopGPSTracking,
    isTracking,
    formatLocationForAI
  } = useLocation();

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
  const [typingMessage, setTypingMessage] = useState('KAM is thinking...');
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  
  // GPS UI state
  const [showGPSStatus, setShowGPSStatus] = useState(false);
  const [gpsEnabled, setGpsEnabled] = useState(false);
  
  // Enhanced error handling state
  const [currentError, setCurrentError] = useState(null);
  
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

  // GPS Location Handling
  const handleGPSToggle = async () => {
    try {
      if (!gpsEnabled) {
        setTypingMessage('Getting your precise location...');
        setIsTyping(true);
        
        const location = await requestGPSLocation();
        setGpsEnabled(true);
        setShowGPSStatus(true);
        
        console.log('📍 GPS enabled with high accuracy:', location);
        console.log(`   Accuracy: ±${location.accuracy?.toFixed(1)}m`);
        
        // Start continuous tracking for Google Maps-like experience
        startGPSTracking();
        console.log('🎯 Continuous GPS tracking active');
        
        setIsTyping(false);
      } else {
        setGpsEnabled(false);
        stopGPSTracking();
        setShowGPSStatus(false);
        console.log('📍 GPS tracking stopped');
      }
    } catch (error) {
      console.error('GPS toggle error:', error);
      setIsTyping(false);
      setCurrentError({
        type: ErrorTypes.PERMISSION_DENIED,
        message: 'Unable to access your location. Please enable location permissions in your browser settings.',
        timestamp: Date.now()
      });
    }
  };

  // Format location context for AI
  const getLocationContext = () => {
    if (!hasLocation) return null;
    
    const locationData = formatLocationForAI();
    return {
      has_location: true,
      latitude: locationData.lat,
      longitude: locationData.lng,
      accuracy: locationData.accuracy,
      neighborhood: locationData.neighborhood || locationSummary,
      source: locationData.source
    };
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
  }, [messages]);

  // Auto-focus input on mount
  useEffect(() => {
    const inputElement = document.getElementById('chat-input');
    if (inputElement) {
      inputElement.focus();
    }
  }, []);

  // Enhanced message sending function
  const handleSendMessage = async (messageText = null, isRegenerate = false) => {
    const textToSend = messageText || input.trim();
    
    if (!textToSend || loading) return;

    console.log('📤 Sending message:', textToSend);

    // Clear current error
    setCurrentError(null);
    
    // Add user message (unless regenerating)
    if (!isRegenerate) {
      const userMessage = {
        type: 'user',
        content: textToSend,
        timestamp: Date.now()
      };
      
      setMessages(prev => {
        const updated = [...prev, userMessage];
        try {
          localStorage.setItem('chat-messages', JSON.stringify(updated));
        } catch (error) {
          console.error('Failed to save messages:', error);
        }
        return updated;
      });
    }

    // Clear input and set loading states
    if (!messageText) setInput('');
    setLoading(true);
    setIsTyping(true);
    setTypingMessage('KAM is thinking...');

    try {
      // Security preprocessing
      const processedInput = preprocessInput(textToSend);
      
      if (!processedInput) {
        throw new Error('Invalid input after security processing');
      }

      // Check for explicit restaurant requests
      if (isExplicitRestaurantRequest(processedInput)) {
        console.log('🍽️ Processing restaurant request');
        setTypingMessage('Finding great restaurants for you...');
        
        const restaurants = await debouncedFetchRestaurants(processedInput);
        const formattedResponse = formatRestaurantRecommendations(restaurants);
        
        const aiMessage = {
          type: 'ai',
          content: formattedResponse,
          restaurants: restaurants,
          timestamp: Date.now()
        };
        
        setMessages(prev => {
          const updated = [...prev, aiMessage];
          try {
            localStorage.setItem('chat-messages', JSON.stringify(updated));
          } catch (error) {
            console.error('Failed to save messages:', error);
          }
          return updated;
        });
        
        return;
      }

      // Check for explicit places/attractions requests
      if (isExplicitPlacesRequest(processedInput)) {
        console.log('🏛️ Processing places request');
        setTypingMessage('Discovering amazing places for you...');
        
        const places = await debouncedFetchPlaces(processedInput);
        const formattedResponse = formatPlacesRecommendations(places);
        
        const aiMessage = {
          type: 'ai',
          content: formattedResponse,
          places: places,
          timestamp: Date.now()
        };
        
        setMessages(prev => {
          const updated = [...prev, aiMessage];
          try {
            localStorage.setItem('chat-messages', JSON.stringify(updated));
          } catch (error) {
            console.error('Failed to save messages:', error);
          }
          return updated;
        });
        
        return;
      }

      // Default: Use streaming AI response
      console.log('🤖 Processing general AI request');
      setTypingMessage('KAM is generating response...');
      
      let aiResponse = '';
      
      const onChunk = (chunk) => {
        aiResponse += chunk;
        
        // Update the last AI message with streaming content
        setMessages(prev => {
          const updated = [...prev];
          const lastMessage = updated[updated.length - 1];
          
          if (lastMessage && lastMessage.type === 'ai' && !lastMessage.isComplete) {
            lastMessage.content = aiResponse;
          } else {
            updated.push({
              type: 'ai',
              content: aiResponse,
              timestamp: Date.now(),
              isComplete: false
            });
          }
          
          return updated;
        });
      };

      const onComplete = (finalResponse, metadata) => {
        console.log('✅ AI response complete');
        if (metadata) {
          console.log('📊 Metadata received:', metadata);
        }
        
        setMessages(prev => {
          const updated = [...prev];
          const lastMessage = updated[updated.length - 1];
          
          if (lastMessage && lastMessage.type === 'ai') {
            lastMessage.content = finalResponse || aiResponse;
            lastMessage.isComplete = true;
            
            // Add metadata if present (map_data, intent, entities, etc.)
            if (metadata) {
              if (metadata.map_data) {
                lastMessage.map_data = metadata.map_data;
                console.log('🗺️ Map data attached to message:', metadata.map_data);
              }
              if (metadata.intent) {
                lastMessage.intent = metadata.intent;
              }
              if (metadata.entities) {
                lastMessage.entities = metadata.entities;
              }
            }
            
            delete lastMessage.isComplete; // Clean up flag
          }
          
          try {
            localStorage.setItem('chat-messages', JSON.stringify(updated));
          } catch (error) {
            console.error('Failed to save messages:', error);
          }
          
          return updated;
        });
      };

      const onError = (error) => {
        console.error('❌ AI response error:', error);
        
        const errorMessage = {
          type: 'ai',
          content: 'I apologize, but I encountered an error while processing your request. Please try again.',
          timestamp: Date.now(),
          isError: true
        };
        
        setMessages(prev => {
          const updated = [...prev, errorMessage];
          try {
            localStorage.setItem('chat-messages', JSON.stringify(updated));
          } catch (error) {
            console.error('Failed to save messages:', error);
          }
          return updated;
        });
      };

      // Get location context if GPS is enabled
      const locationContext = gpsEnabled && hasLocation ? getLocationContext() : null;
      
      if (locationContext) {
        console.log('📍 Including location context:', locationContext);
      }

      // Call streaming AI API with metadata support
      await fetchStreamingResults(
        processedInput,
        onChunk,
        onComplete,
        onError,
        locationContext  // Include location data
      );

    } catch (error) {
      console.error('❌ Error in handleSendMessage:', error);
      
      // Classify and handle error
      const errorType = classifyError(error);
      const friendlyMessage = getUserFriendlyMessage(errorType, error);
      
      setCurrentError({
        type: errorType,
        message: friendlyMessage,
        originalError: error
      });
      
      // Add error message
      const errorMessage = {
        type: 'ai',
        content: friendlyMessage,
        timestamp: Date.now(),
        isError: true
      };
      
      setMessages(prev => {
        const updated = [...prev, errorMessage];
        try {
          localStorage.setItem('chat-messages', JSON.stringify(updated));
        } catch (error) {
          console.error('Failed to save messages:', error);
        }
        return updated;
      });
      
    } finally {
      setLoading(false);
      setIsTyping(false);
      setTypingMessage('KAM is thinking...');
    }
  };

  // Network status monitoring
  useEffect(() => {
    const handleOnlineStatus = () => {
      setIsOnline(navigator.onLine);
      if (navigator.onLine) {
        console.log('🌐 Back online');
      } else {
        console.log('📴 Gone offline');
      }
    };

    window.addEventListener('online', handleOnlineStatus);
    window.addEventListener('offline', handleOnlineStatus);
    
    // Subscribe to network status from error handler
    const unsubscribe = subscribeToNetworkStatus((status) => {
      setIsOnline(status.isOnline);
    });

    // Check API health periodically
    const healthCheckInterval = setInterval(async () => {
      try {
        const health = await checkApiHealth();
        setApiHealth(health.status);
      } catch (error) {
        setApiHealth('error');
      }
    }, 30000); // Check every 30 seconds

    return () => {
      window.removeEventListener('online', handleOnlineStatus);
      window.removeEventListener('offline', handleOnlineStatus);
      unsubscribe();
      clearInterval(healthCheckInterval);
    };
  }, []);

  // Main render function
  return (
    <div className={`chatbot-container ${darkMode ? 'dark-mode' : ''}`}>
      <NetworkStatusIndicator />

      {/* Chat Messages Container */}
      <div 
        id="chat-messages" 
        className="chat-messages"
        role="log"
        aria-live="polite"
        aria-label="Chat conversation"
      >
        {messages.length === 0 && (
          <div className="welcome-message">
            <div className="welcome-card">
              <h3>👋 Welcome to KAM - Your AI Istanbul Guide!</h3>
              <p>I'm KAM, your personal Istanbul guide. Ask me about:</p>
              <div className="suggestion-grid">
                <button 
                  onClick={() => setInput('Best restaurants in Sultanahmet')}
                  className="suggestion-chip"
                >
                  🍽️ Restaurants
                </button>
                <button 
                  onClick={() => setInput('Tourist attractions in Beyoğlu')}
                  className="suggestion-chip"
                >
                  🏛️ Attractions
                </button>
                <button 
                  onClick={() => setInput('Things to do in Taksim')}
                  className="suggestion-chip"
                >
                  🎯 Activities
                </button>
                <button 
                  onClick={() => setInput('How to get around Istanbul')}
                  className="suggestion-chip"
                >
                  🚇 Transportation
                </button>
              </div>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`message-container ${message.type}`}>
            <div className={`message-bubble ${message.type}`}>
              <div className="message-content">
                {message.type === 'ai' && message.restaurants ? (
                  <div className="ai-response-section">
                    <div className="response-text">{message.content}</div>
                    <div className="recommendations-grid">
                      {message.restaurants.map((restaurant, idx) => (
                        <POICard
                          key={`restaurant-${idx}`}
                          poi={restaurant}
                          type="restaurant"
                          darkMode={darkMode}
                        />
                      ))}
                    </div>
                  </div>
                ) : message.type === 'ai' && message.places ? (
                  <div className="ai-response-section">
                    <div className="response-text">{message.content}</div>
                    <div className="recommendations-grid">
                      {message.places.map((place, idx) => (
                        <POICard
                          key={`place-${idx}`}
                          poi={place}
                          type="attraction"
                          darkMode={darkMode}
                        />
                      ))}
                    </div>
                  </div>
                ) : message.type === 'ai' && message.itinerary ? (
                  <div className="ai-response-section">
                    <div className="response-text">{message.content}</div>
                    <ItineraryTimeline 
                      itinerary={message.itinerary} 
                      darkMode={darkMode}
                    />
                  </div>
                ) : message.type === 'ai' && message.district ? (
                  <div className="ai-response-section">
                    <div className="response-text">{message.content}</div>
                    <DistrictInfo 
                      district={message.district} 
                      darkMode={darkMode}
                    />
                  </div>
                ) : message.type === 'ai' && message.insights ? (
                  <div className="ai-response-section">
                    <div className="response-text">{message.content}</div>
                    <MLInsights 
                      insights={message.insights} 
                      darkMode={darkMode}
                    />
                  </div>
                ) : (
                  <div>
                    <div className="simple-message">{message.content}</div>
                    {/* Show map if map_data is present */}
                    {message.type === 'ai' && message.map_data && (
                      <div className="map-container">
                        <ChatMapView 
                          mapData={message.map_data}
                          darkMode={darkMode}
                        />
                      </div>
                    )}
                    {/* Show transportation interface if transportation data is present */}
                    {message.type === 'ai' && message.map_data && message.map_data.locations && message.map_data.locations.length > 1 && (
                      <div className="transportation-container mt-4">
                        <TransportationInterface
                          mapData={message.map_data}
                          initialOrigin={message.map_data.locations[0]}
                          initialDestination={message.map_data.locations[message.map_data.locations.length - 1]}
                          initialMode="transit"
                          darkMode={darkMode}
                        />
                      </div>
                    )}
                    {/* Show route timeline if route_data exists */}
                    {message.type === 'ai' && message.route_data && message.route_data.segments && (
                      <div className="route-timeline-container mt-4">
                        <div className="route-summary">
                          <h4>🗺️ Optimized Route</h4>
                          <div className="route-stats">
                            <span className="stat">
                              <strong>Distance:</strong> {message.route_data.total_distance_km} km
                            </span>
                            <span className="stat">
                              <strong>Duration:</strong> {message.route_data.total_duration_hours} hours
                            </span>
                          </div>
                        </div>
                        <ItineraryTimeline 
                          itinerary={{
                            stops: message.route_data.segments.map((seg, idx) => ({
                              name: seg.from,
                              description: `Walk ${seg.distance_km} km to ${seg.to}`,
                              duration: `${seg.walking_time_min} minutes`,
                              order: idx + 1
                            }))
                          }}
                          darkMode={darkMode}
                        />
                      </div>
                    )}
                    {/* Show ML insights if route has predictions */}
                    {message.type === 'ai' && message.route_data && message.route_data.ml_predictions && (
                      <div className="ml-insights-container mt-4">
                        <MLInsights 
                          insights={{
                            predictions: message.route_data.ml_predictions,
                            crowding_levels: message.route_data.ml_predictions.crowding_levels,
                            weather_impact: message.route_data.ml_predictions.weather_impact,
                            confidence: message.route_data.ml_predictions.confidence_score
                          }}
                          darkMode={darkMode}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="message-container ai">
            <div className="message-bubble ai typing">
              <div className="message-content">
                <TypingIndicator message={typingMessage} />
              </div>
            </div>
          </div>
        )}

        {/* Error Notifications */}
        <ErrorNotification />
      </div>

      {/* Scroll to Bottom Button */}
      {showScrollToBottom && (
        <ScrollToBottom 
          onClick={scrollToBottom}
          darkMode={darkMode}
        />
      )}

      {/* Chat Input */}
      <div className="chat-input-container">
        {/* GPS Location Status */}
        {showGPSStatus && hasLocation && (
          <div className={`gps-status ${darkMode ? 'dark' : ''}`}>
            <span className="location-icon">📍</span>
            <span className="location-text">
              {neighborhood || locationSummary}
            </span>
            <span className="location-accuracy">
              {hasGPSLocation ? '(Live GPS)' : '(Cached)'}
            </span>
          </div>
        )}
        
        <div className="input-wrapper">
          {/* GPS Toggle Button */}
          <button
            onClick={handleGPSToggle}
            className={`gps-toggle-button ${gpsEnabled ? 'active' : ''} ${darkMode ? 'dark' : ''}`}
            aria-label="Toggle GPS location"
            title={gpsEnabled ? 'GPS enabled - Click to disable' : 'Click to enable GPS location'}
            disabled={loading}
          >
            {gpsEnabled ? '📍' : '📍'}
          </button>
          
          <input
            id="chat-input"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !loading && handleSendMessage()}
            placeholder="Ask me about Istanbul restaurants, attractions, or travel tips..."
            disabled={loading}
            className="chat-input"
            maxLength={500}
            aria-label="Chat input"
          />
          
          <button
            onClick={() => handleSendMessage()}
            disabled={loading || !input.trim()}
            className="send-button"
            aria-label="Send message"
          >
            {loading ? (
              <div className="loading-spinner" />
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
              </svg>
            )}
          </button>
        </div>
        
        {/* Input Character Counter */}
        <div className="character-counter">
          <span className={input.length > 450 ? 'warning' : ''}>
            {input.length}/500
          </span>
        </div>
      </div>
    </div>
  );
}

export default Chatbot;
