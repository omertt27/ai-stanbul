import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useRef } from 'react';
import React from 'react';
import { 
  fetchUnifiedChat,
  fetchUnifiedChatV2,
  fetchRestaurantRecommendations, 
  fetchPlacesRecommendations, 
  extractLocationFromQuery,
  subscribeToNetworkStatus,
  checkApiHealth,
  debouncedFetchRestaurants,
  debouncedFetchPlaces,
  getSessionId
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
import ChatSessionsPanel from './components/ChatSessionsPanel';
import MapVisualization from './components/MapVisualization';
import SimpleChatInput from './components/SimpleChatInput';
import RestaurantCard from './components/RestaurantCard';
import MinimizedGPSBanner from './components/MinimizedGPSBanner';
import { useKeyboardDetection, scrollIntoViewSafe } from './utils/keyboardDetection';
import safeStorage from './utils/safeStorage';
import './styles/mobile-ergonomics-phase1.css';

// ChatGPT-style mobile components
import MobileTypingIndicator from './components/mobile/TypingIndicator';
import JumpToBottomFAB from './components/mobile/JumpToBottomFAB';

console.log('🔄 Chatbot component loaded');

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

function Chatbot({ userLocation: propUserLocation }) {
  // CRITICAL: Initialize error state FIRST to prevent undefined errors during navigation
  const [currentError, setCurrentError] = useState(null);
  const [hasRenderError, setHasRenderError] = useState(false);
  
  // Initialize i18next for multi-language support
  const { t, i18n } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  
  // Use standard backend (not Pure LLM mode)
  const usePureLLM = false;
  
  // Component initialization state - FIX FOR NAVIGATION ISSUE
  const [isInitialized, setIsInitialized] = useState(false);
  
  // GPS location state (use prop if provided, otherwise track it ourselves)
  const [userLocation, setUserLocation] = useState(propUserLocation || null);
  const [locationPermission, setLocationPermission] = useState('unknown');
  const [locationError, setLocationError] = useState(null);
  const [showGPSBanner, setShowGPSBanner] = useState(true);
  
  // Initialize component on mount
  useEffect(() => {
    console.log('🚀 Chatbot component mounting...');
    
    // Set initialized after a small delay to ensure all state is ready
    const timer = setTimeout(() => {
      setIsInitialized(true);
      console.log('✅ Chatbot component initialized');
    }, 100);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Reset state when navigating to chat page (fixes navigation from main page issue)
  useEffect(() => {
    console.log('🔄 Route changed, resetting component state:', location.pathname);
    
    // Clear any errors
    setCurrentError(null);
    setHasRenderError(false);
    
    // Ensure component is initialized
    if (!isInitialized) {
      const timer = setTimeout(() => {
        setIsInitialized(true);
        console.log('✅ Component initialized after route change');
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [location.pathname]); // Re-run when route changes
  
  // Request and track GPS location
  useEffect(() => {
    // Skip if location already provided via props
    if (propUserLocation) {
      console.log('✅ Using provided GPS location:', propUserLocation);
      return;
    }

    const requestLocation = async () => {
      if (!('geolocation' in navigator)) {
        console.log('❌ Geolocation not supported');
        setLocationError('GPS not supported by browser');
        return;
      }

      try {
        // Check permission
        if (navigator.permissions) {
          const permission = await navigator.permissions.query({ name: 'geolocation' });
          setLocationPermission(permission.state);
          
          console.log('📍 GPS Permission status:', permission.state);
          
          permission.onchange = () => {
            console.log('📍 GPS Permission changed to:', permission.state);
            setLocationPermission(permission.state);
            if (permission.state === 'granted') {
              getCurrentLocation();
            } else if (permission.state === 'denied') {
              setUserLocation(null);
            }
          };
        }

        // Get location if permission granted
        if (locationPermission === 'granted' || locationPermission === 'prompt') {
          getCurrentLocation();
        }
      } catch (error) {
        console.error('❌ Permission check failed:', error);
      }
    };

    const getCurrentLocation = () => {
      console.log('📍 Requesting GPS location...');
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy: position.coords.accuracy
          };
          console.log('✅ GPS location obtained:', location);
          setUserLocation(location);
          setLocationError(null);
        },
        (error) => {
          console.error('❌ GPS error:', error.message);
          setLocationError(error.message);
          setUserLocation(null);
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000 // 5 minutes
        }
      );
    };

    requestLocation();
  }, [locationPermission, propUserLocation]);
  
  // Manual GPS request handler
  const requestLocationManually = () => {
    console.log('📍 Manual GPS request triggered');
    if (!('geolocation' in navigator)) {
      alert('GPS not supported by your browser');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const location = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy
        };
        console.log('✅ GPS enabled manually:', location);
        setUserLocation(location);
        setLocationError(null);
        setLocationPermission('granted');
        setShowGPSBanner(false);
      },
      (error) => {
        console.error('❌ Manual GPS error:', error);
        alert(`GPS Error: ${error.message}. Please check your browser settings.`);
        setLocationError(error.message);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );
  };
  
  // Mobile ergonomics: Keyboard detection
  const { isKeyboardVisible, keyboardHeight } = useKeyboardDetection();
  
  // Enhanced state management
  const [messages, setMessages] = useState(() => {
    return safeStorage.getJSON('chat-messages', []);
  });
  
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    return safeStorage.getJSON('dark-mode', false);
  });

  // Enhanced UI state
  const [isTyping, setIsTyping] = useState(false);
  const [typingMessage, setTypingMessage] = useState('');
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [isSessionsPanelOpen, setIsSessionsPanelOpen] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(() => {
    return safeStorage.getItem('chat_session_id') || Date.now().toString();
  });
  
  // Network and health monitoring
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [apiHealth, setApiHealth] = useState('unknown');

  // Error handling state (continued) - currentError and hasRenderError already declared at top
  const [retryAction, setRetryAction] = useState(null);
  const [lastFailedMessage, setLastFailedMessage] = useState(null);
  const [isRetrying, setIsRetrying] = useState(false);

  // ChatGPT-style mobile enhancements
  const chatMessagesRef = useRef(null);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);

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
      
      // Track unread messages (only for assistant messages when not at bottom)
      if (sender === 'assistant' && !isScrolledToBottom) {
        setUnreadCount(count => count + 1);
      }
      
      // Persist to localStorage
      safeStorage.setJSON('chat-messages', updated);
      return updated;
    });
  };

  const clearChatHistory = () => {
    setMessages([]);
    safeStorage.removeItem('chat-messages');
    console.log('🗑️ Chat history cleared');
  };

  // Session management functions
  const handleNewSession = (newSession) => {
    // Save current session before creating new one
    saveCurrentSession();
    
    // Clear current messages and start fresh
    setMessages([]);
    setCurrentSessionId(newSession.id);
    safeStorage.setItem('chat_session_id', newSession.id);
    safeStorage.removeItem('chat-messages');
    
    console.log('✨ Created new chat session:', newSession.id);
  };

  const handleSelectSession = (session) => {
    // Save current session
    saveCurrentSession();
    
    // Load selected session
    setCurrentSessionId(session.id);
    safeStorage.setItem('chat_session_id', session.id);
    
    // Load session messages
    const sessionMessages = safeStorage.getJSON(`chat-messages-${session.id}`, []);
    setMessages(sessionMessages);
    
    console.log('📂 Switched to session:', session.id);
  };

  const saveCurrentSession = () => {
    if (!currentSessionId || messages.length === 0) return;
    
    // Save messages for this session
    safeStorage.setJSON(`chat-messages-${currentSessionId}`, messages);
    
    // Update session metadata
    const sessions = safeStorage.getJSON('chat_sessions', []);
    const sessionIndex = sessions.findIndex(s => s.id === currentSessionId);
    
    if (sessionIndex >= 0) {
      sessions[sessionIndex].messageCount = messages.length;
      sessions[sessionIndex].timestamp = new Date().toISOString();
      sessions[sessionIndex].title = messages[0]?.text?.slice(0, 30) + '...' || 'Chat';
    } else {
      sessions.unshift({
        id: currentSessionId,
        title: messages[0]?.text?.slice(0, 30) + '...' || 'Current Chat',
        timestamp: new Date().toISOString(),
        messageCount: messages.length
      });
    }
    
    safeStorage.setJSON('chat_sessions', sessions);
  };

  const toggleSessionsPanel = () => {
    // Save current session before opening panel
    if (!isSessionsPanelOpen) {
      saveCurrentSession();
    }
    setIsSessionsPanelOpen(!isSessionsPanelOpen);
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
    const shareText = `KAM Assistant: ${message.text}`;
    
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'KAM Assistant Response',
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

  // Track scroll position for FAB and unread counter
  useEffect(() => {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
      const atBottom = distanceFromBottom < 50;
      
      setIsScrolledToBottom(atBottom);
      
      // Clear unread count when at bottom
      if (atBottom) {
        setUnreadCount(0);
      }
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    chatMessagesRef.current = container;
    
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    // Persist dark mode      safeStorage.setJSON('dark-mode', darkMode);
  }, [darkMode]);

  useEffect(() => {
    // Auto-save session when messages change
    if (messages.length > 0 && currentSessionId) {
      const timeoutId = setTimeout(() => {
        saveCurrentSession();
      }, 1000); // Debounce saves
      
      return () => clearTimeout(timeoutId);
    }
  }, [messages, currentSessionId]);

  // Handle initial query from navigation state (from main page search)
  useEffect(() => {
    const initialQuery = location.state?.initialQuery;
    if (initialQuery && !loading) {
      console.log('🔍 Processing initial query from navigation:', initialQuery);
      setInput(initialQuery);
      
      // Auto-submit the query after a short delay
      setTimeout(() => {
        handleSend(initialQuery);
        
        // Clear the navigation state to prevent resubmission
        navigate(location.pathname, { replace: true, state: {} });
      }, 300);
    }
  }, [location.state?.initialQuery]);

  // Error boundary handler - catch any render errors
  useEffect(() => {
    const handleError = (event) => {
      console.error('🚨 Render error caught:', event.error);
      setHasRenderError(true);
      // Prevent the error from propagating
      event.preventDefault();
    };

    const handleUnhandledRejection = (event) => {
      console.error('🚨 Unhandled promise rejection:', event.reason);
      event.preventDefault();
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

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

  // Handle initial query from navigation state
  useEffect(() => {
    const initialQuery = location.state?.from?.search || '';
    if (initialQuery) {
      console.log('🔍 Initial query from navigation state:', initialQuery);
      setInput(initialQuery);
      
      // Optionally, you can auto-submit the query
      // handleSend(initialQuery);
    }
  }, [location.state]);

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
    
    // Set retry action if we have a failed message and not already retrying
    if (failedMessage && failedMessage.input && !isRetrying) {
      setRetryAction(() => () => {
        setIsRetrying(true);
        console.log('🔄 Retrying failed message:', failedMessage.input);
        
        // Clear error state before retry
        setCurrentError(null);
        
        // Retry the send operation
        handleSend(failedMessage.input);
        
        // Reset retry flag after a delay to prevent rapid retries
        setTimeout(() => {
          setIsRetrying(false);
        }, 2000);
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
    setIsRetrying(false);
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
        console.log('🌍 Language:', i18n.language);
        
        // CRITICAL: Use sanitized input for API call
        const restaurantData = await fetchRestaurantRecommendations(sanitizedInput, {
          language: i18n.language
        });
        console.log('Restaurant API response:', restaurantData);
        const formattedResponse = formatRestaurantRecommendations(restaurantData.restaurants);
        console.log('Formatted response:', formattedResponse);
        
        addMessage(formattedResponse, 'assistant', {
          type: 'restaurant-recommendation',
          dataSource: 'google-places',
          resultCount: restaurantData.restaurants?.length || 0,
          language: i18n.language,
          restaurants: restaurantData.restaurants // Store raw restaurant data
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
        console.log('🌍 Language:', i18n.language);
        
        // CRITICAL: Use sanitized input for API call
        const placesData = await fetchPlacesRecommendations(sanitizedInput, {
          language: i18n.language
        });
        console.log('Places API response:', placesData);
        const formattedResponse = formatPlacesRecommendations(placesData.places);
        console.log('Formatted response:', formattedResponse);
        
        addMessage(formattedResponse, 'assistant', {
          type: 'places-recommendation',
          dataSource: 'database',
          resultCount: placesData?.places?.length || 0,
          language: i18n.language
        });
        
        // Clear failed message on success
        setLastFailedMessage(null);
        return;
      }

      // Regular response for non-restaurant/places queries - use SANITIZED input
      setTypingMessage('KAM is thinking...');
      
      console.log('🛡️ Sending SANITIZED input to chat API:', sanitizedInput);
      console.log('🌍 Current language:', i18n.language);
      console.log('🦙 Using Pure LLM:', usePureLLM);
      
      // Use unified chat API V2 which supports backend switching and language
      const chatResponse = await fetchUnifiedChatV2(sanitizedInput, {
        sessionId: getSessionId(),
        gpsLocation: userLocation, // Pass GPS location if available
        language: i18n.language, // Pass current language from i18next
        usePureLLM: usePureLLM // Use Pure LLM backend if enabled
      });
      
      // Add the response message
      addMessage(chatResponse.response || chatResponse.message, 'assistant', {
        type: chatResponse.intent || 'general',
        confidence: chatResponse.confidence,
        mapData: chatResponse.map_data, // Include map data if present
        method: chatResponse.method, // Include LLM method (cached/fresh)
        cached: chatResponse.cached,
        responseTime: chatResponse.response_time,
        backend: usePureLLM ? 'pure-llm' : 'standard'
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
      setTypingMessage('');
    }
  };

  const handleSampleClick = (question) => {
    // Automatically send the message
    handleSend(question);
  };

  // Show error state if render error occurred
  if (hasRenderError) {
    return (
      <div className={`flex items-center justify-center h-screen w-full ${
        darkMode ? 'bg-gray-900' : 'bg-gray-100'
      }`}>
        <div className="text-center max-w-md p-6">
          <div className="text-6xl mb-4">⚠️</div>
          <h2 className={`text-xl font-semibold mb-2 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            Something went wrong
          </h2>
          <p className={`text-sm mb-4 ${
            darkMode ? 'text-gray-400' : 'text-gray-600'
          }`}>
            The chat encountered an error. Please try refreshing the page.
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Refresh Page
          </button>
        </div>
      </div>
    );
  }

  // Show loading state until component is initialized
  if (!isInitialized) {
    return (
      <div className={`flex items-center justify-center h-screen w-full ${
        darkMode ? 'bg-gray-900' : 'bg-gray-100'
      }`}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Loading chat...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-screen w-full transition-colors duration-200 ${
      darkMode ? 'bg-gray-900' : 'bg-gray-100'
    }`}>
      
      {/* Chat Sessions Panel */}
      <ChatSessionsPanel
        darkMode={darkMode}
        isOpen={isSessionsPanelOpen}
        onClose={() => setIsSessionsPanelOpen(false)}
        currentSessionId={currentSessionId}
        onNewSession={handleNewSession}
        onSelectSession={handleSelectSession}
      />
      
      {/* Floating Action Button for chat controls */}
      <ChatHeader
        darkMode={darkMode}
        onDarkModeToggle={() => setDarkMode(!darkMode)}
        onClearHistory={clearChatHistory}
        onToggleSessionsPanel={toggleSessionsPanel}
      />

      {/* GPS Location Banner */}
      {!userLocation && showGPSBanner && locationPermission !== 'denied' && (
        <div className={`px-4 py-3 border-b ${
          darkMode ? 'bg-gray-800 border-gray-700' : 'bg-blue-50 border-blue-200'
        }`}>
          <div className="flex items-center justify-between max-w-5xl mx-auto">
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-blue-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Enable GPS for personalized recommendations near you
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={requestLocationManually}
                className="px-3 py-1.5 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors duration-200 font-medium"
              >
                Enable GPS
              </button>
              <button
                onClick={() => setShowGPSBanner(false)}
                className={`px-2 py-1.5 text-sm rounded-lg transition-colors duration-200 ${
                  darkMode ? 'text-gray-400 hover:text-gray-300' : 'text-gray-600 hover:text-gray-700'
                }`}
                aria-label="Dismiss GPS banner"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* GPS Status Indicator (when location is active) */}
      {userLocation && (
        <div className={`px-4 py-2 border-b ${
          darkMode ? 'bg-gray-800 border-gray-700' : 'bg-green-50 border-green-200'
        }`}>
          <div className="flex items-center gap-2 max-w-5xl mx-auto">
            <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0c-4.198 0-8 3.403-8 7.602 0 4.198 3.469 9.21 8 16.398 4.531-7.188 8-12.2 8-16.398 0-4.199-3.801-7.602-8-7.602zm0 11c-1.657 0-3-1.343-3-3s1.343-3 3-3 3 1.343 3 3-1.343 3-3 3z"/>
            </svg>
            <span className={`text-xs ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              GPS active • Accuracy: ±{Math.round(userLocation.accuracy)}m
            </span>
          </div>
        </div>
      )}
      
      {/* Chat Messages Container - With top padding for better UX */}
      <div className="flex-1 overflow-y-auto chat-messages pb-24 md:pb-0 pt-4 md:pt-6" id="chat-messages">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center px-4">
            <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-6 transition-colors duration-200 ${
              darkMode ? 'bg-white' : 'bg-gradient-to-br from-blue-600 to-purple-600'
            }`}>
              <svg className={`w-8 h-8 transition-colors duration-200 ${
                darkMode ? 'text-black' : 'text-white'
              }`} fill="currentColor" viewBox="0 0 24 24">
                <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
              </svg>
            </div>
            <h2 className={`text-2xl md:text-3xl font-bold mb-4 transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>How can I help you today?</h2>
            <p className={`text-center max-w-2xl text-base md:text-lg leading-relaxed mb-8 transition-colors duration-200 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              I'm your KAM assistant for exploring Istanbul. Ask me about restaurants, attractions, 
              neighborhoods, culture, history, or anything else about this amazing city!
            </p>
            
            {/* Enhanced Sample Cards with Better Mobile Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-4xl w-full px-4">
              <div 
                onClick={() => handleSampleClick('Show me the best attractions and landmarks in Istanbul')}
                className={`p-4 md:p-5 rounded-xl border-2 transition-all duration-200 cursor-pointer hover:shadow-xl hover:scale-105 transform ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600' 
                    : 'bg-white border-blue-200 hover:bg-blue-50 hover:border-blue-400 shadow-md hover:shadow-lg'
                }`}
              >
                <div className={`font-bold text-base md:text-lg mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🏛️ Top Attractions</div>
                <div className={`text-xs md:text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-700'
                }`}>Show me the best attractions and landmarks in Istanbul</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Give me restaurant advice - recommend 4 good restaurants')}
                className={`p-4 md:p-5 rounded-xl border-2 transition-all duration-200 cursor-pointer hover:shadow-xl hover:scale-105 transform ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600' 
                    : 'bg-white border-red-200 hover:bg-red-50 hover:border-red-400 shadow-md hover:shadow-lg'
                }`}
              >
                <div className={`font-bold text-base md:text-lg mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🍽️ Restaurants</div>
                <div className={`text-xs md:text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-700'
                }`}>Give me restaurant advice - recommend 4 good restaurants</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Tell me about Istanbul neighborhoods and districts to visit')}
                className={`p-4 md:p-5 rounded-xl border-2 transition-all duration-200 cursor-pointer hover:shadow-xl hover:scale-105 transform ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600' 
                    : 'bg-white border-green-200 hover:bg-green-50 hover:border-green-400 shadow-md hover:shadow-lg'
                }`}
              >
                <div className={`font-bold text-base md:text-lg mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🏘️ Neighborhoods</div>
                <div className={`text-xs md:text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-700'
                }`}>Tell me about Istanbul neighborhoods and districts to visit</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('What are the best cultural experiences and activities in Istanbul?')}
                className={`p-4 md:p-5 rounded-xl border-2 transition-all duration-200 cursor-pointer hover:shadow-xl hover:scale-105 transform ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 hover:border-gray-600' 
                    : 'bg-white border-purple-200 hover:bg-purple-50 hover:border-purple-400 shadow-md hover:shadow-lg'
                }`}
              >
                <div className={`font-bold text-base md:text-lg mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🎭 Culture & Activities</div>
                <div className={`text-xs md:text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-700'
                }`}>What are the best cultural experiences and activities in Istanbul?</div>
              </div>
            </div>
          </div>
        )}
            
        {/* Message Display Area */}
        <div className="max-w-5xl mx-auto px-4 w-full">
          {messages.map((msg, index) => (
            <div key={msg.id || index} className="group py-6">
              {msg.sender === 'user' ? (
                // USER MESSAGE - RIGHT ALIGNED (ChatGPT Style)
                <div className="flex justify-end px-4 mt-2">
                  <div className="flex flex-row-reverse items-start gap-3 max-w-[85%]">
                    {/* Avatar on right side */}
                    <div className={`w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500' 
                        : 'bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600'
                    }`}>
                      <svg className="w-4 h-4 md:w-5 md:h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    
                    {/* Message content - right aligned */}
                    <div className="flex-1 text-right">
                      <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                        darkMode ? 'text-gray-300' : 'text-gray-600'
                      }`}>You</div>
                      
                      {/* Blue bubble for user messages */}
                      <div className={`inline-block px-4 py-3 rounded-2xl text-left ${
                        darkMode
                          ? 'bg-blue-600 text-white'
                          : 'bg-blue-500 text-white'
                      }`}>
                        <div className="text-sm md:text-lg font-medium leading-[1.6] whitespace-pre-wrap">
                          {msg.text}
                        </div>
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
                  </div>
                </div>
              ) : (
                // AI MESSAGE - FULL WIDTH (ChatGPT Style)
                <div className="flex justify-start px-4 md:px-8">
                  <div className="flex items-start gap-3 w-full max-w-full">
                    {/* Avatar */}
                    <div className={`w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                        : 'bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600'
                    }`}>
                      <svg className="w-4 h-4 md:w-5 md:h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                      </svg>
                    </div>
                    
                    {/* Message content - NO BUBBLE, full width */}
                    <div className="flex-1 min-w-0">
                      <div className={`text-xs font-semibold mb-2 transition-colors duration-200 ${
                        darkMode ? 'text-gray-300' : 'text-gray-600'
                      }`}>KAM Assistant</div>
                      
                      {/* NO background, just text - ChatGPT style */}
                      <div className={`text-sm md:text-base whitespace-pre-wrap leading-[1.6] transition-colors duration-200 ${
                        darkMode ? 'text-gray-100' : 'text-gray-800'
                      }`}>
                        {renderMessageContent(msg.text || msg.content, darkMode)}
                      </div>
                      
                      {/* Restaurant Cards - Display when message has restaurant data */}
                      {msg.restaurants && msg.restaurants.length > 0 && (
                        <div className="mt-4 space-y-4">
                          <div className={`text-sm font-medium mb-3 ${
                            darkMode ? 'text-gray-300' : 'text-gray-700'
                          }`}>
                            📍 Restaurant Recommendations:
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {msg.restaurants.slice(0, 4).map((restaurant, idx) => (
                              <RestaurantCard 
                                key={restaurant.place_id || idx}
                                restaurant={restaurant}
                                index={idx}
                              />
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {msg.timestamp && (
                        <div className={`text-xs mt-2 flex items-center space-x-2 transition-colors duration-200 ${
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
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {/* Desktop typing indicator */}
          <TypingIndicator 
            isTyping={isTyping} 
            message={typingMessage}
            darkMode={darkMode}
          />
          
          {/* ChatGPT-style mobile typing indicator (more prominent) */}
          {loading && (
            <div className="flex justify-start px-4 md:px-8">
              <div className="flex items-start gap-3">
                {/* Avatar */}
                <div className={`w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                  darkMode 
                    ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                    : 'bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600'
                }`}>
                  <svg className="w-4 h-4 md:w-5 md:h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                </svg>
                </div>
                
                {/* Typing indicator */}
                <MobileTypingIndicator darkMode={darkMode} />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Scroll to bottom button (desktop) */}
      <ScrollToBottom 
        show={showScrollToBottom}
        onClick={scrollToBottom}
        darkMode={darkMode}
      />

      {/* ChatGPT-style Jump to Bottom FAB (mobile-optimized) */}
      <JumpToBottomFAB
        containerRef={chatMessagesRef}
        unreadCount={unreadCount}
        darkMode={darkMode}
        bottomOffset={100} // Above input area
      />

      {/* Enhanced Input Area - ChatGPT Style - Fixed at bottom on mobile with spacing */}
      <div className={`border-t p-4 md:relative md:bottom-auto md:left-auto md:right-auto fixed bottom-4 left-0 right-0 z-50 transition-colors duration-200 ${
        darkMode 
          ? 'bg-gray-900 border-gray-700' 
          : 'bg-white border-gray-200'
      }`} style={{ paddingBottom: 'max(1rem, env(safe-area-inset-bottom))' }}>
        <div className="max-w-5xl mx-auto">
          <SimpleChatInput
            value={input}
            onChange={setInput}
            onSend={handleSend}
            loading={loading}
            placeholder="Ask about Istanbul..."
            darkMode={darkMode}
          />
          <div className={`text-xs text-center mt-2.5 opacity-50 transition-colors duration-200 ${
            darkMode ? 'text-gray-400' : 'text-gray-500'
          }`}>
            AI-powered Istanbul travel assistant
          </div>
        </div>
      </div>

      {/* Error Notification - with safety check */}
      {currentError && typeof currentError === 'object' && (
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
      
      {/* Mobile-optimized GPS Banner - only show on mobile */}
      {window.innerWidth <= 768 && (
        <MinimizedGPSBanner 
          autoHide={true}
          autoHideDelay={5000}
          allowDismiss={true}
        />
      )}
    </div>
  );
}

export default Chatbot;
