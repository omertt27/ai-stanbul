import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useRef } from 'react';
import React from 'react';

/**
 * CHATBOT.JSX - Istanbul AI City Intelligence Assistant
 * 
 * CANONICAL COMPONENT HIERARCHY (Production):
 * ============================================
 * 
 * CHAT CORE:
 * - Chatbot.jsx (THIS FILE) - Main chat interface, message handling, routing
 * - ChatHeader.jsx - Top header with FAB menu (home, sessions, new chat, dark mode)
 * - ChatSessionsPanel.jsx - Session history sidebar
 * 
 * INPUT COMPONENTS:
 * - SimpleChatInput.jsx - Desktop input (primary)
 * - SmartChatInput.jsx - Mobile input with voice (primary mobile)
 * 
 * MESSAGE DISPLAY:
 * - StreamingMessage.jsx - Streaming response display
 * - ChatMessage.jsx - Sidebar message display (legacy but referenced)
 * - SwipeableMessage.jsx - Mobile message with swipe actions
 * 
 * ROUTE/MAP VISUALIZATION (PRIORITY ORDER):
 * 1. RouteCard.jsx - PRIMARY route display with map + functional CTAs
 * 2. TransportationRouteCard.jsx - Alternative route card (fallback)
 * 3. MultiRouteComparison.jsx - Multi-route comparison view
 * 4. MapVisualization.jsx - General map component
 * 
 * SPECIALIZED CARDS:
 * - TripPlanCard.jsx - Trip itinerary display
 * - RestaurantCard.jsx - Restaurant recommendation display
 * 
 * MOBILE COMPONENTS:
 * - MobileTypingIndicator.jsx - Mobile typing animation
 * - JumpToBottomFAB.jsx - Scroll to bottom FAB
 * - QuickReplies.jsx - Smart quick reply suggestions
 * - SkeletonMessage.jsx - Loading skeleton
 * - MobileErrorNotification.jsx - Mobile error UI
 * 
 * LEGACY/ARCHIVED:
 * - See /frontend/src/_archive_legacy/README.md for archived components
 * - Variants: Chatbot-*.jsx, SimpleChatbot.jsx, etc. (moved to archive)
 * - Experimental: ChatRouteIntegration, LocationAwareChatEnhancer, etc.
 */

import { 
  fetchUnifiedChat,
  fetchUnifiedChatV2,
  fetchStreamingChat,
  fetchRestaurantRecommendations, 
  fetchPlacesRecommendations, 
  extractLocationFromQuery,
  subscribeToNetworkStatus,
  checkApiHealth,
  debouncedFetchRestaurants,
  debouncedFetchPlaces,
  getSessionId,
  resetAllCircuitBreakers
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
import TripPlanCard from './components/TripPlanCard';
import SimpleChatInput from './components/SimpleChatInput';
import RestaurantCard from './components/RestaurantCard';
import RouteCard from './components/RouteCard';
import TransportationRouteCard from './components/TransportationRouteCard';
import MultiRouteComparison from './components/MultiRouteComparison';
import MinimizedGPSBanner from './components/MinimizedGPSBanner';
import StreamingMessage from './components/StreamingMessage';
import MessageRenderer from './components/MessageRenderer';
import LanguageBanner from './components/LanguageBanner';
import { useKeyboardDetection, scrollIntoViewSafe } from './utils/keyboardDetection';
import safeStorage from './utils/safeStorage';
import { trackEvent } from './utils/analytics';
import { AB_TESTS, isTreatment, trackConversion } from './utils/abTesting';
import './styles/mobile-ergonomics-phase1.css';
import './styles/mobile-chat-premium.css';

// ChatGPT-style mobile components
import MobileTypingIndicator from './components/mobile/TypingIndicator';
import JumpToBottomFAB from './components/mobile/JumpToBottomFAB';
import QuickReplies, { getSmartSuggestions } from './components/mobile/QuickReplies';
import SkeletonMessage from './components/mobile/SkeletonMessage';
import SmartChatInput from './components/mobile/SmartChatInput';
import SwipeableMessage from './components/mobile/SwipeableMessage';
import MobileErrorNotification from './components/mobile/MobileErrorNotification';

// Hooks
import useIsMobile from './hooks/useIsMobile';

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
    // Keep Turkish characters: ı, ş, ğ, ü, ö, ç, İ, Ş, Ğ, Ü, Ö, Ç
    // Keep Arabic characters for voice input
    // Keep Cyrillic for Russian voice input
    // Keep letters, numbers, spaces, hyphens, and international chars
    .replace(/[^\p{L}\p{N}\s\-'.,?!]/gu, ' ')
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
  // Handle undefined or null content
  if (!content) {
    console.warn('⚠️ renderMessageContent received undefined/null content');
    return 'Message content not available';
  }
  
  // Convert to string if not already
  const contentStr = String(content);
  
  // Convert Markdown-style links [text](url) to clickable HTML links
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  
  // Removed excessive logging that was causing console spam
  
  const parts = [];
  let lastIndex = 0;
  let match;
  
  while ((match = linkRegex.exec(contentStr)) !== null) {
    const linkText = match[1];
    const linkUrl = match[2];
    
    // Add text before the link
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {contentStr.substring(lastIndex, match.index)}
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
  if (lastIndex < contentStr.length) {
    parts.push(
      <span key={`text-${lastIndex}`}>
        {contentStr.substring(lastIndex)}
      </span>
    );
  }
  
  return parts.length > 0 ? parts : contentStr;
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
  
  // Mobile detection hook - replaces window.innerWidth checks
  const isMobile = useIsMobile();
  
  // Use standard backend (not Pure LLM mode)
  const usePureLLM = true;
  
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
        // Suppress permission check errors (expected on some browsers)
      }
    };

    const getCurrentLocation = () => {
      console.log('📍 Requesting GPS location...');
      
      // Try with high accuracy first
      const tryHighAccuracy = () => {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            const location = {
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
              accuracy: position.coords.accuracy
            };
            console.log('✅ GPS location obtained (high accuracy):', location);
            setUserLocation(location);
            setLocationError(null);
          },
          (error) => {
            // Silently try low accuracy if high accuracy fails (suppress GPS error logging)
            tryLowAccuracy();
          },
          {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
          }
        );
      };
      
      // Fallback to low accuracy GPS
      const tryLowAccuracy = () => {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            const location = {
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
              accuracy: position.coords.accuracy
            };
            console.log('✅ GPS location obtained (low accuracy):', location);
            setUserLocation(location);
            setLocationError(null);
          },
          (error) => {
            // Suppress GPS error logging for cleaner console output
            // The CoreLocation errors are expected on some systems
            
            // Enhanced error messages with troubleshooting
            let userMessage = '';
            switch(error.code) {
              case 1: // PERMISSION_DENIED
                userMessage = 'Location access denied. Please allow location in browser settings.';
                break;
              case 2: // POSITION_UNAVAILABLE
                userMessage = 'GPS signal unavailable. Try: 1) Check device location is ON, 2) Allow this site to use location, 3) Refresh the page, or 4) Use manual location entry.';
                break;
              case 3: // TIMEOUT
                userMessage = 'GPS request timeout. Signal may be weak. Try again or use manual location entry.';
                break;
              default:
                userMessage = `GPS error: ${error.message}`;
            }
            
            setLocationError(userMessage);
            setUserLocation(null);
          },
          {
            enableHighAccuracy: false, // Use less accurate but faster method
            timeout: 5000,
            maximumAge: 600000 // Allow 10-minute cached position
          }
        );
      };
      
      tryHighAccuracy();
    };

    requestLocation();
  }, [locationPermission, propUserLocation]);
  
  // Manual GPS request handler
  const requestLocationManually = () => {
    // Suppress manual GPS logging for cleaner console
    if (!('geolocation' in navigator)) {
      alert('❌ GPS not supported by your browser. Please use a modern browser with location support.');
      return;
    }

    // Show loading state
    setLocationError('Requesting GPS location...');

    // Try high accuracy first, then fallback to low accuracy
    const tryHighAccuracy = () => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy: position.coords.accuracy
          };
          console.log('✅ GPS enabled manually (high accuracy):', location);
          setUserLocation(location);
          setLocationError(null);
          setLocationPermission('granted');
          setShowGPSBanner(false);
        },
        (error) => {
          // Silently try low accuracy fallback
          tryLowAccuracy();
        },
        {
          enableHighAccuracy: true,
          timeout: 8000,
          maximumAge: 0
        }
      );
    };
    
    const tryLowAccuracy = () => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy: position.coords.accuracy
          };
          console.log('✅ GPS enabled manually (low accuracy):', location);
          setUserLocation(location);
          setLocationError(null);
          setLocationPermission('granted');
          setShowGPSBanner(false);
        },
        (error) => {
          // Suppress GPS error logging (CoreLocation errors are expected on some systems)
          
          // Enhanced error messages with actionable troubleshooting
          let errorMessage = '';
          let troubleshootingTips = '';
          
          switch(error.code) {
            case 1: // PERMISSION_DENIED
              errorMessage = '🚫 Location Access Denied';
              troubleshootingTips = '\n\nPlease enable location access:\n' +
                '• iOS: Settings → Privacy → Location Services → Safari → While Using\n' +
                '• Android: Settings → Location → App Permissions → Browser → Allow\n' +
                '• Desktop: Click the location icon in address bar';
              break;
            case 2: // POSITION_UNAVAILABLE
              errorMessage = '📡 GPS Signal Unavailable';
              troubleshootingTips = '\n\nTroubleshooting:\n' +
                '• Make sure Location Services are ON in your device settings\n' +
                '• Check that your browser has location permission\n' +
                '• Try refreshing the page\n' +
                '• If indoors, move closer to a window\n' +
                '• Use "Enter Location Manually" as alternative';
              break;
            case 3: // TIMEOUT
              errorMessage = '⏱️ GPS Request Timeout';
              troubleshootingTips = '\n\nThe GPS signal is weak. Try:\n' +
                '• Moving to an area with better signal\n' +
                '• Waiting a few moments and trying again\n' +
                '• Using manual location entry instead';
              break;
            default:
              errorMessage = `GPS Error: ${error.message}`;
              troubleshootingTips = '\n\nPlease check your browser and device location settings.';
          }
          
          alert(errorMessage + troubleshootingTips);
          setLocationError(errorMessage);
        },
        {
          enableHighAccuracy: false,
          timeout: 5000,
          maximumAge: 600000 // Allow 10-minute cached position
        }
      );
    };
    
    tryHighAccuracy();
  };
  
  // Mobile ergonomics: Keyboard detection
  const { isKeyboardVisible, keyboardHeight } = useKeyboardDetection();
  
  // Enhanced state management with migration for feedback support
  const [messages, setMessages] = useState(() => {
    const savedMessages = safeStorage.getJSON('chat-messages', []);
    // Migrate existing messages to have interaction_id for feedback buttons
    const migratedMessages = savedMessages.map(msg => {
      if (msg.sender === 'assistant' && !msg.interaction_id) {
        return {
          ...msg,
          interaction_id: `migrated_${msg.id || Date.now()}_${Math.random().toString(36).slice(2, 8)}`
        };
      }
      return msg;
    });
    // Save migrated messages if any changes were made
    if (migratedMessages.some((msg, i) => msg.interaction_id !== savedMessages[i]?.interaction_id)) {
      safeStorage.setJSON('chat-messages', migratedMessages);
    }
    return migratedMessages;
  });
  
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    // Default to dark mode, but respect user's saved preference if they've toggled it before
    const savedPreference = safeStorage.getJSON('dark-mode', null);
    return savedPreference !== null ? savedPreference : true;
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
  
  // 🌊 Streaming state - for real-time word-by-word responses
  const [enableStreaming, setEnableStreaming] = useState(true); // Toggle streaming on/off
  const [streamingText, setStreamingText] = useState(''); // Current streaming text
  const [isStreamingResponse, setIsStreamingResponse] = useState(false); // Is currently streaming
  const abortControllerRef = useRef(null); // AbortController for cancelling streaming requests
  
  // Multi-route state
  const [selectedRouteIndex, setSelectedRouteIndex] = useState(null);
  const [hoveredRouteIndex, setHoveredRouteIndex] = useState(null);
  
  // Language tracking state for LanguageBanner
  const [previousLanguage, setPreviousLanguage] = useState(i18n.language);
  const [showLanguageBanner, setShowLanguageBanner] = useState(false);
  
  // Track language changes
  useEffect(() => {
    if (i18n.language !== previousLanguage) {
      setShowLanguageBanner(true);
      setPreviousLanguage(i18n.language);
    }
  }, [i18n.language, previousLanguage]);
  
  // Stop button handler - Cancel streaming response
  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      console.log('🛑 User clicked Stop - Cancelling streaming request');
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsStreamingResponse(false);
      setStreamingText('');
      setIsTyping(false);
      setTypingMessage('');
      
      // Add a message indicating the response was cancelled
      addMessage('Response cancelled by user.', 'assistant', {
        type: 'info',
        cancelled: true
      });
    }
  };
  
  // A/B Testing: Mobile components vs standard UI
  const useMobileComponents = isTreatment(AB_TESTS.MOBILE_COMPONENTS, 50);
  const useSmartQuickReplies = isTreatment(AB_TESTS.SMART_QUICK_REPLIES, 50);
  
  // Quick replies state
  const [quickReplySuggestions, setQuickReplySuggestions] = useState([
    'Show restaurants',
    'Find attractions', 
    'Get directions',
    'Weather today'
  ]);
  const [showQuickReplies, setShowQuickReplies] = useState(true);

  // Update quick replies dynamically based on last AI message (if in treatment group)
  useEffect(() => {
    if (!useSmartQuickReplies) return; // Only for treatment group
    
    const aiMessages = messages.filter(m => m.sender === 'assistant');
    if (aiMessages.length > 0) {
      const lastAIMessage = aiMessages[aiMessages.length - 1];
      const newSuggestions = getSmartSuggestions(lastAIMessage.text, {}, i18n.language);
      setQuickReplySuggestions(newSuggestions);
      setShowQuickReplies(true);
    }
  }, [messages, useSmartQuickReplies, i18n.language]);

  // ===============================================
  // 📊 BEHAVIORAL SIGNALS TRACKING
  // Track user behavior for smart feedback system
  // ===============================================
  const lastResponseRef = useRef({ timestamp: null, interactionId: null, query: null });
  const behaviorSignalsRef = useRef({});
  
  // Track when user sends a message - detect rephrase patterns
  const trackUserMessage = (userQuery, previousInteractionId) => {
    const now = Date.now();
    const lastResponse = lastResponseRef.current;
    
    if (lastResponse.timestamp && previousInteractionId) {
      const timeSinceResponse = (now - lastResponse.timestamp) / 1000;
      
      // Check for quick rephrase (indicates dissatisfaction)
      const isQuickRephrase = timeSinceResponse < 30;
      const isVeryQuickRephrase = timeSinceResponse < 10;
      
      // Simple similarity check (could be improved with Levenshtein distance)
      const isSimilarQuery = lastResponse.query && 
        userQuery.toLowerCase().includes(lastResponse.query.toLowerCase().split(' ')[0]) ||
        lastResponse.query.toLowerCase().includes(userQuery.toLowerCase().split(' ')[0]);
      
      if (isQuickRephrase && isSimilarQuery) {
        // Store rephrase signal for the previous response
        if (!behaviorSignalsRef.current[previousInteractionId]) {
          behaviorSignalsRef.current[previousInteractionId] = {};
        }
        behaviorSignalsRef.current[previousInteractionId].rephrase_within_30s = true;
        if (isVeryQuickRephrase) {
          behaviorSignalsRef.current[previousInteractionId].rephrase_within_10s = true;
        }
        console.log(`📊 Rephrase detected for ${previousInteractionId} (${timeSinceResponse.toFixed(1)}s)`);
      }
    }
  };
  
  // Track when AI response is received
  const trackResponseReceived = (interactionId, userQuery) => {
    lastResponseRef.current = {
      timestamp: Date.now(),
      interactionId,
      query: userQuery
    };
    
    // Initialize behavior signals for this interaction
    behaviorSignalsRef.current[interactionId] = {
      session_continued: false,
      rephrase_within_30s: false,
      rephrase_within_10s: false,
      time_on_response_seconds: 0,
      copied_response: false,
      shared_response: false,
      follow_up_question: false
    };
  };
  
  // Mark session as continued when user asks follow-up
  const markSessionContinued = (previousInteractionId) => {
    if (previousInteractionId && behaviorSignalsRef.current[previousInteractionId]) {
      behaviorSignalsRef.current[previousInteractionId].session_continued = true;
      behaviorSignalsRef.current[previousInteractionId].follow_up_question = true;
    }
  };
  
  // Track copy action
  const trackCopyAction = (interactionId) => {
    if (interactionId && behaviorSignalsRef.current[interactionId]) {
      behaviorSignalsRef.current[interactionId].copied_response = true;
      console.log(`📊 Copy tracked for ${interactionId}`);
    }
  };
  
  // Track share action
  const trackShareAction = (interactionId) => {
    if (interactionId && behaviorSignalsRef.current[interactionId]) {
      behaviorSignalsRef.current[interactionId].shared_response = true;
      console.log(`📊 Share tracked for ${interactionId}`);
    }
  };
  
  // Get behavior signals for an interaction
  const getBehaviorSignals = (interactionId) => {
    const signals = behaviorSignalsRef.current[interactionId] || {};
    
    // Calculate time on response if we have the timestamp
    if (lastResponseRef.current.interactionId === interactionId) {
      signals.time_on_response_seconds = 
        (Date.now() - lastResponseRef.current.timestamp) / 1000;
    }
    
    return signals;
  };
  // ===============================================

  // Enhanced message management
  const addMessage = (text, sender = 'assistant', metadata = {}) => {
    console.log('📨 addMessage called with mapData:', metadata.mapData ? 'YES' : 'NO');
    console.log('📨 addMessage interaction_id:', metadata.interaction_id || 'NOT SET - will generate');
    if (metadata.mapData) {
      console.log('📨 mapData in addMessage:', {
        markers: metadata.mapData.markers?.length || 0,
        routes: metadata.mapData.routes?.length || 0,
        coordinates: metadata.mapData.coordinates?.length || 0
      });
    }
    
    // Generate interaction_id for assistant messages if not provided (for feedback)
    const messageId = Date.now() + Math.random();
    const interaction_id = metadata.interaction_id || 
      (sender === 'assistant' ? `frontend_${messageId.toString(36)}` : null);
    
    const newMessage = {
      id: messageId,
      text: typeof text === 'string' ? text : '',
      sender,
      timestamp: new Date().toISOString(),
      ...metadata,
      interaction_id  // Ensure interaction_id is always set for assistant messages
    };
    
    // 📊 BEHAVIORAL TRACKING
    if (sender === 'user') {
      // Track user message for rephrase detection
      const lastAssistantMsg = messages.filter(m => m.sender === 'assistant').slice(-1)[0];
      if (lastAssistantMsg?.interaction_id) {
        trackUserMessage(text, lastAssistantMsg.interaction_id);
        markSessionContinued(lastAssistantMsg.interaction_id);
      }
    } else if (sender === 'assistant' && interaction_id) {
      // Track when AI responds - store the user query that triggered this
      const lastUserMsg = messages.filter(m => m.sender === 'user').slice(-1)[0];
      trackResponseReceived(interaction_id, lastUserMsg?.text || '');
    }
    
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
  
  // Enhanced clipboard and sharing with behavioral tracking
  const copyMessageToClipboard = async (message) => {
    try {
      await navigator.clipboard.writeText(message.text);
      console.log('📋 Message copied to clipboard');
      
      // Track copy action for behavioral signals
      if (message.interaction_id) {
        trackCopyAction(message.interaction_id);
      }
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  const shareMessage = async (message) => {
    const shareText = `KAM Assistant: ${message.text}`;
    
    // Track share action for behavioral signals
    if (message.interaction_id) {
      trackShareAction(message.interaction_id);
    }
    
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

  // Handle user feedback on AI responses (for model fine-tuning)
  const [showDislikeReasons, setShowDislikeReasons] = useState(null); // interaction_id when showing reasons
  
  const dislikeReasons = [
    { id: 'incorrect', label: '❌ Incorrect', short: 'Wrong info' },
    { id: 'unclear', label: '😕 Unclear', short: 'Confusing' },
    { id: 'too_generic', label: '📋 Generic', short: 'Not specific' },
    { id: 'outdated', label: '📅 Outdated', short: 'Old info' },
    { id: 'off_topic', label: '🎯 Off Topic', short: 'Wrong answer' },
    { id: 'too_long', label: '📏 Too Long', short: 'Verbose' },
    { id: 'too_short', label: '📝 Too Short', short: 'Need more' },
  ];
  
  const handleFeedback = async (interactionId, feedbackType, dislikeReason = 'none') => {
    try {
      // If thumbs down and no reason yet, show reason selector
      if (feedbackType === 'thumbs_down' && dislikeReason === 'none' && showDislikeReasons !== interactionId) {
        setShowDislikeReasons(interactionId);
        return; // Wait for user to select a reason
      }
      
      // Collect behavioral signals for this interaction
      const behaviorSignals = getBehaviorSignals(interactionId);
      
      // Find the message to get the query/response for smart feedback
      const message = messages.find(m => m.interaction_id === interactionId);
      const userMessages = messages.filter(m => m.sender === 'user');
      const messageIndex = messages.findIndex(m => m.interaction_id === interactionId);
      const precedingUserMessage = messageIndex > 0 
        ? messages.slice(0, messageIndex).reverse().find(m => m.sender === 'user')
        : null;
      
      // Use smart feedback endpoint if available
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/feedback/submit-smart`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          interaction_id: interactionId,
          user_query: precedingUserMessage?.text || '',
          llm_response: message?.text || '',
          thumbs_up: feedbackType === 'thumbs_up',
          thumbs_down: feedbackType === 'thumbs_down',
          dislike_reason: dislikeReason,
          // Behavioral signals (flatten the object)
          session_continued: behaviorSignals.session_continued || false,
          rephrase_within_30s: behaviorSignals.rephrase_within_30s || false,
          rephrase_within_10s: behaviorSignals.rephrase_within_10s || false,
          tool_or_link_used: behaviorSignals.tool_or_link_used || false,
          time_on_response_seconds: behaviorSignals.time_on_response_seconds || 0,
          follow_up_question: behaviorSignals.follow_up_question || false,
          copied_response: behaviorSignals.copied_response || false,
          // Metadata
          intent: message?.intent || 'general',  // Include intent for pattern analysis
          session_id: currentSessionId,
          language: i18n.language,
        }),
      });

      if (!response.ok) {
        // Handle rate limiting gracefully
        if (response.status === 429) {
          console.warn('⚠️ Feedback rate limited, will retry later');
          // Still update UI to show feedback was acknowledged
          setMessages(prev => 
            prev.map(msg => 
              msg.interaction_id === interactionId 
                ? { ...msg, feedback: feedbackType, dislikeReason: dislikeReason }
                : msg
            )
          );
          setShowDislikeReasons(null);
          return; // Don't throw, just return
        }
        
        // Fall back to simple feedback endpoint
        const fallbackResponse = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/feedback/submit`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            interaction_id: interactionId,
            feedback_type: feedbackType,
            dislike_reason: dislikeReason,
          }),
        });
        
        if (!fallbackResponse.ok && fallbackResponse.status !== 429) {
          throw new Error('Failed to submit feedback');
        }
      }

      // Update message feedback state
      setMessages(prev => 
        prev.map(msg => 
          msg.interaction_id === interactionId 
            ? { ...msg, feedback: feedbackType, dislikeReason: dislikeReason }
            : msg
        )
      );
      
      // Hide reason selector
      setShowDislikeReasons(null);

      // Track feedback event in analytics
      trackEvent('feedback_submitted', {
        interaction_id: interactionId,
        feedback_type: feedbackType,
        dislike_reason: dislikeReason,
        session_id: currentSessionId,
      });

      console.log(`✅ Feedback submitted: ${feedbackType} for ${interactionId}${dislikeReason !== 'none' ? ` (reason: ${dislikeReason})` : ''}`);
    } catch (error) {
      console.error('❌ Failed to submit feedback:', error);
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

  // Track pending initial query to send after component is ready
  const [pendingInitialQuery, setPendingInitialQuery] = useState(null);
  
  // Capture initial query from navigation state IMMEDIATELY
  useEffect(() => {
    const initialQuery = location.state?.initialQuery;
    if (initialQuery) {
      console.log('🔍 Captured initial query from navigation:', initialQuery);
      setPendingInitialQuery(initialQuery);
      
      // Clear the navigation state immediately to prevent back button issues
      navigate(location.pathname, { replace: true, state: {} });
    }
  }, [location.pathname]); // Only run when pathname changes (navigation occurs)
  
  // Send the pending query once component is initialized and not loading
  useEffect(() => {
    if (pendingInitialQuery && isInitialized && !loading) {
      console.log('📤 Sending pending initial query:', pendingInitialQuery);
      
      // Set the input field
      setInput(pendingInitialQuery);
      
      // Send the query
      const submitQuery = async () => {
        try {
          await handleSend(pendingInitialQuery);
          console.log('✅ Initial query sent successfully');
          
          // Clear the pending query
          setPendingInitialQuery(null);
        } catch (error) {
          console.error('❌ Failed to send initial query:', error);
          // Keep the pending query so user can retry
        }
      };
      
      // Small delay to ensure handleSend is available
      setTimeout(submitQuery, 200);
    }
  }, [pendingInitialQuery, isInitialized, loading]);

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
    // Periodic API health checks - reduced frequency to minimize console spam
    // Skip health checks during streaming to avoid concurrent request issues
    const checkHealth = async () => {
      // Skip if currently streaming to avoid interference
      if (isStreamingResponse || loading) {
        return;
      }
      
      try {
        const result = await checkApiHealth();
        const isHealthy = result?.healthy ?? result;
        setApiHealth(isHealthy ? 'healthy' : 'unhealthy');
        
        // If API is healthy, reset circuit breakers to allow requests
        if (isHealthy) {
          resetAllCircuitBreakers();
        }
      } catch (error) {
        // Silently handle health check failures - don't spam console
        setApiHealth('error');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 60000); // Check every 60 seconds (reduced from 30)
    return () => clearInterval(interval);
  }, [isStreamingResponse, loading]);

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
    
    // Only set retry action if we're not already in a retry attempt
    // This prevents infinite retry loops on persistent errors
    if (failedMessage && failedMessage.input && !isRetrying) {
      const inputToRetry = failedMessage.input;
      setRetryAction(() => () => {
        // Don't retry if already retrying
        if (isRetrying) {
          console.log('⚠️ Retry already in progress, skipping');
          return Promise.resolve();
        }
        
        setIsRetrying(true);
        console.log('🔄 Retrying failed message:', inputToRetry);
        
        // Clear error state before retry
        setCurrentError(null);
        
        // Retry the send operation
        handleSend(inputToRetry);
        
        // Reset retry flag after a delay to prevent rapid retries
        setTimeout(() => {
          setIsRetrying(false);
        }, 2000);
        
        return Promise.resolve();
      });
    }
    
    setLoading(false);
  };

  const handleRetry = () => {
    // Prevent retry if already retrying
    if (isRetrying) {
      console.log('⚠️ Retry already in progress, ignoring duplicate request');
      return Promise.resolve();
    }
    
    if (retryAction) {
      console.log('🔄 Retrying last action');
      // Reset circuit breakers before retrying to give it a fresh chance
      resetAllCircuitBreakers();
      setCurrentError(null);
      
      // Clear the retry action to prevent duplicate retries
      const actionToRun = retryAction;
      setRetryAction(null);
      
      // Return the result of retryAction for Promise-based error handling
      const result = actionToRun();
      // Ensure we return a Promise for the caller
      return result instanceof Promise ? result : Promise.resolve(result);
    }
    return Promise.resolve();
  };

  const dismissError = () => {
    setCurrentError(null);
    setRetryAction(null);
    setLastFailedMessage(null);
    setIsRetrying(false);
    // Reset circuit breakers when user dismisses error to give fresh start
    resetAllCircuitBreakers();
  };

  // Handle quick action clicks - sends pre-defined follow-up questions
  const handleQuickAction = (query) => {
    console.log('🎯 Quick action clicked:', query);
    setInput(query);
    // Automatically send the query
    setTimeout(() => {
      handleSend(query);
    }, 100);
  };

  const handleSend = async (customInput = null) => {
    const originalUserInput = customInput || input.trim();
    if (!originalUserInput) return;

    // Track message sent (analytics)
    const messageStartTime = Date.now();
    try {
      trackEvent('chatMessage', { action: 'sent', length: originalUserInput.length });
    } catch (e) {
      console.warn('Analytics tracking failed:', e);
    }

    // Store the message info for potential retry (handleError will set retryAction if needed)
    // Don't set retryAction here - let handleError set it only when an error actually occurs
    // This prevents competing retryAction setters causing confusion
    setLastFailedMessage({ input: originalUserInput });

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
        // Note: fetchRestaurantRecommendations takes (userInput, limit) - limit is a number
        const restaurantData = await fetchRestaurantRecommendations(sanitizedInput, 6);
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
        // Note: fetchPlacesRecommendations takes (userInput, limit) - limit is a number
        const placesData = await fetchPlacesRecommendations(sanitizedInput, 6);
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
      console.log('🌊 Streaming enabled:', enableStreaming);
      
      // 🚇 TRANSPORTATION QUERIES: Always use regular API for transportation to ensure RAG and map data
      const isTransportationQuery = /\b(nasıl\s+(giderim|gidebilirim)|yolculuk|güzergah|rota|metro|otobüs|tramvay|vapur|dolmuş|minibüs|how\s+to\s+get|route|directions|transportation|transit|taksim|sultan|galata|kadıköy|üsküdar|beşiktaş)\b/i.test(sanitizedInput);
      
      if (isTransportationQuery) {
        console.log('🚇 Transportation query detected, using regular API to ensure RAG and map data');
        console.log('🛡️ Bypassing streaming for transportation query');
        
        // Use regular chat API directly (bypass streaming)
        setIsTyping(true);
        setTypingMessage('Finding the best route for you...');
        
        try {
          const chatResponse = await fetchUnifiedChatV2(sanitizedInput, {
            sessionId: getSessionId(),
            gpsLocation: userLocation,
            language: i18n.language,
            usePureLLM: false // Force standard backend for transportation queries to use RAG
          });
          
          console.log('🔍 Transportation Chat Response:', chatResponse);
          console.log('🗺️ Transportation Map Data:', chatResponse.map_data);
          console.log('🚌 Transportation Route Data:', chatResponse.route_data);
          
          addMessage(chatResponse.response || chatResponse.message, 'assistant', {
            type: chatResponse.intent || 'transportation',
            confidence: chatResponse.confidence,
            mapData: chatResponse.map_data, // Include map data if present (camelCase)
            map_data: chatResponse.map_data, // Also include snake_case version for condition compatibility
            routeData: chatResponse.route_data, // Include route_data for TransportationRouteCard (camelCase)
            route_data: chatResponse.route_data, // Also include snake_case version for condition compatibility 
            route_info: chatResponse.route_info, // Include route_info if present
            llmMode: chatResponse.llm_mode,
            intent: chatResponse.intent,
            method: 'transportation-rag',
            cached: chatResponse.cached,
            responseTime: chatResponse.response_time,
            backend: 'standard-rag',
            // 🚀 UnifiedLLM metadata for badges
            llm_backend: chatResponse.llm_backend,
            cache_hit: chatResponse.cache_hit,
            circuit_breaker_state: chatResponse.circuit_breaker_state,
            llm_latency_ms: chatResponse.llm_latency_ms
          });
          
          // Track message received (analytics)
          try {
            const responseTime = Date.now() - messageStartTime;
            const responseLength = (chatResponse.response || chatResponse.message).length;
            trackEvent('chatMessage', { action: 'received', length: responseLength, responseTime, type: 'transportation' });
          } catch (e) {
            console.warn('Analytics tracking failed:', e);
          }
          
          // Clear failed message on success
          setLastFailedMessage(null);
          setRetryAction(null);
          return;
          
        } catch (error) {
          console.error('❌ Transportation query failed:', error);
          handleError(error, 'transportation query', { input: originalUserInput });
          
          addMessage('Sorry, I encountered an error finding your route. Please try again.', 'assistant', {
            type: 'error',
            errorType: classifyError(error),
            canRetry: true,
            originalInput: originalUserInput
          });
          return;
        } finally {
          setIsTyping(false);
          setTypingMessage('');
          setLoading(false);
        }
      }
      
      // 🌊 STREAMING: If streaming is enabled and NOT a transportation query, use streaming API
      // Note: Streaming works with both Pure LLM and standard modes
      if (enableStreaming && !isTransportationQuery) {
        console.log('🌊 Starting streaming response...');
        setIsStreamingResponse(true);
        setStreamingText('');
        
        // Create AbortController for cancellation
        abortControllerRef.current = new AbortController();
        
        try {
          await fetchStreamingChat(sanitizedInput, {
            sessionId: getSessionId(),
            language: i18n.language,
            gpsLocation: userLocation,
            signal: abortControllerRef.current.signal,  // Pass abort signal
            
            onStart: (data) => {
              console.log('🚀 Streaming started:', data);
              setTypingMessage('');
              setIsTyping(false);
            },
            
            onToken: (token, fullText) => {
              setStreamingText(fullText);
            },
            
            onComplete: (finalText, metadata) => {
              console.log('🎯 onComplete called with:', { 
                finalTextType: typeof finalText, 
                finalTextLength: finalText?.length,
                streamingTextLength: streamingText?.length,
                metadataKeys: metadata ? Object.keys(metadata) : 'no metadata'
              });
              
              // Ensure finalText is never undefined/null
              const content = finalText || streamingText || '';
              
              if (!content) {
                console.error('❌ Streaming completed with NO CONTENT!', {
                  finalText,
                  streamingText,
                  metadata
                });
                // Add error message
                addMessage('Sorry, I encountered an error processing the streaming response.', 'assistant', {
                  type: 'error'
                });
                setIsStreamingResponse(false);
                setStreamingText('');
                setLoading(false);
                return;
              }
              
              console.log('✅ Streaming complete:', content.substring(0, 100) + '...');
              console.log('🗺️ Map data in metadata:', metadata?.map_data ? 'YES' : 'NO');
              console.log('🗓️ Trip plan in metadata:', metadata?.trip_plan ? 'YES' : 'NO');
              console.log('📝 Interaction ID for feedback:', metadata?.interaction_id || 'NOT SET');
              if (metadata?.map_data) {
                console.log('🗺️ Map data details:', {
                  type: metadata.map_data.type || 'route',
                  markers: metadata.map_data.markers?.length || 0,
                  routes: metadata.map_data.routes?.length || 0,
                  coordinates: metadata.map_data.coordinates?.length || 0,
                  hasRouteData: !!metadata.map_data.route_data,
                  days: metadata.map_data.days?.length || 0
                });
              }
              abortControllerRef.current = null; // Clear abort controller
              setIsStreamingResponse(false);
              setStreamingText('');
              setLoading(false);  // Reset loading state
              setIsTyping(false); // Reset typing state
              
              // Add the completed message to chat history
              addMessage(content, 'assistant', {
                type: metadata?.intent || 'general',
                confidence: metadata?.confidence,
                mapData: metadata?.map_data, // Include map data if present (camelCase)
                map_data: metadata?.map_data, // Also include snake_case version for condition compatibility
                routeData: metadata?.route_data, // Include route_data for TransportationRouteCard (camelCase)
                route_data: metadata?.route_data, // Also include snake_case version for condition compatibility 
                route_info: metadata?.route_info, // Include route_info if present
                tripPlan: metadata?.trip_plan,
                llmMode: metadata?.llm_mode,
                intent: metadata?.intent,
                method: 'streaming',
                cached: false,
                responseTime: metadata?.response_time,
                backend: 'streaming',
                interaction_id: metadata?.interaction_id  // Include for feedback
              });
              
              // Track message received (analytics)
              try {
                const responseTime = Date.now() - messageStartTime;
                trackEvent('chatMessage', { action: 'received', length: content.length, responseTime, streaming: true });
              } catch (e) {
                console.warn('Analytics tracking failed:', e);
              }
              
              // Clear failed message on success
              setLastFailedMessage(null);
            },
            
            onError: (error) => {
              console.error('❌ Streaming error:', error);
              abortControllerRef.current = null; // Clear abort controller
              setIsStreamingResponse(false);
              setStreamingText('');
              
              // Don't show error if user cancelled
              if (error.cancelled) {
                console.log('🛑 Streaming cancelled by user');
                return;
              }
              
              // Fall back to regular chat API
              fallbackToRegularChat();
            }
          });
          
          // Exit early - streaming handles its own message adding
          return;
          
        } catch (streamError) {
          console.error('❌ Streaming failed, falling back:', streamError);
          abortControllerRef.current = null; // Clear abort controller
          setIsStreamingResponse(false);
          setStreamingText('');
          
          // Don't show error if user cancelled
          if (streamError.name === 'AbortError' || streamError.cancelled) {
            console.log('🛑 Streaming cancelled by user');
            return; // Exit early
          }
          
          // Continue to regular chat API below
        }
      }
      
      // Regular (non-streaming) fallback function
      // Using function declaration for hoisting (can be called before definition)
      async function fallbackToRegularChat() {
        console.log('⚡ Using regular chat API (fallback)');
        setIsTyping(true);
        setTypingMessage('KAM is thinking...');
        
        try {
          const chatResponse = await fetchUnifiedChatV2(sanitizedInput, {
            sessionId: getSessionId(),
            gpsLocation: userLocation,
            language: i18n.language,
            usePureLLM: usePureLLM
          });
          
          console.log('🔍 Fallback Chat Response:', chatResponse);
          console.log('🗺️ Fallback Map Data:', chatResponse.map_data);
          
          addMessage(chatResponse.response || chatResponse.message, 'assistant', {
            type: chatResponse.intent || 'general',
            confidence: chatResponse.confidence,
            mapData: chatResponse.map_data, // Include map data if present (camelCase)
            map_data: chatResponse.map_data, // Also include snake_case version for condition compatibility
            routeData: chatResponse.route_data, // Include route_data for TransportationRouteCard (camelCase)
            route_data: chatResponse.route_data, // Also include snake_case version for condition compatibility 
            route_info: chatResponse.route_info, // Include route_info if present
            llmMode: chatResponse.llm_mode,
            intent: chatResponse.intent,
            method: chatResponse.method,
            cached: chatResponse.cached,
            responseTime: chatResponse.response_time,
            backend: usePureLLM ? 'pure-llm-fallback' : 'standard',
            // 🚀 UnifiedLLM metadata for badges
            llm_backend: chatResponse.llm_backend,
            cache_hit: chatResponse.cache_hit,
            circuit_breaker_state: chatResponse.circuit_breaker_state,
            llm_latency_ms: chatResponse.llm_latency_ms
          });
          
          setLastFailedMessage(null);
          
        } catch (fallbackError) {
          handleError(fallbackError, 'fallback message sending', { input: originalUserInput });
          addMessage('Sorry, I encountered an error. Please try again.', 'assistant', {
            type: 'error',
            errorType: classifyError(fallbackError),
            canRetry: true,
            originalInput: originalUserInput
          });
        } finally {
          setIsTyping(false);
          setTypingMessage('');
        }
      };
      
      // Use unified chat API V2 which supports backend switching and language
      const chatResponse = await fetchUnifiedChatV2(sanitizedInput, {
        sessionId: getSessionId(),
        gpsLocation: userLocation, // Pass GPS location if available
        language: i18n.language, // Pass current language from i18next
        usePureLLM: usePureLLM // Use Pure LLM backend if enabled
      });
      
      // Debug: Log mapData
      console.log('🔍 Chat Response:', chatResponse);
      console.log('🗺️ Map Data:', chatResponse.map_data);
      
      // Add the response message
      addMessage(chatResponse.response || chatResponse.message, 'assistant', {
        type: chatResponse.intent || 'general',
        confidence: chatResponse.confidence,
        mapData: chatResponse.map_data, // Include map data if present (camelCase)
        map_data: chatResponse.map_data, // Also include snake_case version for condition compatibility
        routeData: chatResponse.route_data, // 🔥 WEEK 2: Include route_data for TransportationRouteCard (camelCase)
        route_data: chatResponse.route_data, // Also include snake_case version for condition compatibility 
        route_info: chatResponse.route_info, // Include route_info if present
        llmMode: chatResponse.llm_mode, // 🔥 WEEK 2: Include llm_mode for conditional rendering
        intent: chatResponse.intent, // 🔥 WEEK 2: Include intent classification
        method: chatResponse.method, // Include LLM method (cached/fresh)
        cached: chatResponse.cached,
        responseTime: chatResponse.response_time,
        backend: usePureLLM ? 'pure-llm' : 'standard',
        // 🚀 UnifiedLLM metadata for badges
        llm_backend: chatResponse.llm_backend,
        cache_hit: chatResponse.cache_hit,
        circuit_breaker_state: chatResponse.circuit_breaker_state,
        llm_latency_ms: chatResponse.llm_latency_ms
      });
      
      // Track message received (analytics)
      try {
        const responseTime = Date.now() - messageStartTime;
        const responseLength = (chatResponse.response || chatResponse.message).length;
        trackEvent('chatMessage', { action: 'received', length: responseLength, responseTime });
      } catch (e) {
        console.warn('Analytics tracking failed:', e);
      }
      
      // Clear failed message on success
      setLastFailedMessage(null);
      setRetryAction(null); // Clear any retry action on success
      
    } catch (error) {
      // Pass the original input for retry purposes
      handleError(error, 'message sending', { input: originalUserInput });
      
      // Track error (analytics)
      try {
        trackEvent('error', { type: classifyError(error), message: error.message, context: 'chatbot' });
      } catch (e) {
        console.warn('Analytics tracking failed:', e);
      }
      
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
        onDarkModeToggle={() => {
          const newDarkMode = !darkMode;
          setDarkMode(newDarkMode);
          safeStorage.setJSON('dark-mode', newDarkMode);
        }}
        onClearHistory={clearChatHistory}
        onToggleSessionsPanel={toggleSessionsPanel}
      />

      {/* Language Change Banner */}
      {showLanguageBanner && (
        <LanguageBanner 
          language={i18n.language}
          darkMode={darkMode}
          onDismiss={() => setShowLanguageBanner(false)}
        />
      )}

      {/* GPS Location Banner */}
      {!userLocation && showGPSBanner && locationPermission !== 'denied' && (
        <div className={`px-4 py-3 border-b relative z-[60] ${
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
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6m-12 0l12 12" />
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
      
      {/* Chat Messages Container - Optimized for mobile */}
      <div 
        className="flex-1 overflow-y-auto chat-messages pt-2 md:pt-6" 
        id="chat-messages"
        style={{ paddingBottom: isMobile ? '120px' : '80px' }}
      >
        {messages.length === 0 && (
          <div className={`h-full flex flex-col items-center ${isMobile ? 'justify-start pt-6' : 'justify-center'} px-3 md:px-4`}>
            {/* Logo - smaller on mobile like Gemini */}
            <div className={`${isMobile ? 'w-11 h-11 mb-3' : 'w-16 h-16 mb-6'} rounded-full flex items-center justify-center transition-colors duration-200 ${
              darkMode ? 'bg-white' : 'bg-gradient-to-br from-blue-600 to-purple-600'
            }`}>
              <svg className={`${isMobile ? 'w-6 h-6' : 'w-8 h-8'} transition-colors duration-200 ${
                darkMode ? 'text-black' : 'text-white'
              }`} fill="currentColor" viewBox="0 0 24 24">
                <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
              </svg>
            </div>
            {/* Title - ChatGPT/Gemini style compact */}
            <h2 className={`${isMobile ? 'text-lg font-semibold' : 'text-2xl md:text-3xl font-bold'} mb-1 text-center transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>{isMobile ? 'Istanbul Assistant' : 'Your Istanbul City Intelligence Assistant'}</h2>
            {/* Subtitle - shorter on mobile */}
            <p className={`text-center max-w-md ${isMobile ? 'text-xs mb-5' : 'text-sm md:text-base mb-6'} leading-relaxed transition-colors duration-200 ${
              darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              {isMobile ? 'Routes, food, attractions & more' : 'Real-time transit, local insights, and personalized recommendations'}
            </p>
            
            {/* Sample Cards - 2x2 grid on mobile like ChatGPT */}
            <div className={`grid ${isMobile ? 'grid-cols-2 gap-2' : 'grid-cols-1 md:grid-cols-2 gap-3'} max-w-4xl w-full px-1`}>
              <div 
                onClick={() => handleSampleClick('How do I get from Taksim to Sultanahmet by metro?')}
                className={`${isMobile ? 'p-2.5' : 'p-4 md:p-5'} rounded-xl border transition-all duration-150 cursor-pointer active:scale-[0.97] ${
                  darkMode 
                    ? 'bg-gray-800/60 border-gray-700/50 active:bg-gray-700' 
                    : 'bg-white border-gray-200 active:bg-gray-50 shadow-sm'
                }`}
              >
                <div className={`font-medium ${isMobile ? 'text-[13px] mb-0.5' : 'text-base md:text-lg mb-2'} transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🚇 {isMobile ? 'Transit' : 'Transit Routes'}</div>
                <div className={`${isMobile ? 'text-[11px] leading-snug text-gray-400' : 'text-xs md:text-sm'} transition-colors duration-200 ${
                  darkMode ? 'text-gray-500' : 'text-gray-500'
                }`}>{isMobile ? 'Taksim → Sultanahmet' : 'How do I get from Taksim to Sultanahmet by metro?'}</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Give me restaurant advice - recommend 4 good restaurants')}
                className={`${isMobile ? 'p-2.5' : 'p-4 md:p-5'} rounded-xl border transition-all duration-150 cursor-pointer active:scale-[0.97] ${
                  darkMode 
                    ? 'bg-gray-800/60 border-gray-700/50 active:bg-gray-700' 
                    : 'bg-white border-gray-200 active:bg-gray-50 shadow-sm'
                }`}
              >
                <div className={`font-medium ${isMobile ? 'text-[13px] mb-0.5' : 'text-base md:text-lg mb-2'} transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🍽️ {isMobile ? 'Food' : 'Local Food'}</div>
                <div className={`${isMobile ? 'text-[11px] leading-snug text-gray-400' : 'text-xs md:text-sm'} transition-colors duration-200 ${
                  darkMode ? 'text-gray-500' : 'text-gray-500'
                }`}>{isMobile ? 'Best restaurants' : 'Give me restaurant advice - recommend 4 good restaurants'}</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Show me the best attractions and landmarks in Istanbul')}
                className={`${isMobile ? 'p-2.5' : 'p-4 md:p-5'} rounded-xl border transition-all duration-150 cursor-pointer active:scale-[0.97] ${
                  darkMode 
                    ? 'bg-gray-800/60 border-gray-700/50 active:bg-gray-700' 
                    : 'bg-white border-gray-200 active:bg-gray-50 shadow-sm'
                }`}
              >
                <div className={`font-medium ${isMobile ? 'text-[13px] mb-0.5' : 'text-base md:text-lg mb-2'} transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🏛️ {isMobile ? 'Explore' : 'Attractions'}</div>
                <div className={`${isMobile ? 'text-[11px] leading-snug text-gray-400' : 'text-xs md:text-sm'} transition-colors duration-200 ${
                  darkMode ? 'text-gray-500' : 'text-gray-500'
                }`}>{isMobile ? 'Top landmarks' : 'Show me the best attractions and landmarks in Istanbul'}</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('What\'s the weather like today and should I bring an umbrella?')}
                className={`${isMobile ? 'p-2.5' : 'p-4 md:p-5'} rounded-xl border transition-all duration-150 cursor-pointer active:scale-[0.97] ${
                  darkMode 
                    ? 'bg-gray-800/60 border-gray-700/50 active:bg-gray-700' 
                    : 'bg-white border-gray-200 active:bg-gray-50 shadow-sm'
                }`}
              >
                <div className={`font-medium ${isMobile ? 'text-[13px] mb-0.5' : 'text-base md:text-lg mb-2'} transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>☀️ {isMobile ? 'Weather' : 'Weather & Alerts'}</div>
                <div className={`${isMobile ? 'text-[11px] leading-snug text-gray-400' : 'text-xs md:text-sm'} transition-colors duration-200 ${
                  darkMode ? 'text-gray-500' : 'text-gray-500'
                }`}>{isMobile ? 'Today\'s forecast' : 'What\'s the weather like today and should I bring an umbrella?'}</div>
              </div>
            </div>
          </div>
        )}
            
        {/* Message Display Area */}
        <div className={`max-w-5xl mx-auto ${isMobile ? 'px-2' : 'px-4'} w-full`}>
          {messages.map((msg, index) => (
            <div key={msg.id || index} className={`group ${isMobile ? 'py-3' : 'py-6'}`}>
              {msg.sender === 'user' ? (
                // USER MESSAGE - RIGHT ALIGNED (ChatGPT Style)
                <div className={`flex justify-end ${isMobile ? 'px-1' : 'px-4'} mt-1`}>
                  <div className={`flex flex-row-reverse items-start ${isMobile ? 'gap-2' : 'gap-3'} max-w-[90%]`}>
                    {/* Avatar on right side - smaller on mobile */}
                    <div className={`${isMobile ? 'w-6 h-6' : 'w-8 h-8 md:w-10 md:h-10'} rounded-full flex items-center justify-center flex-shrink-0 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500' 
                        : 'bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600'
                    }`}>
                      <svg className={`${isMobile ? 'w-3 h-3' : 'w-4 h-4 md:w-5 md:h-5'} text-white`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    
                    {/* Message content - right aligned */}
                    <div className="flex-1 text-right">
                      {!isMobile && (
                        <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                          darkMode ? 'text-gray-300' : 'text-gray-600'
                        }`}>You</div>
                      )}
                      
                      {/* Blue bubble for user messages - smaller on mobile */}
                      <div className={`inline-block ${isMobile ? 'px-3 py-2 rounded-2xl' : 'px-4 py-3 rounded-2xl'} text-left ${
                        darkMode
                          ? 'bg-blue-600 text-white'
                          : 'bg-blue-500 text-white'
                      }`}>
                        <div className={`${isMobile ? 'text-[14px]' : 'text-sm md:text-lg'} font-medium leading-[1.5] whitespace-pre-wrap`}>
                          {msg.text}
                        </div>
                      </div>
                      
                      {msg.timestamp && !isMobile && (
                        <div className={`text-xs mt-1 transition-colors duration-200 ${
                          darkMode ? 'text-gray-500' : 'text-gray-500'
                        }`}>
                          {new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                        </div>
                      )}
                    </div>
                    
                    {!isMobile && (
                      <MessageActions 
                        message={msg}
                        onCopy={copyMessageToClipboard}
                        onShare={shareMessage}
                        darkMode={darkMode}
                      />
                    )}
                  </div>
                </div>
              ) : (
                // AI MESSAGE - FULL WIDTH (ChatGPT Style)
                <div className={`flex justify-start ${isMobile ? 'px-1' : 'px-4 md:px-8'}`}>
                  {isMobile ? (
                    // Mobile: Swipeable message with compact layout
                    <SwipeableMessage
                      onSwipeLeft={() => {
                        // Delete message
                        setMessages(prev => prev.filter(m => m.id !== msg.id));
                      }}
                      onSwipeRight={() => {
                        // Copy message
                        copyMessageToClipboard(msg);
                      }}
                      leftAction="delete"
                      rightAction="copy"
                      darkMode={darkMode}
                    >
                      <div className="flex items-start gap-2 w-full max-w-full">
                        {/* Avatar - smaller on mobile like ChatGPT */}
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${
                          darkMode 
                            ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                            : 'bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600'
                        }`}>
                          <svg className="w-3.5 h-3.5 text-white" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                        </svg>
                        </div>
                        
                        {/* Message content - NO BUBBLE, full width */}
                        <div className="flex-1 min-w-0">
                          {/* Hide verbose indicators on mobile for cleaner look */}
                          
                          {/* Message text - optimized size for mobile */}
                          <div 
                            className={`text-[14px] whitespace-pre-wrap leading-[1.55] select-text ${
                              darkMode ? 'text-gray-100' : 'text-gray-800'
                            }`}
                            style={{ 
                              maxWidth: '100%',
                              wordWrap: 'break-word',
                              overflowWrap: 'break-word'
                            }}
                          >
                            {renderMessageContent(msg.text || msg.content, darkMode)}
                          </div>
                          
                          {/* Route Card for mobile - show if route data present */}
                          {(msg.route_info || msg.map_data || msg.mapData || msg.routeData || (msg.data && (msg.data.route_info || msg.data.map_data))) && (
                              <div className="mt-3">
                                <RouteCard routeData={msg} />
                              </div>
                          )}
                        </div>
                      </div>
                    </SwipeableMessage>
                  ) : (
                    // Desktop: Regular message (same as mobile but without swipe wrapper)
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
                        <div className="flex items-center gap-2 mb-2">
                          <div className={`text-xs font-semibold transition-colors duration-200 ${
                            darkMode ? 'text-gray-300' : 'text-gray-600'
                          }`}>KAM Assistant</div>
                          
                          {/* 🔥 WEEK 2: LLM Mode Indicator */}
                          {msg.llmMode && msg.llmMode !== 'general' && (
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                              msg.llmMode === 'explain' 
                                ? darkMode ? 'bg-blue-900/50 text-blue-200' : 'bg-blue-100 text-blue-700'
                                : msg.llmMode === 'clarify'
                                ? darkMode ? 'bg-yellow-900/50 text-yellow-200' : 'bg-yellow-100 text-yellow-700'
                                : darkMode ? 'bg-red-900/50 text-red-200' : 'bg-red-100 text-red-700'
                            }`}>
                              {msg.llmMode === 'explain' ? '🚇 Route' : msg.llmMode === 'clarify' ? '❓ Clarifying' : '⚠️ Error'}
                            </span>
                          )}
                          
                          {/* Cache indicator */}
                          {msg.cached && (
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                              darkMode ? 'bg-green-900/50 text-green-200' : 'bg-green-100 text-green-700'
                            }`}>
                              ⚡ Cached
                            </span>
                          )}
                          
                          {/* 🚀 UnifiedLLM Backend Indicator */}
                          {msg.llm_backend && (
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                              msg.llm_backend === 'vllm'
                                ? darkMode ? 'bg-purple-900/50 text-purple-200' : 'bg-purple-100 text-purple-700'
                                : darkMode ? 'bg-orange-900/50 text-orange-200' : 'bg-orange-100 text-orange-700'
                            }`} title={`Backend: ${msg.llm_backend.toUpperCase()}`}>
                              {msg.llm_backend === 'vllm' ? '🚀 vLLM' : '🔄 Groq'}
                            </span>
                          )}
                          
                          {/* 🎯 Circuit Breaker State */}
                          {msg.circuit_breaker_state && msg.circuit_breaker_state !== 'closed' && (
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                              darkMode ? 'bg-yellow-900/50 text-yellow-200' : 'bg-yellow-100 text-yellow-700'
                            }`} title={`Circuit Breaker: ${msg.circuit_breaker_state}`}>
                              ⚠️ {msg.circuit_breaker_state === 'open' ? 'Fallback' : 'Half-Open'}
                            </span>
                          )}
                          
                          {/* ⏱️ Response Time (if available) */}
                          {msg.llm_latency_ms && (
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                              darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'
                            }`} title={`LLM Latency: ${msg.llm_latency_ms}ms`}>
                              ⏱️ {msg.llm_latency_ms}ms
                            </span>
                          )}
                        </div>
                        
                        {/* NO background, just text - ChatGPT style */}
                        <div 
                          className={`text-sm md:text-base whitespace-pre-wrap leading-[1.6] transition-colors duration-200 select-text ${
                            darkMode ? 'text-gray-100' : 'text-gray-800'
                          }`}
                          style={{ 
                            display: 'block',
                            visibility: 'visible',
                            opacity: 1,
                            maxWidth: '100%',
                            wordWrap: 'break-word',
                            overflowWrap: 'break-word'
                          }}
                        >
                          {renderMessageContent(msg.text || msg.content, darkMode)}
                        </div>
                        
                        {/* Copy Button - Easy copy functionality */}
                        <button
                          onClick={() => {
                            const textToCopy = msg.text || msg.content || '';
                            navigator.clipboard.writeText(textToCopy).then(() => {
                              const btn = document.getElementById(`copy-btn-${msg.id || index}`);
                              if (btn) {
                                btn.textContent = '✓ Copied!';
                                setTimeout(() => {
                                  btn.textContent = '📋 Copy';
                                }, 2000);
                              }
                            }).catch(err => console.error('Copy failed:', err));
                          }}
                          id={`copy-btn-${msg.id || index}`}
                          className={`mt-2 px-2 py-1 text-xs rounded transition-all duration-200 ${
                            darkMode 
                              ? 'text-gray-400 hover:text-gray-200 hover:bg-gray-700' 
                              : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                          }`}
                          title="Copy message"
                        >
                          📋 Copy
                        </button>
                        
                        {/* Restaurant Cards */}
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
                        
                        {/* PRIORITY 1: Enhanced Route Card with Map - Mobile-style visualization */}
                        {/* Shows when backend provides route_info + map_data (new route system) */}
                        {(msg.route_info || msg.map_data || msg.mapData || msg.routeData || (msg.data && (msg.data.route_info || msg.data.map_data))) && (
                            <div className="mt-4">
                              <RouteCard routeData={msg} />
                            </div>
                        )}
                        
                        {/* PRIORITY 2: FALLBACK - Transportation Route Card (Legacy) */}
                        {/* Only show if RouteCard data is NOT present */}
                        {!(msg.route_info || msg.map_data || (msg.data && (msg.data.route_info || msg.data.map_data))) && 
                         (msg.routeData || (msg.mapData && msg.mapData.route_data && msg.mapData.type !== 'trip_plan')) && (
                          <TransportationRouteCard 
                            routeData={msg.routeData || msg.mapData?.route_data}
                            darkMode={darkMode}
                          />
                        )}
                        
                        {/* Multi-Route Comparison - Show when multiple route alternatives available */}
                        {/* Only if NOT showing single route cards above */}
                        {!(msg.route_info || msg.map_data) &&
                         msg.mapData && (msg.mapData.multi_routes || msg.mapData.alternatives) && (
                          msg.mapData.multi_routes?.length > 0 || msg.mapData.alternatives?.length > 0
                        ) && (
                          <MultiRouteComparison
                            routes={msg.mapData.multi_routes || msg.mapData.alternatives || []}
                            primaryRoute={msg.mapData.primary_route}
                            routeComparison={msg.mapData.route_comparison || {}}
                            onRouteSelect={(route, index) => {
                              console.log('Selected route:', index, route);
                              setSelectedRouteIndex(index);
                            }}
                            darkMode={darkMode}
                            className="mt-4"
                          />
                        )}
                        
                        {/* Trip Plan Card - Show for multi-day trip planning queries */}
                        {(msg.tripPlan || (msg.mapData && msg.mapData.type === 'trip_plan')) && (
                          <TripPlanCard 
                            tripPlan={msg.tripPlan || msg.mapData}
                          />
                        )}
                        
                        {/* Map Visualization - ONLY show if NO route cards are displayed */}
                        {/* Prevents double-map display (RouteCard already has a map) */}
                        {!(msg.route_info || msg.map_data || msg.routeData) &&
                         msg.mapData && msg.mapData.type !== 'trip_plan' && 
                         (msg.mapData.markers || msg.mapData.coordinates) && 
                         !(msg.mapData.multi_routes || msg.mapData.alternatives) && (
                          <div className="mt-4">
                            <div className={`text-sm font-medium mb-3 ${
                              darkMode ? 'text-gray-300' : 'text-gray-700'
                            }`}>
                              🗺️ Map View:
                            </div>
                            <MapVisualization 
                              mapData={msg.mapData} 
                              height="400px" 
                              className="rounded-lg shadow-md"
                              selectedRouteIndex={selectedRouteIndex}
                              onRouteHover={setHoveredRouteIndex}
                            />
                            <div className={`text-xs mt-2 text-center transition-colors duration-200 ${
                              darkMode ? 'text-gray-500' : 'text-gray-600'
                            }`}>
                              📍 {msg.mapData.markers?.length || 0} locations
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
                        
                        {/* Feedback Buttons - For Model Fine-tuning Data Collection */}
                        {msg.interaction_id && (
                          <div className="mt-3 flex items-center gap-2">
                            <button
                              onClick={() => handleFeedback(msg.interaction_id, 'thumbs_up')}
                              disabled={msg.feedback === 'thumbs_up' || msg.feedback === 'thumbs_down'}
                              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                                msg.feedback === 'thumbs_up'
                                  ? darkMode
                                    ? 'bg-green-600 text-white'
                                    : 'bg-green-500 text-white'
                                  : darkMode
                                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                              } disabled:opacity-50 disabled:cursor-not-allowed`}
                              aria-label="Helpful response"
                            >
                              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                            </svg>
                            {msg.feedback === 'thumbs_up' ? 'Helpful!' : 'Helpful'}
                            </button>
                            
                            <button
                              onClick={() => handleFeedback(msg.interaction_id, 'thumbs_down')}
                              disabled={msg.feedback === 'thumbs_up' || msg.feedback === 'thumbs_down'}
                              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                                msg.feedback === 'thumbs_down'
                                  ? darkMode
                                    ? 'bg-red-600 text-white'
                                    : 'bg-red-500 text-white'
                                  : darkMode
                                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                              } disabled:opacity-50 disabled:cursor-not-allowed`}
                              aria-label="Not helpful"
                            >
                              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
                              </svg>
                              {msg.feedback === 'thumbs_down' ? 'Not helpful' : 'Not helpful'}
                            </button>
                            
                            {msg.feedback && (
                              <span className={`text-xs ml-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                Thanks for your feedback!
                              </span>
                            )}
                          </div>
                        )}
                        
                        {/* Dislike Reason Selector - One-click reasons */}
                        {showDislikeReasons === msg.interaction_id && (
                          <div className={`mt-2 p-3 rounded-lg ${darkMode ? 'bg-gray-700/50' : 'bg-gray-100'}`}>
                            <p className={`text-xs mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                              What was wrong with this response?
                            </p>
                            <div className="flex flex-wrap gap-1.5">
                              {dislikeReasons.map((reason) => (
                                <button
                                  key={reason.id}
                                  onClick={() => handleFeedback(msg.interaction_id, 'thumbs_down', reason.id)}
                                  className={`px-2.5 py-1.5 text-xs rounded-md font-medium transition-all duration-200 ${
                                    darkMode
                                      ? 'bg-gray-600 text-gray-200 hover:bg-red-600 hover:text-white'
                                      : 'bg-white text-gray-700 hover:bg-red-500 hover:text-white border border-gray-200'
                                  }`}
                                >
                                  {reason.label}
                                </button>
                              ))}
                              <button
                                onClick={() => setShowDislikeReasons(null)}
                                className={`px-2.5 py-1.5 text-xs rounded-md font-medium transition-all ${
                                  darkMode
                                    ? 'text-gray-400 hover:text-gray-200'
                                    : 'text-gray-500 hover:text-gray-700'
                                }`}
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          
          {/* 🌊 STREAMING MESSAGE: Show real-time streaming response */}
          {isStreamingResponse && streamingText && (
            <div className="group py-6">
              <div className="flex justify-start px-4 md:px-8">
                <div className="flex items-start gap-3 w-full max-w-full">
                  {/* Avatar */}
                  <div className={`w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                    darkMode 
                      ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                      : 'bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600'
                  }`}>
                    <svg className="w-4 h-4 md:w-5 md:h-5 text-white animate-pulse" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                  </svg>
                  </div>
                  
                  {/* Streaming message content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <div className="flex items-center gap-2">
                        <div className={`text-xs font-semibold transition-colors duration-200 ${
                          darkMode ? 'text-gray-300' : 'text-gray-600'
                        }`}>KAM Assistant</div>
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium animate-pulse ${
                          darkMode ? 'bg-blue-900/50 text-blue-200' : 'bg-blue-100 text-blue-700'
                        }`}>
                          🌊 Streaming...
                        </span>
                      </div>
                      
                      {/* ChatGPT-style Stop button */}
                      <button
                        onClick={handleStopStreaming}
                        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 hover:scale-105 active:scale-95 ${
                          darkMode 
                            ? 'bg-red-900/30 text-red-300 hover:bg-red-900/50 border border-red-700/50' 
                            : 'bg-red-50 text-red-700 hover:bg-red-100 border border-red-200'
                        }`}
                        title="Stop generating response"
                      >
                        <div className="flex items-center gap-1.5">
                          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <rect x="4" y="4" width="12" height="12" rx="1" />
                          </svg>
                          Stop
                        </div>
                      </button>
                    </div>
                    
                    <StreamingMessage
                      text={streamingText}
                      isStreaming={true}
                      isBot={true}
                      showCursor={true}
                      enableMarkdown={true}
                      className={`text-sm md:text-base whitespace-pre-wrap leading-[1.6] ${
                        darkMode ? 'text-gray-100' : 'text-gray-800'
                      }`}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Desktop typing indicator */}
          <TypingIndicator 
            isTyping={isTyping} 
            message={typingMessage}
            darkMode={darkMode}
          />
          
          {/* ChatGPT-style mobile typing indicator + skeleton loader */}
          {loading && (
            <div className="flex justify-start px-4 md:px-8">
              <div className="flex items-start gap-3 w-full">
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
                
                {/* Mobile: Show SkeletonMessage, Desktop: Show typing indicator */}
                <div className="flex-1">
                  {isMobile ? (
                    <SkeletonMessage darkMode={darkMode} count={1} />
                  ) : (
                    <MobileTypingIndicator darkMode={darkMode} />
                  )}
                </div>
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
        bottomOffset={isMobile ? 120 : 140} // Raised higher to avoid overlapping with input
      />

      {/* Quick Reply Suggestions - Hidden on mobile for cleaner UI */}
      {!isMobile && (
        <QuickReplies
          suggestions={quickReplySuggestions}
          onSelect={(suggestion, index) => {
            // Track quick reply click (analytics)
            try {
              trackEvent('quickReply', { suggestion, index: index || 0, type: useSmartQuickReplies ? 'smart' : 'static' });
              // Track A/B test conversion
              trackConversion(AB_TESTS.SMART_QUICK_REPLIES, 'quick_reply_click', { suggestion });
              if (useMobileComponents) {
                trackConversion(AB_TESTS.MOBILE_COMPONENTS, 'feature_usage', { feature: 'quick_replies' });
              }
            } catch (e) {
              console.warn('Analytics tracking failed:', e);
            }
            handleSend(suggestion);
            setShowQuickReplies(false);
          }}
          darkMode={darkMode}
          visible={showQuickReplies && !loading}
        />
      )}

      {/* Enhanced Input Area - ChatGPT/Gemini Style - Floating on mobile */}
      <div className={`${isMobile 
        ? 'fixed left-0 right-0 z-50 px-2' 
        : 'border-t p-4'
      } transition-colors duration-200 ${
        darkMode 
          ? isMobile ? 'bg-gray-900' : 'bg-gray-900 border-gray-700'
          : isMobile ? 'bg-white' : 'bg-white border-gray-200'
      }`} style={isMobile ? { 
        bottom: 0,
        paddingBottom: 'max(4px, env(safe-area-inset-bottom))',
        paddingTop: '6px'
      } : { 
        paddingBottom: '1rem' 
      }}>
        <div className={`${isMobile ? 'max-w-full' : 'max-w-5xl'} mx-auto`}>
          {/* Use SmartChatInput on mobile, SimpleChatInput on desktop */}
          {isMobile ? (
            <SmartChatInput
              value={input}
              onChange={setInput}
              onSend={() => handleSend(input)}
              loading={loading}
              placeholder="Ask anything about Istanbul..."
              darkMode={darkMode}
              enableVoice={true}
              showCharCounter={false}
              minimal={true}
              voiceLanguage="auto"
              showLanguagePicker={true}
            />
          ) : (
            <SimpleChatInput
              value={input}
              onChange={setInput}
              onSend={handleSend}
              loading={loading}
              placeholder="Ask about routes, places, or local tips..."
              darkMode={darkMode}
            />
          )}
        </div>
      </div>

      {/* Error Notification - Mobile optimized */}
      {currentError && typeof currentError === 'object' && (
        <>
          {isMobile ? (
            <MobileErrorNotification
              error={currentError}
              onRetry={handleRetry}
              onDismiss={dismissError}
              darkMode={darkMode}
              autoRetry={false}  // Disabled: prevents infinite retry loops on persistent backend errors
              maxRetries={3}
            />
          ) : (
            <ErrorNotification
              error={currentError}
              onRetry={handleRetry}
              onDismiss={dismissError}
              autoHide={false}
              darkMode={darkMode}
            />
          )}
        </>
      )}
      
      {/* Network Status Indicator */}
      <NetworkStatusIndicator darkMode={darkMode} />
    </div>
  );
}

export default Chatbot;
