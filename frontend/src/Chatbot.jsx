import { useState, useEffect } from 'react';
import { fetchStreamingResults, fetchRestaurantRecommendations, fetchPlacesRecommendations, extractLocationFromQuery } from './api/api';

console.log('🔄 Chatbot component loaded with restaurant functionality');

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
    /<embed\b[^<]*>/gi,                   // <embed> tags
    /<link\b[^>]*>/gi,                    // <link> tags
    /<meta\b[^>]*>/gi,                    // <meta> tags
    /<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi,   // <style> tags
    /<[^>]*>/g,                           // Remove all HTML tags
    /javascript:/gi,                      // Remove javascript: protocol
    /vbscript:/gi,                        // Remove vbscript: protocol
    /data:/gi,                            // Remove data: protocol (can be dangerous)
    /on\w+\s*=/gi,                        // Remove event handlers like onclick=
    /expression\s*\(/gi,                  // CSS expressions
  ];
  
  // Remove command injection patterns - ENHANCED  
  const commandPatterns = [
    /[;&|`]/g,                            // Command separators and backticks
    /\$\([^)]*\)/g,                       // Command substitution $(...)
    /`[^`]*`/g,                           // Command substitution `...`
    /\${[^}]*}/g,                         // Variable substitution ${...}
    /\|\s*\w+/g,                          // Pipe commands
  ];
  
  // Remove path traversal patterns
  const pathPatterns = [
    /\.\.\//g,                            // Directory traversal
    /\.\.\\{1,2}/g,                       // Windows directory traversal
    /~\//g,                               // Home directory reference
  ];
  
  // Remove template injection patterns
  const templatePatterns = [
    /\{\{[^}]*\}\}/g,                     // Handlebars/Mustache templates
    /<%[^%]*%>/g,                         // EJS/ASP templates
    /\{%[^%]*%\}/g,                       // Twig/Django templates
  ];

  let sanitized = input;
  [...sqlPatterns, ...xssPatterns, ...commandPatterns, ...pathPatterns, ...templatePatterns].forEach(pattern => {
    sanitized = sanitized.replace(pattern, ' ');
  });
  
  // Additional security: Remove excessive special characters
  sanitized = sanitized.replace(/[^\w\s\-.,!?'"çğıöşüÇĞİÖŞÜ]/g, ' ');
  
  // Normalize whitespace
  sanitized = sanitized.replace(/\s+/g, ' ').trim();
  
  return sanitized;
};

const normalizeInput = (input) => {
  if (!input || typeof input !== 'string') return '';
  
  return input
    .toLowerCase()                    // Convert to lowercase
    .replace(/\s+/g, ' ')            // Replace multiple spaces with single space
    .replace(/[\n\r\t]/g, ' ')       // Replace newlines and tabs with spaces  
    .trim();                         // Remove leading/trailing spaces
};

const addFuzzyMatching = (input) => {
  // Common typos and their corrections
  const typoCorrections = {
    // Restaurant typos
    'restaurnats': 'restaurants',
    'retaurants': 'restaurants', 
    'resturants': 'restaurants',
    'restraurants': 'restaurants',
    'restorans': 'restaurants',      // Turkish spelling
    'restaurents': 'restaurants',
    
    // Places typos  
    'plases': 'places',
    'attrctions': 'attractions',
    'atractions': 'attractions', 
    'attractons': 'attractions',
    'attracitons': 'attractions',
    
    // Location typos
    'istanbull': 'istanbul',
    'istanbool': 'istanbul', 
    'beyogul': 'beyoglu',
    'taksm': 'taksim',
    'fateh': 'fatih',
    'kadkoy': 'kadikoy',
  };
  
  let corrected = input;
  Object.entries(typoCorrections).forEach(([typo, correction]) => {
    const regex = new RegExp(`\\b${typo}\\b`, 'gi');
    corrected = corrected.replace(regex, correction);
  });
  
  return corrected;
};

const preprocessInput = (userInput) => {
  console.log('🔧 Preprocessing input:', userInput);
  
  // Step 1: Sanitize for security
  const sanitized = sanitizeInput(userInput);  
  console.log('🧹 Sanitized:', sanitized);
  
  // Step 2: Normalize whitespace and case
  const normalized = normalizeInput(sanitized);
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
    
    lastIndex = linkRegex.lastIndex;
  }
  
  // Add any remaining text after the last link
  if (lastIndex < content.length) {
    parts.push(
      <span key={`text-${lastIndex}`}>
        {content.substring(lastIndex)}
      </span>
    );
  }
  
  console.log('Generated parts:', parts.length);
  return parts.length > 0 ? parts : content;
};

// Helper function to format restaurant recommendations
const formatRestaurantRecommendations = (restaurants, locationInfo = null) => {
  console.log('formatRestaurantRecommendations called with:', { restaurants, count: restaurants?.length });
  
  if (!restaurants || restaurants.length === 0) {
    console.log('No restaurants found, returning error message');
    return "I'm sorry, I couldn't find any restaurant recommendations at the moment. Please try again or be more specific about your preferences.";
  }

  let formattedResponse = "🍽️ **Here are 4 great restaurant recommendations for you:**\n\n";
  
  restaurants.slice(0, 4).forEach((restaurant, index) => {
    const name = restaurant.name || 'Unknown Restaurant';
    const rating = restaurant.rating ? `⭐ ${restaurant.rating}` : '';
    const address = restaurant.address || restaurant.vicinity || '';
    const description = restaurant.description || 'A popular dining spot in Istanbul.';
    
    formattedResponse += `**${index + 1}. ${name}**\n`;
    if (rating) formattedResponse += `${rating}\n`;
    if (address) formattedResponse += `📍 ${address}\n`;
    formattedResponse += `${description}\n\n`;
  });

  formattedResponse += "Would you like more details about any of these restaurants or recommendations for a specific type of cuisine?";
  
  return formattedResponse;
};

// Simplified helper function - only catch very explicit restaurant+location requests
const isExplicitRestaurantRequest = (userInput) => {
  console.log('🔍 Checking for explicit restaurant request:', userInput);
  
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
    /<[^>]*>/,                      // HTML tags
    /javascript:/i,                 // JavaScript protocol
    /on\w+\s*=/i,                  // Event handlers
    /[;'"`].*(--)|(\/\*)/,         // SQL injection patterns
    /\$\([^)]*\)/,                 // Command substitution
    /\{[^}]*\}/,                   // Template injection
  ];
  
  for (const pattern of suspiciousPatterns) {
    if (pattern.test(processedInput)) {
      console.log('🛡️ Input rejected: Suspicious patterns remain after sanitization');
      return false;
    }
  }
  
  console.log('✅ Input passed security checks, using sanitized version:', processedInput);
  const input = processedInput; // Use ONLY the sanitized input

  // Only intercept very specific restaurant requests with location
  const explicitRestaurantRequests = [
    'restaurants in',        // "restaurants in Beyoglu"
    'where to eat in',       // "where to eat in Sultanahmet"
    'restaurant recommendations for', // "restaurant recommendations for Taksim"
    'good restaurants in',   // "good restaurants in Galata"
    'best restaurants in',   // "best restaurants in Kadikoy"
    'restaurants near',      // "restaurants near Taksim Square"
    'where to eat near',     // "where to eat near Galata Tower"
    'dining in',             // "dining in Beyoglu"
    'food in',               // "food in Sultanahmet"
    'eat kebab in',          // "i want eat kebab in fatih"
    'want kebab in',         // "i want kebab in beyoglu"
    'eat turkish food in',   // "eat turkish food in taksim"
    'eat in',                // "i want to eat in fatih"
    'want to eat in',        // "i want to eat in sultanahmet"
    'find restaurants in',   // "find restaurants in kadikoy"
    'show me restaurants in',// "show me restaurants in galata"
    'give me restaurants in',// "give me restaurants in taksim"
    'best place to eat in',  // "best place to eat in besiktas"
    'good place to eat in',  // "good place to eat in uskudar"
    'recommend restaurants in', // "recommend restaurants in balat"
    'suggest restaurants in',   // "suggest restaurants in eminonu"
    'kebab in',              // "kebab in fatih"
    'seafood in',            // "seafood in kadikoy"
    'pizza in',              // "pizza in sisli"
    'cafe in',               // "cafe in galata"
    'breakfast in',          // "breakfast in ortakoy"
    'brunch in',             // "brunch in bebek"
    'dinner in',             // "dinner in taksim"
    'lunch in',              // "lunch in sultanahmet"
    'eat something in',      // "eat something in karakoy"
    'hungry in',             // "hungry in balat"
    'where can i eat in',    // "where can i eat in eminonu"
    'where should i eat in', // "where should i eat in maltepe"
    'food places in',        // "food places in kadikoy"
    'local food in',         // "local food in fatih"
    'authentic food in',     // "authentic food in balat"
    'traditional food in',   // "traditional food in uskudar"
    'vegetarian in',         // "vegetarian in sisli"
    'vegan in',              // "vegan in besiktas"
    'halal in',              // "halal in uskudar"
    'rooftop in',            // "rooftop in galata"
    'restaurants around',    // "restaurants around sultanahmet"
    'eat around',            // "eat around taksim"
    'food around',           // "food around galata"
    'dining around',         // "dining around kadikoy"
    'places to eat in',      // "places to eat in beyoglu"
    'best food in',          // "best food in istanbul"
    'good food in',          // "good food in istanbul"
    'find food in',          // "find food in fatih"
    'find a restaurant in',  // "find a restaurant in kadikoy"
    'find me a restaurant in', // "find me a restaurant in sultanahmet"
    'suggest a restaurant in', // "suggest a restaurant in galata"
    'recommend a restaurant in', // "recommend a restaurant in taksim"
    'show restaurants in',   // "show restaurants in besiktas"
    'show me food in',       // "show me food in kadikoy"
    'show me places to eat in', // "show me places to eat in fatih"
    'give me food in',       // "give me food in uskudar"
    'give me a restaurant in', // "give me a restaurant in balat"
    'give me places to eat in', // "give me places to eat in eminonu"
    'any restaurants in',    // "any restaurants in maltepe"
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
    /<[^>]*>/,                      // HTML tags
    /javascript:/i,                 // JavaScript protocol
    /on\w+\s*=/i,                  // Event handlers
    /[;'"`].*(--)|(\/\*)/,         // SQL injection patterns
    /\$\([^)]*\)/,                 // Command substitution
    /\{[^}]*\}/,                   // Template injection
  ];
  
  for (const pattern of suspiciousPatterns) {
    if (pattern.test(processedInput)) {
      console.log('🛡️ Input rejected: Suspicious patterns remain after sanitization');
      return false;
    }
  }
  
  console.log('✅ Input passed security checks, using sanitized version:', processedInput);
  const input = processedInput; // Use ONLY the sanitized input

  // Only intercept very specific places/attractions requests with location
  const explicitPlacesRequests = [
    'attractions in',            // "attractions in istanbul"
    'places to visit in',        // "places to visit in sultanahmet"
    'places to see in',          // "places to see in beyoglu"
    'tourist attractions in',    // "tourist attractions in taksim"
    'sights in',                 // "sights in galata"
    'sightseeing in',           // "sightseeing in fatih"
    'landmarks in',             // "landmarks in istanbul"
    'museums in',               // "museums in sultanahmet"
    'historical places in',     // "historical places in fatih"
    'things to do in',          // "things to do in istanbul"
    'things to see in',         // "things to see in beyoglu"
    'what to visit in',         // "what to visit in taksim"
    'what to see in',           // "what to see in istanbul"
    'show me places in',        // "show me places in galata"
    'show me attractions in',   // "show me attractions in istanbul"
    'show me sights in',        // "show me sights in sultanahmet"
    'give me attractions in',   // "give me attractions in fatih"
    'find attractions in',      // "find attractions in istanbul"
    'best places to visit in',  // "best places to visit in beyoglu"
    'top attractions in',       // "top attractions in istanbul"
    'must see in',              // "must see in taksim"
    'must visit in',            // "must visit in sultanahmet"
    'popular places in',        // "popular places in galata"
    'famous places in',         // "famous places in istanbul"
    'historic sites in',        // "historic sites in fatih"
    'monuments in',             // "monuments in istanbul"
    'palaces in',               // "palaces in sultanahmet"
    'mosques in',               // "mosques in fatih"
    'churches in',              // "churches in galata"
    'towers in',                // "towers in beyoglu"
    'bridges in',               // "bridges in istanbul"
    'bazaars in',               // "bazaars in sultanahmet"
    'markets in',               // "markets in fatih"
    'parks in',                 // "parks in istanbul"
    'gardens in',               // "gardens in beyoglu"
    'waterfront in',            // "waterfront in galata"
    'bosphorus in',             // "bosphorus in istanbul"
    'golden horn in',           // "golden horn in fatih"
    'neighborhoods in',         // "neighborhoods in istanbul"
    'districts in',             // "districts in istanbul"
    'areas to explore in',      // "areas to explore in istanbul"
    'cultural sites in',        // "cultural sites in sultanahmet"
    'art galleries in',         // "art galleries in beyoglu"
    'viewpoints in',            // "viewpoints in galata"
    'photo spots in',           // "photo spots in istanbul"
    'instagram spots in',       // "instagram spots in taksim"
    'scenic places in',         // "scenic places in istanbul"
    'beautiful places in',      // "beautiful places in sultanahmet"
    'hidden gems in',           // "hidden gems in fatih"
    'off the beaten path in',   // "off the beaten path in istanbul"
    'local attractions in',     // "local attractions in beyoglu"
    'tourist spots in',         // "tourist spots in galata"
    'places of interest in',    // "places of interest in istanbul"
    'architectural sites in',   // "architectural sites in sultanahmet"
    'religious sites in',       // "religious sites in fatih"
    'shopping areas in',        // "shopping areas in taksim"
    'entertainment in',         // "entertainment in istanbul"
    'nightlife in',             // "nightlife in beyoglu"
    'activities in',            // "activities in istanbul"
    'experiences in',           // "experiences in galata"
    'tours in',                 // "tours in sultanahmet"
    'walks in',                 // "walks in fatih"
    'trips in',                 // "trips in istanbul"
    'destinations in',          // "destinations in beyoglu"
    'highlights in',            // "highlights in istanbul"
  ];

  // Only allow Istanbul or known districts
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
  const normalizedLocation = (district || location).toLowerCase().trim();
  const isIstanbulLocation = istanbulDistricts.some(districtName => 
    normalizedLocation.includes(districtName)
  );
  
  if (!isIstanbulLocation) {
    console.log('❌ Location is not Istanbul or a known district:', normalizedLocation);
    return false;
  }
  return true;
};

// Helper function to format places recommendations
const formatPlacesRecommendations = (places, locationInfo = null) => {
  console.log('formatPlacesRecommendations called with:', { places, count: places?.length });
  
  if (!places || places.length === 0) {
    console.log('No places found, returning error message');
    return "I'm sorry, I couldn't find any places or attractions at the moment. Please try again or be more specific about what you'd like to visit.";
  }

  let formattedResponse = "🏛️ **Here are amazing places to visit in Istanbul:**\n\n";
  
  places.slice(0, 6).forEach((place, index) => {
    const name = place.name || 'Unknown Place';
    const category = place.category || 'Attraction';
    const district = place.district || '';
    const description = place.description || 'A wonderful place to visit in Istanbul.';
    const googleMapsUrl = place.google_maps_url || '';
    
    formattedResponse += `**${index + 1}. ${name}**\n`;
    if (category) formattedResponse += `📍 Category: ${category}\n`;
    if (district) formattedResponse += `🗺️ District: ${district}\n`;
    formattedResponse += `${description}\n`;
    if (googleMapsUrl) formattedResponse += `[🔗 View on Google Maps](${googleMapsUrl})\n`;
    formattedResponse += `\n`;
  });

  formattedResponse += "Would you like more details about any of these places or recommendations for a specific type of attraction?";
  
  return formattedResponse;
};

function Chatbot({ onDarkModeToggle }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [darkMode, setDarkMode] = useState(true)

  // Apply dark mode class to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  const handleSend = async (customInput = null) => {
    const originalUserInput = customInput || input.trim();
    if (!originalUserInput) return;

    // CRITICAL SECURITY: Sanitize input immediately
    const sanitizedInput = preprocessInput(originalUserInput);
    
    // SECURITY CHECK: Reject if sanitization removed everything
    if (!sanitizedInput || sanitizedInput.trim().length === 0) {
      console.log('🛡️ Input rejected: Empty after sanitization');
      const errorMessage = { role: 'assistant', content: 'Sorry, your input contains invalid characters. Please try again with a different message.' };
      setMessages(prev => [...prev, errorMessage]);
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
      const errorMessage = { role: 'assistant', content: 'Sorry, your input appears to contain invalid content. Please try again with a different message.' };
      setMessages(prev => [...prev, errorMessage]);
      setLoading(false);
      return;
    }

    console.log('✅ Using sanitized input:', sanitizedInput);
    
    // Use original input for display, but sanitized for processing
    const userMessage = { role: 'user', content: originalUserInput };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    // Check if user is asking for restaurant recommendations - using SANITIZED input
    if (isExplicitRestaurantRequest(originalUserInput)) {
      try {
        console.log('Detected restaurant advice request, fetching recommendations...');
        console.log('Original input:', originalUserInput);
        console.log('🛡️ Sending SANITIZED input to backend:', sanitizedInput);
        
        // CRITICAL: Use sanitized input for API call
        const restaurantData = await fetchRestaurantRecommendations(sanitizedInput);
        console.log('Restaurant API response:', restaurantData);
        const formattedResponse = formatRestaurantRecommendations(restaurantData.restaurants);
        console.log('Formatted response:', formattedResponse);
        
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: formattedResponse }
        ]);
        setLoading(false);
        return;
      } catch (error) {
        console.error('Restaurant recommendation error:', error);
        // Fall back to regular AI response if restaurant API fails
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Sorry, I had trouble getting restaurant recommendations: ${error.message}. Let me try a different approach.` }
        ]);
        setLoading(false);
        return;
      }
    }

    // Check if user is asking for places/attractions recommendations - using SANITIZED input
    if (isExplicitPlacesRequest(originalUserInput)) {
      try {
        console.log('Detected places/attractions request, fetching recommendations...');
        console.log('Original input:', originalUserInput);
        console.log('🛡️ Sending SANITIZED input to backend:', sanitizedInput);
        
        // CRITICAL: Use sanitized input for API call
        const placesData = await fetchPlacesRecommendations(sanitizedInput);
        console.log('Places API response:', placesData);
        const formattedResponse = formatPlacesRecommendations(placesData.places);
        console.log('Formatted response:', formattedResponse);
        
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: formattedResponse }
        ]);
        setLoading(false);
        return;
      } catch (error) {
        console.error('Places recommendation error:', error);
        // Fall back to regular AI response if places API fails
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Sorry, I had trouble getting places recommendations: ${error.message}. Let me try a different approach.` }
        ]);
        setLoading(false);
        return;
      }
    }

    // Regular streaming response for non-restaurant/places queries - use SANITIZED input
    let streamedContent = '';
    let hasError = false;
    try {
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
    } catch (error) {
      hasError = true;
      console.error('Chat API Error:', error);
      const errorMessage = error.message.includes('fetch')
        ? 'Sorry, I encountered an error connecting to the server. Please make sure the backend is running on http://localhost:8001 and try again.'
        : `Sorry, there was an error: ${error.message}. Please try again.`;
      
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: errorMessage }
      ]);
    } finally {
      setLoading(false);
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
  }

  const handleSampleClick = (question) => {
    // Automatically send the message
    handleSend(question);
  }

  return (
    <div className={`flex flex-col h-screen w-full pt-16 transition-colors duration-200 ${
      darkMode ? 'bg-gray-900' : 'bg-white'
    }`}>
      
      {/* Header - Simplified since nav is handled by parent */}
      <div className={`flex items-center justify-center px-4 py-3 border-b transition-colors duration-200 ${
        darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-white'
      }`}>
        <div className="flex items-center space-x-3">
          <div className={`w-8 h-8 rounded-sm flex items-center justify-center transition-colors duration-200 ${
            darkMode ? 'bg-white' : 'bg-black'
          }`}>
            <svg className={`w-5 h-5 transition-colors duration-200 ${
              darkMode ? 'text-black' : 'text-white'
            }`} fill="currentColor" viewBox="0 0 24 24">
              <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
            </svg>
          </div>
          <h1 className={`text-lg font-semibold transition-colors duration-200 ${
            darkMode ? 'text-white' : 'text-gray-900'
          }`}>Your AI Istanbul Assistant</h1>
        </div>
      </div>

      {/* Chat Messages Container - Full screen like ChatGPT */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center px-4">
            <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-6 transition-colors duration-200 ${
              darkMode ? 'bg-white' : 'bg-black'
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
              darkMode ? 'text-gray-300' : 'text-gray-500'
            }`}>
              I'm your AI assistant for exploring Istanbul. Ask me about restaurants, attractions, 
              neighborhoods, culture, history, or anything else about this amazing city!
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl w-full px-4">
              <div 
                onClick={() => handleSampleClick('Show me the best attractions and landmarks in Istanbul')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>�️ Top Attractions</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Show me the best attractions and landmarks in Istanbul</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Give me restaurant advice - recommend 4 good restaurants')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🍽️ Restaurant Advice</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Give me restaurant advice - recommend 4 good restaurants</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Tell me about Istanbul neighborhoods and districts to visit')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>�️ Neighborhoods</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Tell me about Istanbul neighborhoods and districts to visit</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('What are the best cultural experiences and activities in Istanbul?')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🎭 Culture & Activities</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  What are the best cultural experiences and activities in Istanbul?
                </div>
              </div>
            </div>
          </div>
        )}
            
        <div className="max-w-full mx-auto px-4">
          {messages.map((msg, index) => (
            <div key={index} className="group py-4">
              <div className="flex items-start space-x-3">
                {msg.role === 'user' ? (
                  <>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500' 
                        : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500'
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
                        darkMode ? 'text-white' : 'text-gray-800'
                      }`}>
                        {renderMessageContent(msg.content, darkMode)}
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                        : 'bg-gradient-to-br from_blue-500 via-indigo-500 to-purple-500'
                    }`}>
                      <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                        darkMode ? 'text-gray-300' : 'text-gray-600'
                      }`}>AI Assistant</div>
                      <div className={`text-sm whitespace-pre-wrap leading-relaxed transition-colors duration-200 ${
                        darkMode ? 'text-white' : 'text-gray-800'
                      }`}>
                        {renderMessageContent(msg.content, darkMode)}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="group py-4">
              <div className="flex items-start space-x-3">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                  darkMode 
                    ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                    : 'bg-gradient-to-br from_blue-500 via-indigo-500 to-purple-500'
                }`}>
                  <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                  </svg>
                </div>
                <div className="flex-1">
                  <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>AI Assistant</div>
                  <div className="flex items-center space-x-1">
                    <div className={`w-1.5 h-1.5 rounded-full animate-bounce transition-colors duration-200 ${
                      darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                    }`}></div>
                    <div className={`w-1.5 h-1.5 rounded-full animate-bounce transition-colors duration-200 ${
                      darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                    }`} style={{animationDelay: '0.1s'}}></div>
                    <div className={`w-1.5 h-1.5 rounded-full animate-bounce transition-colors duration-200 ${
                      darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                    }`} style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className={`border-t p-4 transition-colors duration-200 ${
        darkMode 
          ? 'border-gray-700 bg-gray-900' 
          : 'border-gray-200 bg-white'
      }`}>
        <div className="w-full max-w-4xl mx-auto">
          <div className="relative">
            <div className={`flex items-center space-x-3 rounded-xl px-4 py-3 transition-colors duration-200 border ${
              darkMode 
                ? 'bg-gray-800 border-gray-600' 
                : 'bg-white border-gray-300'
            }`}>
              <div className="flex-1 min-h-[20px] max-h-[100px] overflow-y-auto">
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
                className={`p-2 rounded-lg transition-all duration-200 ${
                  darkMode 
                    ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-700 hover:via-indigo-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600' 
                    : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500 hover:from-blue-600 hover:via-indigo-600 hover:to-purple-600 disabled:from-gray-400 disabled:to-gray-400'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
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
          </div>
          <div className={`text-xs text-center mt-2 transition-colors duration-200 ${
            darkMode ? 'text-gray-500' : 'text-gray-500'
          }`}>
            Your AI-powered Istanbul guide
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chatbot
