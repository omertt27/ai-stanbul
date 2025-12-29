/**
 * Response Sanitizer - Prevent Data Leakage in Frontend
 * ======================================================
 * 
 * Filters and cleans LLM responses to prevent internal conversation
 * context, prompts, and metadata from being displayed to users.
 * 
 * This is a defense-in-depth layer - backend should already sanitize,
 * but this catches any leakage that slips through.
 * 
 * Created: December 29, 2025
 */

// Patterns that indicate internal conversation structure leaked into response
const LEAKAGE_PATTERNS = [
  /---\s*User:/g,               // Conversation history markers
  /Response:\s*.+?(?:---|$)/gs, // Response markers
  /Turn \d+:/g,                 // Turn numbering
  /Bot:\s*.+?\n/g,              // Bot response labels
  /Intent:\s*.+?\n/g,           // Intent labels  
  /Locations:\s*.+?\n/g,        // Location lists
  /Session Context:/g,          // Session context headers
  /Last Mentioned/g,            // Context metadata
  /User's GPS Location/g,       // GPS metadata
  /Active Task:/g,              // Task tracking
  /User Preferences:/g,         // Preference data
  /Conversation Age:/g,         // Conversation stats
  /YOUR TASK:/g,                // Prompt headers
  /RETURN FORMAT/g,             // Format instructions
  /EXAMPLES:/g,                 // Example sections
  /"has_references"/g,          // JSON analysis output
  /"resolved_references"/g,     // Reference resolution
  /"implicit_context"/g,        // Context analysis
  /"needs_clarification"/g,     // Clarification flags
  /CONVERSATION HISTORY:/g,     // History section
  /CURRENT QUERY:/g,            // Query markers
];

// Quick check patterns (for fast detection)
const QUICK_CHECK_PATTERNS = [
  '--- User:',
  'Response:',
  'Turn 1:',
  'Session Context:',
  '"has_references"',
  'YOUR TASK:',
  'RETURN FORMAT',
  'CONVERSATION HISTORY:',
];

/**
 * Check if a response contains data leakage
 * @param {string} response - Response text to check
 * @returns {boolean} True if response contains leakage
 */
export const hasLeakage = (response) => {
  if (!response || typeof response !== 'string') return false;
  
  return QUICK_CHECK_PATTERNS.some(pattern => response.includes(pattern));
};

/**
 * Sanitize response to remove any leaked internal data
 * @param {string} response - Raw response text
 * @returns {string} Cleaned response safe to display
 */
export const sanitizeResponse = (response) => {
  if (!response || typeof response !== 'string') return response;
  
  let cleaned = response;
  const originalLength = cleaned.length;
  
  // Check for conversation history leakage
  if (cleaned.includes('--- User:') || cleaned.includes('Response:')) {
    console.warn('⚠️ Frontend detected conversation history leakage!');
    
    // Strategy 1: Extract content before history markers
    const firstHistoryMarker = cleaned.indexOf('--- User:');
    if (firstHistoryMarker > 50) {
      cleaned = cleaned.substring(0, firstHistoryMarker).trim();
      console.log(`✅ Extracted ${cleaned.length} chars before history marker`);
    } else {
      // Strategy 2: Find last "Response:" and get text after it
      const lastResponseIdx = cleaned.lastIndexOf('Response:');
      if (lastResponseIdx !== -1) {
        let afterResponse = cleaned.substring(lastResponseIdx + 'Response:'.length).trim();
        
        // Clean up trailing markers
        const dashIdx = afterResponse.indexOf('---');
        if (dashIdx > 0) {
          afterResponse = afterResponse.substring(0, dashIdx).trim();
        }
        
        if (afterResponse.length > 20) {
          cleaned = afterResponse;
          console.log(`✅ Extracted final response (${cleaned.length} chars)`);
        }
      }
    }
  }
  
  // Remove remaining leakage patterns
  LEAKAGE_PATTERNS.forEach(pattern => {
    cleaned = cleaned.replace(pattern, '');
  });
  
  // Clean up excessive whitespace
  cleaned = cleaned.replace(/\n{3,}/g, '\n\n').trim();
  
  // Validate we have content
  if (!cleaned || cleaned.length < 10) {
    console.error('❌ Sanitization removed too much content');
    return "I'm sorry, but I encountered an issue with the response. Please try again.";
  }
  
  if (cleaned.length < originalLength) {
    console.log(`✅ Sanitized: removed ${originalLength - cleaned.length} chars of internal data`);
  }
  
  return cleaned;
};

/**
 * Process API response and sanitize if needed
 * @param {Object} data - API response data
 * @returns {Object} Sanitized response data
 */
export const processApiResponse = (data) => {
  if (!data) return data;
  
  // Sanitize the main response text
  if (data.response) {
    data.response = sanitizeResponse(data.response);
  }
  
  // Also check message field (some endpoints use this)
  if (data.message && typeof data.message === 'string') {
    data.message = sanitizeResponse(data.message);
  }
  
  return data;
};

export default {
  hasLeakage,
  sanitizeResponse,
  processApiResponse
};
