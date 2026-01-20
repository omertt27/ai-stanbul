/**
 * Conversation Context Manager
 * =============================
 * Manages multi-turn conversation context for better LLM responses
 * Tracks conversation history, user preferences, and session context
 * 
 * Features:
 * - Multi-turn conversation tracking
 * - User preference storage
 * - Context summarization for LLM
 * - Personalized recommendations
 */

const CONTEXT_STORAGE_KEY = 'kam_conversation_context';
const PREFERENCES_STORAGE_KEY = 'kam_user_preferences';
const MAX_CONTEXT_MESSAGES = 10; // Keep last 10 messages for context

class ConversationContextManager {
  constructor() {
    this.contextHistory = [];
    this._userPreferences = null; // Lazy load
    this.sessionMetadata = {};
    this._initialized = false;
  }

  /**
   * Lazy initialization
   */
  _ensureInitialized() {
    if (this._initialized) return;
    this._userPreferences = this.loadPreferences();
    this._initialized = true;
  }

  /**
   * Get user preferences (lazy loaded)
   */
  get userPreferences() {
    this._ensureInitialized();
    return this._userPreferences;
  }

  /**
   * Set user preferences
   */
  set userPreferences(prefs) {
    this._ensureInitialized();
    this._userPreferences = prefs;
  }

  /**
   * Load user preferences from storage
   */
  loadPreferences() {
    try {
      const stored = localStorage.getItem(PREFERENCES_STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (e) {
      console.warn('Failed to load user preferences:', e);
    }

    return {
      favoriteLocations: [],
      dietaryRestrictions: [],
      interests: [],
      language: 'en',
      visitedPlaces: [],
      savedRestaurants: []
    };
  }

  /**
   * Save user preferences to storage
   */
  savePreferences() {
    try {
      localStorage.setItem(PREFERENCES_STORAGE_KEY, JSON.stringify(this.userPreferences));
    } catch (e) {
      console.warn('Failed to save user preferences:', e);
    }
  }

  /**
   * Add message to conversation history
   */
  addMessage(message, sender, metadata = {}) {
    this.contextHistory.push({
      text: message,
      sender,
      timestamp: new Date().toISOString(),
      ...metadata
    });

    // Keep only last N messages for context
    if (this.contextHistory.length > MAX_CONTEXT_MESSAGES) {
      this.contextHistory.shift();
    }

    // Extract preferences from message
    this.extractPreferences(message, sender);
  }

  /**
   * Extract user preferences from messages
   */
  extractPreferences(message, sender) {
    if (sender !== 'user') return;

    const lower = message.toLowerCase();

    // Extract dietary restrictions
    const dietaryKeywords = ['vegetarian', 'vegan', 'halal', 'kosher', 'gluten-free', 'allergic'];
    dietaryKeywords.forEach(keyword => {
      if (lower.includes(keyword) && !this.userPreferences.dietaryRestrictions.includes(keyword)) {
        this.userPreferences.dietaryRestrictions.push(keyword);
        this.savePreferences();
      }
    });

    // Extract interests
    const interestKeywords = ['history', 'art', 'food', 'nightlife', 'shopping', 'culture', 'architecture'];
    interestKeywords.forEach(interest => {
      if (lower.includes(interest) && !this.userPreferences.interests.includes(interest)) {
        this.userPreferences.interests.push(interest);
        this.savePreferences();
      }
    });
  }

  /**
   * Add location to favorites
   */
  addFavoriteLocation(location) {
    if (!this.userPreferences.favoriteLocations.includes(location)) {
      this.userPreferences.favoriteLocations.push(location);
      this.savePreferences();
    }
  }

  /**
   * Add visited place
   */
  addVisitedPlace(place) {
    if (!this.userPreferences.visitedPlaces.includes(place)) {
      this.userPreferences.visitedPlaces.push(place);
      this.savePreferences();
    }
  }

  /**
   * Save restaurant
   */
  saveRestaurant(restaurant) {
    if (!this.userPreferences.savedRestaurants.some(r => r.id === restaurant.id)) {
      this.userPreferences.savedRestaurants.push(restaurant);
      this.savePreferences();
    }
  }

  /**
   * Get conversation context for LLM
   * Formats recent conversation history for LLM prompt
   */
  getContextForLLM() {
    if (this.contextHistory.length === 0) {
      return null;
    }

    // Format last few messages
    const recentMessages = this.contextHistory.slice(-5).map(msg => {
      return `${msg.sender === 'user' ? 'User' : 'Assistant'}: ${msg.text}`;
    }).join('\n');

    // Build context string
    let context = `Recent Conversation:\n${recentMessages}\n`;

    // Add user preferences if available
    if (this.userPreferences.interests.length > 0) {
      context += `\nUser Interests: ${this.userPreferences.interests.join(', ')}`;
    }

    if (this.userPreferences.dietaryRestrictions.length > 0) {
      context += `\nDietary Restrictions: ${this.userPreferences.dietaryRestrictions.join(', ')}`;
    }

    if (this.userPreferences.visitedPlaces.length > 0) {
      context += `\nPlaces Already Visited: ${this.userPreferences.visitedPlaces.join(', ')}`;
    }

    return context;
  }

  /**
   * Get personalized recommendations based on preferences
   */
  getPersonalizedSuggestions() {
    const suggestions = [];

    // Suggest based on interests
    if (this.userPreferences.interests.includes('history')) {
      suggestions.push('Explore Topkapi Palace', 'Visit Basilica Cistern');
    }

    if (this.userPreferences.interests.includes('food')) {
      suggestions.push('Try Turkish breakfast', 'Visit Spice Bazaar');
    }

    if (this.userPreferences.interests.includes('art')) {
      suggestions.push('Istanbul Modern Museum', 'Pera Museum');
    }

    // Avoid already visited places
    return suggestions.filter(s => 
      !this.userPreferences.visitedPlaces.some(p => s.includes(p))
    );
  }

  /**
   * Set session metadata
   */
  setSessionMetadata(key, value) {
    this.sessionMetadata[key] = value;
  }

  /**
   * Get session metadata
   */
  getSessionMetadata(key) {
    return this.sessionMetadata[key];
  }

  /**
   * Clear conversation history (keep preferences)
   */
  clearHistory() {
    this.contextHistory = [];
    this.sessionMetadata = {};
  }

  /**
   * Reset everything including preferences
   */
  reset() {
    this.contextHistory = [];
    this.sessionMetadata = {};
    this.userPreferences = {
      favoriteLocations: [],
      dietaryRestrictions: [],
      interests: [],
      language: 'en',
      visitedPlaces: [],
      savedRestaurants: []
    };
    this.savePreferences();
  }

  /**
   * Get full context summary
   */
  getContextSummary() {
    return {
      messageCount: this.contextHistory.length,
      preferences: this.userPreferences,
      sessionMetadata: this.sessionMetadata,
      hasContext: this.contextHistory.length > 0
    };
  }
}

// Export singleton instance
export const conversationContext = new ConversationContextManager();

// Export convenience functions
export const addMessageToContext = (message, sender, metadata) => 
  conversationContext.addMessage(message, sender, metadata);

export const getContextForLLM = () => 
  conversationContext.getContextForLLM();

export const getPersonalizedSuggestions = () => 
  conversationContext.getPersonalizedSuggestions();

export const addFavoriteLocation = (location) => 
  conversationContext.addFavoriteLocation(location);

export const addVisitedPlace = (place) => 
  conversationContext.addVisitedPlace(place);

export const saveRestaurant = (restaurant) => 
  conversationContext.saveRestaurant(restaurant);

export const clearConversationHistory = () => 
  conversationContext.clearHistory();

export const getContextSummary = () => 
  conversationContext.getContextSummary();

export default conversationContext;
