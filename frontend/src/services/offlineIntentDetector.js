/**
 * Offline Intent Detector
 * Provides basic on-device intent detection for offline queries
 * Uses lightweight pattern matching and keyword detection
 * 
 * @version 1.0.0
 * @priority MEDIUM
 */

class OfflineIntentDetector {
  constructor() {
    this.intents = this.initializeIntents();
    this.offlineResponses = this.initializeResponses();
  }

  /**
   * Initialize intent patterns and keywords
   */
  initializeIntents() {
    return {
      transit_map: {
        keywords: ['map', 'metro', 'tram', 'ferry', 'show', 'display', 'route', 'line'],
        patterns: [
          /show.*map/i,
          /metro.*map/i,
          /transit.*map/i,
          /where.*metro/i,
          /see.*routes/i
        ],
        priority: 10
      },
      
      nearest_station: {
        keywords: ['nearest', 'closest', 'near', 'nearby', 'station', 'stop', 'metro'],
        patterns: [
          /nearest.*station/i,
          /closest.*metro/i,
          /near.*stop/i,
          /where.*nearest/i,
          /find.*station/i
        ],
        priority: 9
      },
      
      route_planning: {
        keywords: ['how', 'get', 'from', 'to', 'go', 'travel', 'route'],
        patterns: [
          /how.*get.*from/i,
          /how.*go.*to/i,
          /route.*from.*to/i,
          /travel.*from/i,
          /directions.*to/i
        ],
        priority: 8
      },
      
      station_info: {
        keywords: ['station', 'stop', 'info', 'information', 'about', 'facilities'],
        patterns: [
          /info.*station/i,
          /about.*station/i,
          /facilities.*at/i,
          /accessible/i
        ],
        priority: 7
      },
      
      weather: {
        keywords: ['weather', 'temperature', 'rain', 'forecast', 'climate'],
        patterns: [
          /weather.*today/i,
          /temperature/i,
          /rain.*forecast/i,
          /how.*weather/i
        ],
        priority: 6,
        needsInternet: true
      },
      
      restaurant: {
        keywords: ['restaurant', 'food', 'eat', 'dining', 'cafe', 'cuisine'],
        patterns: [
          /where.*eat/i,
          /restaurant.*near/i,
          /food.*recommendation/i,
          /best.*restaurant/i
        ],
        priority: 6,
        needsInternet: true
      },
      
      attraction: {
        keywords: ['attraction', 'visit', 'see', 'tourist', 'landmark', 'museum'],
        patterns: [
          /what.*visit/i,
          /tourist.*attraction/i,
          /see.*istanbul/i,
          /best.*places/i
        ],
        priority: 5,
        needsInternet: true
      },
      
      help: {
        keywords: ['help', 'how', 'what', 'can', 'features', 'offline'],
        patterns: [
          /what.*can.*do/i,
          /help.*me/i,
          /how.*use/i,
          /features.*offline/i,
          /^help$/i
        ],
        priority: 4
      }
    };
  }

  /**
   * Initialize offline response templates
   */
  initializeResponses() {
    return {
      transit_map: {
        online: "I'm KAM, and I'll show you the transit map with all metro, tram, and ferry routes.",
        offline: "🗺️ **Transit Map (Offline)**\n\nHi! I'm KAM, your Istanbul AI assistant. I can show you the complete Istanbul transit network! This includes:\n\n• 7 Metro lines (M1A, M1B, M2, M3, M4, M5, M7)\n• 3 Tram lines (T1, T4, T5)\n• 5+ Ferry routes\n• Metrobüs (BRT)\n• 150+ stations with details\n\n📍 All data is available offline. Click 'View Transit Map' to explore!"
      },
      
      nearest_station: {
        online: "I'm KAM, let me find the nearest stations to your location.",
        offline: "🚇 **Find Nearest Station (Offline)**\n\nHi! I'm KAM. I can help you find nearby stations using your GPS location or last known position.\n\n✅ Available offline:\n• Find closest metro/tram/ferry stops\n• Calculate walking distances\n• Show station facilities\n• View accessibility info\n\n📍 Please enable location services for best results."
      },
      
      route_planning: {
        online: "I'm KAM, I'll plan the best route for you with real-time data.",
        offline: "🚇 **Route Planning (Offline Mode)**\n\nI'm KAM, and I can plan basic routes between stations using static schedules:\n\n✅ Available:\n• Multi-modal routing (metro + tram + ferry)\n• Station-to-station connections\n• Transfer point identification\n• Walking distances\n\n⚠️ Unavailable offline:\n• Real-time arrival times\n• Live traffic updates\n• Service disruptions\n\n💡 Connect to internet for optimized routes with live data."
      },
      
      station_info: {
        online: "I'm KAM, I'll get you detailed information about that station.",
        offline: "🚉 **Station Information (Offline)**\n\nI'm KAM! I have detailed offline data for 150+ stations including:\n\n✅ Station details:\n• Exact GPS coordinates\n• English & Turkish names\n• Transfer connections\n• Accessibility features\n• Available facilities\n\n💬 Tell me which station you're interested in!"
      },
      
      weather: {
        online: "I'm KAM, let me check the current weather in Istanbul for you.",
        offline: "🌤️ **Weather Information (Internet Required)**\n\nHi! I'm KAM. Weather data requires an internet connection to provide:\n• Current temperature\n• Forecast\n• Precipitation alerts\n• Air quality\n\n📴 While offline, I recommend:\n• Checking cached weather (if recently loaded)\n• Using device weather app\n• Planning for typical Istanbul weather\n\n🌐 Connect to get live weather updates!"
      },
      
      restaurant: {
        online: "I'm KAM, I'll find great restaurants for you!",
        offline: "🍽️ **Restaurant Search (Internet Required)**\n\nI'm KAM! Restaurant recommendations require internet to access our database.\n\n📴 While offline, you can:\n• View previously saved restaurants\n• Explore transit routes to known areas\n• Check cached restaurant data (if recent)\n\n💡 **Good news!** We're working on offline restaurant database for future updates.\n\n🌐 Connect to search 1000+ restaurants!"
      },
      
      attraction: {
        online: "I'm KAM, let me suggest some amazing places to visit!",
        offline: "🏛️ **Attractions & Tourism (Internet Required)**\n\nHi! I'm KAM. Attraction data requires internet connection for:\n• Latest opening hours\n• Ticket prices\n• Reviews & ratings\n• Special events\n\n📴 While offline, you can:\n• Navigate to major landmarks using transit map\n• View cached attraction info\n• Plan routes to popular areas\n\n🌐 Connect for full attraction database!"
      },
      
      help: {
        online: "Hi! I'm KAM, your Istanbul AI assistant. Here's what I can help you with!",
        offline: "📴 **Offline Mode - Available Features**\n\n👋 Hi! I'm **KAM**, your Istanbul AI assistant.\n\n✅ **Fully functional offline:**\n• View complete transit map (7 metro, 3 tram, 5+ ferry lines)\n• Find nearest stations with GPS\n• Plan basic routes between stations\n• Station lookup and details (150+ stations)\n• Language switching\n• View cached content\n\n⚠️ **Limited offline:**\n• Route planning (static schedules only)\n• Chat responses (queued for later)\n\n❌ **Requires internet:**\n• Real-time arrival times\n• Weather updates\n• Restaurant search\n• Live traffic data\n• Full AI chat responses\n\n💬 Your messages are saved and will sync when you reconnect!"
      },
      
      unknown: {
        online: "I'm KAM, and I'll do my best to help you with that!",
        offline: "📴 **Offline Mode**\n\nHi! I'm **KAM**, your Istanbul AI assistant.\n\nI'm currently offline and have limited capabilities. However, I can help with:\n\n✅ Transit maps and routes\n✅ Station information\n✅ Basic route planning\n\n💬 For complex queries, your message will be saved and processed when you reconnect.\n\nType 'help' to see all offline features!"
      }
    };
  }

  /**
   * Detect intent from user query
   * @param {string} query - User's query text
   * @returns {object} Intent result with confidence score
   */
  detectIntent(query) {
    if (!query || typeof query !== 'string') {
      return { intent: 'unknown', confidence: 0 };
    }

    const normalizedQuery = query.toLowerCase().trim();
    const scores = {};

    // Score each intent
    Object.entries(this.intents).forEach(([intentName, intentData]) => {
      let score = 0;

      // Check pattern matches (high weight)
      intentData.patterns.forEach(pattern => {
        if (pattern.test(normalizedQuery)) {
          score += 5;
        }
      });

      // Check keyword matches (medium weight)
      intentData.keywords.forEach(keyword => {
        if (normalizedQuery.includes(keyword.toLowerCase())) {
          score += 1;
        }
      });

      // Apply priority boost
      score *= (intentData.priority / 10);

      scores[intentName] = score;
    });

    // Find best match
    const bestMatch = Object.entries(scores).reduce((best, [intent, score]) => {
      return score > best.score ? { intent, score } : best;
    }, { intent: 'unknown', score: 0 });

    // Calculate confidence (0-100)
    const maxPossibleScore = 15; // 5 (pattern) + 10 (keywords)
    const confidence = Math.min(100, Math.round((bestMatch.score / maxPossibleScore) * 100));

    return {
      intent: bestMatch.intent,
      confidence: confidence,
      needsInternet: this.intents[bestMatch.intent]?.needsInternet || false
    };
  }

  /**
   * Get offline response for detected intent
   * @param {string} intent - Detected intent
   * @param {boolean} isOnline - Current network status
   * @returns {string} Response text
   */
  getResponse(intent, isOnline = false) {
    const responses = this.offlineResponses[intent] || this.offlineResponses.unknown;
    return isOnline ? responses.online : responses.offline;
  }

  /**
   * Process user query and return appropriate response
   * @param {string} query - User query
   * @param {boolean} isOnline - Current network status
   * @returns {object} Processing result
   */
  process(query, isOnline = false) {
    const detection = this.detectIntent(query);
    const response = this.getResponse(detection.intent, isOnline);

    return {
      intent: detection.intent,
      confidence: detection.confidence,
      response: response,
      needsInternet: detection.needsInternet,
      isOnline: isOnline,
      canHandleOffline: !detection.needsInternet || detection.intent === 'help'
    };
  }

  /**
   * Check if query can be handled offline
   * @param {string} query - User query
   * @returns {boolean} True if can handle offline
   */
  canHandleOffline(query) {
    const detection = this.detectIntent(query);
    return !this.intents[detection.intent]?.needsInternet;
  }
}

// Singleton instance
const offlineIntentDetector = new OfflineIntentDetector();

export default offlineIntentDetector;
