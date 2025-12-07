/**
 * Amplitude Analytics Helper
 * 
 * Helper functions for tracking events across AI Istanbul application
 * Requires Amplitude SDK to be loaded first
 */

// Check if Amplitude is loaded
const isAmplitudeLoaded = () => {
  return typeof window.amplitude !== 'undefined';
};

// Safe track wrapper
const trackEvent = (eventName, properties = {}) => {
  if (!isAmplitudeLoaded()) {
    console.warn('[Analytics] Amplitude not loaded, skipping event:', eventName);
    return;
  }
  
  try {
    window.amplitude.track(eventName, {
      ...properties,
      timestamp: new Date().toISOString(),
      page_url: window.location.pathname,
      page_title: document.title
    });
  } catch (error) {
    console.error('[Analytics] Error tracking event:', eventName, error);
  }
};

// ===== CHAT & MESSAGING =====

export const trackChatMessageSent = (message, context = {}) => {
  trackEvent('Chat Message Sent', {
    message_length: message.length,
    has_location: !!context.user_location,
    language: context.language || 'en',
    ...context
  });
};

export const trackBotResponse = (response, context = {}) => {
  trackEvent('Bot Response Generated', {
    intent: context.intent,
    confidence: context.confidence,
    response_time: context.response_time,
    method: context.method || 'pure_llm',
    response_length: response.length,
    has_suggestions: !!context.suggestions,
    ...context
  });
};

export const trackChatError = (error, context = {}) => {
  trackEvent('Chat Error', {
    error_message: error.message || error,
    error_type: error.name || 'unknown',
    ...context
  });
};

// ===== ROUTE PLANNING =====

export const trackRouteRequested = (origin, destination, preferences = {}) => {
  trackEvent('Route Requested', {
    origin: typeof origin === 'string' ? origin : origin.name,
    destination: typeof destination === 'string' ? destination : destination.name,
    has_preferences: Object.keys(preferences).length > 0,
    preference_types: Object.keys(preferences),
    ...preferences
  });
};

export const trackRouteSelected = (route) => {
  trackEvent('Route Selected', {
    route_id: route.id,
    transport_mode: route.mode || route.transport_mode,
    duration_minutes: Math.round(route.duration / 60),
    distance_km: (route.distance / 1000).toFixed(2),
    cost_tl: route.cost,
    alternative_rank: route.rank || 1
  });
};

export const trackNavigationStarted = (route, context = {}) => {
  trackEvent('Navigation Started', {
    route_id: route.id,
    gps_enabled: !!context.gps_enabled,
    estimated_arrival: context.estimated_arrival,
    transport_mode: route.mode
  });
};

export const trackNavigationCompleted = (route, context = {}) => {
  trackEvent('Navigation Completed', {
    route_id: route.id,
    actual_duration: context.actual_duration,
    completed_successfully: context.success !== false
  });
};

// ===== PLACES & GEMS =====

export const trackGemDiscovered = (gem) => {
  trackEvent('Gem Discovered', {
    gem_name: gem.name,
    gem_category: gem.category,
    gem_id: gem.id,
    discovery_method: gem.discovery_method || 'chat',
    distance_km: gem.distance ? (gem.distance / 1000).toFixed(2) : null
  });
};

export const trackPlaceSearch = (searchTerm, results = [], filters = {}) => {
  trackEvent('Place Search', {
    search_term: searchTerm,
    results_count: results.length,
    has_filters: Object.keys(filters).length > 0,
    filter_types: Object.keys(filters)
  });
};

export const trackPlaceViewed = (place) => {
  trackEvent('Place Viewed', {
    place_name: place.name,
    place_type: place.type || place.category,
    place_id: place.id,
    has_reviews: !!place.reviews
  });
};

// ===== ADMIN DASHBOARD =====

export const trackAdminDashboardAccess = (section = 'dashboard') => {
  trackEvent('Admin Dashboard Accessed', {
    section: section,
    user_agent: navigator.userAgent
  });
};

export const trackExperimentCreated = (experiment) => {
  trackEvent('Experiment Created', {
    experiment_name: experiment.name,
    experiment_id: experiment.id,
    variant_count: Object.keys(experiment.variants || {}).length,
    metrics: experiment.metrics,
    duration_days: experiment.duration_days
  });
};

export const trackExperimentAction = (action, experimentId, context = {}) => {
  trackEvent('Experiment Action', {
    action: action, // 'started', 'stopped', 'deleted'
    experiment_id: experimentId,
    ...context
  });
};

export const trackFeatureFlagAction = (action, flagName, context = {}) => {
  trackEvent('Feature Flag Action', {
    action: action, // 'created', 'toggled', 'updated', 'deleted'
    flag_name: flagName,
    enabled: context.enabled,
    rollout_percentage: context.rollout_percentage,
    ...context
  });
};

export const trackLearningCycleRun = (result) => {
  trackEvent('Learning Cycle Executed', {
    status: result.status,
    patterns_learned: result.patterns_learned || 0,
    feedback_analyzed: result.feedback_analyzed || 0,
    improvements_deployed: result.improvements_deployed || false,
    duration_seconds: result.duration
  });
};

// ===== USER ENGAGEMENT =====

export const trackPageView = (pageName, context = {}) => {
  trackEvent('Page Viewed', {
    page_name: pageName,
    referrer: document.referrer,
    user_language: navigator.language,
    screen_width: window.screen.width,
    screen_height: window.screen.height,
    ...context
  });
};

export const trackButtonClick = (buttonName, context = {}) => {
  trackEvent('Button Clicked', {
    button_name: buttonName,
    ...context
  });
};

export const trackFormSubmitted = (formName, success, context = {}) => {
  trackEvent('Form Submitted', {
    form_name: formName,
    success: success,
    ...context
  });
};

// ===== FEEDBACK & ERRORS =====

export const trackFeedbackSubmitted = (feedback) => {
  trackEvent('Feedback Submitted', {
    feedback_type: feedback.type,
    original_intent: feedback.original_intent,
    correct_intent: feedback.correct_intent,
    confidence: feedback.confidence,
    query: feedback.query
  });
};

export const trackError = (errorType, errorMessage, context = {}) => {
  trackEvent('Error Occurred', {
    error_type: errorType,
    error_message: errorMessage,
    endpoint: context.endpoint,
    status_code: context.status_code,
    stack_trace: context.stack_trace,
    ...context
  });
};

// ===== USER IDENTIFICATION =====

export const identifyUser = (userId, properties = {}) => {
  if (!isAmplitudeLoaded()) return;
  
  try {
    window.amplitude.setUserId(userId);
    
    if (Object.keys(properties).length > 0) {
      const identify = new window.amplitude.Identify();
      
      Object.entries(properties).forEach(([key, value]) => {
        identify.set(key, value);
      });
      
      window.amplitude.identify(identify);
    }
  } catch (error) {
    console.error('[Analytics] Error identifying user:', error);
  }
};

export const incrementUserProperty = (property, value = 1) => {
  if (!isAmplitudeLoaded()) return;
  
  try {
    const identify = new window.amplitude.Identify();
    identify.add(property, value);
    window.amplitude.identify(identify);
  } catch (error) {
    console.error('[Analytics] Error incrementing property:', error);
  }
};

// ===== CONVENIENCE EXPORTS =====

// Export all tracking functions as a namespace
export const Analytics = {
  // Core
  track: trackEvent,
  identify: identifyUser,
  increment: incrementUserProperty,
  
  // Chat
  chatMessageSent: trackChatMessageSent,
  botResponse: trackBotResponse,
  chatError: trackChatError,
  
  // Routes
  routeRequested: trackRouteRequested,
  routeSelected: trackRouteSelected,
  navigationStarted: trackNavigationStarted,
  navigationCompleted: trackNavigationCompleted,
  
  // Places
  gemDiscovered: trackGemDiscovered,
  placeSearch: trackPlaceSearch,
  placeViewed: trackPlaceViewed,
  
  // Admin
  adminAccess: trackAdminDashboardAccess,
  experimentCreated: trackExperimentCreated,
  experimentAction: trackExperimentAction,
  featureFlagAction: trackFeatureFlagAction,
  learningCycleRun: trackLearningCycleRun,
  
  // Engagement
  pageView: trackPageView,
  buttonClick: trackButtonClick,
  formSubmit: trackFormSubmitted,
  
  // Feedback
  feedbackSubmitted: trackFeedbackSubmitted,
  error: trackError
};

// Make it globally available
if (typeof window !== 'undefined') {
  window.AIAnalytics = Analytics;
}

export default Analytics;
