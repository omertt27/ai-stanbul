/**
 * Amplitude Analytics Helper
 * 
 * Helper functions for tracking events across AI Istanbul application
 * Requires Amplitude SDK to be loaded first
 * 
 * Usage: AIAnalytics.track('Event Name', { properties })
 *        AIAnalytics.chatMessageSent(message, context)
 *        etc.
 */

(function() {
  'use strict';

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

const trackChatMessageSent = (message, context = {}) => {
  trackEvent('Chat Message Sent', {
    message_length: message.length,
    has_location: !!context.user_location,
    language: context.language || 'en',
    ...context
  });
};

const trackBotResponse = (response, context = {}) => {
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

const trackChatError = (error, context = {}) => {
  trackEvent('Chat Error', {
    error_message: error.message || error,
    error_type: error.name || 'unknown',
    ...context
  });
};

// ===== ROUTE PLANNING =====

const trackRouteRequested = (origin, destination, preferences = {}) => {
  trackEvent('Route Requested', {
    origin: typeof origin === 'string' ? origin : origin.name,
    destination: typeof destination === 'string' ? destination : destination.name,
    has_preferences: Object.keys(preferences).length > 0,
    preference_types: Object.keys(preferences),
    ...preferences
  });
};

const trackRouteSelected = (route) => {
  trackEvent('Route Selected', {
    route_id: route.id,
    transport_mode: route.mode || route.transport_mode,
    duration_minutes: Math.round(route.duration / 60),
    distance_km: (route.distance / 1000).toFixed(2),
    cost_tl: route.cost,
    alternative_rank: route.rank || 1
  });
};

const trackNavigationStarted = (route, context = {}) => {
  trackEvent('Navigation Started', {
    route_id: route.id,
    gps_enabled: !!context.gps_enabled,
    estimated_arrival: context.estimated_arrival,
    transport_mode: route.mode
  });
};

const trackNavigationCompleted = (route, context = {}) => {
  trackEvent('Navigation Completed', {
    route_id: route.id,
    actual_duration: context.actual_duration,
    completed_successfully: context.success !== false
  });
};

// ===== PLACES & GEMS =====

const trackGemDiscovered = (gem) => {
  trackEvent('Gem Discovered', {
    gem_name: gem.name,
    gem_category: gem.category,
    gem_id: gem.id,
    discovery_method: gem.discovery_method || 'chat',
    distance_km: gem.distance ? (gem.distance / 1000).toFixed(2) : null
  });
};

const trackPlaceSearch = (searchTerm, results = [], filters = {}) => {
  trackEvent('Place Search', {
    search_term: searchTerm,
    results_count: results.length,
    has_filters: Object.keys(filters).length > 0,
    filter_types: Object.keys(filters)
  });
};

const trackPlaceViewed = (place) => {
  trackEvent('Place Viewed', {
    place_name: place.name,
    place_type: place.type || place.category,
    place_id: place.id,
    has_reviews: !!place.reviews
  });
};

// ===== ADMIN DASHBOARD =====

const trackAdminDashboardAccess = (section = 'dashboard') => {
  trackEvent('Admin Dashboard Accessed', {
    section: section,
    user_agent: navigator.userAgent
  });
};

const trackExperimentCreated = (experiment) => {
  trackEvent('Experiment Created', {
    experiment_name: experiment.name,
    experiment_id: experiment.id,
    variant_count: Object.keys(experiment.variants || {}).length,
    metrics: experiment.metrics,
    duration_days: experiment.duration_days
  });
};

const trackExperimentAction = (action, experimentId, context = {}) => {
  trackEvent('Experiment Action', {
    action: action, // 'started', 'stopped', 'deleted'
    experiment_id: experimentId,
    ...context
  });
};

const trackFeatureFlagAction = (action, flagName, context = {}) => {
  trackEvent('Feature Flag Action', {
    action: action, // 'created', 'toggled', 'updated', 'deleted'
    flag_name: flagName,
    enabled: context.enabled,
    rollout_percentage: context.rollout_percentage,
    ...context
  });
};

const trackLearningCycleRun = (result) => {
  trackEvent('Learning Cycle Executed', {
    status: result.status,
    patterns_learned: result.patterns_learned || 0,
    feedback_analyzed: result.feedback_analyzed || 0,
    improvements_deployed: result.improvements_deployed || false,
    duration_seconds: result.duration
  });
};

// ===== USER ENGAGEMENT =====

const trackPageView = (pageName, context = {}) => {
  trackEvent('Page Viewed', {
    page_name: pageName,
    referrer: document.referrer,
    user_language: navigator.language,
    screen_width: window.screen.width,
    screen_height: window.screen.height,
    ...context
  });
};

const trackButtonClick = (buttonName, context = {}) => {
  trackEvent('Button Clicked', {
    button_name: buttonName,
    ...context
  });
};

const trackFormSubmitted = (formName, success, context = {}) => {
  trackEvent('Form Submitted', {
    form_name: formName,
    success: success,
    ...context
  });
};

// ===== FEEDBACK & ERRORS =====

const trackFeedbackSubmitted = (feedback) => {
  trackEvent('Feedback Submitted', {
    feedback_type: feedback.type,
    original_intent: feedback.original_intent,
    correct_intent: feedback.correct_intent,
    confidence: feedback.confidence,
    query: feedback.query
  });
};

const trackError = (errorType, errorMessage, context = {}) => {
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

const identifyUser = (userId, properties = {}) => {
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

const incrementUserProperty = (property, value = 1) => {
  if (!isAmplitudeLoaded()) return;
  
  try {
    const identify = new window.amplitude.Identify();
    identify.add(property, value);
    window.amplitude.identify(identify);
  } catch (error) {
    console.error('[Analytics] Error incrementing property:', error);
  }
};

// ===== GLOBAL EXPORT =====

// Create global Analytics object
window.AIAnalytics = {
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

console.log('[Analytics] AIAnalytics helper loaded successfully');

})(); // End IIFE

