import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { track } from '@vercel/analytics';

// Google Analytics configuration
// TODO: Replace with your actual GA4 tracking ID
const GA_TRACKING_ID = 'G-2XXEMVNC7Z';

// Check if user has consented to analytics
const hasAnalyticsConsent = () => {
  try {
    const consent = localStorage.getItem('ai-istanbul-cookie-consent');
    if (!consent) return false;
    const consentData = JSON.parse(consent);
    return consentData.analytics === true;
  } catch {
    return false;
  }
};

// Initialize Google Analytics with GDPR compliance
export const initGA = () => {
  // Set default consent to denied
  window.gtag = window.gtag || function() {
    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push(arguments);
  };
  
  // Set default consent mode
  window.gtag('consent', 'default', {
    'analytics_storage': 'denied',
    'ad_storage': 'denied',
    'personalization_storage': 'denied'
  });

  // Load gtag script
  const script1 = document.createElement('script');
  script1.async = true;
  script1.src = `https://www.googletagmanager.com/gtag/js?id=${GA_TRACKING_ID}`;
  document.head.appendChild(script1);

  // Initialize gtag
  const script2 = document.createElement('script');
  script2.innerHTML = `
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', '${GA_TRACKING_ID}', {
      page_title: document.title,
      page_location: window.location.href,
    });
  `;
  document.head.appendChild(script2);

  // Check existing consent and update if needed
  if (hasAnalyticsConsent()) {
    window.gtag('consent', 'update', {
      'analytics_storage': 'granted'
    });
  }
};
// Track page views
export const trackPageView = (path, title) => {
  if (typeof window.gtag !== 'undefined' && hasAnalyticsConsent()) {
    window.gtag('config', GA_TRACKING_ID, {
      page_path: path,
      page_title: title,
    });
  }
};

// Track custom events
export const trackEvent = (action, category = 'engagement', label = '', value = 0) => {
  if (typeof window.gtag !== 'undefined' && hasAnalyticsConsent()) {
    window.gtag('event', action, {
      event_category: category,
      event_label: label,
      value: value,
    });
  }
};

// Track chatbot interactions
export const trackChatEvent = (action, message = '') => {
  if (hasAnalyticsConsent()) {
    trackEvent(action, 'chatbot', message.substring(0, 100)); // Limit message length
  }
};

// Track blog interactions
export const trackBlogEvent = (action, postTitle = '') => {
  if (hasAnalyticsConsent()) {
    trackEvent(action, 'blog', postTitle);
  }
};

// Track navigation
export const trackNavigation = (page) => {
  if (hasAnalyticsConsent()) {
    trackEvent('navigate', 'navigation', page);
  }
};

// Track search
export const trackSearch = (searchTerm) => {
  if (hasAnalyticsConsent()) {
    trackEvent('search', 'user_interaction', searchTerm);
  }
};

// Vercel Analytics integration
export const vercelTrackEvent = {
  // Chat interactions
  chatMessage: (messageType = 'general', language = 'en') => {
    track('chat_message', {
      type: messageType,
      language,
      timestamp: new Date().toISOString()
    })
  },

  // Restaurant searches
  restaurantSearch: (location = 'istanbul', searchType = 'general') => {
    track('restaurant_search', {
      location,
      search_type: searchType,
      timestamp: new Date().toISOString()
    })
  },

  // Blog interactions
  blogInteraction: (action, postId = null) => {
    track('blog_interaction', {
      action, // 'view', 'like', 'share'
      post_id: postId,
      timestamp: new Date().toISOString()
    })
  },

  // Error tracking
  error: (errorType, errorMessage, component = null) => {
    track('error_occurred', {
      error_type: errorType,
      error_message: errorMessage,
      component,
      timestamp: new Date().toISOString()
    })
  },

  // Feature usage
  featureUsage: (featureName, action = 'used') => {
    track('feature_usage', {
      feature: featureName,
      action,
      timestamp: new Date().toISOString()
    })
  }
}

// React component for automatic page tracking
export const GoogleAnalytics = () => {
  const location = useLocation();

  useEffect(() => {
    // Initialize GA on first load
    if (!window.gtag) {
      initGA();
    }
  }, []);

  useEffect(() => {
    // Track page views on route changes
    const path = location.pathname + location.search;
    const title = document.title || 'AI-stanbul';
    trackPageView(path, title);
  }, [location]);

  return null; // This component doesn't render anything
};

export default GoogleAnalytics;
