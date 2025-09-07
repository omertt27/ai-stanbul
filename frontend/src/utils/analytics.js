import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

// Google Analytics configuration
const GA_TRACKING_ID = 'G-WRDCM59VZP';

// Initialize Google Analytics
export const initGA = () => {
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

  // Make gtag available globally
  window.dataLayer = window.dataLayer || [];
  window.gtag = function() {
    window.dataLayer.push(arguments);
  };
};

// Track page views
export const trackPageView = (path, title) => {
  if (typeof window.gtag !== 'undefined') {
    window.gtag('config', GA_TRACKING_ID, {
      page_path: path,
      page_title: title,
    });
  }
};

// Track custom events
export const trackEvent = (action, category = 'engagement', label = '', value = 0) => {
  if (typeof window.gtag !== 'undefined') {
    window.gtag('event', action, {
      event_category: category,
      event_label: label,
      value: value,
    });
  }
};

// Track chatbot interactions
export const trackChatEvent = (action, message = '') => {
  trackEvent(action, 'chatbot', message.substring(0, 100)); // Limit message length
};

// Track blog interactions
export const trackBlogEvent = (action, postTitle = '') => {
  trackEvent(action, 'blog', postTitle);
};

// Track navigation
export const trackNavigation = (page) => {
  trackEvent('navigate', 'navigation', page);
};

// Track search
export const trackSearch = (searchTerm) => {
  trackEvent('search', 'user_interaction', searchTerm);
};

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
