/**
 * Sentry Error Tracking Configuration
 * Initialize Sentry for production error monitoring
 */
import * as Sentry from '@sentry/react';
import { BrowserTracing } from '@sentry/tracing';

const isDevelopment = import.meta.env.DEV;
const isProduction = import.meta.env.PROD;

/**
 * Initialize Sentry error tracking
 * Only enables in production if DSN is provided
 */
export const initSentry = () => {
  const sentryDsn = import.meta.env.VITE_SENTRY_DSN;
  
  // Skip if no DSN provided or in development
  if (!sentryDsn || isDevelopment) {
    console.log('ℹ️ Sentry error tracking disabled (development mode or no DSN)');
    return;
  }

  try {
    Sentry.init({
      dsn: sentryDsn,
      environment: import.meta.env.VITE_SENTRY_ENVIRONMENT || 'production',
      release: import.meta.env.VITE_APP_VERSION || '1.0.0',
      
      // Performance Monitoring
      integrations: [
        new BrowserTracing(),
        new Sentry.Replay({
          maskAllText: true,
          blockAllMedia: true,
        }),
      ],
      
      // Set sample rates
      tracesSampleRate: parseFloat(import.meta.env.VITE_SENTRY_TRACES_SAMPLE_RATE || '0.1'),
      
      // Session Replay
      replaysSessionSampleRate: parseFloat(import.meta.env.VITE_SENTRY_REPLAYS_SESSION_SAMPLE_RATE || '0.1'),
      replaysOnErrorSampleRate: parseFloat(import.meta.env.VITE_SENTRY_REPLAYS_ON_ERROR_SAMPLE_RATE || '1.0'),
      
      // Filter out specific errors
      beforeSend(event, hint) {
        // Don't send errors in development
        if (isDevelopment) return null;
        
        // Filter out known non-critical errors
        const error = hint.originalException;
        if (error && typeof error === 'string') {
          // Filter out common browser extension errors
          if (error.includes('chrome-extension://') || 
              error.includes('moz-extension://')) {
            return null;
          }
        }
        
        return event;
      },
      
      // Capture unhandled promise rejections
      onUnhandledRejection: true,
      
      // Breadcrumb settings
      maxBreadcrumbs: 50,
      
      // Attach stacktrace
      attachStacktrace: true,
      
      // Enable debug mode only in development
      debug: isDevelopment,
    });

    // Set user context if available
    const userId = localStorage.getItem('userId');
    if (userId) {
      Sentry.setUser({ id: userId });
    }

    console.log('✅ Sentry error tracking initialized');
  } catch (error) {
    console.error('❌ Failed to initialize Sentry:', error);
  }
};

/**
 * Manually capture an exception
 * @param {Error} error - Error object to capture
 * @param {Object} context - Additional context
 */
export const captureException = (error, context = {}) => {
  if (isProduction && window.Sentry) {
    Sentry.captureException(error, {
      extra: context,
    });
  } else {
    console.error('Error (not sent to Sentry):', error, context);
  }
};

/**
 * Manually capture a message
 * @param {string} message - Message to capture
 * @param {string} level - Severity level (error, warning, info)
 * @param {Object} context - Additional context
 */
export const captureMessage = (message, level = 'info', context = {}) => {
  if (isProduction && window.Sentry) {
    Sentry.captureMessage(message, {
      level,
      extra: context,
    });
  } else {
    console.log(`[${level.toUpperCase()}] ${message}`, context);
  }
};

/**
 * Set user context for error tracking
 * @param {Object} user - User information
 */
export const setUser = (user) => {
  if (window.Sentry) {
    Sentry.setUser(user);
  }
};

/**
 * Clear user context
 */
export const clearUser = () => {
  if (window.Sentry) {
    Sentry.setUser(null);
  }
};

/**
 * Add breadcrumb for debugging
 * @param {string} message - Breadcrumb message
 * @param {Object} data - Additional data
 */
export const addBreadcrumb = (message, data = {}) => {
  if (window.Sentry) {
    Sentry.addBreadcrumb({
      message,
      data,
      level: 'info',
    });
  }
};

export default {
  initSentry,
  captureException,
  captureMessage,
  setUser,
  clearUser,
  addBreadcrumb,
};
