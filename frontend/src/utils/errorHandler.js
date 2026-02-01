import React from 'react';

/**
 * Comprehensive Error Handling Utilities
 * 
 * Features:
 * - Network error detection and handling
 * - Retry mechanisms with exponential backoff
 * - Offline/online state management
 * - User-friendly error messages
 * - Recovery strategies
 */

// Error types for classification
export const ErrorTypes = {
  NETWORK: 'NETWORK',
  SERVER: 'SERVER', 
  CLIENT: 'CLIENT',
  TIMEOUT: 'TIMEOUT',
  OFFLINE: 'OFFLINE',
  SECURITY: 'SECURITY',
  RATE_LIMIT: 'RATE_LIMIT',
  VALIDATION: 'VALIDATION',
  UNKNOWN: 'UNKNOWN'
};

// Network status management
class NetworkStatus {
  constructor() {
    this.isOnline = navigator.onLine;
    this.listeners = [];
    
    // Listen for online/offline events
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.notifyListeners(true);
    });
    
    window.addEventListener('offline', () => {
      this.isOnline = false;
      this.notifyListeners(false);
    });
  }
  
  addListener(callback) {
    this.listeners.push(callback);
  }
  
  removeListener(callback) {
    this.listeners = this.listeners.filter(listener => listener !== callback);
  }
  
  notifyListeners(isOnline) {
    this.listeners.forEach(listener => listener(isOnline));
  }
}

export const networkStatus = new NetworkStatus();

// Retry configuration
const RETRY_CONFIG = {
  maxAttempts: 3,
  baseDelay: 1000, // 1 second
  maxDelay: 10000, // 10 seconds
  backoffMultiplier: 2,
  retryableStatuses: [408, 429, 500, 502, 503, 504],
  retryableErrors: ['NetworkError', 'TimeoutError', 'AbortError']
};

// Sleep utility for delays
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Calculate retry delay with exponential backoff
const calculateRetryDelay = (attempt, baseDelay = RETRY_CONFIG.baseDelay) => {
  const delay = Math.min(
    baseDelay * Math.pow(RETRY_CONFIG.backoffMultiplier, attempt - 1),
    RETRY_CONFIG.maxDelay
  );
  
  // Add jitter to prevent thundering herd
  const jitter = delay * 0.1 * Math.random();
  return delay + jitter;
};

// Error classification
export const classifyError = (error, response = null) => {
  // Check for actual network errors first (not navigator.onLine which is unreliable)
  if (error.name === 'NetworkError' || error.message.includes('Failed to fetch')) {
    return ErrorTypes.NETWORK;
  }
  
  if (error.name === 'TimeoutError' || error.message.includes('timeout')) {
    return ErrorTypes.TIMEOUT;
  }
  
  if (error.name === 'AbortError') {
    return ErrorTypes.TIMEOUT;
  }
  
  // Only classify as OFFLINE if navigator.onLine is false AND we have a network error
  // This reduces false positives
  if (!navigator.onLine && (error.name === 'NetworkError' || error.message.includes('Failed to fetch'))) {
    return ErrorTypes.OFFLINE;
  }
  
  // HTTP status-based classification
  if (response) {
    const status = response.status;
    
    if (status >= 400 && status < 500) {
      if (status === 429) return ErrorTypes.RATE_LIMIT;
      if (status === 403 || status === 401) return ErrorTypes.SECURITY;
      if (status === 422 || status === 400) return ErrorTypes.VALIDATION;
      return ErrorTypes.CLIENT;
    }
    
    if (status >= 500) {
      return ErrorTypes.SERVER;
    }
  }
  
  return ErrorTypes.UNKNOWN;
};

// Check if error is retryable
export const isRetryable = (error, response = null) => {
  const errorType = classifyError(error, response);
  
  // Always retry network and timeout errors
  if ([ErrorTypes.NETWORK, ErrorTypes.TIMEOUT, ErrorTypes.SERVER].includes(errorType)) {
    return true;
  }
  
  // Retry specific HTTP status codes
  if (response && RETRY_CONFIG.retryableStatuses.includes(response.status)) {
    return true;
  }
  
  // Retry specific error types
  if (RETRY_CONFIG.retryableErrors.includes(error.name)) {
    return true;
  }
  
  return false;
};

// Get user-friendly error message
export const getUserFriendlyMessage = (error, response = null) => {
  const errorType = classifyError(error, response);
  
  switch (errorType) {
    case ErrorTypes.OFFLINE:
      return 'You appear to be offline. Please check your internet connection and try again.';
      
    case ErrorTypes.NETWORK:
      return 'Connection problem. Please check your internet connection and try again.';
      
    case ErrorTypes.TIMEOUT:
      return 'Request timed out. The server is taking too long to respond. Please try again.';
      
    case ErrorTypes.SERVER:
      return 'Server error. Our servers are experiencing issues. Please try again in a few moments.';
      
    case ErrorTypes.RATE_LIMIT:
      return 'Too many requests. Please wait a moment before trying again.';
      
    case ErrorTypes.SECURITY:
      return 'Access denied. Please check your permissions or try logging in again.';
      
    case ErrorTypes.VALIDATION:
      return 'Invalid request. Please check your input and try again.';
      
    case ErrorTypes.CLIENT:
      if (response) {
        return `Request failed (${response.status}). Please check your input and try again.`;
      }
      return 'Request failed. Please check your input and try again.';
      
    default:
      return 'An unexpected error occurred. Please try again or contact support if the problem persists.';
  }
};

// Enhanced fetch with retry logic
export const fetchWithRetry = async (url, options = {}, customConfig = {}) => {
  const config = { ...RETRY_CONFIG, ...customConfig };
  let lastError = null;
  
  for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
    try {
      // Use caller's signal directly if provided, otherwise create timeout-only controller
      const timeoutMs = options.timeout || 30000;
      let signal = options.signal;
      let timeoutId = null;
      let controller = null;
      
      // Only create our own controller if caller didn't provide a signal
      if (!signal) {
        controller = new AbortController();
        signal = controller.signal;
        timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      }
      
      const response = await fetch(url, {
        ...options,
        signal
      });
      
      if (timeoutId) clearTimeout(timeoutId);
      
      if (response.ok) {
        return response;
      }
      
      // Non-OK response
      const errorText = await response.text().catch(() => '');
      lastError = new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
      
      if (attempt < config.maxAttempts && isRetryable(lastError, response)) {
        await sleep(calculateRetryDelay(attempt, config.baseDelay));
        continue;
      }
      throw lastError;
      
    } catch (error) {
      lastError = error;
      
      // Don't retry aborted requests
      if (error.name === 'AbortError') {
        throw error;
      }
      
      if (attempt < config.maxAttempts && isRetryable(error)) {
        await sleep(calculateRetryDelay(attempt, config.baseDelay));
        continue;
      }
      throw error;
    }
  }
  
  throw lastError || new Error('All retry attempts failed');
};

// Recovery strategies
export const getRecoveryStrategies = (error, response = null) => {
  const errorType = classifyError(error, response);
  const strategies = [];
  
  switch (errorType) {
    case ErrorTypes.OFFLINE:
      strategies.push({
        type: 'wait_online',
        message: 'Wait for connection to be restored',
        action: () => new Promise(resolve => {
          const checkOnline = () => {
            if (navigator.onLine) {
              networkStatus.removeListener(checkOnline);
              resolve();
            }
          };
          networkStatus.addListener(checkOnline);
          if (navigator.onLine) resolve(); // Already online
        })
      });
      break;
      
    case ErrorTypes.NETWORK:
    case ErrorTypes.TIMEOUT:
      strategies.push({
        type: 'retry',
        message: 'Retry the request',
        action: 'retry'
      });
      break;
      
    case ErrorTypes.SERVER:
      strategies.push({
        type: 'retry_delayed',
        message: 'Retry after a short delay',
        action: 'retry'
      });
      break;
      
    case ErrorTypes.RATE_LIMIT:
      strategies.push({
        type: 'wait_rate_limit',
        message: 'Wait before retrying',
        delay: 60000 // 1 minute
      });
      break;
      
    case ErrorTypes.VALIDATION:
      strategies.push({
        type: 'fix_input',
        message: 'Check and correct your input'
      });
      break;
  }
  
  return strategies;
};

// Error boundary helper for React components
export const createErrorBoundary = (fallbackComponent) => {
  return class ErrorBoundary extends React.Component {
    constructor(props) {
      super(props);
      this.state = { hasError: false, error: null };
    }
    
    static getDerivedStateFromError(error) {
      return { hasError: true, error };
    }
    
    componentDidCatch(error, errorInfo) {
      console.error('Error boundary caught error:', error, errorInfo);
    }
    
    render() {
      if (this.state.hasError) {
        return fallbackComponent ? 
          fallbackComponent(this.state.error) : 
          React.createElement('div', null, 'Something went wrong.');
      }
      
      return this.props.children;
    }
  };
};

// Debounce utility for preventing rapid successive calls
export const debounce = (func, delay) => {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(null, args), delay);
  };
};

// Circuit breaker pattern for failing services
class CircuitBreaker {
  constructor(options = {}) {
    this.failureThreshold = options.failureThreshold || 5;
    this.resetTimeout = options.resetTimeout || 60000; // 1 minute
    this.monitoringPeriod = options.monitoringPeriod || 10000; // 10 seconds
    
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.failureCount = 0;
    this.lastFailureTime = null;
    this.nextAttempt = null;
  }
  
  async call(operation) {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      this.state = 'HALF_OPEN';
    }
    
    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }
  
  onSuccess() {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }
  
  onFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    
    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
      this.nextAttempt = Date.now() + this.resetTimeout;
    }
  }
  
  reset() {
    this.failureCount = 0;
    this.state = 'CLOSED';
    this.lastFailureTime = null;
    this.nextAttempt = null;
  }
}

export const createCircuitBreaker = (options) => new CircuitBreaker(options);

// API Error Handler - provides consistent error handling for API calls
export const handleApiError = (error, response = null) => {
  const errorType = classifyError(error, response);
  const userMessage = getUserFriendlyMessage(error, response);
  const recoveryStrategies = getRecoveryStrategies(error, response);
  
  // Log the error for debugging
  console.error('API Error:', {
    error: error.message || error,
    type: errorType,
    response: response?.status,
    url: response?.url || 'unknown'
  });
  
  return {
    type: errorType,
    message: userMessage,
    originalError: error,
    response: response,
    recoveryStrategies,
    isRetryable: isRetryable(error, response)
  };
};

export default {
  ErrorTypes,
  networkStatus,
  classifyError,
  isRetryable,
  getUserFriendlyMessage,
  fetchWithRetry,
  handleApiError,
  getRecoveryStrategies,
  createErrorBoundary,
  debounce,
  createCircuitBreaker
};
