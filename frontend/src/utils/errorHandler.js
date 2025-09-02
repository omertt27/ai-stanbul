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
  // Network/connectivity errors
  if (!navigator.onLine) {
    return ErrorTypes.OFFLINE;
  }
  
  if (error.name === 'NetworkError' || error.message.includes('Failed to fetch')) {
    return ErrorTypes.NETWORK;
  }
  
  if (error.name === 'TimeoutError' || error.message.includes('timeout')) {
    return ErrorTypes.TIMEOUT;
  }
  
  if (error.name === 'AbortError') {
    return ErrorTypes.TIMEOUT;
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
  let lastResponse = null;
  
  console.log(`🔄 Starting request to: ${url}`);
  
  for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
    try {
      console.log(`🌐 Attempt ${attempt}/${config.maxAttempts} for: ${url}`);
      
      // Check online status before making request
      if (!navigator.onLine) {
        throw new Error('NetworkError: You are currently offline');
      }
      
      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        options.timeout || 30000 // 30 second default timeout
      );
      
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      // Check if response is successful
      if (response.ok) {
        console.log(`✅ Request succeeded on attempt ${attempt}`);
        return response;
      }
      
      // Store response for error handling
      lastResponse = response;
      const errorText = await response.text().catch(() => 'Unknown error');
      lastError = new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
      
      // Check if we should retry
      if (attempt < config.maxAttempts && isRetryable(lastError, response)) {
        const delay = calculateRetryDelay(attempt, config.baseDelay);
        console.log(`⏳ Retrying in ${Math.round(delay)}ms (attempt ${attempt + 1}/${config.maxAttempts})`);
        await sleep(delay);
        continue;
      }
      
      // No more retries or not retryable
      throw lastError;
      
    } catch (error) {
      lastError = error;
      
      // Handle AbortError (timeout)
      if (error.name === 'AbortError') {
        lastError = new Error('TimeoutError: Request timed out');
      }
      
      console.log(`❌ Attempt ${attempt} failed:`, error.message);
      
      // Check if we should retry
      if (attempt < config.maxAttempts && isRetryable(error)) {
        const delay = calculateRetryDelay(attempt, config.baseDelay);
        console.log(`⏳ Retrying in ${Math.round(delay)}ms (attempt ${attempt + 1}/${config.maxAttempts})`);
        await sleep(delay);
        continue;
      }
      
      // No more retries or not retryable
      throw error;
    }
  }
  
  // This should never be reached, but just in case
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
}

export const createCircuitBreaker = (options) => new CircuitBreaker(options);

export default {
  ErrorTypes,
  networkStatus,
  classifyError,
  isRetryable,
  getUserFriendlyMessage,
  fetchWithRetry,
  getRecoveryStrategies,
  createErrorBoundary,
  debounce,
  createCircuitBreaker
};
