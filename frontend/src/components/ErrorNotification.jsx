import React, { useState, useEffect } from 'react';
import { 
  ErrorTypes, 
  getUserFriendlyMessage, 
  classifyError,
  getRecoveryStrategies,
  networkStatus
} from '../utils/errorHandler';

/**
 * Error Notification Component
 * 
 * Features:
 * - User-friendly error messages
 * - Recovery action buttons
 * - Network status indicator
 * - Auto-dismiss for transient errors
 * - Retry functionality
 */

const ErrorNotification = ({ 
  error, 
  onRetry, 
  onDismiss, 
  autoHide = false,
  autoHideDelay = 5000,
  darkMode = false 
}) => {
  const [isVisible, setIsVisible] = useState(!!error);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  
  // Safety check: ensure error is defined and is an object
  if (!error || typeof error !== 'object') {
    console.warn('ErrorNotification: Invalid error object received:', error);
    return null;
  }
  
  useEffect(() => {
    setIsVisible(!!error);
  }, [error]);
  
  useEffect(() => {
    // Subscribe to network status changes
    const unsubscribe = networkStatus.addListener((online) => {
      setIsOnline(online);
    });
    
    return unsubscribe;
  }, []);
  
  useEffect(() => {
    if (error && autoHide) {
      const timer = setTimeout(() => {
        setIsVisible(false);
        onDismiss && onDismiss();
      }, autoHideDelay);
      
      return () => clearTimeout(timer);
    }
  }, [error, autoHide, autoHideDelay, onDismiss]);
  
  if (!error || !isVisible) {
    return null;
  }
  
  // Handle both structured error objects and raw error objects
  const errorType = error.type || classifyError(error, error.response);
  const userMessage = error.message || getUserFriendlyMessage(error, error.response);
  const recoveryStrategies = error.type ? [] : getRecoveryStrategies(error, error.response);
  
  // Determine if error is retryable
  const isRetryable = error.isRetryable !== undefined ? error.isRetryable : true;
  
  const getErrorIcon = (type) => {
    switch (type) {
      case ErrorTypes.OFFLINE:
        return '○';
      case ErrorTypes.NETWORK:
        return '◇';
      case ErrorTypes.TIMEOUT:
        return '◐';
      case ErrorTypes.SERVER:
        return '◆';
      case ErrorTypes.RATE_LIMIT:
        return '◈';
      case ErrorTypes.SECURITY:
        return '◉';
      case ErrorTypes.VALIDATION:
        return '◎';
      default:
        return '!';
    }
  };
  
  const getErrorColor = (type) => {
    switch (type) {
      case ErrorTypes.OFFLINE:
        return darkMode ? 'bg-yellow-900 border-yellow-600' : 'bg-yellow-100 border-yellow-400';
      case ErrorTypes.NETWORK:
      case ErrorTypes.TIMEOUT:
        return darkMode ? 'bg-orange-900 border-orange-600' : 'bg-orange-100 border-orange-400';
      case ErrorTypes.SERVER:
        return darkMode ? 'bg-red-900 border-red-600' : 'bg-red-100 border-red-400';
      case ErrorTypes.SECURITY:
        return darkMode ? 'bg-purple-900 border-purple-600' : 'bg-purple-100 border-purple-400';
      default:
        return darkMode ? 'bg-gray-800 border-gray-600' : 'bg-gray-100 border-gray-400';
    }
  };
  
  const handleRetry = () => {
    if (onRetry) {
      try {
        const result = onRetry();
        // Handle Promise result if onRetry returns a Promise
        if (result && typeof result === 'object' && typeof result.finally === 'function') {
          result.catch((err) => {
            console.error('Retry failed:', err);
          });
        }
      } catch (err) {
        console.error('Retry failed:', err);
      }
    }
  };
  
  const handleDismiss = () => {
    setIsVisible(false);
    if (onDismiss) {
      onDismiss();
    }
  };
  
  return (
    <div className="fixed top-4 right-4 z-50 max-w-md animate-slide-in">
      <div className={`
        border-l-4 p-4 rounded-lg shadow-lg transition-all duration-300
        ${getErrorColor(errorType)}
        ${darkMode ? 'text-white' : 'text-gray-800'}
      `}>
        {/* Network Status Indicator */}
        {!isOnline && (
          <div className={`
            flex items-center mb-2 text-sm font-medium
            ${darkMode ? 'text-yellow-300' : 'text-yellow-700'}
          `}>
            <span className="mr-2">📡</span>
            You are currently offline
          </div>
        )}
        
        {/* Error Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center">
            <span className="text-xl mr-2">{getErrorIcon(errorType)}</span>
            <div>
              <h4 className="font-semibold text-sm mb-1">
                {errorType === ErrorTypes.OFFLINE ? 'Connection Lost' :
                 errorType === ErrorTypes.NETWORK ? 'Network Error' :
                 errorType === ErrorTypes.TIMEOUT ? 'Request Timeout' :
                 errorType === ErrorTypes.SERVER ? 'Server Error' :
                 errorType === ErrorTypes.RATE_LIMIT ? 'Rate Limited' :
                 errorType === ErrorTypes.SECURITY ? 'Access Denied' :
                 errorType === ErrorTypes.VALIDATION ? 'Invalid Input' :
                 'Error Occurred'}
              </h4>
              <p className="text-sm opacity-90">{userMessage}</p>
            </div>
          </div>
          
          {/* Close Button */}
          <button
            onClick={handleDismiss}
            className={`
              ml-2 p-1 rounded-full hover:bg-black/10 transition-colors
              ${darkMode ? 'text-white/70 hover:text-white' : 'text-gray-500 hover:text-gray-700'}
            `}
          >
            ✕
          </button>
        </div>
        
        {/* Recovery Actions */}
        {recoveryStrategies.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {recoveryStrategies.map((strategy, index) => (
              <button
                key={index}
                onClick={strategy.action === 'retry' ? handleRetry : undefined}
                disabled={strategy.type === 'wait_online' && !isOnline}
                className={`
                  px-3 py-1 text-xs rounded-md font-medium transition-all duration-200
                  ${darkMode 
                    ? 'bg-white/20 text-white hover:bg-white/30 disabled:bg-white/10' 
                    : 'bg-black/10 text-gray-700 hover:bg-black/20 disabled:bg-gray-200'
                  }
                  disabled:cursor-not-allowed disabled:opacity-50
                `}
              >
                {strategy.message}
              </button>
            ))}
            
            {error.isRetryable !== undefined && error.isRetryable && onRetry && (
              <button
                onClick={handleRetry}
                className={`
                  px-3 py-1 text-xs rounded-md font-medium transition-all duration-200
                  ${darkMode 
                    ? 'bg-blue-600 text-white hover:bg-blue-700' 
                    : 'bg-blue-500 text-white hover:bg-blue-600'
                  }
                `}
              >
                🔄 Retry
              </button>
            )}
          </div>
        )}
        
        {/* Debug Info (Development Only) */}
        {import.meta.env.DEV && (
          <details className="mt-3">
            <summary className={`
              text-xs cursor-pointer
              ${darkMode ? 'text-white/70' : 'text-gray-500'}
            `}>
              Debug Info
            </summary>
            <div className={`
              mt-2 p-2 rounded text-xs font-mono
              ${darkMode ? 'bg-black/20' : 'bg-white/50'}
            `}>
              <div>Type: {errorType}</div>
              <div>Retryable: {isRetryable ? 'Yes' : 'No'}</div>
              <div>Status: {error.response?.status || error.status || 'N/A'}</div>
              <div>Original: {error.originalError?.message || error.message || 'Unknown'}</div>
              {error.context && <div>Context: {error.context}</div>}
            </div>
          </details>
        )}
      </div>
    </div>
  );
};

// Network Status Indicator Component
export const NetworkStatusIndicator = ({ darkMode = false }) => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [showOfflineMessage, setShowOfflineMessage] = useState(false);
  
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setShowOfflineMessage(false);
    };
    
    const handleOffline = () => {
      setIsOnline(false);
      setShowOfflineMessage(true);
    };
    
    // Subscribe to network status changes
    const unsubscribe = networkStatus.addListener((online) => {
      if (online) {
        handleOnline();
      } else {
        handleOffline();
      }
    });
    
    return unsubscribe;
  }, []);
  
  // Auto-hide offline message after reconnection
  useEffect(() => {
    if (isOnline && showOfflineMessage) {
      const timer = setTimeout(() => {
        setShowOfflineMessage(false);
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [isOnline, showOfflineMessage]);
  
  if (isOnline && !showOfflineMessage) {
    return null;
  }
  
  return (
    <div className="fixed bottom-4 left-4 z-40">
      <div className={`
        px-4 py-2 rounded-lg shadow-lg border transition-all duration-300
        ${isOnline 
          ? (darkMode ? 'bg-green-800 border-green-600 text-green-100' : 'bg-green-100 border-green-400 text-green-800')
          : (darkMode ? 'bg-yellow-800 border-yellow-600 text-yellow-100' : 'bg-yellow-100 border-yellow-400 text-yellow-800')
        }
      `}>
        <div className="flex items-center text-sm">
          <span className="mr-2">
            {isOnline ? '✅' : '📡'}
          </span>
          {isOnline ? 'Connection restored' : 'You are offline'}
        </div>
      </div>
    </div>
  );
};

// Retry Button Component
export const RetryButton = ({ 
  onRetry, 
  loading = false, 
  disabled = false,
  darkMode = false,
  size = 'medium'
}) => {
  const sizeClasses = {
    small: 'px-3 py-1 text-sm',
    medium: 'px-4 py-2 text-base',
    large: 'px-6 py-3 text-lg'
  };
  
  return (
    <button
      onClick={onRetry}
      disabled={disabled || loading}
      className={`
        ${sizeClasses[size]}
        rounded-lg font-medium transition-all duration-200 flex items-center gap-2
        ${darkMode 
          ? 'bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-600' 
          : 'bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-400'
        }
        disabled:cursor-not-allowed disabled:opacity-50
        ${loading ? 'animate-pulse' : ''}
      `}
    >
      <span className={loading ? 'animate-spin' : ''}>
        {loading ? '⏳' : '🔄'}
      </span>
      {loading ? 'Retrying...' : 'Retry'}
    </button>
  );
};

export default ErrorNotification;
