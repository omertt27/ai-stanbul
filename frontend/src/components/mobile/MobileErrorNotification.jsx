/**
 * MobileErrorNotification Component
 * ==================================
 * Mobile-optimized error notifications with retry logic
 * 
 * Features:
 * - User-friendly error messages
 * - Automatic retry with exponential backoff
 * - Dismissible with swipe
 * - Haptic feedback
 * - Network status indicator
 * - Accessibility support
 */

import React, { useState, useEffect } from 'react';
import './MobileErrorNotification.css';

const MobileErrorNotification = ({ 
  error,
  onRetry,
  onDismiss,
  darkMode = false,
  autoRetry = true,
  maxRetries = 3
}) => {
  const [retryCount, setRetryCount] = useState(0);
  const [isRetrying, setIsRetrying] = useState(false);
  const [countdown, setCountdown] = useState(0);

  // User-friendly error messages
  const getErrorMessage = (error) => {
    if (!error) return null;

    const errorMessages = {
      network: {
        title: "No Internet Connection",
        message: "Please check your connection and try again.",
        icon: "📶",
        color: "#f59e0b"
      },
      timeout: {
        title: "Request Timed Out",
        message: "The request took too long. Please try again.",
        icon: "⏱️",
        color: "#ef4444"
      },
      server: {
        title: "Server Error",
        message: "Our servers are busy. We'll retry automatically.",
        icon: "🔧",
        color: "#ef4444"
      },
      validation: {
        title: "Invalid Input",
        message: "Please check your input and try again.",
        icon: "⚠️",
        color: "#f59e0b"
      },
      gps: {
        title: "Location Unavailable",
        message: "Unable to get your location. Try enabling GPS.",
        icon: "📍",
        color: "#f59e0b"
      },
      generic: {
        title: "Something Went Wrong",
        message: "An unexpected error occurred. Please try again.",
        icon: "❌",
        color: "#ef4444"
      }
    };

    return errorMessages[error.type] || errorMessages.generic;
  };

  const errorInfo = getErrorMessage(error);

  // Auto-retry with exponential backoff
  useEffect(() => {
    if (!autoRetry || !error || retryCount >= maxRetries) return;

    const delay = Math.min(1000 * Math.pow(2, retryCount), 10000); // Max 10s
    setCountdown(Math.ceil(delay / 1000));

    const countdownInterval = setInterval(() => {
      setCountdown(prev => Math.max(0, prev - 1));
    }, 1000);

    const retryTimer = setTimeout(() => {
      setIsRetrying(true);
      setRetryCount(prev => prev + 1);
      
      onRetry?.().finally(() => {
        setIsRetrying(false);
      });
    }, delay);

    return () => {
      clearTimeout(retryTimer);
      clearInterval(countdownInterval);
    };
  }, [error, retryCount, autoRetry, maxRetries, onRetry]);

  const handleManualRetry = () => {
    if (isRetrying) return;

    // Haptic feedback
    if ('vibrate' in navigator) {
      navigator.vibrate(50);
    }

    setIsRetrying(true);
    setRetryCount(prev => prev + 1);
    
    onRetry?.().finally(() => {
      setIsRetrying(false);
    });
  };

  const handleDismiss = () => {
    // Haptic feedback
    if ('vibrate' in navigator) {
      navigator.vibrate([50, 100, 50]);
    }

    onDismiss?.();
  };

  if (!error || !errorInfo) return null;

  return (
    <div 
      className={`mobile-error-notification ${darkMode ? 'dark' : 'light'}`}
      role="alert"
      aria-live="assertive"
      style={{ borderLeftColor: errorInfo.color }}
    >
      <div className="error-icon" style={{ color: errorInfo.color }}>
        {errorInfo.icon}
      </div>

      <div className="error-content">
        <h4 className="error-title">{errorInfo.title}</h4>
        <p className="error-message">{errorInfo.message}</p>
        
        {autoRetry && retryCount < maxRetries && countdown > 0 && (
          <p className="error-countdown">
            Retrying in {countdown}s... (Attempt {retryCount + 1}/{maxRetries})
          </p>
        )}
      </div>

      <div className="error-actions">
        <button
          onClick={handleManualRetry}
          disabled={isRetrying}
          className="retry-button"
          aria-label="Retry now"
        >
          {isRetrying ? (
            <span className="spinner-small">⟳</span>
          ) : (
            '🔄'
          )}
        </button>

        <button
          onClick={handleDismiss}
          className="dismiss-button"
          aria-label="Dismiss error"
        >
          ✕
        </button>
      </div>
    </div>
  );
};

// Network status indicator component
export const NetworkStatusIndicator = ({ darkMode = false }) => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [showOffline, setShowOffline] = useState(false);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setShowOffline(false);
    };

    const handleOffline = () => {
      setIsOnline(false);
      setShowOffline(true);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (!showOffline) return null;

  return (
    <div className={`network-status-indicator ${darkMode ? 'dark' : 'light'} ${isOnline ? 'online' : 'offline'}`}>
      <span className="status-icon">{isOnline ? '✓' : '✕'}</span>
      <span className="status-text">
        {isOnline ? 'Back Online' : 'No Connection'}
      </span>
    </div>
  );
};

export default MobileErrorNotification;
