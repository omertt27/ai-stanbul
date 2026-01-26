/**
 * VoiceInputButton Component
 * ===========================
 * A reusable, ChatGPT-style voice input button with visual feedback
 * 
 * Features:
 * - Animated microphone icon
 * - Audio level visualization (pulsing ring)
 * - Interim transcript display
 * - Error notifications
 * - Haptic feedback
 * - Accessible
 */

import React, { useEffect, useState } from 'react';
import './VoiceInputButton.css';

const VoiceInputButton = ({
  isListening,
  isSupported,
  audioLevel = 0,
  interimTranscript = '',
  error = null,
  onToggle,
  onErrorDismiss,
  darkMode = false,
  disabled = false,
  size = 'medium', // 'small', 'medium', 'large'
  showTranscript = true,
  className = ''
}) => {
  const [showError, setShowError] = useState(false);

  // Show error toast when error occurs
  useEffect(() => {
    if (error) {
      setShowError(true);
      const timer = setTimeout(() => {
        setShowError(false);
        onErrorDismiss?.();
      }, 4000);
      return () => clearTimeout(timer);
    }
  }, [error, onErrorDismiss]);

  // Calculate pulse scale based on audio level
  const pulseScale = isListening ? 1 + (audioLevel * 0.5) : 1;

  const handleClick = (e) => {
    e.preventDefault();
    if (!disabled) {
      onToggle?.();
    }
  };

  const handleTouchEnd = (e) => {
    // Prevent double-tap zoom on mobile
    e.preventDefault();
    if (!disabled) {
      onToggle?.();
    }
  };

  const sizeClasses = {
    small: 'voice-btn-small',
    medium: 'voice-btn-medium',
    large: 'voice-btn-large'
  };

  return (
    <div className={`voice-input-container ${className}`}>
      {/* Error Toast */}
      {showError && error && (
        <div 
          className={`voice-error-notification ${darkMode ? 'dark' : 'light'}`}
          onClick={() => {
            setShowError(false);
            onErrorDismiss?.();
          }}
        >
          <span className="voice-error-icon">⚠️</span>
          <span className="voice-error-message">{error}</span>
          <button className="voice-error-dismiss" aria-label="Dismiss">×</button>
        </div>
      )}

      {/* Interim Transcript */}
      {showTranscript && isListening && interimTranscript && (
        <div className={`voice-interim-transcript ${darkMode ? 'dark' : 'light'}`}>
          <div className="voice-interim-icon">
            <MicrophoneIcon size={14} animated />
          </div>
          <span className="voice-interim-text">{interimTranscript}</span>
        </div>
      )}

      {/* Voice Button */}
      <button
        onClick={handleClick}
        onTouchEnd={handleTouchEnd}
        className={`voice-input-button ${sizeClasses[size]} ${isListening ? 'listening' : ''} ${!isSupported ? 'unsupported' : ''} ${darkMode ? 'dark' : 'light'}`}
        disabled={disabled}
        aria-label={isListening ? 'Stop listening' : 'Start voice input'}
        title={
          !isSupported 
            ? 'Voice input not supported in this browser' 
            : isListening 
              ? 'Tap to stop' 
              : 'Tap to speak'
        }
        type="button"
      >
        {/* Audio level ring (shown when listening) */}
        {isListening && (
          <div 
            className="voice-audio-ring"
            style={{ 
              transform: `scale(${pulseScale})`,
              opacity: 0.3 + (audioLevel * 0.7)
            }}
          />
        )}

        {/* Microphone Icon */}
        <div className="voice-icon-wrapper">
          {isListening ? (
            <StopIcon size={size === 'small' ? 16 : size === 'large' ? 24 : 20} />
          ) : (
            <MicrophoneIcon 
              size={size === 'small' ? 18 : size === 'large' ? 26 : 22} 
              strikethrough={!isSupported}
            />
          )}
        </div>
      </button>

      {/* Listening indicator text */}
      {isListening && !interimTranscript && (
        <div className={`voice-listening-hint ${darkMode ? 'dark' : 'light'}`}>
          Listening...
        </div>
      )}
    </div>
  );
};

// Microphone Icon Component
const MicrophoneIcon = ({ size = 20, animated = false, strikethrough = false }) => (
  <svg 
    className={`mic-icon ${animated ? 'animated' : ''}`}
    width={size} 
    height={size} 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round"
  >
    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
    <line x1="12" x2="12" y1="19" y2="22"/>
    {strikethrough && (
      <line x1="2" x2="22" y1="2" y2="22" strokeWidth="2.5" stroke="currentColor"/>
    )}
  </svg>
);

// Stop Icon Component
const StopIcon = ({ size = 20 }) => (
  <svg 
    className="stop-icon"
    width={size} 
    height={size} 
    viewBox="0 0 24 24" 
    fill="currentColor"
  >
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);

export default VoiceInputButton;
export { MicrophoneIcon, StopIcon };
