/**
 * SmartChatInput Component (v2)
 * ==============================
 * Auto-resizing chat input with robust voice support and smart keyboard handling
 * 
 * Features:
 * - Auto-resize as user types (up to 5 lines)
 * - Voice input support (enhanced with useVoiceInput hook)
 * - Character counter
 * - Smart Enter handling (Enter = send, Shift+Enter = new line)
 * - Auto-focus management
 * - Keyboard detection and scroll adjustment
 * - iOS/Android compatibility
 * - English language default for tourism app
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { scrollIntoViewSafe } from '../../utils/keyboardDetection';
import { trackEvent } from '../../utils/analytics';
import useVoiceInput from '../../hooks/useVoiceInput';
import './SmartChatInput.css';

const SmartChatInputV2 = ({ 
  value, 
  onChange, 
  onSend, 
  loading = false, 
  placeholder = "Ask about Istanbul...",
  darkMode = false,
  maxLength = 1000,
  showCharCounter = false,
  enableVoice = true,
  minimal = false,  // Minimal mode for ultra-clean mobile UI
  voiceLanguage = 'en-US' // Default to English for tourism app
}) => {
  const textareaRef = useRef(null);
  const containerRef = useRef(null);
  const [inputHeight, setInputHeight] = useState(44);
  
  // Use ref to access current value in callbacks (avoid stale closure)
  const valueRef = useRef(value);
  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  // Callback when voice recognition completes
  const handleVoiceResult = useCallback((transcript) => {
    const currentValue = valueRef.current;
    const trimmedTranscript = transcript.trim();
    const newValue = currentValue 
      ? currentValue.trim() + ' ' + trimmedTranscript
      : trimmedTranscript;
    
    // Update textarea directly for mobile compatibility
    if (textareaRef.current) {
      textareaRef.current.value = newValue;
      
      // Trigger resize
      textareaRef.current.style.height = 'auto';
      const maxHeight = minimal ? 80 : 100;
      const newHeight = Math.min(textareaRef.current.scrollHeight, maxHeight);
      textareaRef.current.style.height = newHeight + 'px';
    }
    
    // Update React state
    onChange(newValue);
    valueRef.current = newValue;
    
    // Focus the textarea after voice input
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
        textareaRef.current.selectionStart = textareaRef.current.value.length;
        textareaRef.current.selectionEnd = textareaRef.current.value.length;
      }
    }, 100);
    
    // Track successful voice input
    try {
      trackEvent('voice_input_success', 'chat_input', 'Voice recognition completed');
    } catch (e) {
      console.warn('Analytics tracking failed:', e);
    }
  }, [onChange, minimal]);

  // Callback for voice errors
  const handleVoiceError = useCallback((errorMsg) => {
    try {
      trackEvent('voice_input_error', 'chat_input', errorMsg);
    } catch (e) {
      console.warn('Analytics tracking failed:', e);
    }
  }, []);

  // Use the enhanced voice input hook
  const {
    isListening,
    isSupported: voiceSupported,
    interimTranscript,
    error: voiceError,
    audioLevel,
    toggleListening,
    clearError
  } = useVoiceInput({
    language: voiceLanguage,
    continuous: false, // Stop after silence for chat messages
    silenceTimeout: 2500,
    onResult: handleVoiceResult,
    onError: handleVoiceError
  });

  // Auto-resize textarea as content changes
  useEffect(() => {
    if (!textareaRef.current) return;

    const textarea = textareaRef.current;
    textarea.style.height = 'auto';
    const maxHeight = minimal ? 80 : 100;
    const newHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = newHeight + 'px';
    setInputHeight(newHeight);
  }, [value, minimal]);

  const handleSend = useCallback(() => {
    if (!value.trim() || loading) return;
    
    onSend();
    
    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = '44px';
      setInputHeight(44);
    }
    
    // Refocus input
    requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
  }, [value, loading, onSend]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey && !loading) {
      e.preventDefault();
      handleSend();
    }
  }, [loading, handleSend]);

  const handleFormSubmit = useCallback((e) => {
    e.preventDefault();
    if (!loading && value.trim()) {
      handleSend();
    }
  }, [loading, value, handleSend]);

  const handleVoiceToggle = useCallback(async () => {
    if (!enableVoice) return;
    await toggleListening();
  }, [enableVoice, toggleListening]);

  // Auto-scroll into view when keyboard appears
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const handleFocus = () => {
      setTimeout(() => {
        scrollIntoViewSafe(containerRef.current, { block: 'end' });
      }, 300);
    };

    textarea.addEventListener('focus', handleFocus, { passive: true });
    
    return () => {
      textarea.removeEventListener('focus', handleFocus);
    };
  }, []);

  const charCount = value.length;
  const isNearLimit = charCount > maxLength * 0.8;
  
  const showVoice = enableVoice && (!minimal || !value);
  const showCounter = showCharCounter && isNearLimit;

  // Calculate pulse scale for audio level visualization
  const pulseScale = isListening ? 1 + (audioLevel * 0.3) : 1;

  return (
    <div ref={containerRef} className={`smart-chat-input-container ${darkMode ? 'dark' : 'light'} ${minimal ? 'minimal' : ''}`}>
      {/* Voice error notification */}
      {voiceError && (
        <div 
          className={`voice-error-toast ${darkMode ? 'dark' : 'light'}`}
          onClick={clearError}
        >
          <span className="voice-error-icon">⚠️</span>
          <span className="voice-error-text">{voiceError}</span>
        </div>
      )}
      
      {/* Interim transcript indicator */}
      {isListening && interimTranscript && (
        <div className={`interim-transcript ${darkMode ? 'dark' : 'light'}`}>
          <svg className="interim-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{width: '16px', height: '16px'}}>
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" x2="12" y1="19" y2="22"/>
          </svg>
          <span className="interim-text">{interimTranscript}...</span>
        </div>
      )}
      
      {/* Listening indicator without interim transcript */}
      {isListening && !interimTranscript && (
        <div className={`listening-indicator ${darkMode ? 'dark' : 'light'}`}>
          <div className="listening-dots">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
          </div>
          <span className="listening-text">Listening...</span>
        </div>
      )}
      
      {/* Form wrapper enables mobile keyboard "Send" button */}
      <form onSubmit={handleFormSubmit} className="smart-chat-input-form">
        <div 
          className="smart-chat-input-wrapper"
          style={{ minHeight: `${inputHeight + 16}px` }}
        >
          {/* Voice button (left side) */}
          {showVoice && (
            <button
              onClick={handleVoiceToggle}
              onTouchEnd={(e) => {
                e.preventDefault();
                handleVoiceToggle();
              }}
              className={`voice-button ${isListening ? 'listening' : ''} ${!voiceSupported ? 'unsupported' : ''}`}
              disabled={loading}
              aria-label={isListening ? 'Stop listening' : 'Start voice input'}
              title={voiceSupported ? 'Voice input (English)' : 'Voice input not supported'}
              type="button"
              style={{
                transform: isListening ? `scale(${pulseScale})` : 'scale(1)',
                transition: 'transform 0.1s ease-out'
              }}
            >
              {isListening ? (
                <svg className="voice-icon recording" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              ) : (
                <svg className="voice-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" x2="12" y1="19" y2="22"/>
                </svg>
              )}
            </button>
          )}

          {/* Auto-resizing textarea */}
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value.slice(0, maxLength))}
            onKeyDown={handleKeyDown}
            placeholder={isListening ? 'Listening... speak now' : placeholder}
            disabled={loading}
            className="smart-chat-textarea"
            rows={1}
            autoComplete="off"
            autoCorrect="on"
            autoCapitalize="sentences"
            spellCheck="true"
            maxLength={maxLength}
            aria-label="Chat message input"
            enterKeyHint="send"
            inputMode="text"
          />

          {/* Character counter */}
          {showCounter && (
            <span className={`char-counter ${charCount >= maxLength ? 'limit' : ''}`}>
              {charCount}/{maxLength}
            </span>
          )}

          {/* Send button (right side) */}
          <button
            disabled={loading || !value.trim()}
            className="smart-send-button"
            aria-label="Send message"
            type="submit"
          >
            {loading ? (
              <svg className="spinner" viewBox="0 0 24 24">
                <circle 
                  className="spinner-circle"
                  cx="12" 
                  cy="12" 
                  r="10" 
                  fill="none" 
                  strokeWidth="3"
                />
              </svg>
            ) : (
              <svg className="send-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2.5} 
                  d="M5 12h14m0 0l-6-6m6 6l-6 6" 
                />
              </svg>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SmartChatInputV2;
