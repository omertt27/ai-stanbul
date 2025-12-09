/**
 * SmartChatInput Component
 * =========================
 * Auto-resizing chat input with voice support and smart keyboard handling
 * 
 * Features:
 * - Auto-resize as user types (up to 5 lines)
 * - Voice input support (Web Speech API)
 * - Emoji picker button
 * - Character counter
 * - Smart Enter handling (Enter = send, Shift+Enter = new line)
 * - Auto-focus management
 * - Keyboard detection and scroll adjustment
 * - Placeholder with typing hints
 */

import React, { useRef, useEffect, useState } from 'react';
import { scrollIntoViewSafe } from '../../utils/keyboardDetection';
import { trackEvent } from '../../utils/analytics';
import './SmartChatInput.css';

const SmartChatInput = ({ 
  value, 
  onChange, 
  onSend, 
  loading = false, 
  placeholder = "Ask about Istanbul...",
  darkMode = false,
  maxLength = 1000,
  showCharCounter = false,
  enableVoice = true
}) => {
  const textareaRef = useRef(null);
  const containerRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState(null);
  const [inputHeight, setInputHeight] = useState(44);

  // Initialize speech recognition
  useEffect(() => {
    if (!enableVoice) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.log('Speech recognition not supported');
      return;
    }

    const recognitionInstance = new SpeechRecognition();
    recognitionInstance.continuous = false;
    recognitionInstance.interimResults = false;
    recognitionInstance.lang = 'en-US';

    recognitionInstance.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      onChange(value + ' ' + transcript);
      setIsListening(false);
      
      // Track successful voice input (analytics)
      try {
        trackEvent('voice_input_success', 'chat_input', 'Voice recognition completed', event.results[0][0].confidence);
      } catch (e) {
        console.warn('Analytics tracking failed:', e);
      }
    };

    recognitionInstance.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      setIsListening(false);
      
      // Track failed voice input (analytics)
      try {
        trackEvent('voice_input_error', 'chat_input', event.error || 'Unknown error', 0);
      } catch (e) {
        console.warn('Analytics tracking failed:', e);
      }
    };

    recognitionInstance.onend = () => {
      setIsListening(false);
    };

    setRecognition(recognitionInstance);
  }, [enableVoice]);

  // Auto-resize textarea as content changes
  useEffect(() => {
    if (!textareaRef.current) return;

    const textarea = textareaRef.current;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, 120); // Max 120px (approx 5 lines)
    textarea.style.height = newHeight + 'px';
    setInputHeight(newHeight);
  }, [value]);

  const handleSend = () => {
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
  };

  const handleKeyDown = (e) => {
    // Send on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey && !loading) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleVoiceToggle = () => {
    if (!recognition) return;

    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      // Haptic feedback
      if ('vibrate' in navigator) {
        navigator.vibrate(50);
      }

      recognition.start();
      setIsListening(true);
    }
  };

  // Auto-scroll into view when keyboard appears
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const handleFocus = () => {
      setTimeout(() => {
        scrollIntoViewSafe(containerRef.current, { block: 'end' });
      }, 300); // Delay to wait for keyboard
    };

    textarea.addEventListener('focus', handleFocus, { passive: true });
    
    return () => {
      textarea.removeEventListener('focus', handleFocus);
    };
  }, []);

  const charCount = value.length;
  const isNearLimit = charCount > maxLength * 0.8;

  return (
    <div ref={containerRef} className={`smart-chat-input-container ${darkMode ? 'dark' : 'light'}`}>
      <div 
        className="smart-chat-input-wrapper"
        style={{ minHeight: `${inputHeight + 16}px` }}
      >
        {/* Voice button (left side) */}
        {enableVoice && recognition && (
          <button
            onClick={handleVoiceToggle}
            className={`voice-button ${isListening ? 'listening' : ''}`}
            disabled={loading}
            aria-label={isListening ? 'Stop listening' : 'Start voice input'}
            title="Voice input"
          >
            {isListening ? '🔴' : '🎤'}
          </button>
        )}

        {/* Auto-resizing textarea */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value.slice(0, maxLength))}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={loading}
          className="smart-chat-textarea"
          rows={1}
          autoComplete="off"
          autoCorrect="on"
          autoCapitalize="sentences"
          spellCheck="true"
          maxLength={maxLength}
          aria-label="Chat message input"
        />

        {/* Character counter */}
        {showCharCounter && isNearLimit && (
          <span className={`char-counter ${charCount >= maxLength ? 'limit' : ''}`}>
            {charCount}/{maxLength}
          </span>
        )}

        {/* Send button (right side) */}
        <button
          onClick={handleSend}
          disabled={loading || !value.trim()}
          className="smart-send-button"
          aria-label="Send message"
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

      {/* Keyboard hint */}
      {!value && !isListening && (
        <div className="input-hint">
          Press Enter to send • Shift+Enter for new line
        </div>
      )}
    </div>
  );
};

export default SmartChatInput;
