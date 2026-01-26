/**
 * SmartChatInput Component
 * =========================
 * Auto-resizing chat input with voice support and smart keyboard handling
 * 
 * Features:
 * - Auto-resize as user types (up to 5 lines)
 * - Voice input support (enhanced with useVoiceInput hook)
 * - Emoji picker button
 * - Character counter
 * - Smart Enter handling (Enter = send, Shift+Enter = new line)
 * - Auto-focus management
 * - Keyboard detection and scroll adjustment
 * - Placeholder with typing hints
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { scrollIntoViewSafe } from '../../utils/keyboardDetection';
import { trackEvent } from '../../utils/analytics';
import useVoiceInput, { SUPPORTED_VOICE_LANGUAGES, detectBrowserLanguage } from '../../hooks/useVoiceInput';
import VoiceLanguagePicker from './VoiceLanguagePicker';
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
  enableVoice = true,
  minimal = false,  // Minimal mode for ultra-clean mobile UI
  voiceLanguage: initialVoiceLanguage = 'auto', // 'auto' detects from browser, or specify 'en-US', 'tr-TR', etc.
  showLanguagePicker = true // Show language picker next to mic button
}) => {
  const textareaRef = useRef(null);
  const containerRef = useRef(null);
  const [inputHeight, setInputHeight] = useState(44);
  
  // Voice language state - allows user to change language
  const [selectedLanguage, setSelectedLanguage] = useState(() => {
    // Initialize from prop or detect from browser
    if (initialVoiceLanguage === 'auto') {
      return detectBrowserLanguage();
    }
    return initialVoiceLanguage;
  });
  
  // Use ref to access current value in callbacks (avoid stale closure)
  const valueRef = useRef(value);
  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  // Use the enhanced voice input hook
  const {
    isListening,
    isSupported: voiceSupported,
    interimTranscript,
    error: voiceError,
    audioLevel,
    toggleListening,
    clearError,
    setLanguage,
    browserInfo
  } = useVoiceInput({
    language: selectedLanguage,
    continuous: false, // Stop after silence for chat messages
    silenceTimeout: 2500,
    onResult: useCallback((transcript) => {
      // When we get a final result, append it to the input
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
    }, [onChange, minimal]),
    onInterim: useCallback((interim) => {
      // Interim results are handled by the hook's state
    }, []),
    onError: useCallback((errorMsg) => {
      try {
        trackEvent('voice_input_error', 'chat_input', errorMsg);
      } catch (e) {
        console.warn('Analytics tracking failed:', e);
      }
    }, [])
  });

  // Auto-resize textarea as content changes
  useEffect(() => {
    if (!textareaRef.current) return;

    const textarea = textareaRef.current;
    textarea.style.height = 'auto';
    // Reduced max height for mobile: 80px = ~3 lines (was 120px = ~5 lines)
    // Keeps input compact and conversation visible, ChatGPT-style
    const maxHeight = minimal ? 80 : 100;
    const newHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = newHeight + 'px';
    setInputHeight(newHeight);
  }, [value, minimal]);

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
    // Send on Enter (without Shift) - works for both desktop and mobile
    if (e.key === 'Enter' && !e.shiftKey && !loading) {
      e.preventDefault();
      handleSend();
    }
  };

  // Handle form submission (for mobile keyboard "Send" button)
  const handleFormSubmit = (e) => {
    e.preventDefault();
    if (!loading && value.trim()) {
      handleSend();
    }
  };

  // Voice toggle is now handled by the useVoiceInput hook
  const handleVoiceToggle = useCallback(async () => {
    if (!enableVoice) return;
    await toggleListening();
  }, [enableVoice, toggleListening]);

  // Handle voice language change
  const handleLanguageChange = useCallback((langCode) => {
    setSelectedLanguage(langCode);
    setLanguage(langCode);
    
    // Track language change
    try {
      const langInfo = SUPPORTED_VOICE_LANGUAGES[langCode];
      trackEvent('voice_language_changed', 'chat_input', langInfo?.name || langCode);
    } catch (e) {
      console.warn('Analytics tracking failed:', e);
    }
  }, [setLanguage]);

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
  
  // Show voice button: always show if voice is enabled (even if not supported, to show error)
  // In minimal mode: hide voice when typing
  const showVoice = enableVoice && (!minimal || !value);
  const showCounter = showCharCounter && isNearLimit;

  return (
    <div ref={containerRef} className={`smart-chat-input-container ${darkMode ? 'dark' : 'light'} ${minimal ? 'minimal' : ''}`}>
      {/* Voice error notification */}
      {voiceError && (
        <div className={`voice-error-toast ${darkMode ? 'dark' : 'light'}`}>
          <span className="voice-error-icon">⚠️</span>
          <span className="voice-error-text">{voiceError}</span>
        </div>
      )}
      
      {/* Interim transcript indicator - shows selected language */}
      {isListening && interimTranscript && (
        <div className={`interim-transcript ${darkMode ? 'dark' : 'light'}`}>
          <span className="interim-lang-badge">{SUPPORTED_VOICE_LANGUAGES[selectedLanguage]?.flag}</span>
          <svg className="interim-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{width: '16px', height: '16px'}}>
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" x2="12" y1="19" y2="22"/>
          </svg>
          <span className="interim-text">{interimTranscript}...</span>
        </div>
      )}
      
      {/* Listening indicator without transcript */}
      {isListening && !interimTranscript && (
        <div className={`interim-transcript listening-hint ${darkMode ? 'dark' : 'light'}`}>
          <span className="interim-lang-badge">{SUPPORTED_VOICE_LANGUAGES[selectedLanguage]?.flag}</span>
          <span className="interim-text">Listening in {SUPPORTED_VOICE_LANGUAGES[selectedLanguage]?.name}...</span>
        </div>
      )}
      
      {/* Form wrapper enables mobile keyboard "Send" button to work like IG DM */}
      <form onSubmit={handleFormSubmit} className="smart-chat-input-form">
        <div 
          className="smart-chat-input-wrapper"
          style={{ minHeight: `${inputHeight + 16}px` }}
        >
        {/* Language picker (shows when voice is enabled) */}
        {showVoice && showLanguagePicker && !isListening && (
          <VoiceLanguagePicker
            currentLanguage={selectedLanguage}
            onLanguageChange={handleLanguageChange}
            darkMode={darkMode}
            disabled={loading || isListening}
            compact={true}
          />
        )}
        
        {/* Voice button (left side) - hidden in minimal mode when typing */}
        {showVoice && (
          <button
            onClick={handleVoiceToggle}
            onTouchEnd={(e) => {
              // Prevent double-tap zoom on mobile
              e.preventDefault();
              handleVoiceToggle();
            }}
            className={`voice-button ${isListening ? 'listening' : ''} ${!voiceSupported ? 'unsupported' : ''}`}
            disabled={loading}
            aria-label={isListening ? 'Stop listening' : `Start voice input (${SUPPORTED_VOICE_LANGUAGES[selectedLanguage]?.name})`}
            title={voiceSupported ? `Voice input (${SUPPORTED_VOICE_LANGUAGES[selectedLanguage]?.name})` : 'Voice input not supported'}
            type="button"
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
          placeholder={isListening ? 'Listening...' : placeholder}
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

        {/* Character counter - only when near limit in minimal mode */}
        {showCounter && (
          <span className={`char-counter ${charCount >= maxLength ? 'limit' : ''}`}>
            {charCount}/{maxLength}
          </span>
        )}

        {/* Send button (right side) - type="submit" for form integration */}
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

export default SmartChatInput;
