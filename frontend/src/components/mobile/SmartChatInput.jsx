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

import React, { useRef, useEffect, useState, useCallback } from 'react';
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
  enableVoice = true,
  minimal = false  // NEW: Minimal mode for ultra-clean mobile UI
}) => {
  const textareaRef = useRef(null);
  const containerRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState(null);
  const [inputHeight, setInputHeight] = useState(44);
  const [voiceError, setVoiceError] = useState(null);
  const [voiceSupported, setVoiceSupported] = useState(true);
  const [interimTranscript, setInterimTranscript] = useState('');
  
  // Use ref to access current value in callbacks (avoid stale closure)
  const valueRef = useRef(value);
  useEffect(() => {
    valueRef.current = value;
  }, [value]);

  // Detect browser language for speech recognition
  const detectLanguage = useCallback(() => {
    const browserLang = navigator.language || navigator.userLanguage || 'en-US';
    // Support Turkish and English
    if (browserLang.startsWith('tr')) {
      return 'tr-TR';
    }
    return 'en-US';
  }, []);

  // Check if we're on iOS Safari (limited Web Speech API support)
  const isIOSSafari = useCallback(() => {
    const ua = navigator.userAgent;
    const isIOS = /iPad|iPhone|iPod/.test(ua) && !window.MSStream;
    const isSafari = /Safari/.test(ua) && !/Chrome|CriOS|FxiOS/.test(ua);
    return isIOS && isSafari;
  }, []);

  // Initialize speech recognition
  useEffect(() => {
    if (!enableVoice) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.log('Speech recognition not supported in this browser');
      setVoiceSupported(false);
      return;
    }

    // iOS Safari has very limited Web Speech API support
    if (isIOSSafari()) {
      console.log('iOS Safari has limited speech recognition support');
      // Still allow trying, but warn user if it fails
    }

    try {
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = false;
      recognitionInstance.interimResults = true; // Enable interim results for better UX
      recognitionInstance.lang = detectLanguage();
      recognitionInstance.maxAlternatives = 1;

      recognitionInstance.onresult = (event) => {
        let finalTranscript = '';
        let interim = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          if (result.isFinal) {
            finalTranscript += result[0].transcript;
          } else {
            interim += result[0].transcript;
          }
        }
        
        // Show interim results as visual feedback
        if (interim) {
          setInterimTranscript(interim);
        }
        
        // When we get a final result, append it to the input
        if (finalTranscript) {
          // Use functional update to avoid stale closure issues
          const currentValue = valueRef.current;
          const trimmedTranscript = finalTranscript.trim();
          const newValue = currentValue 
            ? currentValue.trim() + ' ' + trimmedTranscript
            : trimmedTranscript;
          
          // CRITICAL: For mobile browsers, we need to:
          // 1. Update the textarea value directly first
          // 2. Then call onChange to sync React state
          // 3. Dispatch an input event to trigger any listeners
          if (textareaRef.current) {
            // Set native value first (mobile browsers need this)
            textareaRef.current.value = newValue;
            
            // Dispatch input event to sync with React and any other listeners
            const inputEvent = new Event('input', { bubbles: true });
            textareaRef.current.dispatchEvent(inputEvent);
            
            // Trigger resize
            textareaRef.current.style.height = 'auto';
            const maxHeight = 100;
            const newHeight = Math.min(textareaRef.current.scrollHeight, maxHeight);
            textareaRef.current.style.height = newHeight + 'px';
          }
          
          // Update React state (belt and suspenders approach for mobile)
          onChange(newValue);
          
          // Also update the ref immediately
          valueRef.current = newValue;
          
          setInterimTranscript('');
          setIsListening(false);
          
          // Focus the textarea after voice input so user can edit or send
          setTimeout(() => {
            if (textareaRef.current) {
              textareaRef.current.focus();
              // Move cursor to end
              textareaRef.current.selectionStart = textareaRef.current.value.length;
              textareaRef.current.selectionEnd = textareaRef.current.value.length;
            }
          }, 100);
          
          // Track successful voice input (analytics)
          try {
            trackEvent('voice_input_success', 'chat_input', 'Voice recognition completed', 
              event.results[0]?.[0]?.confidence || 0);
          } catch (e) {
            console.warn('Analytics tracking failed:', e);
          }
        }
      };

      recognitionInstance.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        setInterimTranscript('');
        
        // Provide user-friendly error messages
        let errorMessage = '';
        switch (event.error) {
          case 'not-allowed':
          case 'permission-denied':
            errorMessage = 'Microphone access denied. Please allow microphone permissions.';
            break;
          case 'no-speech':
            errorMessage = 'No speech detected. Please try again.';
            break;
          case 'audio-capture':
            errorMessage = 'No microphone found. Please check your device.';
            break;
          case 'network':
            errorMessage = 'Network error. Please check your connection.';
            break;
          case 'aborted':
            // User cancelled, no error message needed
            break;
          default:
            errorMessage = 'Voice input failed. Please try again or type your message.';
        }
        
        if (errorMessage) {
          setVoiceError(errorMessage);
          // Clear error after 3 seconds
          setTimeout(() => setVoiceError(null), 3000);
        }
        
        // Track failed voice input (analytics)
        try {
          trackEvent('voice_input_error', 'chat_input', event.error || 'Unknown error', 0);
        } catch (e) {
          console.warn('Analytics tracking failed:', e);
        }
      };

      recognitionInstance.onend = () => {
        setIsListening(false);
        setInterimTranscript('');
      };
      
      recognitionInstance.onaudiostart = () => {
        console.log('Audio capture started');
      };

      setRecognition(recognitionInstance);
      setVoiceSupported(true);
    } catch (e) {
      console.error('Failed to initialize speech recognition:', e);
      setVoiceSupported(false);
    }
  }, [enableVoice, detectLanguage, isIOSSafari, onChange]);

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

  const handleVoiceToggle = async () => {
    if (!recognition) {
      if (!voiceSupported) {
        setVoiceError('Voice input is not supported in this browser. Please try Chrome or Edge.');
        setTimeout(() => setVoiceError(null), 3000);
      }
      return;
    }

    if (isListening) {
      try {
        recognition.stop();
      } catch (e) {
        console.warn('Error stopping recognition:', e);
      }
      setIsListening(false);
      setInterimTranscript('');
    } else {
      // Clear any previous error
      setVoiceError(null);
      
      // Haptic feedback for mobile
      if ('vibrate' in navigator) {
        navigator.vibrate(50);
      }

      try {
        // Request microphone permission first (helps with mobile browsers)
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // Release the stream immediately, we just needed to prompt for permission
            stream.getTracks().forEach(track => track.stop());
          } catch (permError) {
            console.warn('Microphone permission request failed:', permError);
            // Continue anyway, the speech recognition will handle the error
          }
        }
        
        // Update language before starting (in case user changed browser language)
        recognition.lang = detectLanguage();
        
        recognition.start();
        setIsListening(true);
      } catch (e) {
        console.error('Failed to start speech recognition:', e);
        setIsListening(false);
        
        if (e.message?.includes('already started')) {
          // Recognition was already running, try to stop and restart
          try {
            recognition.stop();
            setTimeout(() => {
              recognition.start();
              setIsListening(true);
            }, 100);
          } catch (retryError) {
            setVoiceError('Voice input is busy. Please try again.');
            setTimeout(() => setVoiceError(null), 3000);
          }
        } else {
          setVoiceError('Failed to start voice input. Please try again.');
          setTimeout(() => setVoiceError(null), 3000);
        }
      }
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
      
      {/* Interim transcript indicator */}
      {isListening && interimTranscript && (
        <div className={`interim-transcript ${darkMode ? 'dark' : 'light'}`}>
          <span className="interim-icon">🎤</span>
          <span className="interim-text">{interimTranscript}...</span>
        </div>
      )}
      
      {/* Form wrapper enables mobile keyboard "Send" button to work like IG DM */}
      <form onSubmit={handleFormSubmit} className="smart-chat-input-form">
        <div 
          className="smart-chat-input-wrapper"
          style={{ minHeight: `${inputHeight + 16}px` }}
        >
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
            aria-label={isListening ? 'Stop listening' : 'Start voice input'}
            title={voiceSupported ? 'Voice input' : 'Voice input not supported'}
            type="button"
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
