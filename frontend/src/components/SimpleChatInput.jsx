/**
 * SimpleChatInput Component
 * ==========================
 * Ultra-modern ChatGPT-style chat input with sleek design
 * 
 * Features:
 * - Single-line input with smooth transitions
 * - Voice input support (Web Speech API)
 * - Modern pill-shaped design
 * - Subtle shadows and borders
 * - Smooth focus animations
 * - Mobile-optimized touch targets (44x44px minimum)
 * - Keyboard detection and auto-scroll
 * - Enter to send
 * - Safe area inset support
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { scrollIntoViewSafe } from '../utils/keyboardDetection';
import { trackEvent } from '../utils/analytics';

const SimpleChatInput = ({ 
  value, 
  onChange, 
  onSend, 
  loading = false, 
  placeholder = "Ask about Istanbul...",
  darkMode = false,
  enableVoice = true  // NEW: Enable voice input
}) => {
  const inputRef = useRef(null);
  const containerRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState(null);
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

  // Initialize speech recognition
  useEffect(() => {
    if (!enableVoice) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.log('Speech recognition not supported in this browser');
      setVoiceSupported(false);
      return;
    }

    try {
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = false;
      recognitionInstance.interimResults = true;
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
        
        if (interim) {
          setInterimTranscript(interim);
        }
        
        if (finalTranscript) {
          const currentValue = valueRef.current;
          const newValue = currentValue 
            ? currentValue.trim() + ' ' + finalTranscript.trim()
            : finalTranscript.trim();
          onChange(newValue);
          setInterimTranscript('');
          setIsListening(false);
          
          // Track successful voice input
          try {
            trackEvent('voice_input_success', 'desktop_chat_input', 'Voice recognition completed', 
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
            break;
          default:
            errorMessage = 'Voice input failed. Please try again or type your message.';
        }
        
        if (errorMessage) {
          setVoiceError(errorMessage);
          setTimeout(() => setVoiceError(null), 3000);
        }
        
        try {
          trackEvent('voice_input_error', 'desktop_chat_input', event.error || 'Unknown error', 0);
        } catch (e) {
          console.warn('Analytics tracking failed:', e);
        }
      };

      recognitionInstance.onend = () => {
        setIsListening(false);
        setInterimTranscript('');
      };

      setRecognition(recognitionInstance);
      setVoiceSupported(true);
    } catch (e) {
      console.error('Failed to initialize speech recognition:', e);
      setVoiceSupported(false);
    }
  }, [enableVoice, detectLanguage, onChange]);

  const handleSend = () => {
    if (!value.trim() || loading) return;
    
    // Send the message
    onSend();
    
    // CRITICAL: Immediately refocus the input (ChatGPT-style)
    requestAnimationFrame(() => {
      inputRef.current?.focus();
    });
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !loading) {
      e.preventDefault();
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
      setVoiceError(null);

      try {
        // Request microphone permission
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
          } catch (permError) {
            console.warn('Microphone permission request failed:', permError);
          }
        }
        
        recognition.lang = detectLanguage();
        recognition.start();
        setIsListening(true);
      } catch (e) {
        console.error('Failed to start speech recognition:', e);
        setIsListening(false);
        
        if (e.message?.includes('already started')) {
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
    const input = inputRef.current;
    if (!input) return;

    const handleFocus = () => {
      // Scroll input into view when keyboard appears
      scrollIntoViewSafe(containerRef.current, { block: 'end' });
    };

    input.addEventListener('focus', handleFocus, { passive: true });
    
    return () => {
      input.removeEventListener('focus', handleFocus);
    };
  }, []);

  // Keep focus when blur is caused by clicking send button only
  // DON'T steal focus when user is selecting text in chat messages
  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    
    const handleBlur = (e) => {
      const relatedTarget = e.relatedTarget;
      
      // If blur was to a button (like send), refocus after
      if (relatedTarget?.tagName === 'BUTTON' && 
          relatedTarget.closest('.simple-chat-input-wrapper')) {
        setTimeout(() => input.focus(), 100);
        return;
      }
      
      // DON'T refocus automatically - let users select and copy text
      // The input will be focused again when they click on it or start typing
    };
    
    input.addEventListener('blur', handleBlur, { passive: true });
    return () => input.removeEventListener('blur', handleBlur);
  }, []);

  return (
    <div ref={containerRef} className="simple-chat-input-container">
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
      
      <div className={`simple-chat-input-wrapper ${darkMode ? 'dark' : 'light'} ${loading ? 'disabled' : ''}`}>
        {/* Voice button (left side) */}
        {enableVoice && (
          <button
            onClick={handleVoiceToggle}
            className={`voice-button ${isListening ? 'listening' : ''} ${!voiceSupported ? 'unsupported' : ''}`}
            disabled={loading}
            aria-label={isListening ? 'Stop listening' : 'Start voice input'}
            title={voiceSupported ? 'Voice input' : 'Voice input not supported'}
            type="button"
          >
            {isListening ? '🔴' : '🎤'}
          </button>
        )}
        
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isListening ? 'Listening...' : placeholder}
          disabled={loading}
          className="simple-chat-input"
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="sentences"
          spellCheck="true"
          inputMode="text"
          autoFocus
        />
        
        <button
          onClick={handleSend}
          disabled={loading || !value.trim()}
          className="simple-send-button"
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

      <style>{`
        .simple-chat-input-container {
          width: 100%;
          max-width: 100%;
          padding: 0;
          margin: 0;
          position: relative;
        }

        /* Voice error toast */
        .voice-error-toast {
          position: absolute;
          bottom: calc(100% + 8px);
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 16px;
          border-radius: 12px;
          font-size: 14px;
          font-weight: 500;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          z-index: 1000;
          animation: slideDown 0.3s ease-out;
          max-width: 90%;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .voice-error-toast.light {
          background: #fef2f2;
          color: #991b1b;
          border: 1px solid #fecaca;
        }

        .voice-error-toast.dark {
          background: #7f1d1d;
          color: #fecaca;
          border: 1px solid #991b1b;
        }

        .voice-error-icon {
          font-size: 16px;
          flex-shrink: 0;
        }

        .voice-error-text {
          flex: 1;
          min-width: 0;
        }

        /* Interim transcript indicator */
        .interim-transcript {
          position: absolute;
          bottom: calc(100% + 8px);
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 14px;
          border-radius: 12px;
          font-size: 14px;
          font-style: italic;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          z-index: 999;
          animation: slideDown 0.3s ease-out;
          max-width: 90%;
        }

        .interim-transcript.light {
          background: #eff6ff;
          color: #1e40af;
          border: 1px solid #bfdbfe;
        }

        .interim-transcript.dark {
          background: #1e3a8a;
          color: #bfdbfe;
          border: 1px solid #3b82f6;
        }

        .interim-icon {
          font-size: 16px;
          animation: pulse 1.5s ease-in-out infinite;
        }

        .interim-text {
          flex: 1;
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateX(-50%) translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
          }
        }

        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }

        /* Voice button */
        .voice-button {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          min-width: 32px;
          min-height: 32px;
          border-radius: 50%;
          border: none;
          background: transparent;
          cursor: pointer;
          transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
          padding: 0;
          flex-shrink: 0;
          font-size: 18px;
          opacity: 0.6;
        }

        .voice-button:hover:not(:disabled) {
          opacity: 1;
          background: rgba(59, 130, 246, 0.1);
        }

        .voice-button.listening {
          opacity: 1;
          background: rgba(239, 68, 68, 0.1);
          animation: pulse 1.5s ease-in-out infinite;
        }

        .voice-button.unsupported {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .voice-button:disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .voice-button:focus-visible {
          outline: 2px solid #3b82f6;
          outline-offset: 2px;
        }

        .simple-chat-input-wrapper {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          border-radius: 24px; /* Modern pill shape */
          border: 1.5px solid;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          background: white;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
          position: relative;
        }

        .simple-chat-input-wrapper.light {
          border-color: #e5e7eb;
          background: #ffffff;
        }

        .simple-chat-input-wrapper.dark {
          border-color: #374151;
          background: #1f2937;
        }

        .simple-chat-input-wrapper:focus-within {
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.08), 
                      0 2px 8px rgba(59, 130, 246, 0.12);
        }

        .simple-chat-input-wrapper.disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .simple-chat-input {
          flex: 1;
          border: none;
          outline: none;
          font-size: 15px;
          line-height: 1.6;
          background: transparent;
          color: inherit;
          padding: 6px 0;
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
          font-weight: 400;
        }

        .simple-chat-input-wrapper.light .simple-chat-input {
          color: #111827;
        }

        .simple-chat-input-wrapper.dark .simple-chat-input {
          color: #f9fafb;
        }

        .simple-chat-input::placeholder {
          color: #9ca3af;
          font-weight: 400;
        }

        .simple-chat-input-wrapper.dark .simple-chat-input::placeholder {
          color: #6b7280;
        }

        .simple-chat-input:disabled {
          cursor: not-allowed;
        }

        .simple-send-button {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          min-width: 32px;
          min-height: 32px;
          border-radius: 50%;
          border: none;
          background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
          color: white;
          cursor: pointer;
          transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
          padding: 0;
          flex-shrink: 0;
          box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
        }

        .simple-send-button:hover:not(:disabled) {
          background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
          transform: scale(1.08);
          box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
        }

        .simple-send-button:active:not(:disabled) {
          transform: scale(0.95);
          box-shadow: 0 1px 2px rgba(59, 130, 246, 0.2);
        }

        .simple-send-button:disabled {
          background: #d1d5db;
          cursor: not-allowed;
          opacity: 0.4;
          box-shadow: none;
        }

        .simple-chat-input-wrapper.dark .simple-send-button:disabled {
          background: #4b5563;
        }

        .send-icon {
          width: 16px;
          height: 16px;
        }

        .spinner {
          width: 16px;
          height: 16px;
          animation: spin 1s linear infinite;
        }

        .spinner-circle {
          stroke: currentColor;
          stroke-dasharray: 60;
          stroke-dashoffset: 60;
          animation: dash 1.5s ease-in-out infinite;
        }

        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }

        @keyframes dash {
          0% {
            stroke-dashoffset: 60;
          }
          50% {
            stroke-dashoffset: 15;
          }
          100% {
            stroke-dashoffset: 60;
          }
        }

        /* Mobile Responsive - Ergonomic & Accessible (WCAG AAA) */
        @media (max-width: 768px) {
          .simple-chat-input-wrapper {
            padding: 4px 4px 4px 16px; /* Asymmetric padding for button */
            border-radius: 26px; /* Larger pill for 52px height */
            gap: 8px;
            height: 52px; /* Larger for easier thumb reach */
          }

          .simple-chat-input {
            font-size: 16px !important; /* Prevents iOS zoom on focus */
            padding: 0;
            line-height: 44px; /* Vertically center text */
          }

          .simple-send-button {
            width: 44px !important; /* WCAG 2.5.5 Level AAA (44x44px) */
            height: 44px !important;
            min-width: 44px !important;
            min-height: 44px !important;
            border-radius: 22px !important;
            margin-right: 0;
            flex-shrink: 0;
          }

          .send-icon {
            width: 18px; /* Larger icon for visibility */
            height: 18px;
          }

          .spinner {
            width: 18px;
            height: 18px;
          }
          
          /* Active state feedback */
          .simple-send-button:active:not(:disabled) {
            transform: scale(0.95) !important;
            transition: transform 0.1s ease !important;
          }
        }

        /* Tablet optimization */
        @media (min-width: 769px) and (max-width: 1024px) {
          .simple-chat-input-wrapper {
            padding: 11px 15px;
          }

          .simple-send-button {
            width: 36px;
            height: 36px;
            min-width: 36px;
            min-height: 36px;
          }
        }

        /* Focus visible for accessibility */
        .simple-send-button:focus-visible {
          outline: 2px solid #3b82f6;
          outline-offset: 2px;
        }

        .simple-chat-input:focus-visible {
          outline: none;
        }

        /* Smooth transitions for dark mode */
        .simple-chat-input-wrapper,
        .simple-chat-input,
        .simple-send-button {
          transition-property: all;
          transition-duration: 0.2s;
          transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
        }
      `}</style>
    </div>
  );
};

export default SimpleChatInput;
