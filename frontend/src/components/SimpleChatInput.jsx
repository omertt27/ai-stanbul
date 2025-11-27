/**
 * SimpleChatInput Component
 * ==========================
 * Ultra-modern ChatGPT-style chat input with sleek design
 * 
 * Features:
 * - Single-line input with smooth transitions
 * - Modern pill-shaped design
 * - Subtle shadows and borders
 * - Smooth focus animations
 * - Mobile-optimized touch targets
 * - Enter to send
 */

import React, { useRef, useEffect } from 'react';

const SimpleChatInput = ({ 
  value, 
  onChange, 
  onSend, 
  loading = false, 
  placeholder = "Ask about Istanbul...",
  darkMode = false 
}) => {
  const inputRef = useRef(null);

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

  // Keep focus even when keyboard dismisses temporarily
  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    
    const handleBlur = (e) => {
      const relatedTarget = e.relatedTarget;
      
      // If blur was to a button, allow it
      if (relatedTarget?.tagName === 'BUTTON') {
        return;
      }
      
      // Otherwise, refocus after a short delay
      setTimeout(() => {
        if (document.activeElement !== input && 
            !document.activeElement?.closest('.message-actions')) {
          input.focus();
        }
      }, 100);
    };
    
    input.addEventListener('blur', handleBlur, { passive: true });
    return () => input.removeEventListener('blur', handleBlur);
  }, []);

  return (
    <div className="simple-chat-input-container">
      <div className={`simple-chat-input-wrapper ${darkMode ? 'dark' : 'light'} ${loading ? 'disabled' : ''}`}>
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={loading}
          className="simple-chat-input"
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="sentences"
          spellCheck="true"
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

      <style jsx>{`
        .simple-chat-input-container {
          width: 100%;
          max-width: 100%;
          padding: 0;
          margin: 0;
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

        /* Mobile Responsive - Gemini Style Compact */}
        @media (max-width: 768px) {
          .simple-chat-input-wrapper {
            padding: 8px 12px; /* More compact */
            border-radius: 24px; /* Smaller pill */
            gap: 8px;
          }

          .simple-chat-input {
            font-size: 16px; /* Prevents iOS zoom - CRITICAL */
            padding: 4px 0; /* Tighter */
          }

          .simple-send-button {
            width: 36px; /* Slightly smaller */
            height: 36px;
            min-width: 36px;
            min-height: 36px;
          }

          .send-icon {
            width: 16px; /* Smaller icon */
            height: 16px;
          }

          .spinner {
            width: 16px;
            height: 16px;
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
