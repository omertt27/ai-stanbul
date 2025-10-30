/**
 * SimpleChatInput Component
 * ==========================
 * ChatGPT-style minimalist chat input with clean design
 * 
 * Features:
 * - Single-line input that expands
 * - Rounded send button
 * - Minimal border/shadow
 * - Focus state animation
 * - Enter to send
 */

import React from 'react';

const SimpleChatInput = ({ 
  value, 
  onChange, 
  onSend, 
  loading = false, 
  placeholder = "Message KAM...",
  darkMode = false 
}) => {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !loading) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="simple-chat-input-container">
      <div className={`simple-chat-input-wrapper ${darkMode ? 'dark' : 'light'} ${loading ? 'disabled' : ''}`}>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={loading}
          className="simple-chat-input"
          autoComplete="off"
          autoFocus
        />
        
        <button
          onClick={onSend}
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
          gap: 8px;
          padding: 12px 16px;
          border-radius: 24px;
          border: 1px solid;
          transition: all 0.2s ease;
          background: white;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
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
          box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        }

        .simple-chat-input-wrapper.disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .simple-chat-input {
          flex: 1;
          border: none;
          outline: none;
          font-size: 15px;
          line-height: 1.5;
          background: transparent;
          color: inherit;
          padding: 0;
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }

        .simple-chat-input-wrapper.light .simple-chat-input {
          color: #111827;
        }

        .simple-chat-input-wrapper.dark .simple-chat-input {
          color: #f9fafb;
        }

        .simple-chat-input::placeholder {
          color: #9ca3af;
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
          width: 36px;
          height: 36px;
          border-radius: 50%;
          border: none;
          background: #3b82f6;
          color: white;
          cursor: pointer;
          transition: all 0.2s ease;
          padding: 0;
          flex-shrink: 0;
        }

        .simple-send-button:hover:not(:disabled) {
          background: #2563eb;
          transform: scale(1.05);
        }

        .simple-send-button:active:not(:disabled) {
          transform: scale(0.95);
        }

        .simple-send-button:disabled {
          background: #d1d5db;
          cursor: not-allowed;
          opacity: 0.5;
        }

        .simple-chat-input-wrapper.dark .simple-send-button:disabled {
          background: #4b5563;
        }

        .send-icon {
          width: 18px;
          height: 18px;
        }

        .spinner {
          width: 18px;
          height: 18px;
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

        /* Mobile Responsive */
        @media (max-width: 768px) {
          .simple-chat-input-wrapper {
            padding: 10px 14px;
            border-radius: 20px;
          }

          .simple-chat-input {
            font-size: 16px; /* Prevents iOS zoom */
          }

          .simple-send-button {
            width: 32px;
            height: 32px;
          }

          .send-icon {
            width: 16px;
            height: 16px;
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
      `}</style>
    </div>
  );
};

export default SimpleChatInput;
