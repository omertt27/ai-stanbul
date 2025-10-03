import React, { useRef, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

const SearchBar = ({ value, onChange, onSubmit, placeholder, isLoading = false, onFocus, onBlur, expanded = false }) => {
  const { t } = useTranslation();
  const inputRef = useRef(null);
  const [isFocused, setIsFocused] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Mobile detection
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const displayPlaceholder = placeholder || t('chat.searchPlaceholder', 'Ask me anything about Istanbul...');

  const handleFocus = (e) => {
    setIsFocused(true);
    if (onFocus) onFocus(e);
  };

  const handleBlur = (e) => {
    setIsFocused(false);
    if (onBlur) onBlur(e);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isLoading && value.trim()) {
      onSubmit(e);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="ai-chat-searchbar-container" style={{
      width: '100%',
      maxWidth: 'none',
      position: 'relative'
    }}>
      <form onSubmit={handleSubmit} className="ai-chat-searchbar searchbar" style={{
        display: 'flex',
        alignItems: 'center',
        width: '100%',
        maxWidth: '100%',
        background: expanded 
          ? (isFocused ? '#ffffff' : '#f7f7f8')
          : (isFocused ? 'rgba(40, 40, 40, 0.95)' : 'rgba(30, 30, 30, 0.8)'),
        borderRadius: isMobile ? '18px' : '24px',
        border: expanded
          ? (isFocused ? '1px solid #e5e7eb' : '1px solid #f3f4f6')
          : (isFocused ? '1px solid rgba(255, 255, 255, 0.3)' : '1px solid rgba(255, 255, 255, 0.2)'),
        boxShadow: isFocused 
          ? (expanded ? '0 0 0 1px rgba(0, 0, 0, 0.02)' : '0 0 0 1px rgba(255, 255, 255, 0.1)')
          : 'none',
        transition: 'all 0.2s ease',
        minHeight: isMobile ? '49px' : '62px',
        overflow: 'hidden',
        boxSizing: 'border-box',
        margin: 0
      }}>
        
        {/* Input Container */}
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          position: 'relative',
          paddingLeft: isMobile ? '12px' : '16px'
        }}>
          {/* Text Input */}
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={onChange}
            onFocus={handleFocus}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
            placeholder={displayPlaceholder}
            disabled={isLoading}
            className="ai-chat-input"
            style={{
              flex: 1,
              width: '100%',
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: expanded ? '#374151' : '#ffffff',
              fontSize: isMobile ? '16px' : '16px', // Prevent zoom on iOS
              fontWeight: '400',
              lineHeight: '1.5',
              padding: `${isMobile ? '12px' : '14px'} 0`,
              caretColor: expanded ? '#6366f1' : '#ffffff',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              boxSizing: 'border-box',
              minWidth: 0 // Allows flex item to shrink
            }}
            autoComplete="off"
            spellCheck="false"
          />

          {/* Placeholder Enhancement */}
          <style>{`
            .ai-chat-input::placeholder {
              color: ${expanded ? 'rgba(107, 114, 128, 0.6)' : 'rgba(255, 255, 255, 0.6)'};
              font-weight: 400;
              transition: all 0.2s ease;
            }
            .ai-chat-input:focus::placeholder {
              color: ${expanded ? 'rgba(107, 114, 128, 0.4)' : 'rgba(255, 255, 255, 0.4)'};
            }
          `}</style>
        </div>

        {/* Send Button */}
        <div style={{
          padding: isMobile ? '2px' : '3px',
          display: 'flex',
          alignItems: 'center'
        }}>
          <button
            type="submit"
            disabled={isLoading || !value.trim()}
            className="ai-chat-send-button"
            style={{
              background: (isLoading || !value.trim()) 
                ? '#f3f4f6' 
                : '#10a37f',
              border: 'none',
              borderRadius: '50%',
              padding: 0,
              minWidth: isMobile ? '44px' : '48px',
              height: isMobile ? '44px' : '48px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: (isLoading || !value.trim()) ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: 'none',
              transform: 'scale(1)',
              opacity: (isLoading || !value.trim()) ? 0.4 : 1
            }}
            onMouseDown={(e) => {
              if (!isLoading && value.trim()) {
                e.target.style.transform = 'scale(0.95)';
              }
            }}
            onMouseUp={(e) => {
              e.target.style.transform = 'scale(1)';
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'scale(1)';
            }}
          >
            {isLoading ? (
              <div style={{
                width: '16px',
                height: '16px',
                border: '2px solid #e5e7eb',
                borderTop: '2px solid #6b7280',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }} />
            ) : (
              <svg 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
                style={{
                  color: (isLoading || !value.trim()) ? '#9ca3af' : '#ffffff',
                  transform: 'rotate(45deg)',
                  transition: 'all 0.2s ease'
                }}
              >
                <line x1="7" y1="17" x2="17" y2="7"/>
                <polyline points="7,7 17,7 17,17"/>
              </svg>
            )}
          </button>
        </div>
      </form>

      {/* Spinner Animation */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .ai-chat-send-button:hover:not(:disabled) {
          background: #0d8a6b;
          transform: scale(1.05);
        }
        
        .ai-chat-send-button:hover:not(:disabled) svg {
          transform: rotate(45deg) scale(1.1);
        }
      `}</style>
    </div>
  );
};

export default SearchBar;
