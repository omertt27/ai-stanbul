import React, { useRef, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { LoadingSkeleton } from './LoadingSkeletons';
import { useMobileUtils } from '../hooks/useMobileUtils';
import '../styles/mobile-enhanced.css';

const SearchBar = ({ value, onChange, onSubmit, placeholder, isLoading = false, onFocus, onBlur }) => {
  const { t } = useTranslation();
  const inputRef = useRef(null);
  const [isFocused, setIsFocused] = useState(false);
  const { isMobile, hapticFeedback, preventZoom, restoreZoom } = useMobileUtils();

  const displayPlaceholder = placeholder || t('chat.searchPlaceholder');

  useEffect(() => {
    if (inputRef.current && value === '') {
      inputRef.current.placeholder = displayPlaceholder || '';
    }
  }, [displayPlaceholder, value]);

  // Mobile-specific input handling
  const handleFocus = (e) => {
    setIsFocused(true);
    if (isMobile) {
      // Prevent zoom on iOS when focusing input
      preventZoom();
    }
    // Call external onFocus handler if provided
    if (onFocus) {
      onFocus(e);
    }
  };

  const handleBlur = (e) => {
    setIsFocused(false);
    if (isMobile) {
      // Restore zoom capability
      restoreZoom();
    }
    // Call external onBlur handler if provided
    if (onBlur) {
      onBlur(e);
    }
  };

  const handleSubmitWithHaptic = (e) => {
    if (isMobile && value.trim()) {
      hapticFeedback('light');
    }
    onSubmit(e);
  };

  return (
    <form onSubmit={handleSubmitWithHaptic} className="searchbar" style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      width: '100%',
      background: 'rgba(28,30,42,0.98)',
      borderRadius: '0.8rem',
      boxShadow: isFocused ? '0 8px 32px rgba(99, 102, 241, 0.2)' : '0 4px 16px 0 #0001',
      padding: '0.6rem 1.2rem',
      border: 'none',
      minHeight: '3.2rem',
      maxWidth: 750,
      margin: '0 auto',
      transition: 'all 0.2s ease',
      backdropFilter: 'blur(10px)',
      position: 'relative',
      overflow: 'visible',
    }}>

      
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={onChange}
        onFocus={handleFocus}
        onBlur={handleBlur}
        placeholder={displayPlaceholder || "What would you like to know about Istanbul?"}
        disabled={isLoading}
        className="chat-input"
        style={{
          flex: 1,
          background: 'transparent',
          border: 'none',
          outline: 'none',
          color: '#fff',
          fontSize: '1.2rem',
          padding: '0.4rem 0.5rem 0.4rem 1.5rem',
          fontWeight: 500,
          letterSpacing: '0.01em',
          minWidth: '200px',
          width: '100%',
          opacity: isLoading ? 0.7 : 1,
          lineHeight: 'normal',
          caretColor: '#fff',
        }}
      />
      
      {/* Submit button with loading state - ChatGPT style */}
      <button
        type="submit"
        disabled={isLoading || !value.trim()}
        style={{
          background: isLoading || !value.trim() ? 'rgba(99, 102, 241, 0.5)' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          border: 'none',
          borderRadius: '0.5rem',
          padding: '0.6rem 1.2rem',
          color: 'white',
          fontSize: '0.9rem',
          fontWeight: 600,
          cursor: isLoading || !value.trim() ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          minWidth: '80px',
          height: 'auto',
          justifyContent: 'center',
        }}
      >
        {isLoading ? (
          <div style={{
            width: isMobile ? '16px' : '14px',
            height: isMobile ? '16px' : '14px',
            border: `2px solid ${isMobile ? 'rgba(26,26,26,0.3)' : 'rgba(255,255,255,0.3)'}`,
            borderTop: `2px solid ${isMobile ? '#1a1a1a' : 'white'}`,
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
        ) : (
          <>
            <svg 
              width={isMobile ? "16" : "18"} 
              height={isMobile ? "16" : "18"} 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2.5" 
              strokeLinecap="round" 
              strokeLinejoin="round"
              style={{
                transition: 'all 0.2s ease',
                transform: isMobile ? 'rotate(-45deg)' : 'none'
              }}
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22,2 15,22 11,13 2,9 22,2"></polygon>
            </svg>
            {!isMobile && (
              <span style={{
                fontWeight: 600,
                letterSpacing: '0.025em',
                textShadow: '0 1px 2px rgba(0, 0, 0, 0.2)'
              }}>Send</span>
            )}
          </>
        )}
      </button>
    </form>
  );
};

export default SearchBar;
