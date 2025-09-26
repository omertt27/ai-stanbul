
import React, { useRef, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { LoadingSkeleton } from './LoadingSkeletons';
import { useMobileUtils } from '../hooks/useMobileUtils';

const SearchBar = ({ value, onChange, onSubmit, placeholder, isLoading = false }) => {
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
  const handleFocus = () => {
    setIsFocused(true);
    if (isMobile) {
      // Prevent zoom on iOS when focusing input
      preventZoom();
    }
  };

  const handleBlur = () => {
    setIsFocused(false);
    if (isMobile) {
      // Restore zoom capability
      restoreZoom();
    }
  };

  const handleSubmitWithHaptic = (e) => {
    if (isMobile && value.trim()) {
      hapticFeedback('light');
    }
    onSubmit(e);
  };

  return (
    <form onSubmit={handleSubmitWithHaptic} className="searchbar mobile-touch-target" style={{
      display: 'flex',
      alignItems: 'center',
      width: '100%',
      background: 'rgba(28,30,42,0.98)',
      borderRadius: '0.8rem',
      boxShadow: isFocused ? '0 8px 32px rgba(99, 102, 241, 0.2)' : '0 4px 16px 0 #0001',
      padding: '0.6rem 1.2rem',
      border: isFocused ? '1.5px solid #6366f1' : '1.5px solid #2a2e44',
      minHeight: isMobile ? '44px' : '3.2rem',
      maxWidth: 750,
      margin: '0 auto',
      transition: 'all 0.3s cubic-bezier(.4,2,.6,1)',
      backdropFilter: 'blur(10px)'
    }}>
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={onChange}
        onFocus={handleFocus}
        onBlur={handleBlur}
        placeholder={value === '' ? displayPlaceholder : ''}
        disabled={isLoading}
        className="chat-input placeholder-gray-400"
        style={{
          flex: 1,
          background: 'transparent',
          border: 'none',
          outline: 'none',
          color: '#fff',
          fontSize: isMobile ? '16px' : '1.2rem', // Prevents zoom on iOS
          padding: '0.4rem 0 0.4rem 1.5rem',
          fontWeight: 500,
          letterSpacing: '0.01em',
          minWidth: 0,
          opacity: isLoading ? 0.7 : 1,
          minHeight: isMobile ? '44px' : 'auto'
        }}
      />
      
      {/* Submit button with loading state */}
      <button
        type="submit"
        disabled={isLoading || !value.trim()}
        className="mobile-touch-target haptic-feedback"
        style={{
          background: isLoading || !value.trim() ? 'rgba(99, 102, 241, 0.5)' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          border: 'none',
          borderRadius: '0.5rem',
          padding: isMobile ? '12px 16px' : '0.5rem 1rem',
          color: 'white',
          fontSize: isMobile ? '14px' : '0.9rem',
          fontWeight: 600,
          cursor: isLoading || !value.trim() ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          minWidth: isMobile ? '60px' : '80px',
          minHeight: isMobile ? '44px' : 'auto',
          justifyContent: 'center',
          touchAction: 'manipulation'
        }}
        onMouseOver={(e) => {
          if (!isLoading && value.trim() && !isMobile) {
            e.target.style.transform = 'scale(1.05)';
          }
        }}
        onMouseOut={(e) => {
          if (!isMobile) {
            e.target.style.transform = 'scale(1)';
          }
        }}
        onTouchStart={() => {
          if (isMobile && value.trim() && !isLoading) {
            hapticFeedback('light');
          }
        }}
      >
        {isLoading ? (
          <>
            <div style={{
              width: '14px',
              height: '14px',
              border: '2px solid rgba(255,255,255,0.3)',
              borderTop: '2px solid white',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }} />
            <span>Sending...</span>
            <style>{`
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
            `}</style>
          </>
        ) : (
          <>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22,2 15,22 11,13 2,9 22,2"></polygon>
            </svg>
            <span>Send</span>
          </>
        )}
      </button>
    </form>
  );
};

export default SearchBar;
