
import React, { useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { LoadingSkeleton } from './LoadingSkeletons';

const SearchBar = ({ value, onChange, onSubmit, placeholder, isLoading = false }) => {
  const { t } = useTranslation();
  const inputRef = useRef(null);

  const displayPlaceholder = placeholder || t('chat.searchPlaceholder');

  useEffect(() => {
    if (inputRef.current && value === '') {
      inputRef.current.placeholder = displayPlaceholder || '';
    }
  }, [displayPlaceholder, value]);

  return (
    <form onSubmit={onSubmit} className="searchbar" style={{
      display: 'flex',
      alignItems: 'center',
      width: '100%',
      background: 'rgba(28,30,42,0.98)',
      borderRadius: '0.8rem',
      boxShadow: '0 4px 16px 0 #0001',
      padding: '0.6rem 1.2rem',
      border: '1.5px solid #2a2e44',
      minHeight: '3.2rem',
      maxWidth: 750,
      margin: '0 auto',
      transition: 'max-width 0.3s cubic-bezier(.4,2,.6,1)',
      backdropFilter: 'none'
    }}>
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={onChange}
        placeholder={value === '' ? displayPlaceholder : ''}
        disabled={isLoading}
        style={{
          flex: 1,
          background: 'transparent',
          border: 'none',
          outline: 'none',
          color: '#fff',
          fontSize: '1.2rem',
          padding: '0.4rem 0 0.4rem 1.5rem',
          fontWeight: 500,
          letterSpacing: '0.01em',
          minWidth: 0,
          opacity: isLoading ? 0.7 : 1
        }}
        className="placeholder-gray-400"
      />
      
      {/* Submit button with loading state */}
      <button
        type="submit"
        disabled={isLoading || !value.trim()}
        style={{
          background: isLoading || !value.trim() ? 'rgba(99, 102, 241, 0.5)' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          border: 'none',
          borderRadius: '0.5rem',
          padding: '0.5rem 1rem',
          color: 'white',
          fontSize: '0.9rem',
          fontWeight: 600,
          cursor: isLoading || !value.trim() ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          minWidth: '80px',
          justifyContent: 'center'
        }}
        onMouseOver={(e) => {
          if (!isLoading && value.trim()) {
            e.target.style.transform = 'scale(1.05)';
          }
        }}
        onMouseOut={(e) => {
          e.target.style.transform = 'scale(1)';
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
