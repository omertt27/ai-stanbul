
import React, { useRef, useEffect } from 'react';

const SearchBar = ({ value, onChange, onSubmit, placeholder }) => {
  const inputRef = useRef(null);
  const isLightMode = document.body.classList.contains('light');

  useEffect(() => {
    if (inputRef.current && value === '') {
      inputRef.current.placeholder = placeholder || '';
    }
  }, [placeholder, value]);

  return (
    <form onSubmit={onSubmit} className="searchbar" style={{
      display: 'flex',
      alignItems: 'center',
      width: '100%',
      background: isLightMode ? 'rgba(255, 255, 255, 0.95)' : 'rgba(28,30,42,0.98)',
      borderRadius: '1.5rem',
      boxShadow: isLightMode ? '0 4px 20px rgba(99, 102, 241, 0.15)' : '0 2px 16px 0 #0001',
      padding: '0.6rem 1.2rem',
      border: isLightMode ? '1.5px solid #e2e8f0' : '1.5px solid #2a2e44',
      minHeight: '3.2rem',
      maxWidth: 750,
      margin: '0 auto',
      transition: 'max-width 0.3s cubic-bezier(.4,2,.6,1)',
      backdropFilter: isLightMode ? 'blur(10px)' : 'none'
    }}>
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={onChange}
        placeholder={value === '' ? (placeholder || '') : ''}
        autoFocus
        style={{
          flex: 1,
          background: 'transparent',
          border: 'none',
          outline: 'none',
          color: isLightMode ? '#1f2937' : '#fff',
          fontSize: '1.2rem',
          padding: '0.4rem 0',
          fontWeight: 500,
          letterSpacing: '0.01em',
          minWidth: 0
        }}
      />
    </form>
  );
};

export default SearchBar;
