import React from 'react';

const SearchBar = ({ value, onChange, onSubmit }) => (
  <form onSubmit={onSubmit} className="searchbar" style={{
    display: 'flex',
    alignItems: 'center',
    width: '100%',
    background: 'rgba(35,38,58,0.98)',
    borderRadius: '1.5rem',
    boxShadow: '0 2px 16px 0 #0001',
    padding: '0.5rem 1.25rem',
    border: '1.5px solid #35395a',
    minHeight: '3.5rem',
    maxWidth: 600,
    margin: '0 auto',
    transition: 'max-width 0.3s cubic-bezier(.4,2,.6,1)'
  }}>
    <input
      type="text"
      value={value}
      onChange={onChange}
      placeholder="Type your message..."
      autoFocus
      style={{
        flex: 1,
        background: 'transparent',
        border: 'none',
        outline: 'none',
        color: '#fff',
        fontSize: '1.15rem',
        padding: '0.5rem 0',
        fontWeight: 500,
        letterSpacing: '0.01em',
        minWidth: 0
      }}
    />
    <button type="submit" className="send-btn" aria-label="Send" style={{
      background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)',
      border: 'none',
      borderRadius: '50%',
      width: 40,
      height: 40,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      marginLeft: 10,
      color: '#fff',
      boxShadow: '0 2px 8px 0 #0002',
      cursor: 'pointer',
      transition: 'background 0.2s'
    }}>
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M3 17L17 10L3 3V8.5L12 10L3 11.5V17Z" fill="currentColor"/>
      </svg>
    </button>
  </form>
);

export default SearchBar;
