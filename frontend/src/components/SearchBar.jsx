import React from 'react';

const SearchBar = ({ value, onChange, onSubmit }) => (
  <form onSubmit={onSubmit} className="searchbar" style={{
    display: 'flex',
    alignItems: 'center',
    width: '100%',
    background: 'rgba(28,30,42,0.98)',
    borderRadius: '1.5rem',
    boxShadow: '0 2px 16px 0 #0001',
    padding: '0.5rem 1.25rem',
    border: '1.5px solid #2a2e44',
    minHeight: '3.5rem',
    maxWidth: 750,
    margin: '0 auto',
    transition: 'max-width 0.3s cubic-bezier(.4,2,.6,1)'
  }}>
    <input
      type="text"
      value={value}
      onChange={onChange}
      placeholder="Welcome to Istanbul!"
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
  </form>
);

export default SearchBar;
