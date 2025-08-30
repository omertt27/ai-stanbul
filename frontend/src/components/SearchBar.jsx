import React from 'react';

const SearchBar = ({ value, onChange, onSubmit }) => (
  <form onSubmit={onSubmit} className="searchbar">
    <input
      type="text"
      value={value}
      onChange={onChange}
      placeholder="Type your message..."
      autoFocus
    />
    <button type="submit" className="send-btn" aria-label="Send">
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M3 17L17 10L3 3V8.5L12 10L3 11.5V17Z" fill="currentColor"/>
      </svg>
    </button>
  </form>
);

export default SearchBar;
