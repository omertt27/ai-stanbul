import React from 'react';
import './SuggestionChips.css';

const SuggestionChips = ({ suggestions, onSuggestionClick }) => {
  if (!suggestions || suggestions.length === 0) {
    return null;
  }

  return (
    <div className="suggestion-chips-container">
      <div className="suggestion-chips-label">Suggestions:</div>
      <div className="suggestion-chips">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            className="suggestion-chip"
            onClick={() => onSuggestionClick(suggestion)}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SuggestionChips;
