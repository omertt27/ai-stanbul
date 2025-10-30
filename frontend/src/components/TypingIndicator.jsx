import React from 'react';

const TypingIndicator = ({ message = "Thinking..." }) => {
  return (
    <div className="typing-indicator">
      <span className="typing-message">{message}</span>
      <div className="typing-dots">
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
      </div>
    </div>
  );
};

export default TypingIndicator;
