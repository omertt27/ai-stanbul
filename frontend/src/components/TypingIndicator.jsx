import React from 'react';

const TypingIndicator = ({ isTyping = false, message = "Thinking...", darkMode = false }) => {
  // Only render when actually typing - prevents stray elements from appearing
  if (!isTyping) {
    return null;
  }

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
