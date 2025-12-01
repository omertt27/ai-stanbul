/**
 * TypingIndicator Component
 * =========================
 * ChatGPT-style typing indicator with three pulsing dots
 * Shows when the AI is generating a response
 */

import React from 'react';
import './TypingIndicator.css';

const TypingIndicator = ({ darkMode = false }) => {
  return (
    <div className={`typing-indicator-container ${darkMode ? 'dark' : 'light'}`}>
      <div className="typing-indicator">
        <span className="typing-dot"></span>
        <span className="typing-dot"></span>
        <span className="typing-dot"></span>
      </div>
    </div>
  );
};

export default TypingIndicator;
