import React from 'react';
import './ChatInput.css';

const ChatInput = ({ value, onChange, onSend, disabled, placeholder }) => {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="chat-input-container">
      <div className="chat-input-wrapper">
        <textarea
          className="chat-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={placeholder || "Type your message..."}
          disabled={disabled}
          rows="1"
          maxLength="1000"
        />
        <button
          className="send-button"
          onClick={onSend}
          disabled={disabled || !value.trim()}
          title="Send message"
        >
          {disabled ? '⏳' : '📤'}
        </button>
      </div>
      <div className="input-hint">
        Press Enter to send, Shift+Enter for new line
      </div>
    </div>
  );
};

export default ChatInput;
