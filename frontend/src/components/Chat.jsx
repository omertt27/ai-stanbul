import React from 'react';

const Chat = ({ messages }) => (
  <div className="chat-container">
    {messages.map((msg, idx) => (
      <div key={idx} className="message-row">
        <div className={`message-bubble ${msg.user === 'You' ? 'user' : 'bot'}`}>
          <span>{msg.text}</span>
        </div>
      </div>
    ))}
  </div>
);

export default Chat;
