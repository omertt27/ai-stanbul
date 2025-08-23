import React from 'react';

const Chat = ({ messages }) => (
  <div className="chat-container p-4 bg-gray-100 rounded h-64 overflow-y-auto">
    {messages.map((msg, idx) => (
      <div key={idx} className="mb-2">
        <span className="font-semibold">{msg.user}: </span>
        <span>{msg.text}</span>
      </div>
    ))}
  </div>
);

export default Chat;
