import React from 'react';
import ReactMarkdown from 'react-markdown';
import './ChatMessage.css';

const ChatMessage = ({ message }) => {
  const { text, sender, timestamp, metadata, cached, confidence, responseTime, isError } = message;

  const formatTimestamp = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  if (sender === 'system' || isError) {
    return (
      <div className="message-wrapper system-message">
        <div className="message-content error">
          <span className="error-icon">⚠️</span>
          <span>{text}</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`message-wrapper ${sender}-message`}>
      <div className="message-avatar">
        {sender === 'user' ? '👤' : '🦙'}
      </div>
      
      <div className="message-content">
        <div className="message-header">
          <span className="message-sender">
            {sender === 'user' ? 'You' : 'AI Istanbul'}
          </span>
          <span className="message-timestamp">
            {formatTimestamp(timestamp)}
          </span>
        </div>
        
        <div className="message-text">
          {sender === 'ai' ? (
            <ReactMarkdown>{text}</ReactMarkdown>
          ) : (
            <p>{text}</p>
          )}
        </div>

        {sender === 'ai' && metadata && (
          <div className="message-metadata">
            {cached && (
              <span className="metadata-badge cached" title="Response from cache">
                ⚡ Cached
              </span>
            )}
            {confidence && (
              <span className="metadata-badge confidence" title={`Confidence: ${(confidence * 100).toFixed(0)}%`}>
                📊 {(confidence * 100).toFixed(0)}%
              </span>
            )}
            {responseTime && (
              <span className="metadata-badge response-time" title={`Response time: ${responseTime}ms`}>
                ⏱️ {(responseTime / 1000).toFixed(1)}s
              </span>
            )}
            {metadata.llm_model && (
              <span className="metadata-badge model" title={`Model: ${metadata.llm_model}`}>
                🤖 {metadata.llm_model}
              </span>
            )}
          </div>
        )}
      </div>

      {sender === 'ai' && (
        <div className="message-actions">
          <button 
            className="action-btn" 
            title="Copy message"
            onClick={() => navigator.clipboard.writeText(text)}
          >
            📋
          </button>
          <button 
            className="action-btn" 
            title="Share"
            onClick={() => {
              if (navigator.share) {
                navigator.share({ text });
              }
            }}
          >
            🔗
          </button>
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
