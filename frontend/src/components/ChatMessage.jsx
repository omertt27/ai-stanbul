import React from 'react';
import ReactMarkdown from 'react-markdown';
import RouteCard from './RouteCard';
import './ChatMessage.css';

const ChatMessage = ({ message }) => {
  const { 
    text, 
    sender, 
    timestamp, 
    metadata, 
    cached, 
    confidence, 
    responseTime, 
    isError,
    type,
    data,
    route_info,
    map_data,
    recommendations,
    weather
  } = message;

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
        
        {/* Route Card Visualization */}
        {sender === 'ai' && (route_info || map_data || (data && (data.route_info || data.map_data))) && (
          <div className="message-route-card mb-3">
            <RouteCard routeData={data || message} />
          </div>
        )}

        {/* Smart text rendering: If route card is shown, display condensed summary instead of full details */}
        <div className="message-text">
          {sender === 'ai' ? (
            <ReactMarkdown>
              {(route_info || map_data || (data && (data.route_info || data.map_data))) 
                ? extractRouteSummary(text) 
                : text}
            </ReactMarkdown>
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

/**
 * Extract a concise summary from route text instead of showing full details
 * The route card shows the details, so the chat text should be brief and contextual
 */
function extractRouteSummary(text) {
  if (!text) return '';
  
  // If text contains step-by-step details, extract just the opening summary
  const lines = text.split('\n');
  
  // Find where detailed steps begin (usually after emojis like 🚇, 🚶, numbered lists, or "Here are the steps")
  const detailsStartIndex = lines.findIndex(line => 
    /^\d+\./.test(line.trim()) || // Numbered lists
    /^[🚇🚶🚌🚋⛴️🔄]/.test(line.trim()) || // Transit emojis
    /here are the steps|step-by-step|directions:/i.test(line.toLowerCase())
  );
  
  if (detailsStartIndex > 0) {
    // Return only the summary portion (before the detailed steps)
    const summary = lines.slice(0, detailsStartIndex).join('\n').trim();
    return summary || lines[0]; // Fallback to first line if summary is empty
  }
  
  // If no detailed steps found, return first 2-3 lines as summary
  if (lines.length > 3) {
    return lines.slice(0, 3).join('\n');
  }
  
  return text;
}

export default ChatMessage;
