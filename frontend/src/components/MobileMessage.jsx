import React from 'react';

/**
 * Mobile-optimized Message Component
 * Provides enhanced message bubbles with better mobile UX
 */
const MobileMessage = ({ 
  message, 
  isUser, 
  isTyping = false, 
  onCopy, 
  onShare, 
  onLike, 
  onDislike,
  showActions = true 
}) => {
  const bubbleClass = isUser 
    ? "message-bubble user mobile-user-bubble"
    : "message-bubble bot mobile-bot-bubble";

  return (
    <div className={`message-row mobile-message-row ${isUser ? 'user-row' : 'bot-row'}`}>
      <div className={bubbleClass}>
        {isTyping ? (
          <div className="mobile-typing-content">
            <span>{message}</span>
            <div className="mobile-typing-cursor">|</div>
          </div>
        ) : (
          <div className="mobile-message-content">
            {message}
          </div>
        )}
      </div>
      
      {showActions && !isUser && !isTyping && (
        <div className="mobile-message-actions">
          <button 
            className="mobile-action-btn copy-btn"
            onClick={onCopy}
            aria-label="Copy message"
          >
            <svg viewBox="0 0 24 24" width="16" height="16">
              <path fill="currentColor" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
            </svg>
          </button>
          
          <button 
            className="mobile-action-btn share-btn"
            onClick={onShare}
            aria-label="Share message"
          >
            <svg viewBox="0 0 24 24" width="16" height="16">
              <path fill="currentColor" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z"/>
            </svg>
          </button>
          
          <button 
            className="mobile-action-btn like-btn"
            onClick={onLike}
            aria-label="Like message"
          >
            <svg viewBox="0 0 24 24" width="16" height="16">
              <path fill="currentColor" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
            </svg>
          </button>
        </div>
      )}
    </div>
  );
};

export default MobileMessage;
