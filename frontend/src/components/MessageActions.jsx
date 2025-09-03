import React, { useState } from 'react';

const MessageActions = ({ message, onCopy, onShare, onRetry, darkMode = true }) => {
  const [showActions, setShowActions] = useState(false);
  const [copied, setCopied] = useState(false);
  const [shared, setShared] = useState(false);

  const handleCopy = async () => {
    try {
      await onCopy(message);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy message:', error);
    }
  };

  const handleShare = async () => {
    try {
      await onShare(message);
      setShared(true);
      setTimeout(() => setShared(false), 2000);
    } catch (error) {
      console.error('Failed to share message:', error);
      // Fallback to copy if share fails
      handleCopy();
    }
  };

  const handleRetry = () => {
    if (onRetry) {
      setShowActions(false);
      onRetry(message);
    }
  };

  const isAssistant = message.sender === 'assistant' || message.role === 'assistant';
  const canRetry = message.type === 'error' && message.canRetry && message.originalInput;
  const messageText = message.text || message.content || '';

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      if (showActions && !event.target.closest('.message-actions')) {
        setShowActions(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [showActions]);

  return (
    <div className="relative message-actions">
      {/* Actions trigger button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          setShowActions(!showActions);
        }}
        className={`opacity-0 group-hover:opacity-100 transition-all duration-200 p-1.5 rounded-full hover:scale-110 ${
          darkMode 
            ? 'hover:bg-gray-600 text-gray-400 hover:text-gray-200' 
            : 'hover:bg-gray-200 text-gray-500 hover:text-gray-700'
        }`}
        title="Message actions"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
        </svg>
      </button>

      {/* Enhanced actions dropdown */}
      {showActions && (
        <div className={`absolute right-0 top-8 z-50 rounded-lg shadow-xl border min-w-40 ${
          darkMode 
            ? 'bg-gray-700 border-gray-600' 
            : 'bg-white border-gray-200'
        }`}>
          <div className="py-1">
            {/* Copy button with enhanced feedback */}
            <button
              onClick={handleCopy}
              className={`w-full text-left px-3 py-2.5 text-sm flex items-center space-x-3 transition-colors ${
                darkMode 
                  ? 'hover:bg-gray-600 text-gray-200' 
                  : 'hover:bg-gray-100 text-gray-700'
              } ${copied ? (darkMode ? 'bg-green-800' : 'bg-green-50') : ''}`}
            >
              <svg className={`w-4 h-4 ${copied ? 'text-green-500' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {copied ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                )}
              </svg>
              <span className={copied ? 'text-green-500 font-medium' : ''}>{copied ? 'Copied!' : 'Copy text'}</span>
            </button>

            {/* Share button - enhanced for assistant messages */}
            {isAssistant && messageText.length > 0 && (
              <button
                onClick={handleShare}
                className={`w-full text-left px-3 py-2.5 text-sm flex items-center space-x-3 transition-colors ${
                  darkMode 
                    ? 'hover:bg-gray-600 text-gray-200' 
                    : 'hover:bg-gray-100 text-gray-700'
                } ${shared ? (darkMode ? 'bg-blue-800' : 'bg-blue-50') : ''}`}
              >
                <svg className={`w-4 h-4 ${shared ? 'text-blue-500' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  {shared ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
                  )}
                </svg>
                <span className={shared ? 'text-blue-500 font-medium' : ''}>{shared ? 'Shared!' : 'Share response'}</span>
              </button>
            )}

            {/* Copy as markdown button for assistant messages */}
            {isAssistant && messageText.length > 0 && (
              <button
                onClick={async () => {
                  try {
                    const markdownText = `**Istanbul Travel Guide - KAM Assistant**\n\n${messageText}\n\n---\n*Generated by AI Istanbul Travel Guide*`;
                    await navigator.clipboard.writeText(markdownText);
                    setCopied(true);
                    setTimeout(() => setCopied(false), 2000);
                  } catch (error) {
                    console.error('Failed to copy markdown:', error);
                  }
                }}
                className={`w-full text-left px-3 py-2.5 text-sm flex items-center space-x-3 transition-colors ${
                  darkMode 
                    ? 'hover:bg-gray-600 text-gray-200' 
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>Copy as markdown</span>
              </button>
            )}

            {/* Retry button - only for error messages */}
            {canRetry && (
              <button
                onClick={handleRetry}
                className={`w-full text-left px-3 py-2.5 text-sm flex items-center space-x-3 transition-colors ${
                  darkMode 
                    ? 'hover:bg-gray-600 text-gray-200' 
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span>Retry message</span>
              </button>
            )}

            {/* Message info section - enhanced */}
            {message.type && (
              <div className={`px-3 py-2 text-xs border-t ${
                darkMode 
                  ? 'border-gray-600 text-gray-400' 
                  : 'border-gray-200 text-gray-500'
              }`}>
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span>Type:</span>
                    <span className="font-mono">{message.type}</span>
                  </div>
                  {message.resultCount && (
                    <div className="flex justify-between">
                      <span>Results:</span>
                      <span className="font-mono">{message.resultCount}</span>
                    </div>
                  )}
                  {message.dataSource && (
                    <div className="flex justify-between">
                      <span>Source:</span>
                      <span className="font-mono">{message.dataSource}</span>
                    </div>
                  )}
                  {message.timestamp && (
                    <div className="flex justify-between">
                      <span>Time:</span>
                      <span className="font-mono">{new Date(message.timestamp).toLocaleTimeString()}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageActions;
