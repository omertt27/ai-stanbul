import React, { useState } from 'react';

const MessageActions = ({ message, onCopy, onShare, onRetry, darkMode = true }) => {
  const [showActions, setShowActions] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await onCopy(message);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleShare = () => {
    onShare(message);
  };

  const handleRetry = () => {
    if (onRetry) {
      onRetry(message);
    }
  };

  const isAssistant = message.sender === 'assistant';
  const canRetry = message.type === 'error' && message.canRetry && message.originalInput;

  return (
    <div className="relative">
      {/* Actions trigger button */}
      <button
        onClick={() => setShowActions(!showActions)}
        className={`opacity-0 group-hover:opacity-100 transition-opacity duration-200 p-1 rounded-full ${
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

      {/* Actions dropdown */}
      {showActions && (
        <div className={`absolute right-0 top-8 z-50 rounded-lg shadow-lg border min-w-32 ${
          darkMode 
            ? 'bg-gray-700 border-gray-600' 
            : 'bg-white border-gray-200'
        }`}>
          <div className="py-1">
            {/* Copy button */}
            <button
              onClick={handleCopy}
              className={`w-full text-left px-3 py-2 text-sm flex items-center space-x-2 ${
                darkMode 
                  ? 'hover:bg-gray-600 text-gray-200' 
                  : 'hover:bg-gray-100 text-gray-700'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <span>{copied ? 'Copied!' : 'Copy'}</span>
            </button>

            {/* Share button - only for assistant messages */}
            {isAssistant && (
              <button
                onClick={handleShare}
                className={`w-full text-left px-3 py-2 text-sm flex items-center space-x-2 ${
                  darkMode 
                    ? 'hover:bg-gray-600 text-gray-200' 
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
                </svg>
                <span>Share</span>
              </button>
            )}

            {/* Retry button - only for error messages */}
            {canRetry && (
              <button
                onClick={handleRetry}
                className={`w-full text-left px-3 py-2 text-sm flex items-center space-x-2 ${
                  darkMode 
                    ? 'hover:bg-gray-600 text-gray-200' 
                    : 'hover:bg-gray-100 text-gray-700'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span>Retry</span>
              </button>
            )}

            {/* Message info for debugging */}
            {message.type && (
              <div className={`px-3 py-2 text-xs border-t ${
                darkMode 
                  ? 'border-gray-600 text-gray-400' 
                  : 'border-gray-200 text-gray-500'
              }`}>
                Type: {message.type}
                {message.resultCount && <div>Results: {message.resultCount}</div>}
                {message.dataSource && <div>Source: {message.dataSource}</div>}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageActions;
