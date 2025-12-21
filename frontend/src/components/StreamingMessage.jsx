/**
 * StreamingMessage Component
 * 
 * Displays a message with streaming text animation.
 * Shows a blinking cursor while streaming is in progress.
 */

import React, { memo } from 'react';
import ReactMarkdown from 'react-markdown';

/**
 * Blinking cursor component
 */
const BlinkingCursor = () => (
  <span className="inline-block w-2 h-5 ml-1 bg-current animate-pulse" 
        style={{ animation: 'blink 1s step-end infinite' }}>
    <style>{`
      @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
      }
    `}</style>
  </span>
);

/**
 * StreamingMessage - Shows text with real-time streaming effect
 * 
 * @param {Object} props
 * @param {string} props.text - The text content to display
 * @param {boolean} props.isStreaming - Whether text is currently streaming
 * @param {boolean} props.isBot - Whether this is a bot message
 * @param {boolean} props.showCursor - Whether to show blinking cursor
 * @param {boolean} props.enableMarkdown - Whether to render markdown
 * @param {string} props.className - Additional CSS classes
 */
const StreamingMessage = memo(({ 
  text = '',
  isStreaming = false,
  isBot = true,
  showCursor = true,
  enableMarkdown = true,
  className = ''
}) => {
  // Don't render if no text
  if (!text && !isStreaming) {
    return null;
  }

  const messageClasses = `
    streaming-message
    ${isBot ? 'bot-message' : 'user-message'}
    ${isStreaming ? 'streaming' : ''}
    ${className}
  `.trim();

  return (
    <div className={messageClasses}>
      <div className="message-content">
        {enableMarkdown && text ? (
          <ReactMarkdown
            components={{
              // Custom link rendering
              a: ({ node, ...props }) => (
                <a 
                  {...props} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:text-blue-600 underline"
                />
              ),
              // Custom paragraph to avoid extra spacing
              p: ({ node, ...props }) => (
                <p {...props} className="mb-2 last:mb-0" />
              ),
              // Custom list styling
              ul: ({ node, ...props }) => (
                <ul {...props} className="list-disc list-inside mb-2" />
              ),
              ol: ({ node, ...props }) => (
                <ol {...props} className="list-decimal list-inside mb-2" />
              ),
              // Code blocks
              code: ({ node, inline, ...props }) => (
                inline 
                  ? <code {...props} className="bg-gray-100 dark:bg-gray-800 px-1 rounded text-sm" />
                  : <code {...props} className="block bg-gray-100 dark:bg-gray-800 p-2 rounded text-sm overflow-x-auto" />
              )
            }}
          >
            {text}
          </ReactMarkdown>
        ) : (
          <span>{text}</span>
        )}
        
        {/* Blinking cursor while streaming */}
        {isStreaming && showCursor && <BlinkingCursor />}
      </div>
    </div>
  );
});

StreamingMessage.displayName = 'StreamingMessage';

/**
 * StreamingTypingIndicator - Shows "typing" animation before streaming starts
 */
export const StreamingTypingIndicator = memo(({ 
  text = 'KAM is thinking',
  className = '' 
}) => {
  return (
    <div className={`streaming-typing-indicator flex items-center gap-2 ${className}`}>
      <span className="text-gray-500 dark:text-gray-400">{text}</span>
      <div className="flex gap-1">
        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
    </div>
  );
});

StreamingTypingIndicator.displayName = 'StreamingTypingIndicator';

export default StreamingMessage;
