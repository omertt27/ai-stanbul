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
      <style jsx>{`
        .streaming-message .message-content p {
          margin-bottom: 1rem !important;
          line-height: 1.6;
        }
        .streaming-message .message-content p:last-child {
          margin-bottom: 0 !important;
        }
        .streaming-message .message-content ul,
        .streaming-message .message-content ol {
          margin-bottom: 1rem !important;
          margin-top: 0.5rem !important;
        }
        .streaming-message .message-content li {
          margin-bottom: 0.25rem;
          line-height: 1.5;
        }
        .streaming-message .message-content h1,
        .streaming-message .message-content h2,
        .streaming-message .message-content h3 {
          margin-top: 1.5rem !important;
          margin-bottom: 0.75rem !important;
        }
        .streaming-message .message-content h1:first-child,
        .streaming-message .message-content h2:first-child,
        .streaming-message .message-content h3:first-child {
          margin-top: 0 !important;
        }
        .streaming-message .message-content blockquote {
          margin: 1rem 0 !important;
        }
        .streaming-message .message-content pre,
        .streaming-message .message-content code {
          margin: 0.5rem 0 !important;
        }
      `}</style>
      <div className="message-content">
        {enableMarkdown && text ? (
          <ReactMarkdown
            className="prose prose-sm md:prose-base max-w-none"
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
              // Custom paragraph with proper spacing for multiple paragraphs
              p: ({ node, ...props }) => (
                <p {...props} className="mb-4 last:mb-0 whitespace-pre-wrap" style={{ marginBottom: '1rem' }} />
              ),
              // Custom list styling with proper spacing
              ul: ({ node, ...props }) => (
                <ul {...props} className="list-disc list-inside mb-4 space-y-1" />
              ),
              ol: ({ node, ...props }) => (
                <ol {...props} className="list-decimal list-inside mb-4 space-y-1" />
              ),
              // List items with proper spacing
              li: ({ node, ...props }) => (
                <li {...props} className="mb-1" />
              ),
              // Bold text
              strong: ({ node, ...props }) => (
                <strong {...props} className="font-semibold" />
              ),
              // Code blocks
              code: ({ node, inline, ...props }) => (
                inline 
                  ? <code {...props} className="bg-gray-100 dark:bg-gray-800 px-1 rounded text-sm" />
                  : <code {...props} className="block bg-gray-100 dark:bg-gray-800 p-2 rounded text-sm overflow-x-auto mb-4" />
              ),
              // Headings
              h1: ({ node, ...props }) => (
                <h1 {...props} className="text-xl font-bold mb-3 mt-4 first:mt-0" />
              ),
              h2: ({ node, ...props }) => (
                <h2 {...props} className="text-lg font-bold mb-3 mt-4 first:mt-0" />
              ),
              h3: ({ node, ...props }) => (
                <h3 {...props} className="text-base font-bold mb-2 mt-3 first:mt-0" />
              ),
              // Blockquotes
              blockquote: ({ node, ...props }) => (
                <blockquote {...props} className="border-l-4 border-gray-300 pl-4 italic mb-4" />
              )
            }}
          >
            {text}
          </ReactMarkdown>
        ) : (
          <span className="whitespace-pre-wrap">{text}</span>
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
