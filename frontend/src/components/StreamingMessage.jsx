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
      <div className="message-content" style={{
        // Force proper paragraph spacing
        '& p': {
          marginBottom: '1rem !important',
          lineHeight: '1.6'
        },
        '& p:last-child': {
          marginBottom: '0 !important'
        },
        '& ul, & ol': {
          marginBottom: '1rem !important',
          marginTop: '0.5rem !important'
        },
        '& li': {
          marginBottom: '0.25rem',
          lineHeight: '1.5'
        }
      }}>
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
              // Custom paragraph with FORCED spacing for multiple paragraphs
              p: ({ node, ...props }) => (
                <p {...props} style={{ 
                  marginBottom: '1rem', 
                  lineHeight: '1.6',
                  display: 'block'
                }} />
              ),
              // Custom list styling with proper spacing
              ul: ({ node, ...props }) => (
                <ul {...props} style={{ 
                  listStyleType: 'disc',
                  paddingLeft: '1.5rem',
                  marginBottom: '1rem',
                  marginTop: '0.5rem'
                }} />
              ),
              ol: ({ node, ...props }) => (
                <ol {...props} style={{ 
                  listStyleType: 'decimal',
                  paddingLeft: '1.5rem',
                  marginBottom: '1rem',
                  marginTop: '0.5rem'
                }} />
              ),
              // List items with proper spacing
              li: ({ node, ...props }) => (
                <li {...props} style={{ 
                  marginBottom: '0.25rem',
                  lineHeight: '1.5'
                }} />
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
              // Headings with proper spacing
              h1: ({ node, ...props }) => (
                <h1 {...props} style={{ 
                  fontSize: '1.25rem',
                  fontWeight: 'bold',
                  marginBottom: '0.75rem',
                  marginTop: '1rem',
                  lineHeight: '1.4'
                }} />
              ),
              h2: ({ node, ...props }) => (
                <h2 {...props} style={{ 
                  fontSize: '1.125rem',
                  fontWeight: 'bold',
                  marginBottom: '0.75rem',
                  marginTop: '1rem',
                  lineHeight: '1.4'
                }} />
              ),
              h3: ({ node, ...props }) => (
                <h3 {...props} style={{ 
                  fontSize: '1rem',
                  fontWeight: 'bold',
                  marginBottom: '0.5rem',
                  marginTop: '0.75rem',
                  lineHeight: '1.4'
                }} />
              ),
              // Blockquotes
              blockquote: ({ node, ...props }) => (
                <blockquote {...props} style={{ 
                  borderLeft: '4px solid #d1d5db',
                  paddingLeft: '1rem',
                  fontStyle: 'italic',
                  marginBottom: '1rem',
                  marginTop: '0.5rem'
                }} />
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
