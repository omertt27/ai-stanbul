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
 * Preprocess text to fix common formatting issues and improve markdown rendering
 */
const preprocessTextForMarkdown = (text) => {
  if (!text) return text;
  
  let processed = text;
  
  // Convert numbered lists to proper markdown with better spacing
  // Pattern: "1. Restaurant Name - Description"
  processed = processed.replace(/(\d+\.)\s+([^-\n]+)\s+-\s+([^\d]*?)(?=\s+\d+\.|📍|$)/g, (match, number, name, description) => {
    return `${number} **${name.trim()}** - ${description.trim()}\n\n`;
  });
  
  // Handle simple numbered items without restaurant pattern
  processed = processed.replace(/^(\d+)\.\s+(.+)/gm, '\n$1. $2');
  
  // Add line breaks before location markers and make them bold
  processed = processed.replace(/📍\s*Location:\s*(.+)/g, '\n**📍 Location:** $1');
  
  // Add line breaks before numbered items if they don't have them
  processed = processed.replace(/(\S)\s+(\d+\.)\s+/g, '$1\n\n$2 ');
  
  // Ensure proper paragraph breaks for sentences
  processed = processed.replace(/(\.\s+)([A-Z][^.]*\.\s*)/g, '$1\n\n$2');
  
  // Clean up multiple line breaks but preserve intentional spacing
  processed = processed.replace(/\n{3,}/g, '\n\n');
  
  return processed.trim();
};

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

  // Preprocess the text to fix formatting issues
  const processedText = preprocessTextForMarkdown(text);

  return (
    <div className={messageClasses}>
      <style>
        {`
          .streaming-message-content {
            white-space: pre-wrap;
          }
          .streaming-message-content p {
            margin-bottom: 1rem !important;
            line-height: 1.6 !important;
            display: block !important;
          }
          .streaming-message-content p:last-child {
            margin-bottom: 0 !important;
          }
          .streaming-message-content ul,
          .streaming-message-content ol {
            margin-bottom: 1rem !important;
            margin-top: 0.5rem !important;
            padding-left: 1.5rem !important;
          }
          .streaming-message-content li {
            margin-bottom: 0.25rem !important;
            line-height: 1.5 !important;
          }
          .streaming-message-content h1,
          .streaming-message-content h2,
          .streaming-message-content h3 {
            margin-top: 1rem !important;
            margin-bottom: 0.75rem !important;
            font-weight: bold !important;
          }
          .streaming-message-content blockquote {
            margin: 1rem 0 !important;
            padding-left: 1rem !important;
            border-left: 4px solid #d1d5db !important;
          }
          /* Force line breaks for numbered items */
          .streaming-message-content {
            white-space: pre-wrap !important;
            line-height: 1.6 !important;
          }
        `}
      </style>
      <div className="message-content streaming-message-content">
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
              // Rely on CSS for paragraph spacing
              p: ({ node, ...props }) => <p {...props} />,
              // Rely on CSS for list spacing  
              ul: ({ node, ...props }) => <ul {...props} style={{ listStyleType: 'disc' }} />,
              ol: ({ node, ...props }) => <ol {...props} style={{ listStyleType: 'decimal' }} />,
              li: ({ node, ...props }) => <li {...props} />,
              // Bold text
              strong: ({ node, ...props }) => (
                <strong {...props} className="font-semibold" />
              ),
              // Code blocks
              code: ({ node, inline, ...props }) => (
                inline 
                  ? <code {...props} className="bg-gray-100 dark:bg-gray-800 px-1 rounded text-sm" />
                  : <code {...props} className="block bg-gray-100 dark:bg-gray-800 p-2 rounded text-sm overflow-x-auto" />
              ),
              // Headings - rely mostly on CSS
              h1: ({ node, ...props }) => <h1 {...props} style={{ fontSize: '1.25rem' }} />,
              h2: ({ node, ...props }) => <h2 {...props} style={{ fontSize: '1.125rem' }} />,
              h3: ({ node, ...props }) => <h3 {...props} style={{ fontSize: '1rem' }} />,
              // Blockquotes - rely on CSS
              blockquote: ({ node, ...props }) => <blockquote {...props} />
            }}
          >
            {processedText}
          </ReactMarkdown>
        ) : (
          <span className="whitespace-pre-wrap">{processedText || text}</span>
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
