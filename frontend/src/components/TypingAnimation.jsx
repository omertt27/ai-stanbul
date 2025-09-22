import React, { useState, useEffect } from 'react';

/**
 * Enhanced Typing Indicator with custom animations
 */
const TypingIndicator = ({ variant = 'dots', className = "" }) => {
  switch (variant) {
    case 'enhanced':
      return (
        <div className={`typing-indicator-enhanced ${className}`}>
          <div className="typing-dots">
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
          </div>
          <span style={{ color: '#6366f1', fontSize: '0.9rem', marginLeft: '8px' }}>
            AI is thinking...
          </span>
        </div>
      );
    
    case 'thinking':
      return (
        <div className={`thinking-animation ${className}`}>
          <span className="thinking-brain">🧠</span>
          <span className="thinking-text">Processing your request...</span>
        </div>
      );
    
    case 'wave':
      return (
        <div className={`wave-loading ${className}`}>
          <div className="wave-bar"></div>
          <div className="wave-bar"></div>
          <div className="wave-bar"></div>
          <div className="wave-bar"></div>
          <div className="wave-bar"></div>
        </div>
      );
    
    case 'floating':
      return (
        <div className={`floating-dots ${className}`}>
          <div className="floating-dot"></div>
          <div className="floating-dot"></div>
          <div className="floating-dot"></div>
          <div className="floating-dot"></div>
        </div>
      );
    
    default: // 'dots'
      return (
        <div className={`typing-indicator-enhanced ${className}`}>
          <div className="typing-dots">
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
          </div>
        </div>
      );
  }
};

/**
 * TypingSimulator Component
 * Simulates realistic typing animation for AI responses
 */
const TypingSimulator = ({ 
  text, 
  onComplete, 
  speed = 50, 
  variation = 30, 
  className = "",
  enableEnhancedCursor = true
}) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    if (!text || currentIndex >= text.length) {
      setIsTyping(false);
      if (onComplete) onComplete();
      return;
    }

    const timer = setTimeout(() => {
      setDisplayText(prev => prev + text[currentIndex]);
      setCurrentIndex(prev => prev + 1);
    }, speed + Math.random() * variation);

    return () => clearTimeout(timer);
  }, [text, currentIndex, speed, variation, onComplete]);

  const cursorClass = enableEnhancedCursor ? "typewriter-enhanced" : "typing-cursor animate-pulse";

  return (
    <div className={`typing-container ${className}`}>
      <span>{displayText}</span>
      {isTyping && (
        <span className={cursorClass}>|</span>
      )}
    </div>
  );
};

/**
 * WordByWordTyping Component
 * Types word by word instead of character by character for better readability
 */
const WordByWordTyping = ({ 
  text, 
  onComplete, 
  speed = 80, 
  variation = 40, 
  className = "" 
}) => {
  const [displayText, setDisplayText] = useState('');
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [isTyping, setIsTyping] = useState(true);

  const words = text ? text.split(' ') : [];

  useEffect(() => {
    if (!text || currentWordIndex >= words.length) {
      setIsTyping(false);
      if (onComplete) onComplete();
      return;
    }

    const timer = setTimeout(() => {
      setDisplayText(prev => 
        prev + (prev ? ' ' : '') + words[currentWordIndex]
      );
      setCurrentWordIndex(prev => prev + 1);
    }, speed + Math.random() * variation);

    return () => clearTimeout(timer);
  }, [text, currentWordIndex, speed, variation, onComplete, words]);

  return (
    <div className={`typing-container ${className}`}>
      <span>{displayText}</span>
      {isTyping && (
        <span className="typing-cursor animate-pulse ml-1">|</span>
      )}
    </div>
  );
};

/**
 * StreamingText Component
 * Simulates real-time streaming responses like ChatGPT
 */
const StreamingText = ({ 
  text, 
  onChunk, 
  onComplete, 
  speed = 30,
  className = "",
  enableStreamingGlow = true
}) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (!text || currentIndex >= text.length) {
      if (onComplete) onComplete();
      return;
    }

    const timer = setTimeout(() => {
      const nextChar = text[currentIndex];
      setDisplayText(prev => prev + nextChar);
      setCurrentIndex(prev => prev + 1);
      
      if (onChunk) onChunk(nextChar);
    }, speed + Math.random() * 20);

    return () => clearTimeout(timer);
  }, [text, currentIndex, speed, onChunk, onComplete]);

  const containerClass = enableStreamingGlow 
    ? `streaming-text message-streaming ${className}` 
    : `streaming-text ${className}`;

  return (
    <div className={containerClass}>
      {displayText}
      <span className="streaming-cursor animate-pulse">▎</span>
    </div>
  );
};

/**
 * LoadingSpinner Component with multiple variants
 */
const LoadingSpinner = ({ variant = 'spinner', size = 'medium', className = "" }) => {
  const sizeClass = size === 'large' ? 'spinner-large' : '';
  
  switch (variant) {
    case 'ripple':
      return (
        <div className={`ripple-loading ${className}`}>
          <div className="ripple-circle"></div>
          <div className="ripple-circle"></div>
        </div>
      );
    
    case 'bounce':
      return (
        <div className={`bounce-loading ${className}`}>
          <div className="bounce-ball"></div>
          <div className="bounce-ball"></div>
          <div className="bounce-ball"></div>
        </div>
      );
    
    default: // 'spinner'
      return (
        <div className={`spinner-loading ${sizeClass} ${className}`}></div>
      );
  }
};

/**
 * ConnectionStatus Component
 */
const ConnectionStatus = ({ isConnected = true, className = "" }) => {
  const statusClass = isConnected ? 'connection-indicator' : 'connection-indicator disconnected';
  const statusText = isConnected ? 'Connected' : 'Disconnected';
  
  return (
    <div className={`${statusClass} ${className}`}>
      <div className="connection-dot"></div>
      <span className="connection-text">{statusText}</span>
    </div>
  );
};

export { 
  TypingSimulator, 
  WordByWordTyping, 
  StreamingText, 
  TypingIndicator,
  LoadingSpinner,
  ConnectionStatus
};

// Default export for backward compatibility
export default TypingSimulator;
