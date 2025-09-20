import React, { useState, useEffect } from 'react';

/**
 * TypingSimulator Component
 * Simulates realistic typing animation for AI responses
 */
const TypingSimulator = ({ 
  text, 
  onComplete, 
  speed = 50, 
  variation = 30, 
  className = "" 
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

  return (
    <div className={`typing-container ${className}`}>
      <span>{displayText}</span>
      {isTyping && (
        <span className="typing-cursor animate-pulse">|</span>
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
  className = "" 
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

  return (
    <div className={`streaming-text ${className}`}>
      {displayText}
      <span className="streaming-cursor animate-pulse">▎</span>
    </div>
  );
};

export { TypingSimulator, WordByWordTyping, StreamingText };

// Default export for backward compatibility
export default TypingSimulator;
