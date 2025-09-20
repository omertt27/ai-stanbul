import React, { useState, useEffect, useRef } from 'react';
import { WordByWordTyping, StreamingText } from './TypingAnimation';
import { ChatMessageSkeleton, TypingIndicator } from './LoadingSkeletons';

/**
 * Enhanced Chat Message Component
 * Supports typing animation and different message types
 */
const EnhancedChatMessage = ({ 
  message, 
  isBot = false, 
  enableTyping = true,
  onTypingComplete 
}) => {
  const [showTyping, setShowTyping] = useState(isBot && enableTyping && !message.typed);
  const [isTypingComplete, setIsTypingComplete] = useState(!enableTyping || message.typed);

  const handleTypingComplete = () => {
    setShowTyping(false);
    setIsTypingComplete(true);
    if (onTypingComplete) onTypingComplete(message.id);
  };

  if (!isTypingComplete && showTyping) {
    return (
      <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
        <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
          isBot 
            ? 'bg-gray-100 text-gray-800' 
            : 'bg-blue-500 text-white'
        }`}>
          <WordByWordTyping 
            text={message.content}
            onComplete={handleTypingComplete}
            speed={60}
            variation={30}
          />
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
      <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
        isBot 
          ? 'bg-gray-100 text-gray-800' 
          : 'bg-blue-500 text-white'
      }`}>
        {message.content}
      </div>
    </div>
  );
};

/**
 * Enhanced Chat Interface
 * Complete chat interface with typing animations and loading states
 */
const EnhancedChatInterface = ({ 
  messages = [], 
  onSendMessage, 
  isLoading = false,
  enableTypingAnimation = true,
  placeholder = "Ask me about Istanbul..." 
}) => {
  const [inputValue, setInputValue] = useState('');
  const [typingMessages, setTypingMessages] = useState(new Set());
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleTypingComplete = (messageId) => {
    setTypingMessages(prev => {
      const newSet = new Set(prev);
      newSet.delete(messageId);
      return newSet;
    });
  };

  const isMessageTyping = (messageId) => typingMessages.has(messageId);

  // Add new bot messages to typing queue
  useEffect(() => {
    const newBotMessages = messages
      .filter(msg => msg.type === 'bot' && !msg.typed)
      .map(msg => msg.id);
    
    if (newBotMessages.length > 0) {
      setTypingMessages(prev => new Set([...prev, ...newBotMessages]));
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full max-h-screen bg-white">
      {/* Chat Header */}
      <div className="bg-blue-500 text-white p-4 shadow-md">
        <h2 className="text-xl font-semibold">Istanbul AI Assistant</h2>
        <p className="text-blue-100 text-sm">Ask me anything about Istanbul!</p>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {messages.map((message) => (
          <EnhancedChatMessage
            key={message.id}
            message={message}
            isBot={message.type === 'bot'}
            enableTyping={enableTypingAnimation && !message.typed}
            onTypingComplete={handleTypingComplete}
          />
        ))}
        
        {/* Loading/Typing Indicator */}
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-100 rounded-lg">
              <TypingIndicator />
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-4 border-t bg-gray-50">
        <div className="flex space-x-3">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={placeholder}
            disabled={isLoading}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
};

/**
 * Smart Response Component
 * Handles different types of responses with appropriate animations
 */
const SmartResponse = ({ 
  response, 
  source = 'ai',
  onComplete 
}) => {
  const [displayText, setDisplayText] = useState('');
  const [isComplete, setIsComplete] = useState(false);

  const shouldAnimate = source === 'ai' && !isComplete;
  const speed = source === 'ai' ? 50 : 20; // Slower for AI, faster for fallback

  useEffect(() => {
    if (response && !isComplete) {
      if (source === 'cache' || source === 'fallback') {
        // Show cached/fallback responses immediately
        setDisplayText(response);
        setIsComplete(true);
        if (onComplete) onComplete();
      }
    }
  }, [response, source, isComplete, onComplete]);

  if (shouldAnimate) {
    return (
      <StreamingText
        text={response}
        speed={speed}
        onComplete={() => {
          setIsComplete(true);
          if (onComplete) onComplete();
        }}
        className="whitespace-pre-wrap"
      />
    );
  }

  return (
    <div className="whitespace-pre-wrap">
      {displayText || response}
    </div>
  );
};

/**
 * Response Type Indicator
 * Shows visual indicator for response source
 */
const ResponseTypeIndicator = ({ source }) => {
  const indicators = {
    ai: { icon: '🤖', label: 'AI Generated', color: 'bg-green-100 text-green-800' },
    cache: { icon: '⚡', label: 'Cached', color: 'bg-blue-100 text-blue-800' },
    fallback: { icon: '📚', label: 'Knowledge Base', color: 'bg-yellow-100 text-yellow-800' }
  };

  const indicator = indicators[source] || indicators.fallback;

  return (
    <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${indicator.color} mb-2`}>
      <span className="mr-1">{indicator.icon}</span>
      {indicator.label}
    </div>
  );
};

export { 
  EnhancedChatMessage, 
  EnhancedChatInterface, 
  SmartResponse, 
  ResponseTypeIndicator 
};
