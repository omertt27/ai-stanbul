import React, { useState, useEffect } from 'react';

const TypingIndicator = ({ isTyping, message = "AI is thinking...", darkMode = true, duration = 0 }) => {
  const [showProgress, setShowProgress] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (isTyping && duration > 0) {
      setShowProgress(true);
      const startTime = Date.now();
      const interval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progressPercent = Math.min((elapsed / duration) * 100, 95); // Cap at 95%
        setProgress(progressPercent);
      }, 100);

      return () => clearInterval(interval);
    } else {
      setShowProgress(false);
      setProgress(0);
    }
  }, [isTyping, duration]);

  if (!isTyping) return null;

  // Enhanced message variations based on context
  const getContextualMessage = (msg) => {
    if (msg.includes('restaurant')) return { text: msg, icon: '🍽️' };
    if (msg.includes('places') || msg.includes('attractions')) return { text: msg, icon: '🏛️' };
    if (msg.includes('thinking')) return { text: msg, icon: '🤔' };
    if (msg.includes('searching')) return { text: msg, icon: '🔍' };
    return { text: msg, icon: '💭' };
  };

  const { text, icon } = getContextualMessage(message);

  return (
    <div className="py-4">
      <div className="group">
        <div className="flex items-start space-x-3">
          {/* Enhanced AI Avatar with pulsing animation */}
          <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-200 animate-pulse ${
            darkMode 
              ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
              : 'bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600'
          }`}>
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
            </svg>
          </div>
          
          <div className="flex-1">
            <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>KAM Assistant</div>
            
            {/* Enhanced typing message with icon */}
            <div className={`text-sm mb-3 flex items-center space-x-2 transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>
              <span className="text-base">{icon}</span>
              <span>{text}</span>
            </div>
            
            {/* Enhanced typing dots with different animation */}
            <div className="flex items-center space-x-2">
              <div className="flex space-x-1">
                <div className={`w-2 h-2 rounded-full animate-pulse ${
                  darkMode ? 'bg-blue-400' : 'bg-blue-500'
                }`} style={{ 
                  animation: 'typing-dot 1.4s infinite ease-in-out',
                  animationDelay: '0ms' 
                }}></div>
                <div className={`w-2 h-2 rounded-full animate-pulse ${
                  darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                }`} style={{ 
                  animation: 'typing-dot 1.4s infinite ease-in-out',
                  animationDelay: '200ms' 
                }}></div>
                <div className={`w-2 h-2 rounded-full animate-pulse ${
                  darkMode ? 'bg-purple-400' : 'bg-purple-500'
                }`} style={{ 
                  animation: 'typing-dot 1.4s infinite ease-in-out',
                  animationDelay: '400ms' 
                }}></div>
              </div>
              
              {/* Progress indicator for longer operations */}
              {showProgress && (
                <div className="flex-1 ml-4">
                  <div className={`w-full bg-gray-300 rounded-full h-1 ${
                    darkMode ? 'bg-gray-600' : 'bg-gray-300'
                  }`}>
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-1 rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                  <div className={`text-xs mt-1 ${
                    darkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}>
                    {Math.round(progress)}%
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Custom CSS for typing animation */}
      <style jsx>{`
        @keyframes typing-dot {
          0%, 20% {
            opacity: 0.3;
            transform: scale(0.8);
          }
          50% {
            opacity: 1;
            transform: scale(1.2);
          }
          100% {
            opacity: 0.3;
            transform: scale(0.8);
          }
        }
      `}</style>
    </div>
  );
};

export default TypingIndicator;
