import React from 'react';

const TypingIndicator = ({ isTyping, message = "AI is thinking...", darkMode = true }) => {
  if (!isTyping) return null;

  return (
    <div className={`flex items-start space-x-3 p-4 ${
      darkMode ? 'bg-gray-800' : 'bg-gray-100'
    } rounded-lg mb-4 animate-pulse`}>
      {/* AI Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        darkMode ? 'bg-blue-600' : 'bg-blue-500'
      }`}>
        <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      </div>
      
      <div className="flex-1">
        <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'} mb-2`}>
          {message}
        </div>
        
        {/* Typing dots animation */}
        <div className="flex space-x-1">
          <div className={`w-2 h-2 rounded-full ${
            darkMode ? 'bg-gray-400' : 'bg-gray-500'
          } animate-bounce`} style={{ animationDelay: '0ms' }}></div>
          <div className={`w-2 h-2 rounded-full ${
            darkMode ? 'bg-gray-400' : 'bg-gray-500'
          } animate-bounce`} style={{ animationDelay: '150ms' }}></div>
          <div className={`w-2 h-2 rounded-full ${
            darkMode ? 'bg-gray-400' : 'bg-gray-500'
          } animate-bounce`} style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
