import React, { useState } from 'react';

const ChatHeader = ({ 
  darkMode, 
  onDarkModeToggle, 
  onClearHistory, 
  messageCount = 0, 
  isOnline = true, 
  apiHealth = { healthy: true } 
}) => {
  const [showMenu, setShowMenu] = useState(false);

  const handleClearHistory = () => {
    setShowMenu(false);
    onClearHistory();
  };

  return (
    <div className={`flex items-center justify-between px-4 py-3 border-b transition-colors duration-200 ${
      darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-white'
    }`}>
      {/* Left side - Logo and title */}
      <div className="flex items-center space-x-3">
        <div className={`w-8 h-8 rounded-sm flex items-center justify-center transition-colors duration-200 ${
          darkMode ? 'bg-white' : 'bg-black'
        }`}>
          <svg className={`w-5 h-5 transition-colors duration-200 ${
            darkMode ? 'text-black' : 'text-white'
          }`} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
          </svg>
        </div>
        <div>
          <h1 className={`text-lg font-semibold transition-colors duration-200 ${
            darkMode ? 'text-white' : 'text-black'
          }`}>
            Istanbul Travel Guide
          </h1>
          {messageCount > 0 && (
            <p className={`text-xs transition-colors duration-200 ${
              darkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
              {messageCount} messages
            </p>
          )}
        </div>
      </div>

      {/* Right side - Status and controls */}
      <div className="flex items-center space-x-3">
        {/* Network status indicator */}
        <div className={`flex items-center space-x-1 text-xs px-2 py-1 rounded-full ${
          isOnline && apiHealth.healthy
            ? darkMode ? 'bg-green-900 text-green-300' : 'bg-green-100 text-green-700'
            : darkMode ? 'bg-red-900 text-red-300' : 'bg-red-100 text-red-700'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            isOnline && apiHealth.healthy ? 'bg-green-500' : 'bg-red-500'
          }`}></div>
          <span>
            {isOnline 
              ? (apiHealth.healthy ? 'Online' : 'Service Issues') 
              : 'Offline'
            }
          </span>
        </div>

        {/* Dark mode toggle */}
        <button
          onClick={onDarkModeToggle}
          className={`p-2 rounded-full transition-colors duration-200 ${
            darkMode 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-200 text-gray-600'
          }`}
          title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
        >
          {darkMode ? (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
            </svg>
          )}
        </button>

        {/* Menu button */}
        <div className="relative">
          <button
            onClick={() => setShowMenu(!showMenu)}
            className={`p-2 rounded-full transition-colors duration-200 ${
              darkMode 
                ? 'hover:bg-gray-700 text-gray-300' 
                : 'hover:bg-gray-200 text-gray-600'
            }`}
            title="Chat options"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
            </svg>
          </button>

          {/* Menu dropdown */}
          {showMenu && (
            <div className={`absolute right-0 top-10 z-50 rounded-lg shadow-lg border min-w-48 ${
              darkMode 
                ? 'bg-gray-700 border-gray-600' 
                : 'bg-white border-gray-200'
            }`}>
              <div className="py-1">
                <button
                  onClick={handleClearHistory}
                  className={`w-full text-left px-4 py-2 text-sm flex items-center space-x-2 ${
                    darkMode 
                      ? 'hover:bg-gray-600 text-gray-200' 
                      : 'hover:bg-gray-100 text-gray-700'
                  } ${messageCount === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={messageCount === 0}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  <span>Clear Chat History</span>
                </button>

                <div className={`px-4 py-2 text-xs border-t ${
                  darkMode 
                    ? 'border-gray-600 text-gray-400' 
                    : 'border-gray-200 text-gray-500'
                }`}>
                  <div>Messages stored locally</div>
                  <div>Auto-saves conversation</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatHeader;
