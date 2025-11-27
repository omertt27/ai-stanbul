import React, { useState } from 'react';

const ChatHeader = ({ 
  darkMode, 
  onDarkModeToggle, 
  onClearHistory, 
  messageCount = 0, 
  isOnline = true, 
  apiHealth = 'healthy',
  sessionId = '',
  isHistoryLoading = false,
  onToggleSessionsPanel
}) => {
  const [showMenu, setShowMenu] = useState(false);
  const [clearingHistory, setClearingHistory] = useState(false);

  const handleClearHistory = async () => {
    setClearingHistory(true);
    setShowMenu(false);
    try {
      await onClearHistory();
    } catch (error) {
      console.error('Failed to clear history:', error);
    } finally {
      setClearingHistory(false);
    }
  };

  const handleNewSession = () => {
    setShowMenu(false);
    // Clear local session and reload
    localStorage.removeItem('chat_session_id');
    localStorage.removeItem('chat-messages');
    window.location.reload();
  };

  const copySessionId = async () => {
    try {
      await navigator.clipboard.writeText(sessionId);
      // Could show a toast here
    } catch (error) {
      console.error('Failed to copy session ID:', error);
    }
  };

  // Determine overall health status
  const isHealthy = isOnline && apiHealth === 'healthy';

  return (
    <div className={`fixed top-0 left-0 right-0 h-[60px] z-50 flex items-center justify-between px-4 border-b transition-colors duration-200 ${
      darkMode 
        ? 'bg-gray-900/80 border-gray-700 backdrop-blur-md backdrop-saturate-150' 
        : 'bg-white/80 border-gray-200 backdrop-blur-md backdrop-saturate-150'
    }`}>
      {/* Left side - Control buttons (ChatGPT style) */}
      <div className="flex items-center space-x-2">
        {/* Chat Sessions Button */}
        <button
          onClick={onToggleSessionsPanel}
          className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200 touch-manipulation ${
            darkMode 
              ? 'hover:bg-gray-700 text-gray-300' 
              : 'hover:bg-gray-200 text-gray-600'
          }`}
          title="View and manage chat sessions"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
          </svg>
        </button>

        {/* Menu button with dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowMenu(!showMenu)}
            disabled={clearingHistory}
            className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200 touch-manipulation ${
              darkMode 
                ? 'hover:bg-gray-700 text-gray-300' 
                : 'hover:bg-gray-200 text-gray-600'
            } ${clearingHistory ? 'opacity-50 cursor-not-allowed' : ''}`}
            title="Chat options"
          >
            {clearingHistory ? (
              <div className="w-5 h-5 border border-gray-400 border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
              </svg>
            )}
          </button>

          {/* Menu dropdown */}
          {showMenu && (
            <div className={`absolute left-0 top-12 z-50 rounded-lg shadow-xl border min-w-64 ${
              darkMode 
                ? 'bg-gray-800 border-gray-700' 
                : 'bg-white border-gray-200'
            }`}>
              <div className="py-1">
                {/* New session button */}
                <button
                  onClick={handleNewSession}
                  className={`w-full text-left px-4 py-2.5 text-sm flex items-center space-x-3 ${
                    darkMode 
                      ? 'hover:bg-gray-700 text-gray-200' 
                      : 'hover:bg-gray-100 text-gray-700'
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                  </svg>
                  <span>New chat</span>
                </button>
                
                {/* Clear history button */}
                <button
                  onClick={handleClearHistory}
                  className={`w-full text-left px-4 py-2.5 text-sm flex items-center space-x-3 ${
                    darkMode 
                      ? 'hover:bg-gray-700 text-gray-200' 
                      : 'hover:bg-gray-100 text-gray-700'
                  } ${messageCount === 0 ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={messageCount === 0 || clearingHistory}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  <span>Clear chat</span>
                </button>

                {/* Session info section */}
                {sessionId && (
                  <>
                    <div className={`border-t my-1 ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}></div>
                    <div className={`px-4 py-2 text-xs ${
                      darkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      <div className="mb-1 font-medium">Session ID:</div>
                      <button
                        onClick={copySessionId}
                        className={`font-mono text-xs px-2 py-1 rounded w-full text-left ${
                          darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
                        }`}
                        title="Click to copy"
                      >
                        {sessionId.slice(-12)}...
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Center - Title */}
      <div className="absolute left-1/2 transform -translate-x-1/2">
        <h1 className={`text-base font-semibold transition-colors duration-200 whitespace-nowrap ${
          darkMode ? 'text-white' : 'text-gray-900'
        }`}>
          KAM Assistant
        </h1>
      </div>

      {/* Right side - Status and dark mode */}
      <div className="flex items-center space-x-2">
        {/* Network status indicator */}
        <div className={`hidden md:flex items-center space-x-1.5 text-xs px-2.5 py-1.5 rounded-full transition-all ${
          isHealthy
            ? darkMode ? 'bg-green-900/30 text-green-400' : 'bg-green-100 text-green-700'
            : darkMode ? 'bg-red-900/30 text-red-400' : 'bg-red-100 text-red-700'
        }`}>
          <div className={`w-1.5 h-1.5 rounded-full ${
            isHealthy ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          }`}></div>
          <span className="font-medium">
            {!isOnline 
              ? 'Offline' 
              : apiHealth === 'healthy' ? 'Online' : 
                apiHealth === 'unhealthy' ? 'Issues' : 'Checking...'}
          </span>
        </div>

        {/* Dark mode toggle */}
        <button
          onClick={onDarkModeToggle}
          className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200 touch-manipulation ${
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
      </div>
    </div>
  );
};

export default ChatHeader;
