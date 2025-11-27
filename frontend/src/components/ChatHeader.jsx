import React from 'react';

const ChatHeader = ({ 
  darkMode, 
  onDarkModeToggle, 
  onClearHistory, 
  onToggleSessionsPanel
}) => {
  const handleNewSession = () => {
    // Clear local session and reload
    localStorage.removeItem('chat_session_id');
    localStorage.removeItem('chat-messages');
    window.location.reload();
  };

  return (
    // Floating buttons with labels - positioned lower, no background bar
    <div className="fixed top-20 right-4 z-50 flex items-center gap-2">
      {/* Independent floating buttons with hover labels */}
      
      {/* Sessions Button */}
      <div className="relative group">
        <button
          onClick={onToggleSessionsPanel}
          className={`w-11 h-11 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-105 ${
            darkMode 
              ? 'bg-gray-800 hover:bg-gray-700 text-white shadow-gray-900/50' 
              : 'bg-white hover:bg-gray-50 text-gray-700 shadow-gray-300/50'
          }`}
          aria-label="View chat sessions"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
          </svg>
        </button>
        {/* Tooltip */}
        <div className={`absolute right-0 top-full mt-2 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200 ${
          darkMode 
            ? 'bg-gray-900 text-white' 
            : 'bg-gray-800 text-white'
        }`}>
          Sessions
        </div>
      </div>

      {/* New Chat Button */}
      <div className="relative group">
        <button
          onClick={handleNewSession}
          className={`w-11 h-11 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-105 ${
            darkMode 
              ? 'bg-gray-800 hover:bg-gray-700 text-white shadow-gray-900/50' 
              : 'bg-white hover:bg-gray-50 text-gray-700 shadow-gray-300/50'
          }`}
          aria-label="Start new chat"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
        </button>
        {/* Tooltip */}
        <div className={`absolute right-0 top-full mt-2 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200 ${
          darkMode 
            ? 'bg-gray-900 text-white' 
            : 'bg-gray-800 text-white'
        }`}>
          New Chat
        </div>
      </div>

      {/* Clear History Button */}
      <div className="relative group">
        <button
          onClick={onClearHistory}
          className={`w-11 h-11 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-105 ${
            darkMode 
              ? 'bg-gray-800 hover:bg-gray-700 text-white shadow-gray-900/50' 
              : 'bg-white hover:bg-gray-50 text-gray-700 shadow-gray-300/50'
          }`}
          aria-label="Clear chat history"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
        {/* Tooltip */}
        <div className={`absolute right-0 top-full mt-2 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200 ${
          darkMode 
            ? 'bg-gray-900 text-white' 
            : 'bg-gray-800 text-white'
        }`}>
          Clear
        </div>
      </div>

      {/* Dark Mode Toggle */}
      <div className="relative group">
        <button
          onClick={onDarkModeToggle}
          className={`w-11 h-11 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-105 ${
            darkMode 
              ? 'bg-gray-800 hover:bg-gray-700 text-yellow-400 shadow-gray-900/50' 
              : 'bg-white hover:bg-gray-50 text-gray-700 shadow-gray-300/50'
          }`}
          aria-label={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
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
        {/* Tooltip */}
        <div className={`absolute right-0 top-full mt-2 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200 ${
          darkMode 
            ? 'bg-gray-900 text-white' 
            : 'bg-gray-800 text-white'
        }`}>
          {darkMode ? 'Light Mode' : 'Dark Mode'}
        </div>
      </div>
    </div>
  );
};

export default ChatHeader;
