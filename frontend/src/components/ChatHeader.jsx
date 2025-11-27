import React, { useState } from 'react';

const ChatHeader = ({ 
  darkMode, 
  onDarkModeToggle, 
  onClearHistory, 
  onToggleSessionsPanel
}) => {
  const [menuOpen, setMenuOpen] = useState(false);

  const handleNewSession = () => {
    // Clear local session and reload
    localStorage.removeItem('chat_session_id');
    localStorage.removeItem('chat-messages');
    window.location.reload();
  };

  return (
    <>
      {/* Mobile-First Header - Gemini Style */}
      <div className={`fixed top-0 left-0 right-0 z-50 border-b transition-colors duration-200 ${
        darkMode 
          ? 'bg-gray-900 border-gray-800' 
          : 'bg-white border-gray-200'
      }`}>
        <div className="flex items-center justify-between px-3 py-2 md:px-4 md:py-3">
          {/* Left: Menu Button (Mobile) + Title */}
          <div className="flex items-center gap-2 md:gap-3">
            {/* Hamburger Menu - Mobile Only */}
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className={`md:hidden w-10 h-10 rounded-full flex items-center justify-center transition-colors ${
                darkMode 
                  ? 'hover:bg-gray-800 text-white' 
                  : 'hover:bg-gray-100 text-gray-700'
              }`}
              aria-label="Menu"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            {/* Title - Hidden on very small screens */}
            <div className="flex items-center gap-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                darkMode ? 'bg-white' : 'bg-gradient-to-br from-blue-600 to-purple-600'
              }`}>
                <svg className={`w-4 h-4 ${
                  darkMode ? 'text-black' : 'text-white'
                }`} fill="currentColor" viewBox="0 0 24 24">
                  <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                </svg>
              </div>
              <span className={`hidden sm:block font-semibold text-lg ${
                darkMode ? 'text-white' : 'text-gray-900'
              }`}>
                KAM
              </span>
            </div>
          </div>

          {/* Right: Action Buttons */}
          <div className="flex items-center gap-1 md:gap-2">
            {/* Sessions Button - Desktop */}
            <button
              onClick={onToggleSessionsPanel}
              className={`hidden md:flex w-10 h-10 rounded-full items-center justify-center transition-colors ${
                darkMode 
                  ? 'hover:bg-gray-800 text-white' 
                  : 'hover:bg-gray-100 text-gray-700'
              }`}
              aria-label="View chat sessions"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
              </svg>
            </button>

            {/* New Chat Button */}
            <button
              onClick={handleNewSession}
              className={`w-10 h-10 rounded-full flex items-center justify-center transition-colors ${
                darkMode 
                  ? 'hover:bg-gray-800 text-white' 
                  : 'hover:bg-gray-100 text-gray-700'
              }`}
              aria-label="Start new chat"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>

            {/* Dark Mode Toggle */}
            <button
              onClick={onDarkModeToggle}
              className={`w-10 h-10 rounded-full flex items-center justify-center transition-colors ${
                darkMode 
                  ? 'hover:bg-gray-800 text-yellow-400' 
                  : 'hover:bg-gray-100 text-gray-700'
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
          </div>
        </div>
      </div>

      {/* Mobile Dropdown Menu */}
      {menuOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 bg-black/20 z-40 md:hidden"
            onClick={() => setMenuOpen(false)}
          />
          
          {/* Menu Panel */}
          <div className={`fixed top-[57px] left-0 right-0 z-40 md:hidden shadow-lg border-b transition-colors duration-200 ${
            darkMode 
              ? 'bg-gray-900 border-gray-800' 
              : 'bg-white border-gray-200'
          }`}>
            <div className="py-2">
              {/* Sessions */}
              <button
                onClick={() => {
                  onToggleSessionsPanel();
                  setMenuOpen(false);
                }}
                className={`w-full flex items-center gap-3 px-4 py-3 transition-colors ${
                  darkMode 
                    ? 'hover:bg-gray-800 text-white' 
                    : 'hover:bg-gray-50 text-gray-700'
                }`}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
                </svg>
                <span>Chat Sessions</span>
              </button>

              {/* Clear History */}
              <button
                onClick={() => {
                  onClearHistory();
                  setMenuOpen(false);
                }}
                className={`w-full flex items-center gap-3 px-4 py-3 transition-colors ${
                  darkMode 
                    ? 'hover:bg-gray-800 text-white' 
                    : 'hover:bg-gray-50 text-gray-700'
                }`}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                <span>Clear History</span>
              </button>
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default ChatHeader;
