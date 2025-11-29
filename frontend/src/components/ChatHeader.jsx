import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const ChatHeader = ({ 
  darkMode, 
  onDarkModeToggle, 
  onClearHistory, 
  onToggleSessionsPanel
}) => {
  const [fabOpen, setFabOpen] = useState(false);
  const navigate = useNavigate();

  const handleNewSession = () => {
    // Clear local session and reload
    localStorage.removeItem('chat_session_id');
    localStorage.removeItem('chat-messages');
    window.location.reload();
  };

  return (
    <>
      {/* Floating Action Button (FAB) - Bottom Right */}
      <div className="fixed bottom-16 md:bottom-12 right-4 md:right-6 z-50">
        {/* Action Menu - Shows when FAB is clicked */}
        {fabOpen && (
          <>
            {/* Backdrop for mobile */}
            <div 
              className="fixed inset-0 bg-black/20 -z-10 md:hidden"
              onClick={() => setFabOpen(false)}
            />
            
            {/* Action Buttons - Stack above FAB */}
            <div className="flex flex-col gap-2 mb-3">
              {/* Sessions Button */}
              <button
                onClick={() => {
                  onToggleSessionsPanel();
                  setFabOpen(false);
                }}
                className={`w-12 h-12 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-110 ${
                  darkMode 
                    ? 'bg-gray-800 text-white hover:bg-gray-700' 
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
                aria-label="Chat sessions"
                title="Chat Sessions"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
                </svg>
              </button>

              {/* New Chat Button */}
              <button
                onClick={() => {
                  handleNewSession();
                  setFabOpen(false);
                }}
                className={`w-12 h-12 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-110 ${
                  darkMode 
                    ? 'bg-gray-800 text-white hover:bg-gray-700' 
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
                aria-label="New chat"
                title="New Chat"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>

              {/* Dark Mode Toggle */}
              <button
                onClick={() => {
                  onDarkModeToggle();
                  setFabOpen(false);
                }}
                className={`w-12 h-12 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-110 ${
                  darkMode 
                    ? 'bg-gray-800 text-yellow-400 hover:bg-gray-700' 
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
                aria-label={`${darkMode ? 'Light' : 'Dark'} mode`}
                title={`${darkMode ? 'Light' : 'Dark'} Mode`}
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

              {/* Clear History Button */}
              <button
                onClick={() => {
                  onClearHistory();
                  setFabOpen(false);
                }}
                className={`w-12 h-12 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-110 ${
                  darkMode 
                    ? 'bg-gray-800 text-red-400 hover:bg-gray-700' 
                    : 'bg-white text-red-600 hover:bg-red-50'
                }`}
                aria-label="Clear history"
                title="Clear History"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>

              {/* Navigation Menu Button */}
              <button
                onClick={() => {
                  navigate('/');
                  setFabOpen(false);
                }}
                className={`w-12 h-12 rounded-full shadow-lg flex items-center justify-center transition-all duration-200 hover:scale-110 ${
                  darkMode 
                    ? 'bg-gray-800 text-white hover:bg-gray-700' 
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
                aria-label="Home"
                title="Home"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
              </button>
            </div>
          </>
        )}

        {/* Main FAB Button */}
        <button
          onClick={() => setFabOpen(!fabOpen)}
          className={`w-14 h-14 rounded-full shadow-xl flex items-center justify-center transition-all duration-200 hover:scale-110 ${
            darkMode 
              ? 'bg-gradient-to-br from-blue-600 to-purple-600 text-white' 
              : 'bg-gradient-to-br from-blue-600 to-purple-600 text-white'
          }`}
          aria-label="Menu"
        >
          {fabOpen ? (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
            </svg>
          )}
        </button>
      </div>
    </>
  );
};

export default ChatHeader;
