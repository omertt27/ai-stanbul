import React from 'react';

const ScrollToBottom = ({ show, onClick, darkMode = true, unreadCount = 0 }) => {
  if (!show) return null;

  return (
    <div className="fixed bottom-20 right-6 z-40">
      <button
        onClick={onClick}
        className={`relative p-3 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 ${
          darkMode 
            ? 'bg-blue-600 hover:bg-blue-700 text-white' 
            : 'bg-blue-500 hover:bg-blue-600 text-white'
        }`}
        title="Scroll to bottom"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
        
        {/* Unread message indicator */}
        {unreadCount > 0 && (
          <div className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
            {unreadCount > 99 ? '99+' : unreadCount}
          </div>
        )}
      </button>
    </div>
  );
};

export default ScrollToBottom;
