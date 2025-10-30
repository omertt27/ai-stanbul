import React, { useState, useEffect } from 'react';

const ChatSessionsPanel = ({ 
  darkMode, 
  isOpen, 
  onClose, 
  currentSessionId,
  onNewSession,
  onSelectSession 
}) => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadSessions();
    }
  }, [isOpen]);

  const loadSessions = () => {
    // Load sessions from localStorage or API
    setLoading(true);
    try {
      const savedSessions = localStorage.getItem('chat_sessions');
      if (savedSessions) {
        setSessions(JSON.parse(savedSessions));
      } else {
        // Initialize with current session if exists
        const currentSession = {
          id: currentSessionId || Date.now().toString(),
          title: 'Current Chat',
          timestamp: new Date().toISOString(),
          messageCount: 0
        };
        setSessions([currentSession]);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleNewSession = () => {
    const newSession = {
      id: Date.now().toString(),
      title: 'New Chat',
      timestamp: new Date().toISOString(),
      messageCount: 0
    };
    
    // Save current session state before creating new one
    const updatedSessions = [newSession, ...sessions];
    localStorage.setItem('chat_sessions', JSON.stringify(updatedSessions));
    
    onNewSession(newSession);
    onClose();
  };

  const handleSelectSession = (session) => {
    onSelectSession(session);
    onClose();
  };

  const handleDeleteSession = (sessionId, e) => {
    e.stopPropagation();
    const updatedSessions = sessions.filter(s => s.id !== sessionId);
    setSessions(updatedSessions);
    localStorage.setItem('chat_sessions', JSON.stringify(updatedSessions));
    
    // If deleted session was current, create new session
    if (sessionId === currentSessionId && updatedSessions.length > 0) {
      handleSelectSession(updatedSessions[0]);
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity"
        onClick={onClose}
      />
      
      {/* Sidebar Panel */}
      <div className={`fixed left-0 top-0 h-full w-80 z-50 transform transition-transform duration-300 shadow-2xl ${
        darkMode ? 'bg-gray-800' : 'bg-white'
      }`}>
        {/* Panel Header */}
        <div className={`flex items-center justify-between p-4 border-b ${
          darkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <h2 className={`text-lg font-semibold ${
            darkMode ? 'text-white' : 'text-gray-900'
          }`}>
            Chat Sessions
          </h2>
          <button
            onClick={onClose}
            className={`p-2 rounded-full transition-colors ${
              darkMode 
                ? 'hover:bg-gray-700 text-gray-300' 
                : 'hover:bg-gray-100 text-gray-600'
            }`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <button
            onClick={handleNewSession}
            className={`w-full flex items-center justify-center space-x-2 px-4 py-3 rounded-lg border-2 transition-all font-medium ${
              darkMode 
                ? 'bg-blue-600 hover:bg-blue-700 text-white border-blue-500' 
                : 'bg-blue-500 hover:bg-blue-600 text-white border-blue-400'
            }`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New Chat</span>
          </button>
        </div>

        {/* Sessions List */}
        <div className="overflow-y-auto h-[calc(100vh-140px)]">
          {loading ? (
            <div className="flex items-center justify-center p-8">
              <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          ) : sessions.length === 0 ? (
            <div className={`text-center p-8 ${
              darkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              <p>No chat sessions yet.</p>
              <p className="text-sm mt-2">Start a new chat to begin!</p>
            </div>
          ) : (
            <div className="space-y-1 px-2">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  onClick={() => handleSelectSession(session)}
                  className={`group relative p-3 rounded-lg cursor-pointer transition-all ${
                    session.id === currentSessionId
                      ? darkMode 
                        ? 'bg-gray-700 border-l-4 border-blue-500' 
                        : 'bg-blue-50 border-l-4 border-blue-500'
                      : darkMode
                        ? 'hover:bg-gray-700 border-l-4 border-transparent'
                        : 'hover:bg-gray-50 border-l-4 border-transparent'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className={`font-medium truncate ${
                        darkMode ? 'text-white' : 'text-gray-900'
                      }`}>
                        {session.title || 'Untitled Chat'}
                      </div>
                      <div className={`text-xs mt-1 flex items-center space-x-2 ${
                        darkMode ? 'text-gray-400' : 'text-gray-500'
                      }`}>
                        <span>{formatTimestamp(session.timestamp)}</span>
                        {session.messageCount > 0 && (
                          <>
                            <span>•</span>
                            <span>{session.messageCount} messages</span>
                          </>
                        )}
                      </div>
                    </div>
                    
                    {/* Delete button - shown on hover */}
                    <button
                      onClick={(e) => handleDeleteSession(session.id, e)}
                      className={`opacity-0 group-hover:opacity-100 ml-2 p-1 rounded transition-all ${
                        darkMode 
                          ? 'hover:bg-red-600 text-gray-400 hover:text-white' 
                          : 'hover:bg-red-100 text-gray-400 hover:text-red-600'
                      }`}
                      title="Delete session"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>

                  {/* Current indicator */}
                  {session.id === currentSessionId && (
                    <div className={`absolute right-2 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full ${
                      darkMode ? 'bg-blue-400' : 'bg-blue-500'
                    }`}></div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Panel Footer */}
        <div className={`absolute bottom-0 left-0 right-0 p-4 border-t ${
          darkMode ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'
        }`}>
          <div className={`text-xs text-center ${
            darkMode ? 'text-gray-400' : 'text-gray-500'
          }`}>
            Sessions are saved locally
          </div>
        </div>
      </div>
    </>
  );
};

export default ChatSessionsPanel;
