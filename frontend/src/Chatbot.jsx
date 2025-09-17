import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { fetchStreamingResults } from './api/api';
import { Link, useLocation } from 'react-router-dom';
import { trackNavigation } from './utils/analytics';
import NavBar from './components/NavBar';
import './App.css';



// Helper function to render text with clickable links
const renderMessageContent = (content, darkMode) => {
  // Convert Markdown-style links [text](url) to clickable HTML links
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  
  const parts = [];
  let lastIndex = 0;
  let match;
  
  while ((match = linkRegex.exec(content)) !== null) {
    const linkText = match[1];
    const linkUrl = match[2];
    
    // Add text before the link
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {content.substring(lastIndex, match.index)}
        </span>
      );
    }
    
    // Add the clickable link
    parts.push(
      <a
        key={`link-${match.index}`}
        href={linkUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="underline transition-colors duration-200 hover:opacity-80 cursor-pointer text-blue-400 hover:text-blue-300"
      >
        {linkText}
      </a>
    );
    
    lastIndex = linkRegex.lastIndex;
  }
  
  // Add any remaining text after the last link
  if (lastIndex < content.length) {
    parts.push(
      <span key={`text-${lastIndex}`}>
        {content.substring(lastIndex)}
      </span>
    );
  }
  
  return parts.length > 0 ? parts : content;
};

function Chatbot({ onDarkModeToggle }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [darkMode, setDarkMode] = useState(false)
  const [suggestions, setSuggestions] = useState([])
  const [inputError, setInputError] = useState('')
  const [chatSessions, setChatSessions] = useState([])
  const [currentSessionId, setCurrentSessionId] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [readingMessageId, setReadingMessageId] = useState(null)
  const [speechSupported, setSpeechSupported] = useState(false)
  const messagesEndRef = useRef(null)
  const speechSynthesis = useRef(null)

  // Removed input suggestions as requested

  // Add chatbot-page class to body for proper styling
  useEffect(() => {
    document.body.classList.add('chatbot-page');
    
    // Initialize speech synthesis
    if ('speechSynthesis' in window) {
      setSpeechSupported(true);
      speechSynthesis.current = window.speechSynthesis;
    }
    
    return () => {
      document.body.classList.remove('chatbot-page');
      // Stop any ongoing speech when component unmounts
      if (speechSynthesis.current) {
        speechSynthesis.current.cancel();
      }
    };
  }, []);

  // Load chat sessions from localStorage
  useEffect(() => {
    const savedSessions = localStorage.getItem('chat-sessions');
    if (savedSessions) {
      try {
        const sessions = JSON.parse(savedSessions);
        setChatSessions(sessions);
        
        // Auto-load the most recent session if no current session and sessions exist
        if (!currentSessionId && sessions.length > 0) {
          const mostRecentSession = sessions[0]; // Sessions are ordered by most recent first
          setCurrentSessionId(mostRecentSession.id);
          setMessages(mostRecentSession.messages);
        }
      } catch (error) {
        console.error('Error loading chat sessions:', error);
      }
    }
    // Don't auto-create sessions here - let handleSend create them when needed
  }, []); // Empty dependency array to run only once on mount

  // Save chat sessions to localStorage
  const saveChatSessions = (sessions) => {
    localStorage.setItem('chat-sessions', JSON.stringify(sessions));
    setChatSessions(sessions);
  };

  // Create a new chat session
  const createNewChat = () => {
    const newSessionId = Date.now().toString();
    setCurrentSessionId(newSessionId);
    setMessages([]);
    setInput('');
    setInputError('');
    // Removed suggestions setting as suggestions are removed
  };

  // Load a specific chat session
  const loadChatSession = (sessionId) => {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setCurrentSessionId(sessionId);
      setMessages(session.messages);
      setInput('');
      setInputError('');
      // Removed suggestions clearing as suggestions are removed
    }
  };

  // Save current chat session
  const saveCurrentSession = (newMessages) => {
    if (!currentSessionId || newMessages.length === 0) return;

    const sessionTitle = newMessages[0]?.content?.substring(0, 50) + '...' || 'New Chat';
    const updatedSession = {
      id: currentSessionId,
      title: sessionTitle,
      messages: newMessages,
      lastUpdated: new Date().toISOString()
    };

    const existingIndex = chatSessions.findIndex(s => s.id === currentSessionId);
    let updatedSessions;
    
    if (existingIndex >= 0) {
      updatedSessions = [...chatSessions];
      updatedSessions[existingIndex] = updatedSession;
    } else {
      updatedSessions = [updatedSession, ...chatSessions].slice(0, 50); // Keep only last 50 sessions
    }

    saveChatSessions(updatedSessions);
  };

  // Delete a chat session
  const deleteChatSession = (sessionId) => {
    const updatedSessions = chatSessions.filter(s => s.id !== sessionId);
    saveChatSessions(updatedSessions);
    
    if (currentSessionId === sessionId) {
      createNewChat();
    }
  };

  // Auto-scroll to bottom when new messages arrive - optimized
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages.length]); // Only trigger on message count change, not content changes

  // Removed suggestions functionality as requested

  // Enhanced input validation and processing
  const validateInput = (userInput) => {
    const trimmedInput = userInput.trim()
    
    // Check for empty input
    if (!trimmedInput) {
      setInputError('Please enter a question about Istanbul!')
      return false
    }
    
    // Check for very short input
    if (trimmedInput.length < 2) {
      setInputError('Please enter a more detailed question.')
      return false
    }
    
    // Check for spam-like input (repeated characters)
    if (/(.)\1{4,}/.test(trimmedInput)) {
      setInputError('Please enter a meaningful question.')
      return false
    }
    
    // Check for only special characters
    if (!/[a-zA-Z0-9]/.test(trimmedInput)) {
      setInputError('Please use letters and words in your question.')
      return false
    }
    
    setInputError('')
    return true
  }

  const handleSend = async (customInput = null) => {
    const userInput = customInput || input.trim();
    
    // Prevent sending if already loading
    if (loading) {
      return;
    }
    
    // Enhanced input validation
    if (!validateInput(userInput)) {
      return;
    }

    // Only create new session if none exists
    if (!currentSessionId) {
      const newSessionId = Date.now().toString();
      setCurrentSessionId(newSessionId);
    }

    const userMessage = { role: 'user', content: userInput };
    
    // Immediately update messages and clear input for ChatGPT-like experience
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setInputError('');

    // Start streaming response immediately
    let streamedContent = '';
    
    try {
      await fetchStreamingResults(userInput, (chunk) => {
        streamedContent += chunk;
        
        // Update messages in real-time as chunks come in
        setMessages((prevMessages) => {
          const lastMessage = prevMessages[prevMessages.length - 1];
          
          // If last message is assistant and streaming, update it
          if (lastMessage && lastMessage.role === 'assistant' && lastMessage.streaming) {
            return [
              ...prevMessages.slice(0, -1),
              { role: 'assistant', content: streamedContent, streaming: true }
            ];
          } else {
            // Add new assistant message
            return [
              ...prevMessages,
              { role: 'assistant', content: streamedContent, streaming: true }
            ];
          }
        });
      });
    } catch (error) {
      const errorMessage = error.message?.includes('network') 
        ? 'Network error. Please check your connection and try again.'
        : 'Sorry, I encountered an error. Please try rephrasing your question.';
      
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: errorMessage }
      ]);
    } finally {
      setLoading(false);
      
      // Finalize the last message by removing streaming flag
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage && lastMessage.role === 'assistant' && lastMessage.streaming) {
          const finalMessages = [
            ...prev.slice(0, -1),
            { role: 'assistant', content: lastMessage.content }
          ];
          
          // Save session with final messages
          if (currentSessionId && finalMessages.length > 0) {
            setTimeout(() => saveCurrentSession(finalMessages), 100);
          }
          
          return finalMessages;
        }
        return prev;
      });
      
      // Auto-scroll to bottom
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  }

  // Format date for chat sessions
  const formatSessionDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = (now - date) / (1000 * 60 * 60);
    
    if (diffInHours < 24) {
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } else if (diffInHours < 168) {
      return date.toLocaleDateString('en-US', { weekday: 'short' });
    } else {
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  };

  // Copy message to clipboard
  const copyToClipboard = async (text, messageIndex) => {
    try {
      await navigator.clipboard.writeText(text);
      // Show temporary feedback (you could add a toast notification here)
      console.log('Message copied to clipboard');
    } catch (err) {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      console.log('Message copied to clipboard (fallback)');
    }
  };

  // Read message aloud
  const readAloud = (text, messageIndex) => {
    if (!speechSupported || !speechSynthesis.current) {
      console.log('Speech synthesis not supported');
      return;
    }

    // Stop any current speech
    speechSynthesis.current.cancel();

    // If already reading this message, stop
    if (readingMessageId === messageIndex) {
      setReadingMessageId(null);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    
    // Configure speech settings
    utterance.rate = 0.9; // Slightly slower for better comprehension
    utterance.pitch = 1;
    utterance.volume = 1;
    
    // Use a more natural voice if available
    const voices = speechSynthesis.current.getVoices();
    const englishVoice = voices.find(voice => 
      voice.lang.startsWith('en') && voice.name.includes('Google')
    ) || voices.find(voice => voice.lang.startsWith('en'));
    
    if (englishVoice) {
      utterance.voice = englishVoice;
    }

    // Set reading state
    setReadingMessageId(messageIndex);

    // Handle speech events
    utterance.onend = () => {
      setReadingMessageId(null);
    };

    utterance.onerror = () => {
      setReadingMessageId(null);
      console.error('Speech synthesis error');
    };

    // Start speaking
    speechSynthesis.current.speak(utterance);
  };

  // Stop reading
  const stopReading = () => {
    if (speechSynthesis.current) {
      speechSynthesis.current.cancel();
      setReadingMessageId(null);
    }
  };

  // Handle pending chat query from main page - START NEW CHAT
  useEffect(() => {
    const pendingQuery = localStorage.getItem('pending_chat_query');
    if (pendingQuery) {
      // Clear the pending query immediately
      localStorage.removeItem('pending_chat_query');
      
      // Create a completely new chat session for main page queries
      const newSessionId = Date.now().toString();
      setCurrentSessionId(newSessionId);
      setMessages([]); // Start with empty messages
      setInput(''); // Clear input
      setInputError(''); // Clear any errors
      
      // Set the input with the pending query
      setInput(pendingQuery);
      
      // Automatically send the query after a brief delay to allow state to update
      setTimeout(() => {
        handleSend(pendingQuery);
      }, 100);
    }
  }, []); // Run only once on mount to check for pending queries

  return (
    <div className="chatbot-page">
      {/* Standard Site Navigation */}
      <NavBar />
      
      {/* Chat History Sidebar - Modern Design */}
      <div className={`fixed left-0 top-12 h-[calc(100vh-3rem)] transition-all duration-300 z-40 ${
        sidebarOpen ? 'w-full md:w-80' : 'w-0'
      } overflow-hidden`} style={{
        background: 'linear-gradient(135deg, rgba(15, 16, 17, 0.98) 0%, rgba(26, 27, 29, 0.98) 100%)',
        backdropFilter: 'blur(20px)',
        borderRight: '1px solid rgba(139, 92, 246, 0.3)',
        boxShadow: '4px 0 20px rgba(139, 92, 246, 0.15)'
      }}>
        <div className="flex flex-col h-full">
          {/* Sidebar Header - Modern Style */}
          <div className="p-6" style={{
            borderBottom: '1px solid rgba(139, 92, 246, 0.2)',
            background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%)'
          }}>
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-white" style={{
                background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
              }}>Chat History</h2>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-2 rounded-xl transition-all duration-200"
                style={{
                  background: 'rgba(139, 92, 246, 0.1)',
                  border: '1px solid rgba(139, 92, 246, 0.3)',
                  color: '#e5e7eb'
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = 'rgba(139, 92, 246, 0.2)';
                  e.target.style.transform = 'scale(1.05)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = 'rgba(139, 92, 246, 0.1)';
                  e.target.style.transform = 'scale(1)';
                }}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <button
              onClick={createNewChat}
              className="mt-4 px-4 py-2 text-white rounded-xl transition-all duration-200 flex items-center justify-center font-semibold mx-auto"
              style={{
                background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
                border: '1px solid rgba(139, 92, 246, 0.5)',
                boxShadow: '0 4px 16px rgba(139, 92, 246, 0.3)',
                width: 'auto', // Auto width instead of full width
                minWidth: '120px', // Minimum width for usability
                maxWidth: '160px' // Maximum width to keep it compact
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px) scale(1.02)';
                e.target.style.boxShadow = '0 6px 20px rgba(139, 92, 246, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0) scale(1)';
                e.target.style.boxShadow = '0 4px 16px rgba(139, 92, 246, 0.3)';
              }}
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              New Chat
            </button>
          </div>
          
          {/* Chat Sessions List - Modern Design */}
          <div className="flex-1 overflow-y-auto p-4">
            {chatSessions.length === 0 ? (
              <div className="text-center py-12" style={{color: '#9ca3af'}}>
                <div className="mb-4" style={{
                  background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%)',
                  borderRadius: '50%',
                  width: '80px',
                  height: '80px',
                  margin: '0 auto',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid rgba(139, 92, 246, 0.3)'
                }}>
                  <svg className="w-8 h-8" style={{color: '#8b5cf6'}} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <p className="text-sm font-medium">No conversations yet</p>
                <p className="text-xs mt-1 opacity-70">Start a new chat to see your history</p>
              </div>
            ) : (
              <div className="space-y-2">
                {chatSessions.map((session) => (
                  <div
                    key={session.id}
                    className="group relative cursor-pointer transition-all duration-200"
                    style={{
                      background: currentSessionId === session.id 
                        ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.3) 0%, rgba(99, 102, 241, 0.2) 100%)'
                        : 'rgba(139, 92, 246, 0.05)',
                      border: `1px solid ${currentSessionId === session.id ? 'rgba(139, 92, 246, 0.5)' : 'rgba(139, 92, 246, 0.1)'}`,
                      borderRadius: '12px',
                      padding: '16px',
                      boxShadow: currentSessionId === session.id 
                        ? '0 4px 16px rgba(139, 92, 246, 0.25)'
                        : '0 2px 8px rgba(139, 92, 246, 0.1)'
                    }}
                    onClick={() => loadChatSession(session.id)}
                    onMouseEnter={(e) => {
                      if (currentSessionId !== session.id) {
                        e.target.style.background = 'rgba(139, 92, 246, 0.1)';
                        e.target.style.borderColor = 'rgba(139, 92, 246, 0.3)';
                        e.target.style.transform = 'translateY(-1px)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (currentSessionId !== session.id) {
                        e.target.style.background = 'rgba(139, 92, 246, 0.05)';
                        e.target.style.borderColor = 'rgba(139, 92, 246, 0.1)';
                        e.target.style.transform = 'translateY(0)';
                      }
                    }}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold truncate" style={{
                          color: currentSessionId === session.id ? '#ffffff' : '#e5e7eb'
                        }}>{session.title}</p>
                        <p className="text-xs mt-1" style={{
                          color: currentSessionId === session.id ? 'rgba(255,255,255,0.8)' : 'rgba(229,231,235,0.6)'
                        }}>{formatSessionDate(session.lastUpdated)}</p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteChatSession(session.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-2 rounded-lg transition-all duration-200 ml-2"
                        style={{
                          background: 'rgba(239, 68, 68, 0.1)',
                          border: '1px solid rgba(239, 68, 68, 0.3)',
                          color: '#ef4444'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = 'rgba(239, 68, 68, 0.2)';
                          e.target.style.transform = 'scale(1.05)';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = 'rgba(239, 68, 68, 0.1)';
                          e.target.style.transform = 'scale(1)';
                        }}
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Modern Chat History Button - Following AI Chat App Patterns */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="fixed left-2 top-28 z-50 transition-all duration-300 group"
        style={{ 
          zIndex: 1001,
          background: sidebarOpen 
            ? 'rgba(139, 92, 246, 0.15)' 
            : 'rgba(255, 255, 255, 0.08)',
          backdropFilter: 'blur(20px)',
          border: sidebarOpen 
            ? '1px solid rgba(139, 92, 246, 0.4)' 
            : '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '12px',
          padding: '10px',
          boxShadow: sidebarOpen 
            ? '0 8px 32px rgba(139, 92, 246, 0.25)' 
            : '0 4px 20px rgba(0, 0, 0, 0.1)',
          color: sidebarOpen ? '#8b5cf6' : '#9ca3af',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '44px',
          height: '44px',
          cursor: 'pointer'
        }}
        onMouseEnter={(e) => {
          e.target.style.transform = 'translateY(-1px) scale(1.05)';
          e.target.style.boxShadow = sidebarOpen 
            ? '0 12px 48px rgba(139, 92, 246, 0.3)' 
            : '0 8px 32px rgba(139, 92, 246, 0.2)';
          e.target.style.background = 'rgba(139, 92, 246, 0.15)';
          e.target.style.borderColor = 'rgba(139, 92, 246, 0.4)';
          e.target.style.color = '#8b5cf6';
        }}
        onMouseLeave={(e) => {
          e.target.style.transform = 'translateY(0) scale(1)';
          e.target.style.boxShadow = sidebarOpen 
            ? '0 8px 32px rgba(139, 92, 246, 0.25)' 
            : '0 4px 20px rgba(0, 0, 0, 0.1)';
          e.target.style.background = sidebarOpen 
            ? 'rgba(139, 92, 246, 0.15)' 
            : 'rgba(255, 255, 255, 0.08)';
          e.target.style.borderColor = sidebarOpen 
            ? 'rgba(139, 92, 246, 0.4)' 
            : 'rgba(255, 255, 255, 0.1)';
          e.target.style.color = sidebarOpen ? '#8b5cf6' : '#9ca3af';
        }}
        title="Chat History"
      >
        <svg 
          width="20" 
          height="20" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24" 
          style={{ 
            transition: 'all 0.3s ease',
            transform: sidebarOpen ? 'rotate(180deg)' : 'rotate(0deg)'
          }}
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d={sidebarOpen 
              ? "M15 19l-7-7 7-7" 
              : "M8 9l4-4 4 4m0 6l-4 4-4-4"
            } 
          />
        </svg>
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Main Chat Container with Purple Outline */}
      <div className={`chatbot-main-container transition-all duration-300 ${
        sidebarOpen ? 'md:ml-80 ml-0' : 'ml-0'
      }`}>
        <div className="chatbot-purple-box">

          {/* Chat Messages Area - Separate from Input */}
          <div className="chatbot-messages">
          
          {/* Welcome Screen - Clean and modern */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-start h-full pt-4 px-4">
              
              {/* Main Title */}
              <h1 className="text-4xl font-bold mb-4 text-center max-w-2xl text-white" style={{
                background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
              }}>
                Hello! I'm KAM
              </h1>
              
              {/* Subtitle */}
              <p className={`text-lg mb-6 text-center max-w-xl ${
                darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>
                Your AI assistant for exploring Istanbul. Ask me anything about attractions, restaurants, culture, and more!
              </p>
              
            </div>
          )}
          
          {/* Chat Messages - Clean Style with Action Buttons */}
          <div className="py-1">
            {messages.map((msg, index) => (
              <div key={index} className="mb-4 group" style={{ maxWidth: '100%' }}>
                <div className={`flex ${
                  msg.role === 'user' ? 'justify-end' : 'justify-start'
                }`}>
                  {msg.role === 'assistant' && (
                    <div className="text-xs font-medium mb-1 text-gray-100">
                      KAM
                    </div>
                  )}
                </div>
                <div className={`flex ${
                  msg.role === 'user' ? 'justify-end' : 'justify-start'
                } relative`}>
                  <div className={`max-w-[80%] relative ${
                    msg.role === 'user' 
                      ? 'bg-purple-600 text-white rounded-2xl rounded-br-md px-4 py-3'
                      : 'bg-gray-800 text-gray-100 rounded-2xl rounded-bl-md px-4 py-3 border border-gray-700'
                  }`}>
                    <div className={`${msg.role === 'assistant' ? 'prose prose-sm max-w-none' : ''} leading-relaxed`}>
                      {renderMessageContent(msg.content, darkMode)}
                    </div>
                  </div>
                </div>
                
                {/* Action Buttons - Outside message bubble like ChatGPT */}
                {msg.role === 'assistant' && (
                  <div className={`kam-message-actions flex ${
                    msg.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}>
                    
                    {/* Copy Button */}
                    <button
                      onClick={() => copyToClipboard(msg.content, index)}
                      className="kam-action-button kam-copy-button"
                      title="Copy message"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      <span className="text-xs">Copy</span>
                    </button>
                    
                    {/* Read Aloud Button */}
                    {speechSupported && (
                      <button
                        onClick={() => readingMessageId === index ? stopReading() : readAloud(msg.content, index)}
                        className={`kam-action-button kam-read-button ${readingMessageId === index ? 'active' : ''}`}
                        title={readingMessageId === index ? "Stop reading" : "Read aloud"}
                      >
                        {readingMessageId === index ? (
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-6.219-8.56" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10l2 2 4-4" />
                          </svg>
                        ) : (
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 9H4a1 1 0 00-1 1v4a1 1 0 001 1h1.586l4.707 4.707C10.923 20.337 12 19.575 12 18.586V5.414c0-.989-1.077-1.751-1.707-1.121L5.586 9z" />
                          </svg>
                        )}
                        <span className="text-xs">
                          {readingMessageId === index ? 'Stop' : 'Read'}
                        </span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            ))}
            
            {loading && (
              <div className="mb-4">
                <div className="flex justify-start">
                  <div className="text-xs font-medium mb-1 text-gray-100">
                    KAM
                  </div>
                </div>
                <div className="flex justify-start">
                  <div className="max-w-[80%] bg-gray-800 text-gray-100 rounded-2xl rounded-bl-md px-4 py-3 border border-gray-700">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 rounded-full animate-bounce bg-gray-400"></div>
                      <div className="w-2 h-2 rounded-full animate-bounce bg-gray-400" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 rounded-full animate-bounce bg-gray-400" style={{animationDelay: '0.2s'}}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Redesigned Input Area - Modern and Elegant */}
        <div className="chatbot-input-area">
          
          {/* Error message with improved styling */}
          {inputError && (
            <div className="kam-error-message mb-4">
              <div className="flex items-center space-x-2">
                <svg className="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <span className="text-red-300 font-medium">{inputError}</span>
              </div>
            </div>
          )}
          
          {/* Completely Redesigned Input Container - No surrounding border */}
          <div className="kam-input-wrapper">
            <div className="kam-input-container">
              
              {/* Input Field */}
              <input
                type="text"
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  if (inputError) setInputError(''); // Clear error when typing
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !loading) {
                    e.preventDefault();
                    if (input.trim()) {
                      handleSend();
                    }
                  }
                }}
                placeholder="What would you like to know about Istanbul?"
                className="kam-input-field"
                disabled={loading}
                autoComplete="off"
                autoFocus={false}
              />
              
              {/* Search Icon - positioned over input */}
              <div className="kam-search-icon">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              
              {/* Send Button */}
              <button 
                onClick={() => {
                  if (!loading && input.trim()) {
                    handleSend();
                  }
                }} 
                disabled={loading || !input.trim()}
                className="kam-send-button"
                aria-label="Send message"
              >
                {loading ? (
                  <div className="kam-loading-spinner">
                    <div className="kam-spinner-ring"></div>
                  </div>
                ) : (
                  <span style={{ fontSize: '18px' }}>✈️</span>
                )}
              </button>
            </div>
          </div>
          
          {/* Quick suggestions (optional) */}
          {messages.length === 0 && !loading && (
            <div className="kam-quick-suggestions">
              <div className="text-xs text-gray-400 mb-2">Try asking about:</div>
              <div className="flex flex-wrap gap-2">
                {[
                  "Best restaurants in Sultanahmet",
                  "Things to do in Beyoğlu", 
                  "Ferry routes to the islands",
                  "Turkish breakfast spots"
                ].map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => setInput(suggestion)}
                    className="kam-suggestion-pill"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        </div>
      </div>
    </div>
  );
}

export default Chatbot;
