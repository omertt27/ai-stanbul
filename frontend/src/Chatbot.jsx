
import { useState, useEffect, useRef } from 'react';
import { fetchStreamingResults } from './api/api';
import { Link, useLocation } from 'react-router-dom';
import { trackNavigation } from './utils/analytics';
import './App.css';

// NavBar component for Chatbot
const ChatbotNavBar = () => {
  const location = useLocation();
  const [isScrolled, setIsScrolled] = useState(false);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  
  // Update window width on resize
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Handle scroll to show/hide navbar
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      setIsScrolled(scrollTop > 100); // Show navbar after scrolling 100px
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  const isMobile = windowWidth < 768;
  
  // Logo style
  const logoStyle = {
    textDecoration: 'none',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'transform 0.2s ease, opacity 0.2s ease',
  };

  const logoTextStyle = {
    fontSize: isMobile ? '1.8rem' : '2.2rem',
    fontWeight: 700,
    letterSpacing: '0.15em',
    textTransform: 'uppercase',
    background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    textShadow: '0 4px 20px rgba(99, 102, 241, 0.3)',
    transition: 'all 0.3s ease',
    cursor: 'pointer',
  };

  const linkStyle = (isActive) => ({
    color: isActive ? '#6366f1' : '#c7c9e2',
    textDecoration: 'none',
    borderBottom: isActive ? '2px solid #6366f1' : '2px solid transparent',
    paddingBottom: '0.5rem',
    paddingTop: '0.5rem',
    paddingLeft: '1rem',
    paddingRight: '1rem',
    borderRadius: '0.5rem',
    transition: 'all 0.2s ease',
    fontWeight: 'inherit',
    whiteSpace: 'nowrap',
    cursor: 'pointer',
    fontSize: isMobile ? '0.9rem' : '1rem',
  });
  
  const handleLogoClick = () => {
    trackNavigation('/');
    const hasActiveChat = localStorage.getItem('chat-messages');
    const parsedMessages = hasActiveChat ? JSON.parse(hasActiveChat) : [];
    
    if (parsedMessages && parsedMessages.length > 0) {
      window.dispatchEvent(new CustomEvent('chatStateChanged', { 
        detail: { expanded: true, hasMessages: true } 
      }));
    } else {
      localStorage.removeItem('chat_session_id');
      localStorage.removeItem('chat-messages');
    }
  };

  return (
    <nav 
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled 
          ? 'bg-gray-900 bg-opacity-95 backdrop-blur-md shadow-lg' 
          : 'bg-gray-900 bg-opacity-80 backdrop-blur-sm'
      }`}
      style={{ 
        borderBottom: isScrolled ? '1px solid rgba(99, 102, 241, 0.2)' : '1px solid rgba(255, 255, 255, 0.1)'
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" style={logoStyle} onClick={handleLogoClick}>
            <div style={logoTextStyle}>
              AI Istanbul
            </div>
          </Link>
          
          {/* Navigation Links */}
          <div className="flex items-center space-x-4">
            <Link 
              to="/about" 
              style={linkStyle(location.pathname === '/about')}
              onClick={() => trackNavigation('/about')}
            >
              About
            </Link>
            <Link 
              to="/sources" 
              style={linkStyle(location.pathname === '/sources')}
              onClick={() => trackNavigation('/sources')}
            >
              Sources
            </Link>
            <Link 
              to="/faq" 
              style={linkStyle(location.pathname === '/faq')}
              onClick={() => trackNavigation('/faq')}
            >
              FAQ
            </Link>
            <Link 
              to="/contact" 
              style={linkStyle(location.pathname === '/contact')}
              onClick={() => trackNavigation('/contact')}
            >
              Contact
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Helper function to render text with clickable links
const renderMessageContent = (content, darkMode) => {
  // Convert Markdown-style links [text](url) to clickable HTML links
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  
  console.log('Rendering content:', content.substring(0, 100) + '...');
  
  const parts = [];
  let lastIndex = 0;
  let match;
  
  while ((match = linkRegex.exec(content)) !== null) {
    const linkText = match[1];
    const linkUrl = match[2];
    
    console.log('Found link:', linkText, '->', linkUrl);
    
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
        className={`underline transition-colors duration-200 hover:opacity-80 cursor-pointer ${
          darkMode 
            ? 'text-blue-400 hover:text-blue-300' 
            : 'text-blue-600 hover:text-blue-700'
        }`}
        onClick={(e) => {
          console.log('Link clicked:', linkUrl);
        }}
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
  
  console.log('Generated parts:', parts.length);
  return parts.length > 0 ? parts : content;
};

function Chatbot({ onDarkModeToggle }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [darkMode, setDarkMode] = useState(true)
  const [suggestions, setSuggestions] = useState([])
  const [inputError, setInputError] = useState('')
  const [chatSessions, setChatSessions] = useState([])
  const [currentSessionId, setCurrentSessionId] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const messagesEndRef = useRef(null)

  // Enhanced input suggestions for better user guidance
  const inputSuggestions = [
    "restaurants in Kadıköy",
    "places to visit in Sultanahmet", 
    "Turkish food recommendations",
    "museums in Istanbul",
    "nightlife in Beyoğlu",
    "transportation in Istanbul",
    "shopping at Grand Bazaar",
    "Bosphorus cruise options",
    "family friendly places",
    "romantic restaurants",
    "budget travel tips",
    "weather in Istanbul"
  ]

  // Load chat sessions from localStorage
  useEffect(() => {
    const savedSessions = localStorage.getItem('chat-sessions');
    if (savedSessions) {
      try {
        const sessions = JSON.parse(savedSessions);
        setChatSessions(sessions);
      } catch (error) {
        console.error('Error loading chat sessions:', error);
      }
    }
  }, []);

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
    setSuggestions(inputSuggestions.slice(0, 6));
  };

  // Load a specific chat session
  const loadChatSession = (sessionId) => {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
      setCurrentSessionId(sessionId);
      setMessages(session.messages);
      setInput('');
      setInputError('');
      setSuggestions([]);
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

  // Apply dark mode class to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Show suggestions when input is empty or short
  useEffect(() => {
    if (input.length === 0) {
      setSuggestions(inputSuggestions.slice(0, 6))
    } else if (input.length > 0 && input.length < 3) {
      // Filter suggestions based on input
      const filtered = inputSuggestions.filter(suggestion => 
        suggestion.toLowerCase().includes(input.toLowerCase())
      ).slice(0, 4)
      setSuggestions(filtered)
    } else {
      setSuggestions([])
    }
  }, [input])

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
    
    // Enhanced input validation
    if (!validateInput(userInput)) {
      return;
    }

    // Create new session if none exists
    if (!currentSessionId) {
      const newSessionId = Date.now().toString();
      setCurrentSessionId(newSessionId);
    }

    const userMessage = { role: 'user', content: userInput };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    setSuggestions([]) // Clear suggestions when sending

    // Start streaming response
    let streamedContent = '';
    let hasError = false;
    let finalMessages = newMessages;
    
    try {
      await fetchStreamingResults(userInput, (chunk) => {
        streamedContent += chunk;
        // If assistant message already exists, update it; else, add it
        setMessages((prev) => {
          const updatedMessages = prev.length > 0 && prev[prev.length - 1].role === 'assistant' && prev[prev.length - 1].streaming
            ? [
                ...prev.slice(0, -1),
                { role: 'assistant', content: streamedContent, streaming: true }
              ]
            : [
                ...prev,
                { role: 'assistant', content: streamedContent, streaming: true }
              ];
          finalMessages = updatedMessages;
          return updatedMessages;
        });
      });
    } catch (error) {
      hasError = true;
      const errorMessage = error.message?.includes('network') 
        ? 'Network error. Please check your connection and try again.'
        : 'Sorry, I encountered an error. Please try rephrasing your question.'
      const errorMessages = [
        ...newMessages,
        { role: 'assistant', content: errorMessage }
      ];
      setMessages(errorMessages);
      finalMessages = errorMessages;
    } finally {
      setLoading(false);
      // Remove streaming flag on last assistant message and save session
      setMessages((prev) => {
        const cleanedMessages = prev.length > 0 && prev[prev.length - 1].role === 'assistant' && prev[prev.length - 1].streaming
          ? [
              ...prev.slice(0, -1),
              { role: 'assistant', content: prev[prev.length - 1].content }
            ]
          : prev;
        
        // Save the session with final messages
        if (currentSessionId && cleanedMessages.length > 0) {
          setTimeout(() => saveCurrentSession(cleanedMessages), 100);
        }
        
        return cleanedMessages;
      });
    }
  }

  const handleLogoClick = () => {
    trackNavigation('/');
    const hasActiveChat = localStorage.getItem('chat-messages');
    const parsedMessages = hasActiveChat ? JSON.parse(hasActiveChat) : [];
    
    if (parsedMessages && parsedMessages.length > 0) {
      window.dispatchEvent(new CustomEvent('chatStateChanged', { 
        detail: { expanded: true, hasMessages: true } 
      }));
    } else {
      localStorage.removeItem('chat_session_id');
      localStorage.removeItem('chat-messages');
    }
  };

  const handleSampleClick = (question) => {
    // Automatically send the message
    handleSend(question);
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

  return (
    <>
      {/* Scrollable Navbar */}
      <ChatbotNavBar />
      
      {/* Chat History Sidebar */}
      <div className={`fixed left-0 top-16 h-[calc(100vh-4rem)] bg-gray-800 border-r border-gray-700 transition-all duration-300 z-40 ${
        sidebarOpen ? 'w-full md:w-80' : 'w-0'
      } overflow-hidden`}>
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className="p-4 border-b border-gray-700">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Chat History</h2>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1 rounded-lg hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <button
              onClick={createNewChat}
              className="w-full mt-3 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors duration-200 flex items-center justify-center"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              New Chat
            </button>
          </div>
          
          {/* Chat Sessions List */}
          <div className="flex-1 overflow-y-auto p-2">
            {chatSessions.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <p className="text-sm">No chat history yet</p>
              </div>
            ) : (
              <div className="space-y-1">
                {chatSessions.map((session) => (
                  <div
                    key={session.id}
                    className={`group relative rounded-lg p-3 cursor-pointer transition-colors duration-200 ${
                      currentSessionId === session.id
                        ? 'bg-indigo-600 text-white'
                        : 'text-gray-300 hover:bg-gray-700'
                    }`}
                    onClick={() => loadChatSession(session.id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{session.title}</p>
                        <p className="text-xs opacity-70 mt-1">{formatSessionDate(session.lastUpdated)}</p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteChatSession(session.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-600 transition-all duration-200 ml-2"
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

      {/* Sidebar Toggle Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="fixed left-4 top-20 z-50 p-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg shadow-lg transition-all duration-200"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Main Chat Container */}
      <div className={`chatbot-background flex flex-col min-h-screen w-full transition-all duration-300 pt-16 ${
        sidebarOpen ? 'md:ml-80 ml-0' : 'ml-0'
      }`}>

        {/* Chat Messages Container - Enhanced with better structure and longer height */}
        <div className="flex-1 min-h-[calc(100vh-12rem)] overflow-y-auto">{/* Made taller: increased from 6rem to 12rem */}
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center px-4 py-8">
              {/* Logo positioned like other pages at the top center */}
              <div className="mb-8">
                <Link to="/" onClick={handleLogoClick}>
                  <div className={`text-6xl font-bold transition-colors duration-300 ${
                    darkMode ? 'text-white' : 'text-gray-800'
                  }`}>
                    <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent font-black">AI</span><span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent font-normal">/STANBUL</span>
                  </div>
                </Link>
              </div>
              
              <div className={`w-20 h-20 rounded-full flex items-center justify-center mb-6 transition-colors duration-200 ${
                darkMode ? 'bg-gradient-to-r from-blue-500 to-purple-600' : 'bg-gradient-to-r from-blue-600 to-purple-700'
              }`}>
                <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
              </svg>
            </div>
            <h2 className={`text-3xl font-bold mb-4 transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>How can I help you today?</h2>
            <p className={`text-center max-w-2xl text-lg leading-relaxed mb-8 transition-colors duration-200 ${
              darkMode ? 'text-gray-300' : 'text-gray-500'
            }`}>
              I'm your AI assistant for exploring Istanbul. Ask me about restaurants, attractions, 
              neighborhoods, culture, history, or anything else about this amazing city!
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl w-full px-4">
              <div 
                onClick={() => handleSampleClick('Show me the best attractions and landmarks in Istanbul')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>�️ Top Attractions</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Show me the best attractions and landmarks in Istanbul</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Find authentic Turkish restaurants in Istanbul')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🍽️ Turkish Cuisine</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Find authentic Turkish restaurants in Istanbul</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Tell me about Istanbul neighborhoods and districts to visit')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>�️ Neighborhoods</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Tell me about Istanbul neighborhoods and districts to visit</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('What are the best cultural experiences and activities in Istanbul?')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🎭 Culture & Activities</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>What are the best cultural experiences and activities in Istanbul?</div>
              </div>
            </div>
          </div>
        )}
            
        <div className="max-w-full mx-auto px-4">
          {messages.map((msg, index) => (
            <div key={index} className="group py-4">
              <div className="flex items-start space-x-3">
                {msg.role === 'user' ? (
                  <>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500' 
                        : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500'
                    }`}>
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                        darkMode ? 'text-gray-300' : 'text-gray-600'
                      }`}>You</div>
                      <div className={`text-sm whitespace-pre-wrap transition-colors duration-200 ${
                        darkMode ? 'text-white' : 'text-gray-800'
                      }`}>
                        {renderMessageContent(msg.content, darkMode)}
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                        : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500'
                    }`}>
                      <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                        darkMode ? 'text-gray-300' : 'text-gray-600'
                      }`}>AI Assistant</div>
                      <div className={`text-sm whitespace-pre-wrap leading-relaxed transition-colors duration-200 ${
                        darkMode ? 'text-white' : 'text-gray-800'
                      }`}>
                        {renderMessageContent(msg.content, darkMode)}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="group py-4">
              <div className="flex items-start space-x-3">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                  darkMode 
                    ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                    : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500'
                }`}>
                  <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                  </svg>
                </div>
                <div className="flex-1">
                  <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>AI Assistant</div>
                  <div className="flex items-center space-x-1">
                    <div className={`w-1.5 h-1.5 rounded-full animate-bounce transition-colors duration-200 ${
                      darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                    }`}></div>
                    <div className={`w-1.5 h-1.5 rounded-full animate-bounce transition-colors duration-200 ${
                      darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                    }`} style={{animationDelay: '0.1s'}}></div>
                    <div className={`w-1.5 h-1.5 rounded-full animate-bounce transition-colors duration-200 ${
                      darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                    }`} style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Scroll anchor for auto-scrolling */}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area - Made thinner with more padding for longer chat outline */}
      <div className={`border-t p-4 transition-colors duration-200 ${
        darkMode 
          ? 'border-gray-700 bg-gray-900' 
          : 'border-gray-200 bg-white'
      }`}>
        <div className="w-full max-w-4xl mx-auto">
          {/* Input suggestions when typing or no input */}
          {suggestions.length > 0 && (
            <div className={`mb-3 p-3 rounded-lg ${
              darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-gray-50 border border-gray-200'
            }`}>
              <div className={`text-xs font-medium mb-2 ${
                darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>💡 Suggestions:</div>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setInput(suggestion)
                      setSuggestions([])
                    }}
                    className={`px-3 py-1 text-sm rounded-full transition-colors duration-200 ${
                      darkMode 
                        ? 'bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white' 
                        : 'bg-gray-200 hover:bg-gray-300 text-gray-700 hover:text-gray-900'
                    }`}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {/* Error message */}
          {inputError && (
            <div className="mb-3 p-3 rounded-lg bg-red-100 border border-red-300 dark:bg-red-900 dark:border-red-700">
              <div className="text-sm text-red-700 dark:text-red-300">
                ⚠️ {inputError}
              </div>
            </div>
          )}
          
          <div className="relative">
            <div className={`flex items-center space-x-2 rounded-xl px-2 py-1 transition-all duration-200 border ${
              inputError 
                ? 'border-red-400 dark:border-red-600' 
                : darkMode 
                  ? 'bg-gray-800 border-gray-600 focus-within:border-blue-500' 
                  : 'bg-white border-gray-300 focus-within:border-blue-500'
            }`}>
              <div className="flex-1 min-h-[16px] max-h-[80px] overflow-y-auto">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value)
                    setInputError('') // Clear error when typing
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask me anything about Istanbul... (restaurants, attractions, districts, culture, etc.)"
                  className={`w-full bg-transparent border-0 outline-none focus:outline-none focus:ring-0 text-base resize-none transition-colors duration-200 ${
                    darkMode 
                      ? 'placeholder-gray-400 text-white' 
                      : 'placeholder-gray-500 text-gray-900'
                  }`}
                  disabled={loading}
                  autoComplete="off"
                />
              </div>
              <button 
                onClick={handleSend} 
                disabled={loading || !input.trim()}
                className={`p-2 rounded-lg transition-all duration-200 transform hover:scale-105 ${
                  darkMode 
                    ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-700 hover:via-indigo-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600' 
                    : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500 hover:from-blue-600 hover:via-indigo-600 hover:to-purple-600 disabled:from-gray-400 disabled:to-gray-400'
                } disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none`}
              >
                {loading ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                  </svg>
                )}
              </button>
            </div>
          </div>
          <div className={`text-xs text-center mt-2 transition-colors duration-200 ${
            darkMode ? 'text-gray-500' : 'text-gray-500'
          }`}>
            Your AI-powered Istanbul guide
          </div>
        </div>
      </div>
      </div>
    </>
  );
}

export default Chatbot;
