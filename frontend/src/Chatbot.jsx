import { useState, useEffect, useRef } from 'react';
import { fetchStreamingResults } from './api/api';
import { Link, useLocation } from 'react-router-dom';
import { trackNavigation } from './utils/analytics';
import QuickTester from './QuickTester';
import NavBar from './components/NavBar';
import './App.css';



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
  const [darkMode, setDarkMode] = useState(false)
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

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    const scrollToBottom = () => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
    
    // Immediate scroll
    scrollToBottom()
    
    // Also scroll after a short delay to ensure DOM is updated
    const timeoutId = setTimeout(scrollToBottom, 100)
    
    return () => clearTimeout(timeoutId)
  }, [messages, loading])

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
      
      // Scroll to bottom after loading finishes
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }, 200);
      
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

  // Handle pending chat query from main page
  useEffect(() => {
    const pendingQuery = localStorage.getItem('pending_chat_query');
    if (pendingQuery) {
      // Clear the pending query
      localStorage.removeItem('pending_chat_query');
      
      // Set the input and automatically send the query
      setInput(pendingQuery);
      
      // Create a new session if needed
      if (!currentSessionId) {
        const newSessionId = Date.now().toString();
        setCurrentSessionId(newSessionId);
      }
      
      // Automatically send the query after a brief delay
      setTimeout(() => {
        handleSend(pendingQuery);
      }, 500);
    }
  }, []);

  return (
    <div className="chatbot-page">
      {/* Standard Site Navigation */}
      <NavBar />
      
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
        className="fixed left-4 top-20 z-50 p-3 bg-gray-800 hover:bg-gray-700 text-white rounded-lg shadow-lg transition-all duration-200 border border-gray-600"
        style={{ zIndex: 1001 }}
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Main Chat Container with Purple Outline */}
      <div className={`chatbot-container transition-all duration-300 ${
        sidebarOpen ? 'md:ml-80 ml-0' : 'ml-0'
      }`}>

        {/* Chat Messages Area */}
        <div className="chatbot-messages">
          
          {/* Welcome Screen - GPT Style */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full py-12">
              
              {/* Main Title */}
              <h1 className={`text-4xl font-semibold mb-12 text-center ${
                darkMode ? 'text-white' : 'text-gray-900'
              }`}>
                How can I help you today?
              </h1>
              
              {/* Example Prompts Grid - GPT Style */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl mb-8">
                <button 
                  onClick={() => handleSampleClick('Show me the best attractions and landmarks in Istanbul')}
                  className={`p-4 rounded-lg border text-left transition-all duration-200 ${
                    darkMode 
                      ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 text-gray-200' 
                      : 'bg-gray-50 border-gray-200 hover:bg-gray-100 text-gray-700'
                  }`}
                  style={{ 
                    borderRadius: '12px',
                    border: darkMode ? '1px solid #374151' : '1px solid #e5e7eb',
                    cursor: 'pointer'
                  }}
                >
                  <div className="font-medium mb-1">🏛️ Top Attractions</div>
                  <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Discover Istanbul's must-visit landmarks
                  </div>
                </button>
                
                <button 
                  onClick={() => handleSampleClick('Find authentic Turkish restaurants in Istanbul')}
                  className={`p-4 rounded-lg border text-left transition-all duration-200 ${
                    darkMode 
                      ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 text-gray-200' 
                      : 'bg-gray-50 border-gray-200 hover:bg-gray-100 text-gray-700'
                  }`}
                  style={{ 
                    borderRadius: '12px',
                    border: darkMode ? '1px solid #374151' : '1px solid #e5e7eb',
                    cursor: 'pointer'
                  }}
                >
                  <div className="font-medium mb-1">🍽️ Turkish Cuisine</div>
                  <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Find authentic local restaurants
                  </div>
                </button>
                
                <button 
                  onClick={() => handleSampleClick('Tell me about Istanbul neighborhoods and districts to visit')}
                  className={`p-4 rounded-lg border text-left transition-all duration-200 ${
                    darkMode 
                      ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 text-gray-200' 
                      : 'bg-gray-50 border-gray-200 hover:bg-gray-100 text-gray-700'
                  }`}
                  style={{ 
                    borderRadius: '12px',
                    border: darkMode ? '1px solid #374151' : '1px solid #e5e7eb',
                    cursor: 'pointer'
                  }}
                >
                  <div className="font-medium mb-1">🏙️ Neighborhoods</div>
                  <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Explore different districts of the city
                  </div>
                </button>
                
                <button 
                  onClick={() => handleSampleClick('What are the best cultural experiences and activities in Istanbul?')}
                  className={`p-4 rounded-lg border text-left transition-all duration-200 ${
                    darkMode 
                      ? 'bg-gray-800 border-gray-700 hover:bg-gray-750 text-gray-200' 
                      : 'bg-gray-50 border-gray-200 hover:bg-gray-100 text-gray-700'
                  }`}
                  style={{ 
                    borderRadius: '12px',
                    border: darkMode ? '1px solid #374151' : '1px solid #e5e7eb',
                    cursor: 'pointer'
                  }}
                >
                  <div className="font-medium mb-1">🎭 Culture & Activities</div>
                  <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Cultural experiences and activities
                  </div>
                </button>
              </div>
            </div>
          )}
          
          {/* Chat Messages - GPT/Gemini Style */}
          <div className="py-4">
            {messages.map((msg, index) => (
              <div key={index} className="mb-6" style={{ maxWidth: '100%' }}>
                <div className={`flex items-start gap-3 ${
                  msg.role === 'user' ? 'justify-end' : 'justify-start'
                }`}>
                  {msg.role === 'assistant' && (
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      darkMode ? 'bg-gray-700' : 'bg-gray-100'
                    }`}>
                      <svg className={`w-4 h-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} fill="currentColor" viewBox="0 0 24 24">
                        <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                      </svg>
                    </div>
                  )}
                  
                  <div className={`max-w-[80%] ${
                    msg.role === 'user' 
                      ? 'bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-2'
                      : darkMode 
                        ? 'text-gray-100' 
                        : 'text-gray-900'
                  }`}>
                    <div className={`${msg.role === 'assistant' ? 'prose prose-sm max-w-none' : ''} leading-relaxed`}>
                      {renderMessageContent(msg.content, darkMode)}
                    </div>
                  </div>
                  
                  {msg.role === 'user' && (
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      darkMode ? 'bg-blue-600' : 'bg-blue-600'
                    }`}>
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="mb-6">
                <div className="flex items-start gap-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    darkMode ? 'bg-gray-700' : 'bg-gray-100'
                  }`}>
                    <svg className={`w-4 h-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} fill="currentColor" viewBox="0 0 24 24">
                      <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                    </svg>
                  </div>
                  <div className="flex items-center space-x-1 py-2">
                    <div className={`w-2 h-2 rounded-full animate-bounce ${darkMode ? 'bg-gray-400' : 'bg-gray-500'}`}></div>
                    <div className={`w-2 h-2 rounded-full animate-bounce ${darkMode ? 'bg-gray-400' : 'bg-gray-500'}`} style={{animationDelay: '0.1s'}}></div>
                    <div className={`w-2 h-2 rounded-full animate-bounce ${darkMode ? 'bg-gray-400' : 'bg-gray-500'}`} style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area - GPT/Gemini Style */}
          <div className="chatbot-input-area">
            
            {/* Input suggestions when typing or no input */}
            {suggestions.length > 0 && (
              <div className={`mb-3 p-3 rounded-lg ${
                darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-gray-600 border border-gray-700'
              }`}>
                <div className={`text-xs font-medium mb-2 ${
                  darkMode ? 'text-gray-300' : 'text-gray-200'
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
                          : 'bg-gray-700 hover:bg-gray-600 text-gray-200 hover:text-white'
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
            
            {/* Input Box */}
            <div className="relative">            <div className={`flex items-center space-x-3 rounded-2xl px-4 py-3 transition-all duration-200 border ${
              inputError 
                ? 'border-red-400 dark:border-red-600' 
                : darkMode 
                  ? 'bg-gray-800 border-gray-700 focus-within:border-gray-600' 
                  : 'bg-gray-500 border-gray-600 focus-within:border-gray-700'
            }`} style={{
              borderRadius: '24px',
              boxShadow: 'none'
            }}>
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
                  placeholder="Message AI Istanbul"
                  className={`flex-1 bg-transparent border-0 outline-none focus:outline-none focus:ring-0 text-base resize-none transition-colors duration-200 ${
                    darkMode 
                      ? 'placeholder-gray-300 text-white' 
                      : 'placeholder-gray-300 text-white'
                  }`}
                  disabled={loading}
                  autoComplete="off"
                />
                <button 
                  onClick={handleSend} 
                  disabled={loading || !input.trim()}
                  className={`p-2 rounded-full transition-all duration-200 ${
                    loading || !input.trim()
                      ? 'bg-gray-300 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700 cursor-pointer'
                  } text-white`}
                  style={{ borderRadius: '50%' }}
                >
                  {loading ? (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Quick Tester Component */}
      <QuickTester onTestInput={(input) => {
        setInput(input);
        setTimeout(() => handleSend(input), 100);
      }} />
    </div>
  );
}

export default Chatbot;
