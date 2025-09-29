import { useState, useEffect, useRef } from 'react';
import { fetchStreamingResults } from './api/api';
import { trackNavigation, trackEvent, trackChatEvent } from './utils/analytics';
import NavBar from './components/NavBar';
import MobileOptimizer from './components/MobileOptimizer';
import { 
  TypingSimulator, 
  StreamingText, 
  TypingIndicator, 
  LoadingSpinner
} from './components/TypingAnimation';
import { LoadingSkeleton, ChatMessageSkeleton } from './components/LoadingSkeletons';
import { recordUserInteraction, measureApiResponseTime } from './utils/uxEnhancements';
import './App.css';



// Helper function to render text with clickable links and proper formatting
const renderMessageContent = (content, darkMode) => {
  // Enhanced text processing for better readability
  
  // Step 1: Handle numbered lists (1. 2. 3. etc.)
  let formattedContent = content.replace(/^(\d+\.\s)/gm, '\n$1');
  
  // Step 2: Handle bullet points (-)
  formattedContent = formattedContent.replace(/^(\s*-\s)/gm, '\n$1');
  
  // Step 3: Add extra spacing around main sections (lines ending with :)
  formattedContent = formattedContent.replace(/^([^:\n]+:)\s*$/gm, '\n$1\n');
  
  // Step 4: Split content into paragraphs (double line breaks or section breaks)
  const paragraphs = formattedContent.split(/\n\s*\n/).filter(p => p.trim());
  
  return paragraphs.map((paragraph, paragraphIndex) => {
    // Convert Markdown-style links [text](url) to clickable HTML links
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    
    const parts = [];
    let lastIndex = 0;
    let match;
    
    while ((match = linkRegex.exec(paragraph)) !== null) {
      const linkText = match[1];
      const linkUrl = match[2];
      
      // Add text before the link
      if (match.index > lastIndex) {
        const textContent = paragraph.substring(lastIndex, match.index);
        processTextContent(textContent, parts, `${paragraphIndex}-${lastIndex}`);
      }
      
      // Add the clickable link
      parts.push(
        <a
          key={`link-${paragraphIndex}-${match.index}`}
          href={linkUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: '#60a5fa',
            textDecoration: 'underline',
            cursor: 'pointer',
            transition: 'color 0.2s ease'
          }}
          onMouseOver={(e) => e.target.style.color = '#3b82f6'}
          onMouseOut={(e) => e.target.style.color = '#60a5fa'}
        >
          {linkText}
        </a>
      );
      
      lastIndex = linkRegex.lastIndex;
    }
    
    // Add any remaining text after the last link
    if (lastIndex < paragraph.length) {
      const textContent = paragraph.substring(lastIndex);
      processTextContent(textContent, parts, `${paragraphIndex}-${lastIndex}`);
    }
    
    // If no links were found, handle the whole paragraph
    if (parts.length === 0) {
      processTextContent(paragraph, parts, paragraphIndex);
    }
    
    // Return each paragraph with proper spacing
    return (
      <div 
        key={`paragraph-${paragraphIndex}`} 
        style={{ 
          marginBottom: paragraphIndex < paragraphs.length - 1 ? '1rem' : '0',
          lineHeight: '1.6'
        }}
      >
        {parts}
      </div>
    );
  });
};

// Helper function to process text content with proper line breaks and formatting
const processTextContent = (textContent, parts, keyPrefix) => {
  const lines = textContent.split('\n');
  
  lines.forEach((line, lineIndex) => {
    if (lineIndex > 0) {
      parts.push(<br key={`br-${keyPrefix}-${lineIndex}`} />);
    }
    
    if (line.trim()) {
      // Check if this is a numbered list item
      const isNumberedList = /^\d+\.\s/.test(line.trim());
      // Check if this is a bullet point
      const isBulletPoint = /^\s*-\s/.test(line);
      // Check if this is a section header (ends with :)
      const isSectionHeader = line.trim().endsWith(':') && !line.includes('http');
      
      let processedLine = line;
      
      // Add styling for different content types
      if (isNumberedList) {
        const [number, ...rest] = line.split(/\.\s/);
        parts.push(
          <span key={`numbered-${keyPrefix}-${lineIndex}`} style={{
            display: 'block',
            marginTop: '0.5rem',
            marginBottom: '0.25rem'
          }}>
            <strong style={{ color: '#60a5fa' }}>{number}.</strong>
            <span style={{ marginLeft: '0.5rem' }}>{rest.join('. ')}</span>
          </span>
        );
      } else if (isBulletPoint) {
        const cleanLine = line.replace(/^\s*-\s/, '');
        parts.push(
          <span key={`bullet-${keyPrefix}-${lineIndex}`} style={{
            display: 'block',
            marginLeft: '1rem',
            marginTop: '0.25rem',
            marginBottom: '0.25rem'
          }}>
            <span style={{ color: '#60a5fa', marginRight: '0.5rem' }}>•</span>
            {cleanLine}
          </span>
        );
      } else if (isSectionHeader) {
        parts.push(
          <span key={`header-${keyPrefix}-${lineIndex}`} style={{
            display: 'block',
            fontWeight: '600',
            color: '#f1f5f9',
            marginTop: '1rem',
            marginBottom: '0.5rem',
            fontSize: '1.05em'
          }}>
            {line}
          </span>
        );
      } else {
        parts.push(
          <span key={`text-${keyPrefix}-${lineIndex}`}>
            {line}
          </span>
        );
      }
    }
  });
};

function Chatbot() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [darkMode, setDarkMode] = useState(false)
  const [inputError, setInputError] = useState('')
  const [chatSessions, setChatSessions] = useState([])
  const [currentSessionId, setCurrentSessionId] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [readingMessageId, setReadingMessageId] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  
  // Add liked and disliked messages state
  const [likedMessages, setLikedMessages] = useState(new Set());
  const [dislikedMessages, setDislikedMessages] = useState(new Set());
  const [savedSessions, setSavedSessions] = useState([]);

  // Speech synthesis support
  const speechSynthesis = useRef(window.speechSynthesis);
  const speechSupported = 'speechSynthesis' in window;
  const [speechSynthesisSupported, setSpeechSupported] = useState(speechSupported);
  
  // Ref for auto-scrolling to bottom
  const messagesEndRef = useRef(null);

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
      // Clear skeleton timeout
      if (skeletonTimeoutRef.current) {
        clearTimeout(skeletonTimeoutRef.current);
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

  // Refs for scroll control
  const [userScrolling, setUserScrolling] = useState(false);
  const [inputFocused, setInputFocused] = useState(false);
  const [sendingMessage, setSendingMessage] = useState(false);
  const [showLoadingSkeleton, setShowLoadingSkeleton] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  
  // Ref to store skeleton timeout
  const skeletonTimeoutRef = useRef(null);
  const scrollTimeoutRef = useRef(null);

  // Detect user scrolling behavior
  useEffect(() => {
    const handleScroll = (e) => {
      setUserScrolling(true);
      
      // Check if user is at the bottom and show/hide scroll button
      const chatContainer = e.target.closest('.chatbot-messages') || e.target;
      if (chatContainer && chatContainer.scrollHeight) {
        const isAtBottom = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight < 50;
        setShowScrollButton(!isAtBottom);
      }
      
      // Clear existing timeout
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
      
      // Reset user scrolling flag after shorter time during AI responses
      const isAIResponding = loading || isTyping || messages.some(msg => msg.streaming);
      const timeoutDuration = isAIResponding ? 1000 : 2000; // 1 second during AI responses, 2 seconds normally
      
      scrollTimeoutRef.current = setTimeout(() => {
        setUserScrolling(false);
      }, timeoutDuration);
    };

    // Add scroll listener to chat container and window
    const chatContainer = document.querySelector('.chatbot-messages');
    if (chatContainer) {
      chatContainer.addEventListener('scroll', handleScroll, { passive: true });
    }
    
    window.addEventListener('scroll', handleScroll, { passive: true });

    return () => {
      if (chatContainer) {
        chatContainer.removeEventListener('scroll', handleScroll);
      }
      window.removeEventListener('scroll', handleScroll);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);

  // Auto-scroll to bottom when new messages arrive - more aggressive during AI responses
  useEffect(() => {
    if (messagesEndRef.current) {
      // Get the chat messages container
      const chatContainer = messagesEndRef.current.closest('.chatbot-messages');
      if (chatContainer) {
        // During AI responses (loading or streaming), always scroll to bottom unless user is actively scrolling
        const isAIResponding = loading || isTyping || messages.some(msg => msg.streaming);
        
        if (isAIResponding && !userScrolling) {
          // During AI response, always scroll to bottom regardless of position
          setTimeout(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
          }, 50); // Faster scroll during AI responses
        } else if (!userScrolling && !inputFocused && !sendingMessage) {
          // For regular messages, check if user is near the bottom
          const isNearBottom = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight < 100;
          
          if (isNearBottom) {
            setTimeout(() => {
              chatContainer.scrollTop = chatContainer.scrollHeight;
            }, 100);
          }
        }
      }
    }
  }, [messages.length, messages, loading, isTyping, userScrolling, inputFocused, sendingMessage]); // Added messages array and AI states

  // Check scroll position to show/hide scroll button
  useEffect(() => {
    const checkScrollPosition = () => {
      if (messagesEndRef.current) {
        const chatContainer = messagesEndRef.current.closest('.chatbot-messages');
        if (chatContainer && chatContainer.scrollHeight) {
          const isAtBottom = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight < 50;
          setShowScrollButton(!isAtBottom && messages.length > 0);
        }
      }
    };

    // Check immediately and after a short delay to account for rendering
    checkScrollPosition();
    const timeoutId = setTimeout(checkScrollPosition, 100);

    return () => clearTimeout(timeoutId);
  }, [messages]);

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

  // Enhanced function to scroll to the bottom/newest messages
  const scrollToBottom = (smooth = true) => {
    if (messagesEndRef.current) {
      const chatContainer = messagesEndRef.current.closest('.chatbot-messages');
      if (chatContainer) {
        if (isMobile) {
          // Mobile-optimized scrolling
          chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: smooth ? 'smooth' : 'auto'
          });
        } else {
          // Desktop scrolling
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        setShowScrollButton(false);
      }
    }
  };

  // Mobile-specific auto-scroll behavior
  const autoScrollToBottom = () => {
    if (isMobile && !userScrolling) {
      // On mobile, always auto-scroll during AI responses
      setTimeout(() => scrollToBottom(true), 100);
    } else if (!isMobile) {
      // Desktop behavior - scroll only if near bottom
      scrollToBottom(false);
    }
  };

  const handleSend = async (customInput = null) => {
    const userInput = customInput || input.trim();
    
    // Prevent sending if already loading
    if (loading) {
      return;
    }
    
    // Set sending flag to prevent auto-scroll during message sending
    setSendingMessage(true);
    
    // Record user interaction
    recordUserInteraction('message_sent', { messageLength: userInput.length });
    
    // Track chat message with Vercel Analytics
    trackChatEvent('user_message', userInput);
    
    // Enhanced input validation
    if (!validateInput(userInput)) {
      setSendingMessage(false);
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
    setIsTyping(true);
    setInputError('');
    setShowLoadingSkeleton(false); // Reset skeleton state

    // Start performance measurement
    const startTime = Date.now();

    // Show loading skeleton after a short delay to avoid flickering for fast responses
    skeletonTimeoutRef.current = setTimeout(() => {
      if (loading) {
        setShowLoadingSkeleton(true);
      }
    }, 300); // Show skeleton only if response takes more than 300ms

    // Start streaming response immediately
    let streamedContent = '';
    
    try {
      await fetchStreamingResults(userInput, (chunk) => {
        streamedContent += chunk;
        
        // Hide loading indicator as soon as we get the first chunk
        if (streamedContent.length > 0 && loading) {
          setLoading(false);
          setShowLoadingSkeleton(false);
          if (skeletonTimeoutRef.current) {
            clearTimeout(skeletonTimeoutRef.current); // Clear skeleton timeout
            skeletonTimeoutRef.current = null;
          }
        }
        
        // Show typing indicator for the first chunk, then switch to content
        if (streamedContent.length > 0 && isTyping) {
          setIsTyping(false);
        }
        
        // Update messages in real-time as chunks come in
        setMessages((prevMessages) => {
          const lastMessage = prevMessages[prevMessages.length - 1];
          
          // If last message is assistant and streaming, update it
          if (lastMessage && lastMessage.role === 'assistant' && lastMessage.streaming) {
            const updatedMessages = [
              ...prevMessages.slice(0, -1),
              { 
                role: 'assistant', 
                content: streamedContent, 
                streaming: true,
                isTyping: streamedContent.length < 50 // Show typing animation for short responses
              }
            ];
            
            // Auto-scroll during streaming if user is not actively scrolling
            if (!userScrolling && messagesEndRef.current) {
              setTimeout(() => {
                const chatContainer = messagesEndRef.current?.closest('.chatbot-messages');
                if (chatContainer) {
                  chatContainer.scrollTop = chatContainer.scrollHeight;
                }
              }, 10); // Very fast scroll during streaming
            }
            
            return updatedMessages;
          } else {
            // Add new assistant message
            const newMessages = [
              ...prevMessages,
              { 
                role: 'assistant', 
                content: streamedContent, 
                streaming: true,
                isTyping: streamedContent.length < 50
              }
            ];
            
            // Auto-scroll for new streaming message
            if (!userScrolling && messagesEndRef.current) {
              setTimeout(() => {
                const chatContainer = messagesEndRef.current?.closest('.chatbot-messages');
                if (chatContainer) {
                  chatContainer.scrollTop = chatContainer.scrollHeight;
                }
              }, 10);
            }
            
            return newMessages;
          }
        });
      });
      
      // Record successful API response time
      measureApiResponseTime(Date.now() - startTime);
      
    } catch (error) {
      const errorMessage = error.message?.includes('network') 
        ? 'Network error. Please check your connection and try again.'
        : 'Sorry, I encountered an error. Please try rephrasing your question.';
      
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: errorMessage }
      ]);
      
      // Record error
      recordUserInteraction('api_error', { error: error.message });
      
    } finally {
      setLoading(false);
      setIsTyping(false);
      setSendingMessage(false); // Reset sending flag
      setShowLoadingSkeleton(false); // Hide skeleton
      if (skeletonTimeoutRef.current) {
        clearTimeout(skeletonTimeoutRef.current); // Clear skeleton timeout
        skeletonTimeoutRef.current = null;
      }
      
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
      
      // Note: Auto-scroll is now handled by the useEffect hook above
      // which respects user scrolling behavior
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
    if (!speechSynthesisSupported || !speechSynthesis.current) {
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

  // Like/Unlike message functionality
  const toggleMessageLike = async (messageIndex, isLiked) => {
    const newLikedMessages = new Set(likedMessages);
    const newDislikedMessages = new Set(dislikedMessages);
    const messageId = `${currentSessionId}-${messageIndex}`;
    
    if (isLiked) {
      newLikedMessages.add(messageId);
      // Remove from dislikes if it was disliked
      newDislikedMessages.delete(messageId);
    } else {
      newLikedMessages.delete(messageId);
    }
    
    setLikedMessages(newLikedMessages);
    setDislikedMessages(newDislikedMessages);
    
    // Save feedback to backend
    await saveFeedback(messageIndex, isLiked ? 'like' : 'unlike', messages[messageIndex]?.content);
    
    // Save the chat session when a message is liked
    if (isLiked && currentSessionId) {
      await saveChatSession(currentSessionId, messages);
    }
  };

  // Dislike message functionality
  const toggleMessageDislike = async (messageIndex, isDisliked) => {
    const newLikedMessages = new Set(likedMessages);
    const newDislikedMessages = new Set(dislikedMessages);
    const messageId = `${currentSessionId}-${messageIndex}`;
    
    if (isDisliked) {
      newDislikedMessages.add(messageId);
      // Remove from likes if it was liked
      newLikedMessages.delete(messageId);
    } else {
      newDislikedMessages.delete(messageId);
    }
    
    setLikedMessages(newLikedMessages);
    setDislikedMessages(newDislikedMessages);
    
    // Save feedback to backend
    await saveFeedback(messageIndex, isDisliked ? 'dislike' : 'undislike', messages[messageIndex]?.content);
  };

  // Save chat session to backend
  const saveChatSession = async (sessionId, messagesData) => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/chat-sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          messages: messagesData,
          timestamp: new Date().toISOString(),
        }),
      });
      
      if (!response.ok) {
        console.error('Failed to save chat session');
      }
    } catch (error) {
      console.error('Error saving chat session:', error);
    }
  };

  // Save feedback to backend
  const saveFeedback = async (messageIndex, feedbackType, messageContent) => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: currentSessionId,
          message_index: messageIndex,
          feedback_type: feedbackType,
          message_content: messageContent,
          timestamp: new Date().toISOString(),
        }),
      });
      
      if (!response.ok) {
        console.error('Failed to save feedback');
      }
    } catch (error) {
      console.error('Error saving feedback:', error);
    }
  };

  // Load saved sessions
  const loadSavedSessions = async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/chat-sessions`);
      if (response.ok) {
        const data = await response.json();
        setSavedSessions(data.sessions || []);
      }
    } catch (error) {
      console.error('Error loading saved sessions:', error);
    }
  };

  // Load saved session details and display them
  const loadSavedSession = async (sessionId) => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/chat-sessions/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data.session.messages);
        setCurrentSessionId(sessionId);
        setSidebarOpen(false); // Close sidebar on mobile
      }
    } catch (error) {
      console.error('Error loading saved session:', error);
    }
  };

  // Delete a saved session
  const deleteSavedSession = async (sessionId) => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/chat-sessions/${sessionId}`, {
        method: 'DELETE'
      });
      if (response.ok) {
        // Refresh saved sessions list
        loadSavedSessions();
      }
    } catch (error) {
      console.error('Error deleting saved session:', error);
    }
  };

  // Load saved sessions on component mount
  useEffect(() => {
    loadSavedSessions();
  }, []);

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

  // Mobile-specific state and behavior
  const [isMobile, setIsMobile] = useState(false);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  
  // Mobile detection and keyboard handling
  useEffect(() => {
    const checkIfMobile = () => {
      const isMobileDevice = window.innerWidth <= 768 || /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      setIsMobile(isMobileDevice);
    };

    const handleResize = () => {
      checkIfMobile();
      // Detect virtual keyboard on mobile
      if (isMobile && inputFocused) {
        const heightChange = window.innerHeight < window.screen.height * 0.75;
        setKeyboardVisible(heightChange);
      }
    };

    const handleScroll = () => {
      if (messagesEndRef.current) {
        const chatContainer = messagesEndRef.current.closest('.chatbot-messages');
        if (chatContainer) {
          const isNearBottom = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight < 100;
          setShowScrollButton(!isNearBottom && messages.length > 0);
          
          // Detect user scrolling to prevent auto-scroll interruption
          setUserScrolling(true);
          clearTimeout(window.scrollTimeout);
          window.scrollTimeout = setTimeout(() => setUserScrolling(false), 1000);
        }
      }
    };

    checkIfMobile();
    window.addEventListener('resize', handleResize);
    
    // Add scroll listener for mobile scroll behavior
    const chatContainer = document.querySelector('.chatbot-messages');
    if (chatContainer) {
      chatContainer.addEventListener('scroll', handleScroll);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chatContainer) {
        chatContainer.removeEventListener('scroll', handleScroll);
      }
      if (window.scrollTimeout) {
        clearTimeout(window.scrollTimeout);
      }
    };
  }, [isMobile, inputFocused, messages.length]);

  // Handle keyboard-visible class for mobile optimization
  useEffect(() => {
    if (isMobile) {
      if (keyboardVisible) {
        document.body.classList.add('keyboard-visible');
      } else {
        document.body.classList.remove('keyboard-visible');
      }
    }
    
    return () => {
      document.body.classList.remove('keyboard-visible');
    };
  }, [keyboardVisible, isMobile]);

  // Enhanced mobile scroll management
  useEffect(() => {
    if (messagesEndRef.current) {
      const chatContainer = messagesEndRef.current.closest('.chatbot-messages');
      if (chatContainer) {
        // During AI responses, always scroll to bottom unless user is actively scrolling
        const isAIResponding = loading || isTyping || messages.some(msg => msg.streaming);
        
        if (isAIResponding && !userScrolling) {
          // During AI response, always scroll to bottom regardless of position
          setTimeout(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
          }, 50); // Faster scroll during AI responses
        } else if (!userScrolling && !inputFocused && !sendingMessage) {
          // For regular messages, check if user is near the bottom
          const isNearBottom = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight < 100;
          
          if (isNearBottom) {
            setTimeout(() => {
              chatContainer.scrollTop = chatContainer.scrollHeight;
            }, 100);
          }
        }
      }
    }
  }, [messages.length, messages, loading, isTyping, userScrolling, inputFocused, sendingMessage]); // Added messages array and AI states

  return (
    <div className="chatbot-page">
      {/* Mobile Optimization Component */}
      <MobileOptimizer />
      
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
                  border: '1px solid rgba(139, 92, 246, 0.3)',
                  color: '#e5e7eb'
                }}
                onMouseEnter={(e) => {
                  e.target.style.transform = 'scale(1.05)';
                }}
                onMouseLeave={(e) => {
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
                border: '1px solid rgba(139, 92, 246, 0.5)',
                width: 'auto', // Auto width instead of full width
                minWidth: '120px', // Minimum width for usability
                maxWidth: '160px' // Maximum width to keep it compact
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px) scale(1.02)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0) scale(1)';
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
                      border: `1px solid ${currentSessionId === session.id ? 'rgba(139, 92, 246, 0.5)' : 'rgba(139, 92, 246, 0.1)'}`,
                      borderRadius: '12px',
                      padding: '16px'
                    }}
                    onClick={() => loadChatSession(session.id)}
                    onMouseEnter={(e) => {
                      if (currentSessionId !== session.id) {
                        e.target.style.borderColor = 'rgba(139, 92, 246, 0.3)';
                        e.target.style.transform = 'translateY(-1px)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (currentSessionId !== session.id) {
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
                          border: '1px solid rgba(239, 68, 68, 0.3)',
                          color: '#ef4444'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.transform = 'scale(1.05)';
                        }}
                        onMouseLeave={(e) => {
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
            
            {/* Saved Sessions Section */}
            {savedSessions.length > 0 && (
              <>
                <div className="mt-8 mb-4 px-2">
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.69 4.5 1.79C13.09 3.69 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                    </svg>
                    <h3 className="text-sm font-semibold text-gray-200">Liked Sessions</h3>
                  </div>
                  <div className="h-px bg-gradient-to-r from-transparent via-purple-500/30 to-transparent mt-2"></div>
                </div>
                
                <div className="space-y-2">
                  {savedSessions.map((session) => (
                    <div
                      key={session.id}
                      className="group relative cursor-pointer transition-all duration-200"
                      style={{
                        border: '1px solid rgba(234, 179, 8, 0.2)',
                        borderRadius: '12px',
                        padding: '16px',
                        background: 'rgba(234, 179, 8, 0.05)'
                      }}
                      onClick={() => loadSavedSession(session.id)}
                      onMouseEnter={(e) => {
                        e.target.style.borderColor = 'rgba(234, 179, 8, 0.4)';
                        e.target.style.transform = 'translateY(-1px)';
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.borderColor = 'rgba(234, 179, 8, 0.2)';
                        e.target.style.transform = 'translateY(0)';
                      }}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <svg className="w-3 h-3 text-yellow-400 flex-shrink-0" fill="currentColor" viewBox="0 0 24 24">
                              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.69 4.5 1.79C13.09 3.69 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                            </svg>
                            <p className="text-sm font-semibold truncate text-yellow-100">{session.title}</p>
                          </div>
                          <p className="text-xs mt-1 text-yellow-200/60">{formatSessionDate(session.saved_at)}</p>
                          <p className="text-xs text-yellow-200/40">{session.message_count} messages</p>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteSavedSession(session.id);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-2 rounded-lg transition-all duration-200 ml-2"
                          style={{
                            border: '1px solid rgba(239, 68, 68, 0.3)',
                            color: '#ef4444'
                          }}
                          onMouseEnter={(e) => {
                            e.target.style.transform = 'scale(1.05)';
                          }}
                          onMouseLeave={(e) => {
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
              </>
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
          border: sidebarOpen 
            ? '1px solid rgba(139, 92, 246, 0.4)' 
            : '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '12px',
          padding: '10px',
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
          e.target.style.borderColor = 'rgba(139, 92, 246, 0.4)';
          e.target.style.color = '#8b5cf6';
        }}
        onMouseLeave={(e) => {
          e.target.style.transform = 'translateY(0) scale(1)';
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
              : "M4 6h16M4 12h16M4 18h16"
            } 
          />
        </svg>
      </button>

      {/* Main Chat Container with Purple Outline */}
      <div className={`chatbot-main-container transition-all duration-300 ${
        sidebarOpen ? 'md:ml-80 ml-0' : 'ml-0'
      }`}>
        <div className="chatbot-purple-box">
          


          {/* Chat Messages Area - Separate from Input with Scrolling */}
          <div className="chatbot-messages chatbot-scrollable">
          
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
              
              {/* Quick suggestions moved higher - inside welcome screen */}
              <div className="kam-quick-suggestions">
                <div className="text-xs text-gray-400 mb-3">Try asking about:</div>
                <div className="flex flex-wrap gap-2 justify-center">
                  {[
                    "Best restaurants in Sultanahmet",
                    "Things to do in Beyoğlu", 
                    "Ferry routes to the islands",
                    "Turkish breakfast spots"
                  ].map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => handleSend(suggestion)}
                      className="kam-suggestion-pill"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
              
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
                      {msg.role === 'assistant' && msg.isTyping ? (
                        <StreamingText 
                          text={msg.content} 
                          speed={30}
                          enableStreamingGlow={true}
                          onComplete={() => {
                            // Mark typing as complete for this message
                            setMessages(prev => prev.map((m, i) => 
                              i === index ? { ...m, isTyping: false } : m
                            ));
                          }}
                        />
                      ) : (
                        renderMessageContent(msg.content, darkMode)
                      )}
                    </div>
                  </div>
                </div>
                
                {/* Action Buttons - Outside message bubble like ChatGPT - For both user and assistant messages */}
                <div className={`kam-message-actions flex ${
                  msg.role === 'user' ? 'justify-end' : 'justify-start'
                }`}>
                  
                  {/* Like/Unlike Buttons - Only for assistant messages */}
                  {msg.role === 'assistant' && (
                    <>
                      <button
                        onClick={() => toggleMessageLike(index, !likedMessages.has(`${currentSessionId}-${index}`))}
                        className={`kam-action-button kam-like-button ${
                          likedMessages.has(`${currentSessionId}-${index}`) ? 'active' : ''
                        }`}
                        title={likedMessages.has(`${currentSessionId}-${index}`) ? "Unlike this response" : "Like this response"}
                      >
                        <svg className="w-3 h-3" fill={likedMessages.has(`${currentSessionId}-${index}`) ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                        <span className="text-xs">
                          {likedMessages.has(`${currentSessionId}-${index}`) ? 'Unlike' : 'Like'}
                        </span>
                      </button>
                      
                      <button
                        onClick={() => toggleMessageDislike(index, !dislikedMessages.has(`${currentSessionId}-${index}`))}
                        className={`kam-action-button kam-dislike-button ${
                          dislikedMessages.has(`${currentSessionId}-${index}`) ? 'active' : ''
                        }`}
                        title={dislikedMessages.has(`${currentSessionId}-${index}`) ? "Remove dislike" : "Dislike this response"}
                      >
                        <svg className="w-3 h-3" fill={dislikedMessages.has(`${currentSessionId}-${index}`) ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24" transform="rotate(180)">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                        <span className="text-xs">
                          {dislikedMessages.has(`${currentSessionId}-${index}`) ? 'Undislike' : 'Dislike'}
                        </span>
                      </button>
                    </>
                  )}
                  
                  {/* Copy Button */}
                  <button
                    onClick={() => copyToClipboard(msg.content, index)}
                    className="kam-action-button kam-copy-button"
                    title="Copy message"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    <span className="text-xs">Copy</span>
                  </button>
                  
                  {/* Read Aloud Button */}
                  {speechSynthesisSupported && (
                    <button
                      onClick={() => readingMessageId === index ? stopReading() : readAloud(msg.content, index)}
                      className={`kam-action-button kam-read-button ${readingMessageId === index ? 'active' : ''}`}
                      title={readingMessageId === index ? "Stop reading" : "Read aloud"}
                    >
                      {readingMessageId === index ? (
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-6.219-8.56" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10l2 2 4-4" />
                        </svg>
                      ) : (
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 9H4a1 1 0 00-1 1v4a1 1 0 001 1h1.586l4.707 4.707C10.923 20.337 12 19.575 12 18.586V5.414c0-.989-1.077-1.751-1.707-1.121L5.586 9z" />
                        </svg>
                      )}
                      <span className="text-xs">
                        {readingMessageId === index ? 'Stop' : 'Read'}
                      </span>
                    </button>
                  )}
                </div>
              </div>
            ))}
            
            {showLoadingSkeleton && loading && (
              <div className="mb-4">
                <div className="flex justify-start">
                  <div className="text-xs font-medium mb-1 text-gray-100">
                    KAM
                  </div>
                </div>
                <div className="flex justify-start">
                  <div className="max-w-[80%] bg-gray-800 text-gray-100 rounded-2xl rounded-bl-md px-4 py-3 border border-gray-700">
                    <ChatMessageSkeleton variant="enhanced" />
                  </div>
                </div>
              </div>
            )}
            
            {/* Mobile-optimized Scroll to Bottom Button */}
            {showScrollButton && (
              <div className={`fixed z-10 transition-all duration-200 ${
                isMobile 
                  ? 'bottom-24 right-4' // Mobile positioning - above input area
                  : 'bottom-24 right-6' // Desktop positioning
              }`}>
                <button
                  onClick={() => scrollToBottom(true)}
                  className={`${
                    isMobile
                      ? 'bg-blue-600 hover:bg-blue-700 text-white rounded-full p-3 shadow-lg transition-all duration-200 flex items-center justify-center min-h-12 min-w-12'
                      : 'bg-purple-600 hover:bg-purple-700 text-white rounded-full p-3 shadow-lg transition-all duration-200 flex items-center justify-center'
                  }`}
                  title="Go to newest messages"
                  style={{
                    transform: isMobile ? 'scale(1.1)' : 'scale(1)', // Slightly larger on mobile
                    boxShadow: isMobile ? '0 8px 25px rgba(59, 130, 246, 0.3)' : undefined
                  }}
                >
                  <svg 
                    className={isMobile ? "w-6 h-6" : "w-5 h-5"} 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M19 14l-7 7m0 0l-7-7m7 7V3" 
                    />
                  </svg>
                </button>
              </div>
            )}

            
            <div ref={messagesEndRef} />
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
                onFocus={(e) => {
                  setInputFocused(true);
                  
                  // Mobile-specific focus handling
                  if (isMobile) {
                    // Scroll to bottom when input is focused on mobile
                    setTimeout(() => {
                      if (messagesEndRef.current) {
                        messagesEndRef.current.scrollIntoView({ 
                          behavior: 'smooth', 
                          block: 'end' 
                        });
                      }
                    }, 300); // Delay to account for keyboard animation
                    
                    // Track keyboard state
                    setTimeout(() => setKeyboardVisible(true), 300);
                  } else {
                    // Desktop focus behavior (prevent scroll)
                    e.preventDefault();
                    const currentScrollY = window.scrollY;
                    const currentScrollTop = e.target.closest('.chatbot-messages')?.scrollTop || 0;
                    setTimeout(() => {
                      window.scrollTo(0, currentScrollY);
                      const chatContainer = e.target.closest('.chatbot-messages');
                      if (chatContainer) {
                        chatContainer.scrollTop = currentScrollTop;
                      }
                    }, 0);
                  }
                }}
                onBlur={() => {
                  setInputFocused(false);
                  if (isMobile) {
                    setTimeout(() => setKeyboardVisible(false), 300);
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !loading) {
                    e.preventDefault();
                    if (input.trim()) {
                      handleSend();
                      // On mobile, blur input after sending to hide keyboard
                      if (isMobile) {
                        e.target.blur();
                      }
                    }
                  }
                }}
                placeholder={isMobile ? "Ask about Istanbul..." : "What would you like to know about Istanbul?"}
                className="kam-input-field"
                disabled={loading}
                autoComplete="off"
                autoFocus={false}
                inputMode="text"
                autoCapitalize="sentences"
                autoCorrect="on"
                spellCheck="true"
              />
              
              {/* Enhanced Send Button */}
              <button 
                onClick={() => {
                  if (!loading && input.trim()) {
                    handleSend();
                    // On mobile, blur input after sending to hide keyboard
                    if (isMobile) {
                      const inputElement = document.querySelector('.kam-input-field');
                      if (inputElement) inputElement.blur();
                    }
                  }
                }} 
                disabled={loading || !input.trim()}
                className={`kam-send-button ${loading ? 'send-button-loading' : ''} ${isMobile ? 'mobile-optimized' : ''}`}
                aria-label="Send message"
                type="button"
              >
                {loading ? (
                  <LoadingSpinner variant="spinner" size="medium" />
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                )}
              </button>
            </div>
          </div>
          
          {/* Quick suggestions (optional) - removed from here as moved to welcome screen */}
          </div>
        </div>
        </div>
      </div>
    </div>
  );
}

export default Chatbot;
