import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import SearchBar from './components/SearchBar';
import Chat from './components/Chat';
import ResultCard from './components/ResultCard';
// import DebugInfo from './components/DebugInfo';
import { fetchResults, fetchStreamingResults, getSessionId } from './api/api';
import GoogleAnalytics, { trackChatEvent, trackEvent } from './utils/analytics';
import './App.css';

const App = () => {
  const location = useLocation();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [messages, setMessages] = useState(() => {
    // Load saved messages from localStorage
    try {
      const saved = localStorage.getItem('chat-messages');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [expanded, setExpanded] = useState(() => {
    // If accessed via /chat route, auto-expand
    if (window.location.pathname === '/chat') {
      return true;
    }
    // If there are saved messages, start in expanded mode
    try {
      const saved = localStorage.getItem('chat-messages');
      return saved ? JSON.parse(saved).length > 0 : false;
    } catch {
      return false;
    }
  });
  const [sessionId] = useState(() => getSessionId()); // Get persistent session ID
  const chatScrollRef = useRef(null);

  // Auto-expand when navigating to /chat route
  useEffect(() => {
    if (location.pathname === '/chat') {
      setExpanded(true);
    }
  }, [location.pathname]);

  // Add/remove main-page class on body
  useEffect(() => {
    if (location.pathname === '/') {
      document.body.classList.add('main-page');
    } else {
      document.body.classList.remove('main-page');
    }
    
    // Cleanup on unmount
    return () => {
      document.body.classList.remove('main-page');
    };
  }, [location.pathname]);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('chat-messages', JSON.stringify(messages));
    }
  }, [messages]);

  // Notify AppRouter when chat state changes
  useEffect(() => {
    // Dispatch custom event when expanded state changes
    window.dispatchEvent(new CustomEvent('chatStateChanged', { 
      detail: { expanded, hasMessages: messages.length > 0 } 
    }));
  }, [expanded, messages.length]);

  useEffect(() => {
    // Audio file temporarily disabled - uncomment when audio file is available
    // const audio = new Audio('/welcome_baskan.mp3');
    // audio.volume = 0.5;
    // audio.play().catch(() => {/* ignore audio errors */});
  }, []);

  useEffect(() => {
    if (expanded && chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages, expanded]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    // Track the search event
    trackChatEvent('search_initiated', query);
    
    setMessages([...messages, { user: 'You', text: query }]);
    setExpanded(true);
    const searchQuery = query;
    setQuery('');
    setTimeout(() => {
      document.getElementById('chat-animated-container')?.classList.add('expand-animate');
    }, 10);
    // Streaming KAM response
    let aiMessage = '';
    setMessages(msgs => [...msgs, { user: 'KAM', text: '' }]);
    try {
      await fetchStreamingResults(searchQuery, (chunk) => {
        aiMessage += chunk;
        setMessages(msgs => {
          const updated = [...msgs];
          // Find last KAM message and update its text
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].user === 'KAM') {
              updated[i] = { ...updated[i], text: aiMessage };
              break;
            }
          }
          return updated;
        });
      }, sessionId, (error) => {
        // Handle streaming errors
        console.error('Streaming error:', error);
        setMessages(msgs => {
          const updated = [...msgs];
          // Find last KAM message and set error text
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].user === 'KAM') {
              updated[i] = { ...updated[i], text: 'Sorry, I encountered an error. Please try again.' };
              break;
            }
          }
          return updated;
        });
      }); // Pass sessionId for chat history
      
      // Track successful response completion
      trackChatEvent('response_completed', aiMessage.substring(0, 50));
      
    } catch (err) {
      console.error('API Error:', err);
      
      // Track error
      trackEvent('api_error', 'error', err.message);
      
      setMessages(msgs => [...msgs, {
        user: 'KAM',
        text: `Sorry, I encountered an error connecting to the server: ${err.message}. Please make sure the backend is running and try again.`
      }]);
      setResults([]);
    }
  };

  const handleLogoClick = () => {
    // Track navigation back to home
    trackEvent('logo_click', 'navigation', 'home');
    
    // Reset chat state to go back to homepage
    setExpanded(false);
    setMessages([]);
    setResults([]);
    setQuery('');
    
    // Clear session storage to start fresh conversation
    localStorage.removeItem('chat_session_id');
    localStorage.removeItem('chat-messages');
  };

  return (
    <div style={{ width: '100vw', height: '100vh', minHeight: '100vh', background: 'none', display: 'flex', flexDirection: 'column' }}>
      <GoogleAnalytics />
      {/* <DebugInfo /> */}

      {!expanded ? (
        <div className="main-page-background" style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', width: '100vw', height: '100vh', paddingTop: '12rem'}}>
          {/* Centered logo - bigger than navbar logo */}
          <div style={{textAlign: 'center', marginBottom: '3rem'}} onClick={handleLogoClick}>
            <div className="chat-title logo-istanbul main-page-logo">
              <span className="logo-text" style={{
                fontSize: window.innerWidth < 768 ? '3rem' : '5rem', // Increased from 2.5rem/4rem
                fontWeight: 700,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)', // Same as other pages
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                textShadow: '0 4px 20px rgba(99, 102, 241, 0.8)', // Reduced shadow
                transition: 'all 0.3s ease',
                cursor: 'pointer'
              }}>
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </div>
          <div style={{width: '100%', maxWidth: 1200, minWidth: 320, margin: '0 auto', padding: '1rem'}}>
            <SearchBar
              value={query}
              onChange={e => setQuery(e.target.value)}
              onSubmit={handleSearch}
              placeholder="Welcome to Istanbul!"
            />
          </div>
        </div>
      ) : (
        <>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', transition: 'all 0.4s', height: '100vh', paddingTop: '6rem', paddingBottom: '2rem' }}>
            {/* Logo for chatbot view */}
            <div style={{ textAlign: 'center', marginBottom: '1.5rem', cursor: 'pointer' }} onClick={handleLogoClick}>
              <div className="chat-title logo-istanbul">
                <span className="logo-text" style={{
                  fontSize: window.innerWidth < 768 ? '2.5rem' : '3.5rem', // Smaller than main page but visible
                  fontWeight: 700,
                  letterSpacing: '0.12em',
                  textTransform: 'uppercase',
                  background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  textShadow: '0 2px 10px rgba(99, 102, 241, 0.3)',
                  transition: 'all 0.3s ease'
                }}>
                  A/<span style={{fontWeight: 400}}>STANBUL</span>
                </span>
              </div>
            </div>
            <div style={{ width: '100%', maxWidth: 950, flex: 1, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 12rem)', minHeight: '400px' }}>
              {/* Unified chat area and search bar */}
              <div className="chat-container" style={{display: 'flex', flexDirection: 'column', height: '100%', background: 'none', borderRadius: '1.5rem', boxShadow: '0 4px 24px 0 rgba(20, 20, 40, 0.18)', position: 'relative'}}>
                <div ref={chatScrollRef} className="chat-scroll-area" style={{flex: 1, overflowY: 'scroll', overflowX: 'hidden', marginBottom: '1rem', paddingBottom: '0.5rem', minHeight: 0, paddingTop: '0.5rem'}}>
                  <Chat messages={messages} />
                  {/* Remove or reduce margin below chat */}
                  <div style={{ marginTop: '0.5rem' }}>
                    {results.map((res, idx) => (
                      <ResultCard key={idx} title={res.title} description={res.description} />
                    ))}
                  </div>
                </div>
                <div style={{position: 'relative', bottom: 'auto', left: '0', right: '0', background: 'transparent', padding: '0.5rem 0.5rem 1rem 0.5rem', zIndex: 15}}>
                  <SearchBar
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onSubmit={handleSearch}
                    placeholder=""
                  />
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default App;
