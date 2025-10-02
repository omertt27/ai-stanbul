import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import SearchBar from './components/SearchBar';
import ResultCard from './components/ResultCard';
import InteractiveMainPage from './components/InteractiveMainPage';
import WeatherThemeProvider from './components/WeatherThemeProvider';
import CookieConsent from './components/CookieConsent';
import NavBar from './components/NavBar';
import LanguageSwitcher from './components/LanguageSwitcher';
import MainPageMobileNavbar from './components/MainPageMobileNavbar';
import { useMobileUtils, InstallPWAButton, MobileSwipe } from './hooks/useMobileUtils.jsx';
import { fetchResults, fetchStreamingResults, getSessionId } from './api/api';
import GoogleAnalytics, { trackChatEvent, trackEvent } from './utils/analytics';
import './App.css';
import './components/InteractiveMainPage.css';

const App = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { t, i18n } = useTranslation();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
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

  // Mobile utilities hook
  const { 
    isMobile, 
    isTouch, 
    orientation, 
    hapticFeedback, 
    handleTouchStart, 
    handleTouchEnd 
  } = useMobileUtils();

  // Auto-expand when navigating to /chat route
  useEffect(() => {
    // Ensure page stays at top when component mounts
    window.scrollTo(0, 0);
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
    
    setSearchLoading(true);
    
    // Track the search event
    trackChatEvent('search_initiated', query);
    
    // Store the query and navigate to chat page for consistent experience
    localStorage.setItem('pending_chat_query', query);
    navigate('/chat');
    
    // Reset loading state after navigation
    setTimeout(() => setSearchLoading(false), 1000);
  };

  const handleQuickStart = (quickQuery) => {
    setQuery(quickQuery);
    // Store the query for the chat page
    localStorage.setItem('pending_chat_query', quickQuery);
  };

  const handleLogoClick = () => {
    // Track navigation back to home
    trackEvent('logo_click', 'navigation', 'home');
    
    // Navigate to home page
    navigate('/');
    
    // Reset chat state to go back to homepage
    setExpanded(false);
    setMessages([]);
    setResults([]);
    setQuery('');
    
    // Clear session storage to start fresh conversation
    localStorage.removeItem('chat_session_id');
    localStorage.removeItem('chat-messages');
  };

  const handleLanguageChange = (languageCode) => {
    i18n.changeLanguage(languageCode);
    
    // Handle RTL for Arabic
    if (languageCode === 'ar') {
      document.documentElement.setAttribute('dir', 'rtl');
      document.documentElement.setAttribute('lang', 'ar');
    } else {
      document.documentElement.setAttribute('dir', 'ltr');
      document.documentElement.setAttribute('lang', languageCode);
    }
  };

  return (
    <WeatherThemeProvider>
      <div style={{ width: '100vw', minHeight: '100vh', background: 'none', display: 'flex', flexDirection: 'column' }}>
        <GoogleAnalytics />
        {/* <DebugInfo /> */}

        {/* Mobile Top Navbar for main page only */}
        <MainPageMobileNavbar />

        <div className="main-page-background main-to-chat-transition" style={{
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'flex-start',
          width: '100vw', 
          minHeight: '100vh', 
          paddingTop: isMobile || window.innerWidth <= 768 ? '80px' : '6rem', // Account for mobile navbar
          paddingBottom: '2rem',
          paddingLeft: isMobile || window.innerWidth <= 768 ? '1rem' : '2rem',
          paddingRight: isMobile || window.innerWidth <= 768 ? '1rem' : '2rem'
        }}>
          
          {/* Centered logo - Only show on desktop, mobile has navbar logo */}
          {!(isMobile || window.innerWidth <= 768) && (
            <div 
              style={{
                textAlign: 'center', 
                marginBottom: '1rem',
                marginTop: '15px', // Move logo 15px more down for desktop version
              }} 
              onClick={handleLogoClick}
            >
              <div className="chat-title logo-istanbul main-page-logo">
                <span className="logo-text" style={{
                  fontSize: '4rem',
                  fontWeight: 700,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  color: 'transparent',
                  background: 'linear-gradient(90deg, #e5e7eb 0%, #8b5cf6 50%, #6366f1 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  textShadow: '0 2px 10px rgba(139, 92, 246, 0.3)',
                  transition: 'all 0.3s ease',
                  cursor: 'pointer',
                  lineHeight: '1.2'
                }}>
                  A/STANBUL
                </span>
              </div>
            </div>
          )}
          
          {/* Simple search bar - ChatGPT style */}
          <div style={{
            width: '100%', 
            maxWidth: isMobile || window.innerWidth <= 768 ? '100%' : 1200, 
            margin: isMobile || window.innerWidth <= 768 ? '6rem auto 2rem auto' : '0 auto 2rem auto', 
            padding: isMobile || window.innerWidth <= 768 ? '0 0.5rem' : '1rem',
            zIndex: 10,
          }}>
            <SearchBar
              value={query}
              onChange={e => setQuery(e.target.value)}
              onSubmit={handleSearch}
              placeholder={t('chat.searchPlaceholder')}
              isLoading={searchLoading}
              expanded={expanded}
            />
          </div>
          
          {/* Interactive Main Page Content - Show on all devices including mobile */}
          <div>
            <InteractiveMainPage onQuickStart={handleQuickStart} />
          </div>
          
          {/* Districts and interactive content now visible on all devices */}
          
        </div>

        {/* Cookie Consent Banner */}
        <CookieConsent />
        
        {/* PWA Install Button */}
        <InstallPWAButton />
      </div>
    </WeatherThemeProvider>
  );
};

export default App;
