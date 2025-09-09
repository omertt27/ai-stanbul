import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation, Link } from 'react-router-dom';
import TestApp from './TestApp';
import App from './App';
import Chatbot from './Chatbot';
import SimpleChatbot from './SimpleChatbot';
import TestComponent from './TestComponent';
import About from './pages/About';
import Sources from './pages/Sources';
import Donate from './pages/Donate';
import FAQ from './pages/FAQ';
import Contact from './pages/Contact';
import BlogList from './pages/BlogList';
import BlogPost from './pages/BlogPost';
import NewBlogPost from './pages/NewBlogPost';
import EnhancedDemo from './pages/EnhancedDemo';
import NavBar from './components/NavBar';
import Footer from './components/Footer';
import ForceRefreshRoute from './components/ForceRefreshRoute';
import GoogleAnalytics, { trackNavigation } from './utils/analytics';

const AppRouter = () => {
  const [isLightMode, setIsLightMode] = useState(() => {
    // Load saved light mode preference
    try {
      const saved = localStorage.getItem('light-mode');
      return saved ? JSON.parse(saved) : false;
    } catch {
      return false;
    }
  });
  const [chatExpanded, setChatExpanded] = useState(false);

  // Apply saved light mode immediately on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('light-mode');
      if (saved) {
        const savedMode = JSON.parse(saved);
        setIsLightMode(savedMode);
      }
    } catch (error) {
      console.log('Could not load light mode preference');
    }
  }, []);

  // Listen for chat state changes
  useEffect(() => {
    const handleChatStateChange = (event) => {
      setChatExpanded(event.detail.expanded || event.detail.hasMessages);
    };
    
    window.addEventListener('chatStateChanged', handleChatStateChange);
    
    // Check initial state
    const hasActiveChat = localStorage.getItem('chat-messages') && 
                         JSON.parse(localStorage.getItem('chat-messages') || '[]').length > 0;
    setChatExpanded(hasActiveChat);
    
    return () => {
      window.removeEventListener('chatStateChanged', handleChatStateChange);
    };
  }, []);

  // Toggle light mode on both body and chat containers
  useEffect(() => {
    const body = document.body;
    const root = document.getElementById('root');
    const mainPageBg = document.querySelector('.main-page-background');
    const chatContainers = document.querySelectorAll('.chat-container');
    const searchBars = document.querySelectorAll('.searchbar');
    
    if (isLightMode) {
      body.classList.add('light');
      if (root) root.classList.add('light');
      chatContainers.forEach(el => el.classList.add('light-chat'));
      
      // Force pure white background for light mode on all elements
      body.style.background = '#ffffff !important';
      body.style.backgroundColor = '#ffffff';
      body.style.color = '#2d3748 !important';
      
      if (root) {
        root.style.background = '#ffffff !important';
        root.style.backgroundColor = '#ffffff';
      }
      
      // Force main page background to white
      if (mainPageBg) {
        mainPageBg.style.background = '#ffffff !important';
        mainPageBg.style.backgroundColor = '#ffffff !important';
        mainPageBg.style.backgroundImage = 'none !important';
      }
      
      // Update CSS custom properties for pure white background
      document.documentElement.style.setProperty('--bg-primary', '#ffffff');
      document.documentElement.style.setProperty('--bg-secondary', '#f8fafc');
      document.documentElement.style.setProperty('--text-primary', '#2d3748');
      document.documentElement.style.setProperty('--text-secondary', '#4a5568');
      
    } else {
      body.classList.remove('light');
      if (root) root.classList.remove('light');
      chatContainers.forEach(el => el.classList.remove('light-chat'));
      
      // Restore dark mode styles
      body.style.background = 'linear-gradient(135deg, #0f1011 0%, #1a1b1d 100%) !important';
      body.style.backgroundColor = '';
      body.style.color = '#e5e7eb !important';
      
      if (root) {
        root.style.background = '';
        root.style.backgroundColor = '';
      }
      
      // Restore main page background
      if (mainPageBg) {
        mainPageBg.style.background = '';
        mainPageBg.style.backgroundColor = '';
        mainPageBg.style.backgroundImage = '';
      }
      
      // Update CSS custom properties for dark mode
      document.documentElement.style.setProperty('--bg-primary', '#0f1011');
      document.documentElement.style.setProperty('--bg-secondary', '#1a1b1d');
      document.documentElement.style.setProperty('--text-primary', '#e5e7eb');
      document.documentElement.style.setProperty('--text-secondary', '#c7c9e2');
    }
    
    // Force repaint
    body.style.display = 'none';
    body.offsetHeight; // Trigger reflow
    body.style.display = '';
    
  }, [isLightMode]);

  const toggleTheme = () => {
    const newMode = !isLightMode;
    setIsLightMode(newMode);
    // Save to localStorage
    localStorage.setItem('light-mode', JSON.stringify(newMode));
  };

  const buttonStyle = {
    position: 'fixed',
    top: '0.5rem',  
    right: '0.5rem', // Moved closer to edge for easier access
    zIndex: 9999,
    background: isLightMode 
      ? 'rgba(255, 255, 255, 0.9)' 
      : 'rgba(0, 0, 0, 0.7)', // Added background for better clickability
    border: isLightMode 
      ? '1px solid #e2e8f0' 
      : '1px solid #4b5563',
    borderRadius: '8px',
    padding: '16px', // Much bigger padding for easier clicking
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    color: isLightMode ? '#374151' : '#e5e7eb',
    width: '60px', // Much bigger clickable area
    height: '60px',
    minWidth: '60px',
    minHeight: '60px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transform: 'scale(1)',
    boxShadow: isLightMode 
      ? '0 2px 8px rgba(0, 0, 0, 0.1)' 
      : '0 2px 8px rgba(255, 255, 255, 0.1)',
  };

  return (
    <Router>
      <AppContent 
        isLightMode={isLightMode} 
        toggleTheme={toggleTheme} 
        buttonStyle={buttonStyle}
        chatExpanded={chatExpanded}
      />
    </Router>
  );
};

const AppContent = ({ isLightMode, toggleTheme, buttonStyle, chatExpanded }) => {
  const location = useLocation();
  const [routeKey, setRouteKey] = useState(0);

  // Determine if current page should show fixed navbar - only for specific pages that need it
  const shouldShowFixedNavbar = false; // Disable fixed navbar for now to make all pages consistent
  
  // Hide navbar logo on main page only, show on all other pages
  const shouldHideLogo = location.pathname === '/';

  // Global navigation handler to ensure clean state transitions
  useEffect(() => {
    console.log('🔄 AppRouter: Navigation detected to', location.pathname);
    
    // Track page view with Google Analytics
    trackNavigation(location.pathname);

    // Track page navigation
    const pageName = location.pathname.split('/')[1] || 'home';
    trackNavigation(pageName);

    // Force complete remount by updating key
    setRouteKey(prev => prev + 1);
    
    // Force scroll to top on navigation
    window.scrollTo(0, 0);
    
    // Clear any cached data that might interfere
    if (window.performance && window.performance.clearMarks) {
      window.performance.clearMarks();
    }
    
    // Force a brief delay to ensure clean state
    setTimeout(() => {
      console.log('✅ AppRouter: Route transition complete for', location.pathname);
    }, 100);
  }, [location.pathname, location.search]);

  // Create unique keys for each route to force remounting
  const getRouteKey = (basePath) => `${basePath}-${routeKey}-${location.pathname}-${location.search}`;

  return (
    <>
      <GoogleAnalytics />
      <button
        onClick={toggleTheme}
        style={buttonStyle}
        title={`Click to switch to ${isLightMode ? 'dark' : 'light'} mode`}
        aria-label={`Switch to ${isLightMode ? 'dark' : 'light'} mode`}
        onMouseEnter={(e) => {
          e.target.style.transform = 'scale(1.1)';
          e.target.style.backgroundColor = isLightMode 
            ? 'rgba(255, 255, 255, 1)' 
            : 'rgba(0, 0, 0, 0.9)';
          e.target.style.boxShadow = isLightMode 
            ? '0 4px 12px rgba(0, 0, 0, 0.2)' 
            : '0 4px 12px rgba(255, 255, 255, 0.2)';
        }}
        onMouseLeave={(e) => {
          e.target.style.transform = 'scale(1)';
          e.target.style.backgroundColor = isLightMode 
            ? 'rgba(255, 255, 255, 0.9)' 
            : 'rgba(0, 0, 0, 0.7)';
          e.target.style.boxShadow = isLightMode 
            ? '0 2px 8px rgba(0, 0, 0, 0.1)' 
            : '0 2px 8px rgba(255, 255, 255, 0.1)';
        }}
      >
        {/* Extra big stick symbol with better visibility */}
        <div style={{
          width: '8px', // Even thicker for better clicking
          height: '36px', // Even taller
          backgroundColor: 'currentColor',
          borderRadius: '4px',
          transition: 'all 0.3s ease',
          boxShadow: isLightMode 
            ? '0 2px 4px rgba(0, 0, 0, 0.3)' 
            : '0 2px 4px rgba(255, 255, 255, 0.5)'
        }} />
      </button>

      {/* Show chatbot outline - keep it fixed on all pages */}
      <div className="chatbot-outline" style={{ position: 'fixed', zIndex: 9998 }}></div>
      
      {/* Conditionally show NavBar and Footer - only for pages that need them */}
      {shouldShowFixedNavbar ? (
        <>
          {/* Fixed navbar for pages other than chatbot and main */}
          <div className={`fixed-navbar ${isLightMode ? '' : 'dark'}`}>
            {/* AI Istanbul Logo - Centered */}
            <div className="fixed-navbar-logo">
              <Link to="/" style={{textDecoration: 'none'}}>
                <div className="header-logo logo-istanbul">
                  <span className="logo-text">
                    A/<span style={{fontWeight: 400}}>STANBUL</span>
                  </span>
                </div>
              </Link>
            </div>
            
            {/* Navigation Links */}
            <nav className="fixed-navbar-links">
              <Link 
                to="/blog" 
                className={`fixed-navbar-link ${location.pathname.startsWith('/blog') ? 'active' : ''}`}
              >
                Blog
              </Link>
              <Link 
                to="/about" 
                className={`fixed-navbar-link ${location.pathname === '/about' ? 'active' : ''}`}
              >
                About
              </Link>
              <Link 
                to="/faq" 
                className={`fixed-navbar-link ${location.pathname === '/faq' ? 'active' : ''}`}
              >
                FAQ
              </Link>
              <Link 
                to="/donate" 
                className={`fixed-navbar-link ${location.pathname === '/donate' ? 'active' : ''}`}
              >
                Donate
              </Link>
            </nav>
          </div>
          <Footer />
        </>
      ) : (
        <>
          {/* Regular NavBar for all pages with conditional logo hiding */}
          <NavBar hideLogo={shouldHideLogo} />
          <Footer />
        </>
      )}
      <Routes>
        <Route path="/" element={<ForceRefreshRoute component={App} routeName="Home" />} />
        <Route path="/test" element={<TestComponent key={getRouteKey('test')} />} />
        <Route path="/simple" element={<SimpleChatbot key={getRouteKey('simple')} />} />
        <Route path="/chatbot" element={<Chatbot key={getRouteKey('chatbot')} />} />
        <Route path="/demo" element={<EnhancedDemo key={getRouteKey('demo')} />} />
        <Route path="/about" element={<ForceRefreshRoute component={About} routeName="About" />} />
        <Route path="/sources" element={<ForceRefreshRoute component={Sources} routeName="Sources" />} />
        <Route path="/donate" element={<ForceRefreshRoute component={Donate} routeName="Donate" />} />
        <Route path="/faq" element={<ForceRefreshRoute component={FAQ} routeName="FAQ" />} />
        <Route path="/contact" element={<ForceRefreshRoute component={Contact} routeName="Contact" />} />
        <Route path="/blog" element={<ForceRefreshRoute component={BlogList} routeName="Blog" />} />
        <Route path="/blog/new" element={<ForceRefreshRoute component={NewBlogPost} routeName="New Blog Post" />} />
        <Route path="/blog/:id" element={<ForceRefreshRoute component={BlogPost} routeName="Blog Post" />} />
      </Routes>
    </>
  );
};

export default AppRouter;
