import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
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
  const [isLightMode, setIsLightMode] = useState(false);

  // Toggle light mode on both body and chat containers
  useEffect(() => {
    const body = document.body;
    const chatContainers = document.querySelectorAll('.chat-container');
    
    if (isLightMode) {
      body.classList.add('light');
      chatContainers.forEach(el => el.classList.add('light-chat'));
    } else {
      body.classList.remove('light');
      chatContainers.forEach(el => el.classList.remove('light-chat'));
    }
  }, [isLightMode]);

  const toggleTheme = () => {
    setIsLightMode(!isLightMode);
  };

  const buttonStyle = {
    position: 'fixed',
    top: '2.5rem',  // Same as NavBar
    right: '1.5rem',
    zIndex: 9999,
    background: isLightMode ? 'rgba(255, 255, 255, 0.9)' : 'transparent',
    border: isLightMode ? '1px solid #e2e8f0' : 'none',
    borderRadius: '50%',
    padding: 0,
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    color: isLightMode ? '#475569' : '#e5e7eb',
    boxShadow: isLightMode ? '0 2px 8px rgba(99, 102, 241, 0.15)' : 'none',
    width: '32px',
    height: '32px',
    minWidth: '32px',
    minHeight: '32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  return (
    <Router>
      <AppContent 
        isLightMode={isLightMode} 
        toggleTheme={toggleTheme} 
        buttonStyle={buttonStyle} 
      />
    </Router>
  );
};

const AppContent = ({ isLightMode, toggleTheme, buttonStyle }) => {
  const location = useLocation();
  const [routeKey, setRouteKey] = useState(0);

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
        title={`Switch to ${isLightMode ? 'dark' : 'light'} mode`}
      >
        {isLightMode ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5"></circle>
            <line x1="12" y1="1" x2="12" y2="3"></line>
            <line x1="12" y1="21" x2="12" y2="23"></line>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
            <line x1="1" y1="12" x2="3" y2="12"></line>
            <line x1="21" y1="12" x2="23" y2="12"></line>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
          </svg>
        )}
      </button>

      {/* Show chatbot outline - keep it fixed on all pages */}
      <div className="chatbot-outline" style={{ position: 'fixed', zIndex: 9998 }}></div>
      
      <NavBar />
      <Footer />
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
