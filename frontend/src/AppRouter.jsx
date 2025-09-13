import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation, Link } from 'react-router-dom';
import TestApp from './TestApp';
import App from './App';
import Chatbot from './Chatbot';
import SimpleChatbot from './SimpleChatbot';
import TestComponent from './TestComponent';
import ChatbotTester from './ChatbotTester';
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
  const [chatExpanded, setChatExpanded] = useState(false);

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
    
    // Clean up any old light mode data
    localStorage.removeItem('light-mode');
    
    return () => {
      window.removeEventListener('chatStateChanged', handleChatStateChange);
    };
  }, []);

  return (
    <Router>
      <AppContent 
        chatExpanded={chatExpanded}
      />
    </Router>
  );
};

const AppContent = ({ chatExpanded }) => {
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
        <Route path="/chat" element={<ForceRefreshRoute component={App} routeName="Chat" />} />
        <Route path="/test" element={<TestComponent key={getRouteKey('test')} />} />
        <Route path="/simple" element={<SimpleChatbot key={getRouteKey('simple')} />} />
        <Route path="/chatbot" element={<Chatbot key={getRouteKey('chatbot')} />} />
        <Route path="/demo" element={<EnhancedDemo key={getRouteKey('demo')} />} />
        <Route path="/test-chatbot" element={<ChatbotTester key={getRouteKey('test-chatbot')} />} />
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
