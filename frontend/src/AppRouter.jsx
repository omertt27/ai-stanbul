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

  // Determine if current page should show fixed navbar - only for specific pages that need it
  const shouldShowFixedNavbar = false; // Disable fixed navbar for now to make all pages consistent
  
  // Hide navbar logo on main page only, show on all other pages
  const shouldHideLogo = location.pathname === '/';
  
  // Hide navbar and footer for chatbot page (it has its own navigation)
  const shouldHideNavigation = location.pathname === '/chatbot';

  // Global navigation handler to ensure clean state transitions
  useEffect(() => {
    console.log('🔄 AppRouter: Navigation detected to', location.pathname);
    
    // Track page view with Google Analytics
    trackNavigation(location.pathname);

    // Force scroll to top on navigation
    window.scrollTo(0, 0);
    
  }, [location.pathname, location.search]);

  return (
    <>
      <GoogleAnalytics />

      {/* Show chatbot outline - only on pages that are not the dedicated chatbot page */}
      {!shouldHideNavigation && (
        <div className="chatbot-outline" style={{ position: 'fixed', zIndex: 9998 }}></div>
      )}
      
      {/* Conditionally show NavBar and Footer - only for pages that need them */}
      {!shouldHideNavigation && (shouldShowFixedNavbar ? (
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
      ))}
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/chat" element={<App />} />
        <Route path="/test" element={<TestComponent />} />
        <Route path="/simple" element={<SimpleChatbot />} />
        <Route path="/chatbot" element={<Chatbot />} />
        <Route path="/demo" element={<EnhancedDemo />} />
        <Route path="/test-chatbot" element={<ChatbotTester />} />
        <Route path="/about" element={<About />} />
        <Route path="/sources" element={<Sources />} />
        <Route path="/donate" element={<Donate />} />
        <Route path="/faq" element={<FAQ />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/blog" element={<BlogList />} />
        <Route path="/blog/new" element={<NewBlogPost />} />
        <Route path="/blog/:id" element={<BlogPost />} />
      </Routes>
    </>
  );
};

export default AppRouter;
