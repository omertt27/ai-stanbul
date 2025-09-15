import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { trackNavigation } from '../utils/analytics';

const NavBar = ({ hideLogo = false }) => {
  const location = useLocation();
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  
  // Update window width on resize
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
      // Close mobile menu when resizing to desktop
      if (window.innerWidth >= 768) {
        setIsMobileMenuOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Media query checks
  const isMobile = windowWidth < 768;
  
  // Handle body scroll lock for mobile menu
  useEffect(() => {
    if (isMobile && isMobileMenuOpen) {
      document.body.classList.add('mobile-menu-open');
    } else {
      document.body.classList.remove('mobile-menu-open');
    }
    
    // Cleanup on unmount
    return () => {
      document.body.classList.remove('mobile-menu-open');
    };
  }, [isMobile, isMobileMenuOpen]);
  
  // Close mobile menu when route changes
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [location.pathname]);
  
  // Logo style - responsive positioning - BIGGER, properly positioned
  const logoStyle = {
    position: 'fixed',
    top: isMobile ? '-0.2rem' : '-0.2rem', // Much higher position
    left: isMobile ? '1rem' : '2rem',
    zIndex: 60,
    textDecoration: 'none',
    textAlign: 'center',
    cursor: 'pointer',
    pointerEvents: 'auto',
    transition: 'transform 0.2s ease, opacity 0.2s ease',
  };

  const logoTextStyle = {
    fontSize: isMobile ? '1.8rem' : '2.5rem', // Smaller, more reasonable size
    fontWeight: 700,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    textShadow: '0 2px 10px rgba(99, 102, 241, 0.2)',
    transition: 'all 0.3s ease',
    cursor: 'pointer',
  };

  // Modern cleaner button style for navigation items - no boxes, more compact
  const linkStyle = (isActive) => ({
    color: isActive ? '#8b5cf6' : '#c7c9e2',
    textDecoration: 'none',
    paddingBottom: '0.8rem', // Increased padding
    paddingTop: '0.8rem',
    paddingLeft: '1.2rem', // Increased padding
    paddingRight: '1.2rem',
    borderRadius: '8px',
    transition: 'all 0.3s ease',
    fontWeight: isActive ? '700' : '500',
    whiteSpace: 'nowrap',
    cursor: 'pointer',
    fontSize: isMobile ? '1.2rem' : '1.2rem', // Bigger font size
    display: 'block',
    textAlign: 'center',
    lineHeight: '1.2',
    margin: '0 0.5rem', // Reduced margin
    background: 'transparent',
    border: 'none',
    boxShadow: 'none',
    transform: isActive ? 'translateY(-1px)' : 'translateY(0)',
    borderBottom: isActive ? '2px solid #8b5cf6' : '2px solid transparent',
  });
  
  const handleLogoClick = () => {
    // Track navigation click
    trackNavigation('/');
    
    // Check if there's an active chat session
    const hasActiveChat = localStorage.getItem('chat-messages');
    const parsedMessages = hasActiveChat ? JSON.parse(hasActiveChat) : [];
    
    // If there's an active chat with messages, go back to chat view instead of main page
    if (parsedMessages && parsedMessages.length > 0) {
      // Trigger chat state change event to expand chat
      window.dispatchEvent(new CustomEvent('chatStateChanged', { 
        detail: { expanded: true, hasMessages: true } 
      }));
      // Don't clear session - preserve chat history
    } else {
      // No active chat, go to main page and clear any leftover data
      localStorage.removeItem('chat_session_id');
      localStorage.removeItem('chat-messages');
      
      // Trigger chat state change event to go to main page
      window.dispatchEvent(new CustomEvent('chatStateChanged', { 
        detail: { expanded: false, hasMessages: false } 
      }));
    }
  };

  const handleBlogClick = (e) => {
    // Track navigation click
    trackNavigation('blog');
    
    // If we're already on the main blog list page, reload it
    if (location.pathname === '/blog') {
      e.preventDefault();
      window.location.reload();
    }
    // If we're on any other blog page (like /blog/123 or /blog/new), 
    // let the Link navigate to /blog normally
  };

  const handleAboutClick = (e) => {
    // Add button press animation
    e.target.classList.add('button-press-animation');
    setTimeout(() => e.target.classList.remove('button-press-animation'), 200);
    
    // Track navigation click
    trackNavigation('about');
    
    // If we're already on the About page, reload it
    if (location.pathname === '/about') {
      e.preventDefault();
      window.location.reload();
    }
  };

  const handleFAQClick = (e) => {
    // Add button press animation
    e.target.classList.add('button-press-animation');
    setTimeout(() => e.target.classList.remove('button-press-animation'), 200);
    
    trackNavigation('faq');
  };

  const handleDonateClick = (e) => {
    // Add button press animation
    e.target.classList.add('button-press-animation');
    setTimeout(() => e.target.classList.remove('button-press-animation'), 200);
    
    trackNavigation('donate');
  };

  const handleChatClick = (e) => {
    // Add button press animation
    e.target.classList.add('button-press-animation');
    setTimeout(() => e.target.classList.remove('button-press-animation'), 200);
    
    trackNavigation('chatbot');
  };

  // Track navigation for analytics
  useEffect(() => {
    const handleClick = (e) => {
      const target = e.target.closest('a');
      if (target) {
        const { pathname } = new URL(target.href);
        trackNavigation(pathname);
      }
    };

    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, []);

  return (
    <>
      {/* Logo with purple bar underneath */}
      {!hideLogo && (
        <div style={{position: 'fixed', top: 0, left: 0, zIndex: 60}}>
          <Link 
            to="/" 
            style={logoStyle} 
            onClick={handleLogoClick}
          >
            <div className="chat-title logo-istanbul">
              <span className="logo-text" style={logoTextStyle}>
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </Link>
          {/* Purple bar under logo */}
          <div style={{
            position: 'absolute',
            bottom: isMobile ? '-0.5rem' : '-0.7rem', // Moved down a bit more
            left: isMobile ? '1rem' : '2rem',
            width: isMobile ? '100px' : '140px', // Adjusted to logo size
            height: '3px',
            background: 'linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%)',
            borderRadius: '2px',
            boxShadow: '0 2px 8px rgba(139, 92, 246, 0.4)',
          }} />
        </div>
      )}
      
      {/* Desktop Navigation - Clean with purple separator */}
      {!isMobile && (
        <div style={{position: 'fixed', top: 0, right: 0, left: 0, zIndex: 50}}>
          <nav style={{
            position: 'relative',
            top: '0.8rem', // Reduced top spacing
            right: '1.5rem',
            display: 'flex',
            justifyContent: 'flex-end',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.8rem 1rem', // Reduced padding
            fontWeight: 500,
            letterSpacing: '0.01em',
            background: 'transparent',
          }}>
            <Link to="/chatbot" onClick={handleChatClick} className="navbar-link" style={linkStyle(location.pathname === '/chatbot')}>Chat</Link>
            <Link to="/blog" onClick={handleBlogClick} className="navbar-link" style={linkStyle(location.pathname.startsWith('/blog'))}>Blog</Link>
            <Link to="/about" onClick={handleAboutClick} className="navbar-link" style={linkStyle(location.pathname === '/about')}>About</Link>
            <Link to="/faq" onClick={handleFAQClick} className="navbar-link" style={linkStyle(location.pathname === '/faq')}>FAQ</Link>
            <Link to="/donate" onClick={handleDonateClick} className="navbar-link" style={linkStyle(location.pathname === '/donate')}>Donate</Link>
          </nav>
          {/* Purple separator line across full width */}
          <div style={{
            position: 'absolute',
            bottom: '-0.7rem', // Moved down a bit more
            left: 0,
            right: 0,
            height: '2px',
            background: 'linear-gradient(90deg, transparent 0%, #8b5cf6 20%, #6366f1 50%, #8b5cf6 80%, transparent 100%)',
            boxShadow: '0 1px 8px rgba(139, 92, 246, 0.4)',
          }} />
        </div>
      )}

      {/* Mobile Hamburger Menu Button - positioned for smaller navbar */}
      {isMobile && (
        <button
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="navbar-hamburger"
          style={{
            position: 'fixed',
            top: '1rem', // Adjusted for smaller navbar
            right: '1rem',
            zIndex: 62,
            background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
            border: '2px solid rgba(139, 92, 246, 0.3)',
            borderRadius: '12px',
            padding: '0.8rem', // Slightly smaller
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            minHeight: '46px', // Smaller button
            minWidth: '46px',
            boxShadow: '0 4px 15px rgba(139, 92, 246, 0.4)',
          }}
          aria-label="Toggle navigation menu"
        >
          <svg 
            width="24" 
            height="24" 
            fill="none" 
            stroke="white" 
            viewBox="0 0 24 24"
            style={{
              transform: isMobileMenuOpen ? 'rotate(90deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s ease',
            }}
          >
            {isMobileMenuOpen ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      )}

      {/* Mobile Navigation Menu */}
      {isMobile && isMobileMenuOpen && (
        <nav 
          className="navbar-mobile-menu"
          style={{
            position: 'fixed',
            top: '0',
            right: '0',
            bottom: '0',
            left: '0',
            zIndex: 61,
            background: 'rgba(17, 24, 39, 0.95)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            gap: '2rem',
            padding: '2rem',
            fontSize: '1.25rem',
            fontWeight: 400,
          }}
        >
          <Link 
            to="/chatbot" 
            onClick={handleChatClick}
            className="navbar-link"
            style={{
              ...linkStyle(location.pathname === '/chatbot'),
              fontSize: '1rem',
              padding: '0.75rem 1.5rem',
              textAlign: 'center',
              minWidth: '8rem',
            }}
          >
            Chat
          </Link>
          <Link 
            to="/blog" 
            onClick={handleBlogClick}
            className="navbar-link"
            style={{
              ...linkStyle(location.pathname.startsWith('/blog')),
              fontSize: '1rem',
              padding: '0.75rem 1.5rem',
              textAlign: 'center',
              minWidth: '8rem',
            }}
          >
            Blog
          </Link>
          <Link 
            to="/about" 
            onClick={handleAboutClick}
            className="navbar-link"
            style={{
              ...linkStyle(location.pathname === '/about'),
              fontSize: '1rem',
              padding: '0.75rem 1.5rem',
              textAlign: 'center',
              minWidth: '8rem',
            }}
          >
            About
          </Link>
          <Link 
            to="/faq" 
            onClick={handleFAQClick} 
            style={{
              ...linkStyle(location.pathname === '/faq'),
              fontSize: '1rem',
              padding: '0.75rem 1.5rem',
              textAlign: 'center',
              minWidth: '8rem',
            }}
          >
            FAQ
          </Link>
          <Link 
            to="/donate" 
            onClick={handleDonateClick} 
            style={{
              ...linkStyle(location.pathname === '/donate'),
              fontSize: '1rem',
              padding: '0.75rem 1.5rem',
              textAlign: 'center',
              minWidth: '8rem',
            }}
          >
            Donate
          </Link>
        </nav>
      )}
    </>
  );
};

export default NavBar;
