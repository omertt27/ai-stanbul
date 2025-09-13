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
  
  // Logo style - responsive positioning
  const logoStyle = {
    position: 'fixed',
    top: isMobile ? '0.5rem' : '0.25rem',
    left: isMobile ? '1rem' : '2rem',
    zIndex: 60,
    textDecoration: 'none',
    textAlign: 'center',
    cursor: 'pointer',
    pointerEvents: 'auto',
    transition: 'transform 0.2s ease, opacity 0.2s ease',
  };

  const logoTextStyle = {
    fontSize: isMobile ? '1.8rem' : '3.5rem',
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

  // Link style for navigation items
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
    display: 'block',
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
    // Track navigation click
    trackNavigation('about');
    
    // If we're already on the About page, reload it
    if (location.pathname === '/about') {
      e.preventDefault();
      window.location.reload();
    }
  };

  const handleFAQClick = () => {
    trackNavigation('faq');
  };

  const handleDonateClick = () => {
    trackNavigation('donate');
  };

  const handleChatClick = () => {
    trackNavigation('chat');
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
      {/* Logo in top-left corner with main page styling - conditionally rendered */}
      {!hideLogo && (
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
      )}
      
      {/* Desktop Navigation */}
      {!isMobile && (
        <nav style={{
          position: 'fixed',
          top: '0.25rem',
          right: '1.5rem',
          zIndex: 50,
          display: 'flex',
          justifyContent: 'flex-end',
          alignItems: 'center',
          gap: '1.5rem',
          padding: '0.75rem 1rem',
          fontWeight: 400,
          fontSize: '1rem',
          letterSpacing: '0.01em',
        }}>
          <Link to="/chat" onClick={handleChatClick} style={linkStyle(location.pathname === '/chat')}>Chat</Link>
          <Link to="/blog" onClick={handleBlogClick} style={linkStyle(location.pathname.startsWith('/blog'))}>Blog</Link>
          <Link to="/about" onClick={handleAboutClick} style={linkStyle(location.pathname === '/about')}>About</Link>
          <Link to="/faq" onClick={handleFAQClick} style={linkStyle(location.pathname === '/faq')}>FAQ</Link>
          <Link to="/donate" onClick={handleDonateClick} style={linkStyle(location.pathname === '/donate')}>Donate</Link>
        </nav>
      )}

      {/* Mobile Hamburger Menu Button */}
      {isMobile && (
        <button
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="navbar-hamburger"
          style={{
            position: 'fixed',
            top: '1rem',
            right: '1rem',
            zIndex: 62,
            background: 'rgba(99, 102, 241, 0.9)',
            border: 'none',
            borderRadius: '0.5rem',
            padding: '0.75rem',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            minHeight: '44px',
            minWidth: '44px',
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
            to="/chat" 
            onClick={handleChatClick} 
            style={{
              ...linkStyle(location.pathname === '/chat'),
              fontSize: '1.25rem',
              padding: '1rem 2rem',
              textAlign: 'center',
              minWidth: '8rem',
            }}
          >
            Chat
          </Link>
          <Link 
            to="/blog" 
            onClick={handleBlogClick} 
            style={{
              ...linkStyle(location.pathname.startsWith('/blog')),
              fontSize: '1.25rem',
              padding: '1rem 2rem',
              textAlign: 'center',
              minWidth: '8rem',
            }}
          >
            Blog
          </Link>
          <Link 
            to="/about" 
            onClick={handleAboutClick} 
            style={{
              ...linkStyle(location.pathname === '/about'),
              fontSize: '1.25rem',
              padding: '1rem 2rem',
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
              fontSize: '1.25rem',
              padding: '1rem 2rem',
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
              fontSize: '1.25rem',
              padding: '1rem 2rem',
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
