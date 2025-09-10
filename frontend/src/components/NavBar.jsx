import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { trackNavigation } from '../utils/analytics';

const NavBar = ({ hideLogo = false }) => {
  const location = useLocation();
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  
  // Update window width on resize
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Media query checks
  const isMobile = windowWidth < 768;
  const isSmallMobile = windowWidth < 480;
  const isUltraSmall = windowWidth < 320;
  
  // Logo style - positioned higher up
  const logoStyle = {
    position: 'fixed',
    top: '0.5rem', // Move up even more from 1rem to 0.5rem
    left: '2rem',
    zIndex: 60,
    textDecoration: 'none',
    textAlign: 'center',
    cursor: 'pointer', // Ensure it's clickable
    pointerEvents: 'auto', // Ensure click events work
    transition: 'transform 0.2s ease, opacity 0.2s ease',
  };

  const logoTextStyle = {
    fontSize: isMobile ? '2.5rem' : '3.5rem',
    fontWeight: 700,
    letterSpacing: '0.15em',
    textTransform: 'uppercase',
    background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    textShadow: '0 4px 20px rgba(99, 102, 241, 0.3)',
    transition: 'all 0.3s ease',
    cursor: 'pointer', // Make sure text cursor indicates clickability
  };

  // Navigation style - positioned higher up with bigger buttons
  const navStyle = {
    position: 'fixed',
    top: '0.5rem', // Move up even more from 1rem to 0.5rem
    right: '1.5rem',
    zIndex: 50,
    display: 'flex',
    justifyContent: 'flex-end',
    alignItems: 'center',
    gap: isMobile ? '1rem' : '1.5rem', // Reduced gap for closer buttons
    padding: '0.75rem 1rem', // Increased padding for bigger appearance
    fontWeight: 400,
    fontSize: isMobile ? '1.2rem' : '1.4rem', // Increased font size for bigger buttons
    letterSpacing: '0.01em',
    flexWrap: 'wrap',
  };
  const linkStyle = isActive => ({
    color: isActive 
      ? '#6366f1' 
      : '#c7c9e2', // Always use dark mode colors
    textDecoration: 'none',
    borderBottom: isActive ? '2px solid #6366f1' : '2px solid transparent',
    paddingBottom: '0.7rem', // Increased padding for bigger buttons
    paddingTop: '0.7rem',
    paddingLeft: '1.2rem', // Increased padding for bigger buttons
    paddingRight: '1.2rem',
    borderRadius: '0.5rem',
    transition: 'all 0.2s ease',
    fontWeight: 'inherit',
    whiteSpace: 'nowrap',
    cursor: 'pointer',
    minWidth: '5rem', // Increased minimum width for bigger buttons
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
      
      {/* Navigation in top-right corner */}
      <nav style={navStyle}>
        <Link to="/blog" onClick={handleBlogClick} style={linkStyle(location.pathname.startsWith('/blog'))}>Blog</Link>
        <Link to="/about" onClick={handleAboutClick} style={linkStyle(location.pathname === '/about')}>About</Link>
        <Link to="/faq" onClick={handleFAQClick} style={linkStyle(location.pathname === '/faq')}>FAQ</Link>
        <Link to="/donate" onClick={handleDonateClick} style={linkStyle(location.pathname === '/donate')}>Donate</Link>
      </nav>
    </>
  );
};

export default NavBar;
