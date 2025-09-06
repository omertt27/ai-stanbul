import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';

const NavBar = () => {
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
  
  const navStyle = {
    position: 'fixed',
    // Adjust position based on screen size to avoid logo overlap
    top: isMobile ? (isUltraSmall ? '2.5rem' : '3.5rem') : '1.1rem', // Move down on mobile to clear logo
    right: isUltraSmall ? '0.25rem' : isSmallMobile ? '0.5rem' : '1.5rem',
    left: isMobile ? (isUltraSmall ? '0.25rem' : '0.5rem') : 'auto', // Full width on mobile
    zIndex: 50,
    display: 'flex',
    justifyContent: isMobile ? 'center' : 'flex-end',
    alignItems: 'center',
    gap: isUltraSmall ? '0.5rem' : isSmallMobile ? '0.75rem' : isMobile ? '1rem' : '1.5rem',
    padding: isMobile ? (isUltraSmall ? '0.5rem' : '0.75rem') : '0.5rem 0.75rem',
    background: isMobile ? (isLightMode ? 'rgba(253, 253, 254, 0.95)' : 'rgba(15, 16, 17, 0.95)') : 'none', // Add background on mobile
    backdropFilter: isMobile ? 'blur(10px)' : 'none',
    borderRadius: isMobile ? '0.5rem' : '0',
    border: isMobile ? (isLightMode ? '1px solid rgba(209, 213, 219, 0.6)' : '1px solid rgba(255, 255, 255, 0.08)') : 'none',
    fontWeight: 400,
    fontSize: isUltraSmall ? '0.875rem' : isSmallMobile ? '1rem' : isMobile ? '1.1rem' : '1.25rem',
    letterSpacing: '0.01em',
    width: isMobile ? 'auto' : 'auto',
    flexWrap: 'wrap',
    maxWidth: isMobile ? 'calc(100vw - 1rem)' : 'none',
  };
  
  const isLightMode = document.body.classList.contains('light');
  
  const linkStyle = isActive => ({
    color: isActive 
      ? '#6366f1' 
      : (isLightMode ? '#475569' : '#c7c9e2'),
    textDecoration: 'none',
    borderBottom: isActive ? '2px solid #6366f1' : '2px solid transparent',
    paddingBottom: isUltraSmall ? '0.3rem' : isSmallMobile ? '0.4rem' : '0.625rem',
    paddingTop: isUltraSmall ? '0.3rem' : isSmallMobile ? '0.4rem' : '0.625rem',
    paddingLeft: isUltraSmall ? '0.5rem' : isSmallMobile ? '0.75rem' : '1.25rem',
    paddingRight: isUltraSmall ? '0.5rem' : isSmallMobile ? '0.75rem' : '1.25rem',
    borderRadius: isUltraSmall ? '0.3rem' : isSmallMobile ? '0.4rem' : '0.625rem',
    transition: 'all 0.2s ease',
    minWidth: isUltraSmall ? '2.5rem' : isSmallMobile ? '3rem' : '4rem',
    textAlign: 'center',
    fontWeight: 'inherit',
    whiteSpace: 'nowrap',
    // Hover effects
    cursor: 'pointer',
    ':hover': {
      backgroundColor: isLightMode ? 'rgba(99, 102, 241, 0.1)' : 'rgba(99, 102, 241, 0.2)',
      transform: 'translateY(-1px)',
    }
  });
  
  const handleBlogClick = (e) => {
    // If we're already on the main blog list page, reload it
    if (location.pathname === '/blog') {
      e.preventDefault();
      window.location.reload();
    }
    // If we're on any other blog page (like /blog/123 or /blog/new), 
    // let the Link navigate to /blog normally
  };

  const handleAboutClick = (e) => {
    // If we're already on the About page, reload it
    if (location.pathname === '/about') {
      e.preventDefault();
      window.location.reload();
    }
  };

  return (
    <nav style={navStyle}>
      <Link to="/blog" onClick={handleBlogClick} style={linkStyle(location.pathname.startsWith('/blog'))}>Blog</Link>
      <Link to="/about" onClick={handleAboutClick} style={linkStyle(location.pathname === '/about')}>About</Link>
      <Link to="/faq" style={linkStyle(location.pathname === '/faq')}>FAQ</Link>
      <Link to="/donate" style={linkStyle(location.pathname === '/donate')}>Donate</Link>
    </nav>
  );
};

export default NavBar;
