import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const NavBar = () => {
  const location = useLocation();
  
  // Media query check for mobile
  const isMobile = window.innerWidth < 768;
  
  const navStyle = {
    position: 'fixed',
    top: '1.5rem',
    right: '1rem',
    display: 'flex',
    justifyContent: 'flex-end',
    alignItems: 'center',
    gap: isMobile ? '0.75rem' : '1.5rem',
    padding: '0.75rem',
    background: 'none',
    fontWeight: 500,
    fontSize: isMobile ? '1rem' : '1.125rem',
    letterSpacing: '0.02em',
    zIndex: 50,
    width: 'auto',
    flexWrap: 'wrap',
    maxWidth: 'calc(100vw - 2rem)',
  };
  
  const isLightMode = document.body.classList.contains('light');
  
  const linkStyle = isActive => ({
    color: isActive 
      ? '#6366f1' 
      : (isLightMode ? '#475569' : '#c7c9e2'),
    textDecoration: 'none',
    borderBottom: isActive ? '2.5px solid #6366f1' : '2.5px solid transparent',
    paddingBottom: '0.375rem',
    paddingTop: '0.375rem',
    paddingLeft: '0.75rem',
    paddingRight: '0.75rem',
    borderRadius: '0.5rem',
    transition: 'all 0.2s ease',
    minWidth: '3rem',
    textAlign: 'center',
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
