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
    zIndex: 30,
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
  
  return (
    <nav style={navStyle}>
      <Link to="/blog" style={linkStyle(location.pathname.startsWith('/blog'))}>Blog</Link>
      <Link to="/about" style={linkStyle(location.pathname === '/about')}>About</Link>
      <Link to="/sources" style={linkStyle(location.pathname === '/sources')}>Sources</Link>
      <Link to="/faq" style={linkStyle(location.pathname === '/faq')}>FAQ</Link>
      <Link to="/contact" style={linkStyle(location.pathname === '/contact')}>Contact</Link>
      <Link to="/donate" style={linkStyle(location.pathname === '/donate')}>Donate</Link>
    </nav>
  );
};

export default NavBar;
