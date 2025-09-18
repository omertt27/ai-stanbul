import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';

const Footer = () => {
  const location = useLocation();
  const [showFooter, setShowFooter] = useState(false);
  
  // Always call hooks first, before any conditional returns
  // Show footer on all pages
  useEffect(() => {
    // Always show footer on all pages
    setShowFooter(true);
  }, []);
  
  // Show footer on all pages
  // Footer is static on all pages and appears at the bottom of content
  
  const footerStyle = {
    position: 'static',
    bottom: 'auto',
    left: '0',
    right: '0',
    width: '100vw',
    display: 'flex',
    justifyContent: 'center',
    gap: '2rem',
    fontSize: '0.875rem',
    zIndex: 20,
    backgroundColor: 'rgba(75, 85, 99, 0.7)',
    color: '#ffffff',
    padding: '1rem 2rem',
    backdropFilter: 'blur(8px)',
    borderTop: '1px solid rgba(156, 163, 175, 0.2)',
    pointerEvents: 'auto',
    transform: 'translateY(-10px)', // Move footer 10px up
    willChange: 'transform',
    transition: 'transform 0.3s ease-in-out',
    marginTop: '2rem',
  };
  
  const linkStyle = (isActive) => ({
    color: isActive 
      ? '#818cf8'
      : '#ffffff',
    textDecoration: 'none',
    borderBottom: isActive ? '2px solid #818cf8' : '2px solid transparent',
    paddingBottom: '0.25rem',
    transition: 'all 0.2s ease',
    fontSize: '0.875rem',
    fontWeight: '500',
  });
  
  const handleSourcesClick = (e) => {
    if (location.pathname === '/sources') {
      e.preventDefault();
      window.location.reload();
    }
  };

  const handleContactClick = (e) => {
    if (location.pathname === '/contact') {
      e.preventDefault();
      window.location.reload();
    }
  };

  return (
    <footer style={footerStyle}>
      <Link 
        to="/sources" 
        onClick={handleSourcesClick}
        style={linkStyle(location.pathname === '/sources')}
        onMouseOver={(e) => {
          e.target.style.color = '#818cf8';
          e.target.style.borderBottomColor = '#818cf8';
        }}
        onMouseOut={(e) => {
          e.target.style.color = location.pathname === '/sources' 
            ? '#818cf8' 
            : '#ffffff';
          e.target.style.borderBottomColor = location.pathname === '/sources' 
            ? '#818cf8' 
            : 'transparent';
        }}
      >
        Sources
      </Link>
      <Link 
        to="/privacy" 
        style={linkStyle(location.pathname === '/privacy')}
        onMouseOver={(e) => {
          e.target.style.color = '#818cf8';
          e.target.style.borderBottomColor = '#818cf8';
        }}
        onMouseOut={(e) => {
          e.target.style.color = location.pathname === '/privacy' 
            ? '#818cf8' 
            : '#ffffff';
          e.target.style.borderBottomColor = location.pathname === '/privacy' 
            ? '#818cf8' 
            : 'transparent';
        }}
      >
        Privacy
      </Link>
      <Link 
        to="/gdpr" 
        style={linkStyle(location.pathname === '/gdpr')}
        onMouseOver={(e) => {
          e.target.style.color = '#818cf8';
          e.target.style.borderBottomColor = '#818cf8';
        }}
        onMouseOut={(e) => {
          e.target.style.color = location.pathname === '/gdpr' 
            ? '#818cf8' 
            : '#ffffff';
          e.target.style.borderBottomColor = location.pathname === '/gdpr' 
            ? '#818cf8' 
            : 'transparent';
        }}
      >
        GDPR
      </Link>
      <Link 
        to="/contact" 
        onClick={handleContactClick}
        style={linkStyle(location.pathname === '/contact')}
        onMouseOver={(e) => {
          e.target.style.color = '#818cf8';
          e.target.style.borderBottomColor = '#818cf8';
        }}
        onMouseOut={(e) => {
          e.target.style.color = location.pathname === '/contact' 
            ? '#818cf8' 
            : '#ffffff';
          e.target.style.borderBottomColor = location.pathname === '/contact' 
            ? '#818cf8' 
            : 'transparent';
        }}
      >
        Contact
      </Link>
    </footer>
  );
};

export default Footer;
