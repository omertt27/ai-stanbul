import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Footer = () => {
  const location = useLocation();
  
  // Don't show footer on main page unless chat is expanded
  if (location.pathname === '/') {
    return null;
  }
  
  const isLightMode = document.body.classList.contains('light');
  
  const footerStyle = {
    position: 'relative', // Changed from 'fixed' to 'relative'
    bottom: 'auto',
    left: 'auto',
    right: 'auto',
    width: '100%',
    display: 'flex',
    justifyContent: 'center',
    gap: '2rem',
    fontSize: '0.875rem',
    zIndex: 10,
    backgroundColor: 'rgba(75, 85, 99, 0.9)',
    color: '#ffffff',
    padding: '1rem 2rem',
    backdropFilter: 'blur(8px)',
    borderTop: '1px solid rgba(156, 163, 175, 0.2)',
    marginTop: '2rem', // Add some space above
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
