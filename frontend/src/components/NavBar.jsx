import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const NavBar = () => {
  const location = useLocation();
  const navStyle = {
    position: 'fixed',
    top: '0.5rem',
    right: '5rem',
    display: 'flex',
    justifyContent: 'flex-end',
    alignItems: 'center',
    gap: '2rem',
    padding: '0',
    background: 'none',
    fontWeight: 600,
    fontSize: '1.15rem',
    letterSpacing: '0.03em',
    zIndex: 30,
    width: 'auto',
  };
  const linkStyle = isActive => ({
    color: isActive ? '#818cf8' : '#c7c9e2',
    textDecoration: 'none',
    borderBottom: isActive ? '2.5px solid #818cf8' : '2.5px solid transparent',
    paddingBottom: '0.2rem',
    transition: 'color 0.2s, border-bottom 0.2s',
  });
  return (
    <nav style={navStyle}>
      <Link to="/" style={linkStyle(location.pathname === '/')}>Home</Link>
      <Link to="/about" style={linkStyle(location.pathname === '/about')}>About</Link>
      <Link to="/sources" style={linkStyle(location.pathname === '/sources')}>Sources</Link>
      <Link to="/donate" style={linkStyle(location.pathname === '/donate')}>Donate</Link>
    </nav>
  );
};

export default NavBar;
