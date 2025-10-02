import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import LanguageSwitcher from './LanguageSwitcher';
import { trackEvent } from '../utils/analytics';

const MainPageMobileNavbar = () => {
  const navigate = useNavigate();
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  // Update window width on resize
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const isMobile = windowWidth <= 768;

  const handleLogoClick = () => {
    // Track navigation back to home
    trackEvent('logo_click', 'navigation', 'home');
    
    // Navigate to main page when logo is clicked
    navigate('/');
  };

  // Only show on mobile
  if (!isMobile) {
    return null;
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      width: '100vw',
      height: '60px',
      background: 'rgba(17, 24, 39, 0.95)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 1rem',
      zIndex: 9999,
      borderBottom: '2px solid rgba(139, 92, 246, 0.6)',
      boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)'
    }}>
      {/* Logo on the left - Same style as main page */}
      <div onClick={handleLogoClick} style={{ 
        cursor: 'pointer',
        padding: '0.5rem',
        backgroundColor: 'transparent'
      }}>
        {/* Main page logo style - normal size */}
        <span style={{
          fontSize: '1.5rem',
          fontWeight: 700,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: 'transparent',
          background: 'linear-gradient(90deg, #e5e7eb 0%, #8b5cf6 50%, #6366f1 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          textShadow: '0 2px 10px rgba(139, 92, 246, 0.3)',
          transition: 'all 0.3s ease',
          cursor: 'pointer',
          lineHeight: '1.2'
        }}>
          A/STANBUL
        </span>
      </div>
      
      {/* Language options on the right - All 6 languages */}
      <div style={{
        display: 'flex',
        alignItems: 'center'
      }}>
        <LanguageSwitcher />
      </div>
    </div>
  );
};

export default MainPageMobileNavbar;
