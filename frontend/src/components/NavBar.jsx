import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { trackNavigation } from '../utils/analytics';
import LanguageSwitcher from './LanguageSwitcher';

const NavBar = ({ hideLogo = false }) => {
  const location = useLocation();
  const { t } = useTranslation();
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  
  // Update window width on resize
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Media query checks
  const isMobile = windowWidth < 768;
  
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
    console.log('Logo clicked!'); // Debug log
    // Track navigation click
    trackNavigation('/');
    
    // Navigate to main page when logo is clicked
    window.location.href = '/';
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
      {/* Desktop Navigation - Clean with purple separator - Sticky */}
      {!isMobile && (
        <div style={{
          position: 'fixed', 
          top: 0, // Start at the very top
          right: 0, 
          left: 0, 
          zIndex: 50,
          background: 'transparent', // Made transparent
          backdropFilter: 'none', // Removed blur effect
          borderBottom: 'none', // Removed border for cleaner look
          boxShadow: 'none', // Removed shadow
          height: '70px', // Fixed height for navbar
        }}>
          <nav style={{
            position: 'absolute',
            top: '0.5rem', // Moved down from 0.25rem to 0.5rem
            left: '1.5rem', // Start from left
            right: '1.5rem', // Extend to right
            display: 'flex',
            justifyContent: 'space-between', // Always keep consistent layout: logo on left, links on right
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.6rem 1rem', // Slightly reduced padding
            fontWeight: 500,
            letterSpacing: '0.01em',
            background: 'transparent',
          }}>
            {/* Logo in navbar or invisible spacer to maintain layout */}
            {!hideLogo ? (
              <div 
                style={{
                  cursor: 'pointer',
                  pointerEvents: 'auto',
                  transition: 'transform 0.2s ease, opacity 0.2s ease',
                  display: 'flex',
                  alignItems: 'center',
                }}
                onClick={handleLogoClick}
              >
                <span style={{
                  fontSize: '2.6rem', // Even bigger logo size for desktop
                  fontWeight: 700,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  background: 'linear-gradient(90deg, #e5e7eb 0%, #8b5cf6 50%, #6366f1 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  textShadow: '0 2px 10px rgba(139, 92, 246, 0.3)',
                  transition: 'all 0.3s ease',
                  cursor: 'pointer',
                }}>
                  A/<span style={{fontWeight: 400}}>STANBUL</span>
                </span>
              </div>
            ) : (
              // Invisible spacer to maintain right alignment of navigation links
              <div style={{ flex: 1 }} />
            )}
            
            {/* Navigation links on the right */}
            <div style={{
              display: 'flex',
              gap: '0.5rem',
              alignItems: 'center',
            }}>
              <Link to="/chat" onClick={handleChatClick} className="navbar-link" style={linkStyle(location.pathname === '/chat')}>{t('navigation.chat')}</Link>
              <Link to="/blog" onClick={handleBlogClick} className="navbar-link" style={linkStyle(location.pathname.startsWith('/blog'))}>{t('navigation.blog')}</Link>
              <Link to="/about" onClick={handleAboutClick} className="navbar-link" style={linkStyle(location.pathname === '/about')}>{t('navigation.about')}</Link>
              <Link to="/faq" onClick={handleFAQClick} className="navbar-link" style={linkStyle(location.pathname === '/faq')}>{t('navigation.faq')}</Link>
              <Link to="/donate" onClick={handleDonateClick} className="navbar-link" style={linkStyle(location.pathname === '/donate')}>{t('navigation.donate')}</Link>
              <div style={{ marginLeft: '1rem' }}>
                <LanguageSwitcher />
              </div>
            </div>
          </nav>
          {/* Purple separator line below navbar */}
          <div style={{
            position: 'absolute',
            bottom: '-8px', // Moved up 2px from -10px to -8px
            left: 0,
            right: 0,
            height: '2px',
            background: 'linear-gradient(90deg, transparent 0%, #8b5cf6 20%, #6366f1 50%, #8b5cf6 80%, transparent 100%)',
            boxShadow: '0 1px 8px rgba(139, 92, 246, 0.4)',
          }} />
        </div>
      )}

      {/* Mobile Bottom Tab Bar - Modern Alternative to Hamburger Menu */}
      {isMobile && (
        <div style={{
          position: 'fixed',
          top: 0, // Changed from 30 to 0
          right: 0,
          left: 0,
          zIndex: 1005,
          background: 'rgba(15, 16, 17, 0.95)', // More opaque for better contrast
          backdropFilter: 'blur(10px)', // Add subtle blur back
          padding: '8px 16px 12px', // Reduced padding significantly
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          height: '50px', // Reduced from 70px to 50px
          borderBottom: '1px solid rgba(139, 92, 246, 0.2)', // Add subtle border
        }}>
          {/* Logo in mobile navbar - smaller */}
          {!hideLogo && (
            <div 
              style={{
                cursor: 'pointer',
                pointerEvents: 'auto',
                transition: 'transform 0.2s ease, opacity 0.2s ease',
                display: 'flex',
                alignItems: 'center',
              }}
              onClick={handleLogoClick}
            >
              <span style={{
                fontSize: '1.4rem', // Reduced from 1.8rem
                fontWeight: 700,
                letterSpacing: '0.05em', // Slightly reduced
                textTransform: 'uppercase',
                background: 'linear-gradient(90deg, #e5e7eb 0%, #8b5cf6 50%, #6366f1 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                textShadow: '0 2px 10px rgba(139, 92, 246, 0.3)',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
              }}>
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          )}
        </div>
      )}

      {/* Mobile Bottom Tab Bar Navigation - more compact */}
      {isMobile && (
        <div style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          zIndex: 1006,
          background: 'rgba(15, 16, 17, 0.98)',
          backdropFilter: 'blur(20px)',
          borderTop: '1px solid rgba(139, 92, 246, 0.3)',
          padding: '8px 16px 12px', // Reduced padding
          display: 'flex',
          justifyContent: 'space-around',
          alignItems: 'center',
          boxShadow: '0 -4px 20px rgba(0, 0, 0, 0.3)',
        }}>
          {/* Chat Tab */}
          <Link 
            to="/chat" 
            onClick={handleChatClick}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '2px', // Reduced gap
              textDecoration: 'none',
              color: location.pathname === '/chat' ? '#8b5cf6' : '#9ca3af',
              transition: 'all 0.2s ease',
              padding: '6px', // Reduced padding
              borderRadius: '6px', // Smaller radius
              minWidth: '44px', // Maintain touch target
            }}
          >
            <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24"> {/* Reduced from 24 to 20 */}
              <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <span style={{fontSize: '0.65rem', fontWeight: '500', marginTop: '2px'}}>{t('navigation.chat')}</span> {/* Reduced font size */}
          </Link>

          {/* Blog Tab */}
          <Link 
            to="/blog" 
            onClick={handleBlogClick}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '2px',
              textDecoration: 'none',
              color: location.pathname.startsWith('/blog') ? '#8b5cf6' : '#9ca3af',
              transition: 'all 0.2s ease',
              padding: '6px',
              borderRadius: '6px',
              minWidth: '44px',
            }}
          >
            <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
              <path d="M19 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z" />
            </svg>
            <span style={{fontSize: '0.65rem', fontWeight: '500', marginTop: '2px'}}>{t('navigation.blog')}</span>
          </Link>

          {/* About Tab */}
          <Link 
            to="/about" 
            onClick={handleAboutClick}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '2px',
              textDecoration: 'none',
              color: location.pathname === '/about' ? '#8b5cf6' : '#9ca3af',
              transition: 'all 0.2s ease',
              padding: '6px',
              borderRadius: '6px',
              minWidth: '44px',
            }}
          >
            <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
              <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span style={{fontSize: '0.65rem', fontWeight: '500', marginTop: '2px'}}>{t('navigation.about')}</span>
          </Link>

          {/* FAQ Tab */}
          <Link 
            to="/faq" 
            onClick={handleFAQClick}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '2px',
              textDecoration: 'none',
              color: location.pathname === '/faq' ? '#8b5cf6' : '#9ca3af',
              transition: 'all 0.2s ease',
              padding: '6px',
              borderRadius: '6px',
              minWidth: '44px',
            }}
          >
            <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span style={{fontSize: '0.65rem', fontWeight: '500', marginTop: '2px'}}>{t('navigation.faq')}</span>
          </Link>

          {/* Donate Tab */}
          <Link 
            to="/donate" 
            onClick={handleDonateClick}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '2px',
              textDecoration: 'none',
              color: location.pathname === '/donate' ? '#8b5cf6' : '#9ca3af',
              transition: 'all 0.2s ease',
              padding: '6px',
              borderRadius: '6px',
              minWidth: '44px',
            }}
          >
            <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
            </svg>
            <span style={{fontSize: '0.65rem', fontWeight: '500', marginTop: '2px'}}>{t('navigation.donate')}</span>
          </Link>
        </div>
      )}
    </>
  );
};

export default NavBar;
