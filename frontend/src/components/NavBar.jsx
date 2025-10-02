import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { trackNavigation } from '../utils/analytics';
import LanguageSwitcher from './LanguageSwitcher';
import Logo from './Logo';
import '../styles/mobile-enhanced.css';

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
          height: isMobile ? '60px' : '70px', // Smaller height for mobile
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
              <Logo 
                size="medium"
                onClick={handleLogoClick}
                isMobile={false}
              />
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
              {/* Language Switcher positioned at the far right */}
              <div style={{ 
                marginLeft: '1.5rem',
                padding: '4px 8px',
                borderRadius: '8px',
                background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(139, 92, 246, 0.03) 100%)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
                backdropFilter: 'blur(8px)',
                transition: 'all 0.3s ease',
              }}>
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

      {/* Mobile Top Navigation Bar - Enhanced Modern Aesthetic */}
      {isMobile && (
        <div className="mobile-navbar-glass mobile-gpu-accelerated mobile-safe-top" style={{
          position: 'fixed',
          top: 0,
          right: 0,
          left: 0,
          zIndex: 1005,
          background: 'linear-gradient(135deg, rgba(15, 16, 17, 0.98) 0%, rgba(24, 25, 31, 0.98) 50%, rgba(15, 16, 17, 0.95) 100%)',
          backdropFilter: 'blur(24px) saturate(200%)',
          padding: '16px 16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          height: isMobile ? '60px' : '70px',
          borderBottom: '1px solid rgba(139, 92, 246, 0.2)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2), 0 2px 8px rgba(139, 92, 246, 0.1)',
        }}>
          {/* Enhanced Logo in mobile navbar */}
          {!hideLogo ? (
            <Logo 
              size="small"
              onClick={handleLogoClick}
              isMobile={true}
            />
          ) : (
            // Empty spacer when logo is hidden to push language switcher to the right
            <div style={{ flex: 1 }} />
          )}

          {/* Enhanced Language Switcher for Mobile - Right aligned */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            padding: '6px 10px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(8px)',
            transition: 'all 0.3s ease',
            marginLeft: 'auto', // Ensures it stays on the right
          }}>
            <LanguageSwitcher />
          </div>
        </div>
      )}

      {/* Enhanced Mobile Bottom Tab Bar Navigation */}
      {isMobile && (
        <div className="mobile-navbar-glass mobile-gpu-accelerated mobile-safe-bottom" style={{
          position: 'fixed',
          bottom: '0px',
          left: '0px',
          right: '0px',
          zIndex: 1006,
          background: 'linear-gradient(135deg, rgba(15, 16, 17, 0.98) 0%, rgba(24, 25, 31, 0.98) 50%, rgba(15, 16, 17, 0.96) 100%)',
          backdropFilter: 'blur(24px) saturate(200%)',
          borderTop: '1px solid rgba(139, 92, 246, 0.25)',
          borderRadius: '24px 24px 0 0',
          padding: '16px 0px 24px',
          display: 'flex',
          justifyContent: 'stretch',
          alignItems: 'center',
          boxShadow: '0 -12px 40px rgba(0, 0, 0, 0.4), 0 -4px 16px rgba(139, 92, 246, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.05)',
          height: isMobile ? '60px' : '76px',
          marginBottom: '0px',
        }}>
          {/* Enhanced Chat Tab */}
          <Link 
            to="/chat" 
            onClick={handleChatClick}
            className={`mobile-nav-tab mobile-touch-target mobile-focus-ring haptic-feedback ${location.pathname === '/chat' ? 'mobile-nav-tab-active' : ''}`}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '4px',
              textDecoration: 'none',
              color: location.pathname === '/chat' ? '#f1f5f9' : 'rgb(156, 163, 175)',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              background: location.pathname === '/chat' 
                ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.25) 0%, rgba(99, 102, 241, 0.15) 100%)'
                : 'transparent',
              border: location.pathname === '/chat' 
                ? '1px solid rgba(139, 92, 246, 0.4)' 
                : '1px solid transparent',
              transform: location.pathname === '/chat' ? 'translateY(-2px) scale(1.05)' : 'translateY(0) scale(1)',
              boxShadow: location.pathname === '/chat' 
                ? '0 6px 20px rgba(139, 92, 246, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
                : 'none',
            }}
          >
            <svg 
              width="18" 
              height="18" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2.5" 
              viewBox="0 0 24 24"
              style={{
                filter: location.pathname === '/chat' ? 'drop-shadow(0 0 6px rgba(139, 92, 246, 0.5))' : 'none'
              }}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <span style={{
              fontSize: '10px', 
              fontWeight: location.pathname === '/chat' ? '700' : '500', 
              marginTop: '2px',
              textShadow: location.pathname === '/chat' ? '0 0 6px rgba(139, 92, 246, 0.4)' : 'none'
            }}>Chat</span>
          </Link>

          {/* Enhanced Blog Tab */}
          <Link 
            to="/blog" 
            onClick={handleBlogClick}
            className={`mobile-nav-tab mobile-touch-target mobile-focus-ring haptic-feedback ${location.pathname.startsWith('/blog') ? 'mobile-nav-tab-active' : ''}`}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '4px',
              textDecoration: 'none',
              color: location.pathname.startsWith('/blog') ? '#f1f5f9' : 'rgb(156, 163, 175)',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              background: location.pathname.startsWith('/blog') 
                ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(99, 102, 241, 0.1) 100%)'
                : 'transparent',
              border: location.pathname.startsWith('/blog') 
                ? '1px solid rgba(139, 92, 246, 0.3)' 
                : '1px solid transparent',
              transform: location.pathname.startsWith('/blog') ? 'translateY(-2px) scale(1.05)' : 'translateY(0) scale(1)',
              boxShadow: location.pathname.startsWith('/blog') 
                ? '0 4px 12px rgba(139, 92, 246, 0.2)'
                : 'none',
            }}
          >
            <svg 
              width="18" 
              height="18" 
              fill="currentColor" 
              viewBox="0 0 24 24"
              style={{
                filter: location.pathname.startsWith('/blog') ? 'drop-shadow(0 0 4px rgba(139, 92, 246, 0.4))' : 'none'
              }}
            >
              <path d="M19 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V5a2 2 0 00-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"></path>
            </svg>
            <span style={{
              fontSize: '10px', 
              fontWeight: location.pathname.startsWith('/blog') ? '700' : '500', 
              marginTop: '2px',
              textShadow: location.pathname.startsWith('/blog') ? '0 0 6px rgba(139, 92, 246, 0.4)' : 'none'
            }}>Blog</span>
          </Link>

          {/* Enhanced About Tab */}
          <Link 
            to="/about" 
            onClick={handleAboutClick}
            className={`mobile-nav-tab mobile-touch-target mobile-focus-ring haptic-feedback ${location.pathname === '/about' ? 'mobile-nav-tab-active' : ''}`}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '4px',
              textDecoration: 'none',
              color: location.pathname === '/about' ? '#f1f5f9' : 'rgb(156, 163, 175)',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              background: location.pathname === '/about' 
                ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(99, 102, 241, 0.1) 100%)'
                : 'transparent',
              border: location.pathname === '/about' 
                ? '1px solid rgba(139, 92, 246, 0.3)' 
                : '1px solid transparent',
              transform: location.pathname === '/about' ? 'translateY(-2px) scale(1.05)' : 'translateY(0) scale(1)',
              boxShadow: location.pathname === '/about' 
                ? '0 4px 12px rgba(139, 92, 246, 0.2)'
                : 'none',
            }}
          >
            <svg 
              width="18" 
              height="18" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              viewBox="0 0 24 24"
              style={{
                filter: location.pathname === '/about' ? 'drop-shadow(0 0 4px rgba(139, 92, 246, 0.4))' : 'none'
              }}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span style={{
              fontSize: '10px', 
              fontWeight: location.pathname === '/about' ? '700' : '500', 
              marginTop: '2px',
              textShadow: location.pathname === '/about' ? '0 0 6px rgba(139, 92, 246, 0.4)' : 'none'
            }}>About</span>
          </Link>

          {/* Enhanced FAQ Tab */}
          <Link 
            to="/faq" 
            onClick={handleFAQClick}
            className={`mobile-nav-tab mobile-touch-target mobile-focus-ring haptic-feedback ${location.pathname === '/faq' ? 'mobile-nav-tab-active' : ''}`}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '4px',
              textDecoration: 'none',
              color: location.pathname === '/faq' ? '#f1f5f9' : 'rgb(156, 163, 175)',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              background: location.pathname === '/faq' 
                ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(99, 102, 241, 0.1) 100%)'
                : 'transparent',
              border: location.pathname === '/faq' 
                ? '1px solid rgba(139, 92, 246, 0.3)' 
                : '1px solid transparent',
              transform: location.pathname === '/faq' ? 'translateY(-2px) scale(1.05)' : 'translateY(0) scale(1)',
              boxShadow: location.pathname === '/faq' 
                ? '0 4px 12px rgba(139, 92, 246, 0.2)'
                : 'none',
            }}
          >
            <svg 
              width="18" 
              height="18" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              viewBox="0 0 24 24"
              style={{
                filter: location.pathname === '/faq' ? 'drop-shadow(0 0 4px rgba(139, 92, 246, 0.4))' : 'none'
              }}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span style={{
              fontSize: '10px', 
              fontWeight: location.pathname === '/faq' ? '700' : '500', 
              marginTop: '2px',
              textShadow: location.pathname === '/faq' ? '0 0 6px rgba(139, 92, 246, 0.4)' : 'none'
            }}>FAQ</span>
          </Link>

          {/* Enhanced Donate Tab */}
          <Link 
            to="/donate" 
            onClick={handleDonateClick}
            className={`mobile-nav-tab mobile-touch-target mobile-focus-ring haptic-feedback ${location.pathname === '/donate' ? 'mobile-nav-tab-active' : ''}`}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '4px',
              textDecoration: 'none',
              color: location.pathname === '/donate' ? '#f1f5f9' : 'rgb(156, 163, 175)',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              background: location.pathname === '/donate' 
                ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(99, 102, 241, 0.1) 100%)'
                : 'transparent',
              border: location.pathname === '/donate' 
                ? '1px solid rgba(139, 92, 246, 0.3)' 
                : '1px solid transparent',
              transform: location.pathname === '/donate' ? 'translateY(-2px) scale(1.05)' : 'translateY(0) scale(1)',
              boxShadow: location.pathname === '/donate' 
                ? '0 4px 12px rgba(139, 92, 246, 0.2)'
                : 'none',
            }}
          >
            <svg 
              width="18" 
              height="18" 
              fill="currentColor" 
              viewBox="0 0 24 24"
              style={{
                filter: location.pathname === '/donate' ? 'drop-shadow(0 0 4px rgba(139, 92, 246, 0.4))' : 'none'
              }}
            >
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"></path>
            </svg>
            <span style={{
              fontSize: '10px', 
              fontWeight: location.pathname === '/donate' ? '700' : '500', 
              marginTop: '2px',
              textShadow: location.pathname === '/donate' ? '0 0 6px rgba(139, 92, 246, 0.4)' : 'none'
            }}>Donate</span>
          </Link>
        </div>
      )}
    </>
  );
};

export default NavBar;
