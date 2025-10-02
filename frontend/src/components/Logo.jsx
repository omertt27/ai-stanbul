import React from 'react';

const Logo = ({ 
  size = 'medium', 
  onClick = null, 
  style = {},
  className = '',
  isMobile = false
}) => {
  // Size configurations
  const sizeConfig = {
    small: {
      fontSize: '1.6rem',
      fontWeight: 800,
      stanbulFontSize: '1.4rem',
      stanbulFontWeight: 500,
    },
    medium: {
      fontSize: '2.6rem',
      fontWeight: 700,
      stanbulFontSize: '2.6rem',
      stanbulFontWeight: 400,
    },
    large: {
      fontSize: '3.5rem',
      fontWeight: 700,
      stanbulFontSize: '3.5rem',
      stanbulFontWeight: 400,
    }
  };

  const config = sizeConfig[size] || sizeConfig.medium;

  // Mobile navbar specific styling
  const mobileNavbarStyle = isMobile && size === 'small' ? {
    cursor: 'pointer',
    pointerEvents: 'auto',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    display: 'flex',
    alignItems: 'center',
    padding: '8px 12px',
    borderRadius: '16px',
    background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(99, 102, 241, 0.08) 100%)',
    border: '1px solid rgba(139, 92, 246, 0.25)',
    boxShadow: '0 4px 16px rgba(139, 92, 246, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(10px)',
  } : {};

  // Desktop navbar specific styling
  const desktopNavbarStyle = !isMobile && size === 'medium' && onClick ? {
    cursor: 'pointer',
    pointerEvents: 'auto',
    transition: 'transform 0.2s ease, opacity 0.2s ease',
    display: 'flex',
    alignItems: 'center',
  } : {};

  // Large size (page header) specific styling
  const pageHeaderStyle = size === 'large' ? {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: '2rem'
  } : {};

  const logoTextStyle = {
    fontSize: config.fontSize,
    fontWeight: config.fontWeight,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    background: isMobile && size === 'small' 
      ? 'linear-gradient(135deg, #f1f5f9 0%, #8b5cf6 40%, #6366f1 70%, #7c3aed 100%)'
      : 'linear-gradient(90deg, #e5e7eb 0%, #8b5cf6 50%, #6366f1 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    textShadow: isMobile && size === 'small' 
      ? '0 2px 12px rgba(139, 92, 246, 0.4)'
      : '0 2px 10px rgba(139, 92, 246, 0.3)',
    transition: 'all 0.3s ease',
    cursor: onClick ? 'pointer' : 'default',
    filter: isMobile && size === 'small' ? 'drop-shadow(0 0 8px rgba(139, 92, 246, 0.3))' : 'none',
  };

  const containerStyle = {
    ...mobileNavbarStyle,
    ...desktopNavbarStyle,
    ...pageHeaderStyle,
    ...style,
  };

  return (
    <div 
      className={className}
      style={containerStyle}
      onClick={onClick}
    >
      <span style={logoTextStyle}>
        A/<span style={{
          fontWeight: config.stanbulFontWeight,
          fontSize: config.stanbulFontSize
        }}>STANBUL</span>
      </span>
    </div>
  );
};

export default Logo;
