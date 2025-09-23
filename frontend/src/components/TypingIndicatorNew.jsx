import React from 'react';

const TypingIndicator = ({ message = "" }) => {
  const isLightMode = document.body.classList.contains('light');

  const containerStyle = {
    display: 'flex',
    alignItems: 'center',
    padding: '1rem 1.5rem',
    marginBottom: '1rem',
    background: isLightMode 
      ? 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%)'
      : 'linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.95) 100%)',
    border: isLightMode 
      ? '1px solid rgba(226, 232, 240, 0.8)'
      : '1px solid rgba(71, 85, 105, 0.3)',
    borderRadius: '1rem',
    backdropFilter: 'blur(12px)',
    boxShadow: isLightMode
      ? '0 4px 20px rgba(99, 102, 241, 0.08)'
      : '0 4px 20px rgba(0, 0, 0, 0.25)',
    maxWidth: '80%',
    alignSelf: 'flex-start'
  };

  const avatarStyle = {
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    background: isLightMode 
      ? 'linear-gradient(135deg, #6366f1, #8b5cf6)'
      : 'linear-gradient(135deg, #818cf8, #a78bfa)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: '12px',
    flexShrink: 0
  };

  const textStyle = {
    color: isLightMode ? '#475569' : '#e2e8f0',
    fontSize: '0.95rem',
    fontWeight: '500',
    marginRight: '0.75rem'
  };

  const dotsContainerStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  };

  const dotStyle = {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: isLightMode 
      ? 'linear-gradient(135deg, #6366f1, #8b5cf6)'
      : 'linear-gradient(135deg, #818cf8, #a78bfa)',
    animation: 'typingDots 1.4s infinite ease-in-out'
  };

  return (
    <div style={containerStyle}>
      <style>
        {`
          @keyframes typingDots {
            0%, 80%, 100% {
              transform: scale(0.8);
              opacity: 0.5;
            }
            40% {
              transform: scale(1);
              opacity: 1;
            }
          }
        `}
      </style>
      <div style={avatarStyle}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
          <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
        </svg>
      </div>
      <span style={textStyle}>{message}</span>
      <div style={dotsContainerStyle}>
        <div style={{...dotStyle, animationDelay: '0s'}}></div>
        <div style={{...dotStyle, animationDelay: '0.2s'}}></div>
        <div style={{...dotStyle, animationDelay: '0.4s'}}></div>
      </div>
    </div>
  );
};

export default TypingIndicator;
