import React, { useState } from 'react';

const CopyButton = ({ text }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const isLightMode = document.body.classList.contains('light');

  return (
    <button
      onClick={handleCopy}
      title={copied ? 'Copied!' : 'Copy'}
      style={{
        background: 'transparent',
        border: 'none',
        color: copied ? '#10b981' : (isLightMode ? '#9ca3af' : '#6b7280'),
        padding: '0.5rem',
        borderRadius: '0.5rem',
        cursor: 'pointer',
        fontSize: '1rem',
        transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '2rem',
        height: '2rem',
        minWidth: '2rem'
      }}
      onMouseOver={(e) => {
        if (!copied) {
          e.target.style.background = isLightMode ? 'rgba(229, 231, 235, 0.5)' : 'rgba(58, 59, 61, 0.3)';
          e.target.style.color = '#6366f1';
          e.target.style.transform = 'scale(1.05)';
        }
      }}
      onMouseOut={(e) => {
        if (!copied) {
          e.target.style.background = 'transparent';
          e.target.style.color = isLightMode ? '#9ca3af' : '#6b7280';
          e.target.style.transform = 'scale(1)';
        }
      }}
    >
      {copied ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20,6 9,17 4,12"></polyline>
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      )}
    </button>
  );
};

const ActionButtons = ({ text }) => {
  const handleReadAloud = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      window.speechSynthesis.speak(utterance);
    }
  };

  const isLightMode = document.body.classList.contains('light');

  return (
    <div style={{
      display: 'flex',
      gap: '0.5rem',
      marginTop: '0.75rem',
      alignItems: 'center'
    }}>
      <CopyButton text={text} />
      
      <button
        onClick={handleReadAloud}
        title="Read aloud"
        style={{
          background: 'transparent',
          border: 'none',
          color: isLightMode ? '#9ca3af' : '#6b7280',
          padding: '0.5rem',
          borderRadius: '0.5rem',
          cursor: 'pointer',
          fontSize: '1rem',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '2rem',
          height: '2rem',
          minWidth: '2rem'
        }}
        onMouseOver={(e) => {
          e.target.style.background = isLightMode ? 'rgba(229, 231, 235, 0.5)' : 'rgba(58, 59, 61, 0.3)';
          e.target.style.color = '#6366f1';
          e.target.style.transform = 'scale(1.05)';
        }}
        onMouseOut={(e) => {
          e.target.style.background = 'transparent';
          e.target.style.color = isLightMode ? '#9ca3af' : '#6b7280';
          e.target.style.transform = 'scale(1)';
        }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
      </button>
    </div>
  );
};

const formatMessage = (text, isUser = false) => {
  // Check if it's a restaurant recommendation message
  if (text.includes('🍽️') && text.includes('restaurants I found')) {
    const lines = text.split('\n');
    const header = lines[0]; // The header with emoji
    const restaurants = lines.slice(1).filter(line => line.trim().match(/^\d+\./));
    
    return (
      <div>
        <div style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '1rem', color: '#818cf8' }}>
          {header}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {restaurants.map((restaurant, idx) => {
            // Parse restaurant info
            const match = restaurant.match(/\d+\.\s\*\*(.*?)\*\*\s(⭐.*?)\s(💲.*?)\s\[(.*?)\]\((.*?)\)/);
            if (match) {
              const [, name, rating, price, linkText, url] = match;
              
              // Generate brief info about the restaurant based on name
              const getRestaurantInfo = (restaurantName) => {
                const lowerName = restaurantName.toLowerCase();
                if (lowerName.includes('kebab') || lowerName.includes('döner')) {
                  return "Authentic Turkish kebab house with traditional grilled meats and döner.";
                } else if (lowerName.includes('fish') || lowerName.includes('balık') || lowerName.includes('seafood')) {
                  return "Fresh seafood restaurant specializing in Bosphorus catch and Mediterranean dishes.";
                } else if (lowerName.includes('palace') || lowerName.includes('sultan') || lowerName.includes('ottoman')) {
                  return "Ottoman-style fine dining with traditional Turkish cuisine and elegant atmosphere.";
                } else if (lowerName.includes('café') || lowerName.includes('coffee') || lowerName.includes('kahve')) {
                  return "Cozy Turkish café serving traditional coffee, tea, and light Turkish breakfast.";
                } else if (lowerName.includes('meze') || lowerName.includes('taverna')) {
                  return "Traditional meze house offering small plates and authentic Turkish appetizers.";
                } else if (lowerName.includes('rooftop') || lowerName.includes('terrace')) {
                  return "Scenic rooftop dining with panoramic views of Istanbul and international cuisine.";
                } else if (lowerName.includes('lokanta') || lowerName.includes('ev yemek')) {
                  return "Home-style Turkish cooking with daily specials and comfort food classics.";
                } else if (lowerName.includes('pizza') || lowerName.includes('italian')) {
                  return "Italian restaurant serving authentic pizzas and Mediterranean dishes.";
                } else if (lowerName.includes('asian') || lowerName.includes('sushi') || lowerName.includes('chinese')) {
                  return "Asian fusion restaurant with fresh sushi and contemporary Asian flavors.";
                } else {
                  return "Popular local restaurant known for quality Turkish cuisine and warm hospitality.";
                }
              };
              
              const isLightMode = document.body.classList.contains('light');
              
              return (
                <div key={idx} style={{
                  background: isLightMode ? 'rgba(255, 255, 255, 0.9)' : 'rgba(30, 31, 33, 0.7)',
                  padding: '0.75rem 1rem',
                  borderRadius: '0.5rem',
                  border: isLightMode ? '1px solid rgba(229, 231, 235, 0.8)' : '1px solid rgba(58, 59, 61, 0.5)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.25rem' }}>
                    <h4 style={{ margin: 0, color: isLightMode ? '#1f2937' : '#f3f4f6', fontSize: '1rem', fontWeight: '600' }}>
                      {name.trim()}
                    </h4>
                    <span style={{ color: isLightMode ? '#6b7280' : '#a3a7c2', fontSize: '0.9rem' }}>{price}</span>
                  </div>
                  
                  {/* Restaurant info sentence */}
                  <p style={{ 
                    margin: '0.5rem 0', 
                    color: isLightMode ? '#4b5563' : '#c9d1d9', 
                    fontSize: '0.85rem', 
                    lineHeight: '1.4',
                    fontStyle: 'italic'
                  }}>
                    {getRestaurantInfo(name)}
                  </p>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: isLightMode ? '#374151' : '#e5e7eb', fontSize: '0.9rem' }}>{rating}</span>
                    <a 
                      href={url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{
                        color: '#818cf8',
                        textDecoration: 'none',
                        fontSize: '0.85rem',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '0.25rem',
                        background: isLightMode ? 'rgba(229, 231, 235, 0.5)' : 'rgba(58, 59, 61, 0.3)',
                        transition: 'background 0.2s'
                      }}
                      onMouseOver={(e) => e.target.style.background = isLightMode ? 'rgba(229, 231, 235, 0.8)' : 'rgba(58, 59, 61, 0.5)'}
                      onMouseOut={(e) => e.target.style.background = isLightMode ? 'rgba(229, 231, 235, 0.5)' : 'rgba(58, 59, 61, 0.3)'}
                    >
                      📍 View on Maps
                    </a>
                  </div>
                </div>
              );
            }
            return <div key={idx}>{restaurant}</div>;
          })}
        </div>
      </div>
    );
  }
  
  // For other messages, handle basic markdown-like formatting
  let formattedText = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br />');
  
  return (
    <div dangerouslySetInnerHTML={{ __html: formattedText }} />
  );
};

const Chat = ({ messages }) => (
  <div style={{ 
    maxWidth: '100%',
    padding: '1rem 0',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem'
  }}>
    {messages.map((msg, idx) => {
      const isUser = msg.user === 'You';
      
      if (isUser) {
        // User messages: Right-aligned with box
        return (
          <div key={idx} style={{
            display: 'flex',
            justifyContent: 'flex-end',
            padding: '0 1rem',
            maxWidth: '100%'
          }}>
            <div style={{
              maxWidth: '55%',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              padding: '0.75rem 1rem',
              borderRadius: '1.125rem',
              fontSize: '0.95rem',
              lineHeight: '1.4',
              wordBreak: 'break-word'
            }}>
              {formatMessage(msg.text, true)}
            </div>
          </div>
        );
      } else {
        // KAM messages: Left-aligned without avatar
        return (
          <div key={idx} style={{
            display: 'flex',
            gap: '0.75rem',
            alignItems: 'flex-start',
            padding: '0 1rem',
            maxWidth: '100%'
          }}>
            {/* Content */}
            <div style={{ flex: 1, minWidth: 0 }}>
              {/* Name */}
              <div style={{
                fontSize: '0.875rem',
                fontWeight: '600',
                color: document.body.classList.contains('light') ? '#475569' : '#f3f4f6',
                marginBottom: '0.5rem'
              }}>
                KAM
              </div>
              
              {/* Message */}
              <div style={{
                color: document.body.classList.contains('light') ? '#334155' : '#e5e7eb',
                lineHeight: '1.6',
                fontSize: '0.95rem'
              }}>
                {formatMessage(msg.text, false)}
              </div>
              
              {/* Action buttons for AI messages */}
              <ActionButtons text={msg.text} />
            </div>
          </div>
        );
      }
    })}
  </div>
);

export default Chat;
