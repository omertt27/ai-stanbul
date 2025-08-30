import React from 'react';

const formatMessage = (text) => {
  // Check if it's a restaurant recommendation message
  if (text.includes('🍽️') && text.includes('restaurants I found')) {
    const lines = text.split('\n');
    const header = lines[0]; // The header with emoji
    const restaurants = lines.slice(1).filter(line => line.trim().startsWith(/\d+\./));
    
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
              return (
                <div key={idx} style={{
                  background: 'rgba(129, 140, 248, 0.08)',
                  padding: '0.75rem 1rem',
                  borderRadius: '0.5rem',
                  border: '1px solid rgba(129, 140, 248, 0.15)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.25rem' }}>
                    <h4 style={{ margin: 0, color: '#f3f4f6', fontSize: '1rem', fontWeight: '600' }}>
                      {name.trim()}
                    </h4>
                    <span style={{ color: '#a3a7c2', fontSize: '0.9rem' }}>{price}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: '#e5e7eb', fontSize: '0.9rem' }}>{rating}</span>
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
                        background: 'rgba(129, 140, 248, 0.15)',
                        transition: 'background 0.2s'
                      }}
                      onMouseOver={(e) => e.target.style.background = 'rgba(129, 140, 248, 0.25)'}
                      onMouseOut={(e) => e.target.style.background = 'rgba(129, 140, 248, 0.15)'}
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
  return (
    <div dangerouslySetInnerHTML={{
      __html: text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br />')
    }} />
  );
};

const Chat = ({ messages }) => (
  <div className="chat-container">
    {messages.map((msg, idx) => (
      <div key={idx} className="message-row">
        <div className={`message-bubble ${msg.user === 'You' ? 'user' : 'bot'}`}>
          {formatMessage(msg.text)}
        </div>
      </div>
    ))}
  </div>
);

export default Chat;
