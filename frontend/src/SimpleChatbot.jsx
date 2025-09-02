import React from 'react';

const SimpleChatbot = () => {
  return (
    <div style={{ padding: '20px', minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      <h1>Simple Chatbot - Test</h1>
      <div style={{ 
        backgroundColor: 'white', 
        padding: '20px', 
        borderRadius: '10px',
        marginTop: '20px',
        border: '1px solid #ccc'
      }}>
        <h2>KAM - Your AI Istanbul Guide</h2>
        <p>
          <strong>Kam</strong>, in Turkish, Altaic, and Mongolian folk culture, is a shaman, a religious leader, wisdom person. Also referred to as "Gam" or Ham.
        </p>
        <p>
          <em>A religious leader believed to communicate with supernatural powers within communities.</em>
        </p>
      </div>
      <div style={{ marginTop: '20px' }}>
        <input 
          type="text" 
          placeholder="Ask about Istanbul..." 
          style={{ 
            width: '300px', 
            padding: '10px', 
            border: '1px solid #ccc',
            borderRadius: '5px'
          }}
        />
        <button 
          style={{ 
            padding: '10px 20px', 
            marginLeft: '10px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default SimpleChatbot;
