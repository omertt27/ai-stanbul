import React, { useState, useEffect } from 'react';

const CopyrightNotice = () => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Show notice after a delay
    const timer = setTimeout(() => setVisible(true), 2000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div 
      className={`fixed bottom-4 right-4 z-50 transition-all duration-500 ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
      }`}
      style={{
        background: 'linear-gradient(135deg, #ff4444, #cc0000)',
        color: 'white',
        padding: '12px 16px',
        borderRadius: '8px',
        fontSize: '12px',
        fontWeight: 'bold',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        border: '1px solid rgba(255,255,255,0.2)',
        maxWidth: '250px',
        pointerEvents: 'none',
        userSelect: 'none'
      }}
    >
      <div className="flex items-center space-x-2">
        <span>🔒</span>
        <div>
          <div>© AI Istanbul {new Date().getFullYear()}</div>
          <div style={{ fontSize: '10px', opacity: 0.9 }}>
            Protected Content
          </div>
        </div>
      </div>
    </div>
  );
};

export default CopyrightNotice;
