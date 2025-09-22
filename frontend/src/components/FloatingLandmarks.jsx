import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const FloatingLandmarks = ({ onQuickStart }) => {
  const navigate = useNavigate();
  const [activeIndex, setActiveIndex] = useState(0);

  const landmarks = [
    {
      icon: '◆',
      name: 'Hagia Sophia',
      description: 'Ancient Byzantine cathedral',
      query: 'tell me about Hagia Sophia history and visiting hours',
      position: { top: '20%', left: '15%' },
      color: '#e11d48'
    },
    {
      icon: '◇',
      name: 'Galata Tower',
      description: 'Medieval stone tower',
      query: 'Galata Tower views and how to visit',
      position: { top: '15%', right: '20%' },
      color: '#7c3aed'
    },
    {
      icon: '◈',
      name: 'Bosphorus Bridge',
      description: 'Connecting Europe & Asia',
      query: 'Bosphorus Bridge history and best viewpoints',
      position: { top: '40%', right: '15%' },
      color: '#0891b2'
    },
    {
      icon: '◉',
      name: 'Grand Bazaar',
      description: 'Historic covered market',
      query: 'Grand Bazaar shopping guide and what to buy',
      position: { bottom: '25%', left: '20%' },
      color: '#dc2626'
    },
    {
      icon: '◎',
      name: 'Topkapi Palace',
      description: 'Ottoman imperial palace',
      query: 'Topkapi Palace tours and Ottoman history',
      position: { bottom: '20%', right: '25%' },
      color: '#059669'
    },
    {
      icon: '◐',
      name: 'Blue Mosque',
      description: 'Ottoman architectural marvel',
      query: 'Blue Mosque visiting guidelines and architecture',
      position: { top: '60%', left: '25%' },
      color: '#1d4ed8'
    }
  ];

  // Auto-rotate active landmark every 4 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % landmarks.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [landmarks.length]);

  const handleLandmarkClick = (landmark) => {
    onQuickStart(landmark.query);
    navigate('/chat');
  };

  return (
    <div className="floating-landmarks">
      {landmarks.map((landmark, index) => (
        <div
          key={landmark.name}
          className={`floating-landmark ${index === activeIndex ? 'active' : ''}`}
          style={{
            position: 'absolute',
            ...landmark.position,
            animationDelay: `${index * 0.5}s`
          }}
          onClick={() => handleLandmarkClick(landmark)}
        >
          <div 
            className="landmark-bubble"
            style={{ 
              '--landmark-color': landmark.color,
              backgroundColor: `${landmark.color}20`,
              borderColor: `${landmark.color}60`
            }}
          >
            <div className="landmark-icon">{landmark.icon}</div>
            <div className="landmark-name">{landmark.name}</div>
            <div className="landmark-description">{landmark.description}</div>
            
            {/* Ripple effect */}
            <div className="ripple-effect"></div>
          </div>
          
          {/* Floating particles */}
          <div className="floating-particles">
            {[...Array(3)].map((_, i) => (
              <div 
                key={i} 
                className="particle" 
                style={{ 
                  animationDelay: `${i * 0.8}s`,
                  backgroundColor: landmark.color
                }}
              ></div>
            ))}
          </div>
        </div>
      ))}
      
      {/* Landmark explorer hint */}
      <div className="landmark-hint">
        <div className="hint-bubble">
          <span className="hint-icon">✨</span>
          <span className="hint-text">Click any landmark to explore</span>
        </div>
      </div>
    </div>
  );
};

export default FloatingLandmarks;
