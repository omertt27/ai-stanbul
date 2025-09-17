import React, { useState, useEffect } from 'react';

const LiveActivityFeed = () => {
  const [onlineCount, setOnlineCount] = useState(0);
  const [currentTrend, setCurrentTrend] = useState(0);

  const trendingTopics = [
    { text: 'Turkish Coffee Culture', icon: 'C', growth: '+45%' },
    { text: 'Bosphorus Ferry Routes', icon: 'F', growth: '+32%' },
    { text: 'Rooftop Restaurants', icon: 'R', growth: '+28%' },
    { text: 'Street Food Guide', icon: 'S', growth: '+25%' },
    { text: 'Photography Spots', icon: 'P', growth: '+22%' },
    { text: 'Hidden Neighborhoods', icon: 'H', growth: '+35%' },
    { text: 'Traditional Markets', icon: 'M', growth: '+29%' }
  ];

  useEffect(() => {
    // Update online count with realistic variation
    const countInterval = setInterval(() => {
      setOnlineCount(prev => {
        const base = 45;
        const timeBonus = Math.floor(new Date().getHours() / 2); // More active during day
        const variation = Math.floor(Math.random() * 15) - 7;
        return Math.max(base + timeBonus + variation, 1);
      });
    }, 5000);

    // Cycle through trending topics
    const trendInterval = setInterval(() => {
      setCurrentTrend(prev => (prev + 1) % trendingTopics.length);
    }, 4000);

    // Initialize
    setOnlineCount(52);

    return () => {
      clearInterval(countInterval);
      clearInterval(trendInterval);
    };
  }, [trendingTopics.length]);

  const currentTrendData = trendingTopics[currentTrend];

  return (
    <div className="live-activity-feed-minimal">
      {/* Simplified Online Counter */}
      <div className="minimal-online-counter">
        <div className="pulse-indicator">
          <div className="pulse-dot"></div>
        </div>
        <span className="online-text">
          <strong>{onlineCount}</strong> active users
        </span>
      </div>

      {/* Rotating Trending Topic */}
      <div className="minimal-trending">
        <div className="trending-indicator">
          <span className="trending-icon">▲</span>
        </div>
        <div className="trending-content">
          <span className="trending-label">Trending:</span>
          <span className="trending-topic">
            {currentTrendData.icon} {currentTrendData.text}
          </span>
          <span className="trending-growth">{currentTrendData.growth}</span>
        </div>
      </div>
    </div>
  );
};

export default LiveActivityFeed;
