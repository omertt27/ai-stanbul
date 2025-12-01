/**
 * JumpToBottomFAB Component
 * =========================
 * Floating Action Button that appears when user scrolls up
 * Smoothly scrolls back to bottom of chat
 * 
 * Features:
 * - Only visible when scrolled up
 * - Shows unread message count badge
 * - Smooth scroll animation
 * - ChatGPT-style design
 */

import React, { useState, useEffect } from 'react';
import './JumpToBottomFAB.css';

const JumpToBottomFAB = ({ 
  containerRef, 
  unreadCount = 0,
  darkMode = false,
  bottomOffset = 80 // Offset from bottom (above input)
}) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const container = containerRef?.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
      
      // Show button if scrolled up more than 200px from bottom
      setIsVisible(distanceFromBottom > 200);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, [containerRef]);

  const scrollToBottom = () => {
    const container = containerRef?.current;
    if (!container) return;

    container.scrollTo({
      top: container.scrollHeight,
      behavior: 'smooth'
    });
  };

  if (!isVisible) return null;

  return (
    <button
      onClick={scrollToBottom}
      className={`jump-to-bottom-fab ${darkMode ? 'dark' : 'light'}`}
      style={{ bottom: `${bottomOffset}px` }}
      aria-label="Jump to bottom"
    >
      {unreadCount > 0 && (
        <span className="unread-badge">{unreadCount > 9 ? '9+' : unreadCount}</span>
      )}
      <svg 
        width="20" 
        height="20" 
        viewBox="0 0 24 24" 
        fill="none" 
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <polyline points="6 9 12 15 18 9"></polyline>
      </svg>
    </button>
  );
};

export default JumpToBottomFAB;
