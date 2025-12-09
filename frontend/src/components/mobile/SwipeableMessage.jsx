/**
 * SwipeableMessage Component
 * ===========================
 * Mobile-optimized message bubble with swipe gestures
 * 
 * Features:
 * - Swipe left to delete
 * - Swipe right to copy
 * - Visual feedback during swipe
 * - Haptic feedback (if supported)
 * - Smooth animations
 * - Configurable actions
 */

import React, { useState, useRef, useEffect } from 'react';
import { trackEvent } from '../../utils/analytics';
import './SwipeableMessage.css';

const SwipeableMessage = ({ 
  children,
  onSwipeLeft,
  onSwipeRight,
  leftAction = 'delete',  // 'delete', 'reply', 'forward'
  rightAction = 'copy',   // 'copy', 'share', 'bookmark'
  darkMode = false,
  className = ''
}) => {
  const [touchStartX, setTouchStartX] = useState(0);
  const [touchCurrentX, setTouchCurrentX] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [showAction, setShowAction] = useState(null);
  const containerRef = useRef(null);

  const swipeThreshold = 100; // Minimum swipe distance to trigger action
  const maxSwipe = 150; // Maximum swipe distance

  const handleTouchStart = (e) => {
    setTouchStartX(e.touches[0].clientX);
    setIsDragging(true);
  };

  const handleTouchMove = (e) => {
    if (!isDragging) return;

    const currentX = e.touches[0].clientX;
    const diff = currentX - touchStartX;
    
    // Limit swipe distance
    const limitedDiff = Math.max(-maxSwipe, Math.min(maxSwipe, diff));
    setTouchCurrentX(limitedDiff);

    // Show action indicator
    if (Math.abs(limitedDiff) > swipeThreshold / 2) {
      setShowAction(limitedDiff < 0 ? leftAction : rightAction);
    } else {
      setShowAction(null);
    }
  };

  const handleTouchEnd = () => {
    const swipeDistance = touchCurrentX;

    // Trigger action if threshold met
    if (Math.abs(swipeDistance) >= swipeThreshold) {
      // Haptic feedback
      if ('vibrate' in navigator) {
        navigator.vibrate(50);
      }

      if (swipeDistance < 0 && onSwipeLeft) {
        // Track swipe gesture (analytics)
        try {
          trackEvent('swipe_gesture', { action: leftAction || 'left', context: 'ai' });
        } catch (e) {
          console.warn('Analytics tracking failed:', e);
        }
        onSwipeLeft();
      } else if (swipeDistance > 0 && onSwipeRight) {
        // Track swipe gesture (analytics)
        try {
          trackEvent('swipe_gesture', { action: rightAction || 'right', context: 'ai' });
        } catch (e) {
          console.warn('Analytics tracking failed:', e);
        }
        onSwipeRight();
      }
    }

    // Reset
    setIsDragging(false);
    setTouchCurrentX(0);
    setShowAction(null);
  };

  const getActionIcon = (action) => {
    const icons = {
      delete: '🗑️',
      copy: '📋',
      reply: '↩️',
      forward: '➡️',
      share: '📤',
      bookmark: '🔖'
    };
    return icons[action] || '✓';
  };

  const getActionColor = (action) => {
    const colors = {
      delete: '#ef4444',
      copy: '#10b981',
      reply: '#3b82f6',
      forward: '#8b5cf6',
      share: '#06b6d4',
      bookmark: '#f59e0b'
    };
    return colors[action] || '#6b7280';
  };

  return (
    <div 
      ref={containerRef}
      className={`swipeable-message ${darkMode ? 'dark' : 'light'} ${className}`}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      {/* Left action indicator */}
      {showAction === leftAction && touchCurrentX < 0 && (
        <div 
          className="action-indicator left"
          style={{ 
            backgroundColor: getActionColor(leftAction),
            opacity: Math.min(Math.abs(touchCurrentX) / swipeThreshold, 1)
          }}
        >
          <span className="action-icon">{getActionIcon(leftAction)}</span>
        </div>
      )}

      {/* Message content */}
      <div 
        className="message-content"
        style={{
          transform: `translateX(${touchCurrentX}px)`,
          transition: isDragging ? 'none' : 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
        }}
      >
        {children}
      </div>

      {/* Right action indicator */}
      {showAction === rightAction && touchCurrentX > 0 && (
        <div 
          className="action-indicator right"
          style={{ 
            backgroundColor: getActionColor(rightAction),
            opacity: Math.min(touchCurrentX / swipeThreshold, 1)
          }}
        >
          <span className="action-icon">{getActionIcon(rightAction)}</span>
        </div>
      )}
    </div>
  );
};

export default SwipeableMessage;
