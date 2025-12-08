/**
 * SkeletonMessage Component
 * ==========================
 * Loading skeleton for chat messages
 * Shows animated placeholder while AI generates response
 * Better UX than spinning indicator
 * 
 * Features:
 * - Smooth shimmer animation
 * - Multiple skeleton variations
 * - Dark/light mode support
 * - Mobile-optimized sizing
 */

import React from 'react';
import './SkeletonMessage.css';

const SkeletonMessage = ({ darkMode = false, count = 1 }) => {
  return (
    <div className={`skeleton-message-container ${darkMode ? 'dark' : 'light'}`}>
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className="skeleton-message">
          <div className="skeleton-avatar"></div>
          <div className="skeleton-content">
            <div className="skeleton-line skeleton-line-1"></div>
            <div className="skeleton-line skeleton-line-2"></div>
            <div className="skeleton-line skeleton-line-3"></div>
          </div>
        </div>
      ))}
    </div>
  );
};

// Compact skeleton for inline loading
export const SkeletonMessageCompact = ({ darkMode = false }) => {
  return (
    <div className={`skeleton-message-compact ${darkMode ? 'dark' : 'light'}`}>
      <div className="skeleton-dots">
        <div className="skeleton-dot"></div>
        <div className="skeleton-dot"></div>
        <div className="skeleton-dot"></div>
      </div>
    </div>
  );
};

export default SkeletonMessage;
