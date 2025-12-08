/**
 * QuickReplies Component
 * ======================
 * Context-aware quick reply chips for mobile chat
 * Displays smart suggestions based on conversation context
 * 
 * Features:
 * - Contextual suggestions
 * - Smooth animations
 * - Mobile-optimized tap targets (44px minimum)
 * - Horizontal scrolling on small screens
 * - Auto-hide after use
 */

import React from 'react';
import './QuickReplies.css';

const QuickReplies = ({ 
  suggestions = [], 
  onSelect, 
  darkMode = false,
  visible = true 
}) => {
  if (!visible || suggestions.length === 0) return null;

  return (
    <div className={`quick-replies-container ${darkMode ? 'dark' : 'light'}`}>
      <div className="quick-replies-scroll">
        {suggestions.map((suggestion, index) => (
          <button
            key={`${suggestion}-${index}`}
            onClick={() => onSelect(suggestion)}
            className="quick-reply-chip"
            aria-label={`Send quick reply: ${suggestion}`}
          >
            <span className="chip-text">{suggestion}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

// Helper function to generate smart suggestions based on context
export const getSmartSuggestions = (lastMessage, context = {}) => {
  if (!lastMessage) {
    return ['Show restaurants', 'Find attractions', 'Get directions'];
  }

  const lowerMessage = lastMessage.toLowerCase();

  // Restaurant context
  if (lowerMessage.includes('restaurant') || lowerMessage.includes('dining')) {
    return ['Show on map', 'Get directions', 'More options', 'Find nearby'];
  }

  // Attraction/place context
  if (lowerMessage.includes('museum') || lowerMessage.includes('attraction') || 
      lowerMessage.includes('landmark') || lowerMessage.includes('tower')) {
    return ['Show on map', 'Opening hours', 'How to get there', 'More like this'];
  }

  // Direction context
  if (lowerMessage.includes('direction') || lowerMessage.includes('get there') ||
      lowerMessage.includes('how do i go')) {
    return ['Start navigation', 'Public transport', 'Walking route', 'Drive there'];
  }

  // Location context
  if (lowerMessage.includes('taksim') || lowerMessage.includes('sultanahmet') ||
      lowerMessage.includes('beyoğlu') || lowerMessage.includes('galata')) {
    return ['Show restaurants', 'Find attractions', 'Get directions', 'Tell me more'];
  }

  // Question context
  if (lowerMessage.endsWith('?')) {
    return ['Yes', 'No', 'Tell me more', 'Show examples'];
  }

  // Weather context
  if (lowerMessage.includes('weather') || lowerMessage.includes('temperature')) {
    return ['5-day forecast', 'What to wear', 'Best time to visit', 'Indoor activities'];
  }

  // Default suggestions
  return ['Show restaurants', 'Find attractions', 'Get directions', 'Tell me more'];
};

export default QuickReplies;
