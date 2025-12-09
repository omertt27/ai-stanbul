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
import { getTranslatedSuggestions, detectCategory } from '../../utils/quickReplyTranslations';
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
            onClick={() => onSelect(suggestion, index)}
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
export const getSmartSuggestions = (lastMessage, context = {}, language = 'en') => {
  if (!lastMessage) {
    return getTranslatedSuggestions('default', language);
  }

  const lowerMessage = lastMessage.toLowerCase();
  
  // Use translation system for context-aware suggestions
  const category = detectCategory(lastMessage);
  return getTranslatedSuggestions(category, language);
};

export default QuickReplies;
