/**
 * QuickActions Component
 * 
 * Displays contextual action chips after AI responses
 * to guide users toward common follow-up questions
 * 
 * Examples:
 * - After route query: "How long?", "Alternatives?", "Cost?"
 * - After restaurant: "Similar places?", "Menu?", "Directions?"
 * - After attraction: "Opening hours?", "Tickets?", "How to get there?"
 */

import React from 'react';

/**
 * Get relevant quick actions based on message intent/type
 */
const getQuickActions = (message) => {
  const intent = message.intent || message.type || 'general';
  
  // Route/Transportation queries
  if (intent === 'transportation' || intent === 'directions' || intent === 'route' || message.mapData?.route_info) {
    return [
      { id: 'duration', icon: '⏱️', text: 'How long?', query: 'How long does this route take?' },
      { id: 'alternatives', icon: '🔀', text: 'Alternatives?', query: 'Show me alternative routes' },
      { id: 'cost', icon: '💰', text: 'Cost?', query: 'How much does this route cost?' },
      { id: 'accessibility', icon: '♿', text: 'Accessible?', query: 'Is this route wheelchair accessible?' }
    ];
  }
  
  // Restaurant queries
  if (intent === 'restaurant-recommendation' || message.restaurants) {
    return [
      { id: 'similar', icon: '🍽️', text: 'Similar places?', query: 'Show me similar restaurants' },
      { id: 'directions', icon: '🗺️', text: 'Directions?', query: 'How do I get there?' },
      { id: 'hours', icon: '🕐', text: 'Opening hours?', query: 'What are the opening hours?' },
      { id: 'price', icon: '💵', text: 'Price range?', query: 'What is the price range?' }
    ];
  }
  
  // Places/Attractions queries
  if (intent === 'places-recommendation' || intent === 'attraction') {
    return [
      { id: 'hours', icon: '🕐', text: 'Opening hours?', query: 'What are the opening hours?' },
      { id: 'tickets', icon: '🎫', text: 'Tickets?', query: 'How much are tickets?' },
      { id: 'directions', icon: '🗺️', text: 'How to get there?', query: 'How can I get there?' },
      { id: 'nearby', icon: '📍', text: 'What else nearby?', query: 'What other attractions are nearby?' }
    ];
  }
  
  // Weather queries
  if (intent === 'weather') {
    return [
      { id: 'forecast', icon: '🌤️', text: 'Forecast?', query: 'What is the weather forecast?' },
      { id: 'recommendations', icon: '🌂', text: 'Recommendations?', query: 'What should I do in this weather?' }
    ];
  }
  
  // General/default actions
  return [
    { id: 'route', icon: '🚇', text: 'How to get there?', query: 'How do I get there?' },
    { id: 'restaurants', icon: '🍽️', text: 'Restaurants nearby?', query: 'Show me restaurants nearby' },
    { id: 'attractions', icon: '🏛️', text: 'What to see?', query: 'What should I see in Istanbul?' }
  ];
};

/**
 * QuickActions Component
 * 
 * @param {Object} props
 * @param {Object} props.message - The message object to generate actions for
 * @param {Function} props.onActionClick - Callback when action is clicked
 * @param {boolean} props.darkMode - Dark mode flag
 * @param {number} props.maxActions - Maximum number of actions to show (default: 4)
 */
const QuickActions = ({ 
  message, 
  onActionClick, 
  darkMode = false,
  maxActions = 4 
}) => {
  const actions = getQuickActions(message).slice(0, maxActions);
  
  if (actions.length === 0) {
    return null;
  }
  
  return (
    <div className="mt-4">
      <div className={`text-xs font-medium mb-2 ${
        darkMode ? 'text-gray-400' : 'text-gray-500'
      }`}>
        💡 Quick actions:
      </div>
      <div className="flex flex-wrap gap-2">
        {actions.map((action) => (
          <button
            key={action.id}
            onClick={() => onActionClick(action.query)}
            className={`
              flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium
              transition-all duration-200 transform hover:scale-105 active:scale-95
              ${darkMode 
                ? 'bg-gray-700 hover:bg-gray-600 text-gray-200 border border-gray-600' 
                : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-200 shadow-sm hover:shadow-md'
              }
            `}
            title={`Ask: ${action.query}`}
          >
            <span className="text-base">{action.icon}</span>
            <span>{action.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default QuickActions;
