import React from 'react';
import { useTheme } from '../contexts/ThemeContext';

const ActionButtons = ({ actions = [], contextActions = [] }) => {
  const { theme } = useTheme();
  
  const allActions = [...actions, ...contextActions];
  
  if (!allActions.length) return null;
  
  const handleActionClick = (action) => {
    if (action.url) {
      // Open in new tab for external links
      window.open(action.url, '_blank', 'noopener,noreferrer');
    } else if (action.info) {
      // Show info in alert for now (could be improved with modal)
      alert(action.info);
    }
  };
  
  const getActionIcon = (type) => {
    const icons = {
      navigation: '📍',
      directions: '🗺️',
      booking: '🍽️',
      tickets: '🎫',
      tour: '🗺️',
      schedule: '⛴️',
      metro: '🚇',
      context: '💡'
    };
    return icons[type] || '🔗';
  };
  
  const getActionColor = (type) => {
    const colors = {
      navigation: 'bg-blue-500 hover:bg-blue-600',
      directions: 'bg-green-500 hover:bg-green-600', 
      booking: 'bg-orange-500 hover:bg-orange-600',
      tickets: 'bg-purple-500 hover:bg-purple-600',
      tour: 'bg-indigo-500 hover:bg-indigo-600',
      schedule: 'bg-teal-500 hover:bg-teal-600',
      metro: 'bg-red-500 hover:bg-red-600',
      context: 'bg-gray-500 hover:bg-gray-600'
    };
    return colors[type] || 'bg-blue-500 hover:bg-blue-600';
  };
  
  return (
    <div className={`mt-4 space-y-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>
      <h4 className="font-semibold text-sm">Quick Actions:</h4>
      <div className="flex flex-wrap gap-2">
        {allActions.slice(0, 6).map((action, index) => (
          <button
            key={index}
            onClick={() => handleActionClick(action)}
            className={`
              ${getActionColor(action.type)}
              text-white text-xs px-3 py-2 rounded-lg
              flex items-center gap-1
              transition-all duration-200
              hover:scale-105 active:scale-95
              shadow-sm hover:shadow-md
            `}
            title={action.info || action.text}
          >
            <span>{getActionIcon(action.type)}</span>
            <span className="truncate max-w-32">
              {action.text.replace(/📍|🗺️|🍽️|🎫|⛴️|🚇|💡/g, '').trim()}
            </span>
          </button>
        ))}
      </div>
      
      {allActions.length > 6 && (
        <p className="text-xs text-gray-500 mt-2">
          + {allActions.length - 6} more actions available
        </p>
      )}
    </div>
  );
};

export default ActionButtons;
