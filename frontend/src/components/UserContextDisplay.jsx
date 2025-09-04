import React from 'react';
import { useTheme } from '../contexts/ThemeContext';

const UserContextDisplay = ({ userContext = {} }) => {
  const { theme } = useTheme();
  
  if (!userContext || Object.keys(userContext).length === 0) return null;
  
  const contextItems = [];
  
  if (userContext.dietary) {
    contextItems.push({
      icon: '🌱',
      label: 'Diet',
      value: userContext.dietary,
      color: 'bg-green-100 text-green-800'
    });
  }
  
  if (userContext.staying_in) {
    contextItems.push({
      icon: '🏨',
      label: 'Staying in',
      value: userContext.staying_in,
      color: 'bg-blue-100 text-blue-800'
    });
  }
  
  if (userContext.days_left) {
    contextItems.push({
      icon: '⏰',
      label: 'Days left',
      value: `${userContext.days_left} days`,
      color: 'bg-orange-100 text-orange-800'
    });
  }
  
  if (userContext.budget) {
    contextItems.push({
      icon: '💰',
      label: 'Budget',
      value: userContext.budget,
      color: 'bg-purple-100 text-purple-800'
    });
  }
  
  if (contextItems.length === 0) return null;
  
  return (
    <div className={`mb-3 p-3 rounded-lg border ${
      theme === 'dark' 
        ? 'bg-gray-800 border-gray-700' 
        : 'bg-gray-50 border-gray-200'
    }`}>
      <h4 className={`text-sm font-medium mb-2 ${
        theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
      }`}>
        Your Context:
      </h4>
      <div className="flex flex-wrap gap-2">
        {contextItems.map((item, index) => (
          <div
            key={index}
            className={`
              inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs
              ${theme === 'dark' 
                ? 'bg-gray-700 text-gray-300 border border-gray-600' 
                : item.color
              }
            `}
          >
            <span>{item.icon}</span>
            <span className="font-medium">{item.label}:</span>
            <span>{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default UserContextDisplay;
