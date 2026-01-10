/**
 * MessageRenderer Component
 * 
 * Central component for rendering all AI assistant messages.
 * Handles: text, RouteCard, RestaurantCard, MapVisualization, TripPlan, etc.
 * 
 * Benefits:
 * - DRY principle (single source of truth for message rendering)
 * - Consistent styling across desktop and mobile
 * - Easier to maintain and test
 * - Reusable in different contexts
 */

import React from 'react';
import RouteCard from './RouteCard';
import RestaurantCard from './RestaurantCard';
import MapVisualization from './MapVisualization';
import MultiRouteComparison from './MultiRouteComparison';
import TransportationRouteCard from './TransportationRouteCard';
import QuickActions from './QuickActions';

/**
 * Render clickable links in message text
 */
const renderMessageContent = (content, darkMode = false) => {
  // Handle undefined or null content
  if (!content) {
    console.warn('⚠️ renderMessageContent received undefined/null content');
    return 'Message content not available';
  }
  
  // Convert to string if not already
  const contentStr = String(content);
  
  // Convert Markdown-style links [text](url) to clickable HTML links
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  
  const parts = [];
  let lastIndex = 0;
  let match;
  
  while ((match = linkRegex.exec(contentStr)) !== null) {
    const linkText = match[1];
    const linkUrl = match[2];
    
    // Add text before the link
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {contentStr.substring(lastIndex, match.index)}
        </span>
      );
    }
    
    // Add the clickable link
    parts.push(
      <a
        key={`link-${match.index}`}
        href={linkUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={`underline transition-colors duration-200 hover:opacity-80 cursor-pointer ${
          darkMode 
            ? 'text-blue-400 hover:text-blue-300' 
            : 'text-blue-600 hover:text-blue-700'
        }`}
      >
        {linkText}
      </a>
    );
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < contentStr.length) {
    parts.push(
      <span key={`text-${lastIndex}`}>
        {contentStr.substring(lastIndex)}
      </span>
    );
  }
  
  return parts.length > 0 ? parts : contentStr;
};

/**
 * MessageRenderer Component
 * 
 * @param {Object} props
 * @param {Object} props.message - The message object
 * @param {boolean} props.darkMode - Dark mode flag
 * @param {Function} props.onCopyRoute - Callback for copying routes
 * @param {Function} props.onShareRoute - Callback for sharing routes
 * @param {Function} props.onStartNavigation - Callback for starting navigation
 * @param {Function} props.onQuickAction - Callback for quick action clicks
 * @param {boolean} props.showQuickActions - Whether to show quick actions (default: true)
 */
const MessageRenderer = ({ 
  message, 
  darkMode, 
  onCopyRoute, 
  onShareRoute, 
  onStartNavigation,
  onQuickAction,
  showQuickActions = true
}) => {
  const msg = message;
  
  /**
   * PRIORITY RENDERING LOGIC (Phase 1.1)
   * Prevents duplicate route/map visualizations
   * 
   * Priority 1: RouteCard (if route_info + map_data exist)
   * Priority 2: TransportationRouteCard (fallback)
   * Priority 3: MultiRouteComparison (only if no single route)
   * Priority 4: MapVisualization (only if no route cards)
   */
  
  // Check what visualizations are available
  const hasRouteCard = !!(msg.mapData?.route_info && msg.mapData);
  const hasTransportationCard = !!(msg.routeData && !hasRouteCard);
  const hasMultiRoute = !!(msg.mapData?.routes && msg.mapData.routes.length > 1 && !hasRouteCard && !hasTransportationCard);
  const hasMapVisualization = !!(msg.mapData && !hasRouteCard && !hasTransportationCard && !hasMultiRoute);
  
  return (
    <div className="space-y-4">
      {/* Text Content */}
      {(msg.text || msg.content) && (
        <div className={`text-sm md:text-base whitespace-pre-wrap leading-[1.6] transition-colors duration-200 select-text ${
          darkMode ? 'text-gray-100' : 'text-gray-800'
        }`}>
          {renderMessageContent(msg.text || msg.content, darkMode)}
        </div>
      )}
      
      {/* Priority 1: RouteCard (if available) */}
      {hasRouteCard && (
        <div className="mt-4">
          <RouteCard
            routeInfo={msg.mapData.route_info}
            mapData={msg.mapData}
            darkMode={darkMode}
            onCopy={onCopyRoute}
            onShare={onShareRoute}
            onStartNavigation={onStartNavigation}
          />
        </div>
      )}
      
      {/* Priority 2: TransportationRouteCard (fallback) */}
      {hasTransportationCard && (
        <div className="mt-4">
          <TransportationRouteCard
            routeData={msg.routeData}
            darkMode={darkMode}
          />
        </div>
      )}
      
      {/* Priority 3: MultiRouteComparison (if multiple routes) */}
      {hasMultiRoute && (
        <div className="mt-4">
          <MultiRouteComparison
            routes={msg.mapData.routes}
            darkMode={darkMode}
          />
        </div>
      )}
      
      {/* Priority 4: MapVisualization (non-route queries) */}
      {hasMapVisualization && (
        <div className="mt-4">
          <MapVisualization
            mapData={msg.mapData}
            darkMode={darkMode}
          />
        </div>
      )}
      
      {/* Restaurant Cards */}
      {msg.restaurants && msg.restaurants.length > 0 && (
        <div className="mt-4 space-y-4">
          <div className={`text-sm font-medium mb-3 ${
            darkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            🍽️ Restaurant Recommendations
          </div>
          {msg.restaurants.map((restaurant, idx) => (
            <RestaurantCard
              key={restaurant.id || idx}
              restaurant={restaurant}
              darkMode={darkMode}
            />
          ))}
        </div>
      )}
      
      {/* Metadata Badges */}
      {(msg.cached || msg.confidence || msg.llmMode || msg.responseTime) && (
        <div className="mt-3 flex flex-wrap gap-2">
          {/* LLM Mode Badge */}
          {msg.llmMode && msg.llmMode !== 'general' && (
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
              msg.llmMode === 'explain' 
                ? darkMode ? 'bg-blue-900/50 text-blue-200' : 'bg-blue-100 text-blue-700'
                : msg.llmMode === 'clarify'
                ? darkMode ? 'bg-yellow-900/50 text-yellow-200' : 'bg-yellow-100 text-yellow-700'
                : darkMode ? 'bg-red-900/50 text-red-200' : 'bg-red-100 text-red-700'
            }`}>
              {msg.llmMode === 'explain' ? '🚇 Route' : msg.llmMode === 'clarify' ? '❓ Clarifying' : '⚠️ Error'}
            </span>
          )}
          
          {/* Cache Badge */}
          {msg.cached && (
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
              darkMode ? 'bg-green-900/50 text-green-200' : 'bg-green-100 text-green-700'
            }`}>
              ⚡ Cached
            </span>
          )}
          
          {/* Confidence Badge */}
          {msg.confidence && (
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
              msg.confidence > 0.8
                ? darkMode ? 'bg-green-900/50 text-green-200' : 'bg-green-100 text-green-700'
                : msg.confidence > 0.6
                ? darkMode ? 'bg-yellow-900/50 text-yellow-200' : 'bg-yellow-100 text-yellow-700'
                : darkMode ? 'bg-red-900/50 text-red-200' : 'bg-red-100 text-red-700'
            }`}>
              {Math.round(msg.confidence * 100)}% confidence
            </span>
          )}
          
          {/* Response Time Badge */}
          {msg.responseTime && (
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
              darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'
            }`}>
              ⏱️ {msg.responseTime}ms
            </span>
          )}
        </div>
      )}
      
      {/* Quick Actions (Phase 4.2) */}
      {showQuickActions && onQuickAction && (
        <QuickActions
          message={msg}
          onActionClick={onQuickAction}
          darkMode={darkMode}
          maxActions={4}
        />
      )}
    </div>
  );
};

export default MessageRenderer;
