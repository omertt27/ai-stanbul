/**
 * Multi-Route Comparison Component
 * 
 * Moovit-style multi-route display with:
 * - Side-by-side route comparison
 * - Comfort scoring visualization
 * - Interactive route selection
 * - Transfer quality indicators
 * - Route preference labels (fastest, fewest-transfers, etc.)
 * - Visual comparison charts
 * 
 * Features:
 * - Responsive design (mobile + desktop)
 * - Color-coded comfort scores
 * - Expandable route details
 * - Route highlighting on map hover
 * - Smart route recommendations
 * 
 * Author: AI Istanbul Team
 * Date: January 2025
 */

import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

// Comfort score color mapping
const getComfortColor = (score) => {
  if (score >= 80) return 'bg-green-500';
  if (score >= 60) return 'bg-yellow-500';
  if (score >= 40) return 'bg-orange-500';
  return 'bg-red-500';
};

const getComfortTextColor = (score) => {
  if (score >= 80) return 'text-green-700';
  if (score >= 60) return 'text-yellow-700';
  if (score >= 40) return 'text-orange-700';
  return 'text-red-700';
};

// Preference icons and labels
const PREFERENCE_CONFIG = {
  fastest: {
    icon: '⚡',
    label: 'Fastest',
    color: 'bg-blue-100 text-blue-800',
    description: 'Minimizes total travel time'
  },
  'fewest-transfers': {
    icon: '🔄',
    label: 'Fewest Transfers',
    color: 'bg-purple-100 text-purple-800',
    description: 'Least number of changes'
  },
  comfortable: {
    icon: '🛋️',
    label: 'Most Comfortable',
    color: 'bg-green-100 text-green-800',
    description: 'Best overall comfort'
  },
  'least-walking': {
    icon: '🚶',
    label: 'Least Walking',
    color: 'bg-orange-100 text-orange-800',
    description: 'Minimizes walking distance'
  },
  balanced: {
    icon: '⚖️',
    label: 'Balanced',
    color: 'bg-indigo-100 text-indigo-800',
    description: 'Best balance of all factors'
  }
};

// Route card component
const RouteCard = ({ 
  route, 
  index, 
  isSelected, 
  onSelect, 
  onHover, 
  darkMode,
  isRecommended 
}) => {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  
  const prefConfig = PREFERENCE_CONFIG[route.preference] || PREFERENCE_CONFIG.balanced;
  const comfortScore = route.comfort_score?.overall_comfort || 0;
  
  return (
    <div
      className={`relative rounded-lg border-2 transition-all duration-200 cursor-pointer ${
        isSelected 
          ? darkMode 
            ? 'border-blue-500 bg-blue-900/20' 
            : 'border-blue-500 bg-blue-50'
          : darkMode
            ? 'border-gray-700 bg-gray-800 hover:border-gray-600'
            : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
      onClick={() => onSelect(route, index)}
      onMouseEnter={() => onHover && onHover(index)}
      onMouseLeave={() => onHover && onHover(null)}
    >
      {/* Recommended badge */}
      {isRecommended && (
        <div className="absolute -top-3 -right-3 bg-gradient-to-r from-yellow-400 to-yellow-500 text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg">
          ⭐ Recommended
        </div>
      )}
      
      {/* Preference label */}
      <div className={`px-4 py-2 rounded-t-lg ${prefConfig.color}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-xl">{prefConfig.icon}</span>
            <span className="font-bold">{prefConfig.label}</span>
          </div>
          <span className="text-sm opacity-75">Option {index + 1}</span>
        </div>
      </div>
      
      {/* Main content */}
      <div className="p-4 space-y-3">
        {/* Key metrics */}
        <div className="grid grid-cols-3 gap-3">
          <div className="text-center">
            <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {route.duration_minutes}'
            </div>
            <div className="text-xs opacity-75">Duration</div>
          </div>
          
          <div className="text-center">
            <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {route.num_transfers}
            </div>
            <div className="text-xs opacity-75">Transfers</div>
          </div>
          
          <div className="text-center">
            <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {Math.round(route.walking_meters)}m
            </div>
            <div className="text-xs opacity-75">Walking</div>
          </div>
        </div>
        
        {/* Comfort score bar */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium">Comfort Score</span>
            <span className={`text-xs font-bold ${getComfortTextColor(comfortScore)}`}>
              {Math.round(comfortScore)}/100
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${getComfortColor(comfortScore)}`}
              style={{ width: `${comfortScore}%` }}
            />
          </div>
        </div>
        
        {/* Highlights */}
        {route.highlights && route.highlights.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {route.highlights.slice(0, 3).map((highlight, idx) => (
              <span
                key={idx}
                className={`px-2 py-1 rounded text-xs ${
                  darkMode 
                    ? 'bg-gray-700 text-gray-300' 
                    : 'bg-gray-100 text-gray-700'
                }`}
              >
                {highlight}
              </span>
            ))}
          </div>
        )}
        
        {/* Expand button for details */}
        <button
          className={`w-full py-2 rounded text-sm font-medium transition-colors ${
            darkMode
              ? 'bg-gray-700 hover:bg-gray-600 text-white'
              : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
          }`}
          onClick={(e) => {
            e.stopPropagation();
            setExpanded(!expanded);
          }}
        >
          {expanded ? '▲ Hide Details' : '▼ View Details'}
        </button>
        
        {/* Expanded details */}
        {expanded && (
          <div className="pt-3 border-t border-gray-300 space-y-2">
            {/* LLM summary */}
            {route.llm_summary && (
              <div className={`p-3 rounded ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <div className="text-xs font-semibold mb-1">Summary</div>
                <div className="text-sm">{route.llm_summary}</div>
              </div>
            )}
            
            {/* Comfort breakdown */}
            {route.comfort_score && (
              <div>
                <div className="text-xs font-semibold mb-2">Comfort Breakdown</div>
                <div className="space-y-1">
                  {route.comfort_score.crowding_comfort !== undefined && (
                    <ComfortMetric
                      label="Crowding"
                      score={route.comfort_score.crowding_comfort}
                      darkMode={darkMode}
                    />
                  )}
                  {route.comfort_score.transfer_comfort !== undefined && (
                    <ComfortMetric
                      label="Transfers"
                      score={route.comfort_score.transfer_comfort}
                      darkMode={darkMode}
                    />
                  )}
                  {route.comfort_score.walking_comfort !== undefined && (
                    <ComfortMetric
                      label="Walking"
                      score={route.comfort_score.walking_comfort}
                      darkMode={darkMode}
                    />
                  )}
                  {route.comfort_score.waiting_comfort !== undefined && (
                    <ComfortMetric
                      label="Waiting"
                      score={route.comfort_score.waiting_comfort}
                      darkMode={darkMode}
                    />
                  )}
                </div>
              </div>
            )}
            
            {/* Overall score */}
            {route.overall_score && (
              <div className={`p-2 rounded ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold">Overall Score</span>
                  <span className="text-lg font-bold">{Math.round(route.overall_score)}/100</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Comfort metric component
const ComfortMetric = ({ label, score, darkMode }) => {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="opacity-75">{label}</span>
      <div className="flex items-center space-x-2">
        <div className="w-20 bg-gray-300 rounded-full h-1.5">
          <div
            className={`h-1.5 rounded-full ${getComfortColor(score)}`}
            style={{ width: `${score}%` }}
          />
        </div>
        <span className="font-medium w-8 text-right">{Math.round(score)}</span>
      </div>
    </div>
  );
};

// Main component
const MultiRouteComparison = ({ 
  routes = [], 
  primaryRoute = null,
  routeComparison = {},
  onRouteSelect,
  darkMode = false,
  className = ''
}) => {
  const { t } = useTranslation();
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  
  if (!routes || routes.length === 0) {
    return null;
  }
  
  // Add primary route to beginning if not already in list
  const allRoutes = primaryRoute ? [primaryRoute, ...routes] : routes;
  
  // Determine recommended route (highest overall score)
  const recommendedIndex = allRoutes.reduce((maxIdx, route, idx, arr) => 
    (route.overall_score || 0) > (arr[maxIdx].overall_score || 0) ? idx : maxIdx
  , 0);
  
  const handleSelect = (route, index) => {
    setSelectedIndex(index);
    if (onRouteSelect) {
      onRouteSelect(route, index);
    }
  };
  
  return (
    <div className={`multi-route-comparison ${className}`}>
      {/* Header */}
      <div className={`mb-4 p-4 rounded-lg ${
        darkMode 
          ? 'bg-gradient-to-r from-indigo-900 to-blue-900' 
          : 'bg-gradient-to-r from-indigo-600 to-blue-600'
      }`}>
        <div className="flex items-center justify-between text-white">
          <div>
            <h3 className="text-lg font-bold">Route Options</h3>
            <p className="text-sm opacity-90">
              {allRoutes.length} {allRoutes.length === 1 ? 'route' : 'routes'} available
            </p>
          </div>
          <div className="text-right">
            <div className="text-2xl">🗺️</div>
          </div>
        </div>
      </div>
      
      {/* Route cards grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {allRoutes.map((route, index) => (
          <RouteCard
            key={index}
            route={route}
            index={index}
            isSelected={selectedIndex === index}
            isRecommended={index === recommendedIndex}
            onSelect={handleSelect}
            onHover={setHoveredIndex}
            darkMode={darkMode}
          />
        ))}
      </div>
      
      {/* Comparison summary */}
      {routeComparison && Object.keys(routeComparison).length > 0 && (
        <div className={`mt-4 p-4 rounded-lg ${
          darkMode ? 'bg-gray-800' : 'bg-gray-50'
        }`}>
          <h4 className="font-bold mb-3">Quick Comparison</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {routeComparison.fastest_route !== undefined && (
              <div>
                <div className="opacity-75 mb-1">⚡ Fastest</div>
                <div className="font-bold">Option {routeComparison.fastest_route + 1}</div>
              </div>
            )}
            {routeComparison.fewest_transfers !== undefined && (
              <div>
                <div className="opacity-75 mb-1">🔄 Fewest Transfers</div>
                <div className="font-bold">Option {routeComparison.fewest_transfers + 1}</div>
              </div>
            )}
            {routeComparison.most_comfortable !== undefined && (
              <div>
                <div className="opacity-75 mb-1">🛋️ Most Comfortable</div>
                <div className="font-bold">Option {routeComparison.most_comfortable + 1}</div>
              </div>
            )}
            {routeComparison.least_walking !== undefined && (
              <div>
                <div className="opacity-75 mb-1">🚶 Least Walking</div>
                <div className="font-bold">Option {routeComparison.least_walking + 1}</div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Help text */}
      <div className={`mt-4 p-3 rounded-lg text-sm ${
        darkMode 
          ? 'bg-blue-900/20 text-blue-200' 
          : 'bg-blue-50 text-blue-800'
      }`}>
        <span className="font-semibold">💡 Tip:</span> Click on any route to see it highlighted on the map. 
        Hover over routes to compare them quickly.
      </div>
    </div>
  );
};

export default MultiRouteComparison;
