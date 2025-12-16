/**
 * Transportation Route Card
 * 
 * Displays transit routing information in a visual, easy-to-follow format
 * Integrates with the hybrid intent classifier and Dijkstra routing system
 * 
 * Features:
 * - Step-by-step directions with icons
 * - Transfer points highlighted
 * - Time estimates and confidence indicators
 * - Transit line colors
 * - Responsive mobile-friendly design
 * 
 * Author: AI Istanbul Team
 * Date: December 17, 2025
 */

import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

// Transit line colors (Istanbul metro system)
const LINE_COLORS = {
  'M1': '#E63946',      // Red (M1A, M1B)
  'M2': '#00B050',      // Green
  'M3': '#0070C0',      // Blue
  'M4': '#FF6B35',      // Orange
  'M5': '#9C27B0',      // Purple
  'M6': '#795548',      // Brown
  'M7': '#F06292',      // Pink
  'M11': '#607D8B',     // Grey
  'M9': '#FFC107',      // Yellow
  'T1': '#D32F2F',      // Dark Red
  'T4': '#F44336',      // Light Red
  'T5': '#FF5722',      // Deep Orange
  'MARMARAY': '#1976D2', // Deep Blue
  'FERRY': '#00838F',   // Cyan
  'FUNICULAR': '#388E3C', // Dark Green
  'DEFAULT': '#757575'  // Grey
};

// Get line color or default
const getLineColor = (line) => {
  if (!line) return LINE_COLORS.DEFAULT;
  const lineKey = line.toUpperCase().split(' ')[0]; // Extract M1, M2, etc.
  return LINE_COLORS[lineKey] || LINE_COLORS.DEFAULT;
};

// Transit icons
const TransitIcon = ({ type, line }) => {
  const color = getLineColor(line);
  
  if (type === 'transit') {
    return (
      <div className="flex items-center justify-center w-10 h-10 rounded-full" 
           style={{ backgroundColor: color }}>
        <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2c-4 0-8 .5-8 4v9.5C4 17.43 5.57 19 7.5 19L6 20.5v.5h2l2-2h4l2 2h2v-.5L16.5 19c1.93 0 3.5-1.57 3.5-3.5V6c0-3.5-4-4-8-4zM7.5 17c-.83 0-1.5-.67-1.5-1.5S6.67 14 7.5 14s1.5.67 1.5 1.5S8.33 17 7.5 17zm3.5-7H6V6h5v4zm5.5 7c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5zm1.5-7h-5V6h5v4z"/>
        </svg>
      </div>
    );
  }
  
  if (type === 'transfer') {
    return (
      <div className="flex items-center justify-center w-10 h-10 rounded-full bg-yellow-500">
        <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
          <path d="M6.99 11L3 15l3.99 4v-3H14v-2H6.99v-3zM21 9l-3.99-4v3H10v2h7.01v3L21 9z"/>
        </svg>
      </div>
    );
  }
  
  if (type === 'walk') {
    return (
      <div className="flex items-center justify-center w-10 h-10 rounded-full bg-gray-500">
        <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
          <path d="M13.5 5.5c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zM9.8 8.9L7 23h2.1l1.8-8 2.1 2v6h2v-7.5l-2.1-2 .6-3C14.8 12 16.8 13 19 13v-2c-1.9 0-3.5-1-4.3-2.4l-1-1.6c-.4-.6-1-1-1.7-1-.3 0-.5.1-.8.1L6 8.3V13h2V9.6l1.8-.7"/>
        </svg>
      </div>
    );
  }
  
  return (
    <div className="flex items-center justify-center w-10 h-10 rounded-full bg-blue-500">
      <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
      </svg>
    </div>
  );
};

// Confidence badge
const ConfidenceBadge = ({ level, darkMode }) => {
  const colors = {
    high: darkMode ? 'bg-green-900 text-green-200' : 'bg-green-100 text-green-800',
    medium: darkMode ? 'bg-yellow-900 text-yellow-200' : 'bg-yellow-100 text-yellow-800',
    low: darkMode ? 'bg-orange-900 text-orange-200' : 'bg-orange-100 text-orange-800'
  };
  
  const icons = {
    high: '✓',
    medium: '~',
    low: '⚠'
  };
  
  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[level]}`}>
      {icons[level]} {level.charAt(0).toUpperCase() + level.slice(1)} Confidence
    </span>
  );
};

const TransportationRouteCard = ({ routeData, darkMode = false }) => {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(true);
  
  if (!routeData || !routeData.steps || routeData.steps.length === 0) {
    return null;
  }
  
  const { 
    origin, 
    destination, 
    total_time, 
    total_distance, 
    transfers, 
    lines_used = [], 
    steps = [],
    time_confidence = 'medium'
  } = routeData;
  
  return (
    <div className={`mt-4 rounded-xl overflow-hidden shadow-lg transition-all duration-300 ${
      darkMode ? 'bg-gray-800' : 'bg-white'
    }`}>
      {/* Header */}
      <div className={`p-4 ${
        darkMode ? 'bg-gradient-to-r from-indigo-900 to-blue-900' : 'bg-gradient-to-r from-indigo-600 to-blue-600'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-2 text-white">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
              </svg>
              <h3 className="font-bold text-lg">{origin} → {destination}</h3>
            </div>
            
            {/* Summary Stats */}
            <div className="flex flex-wrap items-center gap-3 mt-2 text-sm text-white/90">
              <div className="flex items-center space-x-1">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z"/>
                </svg>
                <span className="font-semibold">{total_time} min</span>
              </div>
              
              {total_distance && (
                <div className="flex items-center space-x-1">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                  </svg>
                  <span>{total_distance.toFixed(1)} km</span>
                </div>
              )}
              
              <div className="flex items-center space-x-1">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6.99 11L3 15l3.99 4v-3H14v-2H6.99v-3zM21 9l-3.99-4v3H10v2h7.01v3L21 9z"/>
                </svg>
                <span>{transfers} {transfers === 1 ? 'transfer' : 'transfers'}</span>
              </div>
              
              <ConfidenceBadge level={time_confidence} darkMode={false} />
            </div>
            
            {/* Lines Used */}
            {lines_used.length > 0 && (
              <div className="flex items-center space-x-2 mt-2">
                <span className="text-xs text-white/80">Lines:</span>
                <div className="flex flex-wrap gap-1">
                  {lines_used.map((line, idx) => (
                    <span 
                      key={idx}
                      className="px-2 py-0.5 rounded text-xs font-bold text-white"
                      style={{ backgroundColor: getLineColor(line) }}
                    >
                      {line}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Expand/Collapse Button */}
          <button
            onClick={() => setExpanded(!expanded)}
            className="ml-2 p-2 rounded-full hover:bg-white/10 transition-colors"
            aria-label={expanded ? "Collapse route" : "Expand route"}
          >
            <svg 
              className={`w-5 h-5 text-white transform transition-transform ${expanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>
      
      {/* Route Steps */}
      {expanded && (
        <div className="p-4">
          <h4 className={`text-sm font-semibold mb-3 ${
            darkMode ? 'text-gray-300' : 'text-gray-700'
          }`}>
            📍 Step-by-Step Directions:
          </h4>
          
          <div className="space-y-1">
            {steps.map((step, idx) => (
              <div key={idx} className="relative">
                {/* Connecting Line */}
                {idx < steps.length - 1 && (
                  <div 
                    className={`absolute left-5 top-10 bottom-0 w-0.5 ${
                      darkMode ? 'bg-gray-700' : 'bg-gray-300'
                    }`}
                    style={{ height: 'calc(100% + 0.25rem)' }}
                  />
                )}
                
                {/* Step Content */}
                <div className="flex items-start space-x-3">
                  {/* Icon */}
                  <div className="flex-shrink-0">
                    <TransitIcon type={step.type} line={step.line} />
                  </div>
                  
                  {/* Details */}
                  <div className="flex-1 pt-2">
                    <div className={`font-medium ${
                      darkMode ? 'text-gray-200' : 'text-gray-900'
                    }`}>
                      {step.instruction || `${step.type === 'transit' ? 'Take' : 'Transfer to'} ${step.line}`}
                    </div>
                    
                    {step.from && step.to && (
                      <div className={`text-sm mt-1 ${
                        darkMode ? 'text-gray-400' : 'text-gray-600'
                      }`}>
                        {step.from} → {step.to}
                      </div>
                    )}
                    
                    <div className="flex items-center space-x-3 mt-1">
                      {step.duration && (
                        <span className={`text-xs ${
                          darkMode ? 'text-gray-500' : 'text-gray-500'
                        }`}>
                          ⏱️ ~{step.duration} min
                        </span>
                      )}
                      
                      {step.stops && (
                        <span className={`text-xs ${
                          darkMode ? 'text-gray-500' : 'text-gray-500'
                        }`}>
                          🚏 {step.stops} stops
                        </span>
                      )}
                      
                      {step.line && (
                        <span 
                          className="px-2 py-0.5 rounded text-xs font-bold text-white"
                          style={{ backgroundColor: getLineColor(step.line) }}
                        >
                          {step.line}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Footer Note */}
          <div className={`mt-4 pt-3 border-t text-xs text-center ${
            darkMode ? 'border-gray-700 text-gray-500' : 'border-gray-200 text-gray-600'
          }`}>
            💡 Times are estimates and may vary based on traffic and wait times
          </div>
        </div>
      )}
    </div>
  );
};

export default TransportationRouteCard;
