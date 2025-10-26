/**
 * RouteInstructions Component
 * ===========================
 * Step-by-step navigation instructions for Istanbul transportation
 * 
 * Features:
 * - Clear step-by-step directions
 * - Multi-modal transit instructions (metro, tram, bus, ferry, walking)
 * - Transfer points highlighted
 * - Accessibility information
 * - Estimated times and distances
 * - Alternative routes display
 * - Print-friendly layout
 * - Mobile-optimized UI
 * - Share functionality
 */

import React, { useState } from 'react';

const RouteInstructions = ({
  route,
  origin,
  destination,
  showAlternatives = true,
  onSelectAlternative = null,
  className = ''
}) => {
  const [expandedSteps, setExpandedSteps] = useState([]);
  const [showShareMenu, setShowShareMenu] = useState(false);

  if (!route || !route.hasRoute) {
    return (
      <div className={`route-instructions no-route ${className}`}>
        <div className="p-6 text-center text-gray-500">
          <div className="text-4xl mb-3">🗺️</div>
          <p>No route available</p>
          <p className="text-sm mt-2">Enter origin and destination to get directions</p>
        </div>
      </div>
    );
  }

  const { steps = [], metadata = {}, summary = '' } = route;
  const { 
    totalDistance, 
    totalDuration, 
    modes = [], 
    transfers = 0,
    accessibility = {},
    alternatives = []
  } = metadata;

  // Toggle step expansion
  const toggleStep = (index) => {
    setExpandedSteps(prev => 
      prev.includes(index) 
        ? prev.filter(i => i !== index)
        : [...prev, index]
    );
  };

  // Get icon for transport mode
  const getModeIcon = (mode) => {
    const icons = {
      metro: '🚇',
      tram: '🚊',
      bus: '🚌',
      ferry: '⛴️',
      walking: '🚶',
      transit: '🚉',
      driving: '🚗',
      transfer: '🔄'
    };
    return icons[mode?.toLowerCase()] || '📍';
  };

  // Get color for transport mode
  const getModeColor = (mode) => {
    const colors = {
      metro: 'bg-pink-100 text-pink-700 border-pink-300',
      tram: 'bg-blue-100 text-blue-700 border-blue-300',
      bus: 'bg-green-100 text-green-700 border-green-300',
      ferry: 'bg-cyan-100 text-cyan-700 border-cyan-300',
      walking: 'bg-orange-100 text-orange-700 border-orange-300',
      transit: 'bg-purple-100 text-purple-700 border-purple-300',
      transfer: 'bg-yellow-100 text-yellow-700 border-yellow-300'
    };
    return colors[mode?.toLowerCase()] || 'bg-gray-100 text-gray-700 border-gray-300';
  };

  // Format distance
  const formatDistance = (meters) => {
    if (!meters) return '';
    if (meters < 1000) return `${Math.round(meters)}m`;
    return `${(meters / 1000).toFixed(1)}km`;
  };

  // Format duration
  const formatDuration = (minutes) => {
    if (!minutes) return '';
    if (minutes < 60) return `${minutes} min`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hours}h ${mins}min` : `${hours}h`;
  };

  // Share route
  const handleShare = async (method) => {
    const routeText = `Route from ${origin?.name || 'Origin'} to ${destination?.name || 'Destination'}\n\n${summary}\n\nTotal: ${formatDistance(totalDistance)}, ${formatDuration(totalDuration)}`;
    
    if (method === 'copy') {
      await navigator.clipboard.writeText(routeText);
      alert('Route copied to clipboard!');
    } else if (method === 'whatsapp') {
      window.open(`https://wa.me/?text=${encodeURIComponent(routeText)}`, '_blank');
    } else if (method === 'twitter') {
      window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(routeText)}`, '_blank');
    }
    setShowShareMenu(false);
  };

  // Print route
  const handlePrint = () => {
    window.print();
  };

  return (
    <div className={`route-instructions ${className}`}>
      {/* Route Summary Header */}
      <div className="route-summary bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4 mb-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-2xl">🗺️</span>
              <h3 className="text-lg font-bold text-gray-800">Your Route</h3>
            </div>
            
            <div className="text-sm text-gray-700 space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-medium">From:</span>
                <span>{origin?.name || 'Your location'}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-medium">To:</span>
                <span>{destination?.name || 'Destination'}</span>
              </div>
            </div>
          </div>
          
          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => setShowShareMenu(!showShareMenu)}
              className="p-2 hover:bg-blue-100 rounded-lg transition"
              title="Share route"
            >
              📤
            </button>
            <button
              onClick={handlePrint}
              className="p-2 hover:bg-blue-100 rounded-lg transition"
              title="Print route"
            >
              🖨️
            </button>
          </div>
        </div>

        {/* Share menu */}
        {showShareMenu && (
          <div className="mt-3 p-3 bg-white rounded-lg shadow-lg border">
            <p className="text-sm font-medium mb-2">Share via:</p>
            <div className="flex gap-2">
              <button
                onClick={() => handleShare('copy')}
                className="flex-1 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm"
              >
                📋 Copy
              </button>
              <button
                onClick={() => handleShare('whatsapp')}
                className="flex-1 px-3 py-2 bg-green-100 hover:bg-green-200 rounded text-sm"
              >
                💬 WhatsApp
              </button>
              <button
                onClick={() => handleShare('twitter')}
                className="flex-1 px-3 py-2 bg-blue-100 hover:bg-blue-200 rounded text-sm"
              >
                🐦 Twitter
              </button>
            </div>
          </div>
        )}

        {/* Route metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          {totalDuration && (
            <div className="bg-white rounded-lg p-3 border border-blue-200">
              <div className="text-xs text-gray-600">Duration</div>
              <div className="text-lg font-bold text-blue-700">{formatDuration(totalDuration)}</div>
            </div>
          )}
          
          {totalDistance && (
            <div className="bg-white rounded-lg p-3 border border-blue-200">
              <div className="text-xs text-gray-600">Distance</div>
              <div className="text-lg font-bold text-blue-700">{formatDistance(totalDistance)}</div>
            </div>
          )}
          
          {modes.length > 0 && (
            <div className="bg-white rounded-lg p-3 border border-blue-200">
              <div className="text-xs text-gray-600">Transport</div>
              <div className="text-lg font-bold text-blue-700">
                {modes.map(m => getModeIcon(m)).join(' ')}
              </div>
            </div>
          )}
          
          {transfers > 0 && (
            <div className="bg-white rounded-lg p-3 border border-blue-200">
              <div className="text-xs text-gray-600">Transfers</div>
              <div className="text-lg font-bold text-blue-700">{transfers}</div>
            </div>
          )}
        </div>

        {/* Accessibility info */}
        {(accessibility.wheelchairAccessible || accessibility.elevatorAvailable) && (
          <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center gap-2 text-sm text-green-800">
              <span>♿</span>
              <span className="font-medium">Accessibility:</span>
              <span>
                {accessibility.wheelchairAccessible && 'Wheelchair accessible'}
                {accessibility.elevatorAvailable && ' • Elevator available'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Step-by-step instructions */}
      <div className="route-steps space-y-3">
        <h4 className="text-md font-bold text-gray-800 mb-3">Step-by-Step Directions</h4>
        
        {steps.length === 0 ? (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <p className="text-gray-600">{summary || 'Route details will appear here'}</p>
          </div>
        ) : (
          steps.map((step, index) => {
            const isExpanded = expandedSteps.includes(index);
            const mode = step.mode || 'transit';
            const modeColor = getModeColor(mode);
            const isTransfer = step.type === 'transfer' || mode === 'transfer';
            
            return (
              <div
                key={index}
                className={`route-step border rounded-lg overflow-hidden transition ${
                  isTransfer ? 'border-yellow-300' : 'border-gray-200'
                }`}
              >
                {/* Step header */}
                <div
                  className={`step-header p-4 cursor-pointer hover:bg-gray-50 ${
                    isTransfer ? 'bg-yellow-50' : 'bg-white'
                  }`}
                  onClick={() => toggleStep(index)}
                >
                  <div className="flex items-start gap-3">
                    {/* Step number */}
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${modeColor} border-2`}>
                      {index + 1}
                    </div>
                    
                    {/* Step content */}
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xl">{getModeIcon(mode)}</span>
                        <span className={`text-xs font-medium px-2 py-1 rounded ${modeColor}`}>
                          {mode.toUpperCase()}
                        </span>
                        {step.line && (
                          <span className="text-xs font-medium px-2 py-1 rounded bg-gray-100 text-gray-700">
                            {step.line}
                          </span>
                        )}
                      </div>
                      
                      <p className="text-gray-800 font-medium">{step.instruction}</p>
                      
                      <div className="flex gap-4 mt-2 text-xs text-gray-600">
                        {step.distance && <span>📏 {formatDistance(step.distance)}</span>}
                        {step.duration && <span>⏱️ {formatDuration(step.duration)}</span>}
                        {step.stops && <span>🚏 {step.stops} stops</span>}
                      </div>
                    </div>
                    
                    {/* Expand icon */}
                    <div className="text-gray-400">
                      {isExpanded ? '▼' : '▶'}
                    </div>
                  </div>
                </div>
                
                {/* Expanded details */}
                {isExpanded && (
                  <div className="step-details bg-gray-50 p-4 border-t">
                    {step.start && (
                      <div className="text-sm mb-2">
                        <span className="font-medium">Start:</span> {step.start[0].toFixed(5)}, {step.start[1].toFixed(5)}
                      </div>
                    )}
                    {step.end && (
                      <div className="text-sm">
                        <span className="font-medium">End:</span> {step.end[0].toFixed(5)}, {step.end[1].toFixed(5)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Alternative routes */}
      {showAlternatives && alternatives.length > 0 && (
        <div className="alternative-routes mt-6">
          <h4 className="text-md font-bold text-gray-800 mb-3">Alternative Routes</h4>
          <div className="space-y-2">
            {alternatives.map((alt, index) => (
              <div
                key={index}
                className="alt-route bg-white border border-gray-200 rounded-lg p-4 hover:border-blue-400 cursor-pointer transition"
                onClick={() => onSelectAlternative && onSelectAlternative(alt)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-800">{alt.summary}</p>
                    <div className="flex gap-4 mt-1 text-sm text-gray-600">
                      {alt.duration && <span>⏱️ {formatDuration(alt.duration)}</span>}
                      {alt.distance && <span>📏 {formatDistance(alt.distance)}</span>}
                      {alt.modes && <span>{alt.modes.map(m => getModeIcon(m)).join(' ')}</span>}
                    </div>
                  </div>
                  <div className="text-blue-600 text-xl">→</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Print styles */}
      <style>{`
        @media print {
          .route-instructions {
            background: white !important;
          }
          
          button {
            display: none !important;
          }
          
          .route-step {
            page-break-inside: avoid;
          }
        }
      `}</style>
    </div>
  );
};

export default RouteInstructions;
