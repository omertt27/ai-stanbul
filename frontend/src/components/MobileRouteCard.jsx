import React, { useState, useEffect } from 'react';
import RouteBottomSheet from './RouteBottomSheet';
import SwipeableStepNavigation from './SwipeableStepNavigation';

/**
 * MobileRouteCard - Mobile-optimized route display with bottom sheet
 * Full-screen map with draggable bottom sheet for route details
 */
const MobileRouteCard = ({ 
  routeData,
  MapComponent, // Pass the map component to render
  onClose 
}) => {
  const [showBottomSheet, setShowBottomSheet] = useState(true);
  const [showNavigation, setShowNavigation] = useState(false);

  // Extract route data
  const map_data = routeData?.map_data || routeData?.mapData;
  const route_info = routeData?.route_info || routeData?.routeData || routeData?.route_data;
  const actualRouteInfo = route_info || map_data?.route_data || map_data?.metadata?.route_data;

  const origin = actualRouteInfo?.start_location || actualRouteInfo?.origin || 'Starting point';
  const destination = actualRouteInfo?.end_location || actualRouteInfo?.destination || 'Destination';
  const distance = actualRouteInfo?.total_distance 
    ? (actualRouteInfo.total_distance / 1000).toFixed(1) 
    : '0.0';
  const duration = actualRouteInfo?.total_time 
    ? Math.round(actualRouteInfo.total_time / 60) 
    : 0;
  const transfers = actualRouteInfo?.transfer_count || actualRouteInfo?.transfers || 0;
  const lines = actualRouteInfo?.transit_lines || actualRouteInfo?.lines_used || [];
  const steps = actualRouteInfo?.steps || [];

  // Helper to get step icon
  const getStepIcon = (mode) => {
    const icons = {
      walk: '🚶',
      metro: '🚇',
      bus: '🚌',
      tram: '🚋',
      ferry: '⛴️',
      transfer: '🔄',
      funicular: '🚡',
      default: '➡️'
    };
    return icons[mode?.toLowerCase()] || icons.default;
  };

  return (
    <div className="fixed inset-0 z-40 bg-white md:hidden">
      {/* Full-Screen Map */}
      <div className="absolute inset-0">
        {MapComponent}
      </div>

      {/* Close Button */}
      <button
        onClick={onClose}
        className="absolute top-4 left-4 z-50 bg-white rounded-full shadow-lg p-3 hover:bg-gray-100 transition-colors"
        aria-label="Close mobile view"
      >
        <span className="text-xl">←</span>
      </button>

      {/* Bottom Sheet with Route Details */}
      {!showNavigation && (
        <RouteBottomSheet
          isOpen={showBottomSheet}
          onClose={() => setShowBottomSheet(false)}
          snapPoints={[0.25, 0.5, 0.85]}
          initialSnapPoint={0.5}
        >
          {/* Route Header */}
          <div className="mb-4">
            <h2 className="text-xl font-bold text-gray-900 mb-2">
              {origin} → {destination}
            </h2>
            
            {/* Route Stats */}
            <div className="flex items-center space-x-4 text-sm text-gray-600">
              <div className="flex items-center bg-indigo-50 px-3 py-1.5 rounded-lg">
                <span className="mr-1">⏱️</span>
                <span className="font-semibold">{duration} min</span>
              </div>
              <div className="flex items-center bg-indigo-50 px-3 py-1.5 rounded-lg">
                <span className="mr-1">📏</span>
                <span className="font-semibold">{distance} km</span>
              </div>
              {transfers > 0 && (
                <div className="flex items-center bg-indigo-50 px-3 py-1.5 rounded-lg">
                  <span className="mr-1">🔄</span>
                  <span className="font-semibold">{transfers}</span>
                </div>
              )}
            </div>

            {/* Transit Lines */}
            {lines.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {lines.map((line, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1.5 bg-indigo-600 text-white rounded-lg text-sm font-bold"
                  >
                    {line}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Start Navigation Button */}
          {steps.length > 0 && (
            <button
              onClick={() => setShowNavigation(true)}
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-4 px-6 rounded-xl transition-colors shadow-lg flex items-center justify-center space-x-2 mb-4"
            >
              <span className="text-2xl">🧭</span>
              <span className="text-lg">Start Step-by-Step Navigation</span>
            </button>
          )}

          {/* Compact Steps List */}
          {steps.length > 0 && (
            <div className="space-y-2">
              <h3 className="font-semibold text-gray-700 mb-3 flex items-center">
                <span className="mr-2">📋</span>
                All Steps
              </h3>
              {steps.map((step, idx) => (
                <div 
                  key={idx} 
                  className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <div className="flex-shrink-0 w-7 h-7 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center font-bold text-sm">
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-lg">{getStepIcon(step.mode)}</span>
                      <span className="font-medium text-gray-800 text-sm">
                        {step.instruction || step.description}
                      </span>
                    </div>
                    {step.duration && (
                      <span className="text-xs text-gray-500">
                        ~{Math.round(step.duration / 60)} min
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </RouteBottomSheet>
      )}

      {/* Navigation Mode - Full Screen */}
      {showNavigation && steps.length > 0 && (
        <div className="absolute inset-0 bg-white z-50 flex flex-col">
          {/* Header */}
          <div className="bg-indigo-600 text-white p-4 flex items-center justify-between">
            <h2 className="text-lg font-bold">Step-by-Step Navigation</h2>
            <button
              onClick={() => setShowNavigation(false)}
              className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-full p-2 transition-colors"
              aria-label="Exit navigation"
            >
              <span className="text-xl">×</span>
            </button>
          </div>

          {/* Swipeable Navigation */}
          <SwipeableStepNavigation
            steps={steps}
            onStepChange={(stepIndex) => {
              console.log('Current step:', stepIndex);
              // Could update map view based on step
            }}
          />
        </div>
      )}
    </div>
  );
};

export default MobileRouteCard;
