import React from 'react';

const ItineraryTimeline = ({ itinerary, darkMode = false }) => {
  if (!itinerary) return null;

  return (
    <div className={`border rounded-lg p-4 mb-3 transition-colors duration-200 ${
      darkMode 
        ? 'bg-gray-800 border-gray-600' 
        : 'bg-white border-gray-300 shadow-sm'
    }`}>
      <div className="flex items-start space-x-3">
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 ${
          darkMode ? 'bg-indigo-900 text-indigo-300' : 'bg-indigo-100 text-indigo-700'
        }`}>
          📋
        </div>
        
        <div className="flex-1">
          <h3 className={`font-bold text-lg mb-2 ${
            darkMode ? 'text-white' : 'text-gray-900'
          }`}>
            Your Itinerary Summary
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            {itinerary.total_pois && (
              <div className="text-center">
                <div className={`text-2xl font-bold ${
                  darkMode ? 'text-blue-400' : 'text-blue-600'
                }`}>
                  {itinerary.total_pois}
                </div>
                <div className={`text-xs ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  Places
                </div>
              </div>
            )}
            
            {itinerary.total_distance && (
              <div className="text-center">
                <div className={`text-2xl font-bold ${
                  darkMode ? 'text-green-400' : 'text-green-600'
                }`}>
                  {itinerary.total_distance}
                </div>
                <div className={`text-xs ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  Distance
                </div>
              </div>
            )}
            
            {itinerary.estimated_duration && (
              <div className="text-center">
                <div className={`text-2xl font-bold ${
                  darkMode ? 'text-purple-400' : 'text-purple-600'
                }`}>
                  {itinerary.estimated_duration}
                </div>
                <div className={`text-xs ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  Duration
                </div>
              </div>
            )}
            
            {itinerary.best_start_time && (
              <div className="text-center">
                <div className={`text-2xl font-bold ${
                  darkMode ? 'text-orange-400' : 'text-orange-600'
                }`}>
                  {itinerary.best_start_time}
                </div>
                <div className={`text-xs ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  Start Time
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ItineraryTimeline;
