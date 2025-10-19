import React from 'react';

const POICard = ({ poi, darkMode = false }) => {
  if (!poi) return null;

  return (
    <div className={`border rounded-lg p-4 mb-3 transition-colors duration-200 ${
      darkMode 
        ? 'bg-gray-800 border-gray-600' 
        : 'bg-white border-gray-300 shadow-sm'
    }`}>
      <div className="flex items-start space-x-3">
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 ${
          poi.type === 'museum' 
            ? (darkMode ? 'bg-purple-900 text-purple-300' : 'bg-purple-100 text-purple-700')
            : (darkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-700')
        }`}>
          {poi.type === 'museum' ? '🏛️' : '📍'}
        </div>
        
        <div className="flex-1">
          <h3 className={`font-bold text-lg mb-1 ${
            darkMode ? 'text-white' : 'text-gray-900'
          }`}>
            {poi.name}
          </h3>
          
          {poi.description && (
            <p className={`text-sm mb-2 ${
              darkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              {poi.description}
            </p>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
            {poi.opening_hours && (
              <div className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                🕒 {poi.opening_hours}
              </div>
            )}
            
            {poi.entrance_fee && (
              <div className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                🎫 {poi.entrance_fee}
              </div>
            )}
            
            {poi.visit_duration && (
              <div className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                ⏱️ {poi.visit_duration}
              </div>
            )}
            
            {poi.best_time_to_visit && (
              <div className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                ⭐ Best time: {poi.best_time_to_visit}
              </div>
            )}
          </div>
          
          {poi.highlights && poi.highlights.length > 0 && (
            <div className="mt-2">
              <div className={`text-xs font-semibold mb-1 ${
                darkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Highlights:
              </div>
              <div className="flex flex-wrap gap-1">
                {poi.highlights.slice(0, 3).map((highlight, idx) => (
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
            </div>
          )}
          
          {poi.local_tips && poi.local_tips.length > 0 && (
            <div className="mt-2">
              <div className={`text-xs font-semibold mb-1 ${
                darkMode ? 'text-yellow-300' : 'text-yellow-700'
              }`}>
                💡 Local Tips:
              </div>
              <ul className={`text-xs space-y-1 ${
                darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>
                {poi.local_tips.slice(0, 2).map((tip, idx) => (
                  <li key={idx}>• {tip}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default POICard;
