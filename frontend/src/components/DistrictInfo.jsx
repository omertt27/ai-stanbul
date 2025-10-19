import React from 'react';

const DistrictInfo = ({ district, darkMode = false }) => {
  if (!district) return null;

  return (
    <div className={`border rounded-lg p-4 mb-3 transition-colors duration-200 ${
      darkMode 
        ? 'bg-gray-800 border-gray-600' 
        : 'bg-white border-gray-300 shadow-sm'
    }`}>
      <div className="flex items-start space-x-3">
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 ${
          darkMode ? 'bg-green-900 text-green-300' : 'bg-green-100 text-green-700'
        }`}>
          🏘️
        </div>
        
        <div className="flex-1">
          <h3 className={`font-bold text-lg mb-1 ${
            darkMode ? 'text-white' : 'text-gray-900'
          }`}>
            {district.name} District
          </h3>
          
          {district.description && (
            <p className={`text-sm mb-3 ${
              darkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>
              {district.description}
            </p>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
            {district.best_time && (
              <div>
                <div className={`font-semibold text-xs mb-1 ${
                  darkMode ? 'text-blue-300' : 'text-blue-700'
                }`}>
                  ⏰ Best Time to Visit:
                </div>
                <div className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                  {district.best_time}
                </div>
              </div>
            )}
            
            {district.transport && (
              <div>
                <div className={`font-semibold text-xs mb-1 ${
                  darkMode ? 'text-blue-300' : 'text-blue-700'
                }`}>
                  🚇 Transportation:
                </div>
                <div className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                  {district.transport}
                </div>
              </div>
            )}
          </div>
          
          {district.local_tips && district.local_tips.length > 0 && (
            <div className="mt-3">
              <div className={`font-semibold text-xs mb-2 ${
                darkMode ? 'text-yellow-300' : 'text-yellow-700'
              }`}>
                💡 Local Tips:
              </div>
              <ul className={`text-sm space-y-1 ${
                darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>
                {district.local_tips.slice(0, 3).map((tip, idx) => (
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

export default DistrictInfo;
