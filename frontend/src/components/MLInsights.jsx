import React from 'react';

const MLInsights = ({ predictions, darkMode = false }) => {
  if (!predictions) return null;

  const getConfidenceColor = (score) => {
    if (score >= 0.8) return darkMode ? 'text-green-400' : 'text-green-600';
    if (score >= 0.6) return darkMode ? 'text-yellow-400' : 'text-yellow-600';
    return darkMode ? 'text-red-400' : 'text-red-600';
  };

  return (
    <div className={`border rounded-lg p-4 mb-3 transition-colors duration-200 ${
      darkMode 
        ? 'bg-gray-800 border-gray-600' 
        : 'bg-white border-gray-300 shadow-sm'
    }`}>
      <div className="flex items-start space-x-3">
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 ${
          darkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-700'
        }`}>
          🤖
        </div>
        
        <div className="flex-1">
          <h3 className={`font-bold text-lg mb-2 ${
            darkMode ? 'text-white' : 'text-gray-900'
          }`}>
            AI Insights & Predictions
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
            {predictions.confidence_score && (
              <div>
                <div className={`text-xs font-semibold mb-1 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  🎯 Confidence Score:
                </div>
                <div className={`text-lg font-bold ${getConfidenceColor(predictions.confidence_score)}`}>
                  {Math.round(predictions.confidence_score * 100)}%
                </div>
              </div>
            )}
            
            {predictions.weather_impact && (
              <div>
                <div className={`text-xs font-semibold mb-1 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  🌤️ Weather Impact:
                </div>
                <div className={`text-sm ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  {predictions.weather_impact.replace(/_/g, ' ').toUpperCase()}
                </div>
              </div>
            )}
          </div>
          
          {predictions.ml_system_enabled && (
            <div className={`text-xs text-center mt-3 pt-2 border-t ${
              darkMode 
                ? 'border-gray-700 text-gray-500' 
                : 'border-gray-200 text-gray-500'
            }`}>
              ✨ Powered by Machine Learning
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MLInsights;
