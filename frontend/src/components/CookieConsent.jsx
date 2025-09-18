import React, { useState, useEffect } from 'react';
import { useTheme } from '../contexts/ThemeContext';

const CookieConsent = () => {
  const { darkMode } = useTheme();
  const [showBanner, setShowBanner] = useState(false);
  const [preferences, setPreferences] = useState({
    necessary: true, // Always true, can't be disabled
    analytics: false,
    personalization: false
  });
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    // Check if user has already made a choice
    const consent = localStorage.getItem('ai-istanbul-cookie-consent');
    if (!consent) {
      setShowBanner(true);
    }
  }, []);

  const handleAcceptAll = () => {
    const consentData = {
      necessary: true,
      analytics: true,
      personalization: true,
      timestamp: Date.now(),
      version: '1.0'
    };
    
    localStorage.setItem('ai-istanbul-cookie-consent', JSON.stringify(consentData));
    setShowBanner(false);
    
    // Initialize analytics if accepted
    if (window.gtag) {
      window.gtag('consent', 'update', {
        'analytics_storage': 'granted',
        'ad_storage': 'denied',
        'personalization_storage': 'granted'
      });
    }
  };

  const handleRejectAll = () => {
    const consentData = {
      necessary: true,
      analytics: false,
      personalization: false,
      timestamp: Date.now(),
      version: '1.0'
    };
    
    localStorage.setItem('ai-istanbul-cookie-consent', JSON.stringify(consentData));
    setShowBanner(false);
    
    // Disable analytics
    if (window.gtag) {
      window.gtag('consent', 'update', {
        'analytics_storage': 'denied',
        'ad_storage': 'denied',
        'personalization_storage': 'denied'
      });
    }
  };

  const handleSavePreferences = () => {
    const consentData = {
      ...preferences,
      timestamp: Date.now(),
      version: '1.0'
    };
    
    localStorage.setItem('ai-istanbul-cookie-consent', JSON.stringify(consentData));
    setShowBanner(false);
    
    // Update analytics consent
    if (window.gtag) {
      window.gtag('consent', 'update', {
        'analytics_storage': preferences.analytics ? 'granted' : 'denied',
        'ad_storage': 'denied',
        'personalization_storage': preferences.personalization ? 'granted' : 'denied'
      });
    }
  };

  if (!showBanner) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4">
      <div className={`max-w-4xl mx-auto rounded-lg shadow-lg border transition-colors duration-300 ${
        darkMode 
          ? 'bg-gray-800 border-gray-700 text-white' 
          : 'bg-white border-gray-200 text-gray-800'
      }`}>
        <div className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center">
              <span className="text-2xl mr-3">🍪</span>
              <h3 className="text-lg font-semibold">We Value Your Privacy</h3>
            </div>
          </div>
          
          <p className={`text-sm mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            AI Istanbul uses cookies and similar technologies to enhance your experience, 
            provide personalized recommendations, and analyze our traffic. We comply with GDPR 
            and respect your privacy choices.
          </p>

          {!showDetails ? (
            <div className="flex flex-wrap gap-3">
              <button
                onClick={handleAcceptAll}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200"
              >
                Accept All
              </button>
              <button
                onClick={handleRejectAll}
                className={`px-4 py-2 border rounded-lg transition-colors duration-200 ${
                  darkMode 
                    ? 'border-gray-600 hover:bg-gray-700' 
                    : 'border-gray-300 hover:bg-gray-50'
                }`}
              >
                Reject All
              </button>
              <button
                onClick={() => setShowDetails(true)}
                className={`px-4 py-2 border rounded-lg transition-colors duration-200 ${
                  darkMode 
                    ? 'border-gray-600 hover:bg-gray-700' 
                    : 'border-gray-300 hover:bg-gray-50'
                }`}
              >
                Customize
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Necessary Cookies</h4>
                    <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      Essential for the website to function properly
                    </p>
                  </div>
                  <input 
                    type="checkbox" 
                    checked={true} 
                    disabled 
                    className="rounded"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Analytics Cookies</h4>
                    <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      Help us understand how visitors interact with our website
                    </p>
                  </div>
                  <input 
                    type="checkbox" 
                    checked={preferences.analytics}
                    onChange={(e) => setPreferences({...preferences, analytics: e.target.checked})}
                    className="rounded"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Personalization Cookies</h4>
                    <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      Remember your preferences for a better experience
                    </p>
                  </div>
                  <input 
                    type="checkbox" 
                    checked={preferences.personalization}
                    onChange={(e) => setPreferences({...preferences, personalization: e.target.checked})}
                    className="rounded"
                  />
                </div>
              </div>
              
              <div className="flex flex-wrap gap-3 pt-4 border-t border-gray-300 dark:border-gray-600">
                <button
                  onClick={handleSavePreferences}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200"
                >
                  Save Preferences
                </button>
                <button
                  onClick={() => setShowDetails(false)}
                  className={`px-4 py-2 border rounded-lg transition-colors duration-200 ${
                    darkMode 
                      ? 'border-gray-600 hover:bg-gray-700' 
                      : 'border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  Back
                </button>
              </div>
            </div>
          )}
          
          <p className={`text-xs mt-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            For more information, read our{' '}
            <a href="/privacy" className="underline hover:no-underline">
              Privacy Policy
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default CookieConsent;
