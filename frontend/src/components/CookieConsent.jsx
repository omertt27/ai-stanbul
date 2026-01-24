import React, { useState, useEffect, createContext, useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../contexts/ThemeContext';
import { Link } from 'react-router-dom';

// Current consent version - increment when privacy policy changes
const CONSENT_VERSION = '1.1';
const CONSENT_STORAGE_KEY = 'ai-istanbul-cookie-consent';

// Consent Context for app-wide access
const ConsentContext = createContext(null);

export const useConsent = () => {
  const context = useContext(ConsentContext);
  if (!context) {
    // Return default values if used outside provider
    return {
      consent: null,
      hasConsent: false,
      analyticsEnabled: false,
      personalizationEnabled: false,
      openConsentBanner: () => {},
    };
  }
  return context;
};

// Utility to get consent status (can be used outside React)
export const getConsentStatus = () => {
  try {
    const stored = localStorage.getItem(CONSENT_STORAGE_KEY);
    if (stored) {
      const consent = JSON.parse(stored);
      // Check if consent version matches
      if (consent.version === CONSENT_VERSION) {
        return consent;
      }
    }
  } catch (e) {
    console.warn('Failed to read consent status:', e);
  }
  return null;
};

// Check if analytics is allowed
export const isAnalyticsAllowed = () => {
  const consent = getConsentStatus();
  return consent?.analytics === true;
};

// Check if personalization is allowed
export const isPersonalizationAllowed = () => {
  const consent = getConsentStatus();
  return consent?.personalization === true;
};

const CookieConsent = () => {
  const { t } = useTranslation();
  const { darkMode } = useTheme();
  const [showBanner, setShowBanner] = useState(false);
  const [preferences, setPreferences] = useState({
    necessary: true, // Always true, can't be disabled
    analytics: false,
    personalization: false
  });
  const [showDetails, setShowDetails] = useState(false);
  const [consent, setConsent] = useState(null);

  useEffect(() => {
    // Check if user has already made a choice with current version
    const stored = getConsentStatus();
    if (!stored) {
      // No valid consent - show banner
      setShowBanner(true);
    } else {
      setConsent(stored);
      setPreferences({
        necessary: true,
        analytics: stored.analytics,
        personalization: stored.personalization
      });
      // Apply consent on load
      applyConsentSettings(stored);
    }
    
    // Listen for manual consent banner open requests
    const handleOpenConsent = () => setShowBanner(true);
    window.addEventListener('openConsentBanner', handleOpenConsent);
    return () => window.removeEventListener('openConsentBanner', handleOpenConsent);
  }, []);

  const applyConsentSettings = (consentData) => {
    // Update Google Analytics consent
    if (window.gtag) {
      window.gtag('consent', 'update', {
        'analytics_storage': consentData.analytics ? 'granted' : 'denied',
        'ad_storage': 'denied', // We don't use ads
        'personalization_storage': consentData.personalization ? 'granted' : 'denied'
      });
    }
    
    // Dispatch event for other components to react
    window.dispatchEvent(new CustomEvent('consentUpdated', { detail: consentData }));
  };

  const saveConsent = (consentData) => {
    const fullConsentData = {
      ...consentData,
      timestamp: Date.now(),
      version: CONSENT_VERSION
    };
    
    localStorage.setItem(CONSENT_STORAGE_KEY, JSON.stringify(fullConsentData));
    setConsent(fullConsentData);
    setShowBanner(false);
    applyConsentSettings(fullConsentData);
  };

  const handleAcceptAll = () => {
    saveConsent({
      necessary: true,
      analytics: true,
      personalization: true
    });
  };

  const handleRejectAll = () => {
    saveConsent({
      necessary: true,
      analytics: false,
      personalization: false
    });
  };

  const handleSavePreferences = () => {
    saveConsent(preferences);
  };

  // Function to open consent banner from outside
  const openConsentBanner = () => {
    setShowBanner(true);
  };

  // Provide consent context to children
  const contextValue = {
    consent,
    hasConsent: !!consent,
    analyticsEnabled: consent?.analytics || false,
    personalizationEnabled: consent?.personalization || false,
    openConsentBanner,
  };

  if (!showBanner) {
    return (
      <ConsentContext.Provider value={contextValue}>
        {null}
      </ConsentContext.Provider>
    );
  }

  return (
    <ConsentContext.Provider value={contextValue}>
      <div 
        className="fixed bottom-0 left-0 right-0 z-50 p-4"
        role="dialog"
        aria-labelledby="cookie-consent-title"
        aria-describedby="cookie-consent-description"
      >
        <div className={`max-w-4xl mx-auto rounded-lg shadow-2xl border transition-colors duration-300 ${
          darkMode 
            ? 'bg-gray-800 border-gray-700 text-white' 
            : 'bg-white border-gray-200 text-gray-800'
        }`}>
          <div className="p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center">
                <span className="text-2xl mr-3" aria-hidden="true">🍪</span>
                <h3 id="cookie-consent-title" className="text-lg font-semibold">
                  {t('gdpr.title', 'We Value Your Privacy')}
                </h3>
              </div>
              <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded dark:bg-blue-900 dark:text-blue-200">
                🇪🇺 GDPR
              </span>
            </div>
            
            <p 
              id="cookie-consent-description"
              className={`text-sm mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}
            >
              {t('gdpr.description', 'AISTANBUL uses cookies and similar technologies to enhance your experience, provide personalized recommendations, and analyze our traffic. We comply with GDPR and respect your privacy choices.')}
            </p>

            {!showDetails ? (
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={handleAcceptAll}
                  className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  aria-label={t('gdpr.acceptAll', 'Accept All Cookies')}
                >
                  {t('gdpr.acceptAll', 'Accept All')}
                </button>
                <button
                  onClick={handleRejectAll}
                  className={`px-5 py-2.5 border rounded-lg transition-colors duration-200 font-medium focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 ${
                    darkMode 
                      ? 'border-gray-600 hover:bg-gray-700' 
                      : 'border-gray-300 hover:bg-gray-50'
                  }`}
                  aria-label={t('gdpr.rejectAll', 'Reject All Optional Cookies')}
                >
                  {t('gdpr.rejectAll', 'Reject All')}
                </button>
                <button
                  onClick={() => setShowDetails(true)}
                  className={`px-5 py-2.5 border rounded-lg transition-colors duration-200 font-medium focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 ${
                    darkMode 
                      ? 'border-gray-600 hover:bg-gray-700' 
                      : 'border-gray-300 hover:bg-gray-50'
                  }`}
                  aria-expanded={showDetails}
                  aria-controls="cookie-preferences"
                >
                  {t('gdpr.customize', 'Customize')}
                </button>
              </div>
            ) : (
              <div id="cookie-preferences" className="space-y-4">
                <div className="space-y-3">
                  {/* Necessary Cookies */}
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
                    <div className="flex-1">
                      <h4 className="font-medium flex items-center">
                        <span className="mr-2">🔒</span>
                        {t('gdpr.necessary.title', 'Necessary Cookies')}
                        <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded dark:bg-green-900 dark:text-green-200">
                          {t('gdpr.required', 'Required')}
                        </span>
                      </h4>
                      <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {t('gdpr.necessary.description', 'Essential for the website to function properly. Cannot be disabled.')}
                      </p>
                    </div>
                    <input 
                      type="checkbox" 
                      checked={true} 
                      disabled 
                      className="w-5 h-5 rounded text-blue-600"
                      aria-label={t('gdpr.necessary.title', 'Necessary Cookies')}
                    />
                  </div>
                  
                  {/* Analytics Cookies */}
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
                    <div className="flex-1">
                      <h4 className="font-medium flex items-center">
                        <span className="mr-2">📊</span>
                        {t('gdpr.analytics.title', 'Analytics Cookies')}
                      </h4>
                      <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {t('gdpr.analytics.description', 'Help us understand how visitors interact with our website (Google Analytics).')}
                      </p>
                    </div>
                    <input 
                      type="checkbox" 
                      checked={preferences.analytics}
                      onChange={(e) => setPreferences({...preferences, analytics: e.target.checked})}
                      className="w-5 h-5 rounded text-blue-600 cursor-pointer"
                      aria-label={t('gdpr.analytics.title', 'Analytics Cookies')}
                    />
                  </div>
                  
                  {/* Personalization Cookies */}
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
                    <div className="flex-1">
                      <h4 className="font-medium flex items-center">
                        <span className="mr-2">✨</span>
                        {t('gdpr.personalization.title', 'Personalization Cookies')}
                      </h4>
                      <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {t('gdpr.personalization.description', 'Remember your preferences (theme, language) for a better experience.')}
                      </p>
                    </div>
                    <input 
                      type="checkbox" 
                      checked={preferences.personalization}
                      onChange={(e) => setPreferences({...preferences, personalization: e.target.checked})}
                      className="w-5 h-5 rounded text-blue-600 cursor-pointer"
                      aria-label={t('gdpr.personalization.title', 'Personalization Cookies')}
                    />
                  </div>
                </div>
                
                <div className={`flex flex-wrap gap-3 pt-4 border-t ${darkMode ? 'border-gray-600' : 'border-gray-300'}`}>
                  <button
                    onClick={handleSavePreferences}
                    className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  >
                    {t('gdpr.savePreferences', 'Save Preferences')}
                  </button>
                  <button
                    onClick={() => setShowDetails(false)}
                    className={`px-5 py-2.5 border rounded-lg transition-colors duration-200 font-medium focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 ${
                      darkMode 
                        ? 'border-gray-600 hover:bg-gray-700' 
                        : 'border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    {t('gdpr.back', 'Back')}
                  </button>
                </div>
              </div>
            )}
            
            <div className={`flex items-center justify-between mt-4 pt-4 border-t text-xs ${darkMode ? 'border-gray-700 text-gray-400' : 'border-gray-200 text-gray-500'}`}>
              <p>
                {t('gdpr.moreInfo', 'For more information, read our')}{' '}
                <Link to="/privacy" className="underline hover:no-underline text-blue-500">
                  {t('gdpr.privacyPolicy', 'Privacy Policy')}
                </Link>
                {' '}{t('gdpr.and', 'and')}{' '}
                <Link to="/gdpr" className="underline hover:no-underline text-blue-500">
                  {t('gdpr.gdprRights', 'GDPR Rights')}
                </Link>
              </p>
              <span className="text-gray-400">v{CONSENT_VERSION}</span>
            </div>
          </div>
        </div>
      </div>
    </ConsentContext.Provider>
  );
};

// Export a button component to reopen consent settings
export const ManageConsentButton = ({ className = '', children }) => {
  const { t } = useTranslation();
  
  const handleClick = () => {
    window.dispatchEvent(new Event('openConsentBanner'));
  };
  
  return (
    <button 
      onClick={handleClick}
      className={className || 'text-sm text-blue-600 hover:underline dark:text-blue-400'}
    >
      {children || t('gdpr.manageConsent', 'Manage Cookie Preferences')}
    </button>
  );
};

export default CookieConsent;
