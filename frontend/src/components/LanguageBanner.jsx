/**
 * LanguageBanner Component
 * 
 * Shows a subtle banner when language changes
 * Better UX for multilingual users
 * 
 * Auto-dismisses after 3 seconds
 */

import React, { useState, useEffect } from 'react';

const LANGUAGE_NAMES = {
  'en': 'English',
  'tr': 'Türkçe',
  'fr': 'Français',
  'de': 'Deutsch',
  'ru': 'Русский',
  'ar': 'العربية',
  'es': 'Español',
  'it': 'Italiano',
  'zh': '中文',
  'ja': '日本語'
};

const LANGUAGE_FLAGS = {
  'en': '🇬🇧',
  'tr': '🇹🇷',
  'fr': '🇫🇷',
  'de': '🇩🇪',
  'ru': '🇷🇺',
  'ar': '🇸🇦',
  'es': '🇪🇸',
  'it': '🇮🇹',
  'zh': '🇨🇳',
  'ja': '🇯🇵'
};

/**
 * LanguageBanner Component
 * 
 * @param {Object} props
 * @param {string} props.language - Current language code (e.g., 'en', 'tr')
 * @param {boolean} props.darkMode - Dark mode flag
 * @param {number} props.autoDismissMs - Auto-dismiss time in ms (default: 3000)
 */
const LanguageBanner = ({ 
  language = 'en', 
  darkMode = false,
  autoDismissMs = 3000 
}) => {
  const [isVisible, setIsVisible] = useState(true);
  const [prevLanguage, setPrevLanguage] = useState(language);
  
  // Show banner when language changes
  useEffect(() => {
    if (language !== prevLanguage) {
      setIsVisible(true);
      setPrevLanguage(language);
      
      // Auto-dismiss after delay
      const timer = setTimeout(() => {
        setIsVisible(false);
      }, autoDismissMs);
      
      return () => clearTimeout(timer);
    }
  }, [language, prevLanguage, autoDismissMs]);
  
  if (!isVisible) {
    return null;
  }
  
  const languageName = LANGUAGE_NAMES[language] || language.toUpperCase();
  const languageFlag = LANGUAGE_FLAGS[language] || '🌐';
  
  return (
    <div className={`
      fixed top-20 left-1/2 transform -translate-x-1/2 z-50
      px-4 py-2 rounded-lg shadow-lg
      flex items-center gap-2
      animate-fade-in-down
      ${darkMode 
        ? 'bg-gray-800 border border-gray-700 text-gray-200' 
        : 'bg-white border border-gray-200 text-gray-800'
      }
    `}>
      <span className="text-xl">{languageFlag}</span>
      <span className="text-sm font-medium">
        Language: {languageName}
      </span>
      <button
        onClick={() => setIsVisible(false)}
        className={`
          ml-2 p-1 rounded-full transition-colors duration-200
          ${darkMode 
            ? 'hover:bg-gray-700 text-gray-400 hover:text-gray-200' 
            : 'hover:bg-gray-100 text-gray-500 hover:text-gray-700'
          }
        `}
        title="Dismiss"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
};

export default LanguageBanner;
