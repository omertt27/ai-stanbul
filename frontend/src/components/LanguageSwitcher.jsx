import React from 'react';
import { useTranslation } from 'react-i18next';
import './LanguageSwitcher.css';

const LanguageSwitcher = () => {
  const { i18n } = useTranslation();

  const languages = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'tr', name: 'Türkçe', flag: '🇹🇷' },
    { code: 'ru', name: 'Русский', flag: '🇷🇺' },
    { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
    { code: 'fr', name: 'Français', flag: '🇫🇷' },
    { code: 'ar', name: 'العربية', flag: '🇸🇦' }
  ];

  const changeLanguage = (languageCode) => {
    i18n.changeLanguage(languageCode);
    
    // Handle RTL for Arabic
    if (languageCode === 'ar') {
      document.documentElement.setAttribute('dir', 'rtl');
      document.documentElement.setAttribute('lang', 'ar');
    } else {
      document.documentElement.setAttribute('dir', 'ltr');
      document.documentElement.setAttribute('lang', languageCode);
    }
  };

  return (
    <div className="language-switcher">
      <div className="language-dropdown">
        <button className="language-button">
          <span className="current-flag">
            {languages.find(lang => lang.code === i18n.language)?.flag || '🌐'}
          </span>
          <span className="current-language">
            {languages.find(lang => lang.code === i18n.language)?.name || 'Language'}
          </span>
          <span className="dropdown-arrow">▼</span>
        </button>
        <div className="language-options">
          {languages.map((language) => (
            <button
              key={language.code}
              className={`language-option ${i18n.language === language.code ? 'active' : ''}`}
              onClick={() => changeLanguage(language.code)}
              data-lang={language.code}
            >
              <span className="flag">{language.flag}</span>
              <span className="name">{language.name}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LanguageSwitcher;
