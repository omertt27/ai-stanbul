import React, { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import './LanguageSwitcher.css';

const LanguageSwitcher = () => {
  const { i18n } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const languages = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'tr', name: 'Türkçe', flag: '🇹🇷' },
    { code: 'ru', name: 'Русский', flag: '🇷🇺' },
    { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
    { code: 'fr', name: 'Français', flag: '🇫🇷' },
    { code: 'ar', name: 'العربية', flag: '🇸🇦' }
  ];

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('touchstart', handleClickOutside);
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside);
    };
  }, []);

  const changeLanguage = (languageCode) => {
    i18n.changeLanguage(languageCode);
    setIsOpen(false);
    
    // Handle RTL for Arabic
    if (languageCode === 'ar') {
      document.documentElement.setAttribute('dir', 'rtl');
      document.documentElement.setAttribute('lang', 'ar');
    } else {
      document.documentElement.setAttribute('dir', 'ltr');
      document.documentElement.setAttribute('lang', languageCode);
    }
  };

  const toggleDropdown = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsOpen(!isOpen);
  };

  return (
    <div className="language-switcher" ref={dropdownRef}>
      <div className={`language-dropdown ${isOpen ? 'open' : ''}`}>
        <button 
          className="language-button"
          onClick={toggleDropdown}
          onTouchEnd={(e) => {
            e.preventDefault();
            toggleDropdown(e);
          }}
          aria-expanded={isOpen}
          aria-haspopup="listbox"
        >
          <span className="current-flag">
            {languages.find(lang => lang.code === i18n.language)?.flag || '🌐'}
          </span>
          <span className="current-language">
            {languages.find(lang => lang.code === i18n.language)?.name || 'Language'}
          </span>
          <span className={`dropdown-arrow ${isOpen ? 'rotated' : ''}`}>▼</span>
        </button>
        <div className={`language-options ${isOpen ? 'visible' : ''}`} role="listbox">
          {languages.map((language) => (
            <button
              key={language.code}
              className={`language-option ${i18n.language === language.code ? 'active' : ''}`}
              onClick={() => changeLanguage(language.code)}
              data-lang={language.code}
              role="option"
              aria-selected={i18n.language === language.code}
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
