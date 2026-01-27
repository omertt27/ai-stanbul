import React, { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

const LanguageSwitcher = () => {
  const { i18n } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

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
    if (!isOpen) return;
    
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    // Small delay to prevent immediate close
    const timer = setTimeout(() => {
      document.addEventListener('click', handleClickOutside, true);
    }, 50);
    
    return () => {
      clearTimeout(timer);
      document.removeEventListener('click', handleClickOutside, true);
    };
  }, [isOpen]);

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

  const currentLang = languages.find(lang => lang.code === i18n.language) || languages[0];

  // Inline styles to avoid any CSS conflicts
  const containerStyle = {
    position: 'relative',
    display: 'inline-block',
    zIndex: 999999,
  };

  const buttonStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 16px',
    background: 'rgba(30, 41, 59, 0.9)',
    border: '1px solid rgba(139, 92, 246, 0.3)',
    borderRadius: '8px',
    cursor: 'pointer',
    color: '#e2e8f0',
    fontSize: '14px',
    minWidth: '120px',
    transition: 'all 0.2s ease',
  };

  const dropdownStyle = {
    position: 'absolute',
    top: '100%',
    right: 0,
    marginTop: '8px',
    background: 'rgba(15, 16, 17, 0.98)',
    border: '1px solid rgba(139, 92, 246, 0.3)',
    borderRadius: '12px',
    boxShadow: '0 15px 35px rgba(0, 0, 0, 0.3)',
    zIndex: 999999,
    minWidth: '180px',
    padding: '8px',
  };

  const optionStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    width: '100%',
    padding: '10px 12px',
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    color: '#e2e8f0',
    fontSize: '14px',
    borderRadius: '8px',
    textAlign: 'left',
    transition: 'background 0.2s ease',
  };

  const activeOptionStyle = {
    ...optionStyle,
    background: 'rgba(139, 92, 246, 0.25)',
    color: '#8b5cf6',
    fontWeight: 600,
  };

  return (
    <div ref={containerRef} style={containerStyle}>
      <button 
        type="button"
        style={buttonStyle}
        onClick={() => {
          console.log('Button clicked! isOpen was:', isOpen);
          setIsOpen(!isOpen);
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(139, 92, 246, 0.2)';
          e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.5)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(30, 41, 59, 0.9)';
          e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.3)';
        }}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        <span style={{ fontSize: '1.3em' }}>{currentLang.flag}</span>
        <span style={{ flex: 1 }}>{currentLang.name}</span>
        <span style={{ 
          fontSize: '0.7em', 
          transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
          transition: 'transform 0.2s ease'
        }}>▼</span>
      </button>
      
      {isOpen && (
        <div style={dropdownStyle} role="listbox">
          {languages.map((language) => (
            <button
              key={language.code}
              type="button"
              style={i18n.language === language.code ? activeOptionStyle : optionStyle}
              onClick={() => changeLanguage(language.code)}
              onMouseEnter={(e) => {
                if (i18n.language !== language.code) {
                  e.currentTarget.style.background = 'rgba(139, 92, 246, 0.15)';
                }
              }}
              onMouseLeave={(e) => {
                if (i18n.language !== language.code) {
                  e.currentTarget.style.background = 'transparent';
                }
              }}
              role="option"
              aria-selected={i18n.language === language.code}
            >
              <span style={{ fontSize: '1.3em' }}>{language.flag}</span>
              <span>{language.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default LanguageSwitcher;
