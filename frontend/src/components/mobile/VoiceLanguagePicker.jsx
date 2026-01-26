/**
 * VoiceLanguagePicker Component
 * ==============================
 * A compact language selector for voice input
 * Supports: English, Turkish, Russian, French, Arabic, German
 */

import React, { useState, useRef, useEffect } from 'react';
import { SUPPORTED_VOICE_LANGUAGES, LANGUAGE_ORDER } from '../../hooks/useVoiceInput';
import './VoiceLanguagePicker.css';

const VoiceLanguagePicker = ({
  currentLanguage = 'en-US',
  onLanguageChange,
  darkMode = false,
  disabled = false,
  compact = true,  // Show only flag in compact mode
  className = ''
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const pickerRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (pickerRef.current && !pickerRef.current.contains(event.target)) {
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

  const currentLang = SUPPORTED_VOICE_LANGUAGES[currentLanguage] || SUPPORTED_VOICE_LANGUAGES['en-US'];

  const handleSelect = (langCode) => {
    onLanguageChange?.(langCode);
    setIsOpen(false);
    
    // Haptic feedback
    if ('vibrate' in navigator) {
      navigator.vibrate(10);
    }
  };

  const toggleDropdown = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsOpen(!isOpen);
    }
  };

  return (
    <div 
      ref={pickerRef}
      className={`voice-language-picker ${darkMode ? 'dark' : 'light'} ${compact ? 'compact' : ''} ${className}`}
    >
      {/* Current language button */}
      <button
        type="button"
        onClick={toggleDropdown}
        onTouchEnd={(e) => {
          e.preventDefault();
          toggleDropdown(e);
        }}
        className={`language-picker-button ${isOpen ? 'open' : ''}`}
        disabled={disabled}
        aria-label={`Voice language: ${currentLang.name}. Tap to change.`}
        aria-expanded={isOpen}
        title={`Voice: ${currentLang.name}`}
      >
        <span className="language-flag">{currentLang.flag}</span>
        {!compact && <span className="language-name">{currentLang.shortName}</span>}
        <svg 
          className={`dropdown-arrow ${isOpen ? 'open' : ''}`}
          width="10" 
          height="10" 
          viewBox="0 0 10 10" 
          fill="currentColor"
        >
          <path d="M2 4l3 3 3-3" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
        </svg>
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div className="language-dropdown">
          <div className="dropdown-header">Voice Language</div>
          {LANGUAGE_ORDER.map((langCode) => {
            const lang = SUPPORTED_VOICE_LANGUAGES[langCode];
            const isSelected = langCode === currentLanguage;
            
            return (
              <button
                key={langCode}
                type="button"
                onClick={() => handleSelect(langCode)}
                className={`language-option ${isSelected ? 'selected' : ''}`}
                aria-selected={isSelected}
              >
                <span className="option-flag">{lang.flag}</span>
                <span className="option-name">{lang.name}</span>
                {isSelected && (
                  <svg className="check-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default VoiceLanguagePicker;
