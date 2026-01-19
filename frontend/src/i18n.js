import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import { Logger } from './utils/logger.js';

const log = new Logger('i18n');

// Translation resources - bundled for instant loading
import enTranslation from './locales/en/translation.json';
import trTranslation from './locales/tr/translation.json';
import ruTranslation from './locales/ru/translation.json';
import deTranslation from './locales/de/translation.json';
import frTranslation from './locales/fr/translation.json';
import arTranslation from './locales/ar/translation.json';

const resources = {
  en: {
    translation: enTranslation
  },
  tr: {
    translation: trTranslation
  },
  ru: {
    translation: ruTranslation
  },
  de: {
    translation: deTranslation
  },
  fr: {
    translation: frTranslation
  },
  ar: {
    translation: arTranslation
  }
};

i18n
  // Detect user language
  .use(LanguageDetector)
  // Pass the i18n instance to react-i18next
  .use(initReactI18next)
  // Initialize i18next
  .init({
    resources, // Use bundled resources, not HTTP backend
    fallbackLng: 'en',
    lng: 'en', // Default language
    debug: import.meta.env.DEV,

    interpolation: {
      escapeValue: false, // React already escapes values
    },

    // Language detection options
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
      lookupLocalStorage: 'i18nextLng',
    },

    // React i18next options
    react: {
      useSuspense: false, // Don't use Suspense to avoid loading delays
      bindI18n: 'languageChanged',
      bindI18nStore: '',
      transEmptyNodeValue: '',
      transSupportBasicHtmlNodes: true,
      transKeepBasicHtmlNodesFor: ['br', 'strong', 'i'],
    },

    // Supported languages
    supportedLngs: ['en', 'tr', 'ru', 'de', 'fr', 'ar'],
    nonExplicitSupportedLngs: true,

    // Namespace configuration
    defaultNS: 'translation',
    ns: ['translation'],

    // Key separator
    keySeparator: '.',
    nsSeparator: ':',
    
    // Initialize synchronously since we have bundled resources
    initImmediate: false,
  })
  .then(() => {
    log.info('✅ i18n initialized successfully');
  })
  .catch((error) => {
    log.error('❌ i18n initialization error:', error);
  });

export default i18n;
