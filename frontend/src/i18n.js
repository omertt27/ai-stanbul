import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import Backend from 'i18next-http-backend';

// Translation resources
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
  // Load translation using http backend
  .use(Backend)
  // Detect user language
  .use(LanguageDetector)
  // Pass the i18n instance to react-i18next
  .use(initReactI18next)
  // Initialize i18next
  .init({
    resources,
    fallbackLng: 'en',
    lng: 'en', // Default language
    debug: process.env.NODE_ENV === 'development',

    interpolation: {
      escapeValue: false, // React already escapes values
    },

    // Language detection options
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
      lookupLocalStorage: 'i18nextLng',
    },

    // Backend options for loading translations
    backend: {
      loadPath: '/locales/{{lng}}/{{ns}}.json',
    },

    // React i18next options
    react: {
      useSuspense: false,
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
  });

export default i18n;
