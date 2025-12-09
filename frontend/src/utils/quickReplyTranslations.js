/**
 * Quick Reply Translations
 * ========================
 * Multilingual quick reply suggestions for Istanbul AI chatbot
 * Supports: English, Turkish, Arabic, French, German, Russian
 * 
 * Usage:
 *   import { getTranslatedSuggestions } from './quickReplyTranslations';
 *   const suggestions = getTranslatedSuggestions('restaurant', 'tr');
 */

export const QUICK_REPLY_TRANSLATIONS = {
  // General navigation suggestions
  restaurants: {
    en: ['Show restaurants', 'Find dining', 'Turkish cuisine', 'Best food'],
    tr: ['Restoranları göster', 'Yemek bul', 'Türk mutfağı', 'En iyi yemekler'],
    ar: ['أظهر المطاعم', 'ابحث عن طعام', 'المطبخ التركي', 'أفضل الأطعمة'],
    fr: ['Afficher restaurants', 'Trouver repas', 'Cuisine turque', 'Meilleurs plats'],
    de: ['Restaurants zeigen', 'Essen finden', 'Türkische Küche', 'Bestes Essen'],
    ru: ['Показать рестораны', 'Найти еду', 'Турецкая кухня', 'Лучшая еда']
  },
  
  attractions: {
    en: ['Find attractions', 'Museums', 'Historical sites', 'Top places'],
    tr: ['Gezilecek yerler', 'Müzeler', 'Tarihi yerler', 'En iyi yerler'],
    ar: ['أماكن سياحية', 'المتاحف', 'مواقع تاريخية', 'أفضل الأماكن'],
    fr: ['Trouver attractions', 'Musées', 'Sites historiques', 'Meilleurs lieux'],
    de: ['Sehenswürdigkeiten', 'Museen', 'Historische Orte', 'Top-Orte'],
    ru: ['Достопримечательности', 'Музеи', 'Исторические места', 'Лучшие места']
  },
  
  directions: {
    en: ['Get directions', 'Navigate', 'Show route', 'How to get there'],
    tr: ['Yol tarifi al', 'Yönlendir', 'Rotayı göster', 'Nasıl giderim'],
    ar: ['احصل على الاتجاهات', 'التنقل', 'إظهار المسار', 'كيف أصل'],
    fr: ['Obtenir itinéraire', 'Naviguer', 'Afficher route', 'Comment y aller'],
    de: ['Wegbeschreibung', 'Navigieren', 'Route zeigen', 'Wie komme ich hin'],
    ru: ['Получить маршрут', 'Навигация', 'Показать путь', 'Как добраться']
  },
  
  weather: {
    en: ['Weather today', '5-day forecast', 'What to wear', 'Best time to visit'],
    tr: ['Bugün hava', '5 günlük tahmin', 'Ne giymeli', 'En iyi ziyaret zamanı'],
    ar: ['الطقس اليوم', 'توقعات 5 أيام', 'ماذا أرتدي', 'أفضل وقت للزيارة'],
    fr: ['Météo aujourd\'hui', 'Prévisions 5 jours', 'Que porter', 'Meilleur moment'],
    de: ['Wetter heute', '5-Tage-Prognose', 'Was anziehen', 'Beste Besuchszeit'],
    ru: ['Погода сегодня', 'Прогноз на 5 дней', 'Что надеть', 'Лучшее время']
  },
  
  // Context-specific suggestions
  restaurant_context: {
    en: ['Show on map', 'Get directions', 'More options', 'Find nearby'],
    tr: ['Haritada göster', 'Yol tarifi', 'Daha fazla seçenek', 'Yakınları bul'],
    ar: ['إظهار على الخريطة', 'احصل على الاتجاهات', 'المزيد من الخيارات', 'ابحث في الجوار'],
    fr: ['Afficher sur carte', 'Obtenir itinéraire', 'Plus d\'options', 'Trouver à proximité'],
    de: ['Auf Karte zeigen', 'Wegbeschreibung', 'Mehr Optionen', 'In der Nähe finden'],
    ru: ['Показать на карте', 'Маршрут', 'Больше вариантов', 'Найти рядом']
  },
  
  attraction_context: {
    en: ['Show on map', 'Opening hours', 'How to get there', 'More like this'],
    tr: ['Haritada göster', 'Açılış saatleri', 'Nasıl giderim', 'Benzerleri'],
    ar: ['إظهار على الخريطة', 'ساعات العمل', 'كيف أصل', 'المزيد مثل هذا'],
    fr: ['Afficher sur carte', 'Heures d\'ouverture', 'Comment y aller', 'Plus comme ça'],
    de: ['Auf Karte zeigen', 'Öffnungszeiten', 'Wie komme ich hin', 'Mehr davon'],
    ru: ['Показать на карте', 'Часы работы', 'Как добраться', 'Похожие места']
  },
  
  navigation_context: {
    en: ['Start navigation', 'Public transport', 'Walking route', 'Drive there'],
    tr: ['Navigasyonu başlat', 'Toplu taşıma', 'Yürüyüş rotası', 'Araçla git'],
    ar: ['بدء الملاحة', 'النقل العام', 'طريق المشي', 'القيادة هناك'],
    fr: ['Démarrer navigation', 'Transport public', 'Itinéraire piéton', 'Conduire'],
    de: ['Navigation starten', 'Öffentliche Verkehrsmittel', 'Fußweg', 'Fahren'],
    ru: ['Начать навигацию', 'Общественный транспорт', 'Пешком', 'Проехать']
  },
  
  question_context: {
    en: ['Yes', 'No', 'Tell me more', 'Show examples'],
    tr: ['Evet', 'Hayır', 'Daha fazla anlat', 'Örnekleri göster'],
    ar: ['نعم', 'لا', 'أخبرني المزيد', 'أظهر أمثلة'],
    fr: ['Oui', 'Non', 'Dites-m\'en plus', 'Montrer exemples'],
    de: ['Ja', 'Nein', 'Mehr erzählen', 'Beispiele zeigen'],
    ru: ['Да', 'Нет', 'Расскажи больше', 'Покажи примеры']
  },
  
  // Location-specific
  taksim: {
    en: ['Restaurants in Taksim', 'Things to do', 'Nightlife', 'How to get there'],
    tr: ['Taksim\'de restoranlar', 'Yapılacaklar', 'Gece hayatı', 'Nasıl giderim'],
    ar: ['مطاعم في تقسيم', 'أشياء للقيام بها', 'الحياة الليلية', 'كيف أصل'],
    fr: ['Restaurants à Taksim', 'Choses à faire', 'Vie nocturne', 'Comment y aller'],
    de: ['Restaurants in Taksim', 'Was tun', 'Nachtleben', 'Wie komme ich hin'],
    ru: ['Рестораны в Таксим', 'Что делать', 'Ночная жизнь', 'Как добраться']
  },
  
  sultanahmet: {
    en: ['Blue Mosque', 'Hagia Sophia', 'Grand Bazaar', 'Historical tour'],
    tr: ['Sultanahmet Camii', 'Ayasofya', 'Kapalıçarşı', 'Tarihi tur'],
    ar: ['المسجد الأزرق', 'آيا صوفيا', 'البازار الكبير', 'جولة تاريخية'],
    fr: ['Mosquée Bleue', 'Sainte-Sophie', 'Grand Bazar', 'Visite historique'],
    de: ['Blaue Moschee', 'Hagia Sophia', 'Großer Basar', 'Historische Tour'],
    ru: ['Голубая мечеть', 'Айя-София', 'Гранд базар', 'Историческая экскурсия']
  },
  
  // Default fallbacks
  default: {
    en: ['Show restaurants', 'Find attractions', 'Get directions', 'Tell me more'],
    tr: ['Restoranları göster', 'Gezilecek yerler', 'Yol tarifi', 'Daha fazla anlat'],
    ar: ['أظهر المطاعم', 'أماكن سياحية', 'الاتجاهات', 'أخبرني المزيد'],
    fr: ['Afficher restaurants', 'Trouver attractions', 'Itinéraire', 'Dites-m\'en plus'],
    de: ['Restaurants zeigen', 'Sehenswürdigkeiten', 'Wegbeschreibung', 'Mehr erzählen'],
    ru: ['Показать рестораны', 'Достопримечательности', 'Маршрут', 'Расскажи больше']
  }
};

/**
 * Detect category from message context
 */
export function detectCategory(message) {
  if (!message) return 'default';
  
  const lower = message.toLowerCase();
  
  // Restaurant context
  if (lower.includes('restaurant') || lower.includes('dining') || lower.includes('food')) {
    return 'restaurant_context';
  }
  
  // Attraction context
  if (lower.includes('museum') || lower.includes('attraction') || 
      lower.includes('landmark') || lower.includes('tower') || lower.includes('mosque')) {
    return 'attraction_context';
  }
  
  // Navigation context
  if (lower.includes('direction') || lower.includes('get there') ||
      lower.includes('how do i go') || lower.includes('navigate')) {
    return 'navigation_context';
  }
  
  // Question context
  if (lower.endsWith('?')) {
    return 'question_context';
  }
  
  // Weather context
  if (lower.includes('weather') || lower.includes('temperature') || lower.includes('forecast')) {
    return 'weather';
  }
  
  // Location-specific
  if (lower.includes('taksim')) return 'taksim';
  if (lower.includes('sultanahmet')) return 'sultanahmet';
  
  return 'default';
}

/**
 * Get translated suggestions based on context and language
 * 
 * @param {string} context - Context category or message
 * @param {string} language - Language code (en, tr, ar, fr, de, ru)
 * @returns {string[]} - Array of translated suggestions
 */
export function getTranslatedSuggestions(context, language = 'en') {
  // Normalize language code
  const lang = language.toLowerCase().split('-')[0]; // 'en-US' -> 'en'
  
  // Detect category if context is a message
  const category = QUICK_REPLY_TRANSLATIONS[context] 
    ? context 
    : detectCategory(context);
  
  // Get suggestions for category and language
  const suggestions = QUICK_REPLY_TRANSLATIONS[category];
  
  if (!suggestions) {
    return QUICK_REPLY_TRANSLATIONS.default[lang] || QUICK_REPLY_TRANSLATIONS.default.en;
  }
  
  return suggestions[lang] || suggestions.en;
}

/**
 * Get all supported languages
 */
export function getSupportedLanguages() {
  return [
    { code: 'en', name: 'English', nativeName: 'English' },
    { code: 'tr', name: 'Turkish', nativeName: 'Türkçe' },
    { code: 'ar', name: 'Arabic', nativeName: 'العربية' },
    { code: 'fr', name: 'French', nativeName: 'Français' },
    { code: 'de', name: 'German', nativeName: 'Deutsch' },
    { code: 'ru', name: 'Russian', nativeName: 'Русский' }
  ];
}

export default {
  getTranslatedSuggestions,
  detectCategory,
  getSupportedLanguages,
  QUICK_REPLY_TRANSLATIONS
};
