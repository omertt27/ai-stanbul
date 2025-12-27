"""
Multilingual Intent Keywords
============================
Language-aware keyword maps for intent detection across supported languages.

This ensures non-English queries (Turkish, Russian, German, Arabic) are
correctly routed to specialized handlers like Hidden Gems instead of
falling back to the generic LLM.
"""

from typing import Dict, List, Set

# =============================================================================
# HIDDEN GEMS KEYWORDS - Multilingual
# =============================================================================
HIDDEN_GEMS_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "hidden", "secret", "off the beaten path", "local secrets", 
        "hidden gems", "secret spots", "local spot", "locals know",
        "undiscovered", "lesser known", "not touristy", "authentic local",
        "insider", "underground", "obscure"
    ],
    "tr": [
        "gizli", "saklı", "yerel", "kimsenin bilmediği", "turistlerin bilmediği",
        "keşfedilmemiş", "az bilinen", "yerel halkın gittiği", "sır",
        "yerel sırlar", "gizli mekan", "saklı yer", "bilinmeyen",
        "yerel favoriler", "turist olmayan"
    ],
    "ru": [
        "скрытые", "секретные", "местные", "тайные", "неизвестные",
        "скрытые жемчужины", "секретные места", "местные секреты",
        "нетуристические", "малоизвестные", "аутентичные"
    ],
    "de": [
        "versteckt", "geheim", "abseits", "geheimtipp", "lokale",
        "versteckte juwelen", "geheime orte", "unbekannt", "authentisch",
        "nicht touristisch", "einheimische", "insider"
    ],
    "ar": [
        "مخفية", "سرية", "محلية", "أسرار", "غير معروفة",
        "أماكن سرية", "جواهر مخفية", "أماكن محلية",
        "خارج المسار السياحي", "أصيلة"
    ]
}

# =============================================================================
# RESTAURANT KEYWORDS - Multilingual
# =============================================================================
RESTAURANT_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "restaurant", "food", "eat", "dining", "meal", "lunch", "dinner",
        "breakfast", "cafe", "coffee", "hungry", "cuisine", "kitchen",
        "brunch", "snack", "takeaway", "delivery"
    ],
    "tr": [
        "restoran", "yemek", "lokanta", "kahvaltı", "öğle yemeği", "akşam yemeği",
        "kafe", "kahve", "yiyecek", "aç", "mutfak", "mekan",
        "kebap", "döner", "pide", "lahmacun"
    ],
    "ru": [
        "ресторан", "еда", "кафе", "завтрак", "обед", "ужин",
        "кухня", "поесть", "голодный", "кофе"
    ],
    "de": [
        "restaurant", "essen", "café", "frühstück", "mittagessen", "abendessen",
        "küche", "hungrig", "kaffee", "speisen"
    ],
    "ar": [
        "مطعم", "طعام", "أكل", "إفطار", "غداء", "عشاء",
        "مقهى", "قهوة", "مطبخ", "جائع"
    ]
}

# =============================================================================
# TRANSPORTATION KEYWORDS - Multilingual
# =============================================================================
TRANSPORTATION_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "metro", "bus", "tram", "ferry", "taxi", "uber", "transport",
        "how to get", "directions", "route", "travel", "commute",
        "istanbulkart", "public transport", "subway", "train"
    ],
    "tr": [
        "metro", "otobüs", "tramvay", "vapur", "taksi", "ulaşım",
        "nasıl giderim", "yol tarifi", "rota", "toplu taşıma",
        "istanbulkart", "marmaray", "metrobüs"
    ],
    "ru": [
        "метро", "автобус", "трамвай", "паром", "такси", "транспорт",
        "как добраться", "маршрут", "направления"
    ],
    "de": [
        "metro", "bus", "straßenbahn", "fähre", "taxi", "transport",
        "wie komme ich", "wegbeschreibung", "route", "öffentliche verkehrsmittel"
    ],
    "ar": [
        "مترو", "حافلة", "ترام", "عبارة", "تاكسي", "مواصلات",
        "كيف أصل", "الاتجاهات", "طريق"
    ]
}

# =============================================================================
# ATTRACTIONS/PLACES KEYWORDS - Multilingual
# =============================================================================
ATTRACTIONS_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "museum", "mosque", "palace", "attraction", "sightseeing", "visit",
        "tourist", "landmark", "monument", "historical", "tour",
        "hagia sophia", "blue mosque", "topkapi", "basilica cistern"
    ],
    "tr": [
        "müze", "cami", "saray", "gezi", "turistik", "ziyaret",
        "tarihi", "anıt", "tur", "mekan",
        "ayasofya", "sultanahmet", "topkapı", "yerebatan"
    ],
    "ru": [
        "музей", "мечеть", "дворец", "достопримечательность", "экскурсия",
        "посетить", "туристический", "памятник", "исторический"
    ],
    "de": [
        "museum", "moschee", "palast", "sehenswürdigkeit", "besichtigung",
        "besuchen", "touristisch", "denkmal", "historisch"
    ],
    "ar": [
        "متحف", "مسجد", "قصر", "معلم سياحي", "زيارة",
        "سياحي", "أثري", "تاريخي", "جولة"
    ]
}

# =============================================================================
# WEATHER KEYWORDS - Multilingual
# =============================================================================
WEATHER_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "weather", "temperature", "rain", "sunny", "cloudy", "forecast",
        "cold", "hot", "warm", "humid", "wind"
    ],
    "tr": [
        "hava", "hava durumu", "sıcaklık", "yağmur", "güneşli", "bulutlu",
        "soğuk", "sıcak", "nem", "rüzgar"
    ],
    "ru": [
        "погода", "температура", "дождь", "солнечно", "облачно", "прогноз",
        "холодно", "жарко", "тепло"
    ],
    "de": [
        "wetter", "temperatur", "regen", "sonnig", "bewölkt", "vorhersage",
        "kalt", "heiß", "warm"
    ],
    "ar": [
        "طقس", "درجة الحرارة", "مطر", "مشمس", "غائم", "توقعات",
        "بارد", "حار", "دافئ"
    ]
}

# =============================================================================
# EVENTS KEYWORDS - Multilingual
# =============================================================================
EVENTS_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "event", "events", "concert", "festival", "show", "exhibition",
        "happening", "what's on", "tonight", "this weekend", "performance",
        "theater", "theatre", "live music", "party", "nightlife", "club"
    ],
    "tr": [
        "etkinlik", "etkinlikler", "konser", "festival", "gösteri", "sergi",
        "ne var", "bu akşam", "bu hafta sonu", "tiyatro", "canlı müzik",
        "parti", "gece hayatı", "kulüp", "sahne"
    ],
    "ru": [
        "событие", "события", "концерт", "фестиваль", "шоу", "выставка",
        "что происходит", "сегодня вечером", "в эти выходные", "театр",
        "живая музыка", "вечеринка", "ночная жизнь"
    ],
    "de": [
        "veranstaltung", "ereignis", "konzert", "festival", "show", "ausstellung",
        "was ist los", "heute abend", "am wochenende", "theater",
        "live musik", "party", "nachtleben"
    ],
    "ar": [
        "حدث", "أحداث", "حفلة", "مهرجان", "عرض", "معرض",
        "ما يحدث", "الليلة", "نهاية الأسبوع", "مسرح",
        "موسيقى حية", "حفلة", "الحياة الليلية"
    ]
}

# =============================================================================
# NEIGHBORHOOD/AREA GUIDE KEYWORDS - Multilingual
# =============================================================================
NEIGHBORHOOD_GUIDE_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "neighborhood", "area", "district", "guide", "explore", "about",
        "what's in", "things to do in", "worth visiting", "vibe", "atmosphere",
        "local area", "tell me about", "describe"
    ],
    "tr": [
        "mahalle", "semt", "bölge", "rehber", "keşfet", "hakkında",
        "neler var", "yapılacak şeyler", "ziyaret edilmeli", "ortam", "atmosfer",
        "anlat", "tarif et"
    ],
    "ru": [
        "район", "квартал", "округ", "гид", "исследовать", "о",
        "что есть в", "чем заняться", "стоит посетить", "атмосфера",
        "расскажи о", "опиши"
    ],
    "de": [
        "viertel", "stadtteil", "bezirk", "führer", "erkunden", "über",
        "was gibt es in", "aktivitäten", "sehenswert", "atmosphäre",
        "erzähl mir von", "beschreibe"
    ],
    "ar": [
        "حي", "منطقة", "دليل", "استكشف", "عن", "ماذا يوجد في",
        "أشياء للقيام بها", "يستحق الزيارة", "أجواء",
        "أخبرني عن", "صف"
    ]
}

# =============================================================================
# ROUTE PLANNING KEYWORDS - Multilingual (Multi-stop itinerary)
# =============================================================================
ROUTE_PLANNING_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "plan", "itinerary", "day trip", "schedule", "visit multiple",
        "best route", "walking tour", "one day", "two days", "half day",
        "plan my day", "what order", "efficient route", "tour route"
    ],
    "tr": [
        "plan", "gezi planı", "günlük tur", "program", "çoklu ziyaret",
        "en iyi rota", "yürüyüş turu", "bir günlük", "iki günlük", "yarım gün",
        "günümü planla", "hangi sıra", "verimli rota"
    ],
    "ru": [
        "план", "маршрут", "однодневная поездка", "расписание",
        "лучший маршрут", "пешая экскурсия", "один день", "два дня",
        "спланируй мой день", "в каком порядке", "эффективный маршрут"
    ],
    "de": [
        "plan", "reiseroute", "tagesausflug", "zeitplan",
        "beste route", "stadtrundgang", "ein tag", "zwei tage",
        "plane meinen tag", "welche reihenfolge", "effiziente route"
    ],
    "ar": [
        "خطة", "جدول", "رحلة يومية", "جدول زمني",
        "أفضل طريق", "جولة مشي", "يوم واحد", "يومين",
        "خطط يومي", "أي ترتيب", "طريق فعال"
    ]
}

# =============================================================================
# DAILY TALKS / CASUAL CONVERSATION KEYWORDS - Multilingual
# =============================================================================
DAILY_TALKS_KEYWORDS: Dict[str, List[str]] = {
    "en": [
        "hello", "hi", "hey", "good morning", "good evening", "good night",
        "thanks", "thank you", "bye", "goodbye", "how are you",
        "what's up", "help", "bored", "tired", "confused",
        "nice", "great", "awesome", "cool", "wow"
    ],
    "tr": [
        "merhaba", "selam", "günaydın", "iyi akşamlar", "iyi geceler",
        "teşekkürler", "sağol", "hoşçakal", "görüşürüz", "nasılsın",
        "ne var ne yok", "yardım", "sıkıldım", "yoruldum", "kafam karıştı",
        "güzel", "harika", "süper", "vay"
    ],
    "ru": [
        "привет", "здравствуйте", "доброе утро", "добрый вечер", "спокойной ночи",
        "спасибо", "благодарю", "пока", "до свидания", "как дела",
        "что нового", "помощь", "скучно", "устал", "запутался",
        "отлично", "круто", "вау"
    ],
    "de": [
        "hallo", "hi", "guten morgen", "guten abend", "gute nacht",
        "danke", "tschüss", "auf wiedersehen", "wie geht's",
        "was gibt's", "hilfe", "langweilig", "müde", "verwirrt",
        "schön", "toll", "super", "wow"
    ],
    "ar": [
        "مرحبا", "أهلا", "صباح الخير", "مساء الخير", "تصبح على خير",
        "شكرا", "وداعا", "إلى اللقاء", "كيف حالك",
        "ما الجديد", "مساعدة", "ملل", "تعب", "محتار",
        "جميل", "رائع", "واو"
    ]
}

# =============================================================================
# NEIGHBORHOOD KEYWORDS - For location extraction
# =============================================================================
NEIGHBORHOOD_KEYWORDS: Dict[str, List[str]] = {
    "all": [  # These work across all languages (proper nouns)
        "balat", "beyoğlu", "beyoglu", "kadıköy", "kadikoy", "üsküdar", "uskudar",
        "beşiktaş", "besiktas", "sarıyer", "sariyer", "fatih", "sultanahmet",
        "taksim", "galata", "karaköy", "karakoy", "ortaköy", "ortakoy",
        "moda", "cihangir", "bebek", "arnavutköy", "arnavutkoy",
        "eminönü", "eminonu", "kuzguncuk", "çengelköy", "cengelkoy"
    ]
}


def get_keywords_for_intent(intent: str, language: str) -> List[str]:
    """Get keywords for a specific intent and language"""
    keyword_maps = {
        "hidden_gems": HIDDEN_GEMS_KEYWORDS,
        "restaurant": RESTAURANT_KEYWORDS,
        "transportation": TRANSPORTATION_KEYWORDS,
        "attractions": ATTRACTIONS_KEYWORDS,
        "weather": WEATHER_KEYWORDS,
        "events": EVENTS_KEYWORDS,
        "neighborhood_guide": NEIGHBORHOOD_GUIDE_KEYWORDS,
        "route_planning": ROUTE_PLANNING_KEYWORDS,
        "daily_talks": DAILY_TALKS_KEYWORDS
    }
    
    keyword_map = keyword_maps.get(intent, {})
    return keyword_map.get(language, keyword_map.get("en", []))


def get_all_keywords_for_intent(intent: str) -> Dict[str, List[str]]:
    """Get all language keywords for a specific intent"""
    keyword_maps = {
        "hidden_gems": HIDDEN_GEMS_KEYWORDS,
        "restaurant": RESTAURANT_KEYWORDS,
        "transportation": TRANSPORTATION_KEYWORDS,
        "attractions": ATTRACTIONS_KEYWORDS,
        "weather": WEATHER_KEYWORDS,
        "events": EVENTS_KEYWORDS,
        "neighborhood_guide": NEIGHBORHOOD_GUIDE_KEYWORDS,
        "route_planning": ROUTE_PLANNING_KEYWORDS,
        "daily_talks": DAILY_TALKS_KEYWORDS
    }
    return keyword_maps.get(intent, {})


def detect_intent_multilingual(query: str, language: str = None) -> tuple:
    """
    Detect intent from query using multilingual keywords.
    
    ALWAYS checks ALL languages to handle cases like "merhaba" (Turkish word
    with no special characters that would be detected as English).
    
    Args:
        query: User query
        language: Optional language hint (used for response language, not filtering)
    
    Returns:
        (intent, confidence, matched_keywords, detected_language)
    """
    query_lower = query.lower()
    
    # Define intent priority order
    intents = [
        ("hidden_gems", HIDDEN_GEMS_KEYWORDS),
        ("restaurant", RESTAURANT_KEYWORDS),
        ("transportation", TRANSPORTATION_KEYWORDS),
        ("attractions", ATTRACTIONS_KEYWORDS),
        ("weather", WEATHER_KEYWORDS),
        ("events", EVENTS_KEYWORDS),
        ("neighborhood_guide", NEIGHBORHOOD_GUIDE_KEYWORDS),
        ("route_planning", ROUTE_PLANNING_KEYWORDS),
        ("daily_talks", DAILY_TALKS_KEYWORDS)
    ]
    
    best_intent = None
    best_score = 0
    best_matches = []
    detected_lang = language or 'en'
    
    for intent_name, keyword_map in intents:
        # ALWAYS check ALL languages - don't filter by detected language
        # This ensures "merhaba" (Turkish with no special chars) is detected
        for lang in keyword_map.keys():
            keywords = keyword_map.get(lang, [])
            matches = [kw for kw in keywords if kw.lower() in query_lower]
            
            if len(matches) > best_score:
                best_score = len(matches)
                best_intent = intent_name
                best_matches = matches
                detected_lang = lang  # Update detected language based on match
    
    # Calculate confidence based on matches
    confidence = min(best_score * 0.3, 1.0) if best_score > 0 else 0.0
    
    return best_intent, confidence, best_matches, detected_lang


def detect_hidden_gems_intent(query: str, language: str = None) -> bool:
    """
    Specifically detect if query is about hidden gems.
    
    Args:
        query: User query
        language: Optional language code
    
    Returns:
        True if query is about hidden gems
    """
    query_lower = query.lower()
    
    # Check all languages if not specified
    languages = [language] if language else list(HIDDEN_GEMS_KEYWORDS.keys())
    
    for lang in languages:
        keywords = HIDDEN_GEMS_KEYWORDS.get(lang, [])
        if any(keyword.lower() in query_lower for keyword in keywords):
            return True
    
    return False


def extract_neighborhood(query: str) -> str:
    """Extract neighborhood name from query"""
    query_lower = query.lower()
    
    for neighborhood in NEIGHBORHOOD_KEYWORDS["all"]:
        if neighborhood.lower() in query_lower:
            return neighborhood
    
    return None
