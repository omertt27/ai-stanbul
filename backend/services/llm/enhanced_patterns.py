"""
enhanced_patterns.py - Enhanced Signal Patterns with Slang & Colloquial Language

Extends base signal patterns with:
- Slang and colloquial expressions
- Regional variations
- Common misspellings
- Implicit intent phrases

Author: AI Istanbul Team
Date: January 2025
"""

from typing import Dict, List

# Enhanced restaurant patterns with slang/colloquial
RESTAURANT_SLANG = {
    'en': [
        # Slang & colloquial
        r'\b(grab\s+a\s+bite|grab\s+some\s+food|get\s+a\s+bite)\b',
        r'\b(chow\s+down|grub|munchies|nosh)\b',
        r'\b(somewhere\s+to\s+eat|place\s+to\s+eat|spot\s+to\s+eat)\b',
        r'\b(foodie|foodies|food\s+spot|food\s+place)\b',
        r'\b(breakfast\s+spot|brunch\s+place|lunch\s+spot|dinner\s+place)\b',
        # Implicit intent
        r'\b(i\'m\s+hungry|i\'m\s+starving|feeling\s+hungry|getting\s+hungry)\b',
        r'\b(need\s+food|want\s+food|looking\s+for\s+food)\b',
        r'\b(where\s+can\s+we\s+eat|where\s+should\s+we\s+eat)\b',
        # Variations
        r'\b(cheap\s+eat|good\s+eat|best\s+food|tasty|delicious)\b',
    ],
    'tr': [
        # Turkish slang
        r'\b(bir\s+şeyler\s+atıştır|atıştırmak|lokma\s+kap)\b',
        r'\b(acıktım|karnım\s+açtı|midem\s+knurrend)\b',
        r'\b(yemek\s+yeri|bir\s+yer|mekan)\b',
        r'\b(ucuz\s+yemek|lezzetli|enfes)\b',
    ],
    'de': [
        # German slang
        r'\b(was\s+essen|etwas\s+essen|essen\s+gehen)\b',
        r'\b(ich\s+habe\s+hunger|hunger|hungrig)\b',
        r'\b(essensplatz|lokal|gaststätte)\b',
    ],
    'fr': [
        # French slang
        r'\b(casser\s+la\s+croûte|grignoter|bouffe)\b',
        r'\b(j\'ai\s+faim|affamé|crève\s+la\s+dalle)\b',
        r'\b(coin\s+sympa|bon\s+resto|bon\s+plan)\b',
    ],
    'ru': [
        # Russian slang
        r'\b(перекусить|пожрать|жратва)\b',
        r'\b(я\s+голоден|хочу\s+есть|проголодался)\b',
        r'\b(вкусное\s+место|хорошее\s+место)\b',
    ],
    'ar': [
        # Arabic variations
        r'\b(جوعان|أحتاج\s+أكل)\b',
        r'\b(مكان\s+جيد|مكان\s+حلو)\b',
    ]
}

# Enhanced attraction patterns with slang
ATTRACTION_SLANG = {
    'en': [
        # Slang & colloquial
        r'\b(check\s+out|worth\s+seeing|must[\s-]?see)\b',
        r'\b(cool\s+places?|awesome\s+places?|interesting\s+spots?)\b',
        r'\b(sightseeing|sight[\s-]?seeing)\b',
        r'\b(photo\s+ops?|instagram[\s-]?worthy|instagrammable)\b',
        # Tourist questions
        r'\b(what\s+should\s+i\s+see|what\s+to\s+check\s+out)\b',
        r'\b(famous\s+places?|iconic\s+places?|landmark)\b',
    ],
    'tr': [
        r'\b(görmeden\s+dönme|mutlaka\s+gör|kesinlikle\s+gör)\b',
        r'\b(harika\s+yerler|güzel\s+yerler|ilginç\s+yerler)\b',
        r'\b(fotoğraf\s+çek|fotoğraf\s+için)\b',
    ],
    'de': [
        r'\b(sehenswert|lohnt\s+sich|unbedingt\s+sehen)\b',
        r'\b(coole\s+orte|interessante\s+orte)\b',
    ],
    'fr': [
        r'\b(à\s+voir\s+absolument|incontournable|remarquable)\b',
        r'\b(endroits\s+cool|lieux\s+sympas)\b',
    ],
    'ru': [
        r'\b(обязательно\s+посмотреть|стоит\s+посмотреть)\b',
        r'\b(крутые\s+места|интересные\s+места)\b',
    ],
    'ar': [
        r'\b(يجب\s+رؤية|جدير\s+بالمشاهدة)\b',
    ]
}

# Enhanced transportation patterns
TRANSPORTATION_SLANG = {
    'en': [
        r'\b(get\s+there|getting\s+there|make\s+it\s+to)\b',
        r'\b(hop\s+on|catch\s+a|take\s+the)\b',
        r'\b(quickest\s+way|fastest\s+way|easiest\s+way)\b',
    ],
    'tr': [
        r'\b(oraya\s+nasıl|git|varmak)\b',
        r'\b(en\s+hızlı|en\s+kolay)\b',
    ]
}

# Nightlife slang
NIGHTLIFE_SLANG = {
    'en': [
        r'\b(go\s+out|going\s+out|night\s+on\s+the\s+town)\b',
        r'\b(have\s+a\s+drink|grab\s+a\s+drink|drinks?)\b',
        r'\b(party\s+scene|clubbing|bar\s+hopping)\b',
        r'\b(where\'s\s+the\s+action|where\s+to\s+party)\b',
    ],
    'tr': [
        r'\b(eğlenmek|takılmak|gezmek)\b',
        r'\b(içki\s+içmek|rakı\s+balık)\b',
    ]
}

# Shopping slang
SHOPPING_SLANG = {
    'en': [
        r'\b(go\s+shopping|shop\s+around|window\s+shopping)\b',
        r'\b(pick\s+up|buy\s+some|get\s+some)\b',
        r'\b(bargain|deal|cheap\s+stuff)\b',
    ],
    'tr': [
        r'\b(alışverişe\s+çık|pazarla|kelepir)\b',
    ]
}

# Family-friendly slang
FAMILY_SLANG = {
    'en': [
        r'\b(bring\s+the\s+kids|take\s+kids|with\s+the\s+family)\b',
        r'\b(kid[\s-]?safe|child[\s-]?safe|suitable\s+for\s+kids)\b',
        r'\b(fun\s+for\s+all\s+ages|all[\s-]?ages)\b',
    ],
    'tr': [
        r'\b(çocukları\s+getir|ailemle|çocuklar\s+için)\b',
    ]
}

# Context phrases (for disambiguation)
CONTEXT_NEGATIVE = {
    'en': [
        r'\b(not|don\'t|avoid|without|except|no)\b',
        r'\b(anything\s+but|other\s+than)\b',
    ],
    'tr': [
        r'\b(değil|yok|hariç|dışında)\b',
    ]
}

CONTEXT_INTENSIFIERS = {
    'en': [
        r'\b(really|very|super|extremely|absolutely)\b',
        r'\b(best|top|great|amazing|excellent)\b',
    ],
    'tr': [
        r'\b(çok|çok\s+iyi|en\s+iyi|harika|muhteşem)\b',
    ]
}


def get_enhanced_patterns(signal_type: str, language: str = 'en') -> List[str]:
    """
    Get enhanced patterns for a signal type and language.
    
    Args:
        signal_type: Signal type (needs_restaurant, needs_attraction, etc.)
        language: Language code
        
    Returns:
        List of regex patterns
    """
    pattern_map = {
        'needs_restaurant': RESTAURANT_SLANG,
        'needs_attraction': ATTRACTION_SLANG,
        'needs_transportation': TRANSPORTATION_SLANG,
        'needs_nightlife': NIGHTLIFE_SLANG,
        'needs_shopping': SHOPPING_SLANG,
        'needs_family_friendly': FAMILY_SLANG,
    }
    
    patterns_dict = pattern_map.get(signal_type, {})
    return patterns_dict.get(language, [])


def get_fuzzy_keywords(signal_type: str, language: str = 'en') -> List[str]:
    """
    Get keywords for fuzzy matching (common misspellings).
    
    Args:
        signal_type: Signal type
        language: Language code
        
    Returns:
        List of keywords for fuzzy matching
    """
    fuzzy_map = {
        'needs_restaurant': {
            'en': ['restaurant', 'cafe', 'food', 'eat', 'dining', 'lunch', 'dinner', 
                   'breakfast', 'brunch', 'hungry', 'meal'],
            'tr': ['restoran', 'kafe', 'yemek', 'lokanta', 'acıktım'],
            'de': ['restaurant', 'essen', 'hunger'],
            'fr': ['restaurant', 'manger', 'faim'],
            'ru': ['ресторан', 'еда', 'голоден'],
            'ar': ['مطعم', 'طعام', 'جائع'],
        },
        'needs_attraction': {
            'en': ['museum', 'attraction', 'palace', 'mosque', 'church', 'tower', 
                   'sight', 'landmark', 'monument', 'visit'],
            'tr': ['müze', 'saray', 'cami', 'gezilecek', 'anıt'],
            'de': ['museum', 'sehenswürdigkeit', 'palast'],
            'fr': ['musée', 'palais', 'attraction'],
            'ru': ['музей', 'достопримечательность', 'дворец'],
            'ar': ['متحف', 'معالم', 'قصر'],
        },
        'needs_shopping': {
            'en': ['shop', 'shopping', 'mall', 'market', 'store', 'buy'],
            'tr': ['alışveriş', 'mağaza', 'market', 'çarşı'],
        },
        'needs_nightlife': {
            'en': ['bar', 'club', 'nightlife', 'party', 'drink'],
            'tr': ['bar', 'kulüp', 'gece hayatı', 'eğlence'],
        },
    }
    
    keywords_dict = fuzzy_map.get(signal_type, {})
    return keywords_dict.get(language, [])
