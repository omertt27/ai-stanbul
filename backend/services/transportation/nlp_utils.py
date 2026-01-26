"""
NLP Utilities for Transportation RAG
====================================

Text processing utilities including:
- Levenshtein distance for fuzzy matching
- Unicode normalization
- Cyrillic-to-Latin transliteration
- Turkish diacritics removal
- Turkish morphology handling

Author: AI Istanbul Team
Date: December 2024
"""

import re
import unicodedata
import logging
from typing import List

logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.
    
    This is the minimum number of single-character edits (insertions,
    deletions, or substitutions) required to change one string into another.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized similarity based on Levenshtein distance.
    
    Returns a value between 0 (completely different) and 1 (identical).
    """
    if not s1 or not s2:
        return 0.0
    
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1 - (distance / max_len)


def normalize_unicode_text(text: str) -> str:
    """
    Normalize Unicode text for consistent matching.
    """
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    return text.strip().lower()


def transliterate_cyrillic_to_latin(text: str) -> str:
    """
    Transliterate Cyrillic text to Latin characters.
    """
    cyrillic_to_latin = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
        'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
        'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
        'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
        'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
    }
    return ''.join(cyrillic_to_latin.get(c, c) for c in text)


def remove_turkish_diacritics(text: str) -> str:
    """
    Remove Turkish-specific diacritics for ASCII-compatible matching.
    ç→c, ğ→g, ı→i, İ→I, ö→o, ş→s, ü→u
    """
    turkish_map = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U',
    }
    return ''.join(turkish_map.get(c, c) for c in text)


class TurkishMorphologyHandler:
    """
    Handles Turkish agglutinative morphology for location name extraction.
    
    Turkish adds suffixes for grammatical cases:
    - Ablative: -dan/-den (from)
    - Dative: -a/-e (to)  
    - Locative: -da/-de (at/in)
    - Genitive: -ın/-in (of)
    """
    
    ABLATIVE_SUFFIXES = [
        'ından', 'inden', 'undan', 'ünden',
        'ndan', 'nden', 'dan', 'den', 'tan', 'ten',
    ]
    
    DATIVE_SUFFIXES = [
        'ına', 'ine', 'una', 'üne',
        'sına', 'sine', 'suna', 'süne',
        'na', 'ne', 'ya', 'ye', 'a', 'e',
    ]
    
    LOCATIVE_SUFFIXES = [
        'ında', 'inde', 'unda', 'ünde',
        'nda', 'nde', 'da', 'de', 'ta', 'te',
    ]
    
    GENITIVE_SUFFIXES = [
        'nın', 'nin', 'nun', 'nün',
        'ın', 'in', 'un', 'ün',
    ]
    
    POSSESSIVE_SUFFIXES = [
        'ım', 'im', 'um', 'üm',
        'sı', 'si', 'su', 'sü',
    ]
    
    PROTECTED_WORDS = {
        'kadıköy', 'kadikoy', 'kartalden', 'levent',
        'eminönü', 'eminonu', 'üsküdar', 'uskudar',
    }
    
    @classmethod
    def strip_suffixes(cls, word: str) -> str:
        """Strip Turkish grammatical suffixes to get the root location name."""
        if not word or len(word) < 3:
            return word
        
        original = word
        
        # Handle apostrophe usage
        for apostrophe in ["'", "'"]:
            if apostrophe in word:
                root = word.split(apostrophe)[0]
                if len(root) >= 3:
                    return root
        
        word_lower = word.lower()
        
        if word_lower in cls.PROTECTED_WORDS:
            return word
        
        all_suffixes = (
            cls.ABLATIVE_SUFFIXES + 
            cls.DATIVE_SUFFIXES + 
            cls.LOCATIVE_SUFFIXES +
            cls.GENITIVE_SUFFIXES +
            cls.POSSESSIVE_SUFFIXES
        )
        
        all_suffixes = sorted(set(all_suffixes), key=len, reverse=True)
        
        for suffix in all_suffixes:
            if word.lower().endswith(suffix) and len(word) > len(suffix) + 2:
                stripped = word[:-len(suffix)]
                if len(stripped) >= 3:
                    logger.debug(f"🇹🇷 Turkish suffix stripped: '{original}' → '{stripped}'")
                    return stripped
        
        return word
    
    @classmethod
    def generate_suffix_variants(cls, location: str) -> List[str]:
        """Generate common Turkish suffix variants for a location name."""
        variants = [location]
        location_lower = location.lower()
        
        back_vowels = set('aıou')
        front_vowels = set('eiöü')
        
        last_vowel = None
        for c in reversed(location_lower):
            if c in back_vowels | front_vowels:
                last_vowel = c
                break
        
        is_back = last_vowel in back_vowels if last_vowel else True
        
        if is_back:
            variants.extend([
                f"{location}'dan", f"{location}'a", f"{location}'da",
                f"{location}dan", f"{location}a", f"{location}da",
            ])
        else:
            variants.extend([
                f"{location}'den", f"{location}'e", f"{location}'de",
                f"{location}den", f"{location}e", f"{location}de",
            ])
        
        return variants


# Smart default locations
TOURIST_DEFAULT_ORIGINS = [
    "sultanahmet",
    "taksim",
    "galata",
    "kadıköy",
]

AIRPORT_DESTINATIONS = {
    "istanbul havalimanı": ["taksim", "sultanahmet", "şişli"],
    "istanbul havalimani": ["taksim", "sultanahmet", "şişli"],
    "istanbul airport": ["taksim", "sultanahmet", "şişli"],
    "sabiha gökçen": ["kadıköy", "üsküdar", "taksim"],
    "sabiha gokcen": ["kadıköy", "üsküdar", "taksim"],
}


def get_time_based_suggestion(hour: int) -> str:
    """Get contextual suggestion based on time of day."""
    if 6 <= hour < 10:
        return "Morning rush - consider metro over bus"
    elif 16 <= hour < 20:
        return "Evening rush - ferries may be less crowded"
    elif 22 <= hour or hour < 6:
        return "Late night - limited transit, consider taxi/Uber"
    return ""
