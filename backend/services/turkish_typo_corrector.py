"""
Turkish Typo Corrector
Corrects Turkish-specific typos and keyboard errors
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)


class TurkishTypoCorrector:
    """Correct Turkish-specific typos and keyboard errors"""
    
    def __init__(self):
        self.turkish_chars = self._load_turkish_chars()
        self.tourism_vocabulary = self._load_tourism_vocabulary()
        self.common_misspellings = self._load_common_misspellings()
        self.keyboard_layout = self._load_turkish_keyboard_layout()
        logger.info("✅ TurkishTypoCorrector initialized")
    
    def _load_turkish_chars(self) -> Set[str]:
        """Load Turkish-specific characters"""
        return {'ç', 'ğ', 'ı', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'İ', 'Ö', 'Ş', 'Ü'}
    
    def _load_tourism_vocabulary(self) -> Dict[str, str]:
        """Load tourism-specific vocabulary with correct spellings"""
        return {
            # Istanbul landmarks - Common misspellings
            'ayasofya': 'Ayasofya',
            'aya sofya': 'Ayasofya',
            'ayasofia': 'Ayasofya',
            'hagia sophia': 'Ayasofya',
            'sultanahmet': 'Sultanahmet',
            'sultan ahmet': 'Sultanahmet',
            'sultanahmed': 'Sultanahmet',
            'topkapı': 'Topkapı',
            'topkapi': 'Topkapı',
            'topkapi': 'Topkapı',
            'galata': 'Galata',
            'kapalıçarşı': 'Kapalıçarşı',
            'kapali carsi': 'Kapalıçarşı',
            'kapalicarsi': 'Kapalıçarşı',
            'grand bazaar': 'Kapalıçarşı',
            'taksim': 'Taksim',
            'taqsim': 'Taksim',
            'kadıköy': 'Kadıköy',
            'kadikoy': 'Kadıköy',
            'kadiköy': 'Kadıköy',
            'beyoğlu': 'Beyoğlu',
            'beyoglu': 'Beyoğlu',
            'üsküdar': 'Üsküdar',
            'uskudar': 'Üsküdar',
            'beşiktaş': 'Beşiktaş',
            'besiktas': 'Beşiktaş',
            'ortaköy': 'Ortaköy',
            'ortakoy': 'Ortaköy',
            'dolmabahçe': 'Dolmabahçe',
            'dolmabahce': 'Dolmabahçe',
            
            # Common Turkish words
            'restorant': 'restoran',
            'restaurant': 'restoran',
            'otel': 'otel',
            'hotel': 'otel',
            'müze': 'müze',
            'muze': 'müze',
            'museum': 'müze',
            'cami': 'cami',
            'camii': 'cami',
            'mosque': 'cami',
            'çarşı': 'çarşı',
            'carsi': 'çarşı',
            'bazaar': 'çarşı',
            
            # Food and cuisine
            'kebap': 'kebap',
            'kebab': 'kebap',
            'balık': 'balık',
            'balik': 'balık',
            'fish': 'balık',
            'kahve': 'kahve',
            'coffee': 'kahve',
            'çay': 'çay',
            'cay': 'çay',
            'tea': 'çay',
        }
    
    def _load_common_misspellings(self) -> Dict[str, str]:
        """Load common misspelling patterns"""
        return {
            # Turkish character substitutions (common when using English keyboard)
            'i': 'ı',  # dotless i
            'ı': 'i',
            'o': 'ö',
            'ö': 'o',
            'u': 'ü',
            'ü': 'u',
            'c': 'ç',
            'ç': 'c',
            's': 'ş',
            'ş': 's',
            'g': 'ğ',
            'ğ': 'g',
        }
    
    def _load_turkish_keyboard_layout(self) -> Dict[str, List[str]]:
        """
        Load Turkish Q keyboard layout for proximity detection
        Keyboard layout (Turkish Q):
        Q W E R T Y U I O P Ğ Ü
        A S D F G H J K L Ş İ
        Z X C V B N M Ö Ç
        """
        return {
            'q': ['w', 'a'],
            'w': ['q', 'e', 's', 'a'],
            'e': ['w', 'r', 'd', 's'],
            'r': ['e', 't', 'f', 'd'],
            't': ['r', 'y', 'g', 'f'],
            'y': ['t', 'u', 'h', 'g'],
            'u': ['y', 'ı', 'j', 'h'],
            'ı': ['u', 'o', 'k', 'j'],
            'o': ['ı', 'p', 'l', 'k'],
            'p': ['o', 'ğ', 'ş', 'l'],
            'ğ': ['p', 'ü', 'i', 'ş'],
            'ü': ['ğ', 'i'],
            'a': ['q', 'w', 's', 'z'],
            's': ['a', 'w', 'e', 'd', 'z', 'x'],
            'd': ['s', 'e', 'r', 'f', 'x', 'c'],
            'f': ['d', 'r', 't', 'g', 'c', 'v'],
            'g': ['f', 't', 'y', 'h', 'v', 'b'],
            'h': ['g', 'y', 'u', 'j', 'b', 'n'],
            'j': ['h', 'u', 'ı', 'k', 'n', 'm'],
            'k': ['j', 'ı', 'o', 'l', 'm', 'ö'],
            'l': ['k', 'o', 'p', 'ş', 'ö', 'ç'],
            'ş': ['l', 'p', 'ğ', 'i', 'ç'],
            'i': ['ş', 'ğ', 'ü'],
            'z': ['a', 's', 'x'],
            'x': ['z', 's', 'd', 'c'],
            'c': ['x', 'd', 'f', 'v'],
            'v': ['c', 'f', 'g', 'b'],
            'b': ['v', 'g', 'h', 'n'],
            'n': ['b', 'h', 'j', 'm'],
            'm': ['n', 'j', 'k', 'ö'],
            'ö': ['m', 'k', 'l', 'ç'],
            'ç': ['ö', 'l', 'ş'],
        }
    
    def correct_typos(self, query: str, aggressive: bool = False) -> Tuple[str, List[str]]:
        """
        Correct typos in query with Turkish-aware algorithms
        
        Args:
            query: Query to correct
            aggressive: If True, applies more corrections (may over-correct)
            
        Returns:
            Tuple of (corrected_query, list_of_corrections)
        """
        words = query.split()
        corrected_words = []
        corrections = []
        
        for word in words:
            # Skip very short words
            if len(word) <= 2:
                corrected_words.append(word)
                continue
            
            original_word = word
            
            # Try correction strategies in order
            corrected = (
                self._try_tourism_vocabulary(word) or
                self._try_turkish_char_substitution(word) or
                (self._try_keyboard_proximity(word) if aggressive else None) or
                word  # Keep original if no correction found
            )
            
            if corrected != original_word:
                corrections.append(f"{original_word} → {corrected}")
            
            corrected_words.append(corrected)
        
        corrected_query = ' '.join(corrected_words)
        
        if corrections:
            logger.info(f"🔧 Corrected typos: '{query}' → '{corrected_query}'")
            logger.info(f"   Applied {len(corrections)} corrections")
        
        return corrected_query, corrections
    
    def _try_tourism_vocabulary(self, word: str) -> Optional[str]:
        """
        Try to correct using tourism vocabulary
        
        Args:
            word: Word to check
            
        Returns:
            Corrected word or None
        """
        word_lower = word.lower()
        
        # Direct match
        if word_lower in self.tourism_vocabulary:
            return self.tourism_vocabulary[word_lower]
        
        # Try removing Turkish suffixes and check again
        # Common Turkish suffixes: 'de, 'da, 'te, 'ta, 'den, 'dan, 'ten, 'tan, etc.
        suffixes_to_strip = ["'de", "'da", "'te", "'ta", "'den", "'dan", "'ten", "'tan",
                             "'nde", "'nda", "'nte", "'nta", "'nden", "'ndan", "'nten", "'ntan",
                             "'e", "'a", "'i", "'ı", "'ü", "'u", "'ö", "'o",
                             "'in", "'ın", "'ün", "'un", "'nin", "'nın", "'nün", "'nun"]
        
        for suffix in suffixes_to_strip:
            if word_lower.endswith(suffix):
                base_word = word_lower[:-len(suffix)]
                if base_word in self.tourism_vocabulary:
                    # Return corrected word with original suffix
                    # For place names, always use the vocabulary's capitalization (proper nouns)
                    corrected_base = self.tourism_vocabulary[base_word]
                    original_suffix = word[len(base_word):]
                    return corrected_base + original_suffix
        
        return None
    
    def _try_turkish_char_substitution(self, word: str) -> Optional[str]:
        """
        Fix common Turkish character mistakes
        
        Examples:
        - "Ayasofia" -> "Ayasofya" (i -> y)
        - "Sultanahment" -> "Sultanahmet" (n -> -)
        - "cami" -> "cami" (correct)
        
        Args:
            word: Word to correct
            
        Returns:
            Corrected word or None
        """
        word_lower = word.lower()
        
        # Try each Turkish character substitution
        for original, replacement in self._load_common_misspellings().items():
            if original in word_lower:
                candidate = word_lower.replace(original, replacement)
                
                # Check if candidate is in tourism vocabulary
                if candidate in self.tourism_vocabulary:
                    # Preserve original capitalization pattern
                    return self._preserve_case(word, self.tourism_vocabulary[candidate])
        
        return None
    
    def _try_keyboard_proximity(self, word: str) -> Optional[str]:
        """
        Fix typos based on Turkish Q keyboard layout
        
        Args:
            word: Word to correct
            
        Returns:
            Corrected word or None
        """
        word_lower = word.lower()
        
        # Generate candidates by replacing each character with nearby keys
        for i, char in enumerate(word_lower):
            if char in self.keyboard_layout:
                for nearby_key in self.keyboard_layout[char]:
                    candidate = word_lower[:i] + nearby_key + word_lower[i+1:]
                    
                    # Check if candidate is valid
                    if candidate in self.tourism_vocabulary:
                        return self._preserve_case(word, self.tourism_vocabulary[candidate])
        
        return None
    
    def _preserve_case(self, original: str, corrected: str) -> str:
        """
        Preserve the case pattern of the original word
        
        Args:
            original: Original word with case
            corrected: Corrected word (may have different case)
            
        Returns:
            Corrected word with original case pattern
        """
        if original.isupper():
            return corrected.upper()
        elif original[0].isupper():
            return corrected.capitalize()
        else:
            return corrected.lower()
    
    def correct_silent(self, query: str) -> str:
        """
        Correct query without returning correction details
        
        Args:
            query: Query to correct
            
        Returns:
            Corrected query
        """
        corrected, _ = self.correct_typos(query, aggressive=False)
        return corrected
    
    def has_typos(self, query: str) -> bool:
        """
        Check if query likely contains typos
        
        Args:
            query: Query to check
            
        Returns:
            True if typos detected
        """
        _, corrections = self.correct_typos(query, aggressive=False)
        return len(corrections) > 0
    
    def get_correction_statistics(self, queries: List[str]) -> Dict:
        """
        Get statistics about typo corrections in queries
        
        Args:
            queries: List of queries to analyze
            
        Returns:
            Dictionary with correction statistics
        """
        total = len(queries)
        with_typos = 0
        total_corrections = 0
        correction_types = {
            'tourism_vocabulary': 0,
            'char_substitution': 0,
            'keyboard_proximity': 0
        }
        
        for query in queries:
            _, corrections = self.correct_typos(query, aggressive=False)
            if corrections:
                with_typos += 1
                total_corrections += len(corrections)
        
        return {
            'total_queries': total,
            'queries_with_typos': with_typos,
            'typo_percentage': (with_typos / total * 100) if total > 0 else 0,
            'total_corrections': total_corrections,
            'avg_corrections_per_query': (total_corrections / with_typos) if with_typos > 0 else 0
        }


# Singleton instance
_corrector_instance = None

def get_typo_corrector() -> TurkishTypoCorrector:
    """Get or create typo corrector singleton"""
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = TurkishTypoCorrector()
    return _corrector_instance
