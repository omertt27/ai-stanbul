"""
fuzzy_matcher.py - Fuzzy String Matching for Signal Detection

Handles misspellings, typos, and phonetic variations in user queries.

Features:
- Levenshtein distance matching (character-level)
- Phonetic matching (sound-alike words)
- Multi-language support
- Performance optimized with caching

Author: AI Istanbul Team
Date: January 2025
"""

import logging
from typing import List, Dict, Set, Optional, Tuple
from functools import lru_cache
import re

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        RAPIDFUZZ_AVAILABLE = False
    except ImportError:
        RAPIDFUZZ_AVAILABLE = False
        logging.warning("⚠️ No fuzzy matching library available. Install rapidfuzz or fuzzywuzzy.")

try:
    import jellyfish
    PHONETIC_AVAILABLE = True
except ImportError:
    PHONETIC_AVAILABLE = False
    logging.warning("⚠️ Phonetic matching not available. Install jellyfish for better matching.")

logger = logging.getLogger(__name__)


class FuzzyMatcher:
    """
    Fuzzy string matching for signal detection.
    
    Supports:
    - Character-level fuzzy matching (typos)
    - Phonetic matching (sound-alike)
    - Multi-language support
    """
    
    def __init__(
        self,
        fuzzy_threshold: int = 85,
        phonetic_threshold: int = 80,
        enable_phonetic: bool = True,
        enable_cache: bool = True
    ):
        """
        Initialize fuzzy matcher.
        
        Args:
            fuzzy_threshold: Minimum similarity score (0-100) for fuzzy match
            phonetic_threshold: Minimum similarity score for phonetic match
            enable_phonetic: Enable phonetic matching
            enable_cache: Enable caching for performance
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.phonetic_threshold = phonetic_threshold
        self.enable_phonetic = enable_phonetic and PHONETIC_AVAILABLE
        self.enable_cache = enable_cache
        
        if not RAPIDFUZZ_AVAILABLE:
            logger.info("ℹ️  Fuzzy matching library not available - using exact matching")
        
        logger.info(f"✅ FuzzyMatcher initialized (threshold={fuzzy_threshold}, phonetic={enable_phonetic})")
    
    def extract_words(self, text: str) -> List[str]:
        """Extract words from text, removing punctuation."""
        # Remove punctuation but keep letters (including non-ASCII)
        words = re.findall(r'\b[\w]+\b', text.lower(), re.UNICODE)
        return [w for w in words if len(w) > 2]  # Skip very short words
    
    @lru_cache(maxsize=1000)
    def fuzzy_match_word(self, word: str, target: str) -> int:
        """
        Calculate fuzzy match score between two words.
        
        Args:
            word: Input word (possibly misspelled)
            target: Target keyword to match against
            
        Returns:
            Similarity score (0-100)
        """
        if not RAPIDFUZZ_AVAILABLE:
            # Fallback: simple character overlap
            return int(100 * len(set(word) & set(target)) / max(len(set(word)), len(set(target))))
        
        # Use partial ratio for substring matching
        score = fuzz.ratio(word, target)
        partial_score = fuzz.partial_ratio(word, target)
        
        return max(score, partial_score)
    
    @lru_cache(maxsize=1000)
    def phonetic_match_word(self, word: str, target: str) -> int:
        """
        Calculate phonetic match score (for sound-alike words).
        
        Args:
            word: Input word
            target: Target keyword
            
        Returns:
            Phonetic similarity score (0-100)
        """
        if not self.enable_phonetic:
            return 0
        
        try:
            # Metaphone for English-like phonetic matching
            word_ph = jellyfish.metaphone(word)
            target_ph = jellyfish.metaphone(target)
            
            if word_ph == target_ph:
                return 100
            
            # Calculate similarity between phonetic codes
            return fuzz.ratio(word_ph, target_ph) if RAPIDFUZZ_AVAILABLE else 0
        except Exception as e:
            logger.debug(f"Phonetic matching error: {e}")
            return 0
    
    def match_keywords(
        self,
        query: str,
        keywords: List[str],
        use_phonetic: bool = True
    ) -> Tuple[bool, float, List[str]]:
        """
        Check if query contains any keywords (with fuzzy/phonetic matching).
        
        Args:
            query: User query
            keywords: List of keywords to match
            use_phonetic: Enable phonetic matching
            
        Returns:
            Tuple of (matched, confidence, matched_terms)
        """
        if not keywords:
            return False, 0.0, []
        
        query_words = self.extract_words(query)
        if not query_words:
            return False, 0.0, []
        
        matches = []
        scores = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # 1. Exact match (highest priority)
            if keyword_lower in query.lower():
                matches.append(keyword)
                scores.append(100)
                continue
            
            # 2. Fuzzy match (typos)
            for word in query_words:
                fuzzy_score = self.fuzzy_match_word(word, keyword_lower)
                
                if fuzzy_score >= self.fuzzy_threshold:
                    matches.append(f"{keyword}~{word}")
                    scores.append(fuzzy_score)
                    break
                
                # 3. Phonetic match (sound-alike)
                if use_phonetic and self.enable_phonetic:
                    phonetic_score = self.phonetic_match_word(word, keyword_lower)
                    
                    if phonetic_score >= self.phonetic_threshold:
                        matches.append(f"{keyword}≈{word}")
                        scores.append(phonetic_score * 0.9)  # Slightly lower confidence
                        break
        
        if matches:
            # Average confidence from all matches
            confidence = sum(scores) / len(scores) / 100.0  # Normalize to 0-1
            return True, confidence, matches
        
        return False, 0.0, []
    
    def match_patterns_fuzzy(
        self,
        query: str,
        patterns: List[str],
        fuzzy_keywords: List[str]
    ) -> Tuple[bool, float, List[str]]:
        """
        Match query against regex patterns + fuzzy keywords.
        
        Args:
            query: User query
            patterns: List of regex patterns
            fuzzy_keywords: Keywords to match with fuzzy logic
            
        Returns:
            Tuple of (matched, confidence, details)
        """
        # 1. Try regex patterns first (exact match)
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE | re.UNICODE)
            if match:
                return True, 1.0, [f"regex:{match.group(0)}"]
        
        # 2. Fall back to fuzzy matching
        return self.match_keywords(query, fuzzy_keywords)


# Global fuzzy matcher instance
_fuzzy_matcher = None


def get_fuzzy_matcher(
    fuzzy_threshold: int = 85,
    phonetic_threshold: int = 80,
    enable_phonetic: bool = True
) -> FuzzyMatcher:
    """Get or create global fuzzy matcher instance."""
    global _fuzzy_matcher
    
    if _fuzzy_matcher is None:
        _fuzzy_matcher = FuzzyMatcher(
            fuzzy_threshold=fuzzy_threshold,
            phonetic_threshold=phonetic_threshold,
            enable_phonetic=enable_phonetic
        )
    
    return _fuzzy_matcher


# Common misspellings and variations for quick reference
COMMON_MISSPELLINGS = {
    'restaurant': ['restarant', 'resturant', 'restraunt', 'restaurent', 'restorant'],
    'nearby': ['nearbye', 'nearbuy', 'near-by', 'neerby'],
    'museum': ['musem', 'musium', 'museam', 'musuem'],
    'attraction': ['atraction', 'attracton', 'atracttion'],
    'accommodation': ['accomodation', 'acommodation', 'acomodation'],
}
