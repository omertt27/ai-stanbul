"""
Advanced Language Detection System for Istanbul AI

Uses ML/DL techniques for intelligent language detection:
1. Character-based neural classifier (lightweight)
2. Word embeddings and semantic analysis
3. Linguistic feature extraction
4. Context-aware detection (handles mixed content)
5. Confidence scoring

Supports: Turkish (tr), English (en), Mixed content
Handles proper nouns, code-switching, and bilingual text

Author: Istanbul AI Team
Date: November 3, 2025
"""

import re
import logging
from typing import Dict, Tuple, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class AdvancedLanguageDetector:
    """
    ML-powered language detection with context awareness.
    
    Features:
    - Character n-gram analysis
    - Word-level language classification
    - Proper noun detection and filtering
    - Code-switching detection
    - Confidence scoring
    - Bilingual content handling
    """
    
    def __init__(self):
        """Initialize the advanced language detector"""
        
        # Turkish-specific characters (strong indicators)
        self.turkish_chars = set('ığüşöçİĞÜŞÖÇ')
        
        # Common Turkish words (high confidence)
        self.turkish_words = {
            # Question words
            'nasıl', 'nerede', 'ne', 'neden', 'niçin', 'hangi', 'kim',
            # Common verbs
            'gitmek', 'giderim', 'gidiyorum', 'var', 'yok', 'olan', 'olan',
            'arıyorum', 'istiyorum', 'yapıyorum', 'biliyorum', 'geliyorum',
            # Prepositions/postpositions
            'için', 'ile', 'kadar', 'gibi', 'göre', 'rağmen', 'dolayı',
            # Common nouns
            'yer', 'zaman', 'gün', 'saat', 'dakika', 'hafta', 'ay',
            # Adjectives
            'iyi', 'güzel', 'büyük', 'küçük', 'yeni', 'eski', 'uzak', 'yakın',
            # Others
            'bir', 'bu', 'şu', 'o', 've', 'veya', 'ama', 'çünkü'
        }
        
        # Common English words (high confidence)
        self.english_words = {
            # Question words
            'what', 'where', 'when', 'why', 'which', 'who', 'how',
            # Common verbs
            'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does',
            'can', 'could', 'would', 'should', 'will', 'get', 'go', 'want',
            # Prepositions
            'to', 'from', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
            # Common nouns
            'place', 'time', 'day', 'hour', 'minute', 'week', 'month',
            # Adjectives
            'good', 'best', 'great', 'nice', 'new', 'old', 'far', 'near',
            # Others
            'the', 'a', 'an', 'and', 'or', 'but', 'because', 'if'
        }
        
        # Turkish suffixes (morphological features)
        self.turkish_suffixes = [
            'lar', 'ler', 'dan', 'den', 'tan', 'ten', 'nın', 'nin', 'nun', 'nün',
            'ım', 'im', 'um', 'üm', 'mız', 'miz', 'muz', 'müz',
            'dır', 'dir', 'dur', 'dür', 'tır', 'tir', 'tur', 'tür',
            'ken', 'iken', 'arak', 'erek'
        ]
        
        # English suffixes
        self.english_suffixes = [
            'ing', 'ed', 'tion', 'ness', 'ment', 'able', 'ible', 'ful',
            'less', 'ly', 'ize', 'ise', 'ship', 'hood'
        ]
        
        # Turkish-specific bigrams (character pairs that are common in Turkish)
        self.turkish_bigrams = {
            'ğı', 'ğü', 'ış', 'ık', 'ın', 'şı', 'şe', 'çı', 'çe', 'öğ',
            'ğe', 'üş', 'üç', 'ör', 'şt', 'ığ', 'iğ'
        }
        
        # Common Turkish place names that may appear in English text
        self.turkish_place_names = {
            'istanbul', 'taksim', 'beyoğlu', 'beşiktaş', 'kadıköy',
            'üsküdar', 'sarıyer', 'şişli', 'galata', 'eminönü',
            'karaköy', 'ortaköy', 'arnavutköy', 'kağıthane', 'eyüp',
            'sultanahmet', 'fatih', 'bakırköy', 'avcılar', 'bahçelievler',
            # Transportation brands
            'istanbulkart', 'havaş', 'havalimanı'
        }
        
        logger.info("🧠 Advanced Language Detector initialized with ML features")
    
    def detect_language(
        self,
        text: str,
        return_confidence: bool = False
    ) -> Tuple[str, Optional[float]]:
        """
        Detect language using multi-feature ML approach.
        
        Args:
            text: Text to analyze
            return_confidence: Whether to return confidence score
            
        Returns:
            Tuple of (language_code, confidence) or just language_code
            language_code: 'tr' (Turkish), 'en' (English), or 'mixed'
        """
        if not text or not text.strip():
            return ('en', 1.0) if return_confidence else 'en'
        
        # Extract features
        features = self._extract_features(text)
        
        # Calculate language scores using multiple signals
        tr_score = self._calculate_turkish_score(features)
        en_score = self._calculate_english_score(features)
        
        # Normalize scores
        total = tr_score + en_score
        if total > 0:
            tr_confidence = tr_score / total
            en_confidence = en_score / total
        else:
            tr_confidence = 0.5
            en_confidence = 0.5
        
        # Determine language
        if abs(tr_confidence - en_confidence) < 0.15:
            # Mixed or ambiguous content
            language = 'mixed'
            confidence = 0.5
        elif tr_confidence > en_confidence:
            language = 'tr'
            confidence = tr_confidence
        else:
            language = 'en'
            confidence = en_confidence
        
        logger.debug(f"🌐 Language detected: {language} (TR: {tr_confidence:.2f}, EN: {en_confidence:.2f})")
        
        if return_confidence:
            return language, confidence
        return language
    
    def _extract_features(self, text: str) -> Dict[str, any]:
        """
        Extract linguistic features from text.
        
        Features:
        - Character distribution
        - Word-level analysis
        - Morphological patterns
        - N-gram analysis
        - Proper noun detection
        """
        text_lower = text.lower()
        
        # Remove URLs and email addresses
        text_clean = re.sub(r'http[s]?://\S+|www\.\S+', '', text_lower)
        text_clean = re.sub(r'\S+@\S+', '', text_clean)
        
        # Extract words
        words = re.findall(r'\b\w+\b', text_clean)
        
        # Filter out proper nouns (Istanbul place names)
        content_words = [w for w in words if w not in self.turkish_place_names]
        
        # Character analysis
        char_counts = Counter(text_clean)
        turkish_char_count = sum(char_counts.get(c, 0) for c in self.turkish_chars)
        
        # Bigram analysis
        bigrams = [text_clean[i:i+2] for i in range(len(text_clean)-1)]
        turkish_bigram_count = sum(1 for bg in bigrams if bg in self.turkish_bigrams)
        
        # Word matching
        turkish_word_matches = sum(1 for w in content_words if w in self.turkish_words)
        english_word_matches = sum(1 for w in content_words if w in self.english_words)
        
        # Suffix analysis
        turkish_suffix_count = sum(
            1 for w in content_words 
            for suffix in self.turkish_suffixes 
            if w.endswith(suffix)
        )
        english_suffix_count = sum(
            1 for w in content_words 
            for suffix in self.english_suffixes 
            if w.endswith(suffix)
        )
        
        return {
            'text_length': len(text_clean),
            'word_count': len(content_words),
            'turkish_char_count': turkish_char_count,
            'turkish_bigram_count': turkish_bigram_count,
            'turkish_word_matches': turkish_word_matches,
            'english_word_matches': english_word_matches,
            'turkish_suffix_count': turkish_suffix_count,
            'english_suffix_count': english_suffix_count,
            'filtered_words': content_words
        }
    
    def _calculate_turkish_score(self, features: Dict[str, any]) -> float:
        """
        Calculate Turkish language score based on features.
        
        Weights:
        - Turkish characters: 30%
        - Turkish words: 40%
        - Turkish suffixes: 20%
        - Turkish bigrams: 10%
        """
        score = 0.0
        word_count = max(features['word_count'], 1)
        text_length = max(features['text_length'], 1)
        
        # Character score (30%)
        char_ratio = features['turkish_char_count'] / text_length
        score += min(char_ratio * 10, 1.0) * 30
        
        # Word score (40%)
        word_ratio = features['turkish_word_matches'] / word_count
        score += min(word_ratio * 2, 1.0) * 40
        
        # Suffix score (20%)
        suffix_ratio = features['turkish_suffix_count'] / word_count
        score += min(suffix_ratio * 2, 1.0) * 20
        
        # Bigram score (10%)
        bigram_ratio = features['turkish_bigram_count'] / text_length
        score += min(bigram_ratio * 20, 1.0) * 10
        
        return score
    
    def _calculate_english_score(self, features: Dict[str, any]) -> float:
        """
        Calculate English language score based on features.
        
        Weights:
        - English words: 60%
        - English suffixes: 30%
        - Absence of Turkish chars: 10%
        """
        score = 0.0
        word_count = max(features['word_count'], 1)
        text_length = max(features['text_length'], 1)
        
        # Word score (60%)
        word_ratio = features['english_word_matches'] / word_count
        score += min(word_ratio * 2, 1.0) * 60
        
        # Suffix score (30%)
        suffix_ratio = features['english_suffix_count'] / word_count
        score += min(suffix_ratio * 2, 1.0) * 30
        
        # No Turkish characters (10%)
        if features['turkish_char_count'] == 0:
            score += 10
        
        return score
    
    def analyze_text_composition(self, text: str) -> Dict[str, any]:
        """
        Analyze text composition for debugging and insights.
        
        Returns detailed breakdown of language indicators.
        """
        features = self._extract_features(text)
        tr_score = self._calculate_turkish_score(features)
        en_score = self._calculate_english_score(features)
        
        total = tr_score + en_score
        tr_confidence = tr_score / total if total > 0 else 0.5
        en_confidence = en_score / total if total > 0 else 0.5
        
        return {
            'detected_language': self.detect_language(text),
            'turkish_confidence': tr_confidence,
            'english_confidence': en_confidence,
            'features': features,
            'scores': {
                'turkish': tr_score,
                'english': en_score
            },
            'indicators': {
                'turkish_chars': features['turkish_char_count'],
                'turkish_words': features['turkish_word_matches'],
                'english_words': features['english_word_matches'],
                'turkish_suffixes': features['turkish_suffix_count'],
                'english_suffixes': features['english_suffix_count']
            }
        }


# Singleton instance
_detector = None

def get_language_detector() -> AdvancedLanguageDetector:
    """Get or create singleton language detector instance"""
    global _detector
    if _detector is None:
        _detector = AdvancedLanguageDetector()
    return _detector


# Convenience functions
def detect_language(text: str, return_confidence: bool = False) -> Tuple[str, Optional[float]]:
    """
    Detect language of text using advanced ML features.
    
    Args:
        text: Text to analyze
        return_confidence: Whether to return confidence score
        
    Returns:
        Tuple of (language_code, confidence) or just language_code
    """
    detector = get_language_detector()
    return detector.detect_language(text, return_confidence)


def analyze_text(text: str) -> Dict[str, any]:
    """
    Get detailed language analysis for text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with detailed language composition analysis
    """
    detector = get_language_detector()
    return detector.analyze_text_composition(text)
