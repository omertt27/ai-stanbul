"""
Signal Detection Module
Multi-intent signal detection using semantic embeddings

Responsibilities:
- Semantic embedding model management
- Signal pattern embeddings
- Multi-intent signal detection
- Language-specific thresholds
- Signal caching

Extracted from: pure_llm_handler.py (_init_signal_embeddings, _init_language_thresholds, _detect_language)

Author: Istanbul AI Team
Date: November 14, 2025
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

# Try to import optional dependencies
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available - language detection disabled")

logger = logging.getLogger(__name__)


class SignalDetector:
    """
    Detects service signals from user queries
    
    Uses semantic embeddings for language-independent detection:
    - Supports 6+ languages (EN, TR, AR, DE, RU, FR)
    - Multi-intent detection
    - Configurable per-language thresholds
    - Semantic similarity matching
    """
    
    def __init__(self, embedding_model=None, language_thresholds=None):
        """
        Initialize signal detector
        
        Args:
            embedding_model: Sentence transformer model
            language_thresholds: Per-language detection thresholds
        """
        self.embedding_model = embedding_model
        self._signal_embeddings = {}
        
        # Initialize or use provided thresholds
        if language_thresholds:
            self.thresholds = language_thresholds
        else:
            self._init_language_thresholds()
        
        # Initialize signal embeddings if model available
        if self.embedding_model:
            self._init_signal_embeddings()
            self._prewarm_model()
        
        logger.info("🎯 Signal detector initialized")
    
    def _init_language_thresholds(self):
        """
        Initialize per-language semantic similarity thresholds
        
        Extracted from pure_llm_handler._init_language_thresholds()
        
        Different languages have different semantic spaces and embedding quality.
        These thresholds are tuned based on:
        - Embedding model performance per language
        - Query patterns in each language
        - False positive/negative rates
        """
        self.thresholds = {
            # English: Highest quality embeddings, can use higher thresholds
            "en": {
                "needs_map": 0.35,
                "needs_gps_routing": 0.48,
                "needs_weather": 0.33,
                "needs_events": 0.38,
                "needs_hidden_gems": 0.30,
                "has_budget_constraint": 0.38,
                "likely_restaurant": 0.33,
                "likely_attraction": 0.28
            },
            # Turkish: Good support, moderate thresholds
            "tr": {
                "needs_map": 0.32,
                "needs_gps_routing": 0.45,
                "needs_weather": 0.30,
                "needs_events": 0.35,
                "needs_hidden_gems": 0.28,
                "has_budget_constraint": 0.35,
                "likely_restaurant": 0.30,
                "likely_attraction": 0.25
            },
            # Arabic: Moderate support, lower thresholds
            "ar": {
                "needs_map": 0.28,
                "needs_gps_routing": 0.42,
                "needs_weather": 0.27,
                "needs_events": 0.32,
                "needs_hidden_gems": 0.25,
                "has_budget_constraint": 0.32,
                "likely_restaurant": 0.27,
                "likely_attraction": 0.22
            },
            # German: Good support, moderate thresholds
            "de": {
                "needs_map": 0.33,
                "needs_gps_routing": 0.46,
                "needs_weather": 0.31,
                "needs_events": 0.36,
                "needs_hidden_gems": 0.29,
                "has_budget_constraint": 0.36,
                "likely_restaurant": 0.31,
                "likely_attraction": 0.26
            },
            # Russian: Moderate support, lower thresholds
            "ru": {
                "needs_map": 0.30,
                "needs_gps_routing": 0.43,
                "needs_weather": 0.28,
                "needs_events": 0.33,
                "needs_hidden_gems": 0.26,
                "has_budget_constraint": 0.33,
                "likely_restaurant": 0.28,
                "likely_attraction": 0.23
            },
            # French: Good support, moderate thresholds
            "fr": {
                "needs_map": 0.34,
                "needs_gps_routing": 0.47,
                "needs_weather": 0.32,
                "needs_events": 0.37,
                "needs_hidden_gems": 0.29,
                "has_budget_constraint": 0.37,
                "likely_restaurant": 0.32,
                "likely_attraction": 0.27
            },
            # Default: Conservative thresholds for unknown languages
            "default": {
                "needs_map": 0.32,
                "needs_gps_routing": 0.45,
                "needs_weather": 0.30,
                "needs_events": 0.35,
                "needs_hidden_gems": 0.28,
                "has_budget_constraint": 0.35,
                "likely_restaurant": 0.30,
                "likely_attraction": 0.25
            }
        }
    
    def _init_signal_embeddings(self):
        """
        Pre-compute embeddings for signal patterns
        
        Extracted from pure_llm_handler._init_signal_embeddings()
        
        This enables semantic similarity matching for language-independent detection.
        """
        if not self.embedding_model:
            return
        
        try:
            # Define signal patterns in 6 languages (abbreviated for space)
            signal_patterns = {
                'map_routing': [
                    "How do I get there?", "Show me directions", "Navigate",
                    "Oraya nasıl gidilir?", "Yol tarifi",
                    "كيف أصل؟", "Wie komme ich?", "Как добраться?", "Comment y aller?"
                ],
                'weather': [
                    "What's the weather?", "Will it rain?", "Temperature",
                    "Hava durumu?", "كيف الطقس؟", "Wie ist das Wetter?",
                    "Какая погода?", "Quel temps?"
                ],
                'events': [
                    "What events?", "Concerts", "Festivals",
                    "Etkinlikler?", "فعاليات", "Veranstaltungen",
                    "События", "Événements"
                ],
                'hidden_gems': [
                    "Local secrets", "Off beaten path", "Where locals go",
                    "Gizli yerler", "أماكن سرية", "Geheimtipps",
                    "Секретные места", "Secrets locaux"
                ],
                'budget': [
                    "Cheap", "Budget", "Expensive", "Affordable",
                    "Ucuz", "رخيص", "Günstig", "Дешевый", "Bon marché"
                ],
                'restaurant': [
                    "Where to eat?", "Restaurants", "Food",
                    "Nerede yemek?", "أين آكل؟", "Wo essen?",
                    "Где поесть?", "Où manger?"
                ],
                'attraction': [
                    "What to visit?", "Museums", "Landmarks",
                    "Nereleri gezmeliyim?", "ماذا أزور؟", "Was besichtigen?",
                    "Что посетить?", "Que visiter?"
                ]
            }
            
            # Pre-compute embeddings for each signal pattern
            for signal, patterns in signal_patterns.items():
                embeddings = self.embedding_model.encode(
                    patterns, 
                    convert_to_numpy=True, 
                    show_progress_bar=False
                )
                self._signal_embeddings[signal] = embeddings
            
            logger.debug(f"   Pre-computed {len(self._signal_embeddings)} signal embeddings")
            
        except Exception as e:
            logger.warning(f"Failed to initialize signal embeddings: {e}")
            self._signal_embeddings = {}
    
    def _prewarm_model(self):
        """
        Pre-warm the embedding model with sample queries
        
        Extracted from pure_llm_handler._prewarm_model()
        
        This eliminates the cold-start delay on the first user query.
        """
        if not self.embedding_model:
            return
        
        try:
            warm_up_queries = [
                "Where should I eat?",
                "How do I get there?",
                "What's the weather?",
                "Show me events",
                "Cheap restaurants"
            ]
            
            # Encode sample queries to warm up the model
            self.embedding_model.encode(
                warm_up_queries, 
                convert_to_numpy=True, 
                show_progress_bar=False
            )
            
            logger.debug("   Model pre-warmed (faster first query)")
            
        except Exception as e:
            logger.debug(f"   Model pre-warming failed (non-critical): {e}")
    
    async def detect_signals(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en"
    ) -> Dict[str, bool]:
        """
        Detect service signals from query using semantic similarity
        
        Args:
            query: User query string
            user_location: Optional GPS location
            language: Query language
            
        Returns:
            Dict of signal_name -> bool
        """
        signals = {
            "needs_map": False,
            "needs_gps_routing": False,
            "needs_weather": False,
            "needs_events": False,
            "needs_hidden_gems": False,
            "has_budget_constraint": False,
            "likely_restaurant": False,
            "likely_attraction": False
        }
        
        if not self.embedding_model or not self._signal_embeddings:
            return signals
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(
                [query], 
                convert_to_numpy=True, 
                show_progress_bar=False
            )[0]
            
            # Get thresholds for language
            lang_thresholds = self.thresholds.get(language, self.thresholds['default'])
            
            # Compute similarities for each signal
            for signal, pattern_embeddings in self._signal_embeddings.items():
                # Compute cosine similarity
                similarities = np.dot(pattern_embeddings, query_embedding) / (
                    np.linalg.norm(pattern_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                max_similarity = np.max(similarities)
                
                # Map signal patterns to actual signals
                signal_mapping = {
                    'map_routing': ['needs_map', 'needs_gps_routing'],
                    'weather': ['needs_weather'],
                    'events': ['needs_events'],
                    'hidden_gems': ['needs_hidden_gems'],
                    'budget': ['has_budget_constraint'],
                    'restaurant': ['likely_restaurant'],
                    'attraction': ['likely_attraction']
                }
                
                # Check threshold for each mapped signal
                for actual_signal in signal_mapping.get(signal, []):
                    threshold = lang_thresholds.get(actual_signal, 0.30)
                    if max_similarity >= threshold:
                        signals[actual_signal] = True
            
        except Exception as e:
            logger.error(f"Signal detection failed: {e}")
        
        return signals
    
    def detect_language(self, query: str) -> str:
        """
        Enhanced automatic language detection from query text
        
        Uses multiple detection methods for accuracy:
        1. Character set analysis (fast, reliable for Arabic/Russian/Turkish special chars)
        2. Keyword patterns (reliable for common phrases)
        3. langdetect library (good for longer texts)
        
        Supported languages: en, tr, ar, de, ru, fr
        
        Args:
            query: User query string
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'tr')
        """
        # Method 1: Character set analysis (most reliable for scripts with unique chars)
        char_lang = self._detect_by_charset(query)
        if char_lang:
            logger.info(f"   🌍 Language detected (charset): {char_lang}")
            return char_lang
        
        # Method 2: Keyword pattern matching (reliable for common phrases)
        keyword_lang = self._detect_by_keywords(query)
        if keyword_lang:
            logger.info(f"   🌍 Language detected (keywords): {keyword_lang}")
            return keyword_lang
        
        # Method 3: langdetect library (good for longer texts)
        if LANGDETECT_AVAILABLE:
            try:
                # Use detect_langs for confidence scores
                from langdetect import detect_langs
                detected_langs = detect_langs(query)
                
                if detected_langs:
                    top_lang = detected_langs[0]
                    lang_code = top_lang.lang
                    confidence = top_lang.prob
                    
                    # Map to supported languages
                    supported_langs = {"en", "tr", "ar", "de", "ru", "fr", "es", "it", "pt", "nl"}
                    
                    if lang_code in supported_langs:
                        # Accept if confidence is reasonable (>0.5) or if it's not English
                        # (langdetect often defaults to English with low confidence)
                        if confidence > 0.5 or (lang_code != 'en' and confidence > 0.3):
                            logger.info(f"   🌍 Language detected (langdetect): {lang_code} (confidence: {confidence:.2f})")
                            return lang_code
                        else:
                            logger.debug(f"   ⚠️ langdetect returned {lang_code} with low confidence {confidence:.2f}")
                    else:
                        logger.debug(f"   🌍 Language detected as {lang_code}, not in supported set")
                    
            except Exception as e:
                logger.debug(f"   ⚠️ langdetect failed: {e}")
        
        # Final fallback - default to English
        logger.debug(f"   ⚠️ Could not confidently detect language, defaulting to 'en'")
        return "en"
    
    def _detect_by_charset(self, text: str) -> Optional[str]:
        """Detect language by character set analysis"""
        # Count characters in different unicode ranges
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        
        # Arabic (needs at least 3 Arabic characters)
        if arabic_chars >= 3:
            return 'ar'
        
        # Russian (needs at least 3 Cyrillic characters)
        if cyrillic_chars >= 3:
            return 'ru'
        
        # Turkish (check for Turkish-specific characters)
        turkish_chars = set('ğĞıİöÖüÜşŞçÇ')
        if any(c in text for c in turkish_chars):
            return 'tr'
        
        return None
    
    def _detect_by_keywords(self, text: str) -> Optional[str]:
        """Detect language by common keywords and patterns"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Turkish keywords - expanded list with common words and suffixes
        turkish_words = {
            # Question words
            'nerede', 'nasıl', 'hangi', 'ne', 'nereye', 'nereden', 'neler', 'kaç', 'kim',
            # Common words
            'var', 'yok', 'için', 'ile', 'bir', 'bu', 've', 'da', 'de', 'den', 'dan',
            # Question suffixes
            'mi', 'mı', 'mu', 'mü', 'misin', 'mısın', 'musun', 'müsün',
            # Verbs and common patterns
            'istiyorum', 'gitmek', 'gidecek', 'gidilecek', 'yemek', 'bulmak', 
            'önerir', 'tavsiye', 'edersiniz', 'misiniz', 'lazım', 'gerek',
            # Nouns
            'restoran', 'restaurantlar', 'yer', 'yerler', 'mekan', 'mekanlar',
            'otel', 'cafe', 'kahve', 'yemek', 'içecek',
            # Location suffixes (Turkish locative -da/-de/-ta/-te)
            'istanbulda', 'taksimde', 'kadıköyde', 'beşiktaşta', 'üsküdarda',
            # Time words
            'bugün', 'yarın', 'şimdi', 'akşam', 'sabah', 'gece',
            # Adjectives
            'güzel', 'iyi', 'en', 'çok', 'ucuz', 'pahalı'
        }
        
        # Check for Turkish words
        if any(word in words for word in turkish_words):
            return 'tr'
        
        # Check for Turkish suffixes in any word (locative -da/-de, ablative -dan/-den, etc.)
        turkish_suffixes = ['da', 'de', 'ta', 'te', 'dan', 'den', 'tan', 'ten', 'lar', 'ler', 'lık', 'lik']
        for word in words:
            if len(word) > 4:  # Only check longer words
                for suffix in turkish_suffixes:
                    if word.endswith(suffix) and not word in {'ында', 'ында'}:  # Exclude Russian
                        # Additional check: word should have Turkish-compatible characters
                        if all(c in 'abcçdefgğhıijklmnoöprsştuüvyzw0123456789' for c in word):
                            logger.debug(f"   🇹🇷 Turkish detected via suffix '{suffix}' in word '{word}'")
                            return 'tr'
        
        # German keywords
        german_words = {'wo', 'wie', 'welche', 'der', 'die', 'das', 'ist', 'sind', 'können', 'gibt', 
                       'bitte', 'danke', 'guten', 'morgen', 'abend', 'nach', 'zum', 'zur'}
        if any(word in words for word in german_words):
            return 'de'
        
        # French keywords
        french_words = {'où', 'comment', 'quel', 'quelle', 'le', 'la', 'les', 'est', 'sont', 'puis',
                       'je', 'tu', 'nous', 'vous', 'merci', 'bonjour', 'bonsoir', 'au', 'aux'}
        if any(word in words for word in french_words):
            return 'fr'
        
        # Russian keywords (Latin transliteration)
        russian_words = {'gde', 'kak', 'chto', 'kogda', 'pochemu', 'skolko'}
        if any(word in words for word in russian_words):
            return 'ru'
        
        return None
