"""
Advanced NLP Service using spaCy

Provides enhanced natural language processing capabilities:
- Named Entity Recognition (NER)
- Intent classification
- Entity extraction
- Language detection
- Semantic similarity

Author: AI Istanbul Team
Date: December 2024
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import spaCy
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("⚠️ spaCy not installed. Install with: pip install spacy")

# Try to import language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("⚠️ langdetect not installed")


class IntentType(str, Enum):
    """Supported intent types"""
    RESTAURANT = "restaurant"
    ATTRACTION = "attraction"
    TRANSPORTATION = "transportation"
    HOTEL = "hotel"
    WEATHER = "weather"
    GREETING = "greeting"
    FAREWELL = "farewell"
    HELP = "help"
    HIDDEN_GEM = "hidden_gem"
    ITINERARY = "itinerary"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "normalized": self.normalized
        }


@dataclass
class NLPResult:
    """Result from NLP processing"""
    original_text: str
    processed_text: str
    language: str
    language_confidence: float
    intent: IntentType
    intent_confidence: float
    entities: List[ExtractedEntity] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "processed_text": self.processed_text,
            "language": self.language,
            "language_confidence": self.language_confidence,
            "intent": self.intent.value,
            "intent_confidence": self.intent_confidence,
            "entities": [e.to_dict() for e in self.entities],
            "keywords": self.keywords,
            "sentiment": self.sentiment
        }


class AdvancedNLPService:
    """
    Advanced NLP service for enhanced text understanding.
    
    Uses spaCy for:
    - Named Entity Recognition
    - Part-of-speech tagging
    - Dependency parsing
    - Lemmatization
    """
    
    # Intent keywords mapping
    INTENT_KEYWORDS = {
        IntentType.RESTAURANT: [
            "restaurant", "food", "eat", "dinner", "lunch", "breakfast",
            "cafe", "coffee", "cuisine", "yemek", "restoran", "lokanta",
            "kebab", "döner", "meze", "baklava", "hungry"
        ],
        IntentType.ATTRACTION: [
            "visit", "see", "attraction", "museum", "mosque", "palace",
            "hagia sophia", "blue mosque", "topkapi", "basilica cistern",
            "galata", "bosphorus", "tour", "sightseeing", "gezilecek"
        ],
        IntentType.TRANSPORTATION: [
            "go", "get", "metro", "bus", "tram", "ferry", "taxi",
            "transport", "direction", "route", "how to get", "nasıl gidilir",
            "ulaşım", "metro", "otobüs", "vapur", "marmaray"
        ],
        IntentType.HOTEL: [
            "hotel", "stay", "accommodation", "hostel", "airbnb",
            "book", "room", "otel", "konaklama", "oda"
        ],
        IntentType.WEATHER: [
            "weather", "temperature", "rain", "sunny", "forecast",
            "hava", "sıcaklık", "yağmur", "güneşli"
        ],
        IntentType.GREETING: [
            "hi", "hello", "hey", "good morning", "good evening",
            "merhaba", "selam", "günaydın", "iyi akşamlar"
        ],
        IntentType.FAREWELL: [
            "bye", "goodbye", "see you", "thanks", "thank you",
            "hoşça kal", "güle güle", "teşekkürler", "sağol"
        ],
        IntentType.HELP: [
            "help", "what can you do", "how to use", "yardım",
            "ne yapabilirsin", "nasıl kullanılır"
        ],
        IntentType.HIDDEN_GEM: [
            "hidden gem", "secret", "local", "off the beaten path",
            "unknown", "authentic", "gizli", "yerel", "bilinmeyen"
        ],
        IntentType.ITINERARY: [
            "itinerary", "plan", "schedule", "day trip", "tour",
            "program", "günlük", "plan", "tur"
        ]
    }
    
    # Istanbul-specific entities
    ISTANBUL_LOCATIONS = {
        "sultanahmet": {"type": "district", "normalized": "Sultanahmet"},
        "taksim": {"type": "district", "normalized": "Taksim"},
        "kadikoy": {"type": "district", "normalized": "Kadıköy"},
        "kadıköy": {"type": "district", "normalized": "Kadıköy"},
        "besiktas": {"type": "district", "normalized": "Beşiktaş"},
        "beşiktaş": {"type": "district", "normalized": "Beşiktaş"},
        "eminonu": {"type": "district", "normalized": "Eminönü"},
        "eminönü": {"type": "district", "normalized": "Eminönü"},
        "karakoy": {"type": "district", "normalized": "Karaköy"},
        "karaköy": {"type": "district", "normalized": "Karaköy"},
        "galata": {"type": "landmark", "normalized": "Galata"},
        "hagia sophia": {"type": "landmark", "normalized": "Hagia Sophia"},
        "ayasofya": {"type": "landmark", "normalized": "Hagia Sophia"},
        "blue mosque": {"type": "landmark", "normalized": "Blue Mosque"},
        "sultan ahmed": {"type": "landmark", "normalized": "Blue Mosque"},
        "sultanahmet camii": {"type": "landmark", "normalized": "Blue Mosque"},
        "topkapi": {"type": "landmark", "normalized": "Topkapi Palace"},
        "topkapı": {"type": "landmark", "normalized": "Topkapi Palace"},
        "grand bazaar": {"type": "landmark", "normalized": "Grand Bazaar"},
        "kapalıçarşı": {"type": "landmark", "normalized": "Grand Bazaar"},
        "spice bazaar": {"type": "landmark", "normalized": "Spice Bazaar"},
        "mısır çarşısı": {"type": "landmark", "normalized": "Spice Bazaar"},
        "bosphorus": {"type": "landmark", "normalized": "Bosphorus"},
        "boğaz": {"type": "landmark", "normalized": "Bosphorus"},
        "dolmabahce": {"type": "landmark", "normalized": "Dolmabahçe Palace"},
        "dolmabahçe": {"type": "landmark", "normalized": "Dolmabahçe Palace"},
        "basilica cistern": {"type": "landmark", "normalized": "Basilica Cistern"},
        "yerebatan": {"type": "landmark", "normalized": "Basilica Cistern"},
        "istiklal": {"type": "landmark", "normalized": "Istiklal Avenue"},
        "princess islands": {"type": "landmark", "normalized": "Princess Islands"},
        "adalar": {"type": "landmark", "normalized": "Princess Islands"},
    }
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the NLP service.
        
        Args:
            model_name: spaCy model to load (default: en_core_web_sm)
        """
        self.nlp = None
        self.model_name = model_name
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"✅ spaCy model '{model_name}' loaded")
            except OSError:
                logger.warning(f"⚠️ spaCy model '{model_name}' not found")
                logger.info("   Install with: python -m spacy download en_core_web_sm")
                # Try to download
                try:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
                    self.nlp = spacy.load(model_name)
                    logger.info(f"✅ Downloaded and loaded '{model_name}'")
                except Exception as e:
                    logger.error(f"Failed to download spaCy model: {e}")
        else:
            logger.warning("spaCy not available - using rule-based NLP")
    
    def process(self, text: str) -> NLPResult:
        """
        Process text through the NLP pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            NLPResult with extracted information
        """
        # Detect language
        language, lang_confidence = self._detect_language(text)
        
        # Clean and normalize text
        processed_text = self._preprocess_text(text)
        
        # Classify intent
        intent, intent_confidence = self._classify_intent(processed_text, language)
        
        # Extract entities
        entities = self._extract_entities(processed_text, language)
        
        # Extract keywords
        keywords = self._extract_keywords(processed_text)
        
        # Analyze sentiment (basic)
        sentiment = self._analyze_sentiment(processed_text)
        
        return NLPResult(
            original_text=text,
            processed_text=processed_text,
            language=language,
            language_confidence=lang_confidence,
            intent=intent,
            intent_confidence=intent_confidence,
            entities=entities,
            keywords=keywords,
            sentiment=sentiment
        )
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the language of the text."""
        if not text or len(text.strip()) < 3:
            return "en", 0.5
        
        # Quick check for Turkish characters
        turkish_chars = set("şğüöçıŞĞÜÖÇİ")
        if any(c in text for c in turkish_chars):
            return "tr", 0.95
        
        # Quick check for Arabic
        if re.search(r'[\u0600-\u06FF]', text):
            return "ar", 0.95
        
        # Quick check for Russian/Cyrillic
        if re.search(r'[\u0400-\u04FF]', text):
            return "ru", 0.95
        
        if LANGDETECT_AVAILABLE:
            try:
                langs = detect_langs(text)
                if langs:
                    top_lang = langs[0]
                    return top_lang.lang, top_lang.prob
            except Exception as e:
                logger.debug(f"Language detection failed: {e}")
        
        return "en", 0.7
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Turkish/Arabic
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0400-\u04FFşğüöçıŞĞÜÖÇİ]', '', text)
        
        return text
    
    def _classify_intent(self, text: str, language: str) -> Tuple[IntentType, float]:
        """
        Classify the intent of the text.
        
        Uses keyword matching with scoring.
        """
        text_lower = text.lower()
        
        scores: Dict[IntentType, float] = {}
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = 0.0
            matches = 0
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Exact word match gets higher score
                    if re.search(rf'\b{re.escape(keyword.lower())}\b', text_lower):
                        score += 2.0
                        matches += 1
                    else:
                        score += 1.0
                        matches += 1
            
            # Normalize by keyword count
            if matches > 0:
                scores[intent] = score / len(keywords) * matches
        
        if not scores:
            return IntentType.GENERAL, 0.5
        
        # Get top intent
        top_intent = max(scores, key=scores.get)
        top_score = scores[top_intent]
        
        # Normalize confidence
        confidence = min(1.0, top_score / 3.0)
        
        return top_intent, confidence
    
    def _extract_entities(self, text: str, language: str) -> List[ExtractedEntity]:
        """Extract named entities from text."""
        entities = []
        text_lower = text.lower()
        
        # 1. Extract Istanbul-specific locations
        for location, info in self.ISTANBUL_LOCATIONS.items():
            if location.lower() in text_lower:
                # Find position
                start = text_lower.find(location.lower())
                end = start + len(location)
                
                entities.append(ExtractedEntity(
                    text=location,
                    label=f"ISTANBUL_{info['type'].upper()}",
                    start=start,
                    end=end,
                    confidence=0.95,
                    normalized=info['normalized']
                ))
        
        # 2. Use spaCy for general NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Skip if already found as Istanbul location
                if any(e.text.lower() == ent.text.lower() for e in entities):
                    continue
                
                entities.append(ExtractedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8
                ))
        
        # 3. Extract time expressions
        time_patterns = [
            (r'\b(\d{1,2}:\d{2})\b', "TIME"),
            (r'\b(today|tomorrow|yesterday|bugün|yarın|dün)\b', "DATE"),
            (r'\b(morning|afternoon|evening|night|sabah|öğle|akşam|gece)\b', "TIME_OF_DAY"),
            (r'\b(\d+)\s*(hours?|minutes?|saat|dakika)\b', "DURATION"),
        ]
        
        for pattern, label in time_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        keywords = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract nouns and proper nouns
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                    keywords.append(token.lemma_)
            
            # Extract noun chunks
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1:
                    keywords.append(chunk.text)
        else:
            # Simple keyword extraction
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "from", "in", "on", "at"}
            words = text.lower().split()
            keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(keywords))[:10]
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Basic sentiment analysis.
        
        Returns a score from -1 (negative) to 1 (positive).
        """
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "beautiful",
            "best", "love", "like", "nice", "recommend", "perfect", "awesome",
            "güzel", "harika", "mükemmel", "süper", "iyi"
        }
        
        negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike",
            "expensive", "crowded", "dirty", "disappointing", "avoid",
            "kötü", "berbat", "pahalı", "kalabalık", "pis"
        }
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def get_location_suggestions(self, text: str) -> List[Dict[str, Any]]:
        """
        Get location suggestions based on partial text input.
        
        Useful for autocomplete functionality.
        """
        suggestions = []
        text_lower = text.lower()
        
        for location, info in self.ISTANBUL_LOCATIONS.items():
            if location.startswith(text_lower) or info['normalized'].lower().startswith(text_lower):
                suggestions.append({
                    "text": info['normalized'],
                    "type": info['type'],
                    "match_score": len(text_lower) / len(location)
                })
        
        # Sort by match score
        suggestions.sort(key=lambda x: x['match_score'], reverse=True)
        
        return suggestions[:5]


# Singleton instance
_nlp_service: Optional[AdvancedNLPService] = None


def get_nlp_service() -> AdvancedNLPService:
    """Get or create the NLP service singleton."""
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = AdvancedNLPService()
    return _nlp_service
