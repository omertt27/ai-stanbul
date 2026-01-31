"""
Lightweight NLP Service for Istanbul Travel Assistant

Provides basic text processing capabilities:
- Istanbul location extraction
- Text preprocessing
- Basic entity extraction

NOTE: Intent detection and language detection are handled by the LLM directly.
This service is kept lightweight for fast startup and processing.

Author: AI Istanbul Team
Date: December 2024
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


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
    Lightweight NLP service for text processing.
    
    NOTE: Language detection and intent classification are handled by the LLM.
    This service only handles:
    - Istanbul location extraction
    - Text preprocessing
    - Basic entity extraction (time, locations)
    """
    
    # Istanbul-specific entities (kept for location extraction)
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
    
    def __init__(self):
        """
        Initialize the lightweight NLP service.
        No external dependencies needed - instant startup.
        """
        logger.info("✅ Lightweight NLP service initialized (no spaCy/langdetect)")
    
    def process(self, text: str) -> NLPResult:
        """
        Process text through the lightweight NLP pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            NLPResult with extracted information
        """
        # Clean and normalize text
        processed_text = self._preprocess_text(text)
        
        # Extract Istanbul locations
        entities = self._extract_entities(processed_text)
        
        # Extract basic keywords
        keywords = self._extract_keywords(processed_text)
        
        # NOTE: Language and intent detection are done by the LLM
        # We return defaults here - the LLM will determine the actual values
        return NLPResult(
            original_text=text,
            processed_text=processed_text,
            language="auto",  # LLM will detect
            language_confidence=1.0,
            intent=IntentType.GENERAL,  # LLM will determine
            intent_confidence=1.0,
            entities=entities,
            keywords=keywords,
            sentiment=0.0
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Turkish/Arabic
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0400-\u04FFşğüöçıŞĞÜÖÇİ]', '', text)
        
        return text
    
    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract Istanbul locations and time expressions from text."""
        entities = []
        text_lower = text.lower()
        
        # 1. Extract Istanbul-specific locations
        for location, info in self.ISTANBUL_LOCATIONS.items():
            if location.lower() in text_lower:
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
        
        # 2. Extract time expressions
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
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "from", 
                      "in", "on", "at", "for", "with", "and", "or", "but", "i", 
                      "you", "we", "they", "it", "this", "that", "can", "do", "how"}
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return list(set(keywords))[:10]
    
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
