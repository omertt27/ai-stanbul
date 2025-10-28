#!/usr/bin/env python3
"""
Main System Integration with Neural Query Classifier
Hybrid approach: DistilBERT classifier with rule-based fallback
Uses newly trained DistilBERT model (91.3% accuracy, 30 intents)
"""

import logging
from typing import Dict, Tuple, Optional
from distilbert_intent_inference import get_distilbert_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralIntentRouter:
    """
    Hybrid intent routing system
    - Primary: Neural classifier (fast, learns from data)
    - Fallback: Rule-based patterns (reliable for low-confidence)
    """
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.80  # Use neural prediction directly
    MEDIUM_CONFIDENCE = 0.60  # Use neural with validation
    LOW_CONFIDENCE = 0.40  # Fall back to rules
    
    # Keyword-based fallback patterns (Turkish + English)
    KEYWORD_PATTERNS = {
        "emergency": [
            # Turkish
            "acil", "polis", "hastane", "ambulans", "kayboldum", "tehlike", "yardım edin",
            # English
            "emergency", "police", "hospital", "ambulance", "lost", "help", "danger"
        ],
        "weather": [
            # Turkish
            "hava", "yağmur", "güneş", "sıcaklık", "derece", "soğuk", "sıcak",
            # English
            "weather", "rain", "sunny", "temperature", "cold", "hot", "forecast"
        ],
        "restaurant": [
            # Turkish
            "restoran", "yemek", "lokanta", "balık", "kebap", "meze",
            # English
            "restaurant", "food", "eat", "dining", "fish", "kebab"
        ],
        "attraction": [
            # Turkish
            "ayasofya", "topkapı", "galata", "cistern", "sultanahmet", "görülecek",
            # English
            "hagia sophia", "topkapi", "galata", "attraction", "visit", "sightseeing", "places to see"
        ],
        "transportation": [
            # Turkish
            "metro", "tramvay", "otobüs", "taksi", "ferry", "ulaşım",
            # English
            "metro", "tram", "bus", "taxi", "ferry", "transport", "get to"
        ],
        "museum": [
            # Turkish
            "müze", "galeri", "sanat", "sergi",
            # English
            "museum", "gallery", "art", "exhibition"
        ],
        "accommodation": [
            # Turkish
            "otel", "hostel", "konaklama", "pansiyon", "ucuz otel",
            # English
            "hotel", "hostel", "stay", "accommodation", "lodging", "cheap hotel", "looking for"
        ],
        "booking": [
            # Turkish
            "rezervasyon", "ayırt", "bilet al", "online",
            # English
            "book", "reservation", "reserve", "ticket"
        ],
        "shopping": [
            # Turkish
            "alışveriş", "çarşı", "market", "mağaza", "kapalıçarşı",
            # English
            "shopping", "shop", "mall", "bazaar", "grand bazaar", "market"
        ],
        "route_planning": [
            # Turkish
            "rota", "güzergah", "itiner", "plan",
            # English
            "route", "itinerary", "plan", "path"
        ],
        "gps_navigation": [
            # Turkish
            "konum", "gps", "harita", "navigasyon",
            # English
            "location", "gps", "map", "navigation", "navigate"
        ],
        "romantic": [
            # Turkish
            "romantik", "çift", "balayı", "gün batımı",
            # English
            "romantic", "couple", "honeymoon", "sunset", "date"
        ],
        "family_activities": [
            # Turkish
            "çocuk", "aile", "ailece", "çocuklarla",
            # English
            "family", "kids", "children", "child", "with kids", "with children"
        ],
        "nightlife": [
            # Turkish
            "gece", "bar", "kulüp", "eğlence",
            # English
            "nightlife", "bar", "club", "night", "party"
        ],
        "food": [
            # Turkish
            "türk mutfağı", "kahvaltı", "baklava", "yemek kültürü",
            # English
            "turkish food", "cuisine", "breakfast", "baklava", "traditional food"
        ],
        "history": [
            # Turkish
            "tarih", "bizans", "osmanlı",
            # English
            "history", "historical", "byzantine", "ottoman"
        ],
    }
    
    def __init__(self):
        """Initialize neural intent router with DistilBERT classifier"""
        logger.info("Initializing Neural Intent Router with DistilBERT...")
        self.neural_classifier = get_distilbert_classifier()
        logger.info("✅ Neural Intent Router ready (DistilBERT 91.3% accuracy, 30 intents)")
    
    def route_query(self, query: str, user_context: Optional[Dict] = None) -> Dict:
        """
        Route query to appropriate intent handler
        
        Args:
            query: User query in Turkish
            user_context: Optional user context (location, preferences, etc.)
            
        Returns:
            Dict with intent, confidence, method, and metadata
        """
        # Step 1: Get neural prediction
        neural_intent, neural_confidence = self.neural_classifier.predict(query)
        
        # Step 2: Apply confidence-based routing
        if neural_confidence >= self.HIGH_CONFIDENCE:
            # High confidence - use neural prediction directly
            return {
                'intent': neural_intent,
                'confidence': neural_confidence,
                'method': 'neural',
                'fallback_used': False,
                'query': query
            }
        
        elif neural_confidence >= self.MEDIUM_CONFIDENCE:
            # Medium confidence - validate with keywords
            keyword_intent = self._check_keywords(query)
            
            if keyword_intent and keyword_intent == neural_intent:
                # Neural and keywords agree
                return {
                    'intent': neural_intent,
                    'confidence': neural_confidence,
                    'method': 'neural_validated',
                    'fallback_used': False,
                    'query': query
                }
            elif keyword_intent:
                # Keywords override neural
                return {
                    'intent': keyword_intent,
                    'confidence': 0.75,  # Medium confidence for keyword match
                    'method': 'keyword_override',
                    'fallback_used': True,
                    'neural_prediction': neural_intent,
                    'query': query
                }
            else:
                # No keyword match, trust neural
                return {
                    'intent': neural_intent,
                    'confidence': neural_confidence,
                    'method': 'neural_unvalidated',
                    'fallback_used': False,
                    'query': query
                }
        
        else:
            # Low confidence - use keyword fallback
            keyword_intent = self._check_keywords(query)
            
            if keyword_intent:
                return {
                    'intent': keyword_intent,
                    'confidence': 0.70,
                    'method': 'keyword_fallback',
                    'fallback_used': True,
                    'neural_prediction': neural_intent,
                    'neural_confidence': neural_confidence,
                    'query': query
                }
            else:
                # No keywords found, use general_info as safe default
                return {
                    'intent': 'general_info',
                    'confidence': 0.50,
                    'method': 'default_fallback',
                    'fallback_used': True,
                    'neural_prediction': neural_intent,
                    'neural_confidence': neural_confidence,
                    'query': query
                }
    
    def _check_keywords(self, query: str) -> Optional[str]:
        """
        Check for keyword patterns in query
        Prioritizes longer matches and specific phrases
        
        Returns:
            Intent name if keywords match, None otherwise
        """
        query_lower = query.lower()
        
        # Priority matches - check specific phrases first
        priority_patterns = {
            "family_activities": ["with kids", "with children", "çocuklarla"],
            "accommodation": ["cheap hotel", "ucuz otel", "looking for"],
            "attraction": ["hagia sophia", "ayasofya", "bosphorus", "boğaz"],
        }
        
        for intent, patterns in priority_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return intent
        
        # General keyword matching
        for intent, keywords in self.KEYWORD_PATTERNS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent
        
        return None
    
    def get_top_intents(self, query: str, k: int = 3) -> list:
        """Get top K most likely intents"""
        return self.neural_classifier.get_top_k_intents(query, k)
    
    def get_statistics(self) -> Dict:
        """Get classifier statistics"""
        return self.neural_classifier.get_stats()


# Singleton instance
_router_instance = None


def get_neural_router() -> NeuralIntentRouter:
    """Get singleton neural router instance"""
    global _router_instance
    
    if _router_instance is None:
        _router_instance = NeuralIntentRouter()
    
    return _router_instance


def test_neural_router():
    """Test the neural router with various queries"""
    print("=" * 80)
    print("NEURAL INTENT ROUTER TEST")
    print("=" * 80)
    print()
    
    router = get_neural_router()
    
    test_queries = [
        # High confidence expected
        "Ayasofya'yı görmek istiyorum",
        "Hava durumu nasıl?",
        "Müze önerileri",
        
        # Medium confidence (needs validation)
        "En yakın restoran nerede?",
        "Ucuz otel",
        "Romantik yerler",
        
        # Low confidence (may need fallback)
        "Acil durum!",
        "Boğaz turu ne kadar?",
        "Çocuklarla nereye gidebilirim?",
        
        # Edge cases
        "Merhaba",
        "Yardım",
        "Ne yapabilirim?",
    ]
    
    for query in test_queries:
        result = router.route_query(query)
        
        # Format output
        conf_marker = "🔥" if result['confidence'] >= 0.80 else "✅" if result['confidence'] >= 0.60 else "⚠️"
        fallback_marker = " (FB)" if result['fallback_used'] else ""
        
        print(f"{conf_marker} Query: '{query}'")
        print(f"   Intent: {result['intent']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Method: {result['method']}{fallback_marker}")
        
        if 'neural_prediction' in result:
            print(f"   (Neural predicted: {result['neural_prediction']})")
        
        print()
    
    # Show statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    stats = router.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    test_neural_router()
