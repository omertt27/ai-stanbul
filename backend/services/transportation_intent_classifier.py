"""
Hybrid Transportation Intent Classifier

Combines multiple signals to determine if a query is transportation-related:
1. Regex patterns (fast, catches obvious cases)
2. Semantic similarity (catches human phrasing)
3. Location mentions (contextual signal)
4. LLM fallback (only when uncertain)

This prevents fragile keyword-matching and handles:
- "I'm near the ferry, how do I reach the square?"
- "What's the best way to cross to Europe side?"
- "I'm lost, can you help?"
- "Need to be in Taksim by 9am"

Author: AI Istanbul Team
Date: December 16, 2025
"""

import logging
import re
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class TransportationIntentClassifier:
    """
    Hybrid intent classifier that combines:
    - Pattern matching (fast)
    - Semantic embeddings (robust)
    - LLM clarification (uncertain cases)
    
    Returns confidence score (0.0-1.0) instead of boolean.
    """
    
    def __init__(self):
        """Initialize the classifier"""
        self.high_confidence_patterns, self.normal_patterns = self._build_patterns()
        self.semantic_examples = self._build_semantic_examples()
        self.location_keywords = self._build_location_keywords()
        
        # Initialize embedding service (if available)
        self.embedding_service = None
        try:
            from services.llm.embedding_service import get_embedding_service
            self.embedding_service = get_embedding_service()
            logger.info("✅ Hybrid Intent Classifier initialized with embeddings")
        except Exception as e:
            logger.warning(f"⚠️ Embeddings not available, using patterns only: {e}")
    
    def _build_patterns(self) -> Tuple[list, list]:
        """
        Build regex patterns for transportation queries.
        Returns (high_confidence_patterns, normal_patterns)
        """
        # Explicit routing queries - give EXTRA confidence boost
        high_confidence = [
            # Explicit "directions/route from X to Y"
            r'\b(directions?|route|routing)\s+(from|to)\b',  # "directions from X" or "directions to Y"
            r'\b(directions?|route|routing)\b.*\b(from|to)\b',  # "directions between X and Y"
            
            # "How to get/go from/to"
            r'\bhow\s+(do|can|could|to)\s+(i|we|you)\s+(get|go|travel|reach|navigate)\s+(from|to)\b',
            r'\bhow\s+(to|can\s+i|do\s+i)\s+(get|go|travel|reach)\b',  # "how to get"
            
            # Imperative directions
            r'\b(show|tell|give|find)\s+(me\s+)?(the\s+)?(way|route|directions?|path)\s+(from|to|between)\b',
            r'\b(show|tell|give)\s+.*\b(directions?|route|way)\b',  # "show directions"
            
            # Need/want directions
            r'\b(need|want)\s+.*\b(directions?|route|way|navigation)\b',
            r'\bi\s+need\s+to\s+(get|go|reach|arrive)\b',
        ]
        
        # Normal patterns
        normal = [
            # Core routing verbs
            r'\b(go|get|reach|travel|arrive)\b.*\b(to|from)\b',  # "go to", "get to", "reach X"
            r'\b(how|what).*(way|route|get|go|reach)\b',  # "how can I go", "what's the way"
            r'\b(directions?|route|routing|way|path)\b',  # Direct mention of directions
            
            # From/to patterns
            r'\b(from .+ to |to .+ from )\b',  # "from X to Y" or "to Y from X"
            r'\bfrom\b.*\bto\b',  # Simple "from ... to" pattern
            
            # Navigation verbs
            r'\b(navigate|find my way|get there)\b',
            r'\b(take me|guide me|show me|tell me)\b.*\b(to|there|way)\b',
            
            # Public transport specific
            r'\b(metro|tram|bus|ferry|marmaray|funicular|train)\b.*\b(to|from|station)\b',
            r'\b(station|stop|terminal|pier)\b.*\b(to|from)\b',
            
            # Situational (catches "I'm lost" type queries)
            r'\b(lost|stuck|can.?t find|where am i|help me (get|find))\b',
            r'\b(need to (get|be|reach|arrive))\b',
            r'\b(best way|quickest way|fastest way)\b',
        ]
        
        return high_confidence, normal
    
    def _build_semantic_examples(self) -> list:
        """
        Build example queries that represent transportation intent.
        Used for semantic similarity matching.
        """
        return [
            # Explicit routing queries
            "Give me directions from place A to place B",
            "I need directions to a location",
            "Show me the route from one place to another",
            "How do I get from here to there",
            "What's the way from point A to point B",
            
            # General navigation
            "I want to get somewhere",
            "How do I reach a place",
            "Help me find my way",
            "I need to go somewhere",
            "What's the best route",
            "How can I travel from one place to another",
            "I'm trying to get to a destination",
            "Show me how to reach my destination",
            "I need navigation help",
            "How do I get there",
            "Help me cross to another area",
            "I'm lost and need directions",
            "Need to arrive at a place",
        ]
    
    def _build_location_keywords(self) -> set:
        """Keywords that suggest geographic/location context"""
        return {
            # Istanbul neighborhoods
            'taksim', 'kadıköy', 'kadikoy', 'beşiktaş', 'besiktas', 'sultanahmet',
            'eminönü', 'eminonu', 'üsküdar', 'uskudar', 'beyoğlu', 'beyoglu',
            'galata', 'karaköy', 'karakoy', 'levent', 'şişli', 'sisli',
            
            # Transport modes
            'metro', 'tram', 'bus', 'ferry', 'marmaray', 'funicular', 'train',
            'station', 'stop', 'terminal', 'pier', 'line',
            
            # Geographic terms
            'side', 'shore', 'coast', 'european', 'asian', 'bosphorus',
            'square', 'street', 'district', 'area', 'neighborhood',
            'near', 'close', 'far', 'distance',
        }
    
    def classify_intent(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, any]]:
        """
        Classify transportation intent with confidence score.
        
        Args:
            query: User's query text
            user_location: Optional GPS coordinates
            
        Returns:
            Tuple of (confidence_score, debug_info)
            
        Confidence breakdown:
            0.0-0.4: Unlikely transportation query
            0.4-0.6: Uncertain, may need clarification
            0.6-0.8: Likely transportation query
            0.8-1.0: Highly confident transportation query
        """
        query_lower = query.lower().strip()
        
        # Track confidence contributions
        confidence = 0.0
        debug_info = {
            'query': query,
            'regex_hit': False,
            'semantic_score': 0.0,
            'location_mentions': 0,
            'has_gps': user_location is not None,
            'signals': []
        }
        
        # Signal 1: Regex pattern matching
        regex_hit = False
        high_confidence_hit = False
        
        # Check high-confidence patterns first (+0.6)
        for pattern in self.high_confidence_patterns:
            if re.search(pattern, query_lower):
                regex_hit = True
                high_confidence_hit = True
                confidence += 0.6
                debug_info['regex_hit'] = True
                debug_info['high_confidence_pattern'] = True
                debug_info['signals'].append('high_confidence_regex')
                logger.debug(f"✅ High-confidence regex hit: {pattern[:50]}...")
                break
        
        # If no high-confidence hit, check normal patterns (+0.4)
        if not high_confidence_hit:
            for pattern in self.normal_patterns:
                if re.search(pattern, query_lower):
                    regex_hit = True
                    confidence += 0.4
                    debug_info['regex_hit'] = True
                    debug_info['signals'].append('regex_pattern_match')
                    logger.debug(f"✅ Normal regex hit: {pattern[:50]}...")
                    break
        
        # Signal 2: Semantic similarity (+0.4 if high similarity)
        semantic_score = 0.0
        used_embeddings = False
        
        if self.embedding_service and not self.embedding_service.offline_mode:
            try:
                # Get query embedding
                query_embedding = self.embedding_service.encode(query)
                
                if query_embedding is not None:
                    used_embeddings = True
                    # Compare against transportation examples
                    max_similarity = 0.0
                    for example in self.semantic_examples:
                        example_embedding = self.embedding_service.encode(example)
                        if example_embedding is not None:
                            similarity = self._cosine_similarity(query_embedding, example_embedding)
                            max_similarity = max(max_similarity, similarity)
                    
                    semantic_score = max_similarity
                    debug_info['semantic_score'] = round(semantic_score, 3)
                    
                    # Add confidence based on semantic score (more lenient thresholds)
                    if semantic_score > 0.60:  # Lowered from 0.75
                        confidence += 0.4
                        debug_info['signals'].append('high_semantic_similarity')
                    elif semantic_score > 0.50:  # Lowered from 0.65
                        confidence += 0.3
                        debug_info['signals'].append('medium_semantic_similarity')
                    elif semantic_score > 0.40:  # Lowered from 0.55
                        confidence += 0.2
                        debug_info['signals'].append('low_semantic_similarity')
                    
                    logger.debug(f"🔍 Semantic score: {semantic_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {e}")
        
        # Fallback: keyword-based semantic matching (if embeddings not used)
        if not used_embeddings:
            transport_keywords = {
                'how', 'way', 'route', 'get', 'go', 'reach', 'travel', 'arrive',
                'directions', 'navigate', 'from', 'to', 'take', 'guide', 'show'
            }
            query_words = set(query_lower.split())
            overlap = len(query_words.intersection(transport_keywords))
            
            if overlap >= 2:
                confidence += 0.3
                debug_info['signals'].append('keyword_overlap_fallback')
                debug_info['keyword_overlap'] = overlap
                logger.debug(f"🔍 Keyword overlap: {overlap} words")
        
        # Signal 3: Location mentions (+0.2)
        location_mentions = 0
        for keyword in self.location_keywords:
            if keyword in query_lower:
                location_mentions += 1
        
        if location_mentions > 0:
            confidence += min(0.2, location_mentions * 0.1)
            debug_info['location_mentions'] = location_mentions
            debug_info['signals'].append(f'{location_mentions}_location_keywords')
            logger.debug(f"📍 Found {location_mentions} location keywords")
        
        # Signal 4: GPS presence (minor boost +0.1)
        if user_location:
            confidence += 0.1
            debug_info['signals'].append('gps_available')
            logger.debug(f"📍 GPS coordinates available")
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        debug_info['final_confidence'] = round(confidence, 3)
        
        logger.info(
            f"🎯 Intent classification: confidence={confidence:.3f}, "
            f"regex={regex_hit}, semantic={semantic_score:.3f}, "
            f"locations={location_mentions}, gps={user_location is not None}"
        )
        
        return confidence, debug_info
    
    async def clarify_with_llm(
        self,
        query: str,
        llm_service
    ) -> bool:
        """
        Use LLM to clarify intent when confidence is uncertain (0.4-0.6).
        
        This is CHEAP and FAST:
        - max_tokens: 3
        - temperature: 0
        - Expected latency: ~300ms
        
        Only called when needed, not for every query.
        """
        try:
            prompt = f"""Does this user want directions or navigation help?
Query: "{query}"

Answer only YES or NO."""

            response = await llm_service.generate(
                prompt=prompt,
                max_tokens=3,
                temperature=0
            )
            
            answer = response.strip().upper()
            is_transportation = 'YES' in answer
            
            logger.info(f"🤖 LLM clarification: '{query[:50]}...' → {answer}")
            
            return is_transportation
            
        except Exception as e:
            logger.error(f"LLM clarification failed: {e}")
            # Fail safe: assume not transportation
            return False
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# Singleton instance
_intent_classifier = None

def get_transportation_intent_classifier() -> TransportationIntentClassifier:
    """Get or create intent classifier singleton"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = TransportationIntentClassifier()
    return _intent_classifier
