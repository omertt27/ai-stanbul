"""
Lightweight Neural Query Enhancement System
===========================================

Optimized for student budget and CPU-only deployment (~10k users/month)
- Uses small, efficient models (<100MB each)
- Fast CPU inference (<100ms per query)
- Low memory footprint (<1GB total)
- Template-based responses with neural ranking (no generative models)

Budget-friendly architecture for Istanbul tourism queries
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import re
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

# Lightweight ML imports (all budget-friendly)
# Singleton pattern for spaCy model to avoid multiple loads
_spacy_nlp = None
SPACY_AVAILABLE = False

try:
    import spacy
    try:
        _spacy_nlp = spacy.load("en_core_web_sm")  # Only 12MB model
        SPACY_AVAILABLE = True
        logger.info("✅ spaCy NLP with en_core_web_sm model loaded successfully")
    except OSError:
        _spacy_nlp = None
        logger.info("ℹ️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    _spacy_nlp = None
    logger.info("ℹ️  spaCy not installed - using rule-based NLP fallback")

# Singleton for TextBlob (lazy import pattern)
_textblob_class = None
TEXTBLOB_AVAILABLE = False

try:
    from textblob import TextBlob as _TextBlobClass
    _textblob_class = _TextBlobClass
    TEXTBLOB_AVAILABLE = True
    logger.info("✅ TextBlob sentiment analysis loaded successfully")
except ImportError:
    logger.info("ℹ️  TextBlob not installed - using basic sentiment analysis")

# Alias for backward compatibility
nlp = _spacy_nlp
TextBlob = _textblob_class

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️  scikit-learn not available")


@dataclass
class LightweightNeuralInsights:
    """Neural insights from lightweight processing"""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    semantic_features: Dict[str, Any] = field(default_factory=dict)
    sentiment: Optional[str] = None
    sentiment_score: float = 0.0
    keywords: List[str] = field(default_factory=list)
    query_complexity: str = "simple"
    location_context: Optional[str] = None
    temporal_context: Optional[str] = None
    processing_time_ms: float = 0.0


class LightweightNeuralProcessor:
    """
    Budget-friendly neural processor optimized for CPU deployment
    
    Memory: <500MB
    Latency: <100ms per query
    Cost: Fits in basic Google Cloud instance (e2-small: ~$15/month)
    """
    
    def __init__(self):
        """Initialize lightweight neural processor"""
        self.nlp = nlp if SPACY_AVAILABLE else None
        
        # Rule-based patterns (no memory cost!)
        self.entity_patterns = self._initialize_entity_patterns()
        self.intent_keywords = self._initialize_intent_keywords()
        self.location_keywords = self._initialize_location_keywords()
        self.temporal_keywords = self._initialize_temporal_keywords()
        
        # Lightweight TF-IDF for semantic similarity (minimal memory)
        self.vectorizer = None
        self.query_cache = {}
        
        if SKLEARN_AVAILABLE:
            self._initialize_semantic_vectors()
        
        logger.info("✅ Lightweight neural processor initialized (Budget-friendly mode)")
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns"""
        return {
            'location': [
                r'\b(hagia sophia|blue mosque|topkapi|galata|bosphorus|taksim|sultanahmet|'
                r'grand bazaar|spice bazaar|dolmabahce|maiden tower|ortakoy|eminonu|'
                r'kadikoy|besiktas|sisli|beyoglu)\b',
                r'\b(mosque|palace|museum|tower|square|market|bazaar|bridge|strait)\b'
            ],
            'cuisine': [
                r'\b(kebab|baklava|turkish coffee|tea|simit|borek|meze|raki|lahmacun|'
                r'pide|kofte|doner|iskender|turkish delight|lokum)\b',
                r'\b(restaurant|cafe|food|eat|drink|cuisine|dish|meal)\b'
            ],
            'activity': [
                r'\b(visit|tour|see|explore|walk|cruise|shop|eat|experience|discover|'
                r'enjoy|watch|attend|participate)\b',
                r'\b(activity|attraction|thing to do|experience|tour|excursion)\b'
            ],
            'time': [
                r'\b(morning|afternoon|evening|night|sunrise|sunset|weekend|weekday|'
                r'today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                r'\b(\d{1,2}:\d{2}|\d{1,2}\s*(am|pm))\b'
            ],
            'transportation': [
                r'\b(metro|tram|bus|ferry|taxi|dolmus|funicular|cable car|'
                r'istanbulkart|transportation|transport|travel|get to)\b'
            ]
        }
    
    def _initialize_intent_keywords(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intent classification keywords"""
        return {
            'attraction': {
                'keywords': ['visit', 'see', 'attraction', 'monument', 'palace', 'museum', 
                            'mosque', 'church', 'tower', 'landmark', 'famous', 'historic'],
                'weight': 1.0
            },
            'food': {
                'keywords': ['eat', 'food', 'restaurant', 'cuisine', 'dish', 'meal', 
                            'breakfast', 'lunch', 'dinner', 'cafe', 'taste', 'try'],
                'weight': 1.0
            },
            'transportation': {
                'keywords': ['get to', 'how to reach', 'transport', 'metro', 'bus', 'ferry', 
                            'taxi', 'travel', 'route', 'direction', 'way to'],
                'weight': 1.0
            },
            'activity': {
                'keywords': ['do', 'activity', 'things to do', 'experience', 'tour', 
                            'excursion', 'adventure', 'fun', 'entertainment'],
                'weight': 1.0
            },
            'accommodation': {
                'keywords': ['hotel', 'stay', 'accommodation', 'hostel', 'lodge', 
                            'booking', 'reserve', 'where to stay'],
                'weight': 1.0
            },
            'shopping': {
                'keywords': ['shop', 'buy', 'market', 'bazaar', 'mall', 'store', 
                            'souvenir', 'shopping', 'purchase'],
                'weight': 1.0
            },
            'culture': {
                'keywords': ['culture', 'tradition', 'history', 'heritage', 'art', 
                            'music', 'festival', 'event', 'celebration'],
                'weight': 1.0
            },
            'nightlife': {
                'keywords': ['night', 'nightlife', 'club', 'bar', 'pub', 'party', 
                            'evening', 'entertainment', 'drinks'],
                'weight': 1.0
            },
            'general_info': {
                'keywords': ['info', 'information', 'about', 'tell me', 'what is', 
                            'explain', 'describe', 'help', 'question'],
                'weight': 0.8
            },
            'recommendation': {
                'keywords': ['recommend', 'suggest', 'best', 'top', 'favorite', 
                            'popular', 'must see', 'should visit', 'worth'],
                'weight': 1.0
            }
        }
    
    def _initialize_location_keywords(self) -> Dict[str, List[str]]:
        """Initialize location context keywords"""
        return {
            'sultanahmet': ['sultanahmet', 'old city', 'historic peninsula', 'hagia sophia', 'blue mosque'],
            'beyoglu': ['beyoglu', 'taksim', 'istiklal', 'galata'],
            'bosphorus': ['bosphorus', 'strait', 'waterway', 'asian side', 'european side'],
            'asian_side': ['kadikoy', 'uskudar', 'asian side', 'anadolu'],
            'european_side': ['european side', 'avrupa', 'fatih', 'beyoglu', 'besiktas']
        }
    
    def _initialize_temporal_keywords(self) -> Dict[str, List[str]]:
        """Initialize temporal context keywords"""
        return {
            'morning': ['morning', 'breakfast', 'sunrise', 'early', 'am'],
            'afternoon': ['afternoon', 'lunch', 'midday', 'noon'],
            'evening': ['evening', 'dinner', 'sunset', 'dusk'],
            'night': ['night', 'midnight', 'late', 'nightlife'],
            'weekend': ['weekend', 'saturday', 'sunday'],
            'weekday': ['weekday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        }
    
    def _initialize_semantic_vectors(self):
        """Initialize lightweight semantic similarity system"""
        try:
            # Sample queries for TF-IDF training (minimal memory)
            sample_queries = [
                "where is hagia sophia",
                "best restaurants in istanbul",
                "how to get to topkapi palace",
                "things to do in taksim",
                "where to stay in sultanahmet",
                "turkish food recommendations",
                "bosphorus cruise tour",
                "grand bazaar shopping",
                "istanbul nightlife",
                "metro to airport"
            ]
            
            self.vectorizer = TfidfVectorizer(
                max_features=100,  # Keep it small!
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.vectorizer.fit(sample_queries)
            
            logger.info("✅ Semantic vectors initialized (lightweight TF-IDF)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize semantic vectors: {e}")
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> LightweightNeuralInsights:
        """
        Process query with lightweight neural enhancements
        
        Args:
            query: User query text
            context: Optional context information
            
        Returns:
            LightweightNeuralInsights with processing results
        """
        start_time = datetime.now()
        
        try:
            # Extract entities (fast rule-based + spaCy if available)
            entities = await self._extract_entities(query)
            
            # Classify intent (keyword-based, very fast)
            intent, confidence = await self._classify_intent(query)
            
            # Extract semantic features (TF-IDF)
            semantic_features = await self._extract_semantic_features(query)
            
            # Analyze sentiment (lightweight)
            sentiment, sentiment_score = await self._analyze_sentiment(query)
            
            # Extract keywords
            keywords = await self._extract_keywords(query)
            
            # Determine query complexity
            complexity = await self._determine_complexity(query, entities)
            
            # Extract location context
            location_context = await self._extract_location_context(query)
            
            # Extract temporal context
            temporal_context = await self._extract_temporal_context(query)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return LightweightNeuralInsights(
                entities=entities,
                intent=intent,
                intent_confidence=confidence,
                semantic_features=semantic_features,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                keywords=keywords,
                query_complexity=complexity,
                location_context=location_context,
                temporal_context=temporal_context,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return LightweightNeuralInsights()
    
    async def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities using rule-based patterns + spaCy"""
        entities = []
        query_lower = query.lower()
        
        # Rule-based extraction (very fast, no model needed)
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query_lower, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9,
                        'source': 'pattern'
                    })
        
        # SpaCy NER if available (still fast on small model)
        if self.nlp is not None:
            try:
                doc = self.nlp(query)
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'type': ent.label_.lower(),
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.85,
                        'source': 'spacy'
                    })
            except Exception as e:
                logger.debug(f"SpaCy processing error: {e}")
        
        # Deduplicate entities
        unique_entities = []
        seen_texts = set()
        for entity in entities:
            if entity['text'] not in seen_texts:
                unique_entities.append(entity)
                seen_texts.add(entity['text'])
        
        return unique_entities
    
    async def _classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify intent using keyword matching (no ML model needed!)"""
        query_lower = query.lower()
        intent_scores = {}
        
        # Score each intent based on keyword matches
        for intent, config in self.intent_keywords.items():
            score = 0.0
            keyword_count = 0
            
            for keyword in config['keywords']:
                if keyword in query_lower:
                    score += config['weight']
                    keyword_count += 1
            
            # Normalize score
            if keyword_count > 0:
                intent_scores[intent] = score / len(config['keywords'])
        
        # Get highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            return best_intent, min(confidence * 2, 1.0)  # Scale confidence
        
        return 'general_info', 0.5
    
    async def _extract_semantic_features(self, query: str) -> Dict[str, Any]:
        """Extract semantic features using TF-IDF"""
        features = {
            'query_length': len(query),
            'word_count': len(query.split()),
            'has_question': '?' in query,
            'is_imperative': query.lower().startswith(('find', 'show', 'get', 'tell', 'give'))
        }
        
        if self.vectorizer is not None:
            try:
                # Get TF-IDF vector
                vector = self.vectorizer.transform([query])
                features['tfidf_score'] = float(vector.sum())
                
                # Get top features
                feature_names = self.vectorizer.get_feature_names_out()
                scores = vector.toarray()[0]
                top_indices = scores.argsort()[-5:][::-1]
                features['top_terms'] = [feature_names[i] for i in top_indices if scores[i] > 0]
            except Exception as e:
                logger.debug(f"TF-IDF extraction error: {e}")
        
        return features
    
    async def _analyze_sentiment(self, query: str) -> Tuple[str, float]:
        """Analyze sentiment using TextBlob"""
        if not TEXTBLOB_AVAILABLE or TextBlob is None:
            return 'neutral', 0.0
        
        try:
            blob = TextBlob(query)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return sentiment, polarity
        except Exception as e:
            logger.debug(f"Sentiment analysis error: {e}")
            return 'neutral', 0.0
    
    async def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords"""
        # Simple keyword extraction (remove stop words)
        stop_words = {'a', 'an', 'the', 'is', 'are', 'in', 'to', 'of', 'for', 'on', 'at', 'by'}
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:10]  # Top 10 keywords
    
    async def _determine_complexity(self, query: str, entities: List[Dict]) -> str:
        """Determine query complexity"""
        word_count = len(query.split())
        entity_count = len(entities)
        
        if word_count < 5 and entity_count < 2:
            return 'simple'
        elif word_count < 15 and entity_count < 5:
            return 'moderate'
        else:
            return 'complex'
    
    async def _extract_location_context(self, query: str) -> Optional[str]:
        """Extract location context from query"""
        query_lower = query.lower()
        
        for location, keywords in self.location_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return location
        
        return None
    
    async def _extract_temporal_context(self, query: str) -> Optional[str]:
        """Extract temporal context from query"""
        query_lower = query.lower()
        
        for time_period, keywords in self.temporal_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return time_period
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'mode': 'lightweight',
            'memory_efficient': True,
            'cpu_optimized': True,
            'spacy_available': SPACY_AVAILABLE,
            'textblob_available': TEXTBLOB_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'estimated_memory_mb': '<500MB',
            'estimated_latency_ms': '<100ms'
        }


# Global instance
_lightweight_processor = None

def get_lightweight_neural_processor() -> LightweightNeuralProcessor:
    """Get or create lightweight neural processor instance"""
    global _lightweight_processor
    if _lightweight_processor is None:
        _lightweight_processor = LightweightNeuralProcessor()
    return _lightweight_processor
