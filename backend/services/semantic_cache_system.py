"""
Enhanced Semantic Cache System for AI Istanbul
Replaces GPT dependency with intelligent caching and retrieval
"""

import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

@dataclass
class CachedResponse:
    """Cached query-response pair with metadata"""
    query: str
    response: str
    query_type: str
    language: str
    context: Dict
    usage_count: int
    last_used: datetime
    created_at: datetime
    user_satisfaction: float  # 0-1 based on feedback
    semantic_embedding: np.ndarray

@dataclass
class QueryPattern:
    """User behavioral pattern"""
    pattern_id: str
    query_sequence: List[str]
    frequency: int
    success_rate: float
    avg_satisfaction: float
    next_likely_queries: List[str]

class SemanticCacheSystem:
    """
    Advanced semantic caching system to replace GPT dependency
    Integrates with existing AI Istanbul components
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.cache: Dict[str, CachedResponse] = {}
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.similarity_threshold = similarity_threshold
        self.is_trained = False
        
        # Knowledge graph for context enhancement
        self.knowledge_graph = self._build_istanbul_knowledge_graph()
        
        # User behavior tracking
        self.user_sessions: Dict[str, List[str]] = {}
        
    def _build_istanbul_knowledge_graph(self) -> Dict[str, Dict]:
        """Build Istanbul-specific knowledge relationships"""
        return {
            "hagia_sophia": {
                "type": "attraction",
                "district": "sultanahmet",
                "nearby": ["blue_mosque", "topkapi_palace", "grand_bazaar"],
                "recommended_time": "morning",
                "combine_with": ["blue_mosque", "sultanahmet_walk"],
                "transport": ["tram_sultanahmet", "walking"],
                "restaurants_nearby": ["pandeli", "seasons_restaurant"],
                "tags": ["historical", "religious", "unesco", "museum"]
            },
            "blue_mosque": {
                "type": "attraction", 
                "district": "sultanahmet",
                "nearby": ["hagia_sophia", "hippodrome", "grand_bazaar"],
                "recommended_time": "morning",
                "combine_with": ["hagia_sophia", "sultanahmet_walk"],
                "transport": ["tram_sultanahmet", "walking"],
                "restaurants_nearby": ["deraliye", "pandeli"],
                "tags": ["historical", "religious", "architecture", "free"]
            },
            "galata_tower": {
                "type": "attraction",
                "district": "beyoglu", 
                "nearby": ["karakoy", "istiklal_street", "galata_bridge"],
                "recommended_time": "sunset",
                "combine_with": ["istiklal_street", "karakoy_walk"],
                "transport": ["metro_karakoy", "funicular", "walking"],
                "restaurants_nearby": ["mikla", "galata_house"],
                "tags": ["panoramic", "historical", "paid", "tower"]
            },
            "grand_bazaar": {
                "type": "shopping",
                "district": "beyazit",
                "nearby": ["hagia_sophia", "blue_mosque", "spice_bazaar"],
                "recommended_time": "afternoon",
                "combine_with": ["spice_bazaar", "sultanahmet_walk"],
                "transport": ["tram_beyazit", "metro_vezneciler"],
                "restaurants_nearby": ["havuzlu_restaurant", "sark_kahvesi"],
                "tags": ["shopping", "historical", "covered", "souvenirs"]
            }
        }
    
    def add_to_cache(self, query: str, response: str, query_type: str, 
                    language: str, context: Dict = None):
        """Add a new query-response pair to semantic cache"""
        try:
            # Generate embedding
            embedding = self._get_query_embedding(query)
            
            cache_key = self._generate_cache_key(query, context)
            
            cached_response = CachedResponse(
                query=query,
                response=response,
                query_type=query_type,
                language=language,
                context=context or {},
                usage_count=1,
                last_used=datetime.now(),
                created_at=datetime.now(),
                user_satisfaction=0.8,  # Default assumption
                semantic_embedding=embedding
            )
            
            self.cache[cache_key] = cached_response
            logger.info(f"Added to cache: {query[:50]}... -> {len(response)} chars")
            
        except Exception as e:
            logger.error(f"Error adding to cache: {e}")
    
    def retrieve_from_cache(self, query: str, context: Dict = None, 
                          user_id: str = None) -> Optional[Tuple[str, float]]:
        """Retrieve similar response from cache"""
        try:
            # Track user session for behavioral patterns
            if user_id:
                self._track_user_query(user_id, query)
            
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            best_match = None
            best_similarity = 0.0
            
            # Search through cache for similar queries
            for cache_key, cached_response in self.cache.items():
                # Calculate semantic similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    cached_response.semantic_embedding.reshape(1, -1)
                )[0][0]
                
                # Context matching bonus
                context_bonus = self._calculate_context_similarity(
                    context or {}, cached_response.context
                )
                
                total_similarity = similarity + (context_bonus * 0.1)
                
                if total_similarity > best_similarity and total_similarity >= self.similarity_threshold:
                    best_similarity = total_similarity
                    best_match = cached_response
            
            if best_match:
                # Update usage statistics
                best_match.usage_count += 1
                best_match.last_used = datetime.now()
                
                # Enhance response with behavioral predictions
                enhanced_response = self._enhance_with_behavioral_patterns(
                    best_match.response, user_id, query
                )
                
                logger.info(f"Cache hit: {best_similarity:.3f} similarity")
                return enhanced_response, best_similarity
            
            logger.info("Cache miss")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query using TF-IDF"""
        if not self.is_trained:
            # Train on existing cache queries
            if self.cache:
                queries = [cr.query for cr in self.cache.values()]
                self.vectorizer.fit(queries)
                self.is_trained = True
            else:
                # Bootstrap with common Istanbul queries
                bootstrap_queries = [
                    "How to get to Hagia Sophia",
                    "Best restaurants in Sultanahmet", 
                    "Galata Tower opening hours",
                    "Blue Mosque visit times",
                    "Grand Bazaar shopping guide",
                    "Istanbul transport card",
                    "Bosphorus cruise times",
                    "Turkish breakfast places"
                ]
                self.vectorizer.fit(bootstrap_queries)
                self.is_trained = True
        
        return self.vectorizer.transform([query]).toarray()[0]
    
    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between contexts"""
        if not context1 or not context2:
            return 0.0
        
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for key in common_keys if context1[key] == context2[key])
        return matches / len(common_keys)
    
    def _generate_cache_key(self, query: str, context: Dict = None) -> str:
        """Generate unique cache key"""
        content = query
        if context:
            content += str(sorted(context.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _track_user_query(self, user_id: str, query: str):
        """Track user query patterns for behavioral prediction"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        
        self.user_sessions[user_id].append(query)
        
        # Keep only recent queries (last 10)
        self.user_sessions[user_id] = self.user_sessions[user_id][-10:]
        
        # Update behavioral patterns
        self._update_query_patterns(user_id)
    
    def _update_query_patterns(self, user_id: str):
        """Update query patterns based on user behavior"""
        queries = self.user_sessions.get(user_id, [])
        if len(queries) < 2:
            return
        
        # Look for patterns in recent queries
        for i in range(len(queries) - 1):
            current_query = queries[i]
            next_query = queries[i + 1]
            
            # Extract entities/intents for pattern matching
            current_entities = self._extract_entities(current_query)
            next_entities = self._extract_entities(next_query)
            
            # Update pattern frequency
            pattern_key = f"{current_entities['type']}_{current_entities.get('location', 'any')}"
            
            if pattern_key not in self.query_patterns:
                self.query_patterns[pattern_key] = QueryPattern(
                    pattern_id=pattern_key,
                    query_sequence=[current_query],
                    frequency=1,
                    success_rate=0.8,
                    avg_satisfaction=0.8,
                    next_likely_queries=[next_query]
                )
            else:
                pattern = self.query_patterns[pattern_key]
                pattern.frequency += 1
                if next_query not in pattern.next_likely_queries:
                    pattern.next_likely_queries.append(next_query)
    
    def _extract_entities(self, query: str) -> Dict:
        """Extract entities from query for pattern matching"""
        query_lower = query.lower()
        
        # Simple entity extraction (can be enhanced)
        entities = {"type": "general"}
        
        # Location entities
        locations = ["sultanahmet", "beyoglu", "taksim", "kadikoy", "galata", "eminonu"]
        for location in locations:
            if location in query_lower:
                entities["location"] = location
                break
        
        # Intent entities
        if any(word in query_lower for word in ["restaurant", "eat", "food", "lunch", "dinner"]):
            entities["type"] = "restaurant"
        elif any(word in query_lower for word in ["attraction", "visit", "see", "museum", "mosque"]):
            entities["type"] = "attraction"
        elif any(word in query_lower for word in ["transport", "metro", "bus", "get", "go"]):
            entities["type"] = "transport"
        elif any(word in query_lower for word in ["hotel", "stay", "accommodation"]):
            entities["type"] = "accommodation"
        
        return entities
    
    def _enhance_with_behavioral_patterns(self, response: str, user_id: str, 
                                        current_query: str) -> str:
        """Enhance response with behavioral predictions"""
        if not user_id or user_id not in self.user_sessions:
            return response
        
        # Get current entities
        current_entities = self._extract_entities(current_query)
        pattern_key = f"{current_entities['type']}_{current_entities.get('location', 'any')}"
        
        # Check if we have behavioral patterns for this type of query
        if pattern_key in self.query_patterns:
            pattern = self.query_patterns[pattern_key]
            
            # Add predictive suggestions to response
            if pattern.next_likely_queries:
                suggestions = pattern.next_likely_queries[:3]  # Top 3 suggestions
                
                response += "\n\n🔮 **You might also be interested in:**\n"
                for i, suggestion in enumerate(suggestions, 1):
                    response += f"{i}. {suggestion}\n"
        
        # Add knowledge graph enhancements
        response = self._enhance_with_knowledge_graph(response, current_entities)
        
        return response
    
    def _enhance_with_knowledge_graph(self, response: str, entities: Dict) -> str:
        """Enhance response using knowledge graph connections"""
        location = entities.get('location')
        query_type = entities.get('type')
        
        if location in self.knowledge_graph:
            location_data = self.knowledge_graph[location]
            
            # Add contextual suggestions based on knowledge graph
            if query_type == "attraction" and location_data.get('combine_with'):
                response += f"\n\n💡 **Perfect to combine with:** {', '.join(location_data['combine_with'])}"
            
            if location_data.get('recommended_time'):
                response += f"\n⏰ **Best time to visit:** {location_data['recommended_time']}"
            
            if location_data.get('nearby') and query_type == "restaurant":
                nearby_attractions = location_data['nearby'][:2]
                response += f"\n🏛️ **Nearby attractions:** {', '.join(nearby_attractions)}"
        
        return response
    
    def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics"""
        if not self.cache:
            return {"status": "empty"}
        
        total_entries = len(self.cache)
        total_usage = sum(cr.usage_count for cr in self.cache.values())
        avg_satisfaction = sum(cr.user_satisfaction for cr in self.cache.values()) / total_entries
        
        # Calculate hit rate (approximate)
        recent_usage = sum(1 for cr in self.cache.values() 
                          if (datetime.now() - cr.last_used).days <= 7)
        
        return {
            "total_entries": total_entries,
            "total_usage": total_usage,
            "average_satisfaction": avg_satisfaction,
            "recent_activity": recent_usage,
            "cache_size_mb": len(pickle.dumps(self.cache)) / (1024 * 1024),
            "patterns_learned": len(self.query_patterns)
        }
    
    def cleanup_cache(self, max_age_days: int = 30, min_usage: int = 2):
        """Clean up old, unused cache entries"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        keys_to_remove = []
        for key, cached_response in self.cache.items():
            if (cached_response.last_used < cutoff_date and 
                cached_response.usage_count < min_usage):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        logger.info(f"Cleaned up {len(keys_to_remove)} cache entries")
    
    def export_cache(self, filepath: str):
        """Export cache to disk"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'patterns': self.query_patterns,
                    'vectorizer': self.vectorizer
                }, f)
            logger.info(f"Cache exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting cache: {e}")
    
    def import_cache(self, filepath: str):
        """Import cache from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.cache = data.get('cache', {})
                self.query_patterns = data.get('patterns', {})
                self.vectorizer = data.get('vectorizer', self.vectorizer)
                self.is_trained = True
            logger.info(f"Cache imported from {filepath}")
        except Exception as e:
            logger.error(f"Error importing cache: {e}")

# Integration with existing query router
class GPTFreeQueryRouter:
    """Enhanced query router that eliminates GPT dependency"""
    
    def __init__(self):
        from .query_router import IndustryQueryRouter
        from .template_engine import TemplateEngine
        from .recommendation_engine import RecommendationEngine
        from .route_planner import RouteOptimizer
        
        self.base_router = IndustryQueryRouter()
        self.template_engine = TemplateEngine()
        self.recommendation_engine = RecommendationEngine()
        self.route_planner = RouteOptimizer()
        self.semantic_cache = SemanticCacheSystem()
        
    def process_query(self, query: str, user_id: str = "anonymous", 
                     context: Dict = None) -> Dict:
        """Process query without GPT dependency"""
        
        # Step 1: Check semantic cache first
        cached_result = self.semantic_cache.retrieve_from_cache(
            query, context, user_id
        )
        
        if cached_result:
            response, confidence = cached_result
            return {
                "response": response,
                "source": "semantic_cache",
                "confidence": confidence,
                "processing_time_ms": 10  # Very fast cache retrieval
            }
        
        # Step 2: Use existing classification and routing
        classification = self.base_router.classify_query(query, context)
        
        # Step 3: Generate response using template engine
        template_response = self.template_engine.generate_response(
            classification.query_type,
            classification.extracted_entities,
            classification.language,
            context
        )
        
        # Step 4: Enhance with recommendations if relevant
        if classification.query_type.value in ['attraction_search', 'restaurant_search']:
            user_profile = self.recommendation_engine.create_user_profile(
                context.get('user_preferences', {}) if context else {}
            )
            recommendations = self.recommendation_engine.get_advanced_recommendations(
                user_profile, n_recommendations=3, context=context
            )
            
            if recommendations:
                template_response += "\n\n🎯 **Personalized Suggestions:**\n"
                for i, rec in enumerate(recommendations[:3], 1):
                    template_response += f"{i}. {rec.name} ({rec.category})\n"
        
        # Step 5: Add to cache for future use
        self.semantic_cache.add_to_cache(
            query=query,
            response=template_response,
            query_type=classification.query_type.value,
            language=classification.language,
            context=context
        )
        
        return {
            "response": template_response,
            "source": "template_engine",
            "confidence": classification.confidence,
            "processing_time_ms": 50,  # Fast template generation
            "classification": classification.query_type.value
        }

if __name__ == "__main__":
    # Example usage
    cache_system = SemanticCacheSystem()
    
    # Add some sample responses to cache
    sample_queries = [
        ("How to get to Hagia Sophia?", "Take the T1 tram to Sultanahmet station. Hagia Sophia is a 2-minute walk from the station.", "transport", "english"),
        ("Best restaurants in Sultanahmet", "Top restaurants: 1. Pandeli (Ottoman cuisine), 2. Seasons Restaurant (fine dining), 3. Deraliye (palace cuisine)", "restaurant", "english"),
        ("Ayasofya'ya nasıl giderim?", "Sultanahmet durağına T1 tramvayı ile gidin. Ayasofya istasyondan 2 dakika yürüme mesafesinde.", "transport", "turkish")
    ]
    
    for query, response, query_type, language in sample_queries:
        cache_system.add_to_cache(query, response, query_type, language)
    
    # Test retrieval
    result = cache_system.retrieve_from_cache("How do I get to Hagia Sophia?")
    if result:
        response, similarity = result
        print(f"Found similar response (similarity: {similarity:.3f}): {response}")
    
    # Show statistics
    stats = cache_system.get_cache_statistics()
    print(f"Cache statistics: {stats}")
