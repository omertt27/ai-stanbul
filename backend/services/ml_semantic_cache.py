"""
ML-Powered Semantic Cache System for AI Istanbul
Uses FAISS for high-performance similarity search and embeddings
Targets 70-80% cache hit rate to replace GPT dependency
"""

import os
import json
import pickle
import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path

# ML and embedding libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ SentenceTransformers not available. Install with: pip install sentence-transformers")

# Fallback to sklearn if advanced ML not available
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class CachedResponse:
    """Enhanced cached response with ML metadata"""
    query_id: str
    original_query: str
    normalized_query: str
    response: str
    query_type: str
    language: str
    context: Dict[str, Any]
    embedding: np.ndarray
    
    # Usage statistics
    usage_count: int = 1
    last_used: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Quality metrics
    user_satisfaction: float = 0.8  # 0-1 score
    response_time_ms: float = 0.0
    confidence_score: float = 0.9
    
    # ML metadata
    cluster_id: Optional[str] = None
    template_used: Optional[str] = None
    similarity_threshold: float = 0.85
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'query_id': self.query_id,
            'original_query': self.original_query,
            'normalized_query': self.normalized_query,
            'response': self.response,
            'query_type': self.query_type,
            'language': self.language,
            'context': self.context,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat(),
            'created_at': self.created_at.isoformat(),
            'user_satisfaction': self.user_satisfaction,
            'response_time_ms': self.response_time_ms,
            'confidence_score': self.confidence_score,
            'cluster_id': self.cluster_id,
            'template_used': self.template_used,
            'similarity_threshold': self.similarity_threshold
        }

class MLSemanticCache:
    """
    Advanced ML-powered semantic cache using FAISS for vector similarity search
    Integrates with existing AI Istanbul components for optimal performance
    """
    
    def __init__(self, cache_dir: str = "cache_data", embedding_model: str = "all-MiniLM-L6-v2"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache storage
        self.cached_responses: Dict[str, CachedResponse] = {}
        self.query_embeddings: List[np.ndarray] = []
        self.query_ids: List[str] = []
        
        # ML Components
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.faiss_index = None
        self.embedding_dim = 384  # Default for MiniLM
        
        # Fallback components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
        self.tfidf_fitted = False
        
        # Performance settings
        self.similarity_threshold = 0.75
        self.max_cache_size = 10000
        self.cleanup_interval_hours = 24
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'embeddings_generated': 0,
            'last_cleanup': datetime.now()
        }
        
        # Initialize ML components
        self._initialize_ml_components()
        self._load_cache_from_disk()
    
    def _initialize_ml_components(self):
        """Initialize ML models and FAISS index"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"✅ Loaded embedding model with dimension: {self.embedding_dim}")
            else:
                logger.warning("⚠️ Using TF-IDF fallback for embeddings")
            
            if FAISS_AVAILABLE:
                # Initialize FAISS index for fast similarity search
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                logger.info("✅ FAISS index initialized")
            else:
                logger.warning("⚠️ FAISS not available, using sklearn fallback")
                
        except Exception as e:
            logger.error(f"❌ Error initializing ML components: {e}")
            logger.info("📦 Install requirements: pip install faiss-cpu sentence-transformers")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better matching"""
        # Basic normalization
        normalized = query.lower().strip()
        
        # Remove common variations
        replacements = {
            "how can i get to": "how to get to",
            "how do i go to": "how to get to", 
            "what's the best way to": "how to get to",
            "can you tell me about": "tell me about",
            "i want to know about": "tell me about",
            "what are the": "list",
            "show me": "list",
            "recommend": "suggest",
            "ayasofya": "hagia sophia",
            "sultanahmet camii": "blue mosque",
            "kapalıçarşı": "grand bazaar"
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the best available method"""
        try:
            if self.embedding_model:
                # Use SentenceTransformers for high-quality embeddings
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                self.stats['embeddings_generated'] += 1
                return embedding
            else:
                # Fallback to TF-IDF
                if not self.tfidf_fitted:
                    # Bootstrap with common Istanbul queries
                    bootstrap_queries = [
                        "how to get to hagia sophia",
                        "best restaurants in sultanahmet",
                        "blue mosque opening hours", 
                        "galata tower tickets",
                        "istanbul metro map",
                        "bosphorus cruise prices",
                        "grand bazaar shopping",
                        "turkish breakfast places"
                    ]
                    self.tfidf_vectorizer.fit(bootstrap_queries + [text])
                    self.tfidf_fitted = True
                
                embedding = self.tfidf_vectorizer.transform([text]).toarray()[0]
                self.stats['embeddings_generated'] += 1
                return embedding
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def _generate_query_id(self, query: str, context: Dict = None) -> str:
        """Generate unique ID for query"""
        content = f"{query}_{json.dumps(context or {}, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_to_cache(self, query: str, response: str, query_type: str, 
                    language: str, context: Dict = None, template_used: str = None) -> str:
        """Add query-response pair to semantic cache"""
        try:
            # Generate components
            normalized_query = self._normalize_query(query)
            query_id = self._generate_query_id(normalized_query, context)
            embedding = self._generate_embedding(normalized_query)
            
            # Create cached response
            cached_response = CachedResponse(
                query_id=query_id,
                original_query=query,
                normalized_query=normalized_query,
                response=response,
                query_type=query_type,
                language=language,
                context=context or {},
                embedding=embedding,
                template_used=template_used,
                created_at=datetime.now()
            )
            
            # Store in memory
            self.cached_responses[query_id] = cached_response
            
            # Add to FAISS index if available
            if self.faiss_index is not None:
                # Normalize embedding for cosine similarity
                norm_embedding = embedding / np.linalg.norm(embedding)
                self.faiss_index.add(norm_embedding.reshape(1, -1))
                self.query_ids.append(query_id)
                self.query_embeddings.append(norm_embedding)
            
            # Cleanup if cache is getting too large
            if len(self.cached_responses) > self.max_cache_size:
                self._cleanup_cache()
            
            logger.info(f"✅ Added to cache: {query[:50]}... -> {len(response)} chars")
            return query_id
            
        except Exception as e:
            logger.error(f"❌ Error adding to cache: {e}")
            return ""
    
    def cache_response(self, query: str, response: str, query_type: str = 'general', 
                      language: str = 'en', context: Dict = None) -> str:
        """Alias for add_to_cache for compatibility"""
        return self.add_to_cache(query, response, query_type, language, context)
    
    def search_cache(self, query: str, context: Dict = None, 
                    top_k: int = 5) -> List[Tuple[CachedResponse, float]]:
        """Search cache for similar queries"""
        try:
            self.stats['total_queries'] += 1
            
            # Generate embedding for search query
            normalized_query = self._normalize_query(query)
            query_embedding = self._generate_embedding(normalized_query)
            
            if self.faiss_index is not None and len(self.query_ids) > 0:
                # Use FAISS for fast similarity search
                norm_embedding = query_embedding / np.linalg.norm(query_embedding)
                
                # Search FAISS index
                similarities, indices = self.faiss_index.search(
                    norm_embedding.reshape(1, -1), 
                    min(top_k, len(self.query_ids))
                )
                
                # Get results
                results = []
                for similarity, idx in zip(similarities[0], indices[0]):
                    if idx >= 0 and idx < len(self.query_ids):
                        query_id = self.query_ids[idx]
                        if query_id in self.cached_responses:
                            cached_response = self.cached_responses[query_id]
                            # Add context similarity bonus
                            context_bonus = self._calculate_context_similarity(
                                context or {}, cached_response.context
                            )
                            total_similarity = float(similarity) + (context_bonus * 0.1)
                            results.append((cached_response, total_similarity))
                
                # Sort by similarity
                results.sort(key=lambda x: x[1], reverse=True)
                return results
            
            else:
                # Fallback to manual similarity calculation
                results = []
                for cached_response in self.cached_responses.values():
                    # Calculate similarity
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        cached_response.embedding.reshape(1, -1)
                    )[0][0]
                    
                    # Add context similarity bonus
                    context_bonus = self._calculate_context_similarity(
                        context or {}, cached_response.context
                    )
                    total_similarity = float(similarity) + (context_bonus * 0.1)
                    
                    if total_similarity >= self.similarity_threshold:
                        results.append((cached_response, total_similarity))
                
                # Sort and return top results
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
        
        except Exception as e:
            logger.error(f"❌ Error searching cache: {e}")
            return []
    
    def get_cached_response(self, query: str, context: Dict = None, 
                          user_id: str = None) -> Optional[Tuple[str, float, Dict]]:
        """Get best cached response for query"""
        try:
            # Search for similar queries
            results = self.search_cache(query, context, top_k=1)
            
            if results:
                cached_response, similarity = results[0]
                
                if similarity >= self.similarity_threshold:
                    # Update usage statistics
                    cached_response.usage_count += 1
                    cached_response.last_used = datetime.now()
                    
                    # Enhance response with current context
                    enhanced_response = self._enhance_response_with_context(
                        cached_response.response, query, context, user_id
                    )
                    
                    self.stats['cache_hits'] += 1
                    
                    metadata = {
                        'source': 'ml_semantic_cache',
                        'similarity': similarity,
                        'original_query': cached_response.original_query,
                        'query_type': cached_response.query_type,
                        'template_used': cached_response.template_used,
                        'usage_count': cached_response.usage_count,
                        'confidence': cached_response.confidence_score
                    }
                    
                    logger.info(f"🎯 Cache HIT: {similarity:.3f} similarity for '{query[:50]}...'")
                    return enhanced_response, similarity, metadata
            
            self.stats['cache_misses'] += 1
            logger.info(f"❌ Cache MISS for '{query[:50]}...'")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting cached response: {e}")
            return None
    
    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between contexts"""
        if not context1 or not context2:
            return 0.0
        
        # Check for common keys
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        # Calculate matches
        matches = sum(1 for key in common_keys if context1[key] == context2[key])
        return matches / len(common_keys)
    
    def _enhance_response_with_context(self, response: str, current_query: str, 
                                     context: Dict, user_id: str) -> str:
        """Enhance cached response with current context"""
        enhanced = response
        
        # Add personalization if user context available
        if context and context.get('user_preferences'):
            preferences = context['user_preferences']
            if 'interests' in preferences:
                interests = preferences['interests']
                if 'historical' in interests:
                    enhanced += "\n\n🏛️ *Since you're interested in history, you might also enjoy the Istanbul Archaeological Museums nearby!*"
                elif 'food' in interests:
                    enhanced += "\n\n🍽️ *For food lovers like you, check out the local restaurants in this area!*"
        
        # Add current time-based enhancements
        current_hour = datetime.now().hour
        if 'opening' in current_query.lower() or 'hours' in current_query.lower():
            if current_hour < 9:
                enhanced += "\n\n⏰ *Currently closed - opens at 9:00 AM*"
            elif current_hour > 17:
                enhanced += "\n\n⏰ *Currently closed - opens tomorrow at 9:00 AM*"
        
        # Add location-based enhancements
        if context and context.get('location'):
            enhanced += f"\n\n📍 *Based on your current location, this is approximately {context.get('distance', 'nearby')}*"
        
        return enhanced
    
    def _cleanup_cache(self):
        """Clean up old and unused cache entries"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=30)
            
            # Find entries to remove
            entries_to_remove = []
            for query_id, cached_response in self.cached_responses.items():
                # Remove if old and rarely used
                if (cached_response.last_used < cutoff_time and 
                    cached_response.usage_count < 3):
                    entries_to_remove.append(query_id)
            
            # Remove from memory
            for query_id in entries_to_remove:
                del self.cached_responses[query_id]
            
            # Rebuild FAISS index (expensive but necessary)
            if self.faiss_index is not None and entries_to_remove:
                self._rebuild_faiss_index()
            
            self.stats['last_cleanup'] = current_time
            logger.info(f"🧹 Cleaned up {len(entries_to_remove)} cache entries")
            
        except Exception as e:
            logger.error(f"❌ Error during cache cleanup: {e}")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index after cleanup"""
        try:
            if self.faiss_index is not None:
                # Reset index
                self.faiss_index.reset()
                self.query_ids = []
                self.query_embeddings = []
                
                # Re-add all current entries
                for query_id, cached_response in self.cached_responses.items():
                    norm_embedding = cached_response.embedding / np.linalg.norm(cached_response.embedding)
                    self.faiss_index.add(norm_embedding.reshape(1, -1))
                    self.query_ids.append(query_id)
                    self.query_embeddings.append(norm_embedding)
                
                logger.info(f"🔄 Rebuilt FAISS index with {len(self.query_ids)} entries")
                
        except Exception as e:
            logger.error(f"❌ Error rebuilding FAISS index: {e}")
    
    def save_cache_to_disk(self):
        """Save cache to disk for persistence"""
        try:
            cache_file = self.cache_dir / "semantic_cache.pkl"
            index_file = self.cache_dir / "faiss_index.bin"
            stats_file = self.cache_dir / "cache_stats.json"
            
            # Save cached responses
            with open(cache_file, 'wb') as f:
                # Convert to serializable format
                serializable_cache = {}
                for query_id, cached_response in self.cached_responses.items():
                    data = cached_response.to_dict()
                    data['embedding'] = cached_response.embedding.tolist()  # Convert numpy to list
                    serializable_cache[query_id] = data
                
                pickle.dump(serializable_cache, f)
            
            # Save FAISS index
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(index_file))
            
            # Save query IDs and stats
            metadata = {
                'query_ids': self.query_ids,
                'stats': self.stats,
                'embedding_dim': self.embedding_dim,
                'similarity_threshold': self.similarity_threshold
            }
            
            # Convert datetime objects to strings
            if 'last_cleanup' in metadata['stats']:
                metadata['stats']['last_cleanup'] = metadata['stats']['last_cleanup'].isoformat()
            
            with open(stats_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Cache saved to disk: {len(self.cached_responses)} entries")
            
        except Exception as e:
            logger.error(f"❌ Error saving cache to disk: {e}")
    
    def _load_cache_from_disk(self):
        """Load cache from disk if available"""
        try:
            cache_file = self.cache_dir / "semantic_cache.pkl"
            index_file = self.cache_dir / "faiss_index.bin"
            stats_file = self.cache_dir / "cache_stats.json"
            
            if not cache_file.exists():
                logger.info("📦 No existing cache found, starting fresh")
                return
            
            # Load cached responses
            with open(cache_file, 'rb') as f:
                serializable_cache = pickle.load(f)
            
            # Convert back to CachedResponse objects
            for query_id, data in serializable_cache.items():
                # Convert embedding back to numpy array
                embedding = np.array(data['embedding'])
                
                # Parse datetime strings
                last_used = datetime.fromisoformat(data['last_used'])
                created_at = datetime.fromisoformat(data['created_at'])
                
                cached_response = CachedResponse(
                    query_id=data['query_id'],
                    original_query=data['original_query'],
                    normalized_query=data['normalized_query'],
                    response=data['response'],
                    query_type=data['query_type'],
                    language=data['language'],
                    context=data['context'],
                    embedding=embedding,
                    usage_count=data['usage_count'],
                    last_used=last_used,
                    created_at=created_at,
                    user_satisfaction=data['user_satisfaction'],
                    response_time_ms=data['response_time_ms'],
                    confidence_score=data['confidence_score'],
                    cluster_id=data.get('cluster_id'),
                    template_used=data.get('template_used'),
                    similarity_threshold=data.get('similarity_threshold', 0.85)
                )
                
                self.cached_responses[query_id] = cached_response
            
            # Load FAISS index
            if index_file.exists() and FAISS_AVAILABLE:
                self.faiss_index = faiss.read_index(str(index_file))
            
            # Load metadata
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    metadata = json.load(f)
                
                self.query_ids = metadata.get('query_ids', [])
                self.stats = metadata.get('stats', self.stats)
                self.embedding_dim = metadata.get('embedding_dim', self.embedding_dim)
                self.similarity_threshold = metadata.get('similarity_threshold', self.similarity_threshold)
                
                # Parse datetime
                if 'last_cleanup' in self.stats:
                    self.stats['last_cleanup'] = datetime.fromisoformat(self.stats['last_cleanup'])
            
            logger.info(f"📂 Loaded cache from disk: {len(self.cached_responses)} entries")
            
        except Exception as e:
            logger.error(f"❌ Error loading cache from disk: {e}")
            # Continue with empty cache if loading fails
            self.cached_responses = {}
            self.query_ids = []
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if self.stats['total_queries'] > 0:
            hit_rate = (self.stats['cache_hits'] / self.stats['total_queries']) * 100
        else:
            hit_rate = 0.0
        
        return {
            'total_entries': len(self.cached_responses),
            'total_queries': self.stats['total_queries'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'embeddings_generated': self.stats['embeddings_generated'],
            'last_cleanup': self.stats['last_cleanup'].isoformat(),
            'embedding_model': self.embedding_model_name,
            'similarity_threshold': self.similarity_threshold,
            'faiss_available': FAISS_AVAILABLE,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'cache_size_mb': len(pickle.dumps(self.cached_responses)) / (1024 * 1024),
            'avg_usage_per_entry': sum(cr.usage_count for cr in self.cached_responses.values()) / len(self.cached_responses) if self.cached_responses else 0
        }
    
    def update_response_quality(self, query_id: str, satisfaction_score: float, 
                              response_time_ms: float = None):
        """Update quality metrics for a cached response"""
        if query_id in self.cached_responses:
            cached_response = self.cached_responses[query_id]
            cached_response.user_satisfaction = satisfaction_score
            if response_time_ms:
                cached_response.response_time_ms = response_time_ms
            logger.info(f"📊 Updated quality metrics for {query_id}: satisfaction={satisfaction_score}")
    
    def export_training_data(self, output_file: str = None) -> List[Dict]:
        """Export cache data for model training or analysis"""
        if not output_file:
            output_file = self.cache_dir / "training_data.json"
        
        training_data = []
        for cached_response in self.cached_responses.values():
            training_data.append({
                'query': cached_response.original_query,
                'normalized_query': cached_response.normalized_query,
                'response': cached_response.response,
                'query_type': cached_response.query_type,
                'language': cached_response.language,
                'context': cached_response.context,
                'usage_count': cached_response.usage_count,
                'satisfaction': cached_response.user_satisfaction,
                'template_used': cached_response.template_used
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📤 Exported {len(training_data)} training examples to {output_file}")
        return training_data

# Integration with existing query router
def integrate_ml_cache_with_router():
    """Integration example with existing query router"""
    
    from .query_router import IndustryQueryRouter
    from .template_engine import TemplateEngine
    from .recommendation_engine import RecommendationEngine
    
    class MLCachedQueryRouter(IndustryQueryRouter):
        """Enhanced query router with ML semantic caching"""
        
        def __init__(self):
            super().__init__()
            self.ml_cache = MLSemanticCache()
            
        def process_query_with_ml_cache(self, query: str, user_id: str = "anonymous", 
                                      context: Dict = None) -> Dict[str, Any]:
            """Process query with ML cache first, then fallback to original processing"""
            start_time = time.time()
            
            # Step 1: Check ML semantic cache
            cached_result = self.ml_cache.get_cached_response(query, context, user_id)
            
            if cached_result:
                response, similarity, metadata = cached_result
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    'response': response,
                    'source': 'ml_semantic_cache',
                    'similarity': similarity,
                    'confidence': metadata['confidence'],
                    'processing_time_ms': processing_time,
                    'metadata': metadata,
                    'cache_hit': True
                }
            
            # Step 2: Process normally and add to cache
            classification = self.classify_query(query, context)
            
            # Generate response using existing systems
            if hasattr(self, 'template_engine') and self.template_engine:
                response = self.template_engine.generate_response(
                    classification.query_type,
                    classification.extracted_entities,
                    classification.language,
                    context
                )
            else:
                response = f"I understand you're asking about {classification.query_type.value}. Here's what I can help you with..."
            
            # Add to ML cache for future use
            query_id = self.ml_cache.add_to_cache(
                query=query,
                response=response,
                query_type=classification.query_type.value,
                language=classification.language,
                context=context,
                template_used="generated"
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'response': response,
                'source': 'template_engine',
                'confidence': classification.confidence,
                'processing_time_ms': processing_time,
                'classification': classification.query_type.value,
                'cached_for_future': True,
                'cache_hit': False,
                'query_id': query_id
            }

if __name__ == "__main__":
    # Example usage and testing
    print("🚀 Initializing ML Semantic Cache...")
    cache = MLSemanticCache()
    
    # Add some sample queries
    sample_data = [
        ("How to get to Hagia Sophia?", "Take the T1 tram to Sultanahmet station. Hagia Sophia is a 2-minute walk from the station.", "transport", "english"),
        ("What are the opening hours of Blue Mosque?", "Blue Mosque is open daily from 9:00 AM to 6:00 PM, except during prayer times.", "attraction_info", "english"),
        ("Best restaurants in Sultanahmet", "Top restaurants in Sultanahmet: 1. Pandeli (Ottoman cuisine), 2. Seasons Restaurant (fine dining), 3. Deraliye (palace cuisine)", "restaurant_search", "english"),
        ("Ayasofya'ya nasıl giderim?", "Sultanahmet durağına T1 tramvayı ile gidin. Ayasofya istasyondan 2 dakika yürüme mesafesinde.", "transport", "turkish")
    ]
    
    print("\n📝 Adding sample data to cache...")
    for query, response, query_type, language in sample_data:
        cache.add_to_cache(query, response, query_type, language)
    
    # Test cache retrieval
    test_queries = [
        "How do I get to Hagia Sophia?",  # Similar to cached query
        "Ayasofya nasıl gidilir?",        # Similar to Turkish query
        "Blue Mosque hours",              # Similar to opening hours query
        "Completely new query about something else"  # Should not match
    ]
    
    print("\n🎯 Testing cache retrieval...")
    for test_query in test_queries:
        result = cache.get_cached_response(test_query)
        if result:
            response, similarity, metadata = result
            print(f"✅ '{test_query}' -> MATCH (similarity: {similarity:.3f})")
            print(f"   Response: {response[:100]}...")
        else:
            print(f"❌ '{test_query}' -> NO MATCH")
    
    # Show statistics
    print("\n📊 Cache Statistics:")
    stats = cache.get_cache_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Save cache
    cache.save_cache_to_disk()
    print("\n💾 Cache saved to disk!")
