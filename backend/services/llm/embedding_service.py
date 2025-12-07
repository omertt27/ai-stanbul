"""
Semantic Embedding Service for Signal Detection
Phase 4 - Priority 1: Real Semantic Embeddings Integration

This module provides sentence embeddings for improved semantic signal detection,
replacing the template-based approach with real embedding models.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Provides semantic embeddings for intent classification.
    
    Features:
    - Real sentence embeddings (not just templates)
    - Multi-language support
    - Intent classification with confidence scores
    - Caching for performance
    - Fallback to simple similarity for offline mode
    """
    
    def __init__(self, model_name: str = 'lightweight', cache_dir: Optional[str] = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: 'lightweight' (fast) or 'high_quality' (accurate)
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '.embedding_cache')
        self.model = None
        self.intent_embeddings = {}
        self.embedding_cache = {}
        self.offline_mode = False
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model
        self._load_model()
        
        # Load intent examples
        self._load_intent_examples()
        
        logger.info(f"✅ EmbeddingService initialized (mode: {self.model_name}, offline: {self.offline_mode})")
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            if self.model_name == 'lightweight':
                # Fast, good for production (384 dimensions, ~80MB)
                model_id = 'all-MiniLM-L6-v2'
            elif self.model_name == 'high_quality':
                # Better quality, slower (768 dimensions, ~420MB)
                model_id = 'all-mpnet-base-v2'
            else:
                model_id = 'all-MiniLM-L6-v2'
            
            self.model = SentenceTransformer(model_id)
            logger.info(f"✅ Loaded embedding model: {model_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load SentenceTransformer: {e}")
            logger.warning("⚠️ Falling back to offline mode (simple similarity)")
            self.offline_mode = True
            self.model = None
    
    def _load_intent_examples(self):
        """Load intent training examples and compute embeddings."""
        
        # Intent examples database (50-100 examples per intent)
        intent_examples = {
            'needs_restaurant': [
                # Direct requests
                "where can I eat",
                "find a restaurant",
                "I'm hungry",
                "looking for food",
                "place to eat",
                "find food nearby",
                "restaurant recommendations",
                "good place for lunch",
                "dinner options",
                "breakfast spot",
                "where to eat",
                "food around here",
                "hungry need food",
                "want to eat something",
                "grab a bite",
                
                # Implicit/conversational
                "something to eat",
                "place for food",
                "anywhere to eat",
                "somewhere for dinner",
                "lunch place",
                "breakfast place",
                "dining options",
                "eatery nearby",
                "café or restaurant",
                "food spot",
                
                # Cuisine-specific
                "turkish restaurant",
                "seafood place",
                "italian food",
                "kebab shop",
                "sushi restaurant",
                "vegetarian restaurant",
                "vegan options",
                "asian food",
                "pizza place",
                "burger joint",
                
                # Multi-language variations
                "restoran önerisi",  # Turkish
                "nerede yemek yenir",  # Turkish
                "où manger",  # French
                "wo kann ich essen",  # German
                "dónde comer",  # Spanish
                "أين يمكنني أن آكل",  # Arabic
                
                # Context variations
                "meal nearby",
                "place to dine",
                "eating establishment",
                "culinary options",
                "food venue",
            ],
            
            'needs_attraction': [
                # Direct requests
                "what to see",
                "tourist attractions",
                "visit museums",
                "historical places",
                "sightseeing",
                "places to visit",
                "landmarks",
                "monuments",
                "cultural sites",
                "tourist spots",
                "things to do",
                "attractions nearby",
                "famous places",
                "must-see places",
                "interesting places",
                
                # Implicit
                "what's interesting here",
                "anything to see",
                "worth visiting",
                "popular spots",
                "notable places",
                "scenic locations",
                "nice places to visit",
                "beautiful spots",
                
                # Specific types
                "art galleries",
                "history museum",
                "ancient ruins",
                "palace tour",
                "castle visit",
                "temple tour",
                "mosque visit",
                "church tour",
                "archaeological site",
                "heritage site",
                
                # Multi-language
                "gezilecek yerler",  # Turkish
                "müze önerisi",  # Turkish
                "que visiter",  # French
                "sehenswürdigkeiten",  # German
                "qué visitar",  # Spanish
                "أماكن للزيارة",  # Arabic
            ],
            
            'needs_transportation': [
                # Direct routing
                "how to get to",
                "directions to",
                "route to",
                "way to",
                "navigate to",
                "travel to",
                "go to",
                "reach",
                "find way",
                "path to",
                
                # Transportation modes
                "take bus",
                "metro route",
                "tram line",
                "ferry schedule",
                "taxi to",
                "walk to",
                "drive to",
                "public transport",
                "transportation options",
                
                # Implicit
                "get me there",
                "how far is",
                "distance to",
                "travel time",
                "which line",
                "which bus",
                "best route",
                "fastest way",
                "cheapest way",
                
                # Istanbul-specific
                "dolmuş to",
                "vapur to",
                "marmaray route",
                "metrobüs line",
                "iskele nerede",  # Turkish: where is pier
                
                # Multi-language
                "nasıl gidilir",  # Turkish
                "comment y aller",  # French
                "wie komme ich",  # German
                "cómo llegar",  # Spanish
                "كيف أصل",  # Arabic
            ],
            
            'needs_nearby': [
                # Explicit nearby
                "near me",
                "nearby",
                "around here",
                "close to me",
                "close by",
                "in the area",
                "around",
                "vicinity",
                "surrounding",
                "neighborhood",
                
                # Implicit proximity
                "what's here",
                "anything here",
                "options here",
                "this area",
                "local",
                "walking distance",
                "within reach",
                
                # Multi-language
                "yakınımda",  # Turkish
                "civarında",  # Turkish
                "près de moi",  # French
                "in der nähe",  # German
                "cerca de mí",  # Spanish
                "بالقرب مني",  # Arabic
            ],
            
            'needs_neighborhood': [
                # Direct requests
                "neighborhood info",
                "area information",
                "about this area",
                "district guide",
                "locality details",
                "quarter information",
                
                # Implicit
                "what's this place like",
                "tell me about here",
                "this area",
                "around here",
                
                # Specific neighborhoods
                "about Beyoğlu",
                "Kadıköy area",
                "Sultanahmet district",
                "Taksim neighborhood",
                
                # Multi-language
                "semt hakkında",  # Turkish
                "mahalle bilgisi",  # Turkish
                "quartier",  # French
                "viertel",  # German
                "barrio",  # Spanish
            ],
            
            'needs_events': [
                # Direct requests
                "events today",
                "what's happening",
                "activities",
                "shows",
                "concerts",
                "festivals",
                "exhibitions",
                "performances",
                
                # Implicit
                "anything going on",
                "what to do tonight",
                "fun activities",
                "entertainment",
                
                # Multi-language
                "etkinlikler",  # Turkish
                "ne var ne yok",  # Turkish
                "événements",  # French
                "veranstaltungen",  # German
                "eventos",  # Spanish
            ],
        }
        
        # Compute embeddings for all examples
        if not self.offline_mode and self.model:
            for intent, examples in intent_examples.items():
                try:
                    embeddings = self.model.encode(examples, convert_to_numpy=True)
                    self.intent_embeddings[intent] = {
                        'examples': examples,
                        'embeddings': embeddings,
                        'mean_embedding': np.mean(embeddings, axis=0)
                    }
                    logger.debug(f"Loaded {len(examples)} examples for {intent}")
                except Exception as e:
                    logger.error(f"Error encoding examples for {intent}: {e}")
        else:
            # Offline mode: just store examples
            for intent, examples in intent_examples.items():
                self.intent_embeddings[intent] = {
                    'examples': examples,
                    'embeddings': None,
                    'mean_embedding': None
                }
        
        logger.info(f"✅ Loaded intent examples for {len(self.intent_embeddings)} intents")
    
    def encode(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text to embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector or None if offline
        """
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        if self.offline_mode or not self.model:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None
    
    def encode_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Encode multiple texts efficiently.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings or None if offline
        """
        if self.offline_mode or not self.model:
            return None
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Cache individual embeddings
            for text, embedding in zip(texts, embeddings):
                self.embedding_cache[text] = embedding
            
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            return None
    
    def classify_intent(
        self,
        query: str,
        threshold: float = 0.65,
        top_k: int = 3
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Classify query intent using semantic similarity.
        
        Args:
            query: Query text
            threshold: Minimum similarity score
            top_k: Number of top examples to consider for scoring
            
        Returns:
            Dict of {intent_name: (detected, confidence)}
        """
        if self.offline_mode or not self.model:
            # Fallback to simple keyword matching
            return self._classify_intent_offline(query, threshold)
        
        # Encode query
        query_embedding = self.encode(query)
        if query_embedding is None:
            return {}
        
        results = {}
        
        for intent_name, intent_data in self.intent_embeddings.items():
            if intent_data['embeddings'] is None:
                continue
            
            # Compute cosine similarity with all examples
            similarities = self._cosine_similarity_batch(
                query_embedding,
                intent_data['embeddings']
            )
            
            # Scoring strategy: combine max similarity with top-k average
            max_sim = np.max(similarities)
            top_k_sims = np.sort(similarities)[-top_k:]
            top_k_avg = np.mean(top_k_sims)
            
            # Weighted combination (favor strong matches but consider multiple)
            confidence = 0.6 * max_sim + 0.4 * top_k_avg
            
            # Check threshold
            detected = confidence >= threshold
            
            results[intent_name] = (detected, float(confidence))
            
            if detected:
                logger.debug(
                    f"Intent detected: {intent_name} "
                    f"(conf: {confidence:.3f}, max: {max_sim:.3f}, avg: {top_k_avg:.3f})"
                )
        
        return results
    
    def _classify_intent_offline(
        self,
        query: str,
        threshold: float = 0.65
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Fallback intent classification using simple keyword matching.
        
        This is used when the embedding model is not available.
        """
        query_lower = query.lower()
        results = {}
        
        for intent_name, intent_data in self.intent_embeddings.items():
            # Count matching keywords from examples
            matches = 0
            total_keywords = len(intent_data['examples'])
            
            for example in intent_data['examples']:
                # Simple word overlap check
                example_words = set(example.lower().split())
                query_words = set(query_lower.split())
                
                overlap = len(example_words & query_words)
                if overlap > 0:
                    matches += overlap / len(example_words)
            
            # Normalize score
            confidence = min(1.0, matches / max(1, total_keywords * 0.1))
            detected = confidence >= threshold
            
            results[intent_name] = (detected, confidence)
        
        return results
    
    def _cosine_similarity_batch(
        self,
        query_embedding: np.ndarray,
        example_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and multiple examples.
        
        Args:
            query_embedding: Single query embedding
            example_embeddings: Array of example embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        examples_norm = example_embeddings / np.linalg.norm(
            example_embeddings, axis=1, keepdims=True
        )
        
        # Dot product gives cosine similarity for normalized vectors
        similarities = np.dot(examples_norm, query_norm)
        
        return similarities
    
    def get_most_similar_examples(
        self,
        query: str,
        intent: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get most similar examples for an intent.
        
        Args:
            query: Query text
            intent: Intent name
            top_k: Number of examples to return
            
        Returns:
            List of (example, similarity) tuples
        """
        if intent not in self.intent_embeddings:
            return []
        
        if self.offline_mode or not self.model:
            # Just return first few examples
            return [(ex, 0.0) for ex in self.intent_embeddings[intent]['examples'][:top_k]]
        
        query_embedding = self.encode(query)
        if query_embedding is None:
            return []
        
        intent_data = self.intent_embeddings[intent]
        if intent_data['embeddings'] is None:
            return []
        
        similarities = self._cosine_similarity_batch(
            query_embedding,
            intent_data['embeddings']
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (intent_data['examples'][i], float(similarities[i]))
            for i in top_indices
        ]
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """
        Get service health status.
        
        Returns:
            Health status dictionary
        """
        return {
            'status': 'online' if not self.offline_mode else 'offline',
            'model': self.model_name,
            'intents_loaded': len(self.intent_embeddings),
            'cache_size': len(self.embedding_cache),
            'model_available': self.model is not None
        }


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(
    model_name: str = 'lightweight',
    force_reload: bool = False
) -> EmbeddingService:
    """
    Get singleton embedding service instance.
    
    Args:
        model_name: Model to use ('lightweight' or 'high_quality')
        force_reload: Force reload of service
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None or force_reload:
        _embedding_service = EmbeddingService(model_name=model_name)
    
    return _embedding_service
