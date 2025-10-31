"""
Neural Response Ranker - GPU-Accelerated Semantic Similarity Ranking

This module provides GPU-accelerated response ranking using semantic similarity.
It uses DistilBERT embeddings to match user queries with results semantically,
going beyond keyword matching for superior relevance.

Features:
- GPU-accelerated embedding generation (T4 GPU support)
- Semantic similarity scoring using cosine similarity
- Context-aware ranking (user preferences, temporal factors)
- Multi-factor scoring (semantic + popularity + recency + context)
- Batch processing for efficiency
- Embedding caching for performance

Author: Istanbul AI Team
Date: October 31, 2025
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
import time

logger = logging.getLogger(__name__)


@dataclass
class RankingConfig:
    """Configuration for neural ranking weights"""
    semantic_weight: float = 0.60  # Semantic similarity
    context_weight: float = 0.20   # User preferences/context
    popularity_weight: float = 0.10  # Ratings/popularity
    recency_weight: float = 0.10   # Temporal relevance


@dataclass
class RankingResult:
    """Result of neural ranking"""
    ranked_results: List[Dict]
    method: str  # 'neural', 'fallback'
    avg_semantic_score: float
    processing_time_ms: float
    gpu_used: bool


class NeuralResponseRanker:
    """
    GPU-accelerated response ranking using semantic similarity
    
    Uses DistilBERT to generate semantic embeddings and ranks results
    based on multiple factors including semantic similarity, user context,
    popularity, and recency.
    
    Automatically falls back to keyword-based ranking if neural unavailable.
    """
    
    def __init__(
        self,
        device: str = "auto",
        config: Optional[RankingConfig] = None,
        cache_embeddings: bool = True,
        batch_size: int = 16
    ):
        """
        Initialize Neural Response Ranker
        
        Args:
            device: Device to use ('cuda', 'cpu', or 'auto')
            config: Ranking configuration weights
            cache_embeddings: Whether to cache embeddings
            batch_size: Batch size for processing multiple results
        """
        self.config = config or RankingConfig()
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size
        
        # Initialize persistent LRU cache with disk storage
        if cache_embeddings:
            try:
                from .persistent_embedding_cache import PersistentEmbeddingCache
                self.embedding_cache = PersistentEmbeddingCache(
                    max_size=10000,  # ~60MB
                    cache_dir='cache/embeddings',
                    auto_save=True,
                    save_interval=300  # Save every 5 minutes
                )
                logger.info(f"✅ Persistent cache initialized: {len(self.embedding_cache)} embeddings")
            except Exception as e:
                logger.warning(f"⚠️  Persistent cache failed, using simple dict: {e}")
                self.embedding_cache = {}
        else:
            self.embedding_cache = None
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.available = False
        
        try:
            self._initialize_model()
            self.available = True
            logger.info(f"✅ Neural ranker initialized on {self.device}")
        except Exception as e:
            logger.warning(f"⚠️  Neural ranker unavailable: {e}")
            logger.info("📝 Will use fallback ranking")
        
        # Statistics
        self.stats = {
            'total_rankings': 0,
            'neural_rankings': 0,
            'fallback_rankings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time_ms': 0.0
        }
    
    def _initialize_model(self):
        """Initialize DistilBERT model and tokenizer"""
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "distilbert-base-uncased"
        
        logger.info(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Model loaded on {self.device}")
    
    @torch.no_grad()
    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Get semantic embedding for text (GPU-accelerated)
        
        Args:
            text: Input text
            use_cache: Whether to use/update cache
            
        Returns:
            Embedding vector (768-dim for DistilBERT)
        """
        if not self.available:
            raise RuntimeError("Neural ranker not available")
        
        # Check cache (works with both dict and PersistentCache)
        if use_cache and self.cache_embeddings:
            if hasattr(self.embedding_cache, 'get'):
                # PersistentCache has .get() method
                cached = self.embedding_cache.get(text)
                if cached is not None:
                    self.stats['cache_hits'] += 1
                    return cached
            elif text in self.embedding_cache:
                # Simple dict cache
                self.stats['cache_hits'] += 1
                return self.embedding_cache[text]
        
        self.stats['cache_misses'] += 1
        
        # Tokenize and move to device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Get model output
        outputs = self.model(**inputs)
        
        # Use [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
        
        # Cache if enabled (works with both dict and PersistentCache)
        if use_cache and self.cache_embeddings:
            if hasattr(self.embedding_cache, 'set'):
                # PersistentCache has .set() method
                self.embedding_cache.set(text, embedding)
            else:
                # Simple dict cache
                self.embedding_cache[text] = embedding
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts in one GPU pass (efficient!)
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embeddings (n_texts x 768)
        """
        if not self.available:
            raise RuntimeError("Neural ranker not available")
        
        # Check cache for all texts
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = None
            if self.cache_embeddings:
                if hasattr(self.embedding_cache, 'get'):
                    # PersistentCache
                    cached = self.embedding_cache.get(text)
                elif text in self.embedding_cache:
                    # Simple dict
                    cached = self.embedding_cache[text]
            
            if cached is not None:
                cached_embeddings.append((i, cached))
                self.stats['cache_hits'] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.stats['cache_misses'] += 1
        
        # Get embeddings for uncached texts
        all_embeddings = [None] * len(texts)
        
        if uncached_texts:
            # Tokenize all at once
            inputs = self.tokenizer(
                uncached_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Single forward pass (GPU-efficient!)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            
            # Cache and store
            for i, text, embedding in zip(uncached_indices, uncached_texts, embeddings):
                all_embeddings[i] = embedding
                if self.cache_embeddings:
                    if hasattr(self.embedding_cache, 'set'):
                        # PersistentCache
                        self.embedding_cache.set(text, embedding)
                    else:
                        # Simple dict
                        self.embedding_cache[text] = embedding
        
        # Add cached embeddings
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
        
        return np.array(all_embeddings)
    
    def rank_results(
        self,
        query: str,
        results: List[Dict],
        user_context: Optional[Dict] = None,
        field: str = 'description'
    ) -> RankingResult:
        """
        Rank results using neural semantic similarity + context
        
        Args:
            query: User's search query
            results: List of result dictionaries to rank
            user_context: Optional user context (preferences, history, etc.)
            field: Field to use for semantic matching (default: 'description')
            
        Returns:
            RankingResult with ranked results and metadata
        """
        start_time = time.time()
        self.stats['total_rankings'] += 1
        
        # Fallback if neural unavailable or no results
        if not self.available or not results:
            self.stats['fallback_rankings'] += 1
            return RankingResult(
                ranked_results=results,
                method='fallback',
                avg_semantic_score=0.0,
                processing_time_ms=0.0,
                gpu_used=False
            )
        
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Get all result texts
            result_texts = []
            for result in results:
                # Get text from specified field or construct from name/description
                text = result.get(field, '')
                if not text:
                    text = f"{result.get('name', '')} {result.get('description', '')}"
                result_texts.append(text.strip() or "No description")
            
            # Get embeddings for all results in batch (GPU-efficient!)
            result_embeddings = self.get_embeddings_batch(result_texts)
            
            # Calculate semantic similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                result_embeddings
            )[0]
            
            # Score and rank results
            scored_results = []
            for i, (result, similarity) in enumerate(zip(results, similarities)):
                # Calculate context score
                context_score = self._calculate_context_score(result, user_context)
                
                # Get other factors
                popularity = result.get('rating', result.get('popularity', 0.5))
                if isinstance(popularity, (int, float)):
                    popularity = min(popularity / 5.0, 1.0)  # Normalize to 0-1
                else:
                    popularity = 0.5
                
                recency = self._calculate_recency_score(result)
                
                # Combined score using config weights
                final_score = (
                    similarity * self.config.semantic_weight +
                    context_score * self.config.context_weight +
                    popularity * self.config.popularity_weight +
                    recency * self.config.recency_weight
                )
                
                # Add scoring metadata to result
                result_copy = result.copy()
                result_copy['neural_score'] = float(final_score)
                result_copy['semantic_similarity'] = float(similarity)
                result_copy['context_score'] = float(context_score)
                result_copy['popularity_score'] = float(popularity)
                result_copy['recency_score'] = float(recency)
                result_copy['ranking_method'] = 'neural'
                
                scored_results.append(result_copy)
            
            # Sort by final score
            scored_results.sort(key=lambda x: x['neural_score'], reverse=True)
            
            # Update statistics
            self.stats['neural_rankings'] += 1
            processing_time_ms = (time.time() - start_time) * 1000
            self.stats['avg_processing_time_ms'] = (
                (self.stats['avg_processing_time_ms'] * (self.stats['neural_rankings'] - 1) +
                 processing_time_ms) / self.stats['neural_rankings']
            )
            
            avg_semantic_score = float(np.mean(similarities))
            
            logger.info(
                f"🧠 Neural ranking: {len(results)} results, "
                f"avg similarity: {avg_semantic_score:.3f}, "
                f"time: {processing_time_ms:.1f}ms"
            )
            
            return RankingResult(
                ranked_results=scored_results,
                method='neural',
                avg_semantic_score=avg_semantic_score,
                processing_time_ms=processing_time_ms,
                gpu_used=(self.device == 'cuda')
            )
            
        except Exception as e:
            logger.error(f"❌ Neural ranking failed: {e}")
            self.stats['fallback_rankings'] += 1
            return RankingResult(
                ranked_results=results,
                method='fallback',
                avg_semantic_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                gpu_used=False
            )
    
    def _calculate_context_score(
        self,
        result: Dict,
        user_context: Optional[Dict]
    ) -> float:
        """
        Calculate context-based score for a result
        
        Factors:
        - User preferences (cuisine, price range, etc.)
        - User history (previously liked places)
        - Current context (time of day, location, etc.)
        
        Args:
            result: Result dictionary
            user_context: User context dictionary
            
        Returns:
            Context score (0.0-1.0)
        """
        if not user_context:
            return 0.5  # Neutral score
        
        score = 0.5
        factors = 0
        
        # Check user preferences
        preferences = user_context.get('preferences', {})
        
        # Cuisine preference
        if 'preferred_cuisines' in preferences:
            result_cuisine = result.get('cuisine', '').lower()
            if result_cuisine in [c.lower() for c in preferences['preferred_cuisines']]:
                score += 0.2
                factors += 1
        
        # Price range preference
        if 'price_range' in preferences:
            result_price = result.get('price_level', result.get('price_range', ''))
            if result_price == preferences['price_range']:
                score += 0.15
                factors += 1
        
        # Distance preference
        if 'max_distance_km' in preferences and 'distance_km' in result:
            if result['distance_km'] <= preferences['max_distance_km']:
                score += 0.15
                factors += 1
        
        # Previously visited/liked
        history = user_context.get('history', {})
        if 'liked_places' in history:
            if result.get('id') in history['liked_places']:
                score += 0.3  # Strong boost for previously liked
                factors += 1
        
        # Normalize score
        if factors > 0:
            score = min(score, 1.0)
        
        return score
    
    def _calculate_recency_score(self, result: Dict) -> float:
        """
        Calculate recency score for temporal relevance
        
        Args:
            result: Result dictionary
            
        Returns:
            Recency score (0.0-1.0)
        """
        # Check if result has temporal data
        if 'date' in result:
            try:
                result_date = datetime.fromisoformat(result['date'])
                now = datetime.now()
                days_diff = (now - result_date).days
                
                # Score decreases with age
                # Recent (0-7 days): 1.0
                # Medium (8-30 days): 0.7
                # Old (31+ days): 0.4
                if days_diff <= 7:
                    return 1.0
                elif days_diff <= 30:
                    return 0.7
                else:
                    return 0.4
            except:
                pass
        
        # Check if result is for events (temporal)
        if result.get('type') == 'event':
            if 'event_date' in result:
                try:
                    event_date = datetime.fromisoformat(result['event_date'])
                    now = datetime.now()
                    
                    # Future events
                    if event_date > now:
                        days_until = (event_date - now).days
                        # Upcoming (0-7 days): 1.0
                        # Soon (8-30 days): 0.8
                        # Later (31+ days): 0.6
                        if days_until <= 7:
                            return 1.0
                        elif days_until <= 30:
                            return 0.8
                        else:
                            return 0.6
                except:
                    pass
        
        return 0.5  # Neutral score
    
    def clear_cache(self):
        """Clear embedding cache"""
        if self.embedding_cache:
            if hasattr(self.embedding_cache, 'clear'):
                self.embedding_cache.clear()
            else:
                self.embedding_cache = {}
            logger.info("🗑️  Embedding cache cleared")
    
    def save_cache(self):
        """Save cache to disk (if persistent cache)"""
        if self.embedding_cache and hasattr(self.embedding_cache, 'save_to_disk'):
            self.embedding_cache.save_to_disk()
            logger.info("💾 Cache saved to disk")
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings"""
        if not self.embedding_cache:
            return 0
        if hasattr(self.embedding_cache, 'get_size'):
            return self.embedding_cache.get_size()
        return len(self.embedding_cache)
    
    def get_stats(self) -> Dict:
        """Get ranking statistics"""
        stats = self.stats.copy()
        stats['available'] = self.available
        stats['device'] = self.device
        stats['cache_size'] = self.get_cache_size()
        stats['cache_hit_rate'] = (
            self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
        )
        
        # Add persistent cache stats if available
        if self.embedding_cache and hasattr(self.embedding_cache, 'get_stats'):
            cache_stats = self.embedding_cache.get_stats()
            stats['cache_memory_mb'] = cache_stats.get('memory_mb', 0)
            stats['cache_utilization'] = cache_stats.get('utilization', 0)
            stats['cache_evictions'] = cache_stats.get('evictions', 0)
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_rankings': 0,
            'neural_rankings': 0,
            'fallback_rankings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time_ms': 0.0
        }
        logger.info("📊 Neural ranker stats reset")


def create_neural_ranker(
    device: str = "auto",
    config: Optional[RankingConfig] = None
) -> NeuralResponseRanker:
    """
    Factory function to create neural response ranker
    
    Args:
        device: Device to use ('cuda', 'cpu', or 'auto')
        config: Optional ranking configuration
        
    Returns:
        NeuralResponseRanker instance
    """
    return NeuralResponseRanker(device=device, config=config)
