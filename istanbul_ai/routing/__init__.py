"""
Routing Layer - Intent Classification, Entity Extraction, and Query Routing

This module provides the routing layer for the Istanbul AI system, responsible for:
- Intent classification from user messages (keyword + neural hybrid)
- Entity extraction and enhancement
- Query routing to appropriate handlers
- Query preprocessing and normalization
- Neural response ranking (GPU-accelerated semantic similarity)
- Persistent embedding caching with pre-warming

Week 2 Refactoring: Extracted from main_system.py for better modularity
ML Enhancement: Added HybridIntentClassifier for GPU-accelerated neural + keyword ensemble
Phase 2: Added NeuralResponseRanker for semantic similarity ranking
Phase 3: Added PersistentEmbeddingCache and CachePrewarmer for optimized caching
"""

from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .response_router import ResponseRouter
from .query_preprocessor import QueryPreprocessor
from .hybrid_intent_classifier import HybridIntentClassifier
from .neural_response_ranker import NeuralResponseRanker, RankingConfig
from .persistent_embedding_cache import PersistentEmbeddingCache, create_persistent_cache
from .cache_prewarmer import CachePrewarmer, prewarm_cache

__all__ = [
    'IntentClassifier',
    'EntityExtractor',
    'ResponseRouter',
    'QueryPreprocessor',
    'HybridIntentClassifier',
    'NeuralResponseRanker',
    'RankingConfig',
    'PersistentEmbeddingCache',
    'create_persistent_cache',
    'CachePrewarmer',
    'prewarm_cache'
]
