"""
Cache Pre-warmer - Pre-generate embeddings for common queries

This module pre-warms the embedding cache with frequently used queries
to achieve 95%+ cache hit rate from the start.

Author: Istanbul AI Team
Date: October 31, 2025
"""

import logging
from typing import List, Optional
import time

logger = logging.getLogger(__name__)


class CachePrewarmer:
    """
    Pre-warm embedding cache with common queries
    
    Analyzes common query patterns and pre-generates embeddings
    for the most frequent queries to maximize cache hit rate.
    """
    
    def __init__(self, neural_ranker):
        """
        Initialize cache pre-warmer
        
        Args:
            neural_ranker: NeuralResponseRanker instance with embedding cache
        """
        self.ranker = neural_ranker
        self.common_queries = self._get_common_queries()
    
    def _get_common_queries(self) -> List[str]:
        """
        Get list of most common queries
        
        These are based on typical user queries and can be updated
        based on actual query logs in production.
        
        Returns:
            List of common query strings
        """
        return [
            # Restaurant queries (most common)
            "best restaurants in Sultanahmet",
            "traditional Turkish restaurants",
            "cheap restaurants near Taksim",
            "seafood restaurants with Bosphorus view",
            "authentic Ottoman cuisine",
            "rooftop restaurants Istanbul",
            "restaurants in Kadıköy",
            "best kebab restaurants",
            "vegetarian restaurants Istanbul",
            "romantic restaurants for dinner",
            "breakfast places with good coffee",
            "Turkish street food",
            "fine dining restaurants",
            "family-friendly restaurants",
            "late night restaurants",
            
            # Attraction queries
            "things to do in Istanbul",
            "historical sites Istanbul",
            "museums in Istanbul",
            "hidden gems Istanbul",
            "best places to visit",
            "Ottoman palaces",
            "Byzantine architecture",
            "mosques to visit",
            "bazaars and markets",
            "Bosphorus cruise",
            "photography spots Istanbul",
            "sunset viewpoints",
            "architectural landmarks",
            "religious sites",
            "off the beaten path Istanbul",
            
            # Transportation queries
            "how to get to Taksim Square",
            "airport to city center",
            "metro map Istanbul",
            "public transportation Istanbul",
            "ferry schedule",
            "taxi to Blue Mosque",
            "bus routes Istanbul",
            "transportation from airport",
            "getting around Istanbul",
            "metro card Istanbul",
            
            # Event queries
            "events this weekend",
            "concerts in Istanbul",
            "cultural events",
            "art exhibitions",
            "music festivals",
            "theater shows",
            "nightlife Istanbul",
            "live music venues",
            "whirling dervishes show",
            "traditional Turkish performance",
            
            # Neighborhood queries
            "best neighborhoods to explore",
            "Sultanahmet area",
            "Beyoğlu district",
            "Asian side Istanbul",
            "Kadıköy neighborhood",
            "Beşiktaş area",
            "trendy neighborhoods",
            "local neighborhoods",
            "shopping districts",
            "nightlife areas",
            
            # Weather/timing queries
            "weather in Istanbul",
            "best time to visit",
            "opening hours museums",
            "prayer times",
            "sunset time",
            
            # Specific landmarks (very common)
            "Hagia Sophia",
            "Blue Mosque",
            "Topkapi Palace",
            "Grand Bazaar",
            "Spice Bazaar",
            "Galata Tower",
            "Dolmabahçe Palace",
            "Basilica Cistern",
            "Chora Church",
            "Süleymaniye Mosque",
            "Maiden's Tower",
            "Istiklal Avenue",
            "Bosphorus Bridge",
            "Pierre Loti Hill",
            "Princes' Islands",
        ]
    
    def prewarm(self, progress_callback: Optional[callable] = None) -> dict:
        """
        Pre-warm cache with common queries
        
        Generates embeddings for all common queries and stores them in cache.
        This ensures high cache hit rate from the start.
        
        Args:
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Dict with pre-warming statistics
        """
        if not self.ranker or not self.ranker.available:
            logger.warning("⚠️  Neural ranker not available, skipping pre-warm")
            return {'success': False, 'reason': 'ranker_unavailable'}
        
        if not self.ranker.cache_embeddings:
            logger.warning("⚠️  Caching disabled, skipping pre-warm")
            return {'success': False, 'reason': 'caching_disabled'}
        
        logger.info(f"🔥 Pre-warming cache with {len(self.common_queries)} queries...")
        start_time = time.time()
        
        success_count = 0
        failure_count = 0
        
        for i, query in enumerate(self.common_queries, 1):
            try:
                # Generate and cache embedding
                self.ranker.get_embedding(query, use_cache=True)
                success_count += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(i, len(self.common_queries))
                
            except Exception as e:
                logger.warning(f"Failed to prewarm '{query}': {e}")
                failure_count += 1
        
        elapsed_time = time.time() - start_time
        
        # Get final cache stats
        cache_stats = self.ranker.embedding_cache.get_stats() if hasattr(self.ranker.embedding_cache, 'get_stats') else {}
        
        result = {
            'success': True,
            'total_queries': len(self.common_queries),
            'success_count': success_count,
            'failure_count': failure_count,
            'elapsed_time': elapsed_time,
            'cache_size': cache_stats.get('size', success_count),
            'cache_memory_mb': cache_stats.get('memory_mb', 0)
        }
        
        logger.info(
            f"✅ Cache pre-warmed: {success_count}/{len(self.common_queries)} queries, "
            f"{elapsed_time:.1f}s, {cache_stats.get('memory_mb', 0):.1f} MB"
        )
        
        return result
    
    def prewarm_categories(self, categories: List[str]) -> dict:
        """
        Pre-warm specific categories only
        
        Args:
            categories: List of categories ('restaurants', 'attractions', etc.)
            
        Returns:
            Pre-warming statistics
        """
        category_keywords = {
            'restaurants': ['restaurant', 'food', 'dining', 'cuisine', 'cafe'],
            'attractions': ['attraction', 'museum', 'site', 'place', 'landmark'],
            'transportation': ['transport', 'metro', 'bus', 'taxi', 'ferry'],
            'events': ['event', 'concert', 'show', 'festival', 'performance'],
            'neighborhoods': ['neighborhood', 'district', 'area', 'quarter']
        }
        
        # Filter queries by category
        filtered_queries = []
        for query in self.common_queries:
            query_lower = query.lower()
            for category in categories:
                if category in category_keywords:
                    if any(keyword in query_lower for keyword in category_keywords[category]):
                        filtered_queries.append(query)
                        break
        
        # Temporarily replace common_queries
        original_queries = self.common_queries
        self.common_queries = filtered_queries
        
        result = self.prewarm()
        
        # Restore original queries
        self.common_queries = original_queries
        
        return result
    
    def add_custom_queries(self, queries: List[str]):
        """
        Add custom queries to pre-warm list
        
        Args:
            queries: List of query strings to add
        """
        self.common_queries.extend(queries)
        logger.info(f"✅ Added {len(queries)} custom queries to pre-warm list")
    
    def get_query_count(self) -> int:
        """Get total number of queries to pre-warm"""
        return len(self.common_queries)


def prewarm_cache(neural_ranker, categories: Optional[List[str]] = None) -> dict:
    """
    Convenience function to pre-warm cache
    
    Args:
        neural_ranker: NeuralResponseRanker instance
        categories: Optional list of categories to pre-warm
        
    Returns:
        Pre-warming statistics
    """
    prewarmer = CachePrewarmer(neural_ranker)
    
    if categories:
        return prewarmer.prewarm_categories(categories)
    else:
        return prewarmer.prewarm()
