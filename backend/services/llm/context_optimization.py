"""
Context Optimization Module for Phase 3

This module provides:
1. Context caching for popular items
2. Context ranking by relevance
3. Context compression for token optimization
4. Prompt optimization with dynamic selection

Author: AI Istanbul Team
Date: January 2025
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import asyncio

logger = logging.getLogger(__name__)


class ContextCache:
    """
    Cache for popular context items to reduce database queries.
    
    Features:
    - Cache popular restaurants, museums, districts
    - TTL-based expiration
    - Hit rate tracking
    - Smart invalidation
    """
    
    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_items: int = 100
    ):
        """
        Initialize context cache.
        
        Args:
            ttl_seconds: Time-to-live for cached items
            max_items: Maximum items to cache
        """
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        
        # Cache storage
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Popular items tracking
        self.access_counts: Counter = Counter()
        
        logger.info(f"✅ Context Cache initialized (TTL={ttl_seconds}s, Max={max_items})")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        # Check TTL
        if datetime.now() - self.cache_timestamps[key] > timedelta(seconds=self.ttl_seconds):
            # Expired
            del self.cache[key]
            del self.cache_timestamps[key]
            self.misses += 1
            return None
        
        self.hits += 1
        self.access_counts[key] += 1
        return self.cache[key]
    
    def put(self, key: str, data: Dict[str, Any]):
        """
        Put item in cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        # Evict if at capacity
        if len(self.cache) >= self.max_items and key not in self.cache:
            self._evict_lru()
        
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Find oldest item
        oldest_key = min(self.cache_timestamps, key=self.cache_timestamps.get)
        del self.cache[oldest_key]
        del self.cache_timestamps[oldest_key]
        self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_items,
            'evictions': self.evictions,
            'most_popular': self.access_counts.most_common(10)
        }
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Context cache cleared")


class ContextRanker:
    """
    Rank context items by relevance to query and signals.
    
    Features:
    - Relevance scoring
    - Signal-based boosting
    - Recency boosting
    - Diversity enforcement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize context ranker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Scoring weights
        self.signal_match_weight = self.config.get('signal_match_weight', 2.0)
        self.recency_weight = self.config.get('recency_weight', 0.5)
        self.popularity_weight = self.config.get('popularity_weight', 0.3)
        self.diversity_penalty = self.config.get('diversity_penalty', 0.2)
        
        logger.info("✅ Context Ranker initialized")
    
    def rank_items(
        self,
        items: List[Dict[str, Any]],
        signals: List[str],
        query: str,
        max_items: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Rank items by relevance.
        
        Args:
            items: List of context items
            signals: Detected signals
            query: User query
            max_items: Maximum items to return
            
        Returns:
            Ranked list of items
        """
        if not items:
            return []
        
        # Score each item
        scored_items = []
        for item in items:
            score = self._calculate_relevance_score(item, signals, query)
            scored_items.append((score, item))
        
        # Sort by score (highest first)
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # Apply diversity
        ranked = self._apply_diversity(scored_items, max_items)
        
        logger.debug(f"Ranked {len(items)} items to {len(ranked)} items")
        
        return ranked
    
    def _calculate_relevance_score(
        self,
        item: Dict[str, Any],
        signals: List[str],
        query: str
    ) -> float:
        """Calculate relevance score for an item"""
        score = 1.0  # Base score
        
        # Signal match boost
        item_type = item.get('type', '')
        for signal in signals:
            if 'restaurant' in signal and item_type == 'restaurant':
                score += self.signal_match_weight
            elif 'attraction' in signal and item_type in ['museum', 'attraction']:
                score += self.signal_match_weight
            elif 'shopping' in signal and item_type in ['shopping', 'market']:
                score += self.signal_match_weight
            elif 'nightlife' in signal and item_type in ['bar', 'club', 'nightlife']:
                score += self.signal_match_weight
        
        # Recency boost (if item has timestamp)
        if 'updated_at' in item or 'created_at' in item:
            # Boost newer items slightly
            score += self.recency_weight * 0.5
        
        # Popularity boost (if item has rating or review count)
        if 'rating' in item:
            rating = float(item.get('rating', 0))
            score += self.popularity_weight * (rating / 5.0)
        
        if 'review_count' in item:
            review_count = int(item.get('review_count', 0))
            # Logarithmic boost for popularity
            import math
            score += self.popularity_weight * math.log10(max(review_count, 1) + 1) * 0.1
        
        return score
    
    def _apply_diversity(
        self,
        scored_items: List[Tuple[float, Dict[str, Any]]],
        max_items: int
    ) -> List[Dict[str, Any]]:
        """Apply diversity to avoid too many similar items"""
        if len(scored_items) <= max_items:
            return [item for _, item in scored_items]
        
        selected = []
        used_types = Counter()
        used_districts = Counter()
        
        for score, item in scored_items:
            if len(selected) >= max_items:
                break
            
            # Penalize if too many of same type/district
            item_type = item.get('type', '')
            district = item.get('district', '')
            
            penalty = 0
            if used_types[item_type] > 3:
                penalty += self.diversity_penalty
            if used_districts[district] > 4:
                penalty += self.diversity_penalty
            
            adjusted_score = score - penalty
            
            # Still add if score is decent
            if adjusted_score > 0.5 or len(selected) < max_items // 2:
                selected.append(item)
                used_types[item_type] += 1
                used_districts[district] += 1
        
        return selected


class ContextCompressor:
    """
    Compress context to optimize token usage.
    
    Features:
    - Summarize long descriptions
    - Remove redundant information
    - Keep only essential fields
    - Token counting
    """
    
    def __init__(self, max_tokens: int = 2000):
        """
        Initialize context compressor.
        
        Args:
            max_tokens: Maximum tokens for context
        """
        self.max_tokens = max_tokens
        
        # Essential fields by type
        self.essential_fields = {
            'restaurant': ['name', 'cuisine', 'price_level', 'district', 'rating'],
            'museum': ['name', 'type', 'district', 'opening_hours'],
            'attraction': ['name', 'type', 'district', 'description'],
            'shopping': ['name', 'type', 'district'],
            'nightlife': ['name', 'type', 'district', 'atmosphere']
        }
        
        logger.info(f"✅ Context Compressor initialized (max_tokens={max_tokens})")
    
    def compress_context(
        self,
        context: Dict[str, Any],
        signals: List[str]
    ) -> Dict[str, Any]:
        """
        Compress context to fit token budget.
        
        Args:
            context: Full context dictionary
            signals: Detected signals for prioritization
            
        Returns:
            Compressed context
        """
        compressed = {}
        
        # Database context
        if 'database' in context and context['database']:
            compressed['database'] = self._compress_database_context(
                context['database'],
                signals
            )
        
        # RAG context
        if 'rag' in context and context['rag']:
            compressed['rag'] = self._compress_rag_context(
                context['rag']
            )
        
        # Services context
        if 'services' in context and context['services']:
            compressed['services'] = self._compress_services_context(
                context['services'],
                signals
            )
        
        # Copy other fields as-is
        for key in context:
            if key not in ['database', 'rag', 'services']:
                compressed[key] = context[key]
        
        logger.debug(f"Context compressed")
        
        return compressed
    
    def _compress_database_context(
        self,
        db_context: str,
        signals: List[str]
    ) -> str:
        """Compress database context"""
        # If already short, return as-is
        if len(db_context) < 500:
            return db_context
        
        # Extract key information based on signals
        # This is a simplified version - could be improved with NLP
        lines = db_context.split('\n')
        important_lines = []
        
        for line in lines:
            # Keep lines that match signals
            for signal in signals:
                signal_keyword = signal.replace('needs_', '')
                if signal_keyword in line.lower():
                    important_lines.append(line)
                    break
        
        # If we filtered too much, keep more
        if len(important_lines) < 10 and len(lines) > 10:
            important_lines = lines[:15]
        
        return '\n'.join(important_lines[:20])  # Max 20 lines
    
    def _compress_rag_context(self, rag_context: str) -> str:
        """Compress RAG context"""
        # Truncate to reasonable length
        if len(rag_context) < 500:
            return rag_context
        
        # Keep first 400 characters (usually most relevant)
        return rag_context[:400] + "..."
    
    def _compress_services_context(
        self,
        services: List[Dict[str, Any]],
        signals: List[str]
    ) -> List[Dict[str, Any]]:
        """Compress services context"""
        compressed = []
        
        for item in services[:10]:  # Max 10 items
            item_type = item.get('type', 'unknown')
            essential = self.essential_fields.get(item_type, ['name', 'type', 'district'])
            
            # Keep only essential fields
            compressed_item = {
                field: item.get(field)
                for field in essential
                if field in item
            }
            
            # Truncate description if present
            if 'description' in compressed_item:
                desc = compressed_item['description']
                if len(desc) > 100:
                    compressed_item['description'] = desc[:97] + "..."
            
            compressed.append(compressed_item)
        
        return compressed
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: ~4 characters per token
        return len(text) // 4


class PromptOptimizer:
    """
    Optimize prompts with dynamic selection and few-shot examples.
    
    Features:
    - Template selection based on signals
    - Few-shot examples
    - Chain-of-thought reasoning
    - Token-aware truncation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize prompt optimizer"""
        self.config = config or {}
        
        # Few-shot examples by signal type
        self.few_shot_examples = self._load_few_shot_examples()
        
        logger.info("✅ Prompt Optimizer initialized")
    
    def _load_few_shot_examples(self) -> Dict[str, List[str]]:
        """Load few-shot examples for each signal type"""
        return {
            'needs_restaurant': [
                "Q: Best Italian restaurants in Beyoglu?\nA: Here are top Italian restaurants in Beyoglu: [specific recommendations with details]",
            ],
            'needs_shopping': [
                "Q: Where can I buy souvenirs?\nA: For authentic souvenirs, visit the Grand Bazaar or Spice Market: [specific recommendations]",
            ],
            'needs_nightlife': [
                "Q: Best bars in Kadikoy?\nA: Kadikoy has excellent nightlife. Top bars include: [specific recommendations with atmosphere]",
            ],
            'needs_family_friendly': [
                "Q: Activities for kids in Istanbul?\nA: Great family-friendly activities: Istanbul Aquarium, Miniaturk, [specific recommendations]",
            ]
        }
    
    def optimize_prompt(
        self,
        base_prompt: str,
        signals: List[str],
        context: Dict[str, Any],
        max_tokens: int = 3000
    ) -> str:
        """
        Optimize prompt for better LLM performance.
        
        Args:
            base_prompt: Base prompt template
            signals: Detected signals
            context: Context dictionary
            max_tokens: Maximum prompt tokens
            
        Returns:
            Optimized prompt
        """
        parts = []
        
        # Add few-shot examples if relevant
        for signal in signals[:2]:  # Max 2 examples
            if signal in self.few_shot_examples:
                examples = self.few_shot_examples[signal]
                parts.append("Example:\n" + examples[0] + "\n")
        
        # Add chain-of-thought instruction for complex queries
        if len(signals) > 2:
            parts.append(
                "Think step by step:\n"
                "1. Identify the main need\n"
                "2. Consider the context\n"
                "3. Provide specific recommendations\n\n"
            )
        
        # Add base prompt
        parts.append(base_prompt)
        
        # Combine and ensure within token budget
        full_prompt = "\n".join(parts)
        
        # Rough token estimation and truncation
        estimated_tokens = len(full_prompt) // 4
        if estimated_tokens > max_tokens:
            # Truncate context portion
            truncate_at = max_tokens * 4
            full_prompt = full_prompt[:truncate_at]
        
        return full_prompt
