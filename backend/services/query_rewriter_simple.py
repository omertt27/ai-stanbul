"""
Simple Query Rewriting System - LLM-Based Approach

PRIORITY 3.3: Query rewriting using LLM for intelligent query enhancement.

Philosophy: Don't use complex rules. Let the LLM enhance unclear queries.
The LLM can:
- Expand abbreviations and shorthand
- Add context and clarity
- Fix grammar and spelling
- Make implicit information explicit

Updated: December 2024 - Using unified PromptBuilder from prompts.py

Author: AI Istanbul Team
Date: November 14, 2025
"""

import logging
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import redis

logger = logging.getLogger(__name__)


class SimpleQueryRewriter:
    """
    Simple query rewriter that uses LLM to enhance unclear queries.
    
    Instead of complex rule-based rewriting, we ask the LLM to:
    1. Detect if a query is unclear
    2. Rewrite it to be more specific and clear
    
    This is:
    - More accurate (LLM understands nuance)
    - Simpler (no complex rules)
    - Multilingual (works in any language)
    - Flexible (adapts to any query type)
    """
    
    def __init__(
        self,
        llm_client=None,
        redis_client: Optional[redis.Redis] = None,
        cache_ttl: int = 86400,  # 24 hours
        min_query_length: int = 2,
        rewrite_threshold: float = 0.7
    ):
        """
        Initialize query rewriter.
        
        Args:
            llm_client: LLM client for query rewriting
            redis_client: Redis client for caching rewrites
            cache_ttl: Cache time-to-live in seconds
            min_query_length: Minimum query length (in words) before rewriting
            rewrite_threshold: Confidence threshold for applying rewrites
        """
        self.llm = llm_client
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.min_query_length = min_query_length
        self.rewrite_threshold = rewrite_threshold
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'rewrites_attempted': 0,
            'rewrites_applied': 0,
            'cache_hits': 0,
            'improvements': 0
        }
        
        logger.info("✍️ Simple Query Rewriter initialized")
    
    def should_rewrite(self, query: str, conversation_context: Optional[str] = None) -> bool:
        """
        Determine if query needs rewriting.
        
        Quick heuristics to avoid unnecessary LLM calls:
        - Very short queries (< 2 words)
        - Queries with many abbreviations
        - Queries with unclear references without context
        
        Args:
            query: User query
            conversation_context: Optional conversation context
            
        Returns:
            True if query likely needs enhancement
        """
        query = query.strip()
        
        # Don't rewrite empty queries
        if not query:
            return False
        
        # Count words
        words = query.split()
        word_count = len(words)
        
        # Very short queries might need expansion
        if word_count <= self.min_query_length:
            return True
        
        # Check for unclear references without context
        unclear_patterns = ['there', 'it', 'that', 'this', 'what about', 'how about']
        query_lower = query.lower()
        
        has_unclear_ref = any(pattern in query_lower for pattern in unclear_patterns)
        
        # Only rewrite unclear references if no conversation context
        if has_unclear_ref and not conversation_context:
            return True
        
        # Check for common abbreviations or shorthand
        common_abbreviations = ['tmrw', 'asap', 'rn', 'pls', 'thx', 'u', 'r']
        has_abbreviation = any(abbr in query_lower for abbr in common_abbreviations)
        
        if has_abbreviation:
            return True
        
        # Otherwise, query seems clear enough
        return False
    
    async def rewrite_query(
        self,
        query: str,
        conversation_context: Optional[str] = None,
        user_location: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Rewrite query to be clearer and more specific.
        
        Args:
            query: Original user query
            conversation_context: Optional conversation history
            user_location: Optional user location info
            language: Query language
            
        Returns:
            Tuple of (rewritten_query, was_rewritten, metadata)
        """
        self.stats['total_queries'] += 1
        
        # Quick check: should we rewrite?
        if not self.should_rewrite(query, conversation_context):
            return query, False, {'reason': 'query_clear_enough'}
        
        # Check cache first
        cache_key = self._get_cache_key(query, conversation_context, language)
        cached_rewrite = self._get_from_cache(cache_key)
        
        if cached_rewrite:
            self.stats['cache_hits'] += 1
            self.stats['rewrites_applied'] += 1
            return cached_rewrite['rewritten'], True, {
                'reason': 'cached',
                'confidence': cached_rewrite.get('confidence', 1.0)
            }
        
        # No LLM client, return original
        if not self.llm:
            return query, False, {'reason': 'no_llm_client'}
        
        self.stats['rewrites_attempted'] += 1
        
        try:
            # Build rewriting prompt
            rewrite_prompt = self._build_rewrite_prompt(
                query=query,
                conversation_context=conversation_context,
                user_location=user_location,
                language=language
            )
            
            # Ask LLM to rewrite
            response = await self.llm.generate(
                prompt=rewrite_prompt,
                max_tokens=150,
                temperature=0.3  # Lower temperature for more consistent rewrites
            )
            
            if not response or 'generated_text' not in response:
                return query, False, {'reason': 'llm_error'}
            
            rewritten = response['generated_text'].strip()
            
            # Validate rewrite
            is_improvement = self._validate_rewrite(query, rewritten)
            
            if is_improvement:
                # Cache the rewrite
                self._cache_rewrite(cache_key, {
                    'original': query,
                    'rewritten': rewritten,
                    'confidence': 0.9,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.stats['rewrites_applied'] += 1
                self.stats['improvements'] += 1
                
                logger.info(f"✍️ Rewrote query: '{query}' → '{rewritten}'")
                
                return rewritten, True, {
                    'reason': 'improved',
                    'confidence': 0.9,
                    'original': query
                }
            else:
                # Rewrite didn't improve query
                return query, False, {
                    'reason': 'no_improvement',
                    'attempted_rewrite': rewritten
                }
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query, False, {'reason': 'error', 'error': str(e)}
    
    def _build_rewrite_prompt(
        self,
        query: str,
        conversation_context: Optional[str],
        user_location: Optional[Dict[str, Any]],
        language: str
    ) -> str:
        """
        Build prompt for LLM to rewrite query using improved template.
        
        Args:
            query: Original query
            conversation_context: Conversation context
            user_location: User location
            language: Query language
            
        Returns:
            Formatted prompt
        """
        # Build context sections
        conversation_section = ""
        if conversation_context:
            conversation_section = f"\n**Conversation Context**: {conversation_context}"
        
        location_section = ""
        if user_location:
            location_str = f"{user_location.get('district', 'Istanbul')}"
            location_section = f"\n**User Location**: {location_str}"
        
        # Build query rewriter prompt (inline, no legacy imports)
        prompt = f"""Rewrite this Istanbul travel query to be clearer and more specific.

Original Query: "{query}"{conversation_section}{location_section}

Rules:
1. If the query is already clear, return it unchanged
2. Expand abbreviations (e.g., "hm" → "Hagia Sophia Museum")
3. Fix spelling/grammar errors
4. Add implicit context (e.g., "restaurants" → "restaurants in Istanbul")
5. Keep the same language as the original query
6. Don't add information the user didn't ask for

Rewritten Query (or original if already clear):"""
        
        return prompt
    
    def _validate_rewrite(self, original: str, rewritten: str) -> bool:
        """
        Validate that rewrite is an improvement.
        
        Args:
            original: Original query
            rewritten: Rewritten query
            
        Returns:
            True if rewrite is valid improvement
        """
        # Same query - no improvement
        if original.strip().lower() == rewritten.strip().lower():
            return False
        
        # Rewrite is much longer - might be over-explaining
        if len(rewritten) > len(original) * 3:
            return False
        
        # Rewrite is too short
        if len(rewritten.split()) < 2:
            return False
        
        # Rewrite contains original intent (fuzzy check)
        original_words = set(original.lower().split())
        rewritten_words = set(rewritten.lower().split())
        
        # At least some overlap in words (maintains intent)
        overlap = len(original_words & rewritten_words)
        if overlap == 0 and len(original_words) > 2:
            return False  # Completely different - lost original intent
        
        # Looks good!
        return True
    
    def _get_cache_key(
        self,
        query: str,
        conversation_context: Optional[str],
        language: str
    ) -> str:
        """Generate cache key for query rewrite."""
        context_str = conversation_context or ""
        combined = f"{query}:{context_str}:{language}"
        return f"query_rewrite:{hashlib.md5(combined.encode()).hexdigest()}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached rewrite."""
        if not self.redis:
            return None
        
        try:
            import json
            data = self.redis.get(cache_key)
            if data:
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_rewrite(self, cache_key: str, rewrite_data: Dict[str, Any]):
        """Cache rewrite result."""
        if not self.redis:
            return
        
        try:
            import json
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(rewrite_data)
            )
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rewriter statistics.
        
        Returns:
            Statistics dict
        """
        total = self.stats['total_queries']
        if total == 0:
            return {**self.stats, 'rewrite_rate': 0.0, 'cache_hit_rate': 0.0}
        
        return {
            **self.stats,
            'rewrite_rate': self.stats['rewrites_applied'] / total,
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['rewrites_attempted']),
            'improvement_rate': self.stats['improvements'] / max(1, self.stats['rewrites_attempted'])
        }


# Example usage:
"""
# Initialize
rewriter = SimpleQueryRewriter(
    llm_client=llm_client,
    redis_client=redis_client,
    min_query_length=2
)

# Rewrite a query
query = "cheap food"
rewritten, was_rewritten, metadata = await rewriter.rewrite_query(
    query=query,
    user_location={'district': 'Sultanahmet'},
    language='en'
)

if was_rewritten:
    print(f"Enhanced: {query} → {rewritten}")
    # Use rewritten query for processing
else:
    print(f"Query clear enough: {query}")
    # Use original query

# Examples of rewrites:
# "cheap food" → "affordable restaurants in Sultanahmet"
# "there" + context → "Hagia Sophia" (from conversation)
# "how to go" → "directions to Blue Mosque"
# "tmrw weather" → "tomorrow's weather in Istanbul"
"""
