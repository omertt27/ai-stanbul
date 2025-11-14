"""
Priority 4.1: Smart Query Suggestions
Autocomplete, spell correction, and related queries

Features:
- Fast autocomplete using trie structure (<10ms)
- Spell correction with fuzzy matching
- LLM-generated related query suggestions
- Redis caching for performance
- Multilingual support

Author: AI Istanbul Team
Date: November 14, 2025
"""

import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from collections import defaultdict
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis not available. Caching disabled.")


class TrieNode:
    """Node in trie data structure for fast prefix search."""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0


class QueryTrie:
    """
    Trie structure for fast autocomplete suggestions.
    
    Supports:
    - O(m) prefix search (m = query length)
    - Frequency-based ranking
    - Case-insensitive matching
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.total_queries = 0
    
    def insert(self, query: str, frequency: int = 1):
        """Insert query into trie."""
        query = query.lower().strip()
        if not query:
            return
        
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.frequency += frequency
        self.total_queries += 1
    
    def search_prefix(self, prefix: str, max_results: int = 5) -> List[tuple]:
        """
        Search for queries with given prefix.
        
        Returns:
            List of (query, frequency) tuples
        """
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all words from this node
        results = []
        self._collect_words(node, prefix, results)
        
        # Sort by frequency (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:max_results]
    
    def _collect_words(self, node: TrieNode, current_word: str, results: List):
        """Recursively collect all words from node."""
        if node.is_end_of_word:
            results.append((current_word, node.frequency))
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, results)


class FuzzyLocationMatcher:
    """
    Fuzzy matching for location names.
    Handles typos and variations in location names.
    """
    
    def __init__(self, location_names: Optional[List[str]] = None):
        """
        Initialize with known location names.
        
        Args:
            location_names: List of correct location names
        """
        self.locations = location_names or self._get_default_locations()
        self.location_lower = {loc.lower(): loc for loc in self.locations}
    
    def _get_default_locations(self) -> List[str]:
        """Get default Istanbul locations."""
        return [
            "Taksim", "Sultanahmet", "Beyoğlu", "Kadıköy", "Beşiktaş",
            "Üsküdar", "Fatih", "Eminönü", "Şişli", "Sarıyer",
            "Ortaköy", "Bebek", "Galata", "Karaköy", "Nişantaşı",
            "Istiklal", "Bosphorus", "Golden Horn", "Princes Islands",
            "Blue Mosque", "Hagia Sophia", "Topkapi Palace", "Grand Bazaar",
            "Spice Bazaar", "Dolmabahce Palace", "Galata Tower"
        ]
    
    def find_best_match(
        self,
        query: str,
        threshold: float = 0.80
    ) -> Optional[Dict[str, Any]]:
        """
        Find best matching location name.
        
        Args:
            query: User's input (possibly misspelled)
            threshold: Minimum similarity score (0-1)
        
        Returns:
            Dict with 'name' and 'score', or None
        """
        if not query or len(query) < 2:
            return None
        
        # Use fuzzywuzzy for fuzzy matching
        best_match = process.extractOne(
            query,
            self.locations,
            scorer=fuzz.ratio
        )
        
        if best_match and best_match[1] >= (threshold * 100):
            return {
                "name": best_match[0],
                "score": best_match[1] / 100.0
            }
        
        return None
    
    def find_multiple_matches(
        self,
        query: str,
        max_results: int = 3,
        threshold: float = 0.75
    ) -> List[Dict[str, Any]]:
        """Find multiple possible matches."""
        if not query or len(query) < 2:
            return []
        
        matches = process.extract(
            query,
            self.locations,
            scorer=fuzz.ratio,
            limit=max_results
        )
        
        return [
            {"name": match[0], "score": match[1] / 100.0}
            for match in matches
            if match[1] >= (threshold * 100)
        ]


class QuerySuggester:
    """
    Smart query suggestions with autocomplete, spell check, and related queries.
    
    Features:
    - Fast autocomplete (<10ms)
    - Spell correction (fuzzy matching)
    - Related queries (LLM-generated)
    - Multilingual support
    - Redis caching
    """
    
    def __init__(
        self,
        llm_client: Any,
        redis_client: Optional[redis.Redis] = None,
        location_names: Optional[List[str]] = None,
        cache_ttl: int = 3600
    ):
        """
        Initialize Query Suggester.
        
        Args:
            llm_client: LLM client for generating related queries
            redis_client: Optional Redis client for caching
            location_names: List of known location names
            cache_ttl: Cache TTL in seconds (default: 1 hour)
        """
        self.llm = llm_client
        self.redis = redis_client if REDIS_AVAILABLE else None
        self.cache_ttl = cache_ttl
        
        # Autocomplete trie
        self.trie = QueryTrie()
        self.query_frequencies = defaultdict(int)
        
        # Spell checker
        self.fuzzy_matcher = FuzzyLocationMatcher(location_names)
        
        # Statistics
        self.stats = {
            "total_suggestions": 0,
            "autocomplete_requests": 0,
            "spell_check_requests": 0,
            "spell_corrections_made": 0,
            "related_query_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0
        }
        
        # Load popular queries if available
        self._load_popular_queries()
        
        logger.info("✅ QuerySuggester initialized")
        if self.redis:
            logger.info("✅ Redis caching enabled (TTL: %ds)", cache_ttl)
    
    def _load_popular_queries(self):
        """Load popular queries from Redis if available."""
        if not self.redis:
            return
        
        try:
            # Load from Redis
            popular = self.redis.get("query_suggester:popular_queries")
            if popular:
                queries_data = json.loads(popular)
                for query, freq in queries_data.items():
                    self.trie.insert(query, freq)
                    self.query_frequencies[query] = freq
                logger.info(f"✅ Loaded {len(queries_data)} popular queries")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load popular queries: {e}")
    
    def _save_popular_queries(self):
        """Save popular queries to Redis."""
        if not self.redis:
            return
        
        try:
            # Save top 1000 queries
            top_queries = dict(
                sorted(
                    self.query_frequencies.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:1000]
            )
            self.redis.setex(
                "query_suggester:popular_queries",
                86400,  # 24 hours
                json.dumps(top_queries)
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to save popular queries: {e}")
    
    async def suggest_completions(
        self,
        partial_query: str,
        language: str = "en",
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Fast autocomplete suggestions.
        
        Uses trie structure for <10ms latency.
        Ranked by popularity.
        
        Args:
            partial_query: Incomplete user query
            language: Language code (en, tr)
            max_suggestions: Maximum suggestions to return
        
        Returns:
            List of suggested queries
        """
        self.stats["autocomplete_requests"] += 1
        self.stats["total_suggestions"] += 1
        
        if not partial_query or len(partial_query) < 2:
            return []
        
        # Search trie
        results = self.trie.search_prefix(partial_query, max_suggestions)
        
        # Extract query strings
        suggestions = [query for query, freq in results]
        
        logger.debug(f"Autocomplete: '{partial_query}' → {len(suggestions)} suggestions")
        
        return suggestions
    
    async def suggest_correction(
        self,
        query: str,
        language: str = "en",
        threshold: float = 0.75
    ) -> Optional[Dict[str, Any]]:
        """
        Detect and correct spelling mistakes.
        
        Args:
            query: User query (possibly with typos)
            language: Language code
            threshold: Minimum similarity for correction (0-1)
        
        Returns:
            Dict with corrected query and changes, or None
        """
        self.stats["spell_check_requests"] += 1
        self.stats["total_suggestions"] += 1
        
        if not query or len(query) < 3:
            return None
        
        # Extract potential location names (simple word extraction)
        words = query.split()
        corrections = []
        
        for word in words:
            # Skip common words and very short words
            if len(word) < 4 or word.lower() in ['the', 'to', 'in', 'at', 'from', 'how', 'what', 'where', 'for', 'near', 'best']:
                continue
            
            # Skip if word is all lowercase (likely not a location)
            if word.islower():
                continue
            
            # Try fuzzy match with lower threshold for better matches
            match = self.fuzzy_matcher.find_best_match(word, threshold)
            if match and match['name'].lower() != word.lower():
                corrections.append({
                    "original": word,
                    "corrected": match['name'],
                    "confidence": match['score']
                })
        
        if not corrections:
            return None
        
        # Apply corrections (case-insensitive replace)
        corrected_query = query
        for corr in corrections:
            # Use regex for case-insensitive replacement
            import re
            pattern = re.compile(re.escape(corr['original']), re.IGNORECASE)
            corrected_query = pattern.sub(corr['corrected'], corrected_query, count=1)
        
        self.stats["spell_corrections_made"] += 1
        
        logger.info(f"Spell check: '{query}' → '{corrected_query}'")
        
        return {
            "corrected_query": corrected_query,
            "confidence": sum(c['confidence'] for c in corrections) / len(corrections),
            "changes": corrections
        }
    
    async def suggest_related(
        self,
        query: str,
        response: str,
        signals: Dict[str, Any],
        language: str = "en",
        max_suggestions: int = 3
    ) -> List[str]:
        """
        Generate related query suggestions (LLM-based).
        
        Cached for performance.
        
        Args:
            query: Original user query
            response: System response
            signals: Detected signals/intents
            language: Language code
            max_suggestions: Maximum suggestions to return
        
        Returns:
            List of related query suggestions
        """
        self.stats["related_query_requests"] += 1
        self.stats["total_suggestions"] += 1
        
        # Check cache
        cache_key = f"related:{language}:{hashlib.md5(query.encode()).hexdigest()}"
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    self.stats["cache_hits"] += 1
                    logger.debug(f"Cache hit for related queries: '{query[:30]}...'")
                    return json.loads(cached)
                else:
                    self.stats["cache_misses"] += 1
            except Exception as e:
                logger.warning(f"Cache retrieval error: {e}")
        
        # Build prompt for LLM
        prompt = self._build_related_queries_prompt(query, response, signals, language)
        
        try:
            # Call LLM
            self.stats["llm_calls"] += 1
            start_time = time.time()
            
            llm_response = await self.llm.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            
            llm_time = time.time() - start_time
            
            # Parse suggestions
            suggestions = self._parse_suggestions(
                llm_response.choices[0].message.content,
                max_suggestions
            )
            
            # Cache results
            if self.redis and suggestions:
                try:
                    self.redis.setex(cache_key, self.cache_ttl, json.dumps(suggestions))
                except Exception as e:
                    logger.warning(f"Cache storage error: {e}")
            
            logger.info(
                f"Generated {len(suggestions)} related queries in {llm_time:.2f}s (lang: {language})"
            )
            
            return suggestions
        
        except Exception as e:
            logger.error(f"Failed to generate related queries: {e}")
            return []
    
    def _build_related_queries_prompt(
        self,
        query: str,
        response: str,
        signals: Dict[str, Any],
        language: str
    ) -> str:
        """Build prompt for generating related queries."""
        
        detected_signals = signals.get("detected_signals", [])
        primary_intent = signals.get("primary_intent", "general")
        
        # Language-specific instructions
        if language.lower() == "tr":
            lang_instruction = "Generate suggestions in Turkish (Türkçe)."
            format_instruction = "Sadece soruları listele, numaralandırma olmadan."
        else:
            lang_instruction = "Generate suggestions in English."
            format_instruction = "List only the questions, no numbering."
        
        prompt = f"""Based on this conversation, suggest 3 related follow-up questions a user might ask.

USER ASKED: "{query}"
DETECTED INTENT: {primary_intent}
DETECTED SIGNALS: {", ".join(detected_signals) if detected_signals else "general"}

SYSTEM ANSWERED: {response[:200]}...

YOUR TASK: Suggest 3 natural follow-up questions that would be helpful.

{lang_instruction}
{format_instruction}
Keep questions short and natural.

Example format:
What are the opening hours?
How much does it cost?
Is it wheelchair accessible?

Generate 3 related questions:"""
        
        return prompt
    
    def _parse_suggestions(self, llm_text: str, max_suggestions: int = 3) -> List[str]:
        """Parse LLM output to extract suggestions."""
        suggestions = []
        
        # Split by lines
        lines = llm_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove numbering (1., 2., -, *, etc.)
            line = line.lstrip('0123456789.-*• ')
            
            # Remove quotes
            line = line.strip('"\'')
            
            if line and line.endswith('?'):
                suggestions.append(line)
            elif line and len(line) > 10:  # Reasonable question length
                # Add question mark if missing
                if not line.endswith('?'):
                    line += '?'
                suggestions.append(line)
            
            if len(suggestions) >= max_suggestions:
                break
        
        return suggestions[:max_suggestions]
    
    def track_query(self, query: str):
        """
        Track query for autocomplete ranking.
        
        Args:
            query: User query to track
        """
        if not query or len(query) < 3:
            return
        
        query = query.strip()
        
        # Update frequency
        self.query_frequencies[query] += 1
        
        # Insert into trie
        self.trie.insert(query, 1)
        
        # Periodically save to Redis (every 100 queries)
        if len(self.query_frequencies) % 100 == 0:
            self._save_popular_queries()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get suggester statistics."""
        total_requests = self.stats["total_suggestions"]
        if total_requests > 0:
            cache_hit_rate = (self.stats["cache_hits"] / total_requests) * 100
        else:
            cache_hit_rate = 0.0
        
        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "tracked_queries": len(self.query_frequencies),
            "trie_size": self.trie.total_queries
        }
    
    def clear_cache(self) -> int:
        """Clear all cached suggestions. Returns count of cleared items."""
        if not self.redis:
            return 0
        
        try:
            # Find all suggestion cache keys
            keys = []
            cursor = 0
            while True:
                cursor, partial_keys = self.redis.scan(
                    cursor,
                    match="related:*",
                    count=100
                )
                keys.extend(partial_keys)
                if cursor == 0:
                    break
            
            # Delete all keys
            if keys:
                count = self.redis.delete(*keys)
                logger.info("✅ Cleared %d cached suggestions", count)
                return count
            else:
                logger.info("✅ No cached suggestions to clear")
                return 0
        
        except Exception as e:
            logger.error("❌ Error clearing cache: %s", str(e))
            return 0


# Convenience function
def create_query_suggester(
    llm_client: Any,
    redis_url: Optional[str] = None,
    location_names: Optional[List[str]] = None
) -> QuerySuggester:
    """
    Create QuerySuggester with default settings.
    
    Args:
        llm_client: LLM client for generating suggestions
        redis_url: Optional Redis URL for caching
        location_names: Optional list of location names
    
    Returns:
        Configured QuerySuggester instance
    """
    redis_client = None
    if redis_url and REDIS_AVAILABLE:
        try:
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            logger.info("✅ Connected to Redis for query suggestions")
        except Exception as e:
            logger.warning("⚠️ Failed to connect to Redis: %s", str(e))
            redis_client = None
    
    return QuerySuggester(
        llm_client=llm_client,
        redis_client=redis_client,
        location_names=location_names
    )
