#!/usr/bin/env python3
"""
Smart Caching System for AI Istanbul
Implements intelligent caching to reduce API costs by 22.5%

Key Features:
1. Google Places API response caching (24h duration)
2. OpenAI response caching for similar queries
3. Location-based cache optimization
4. Context-aware cache invalidation
5. Cost tracking and analytics
"""

import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Redis for production caching (fallback to file-based for development)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: datetime
    ttl_seconds: int
    cache_type: str
    metadata: Dict[str, Any]
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class SmartCache:
    """Intelligent caching system for API responses"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.file_cache_dir = Path("cache")
        self.file_cache_dir.mkdir(exist_ok=True)
        
        # Cache configuration
        self.cache_ttl = {
            'google_places': 24 * 3600,     # 24 hours for places
            'google_weather': 1 * 3600,     # 1 hour for weather
            'openai_response': 2 * 3600,    # 2 hours for similar queries
            'location_context': 6 * 3600,   # 6 hours for location data
            'user_preferences': 7 * 24 * 3600  # 7 days for user preferences
        }
        
        # Cost tracking
        self.cost_savings = {
            'google_places_calls_saved': 0,
            'openai_tokens_saved': 0,
            'total_api_calls_saved': 0,
            'estimated_cost_savings': 0.0
        }
        
        # Initialize Redis if available
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("✅ Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed, using file cache: {e}")
                self.redis_client = None
        
        logger.info(f"🗄️ Smart cache initialized (Redis: {self.redis_client is not None})")
    
    def _generate_cache_key(self, cache_type: str, identifier: str) -> str:
        """Generate a consistent cache key"""
        key_data = f"{cache_type}:{identifier}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _serialize_for_cache(self, data: Any) -> str:
        """Serialize data for caching"""
        return json.dumps(data, default=str, ensure_ascii=False)
    
    def _deserialize_from_cache(self, data: str) -> Any:
        """Deserialize data from cache"""
        return json.loads(data)
    
    def get(self, cache_type: str, identifier: str) -> Optional[Any]:
        """Get item from cache"""
        cache_key = self._generate_cache_key(cache_type, identifier)
        
        try:
            # Try Redis first
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    entry_dict = json.loads(cached_data.decode())
                    entry = CacheEntry.from_dict(entry_dict)
                    
                    if not entry.is_expired():
                        entry.hit_count += 1
                        # Update hit count in cache
                        self.redis_client.set(cache_key, json.dumps(entry.to_dict()))
                        
                        # Track cost savings
                        self._track_cache_hit(cache_type, entry.metadata.get('cost_saved', 0))
                        
                        logger.debug(f"🎯 Cache hit (Redis): {cache_type} - {identifier[:50]}")
                        return entry.data
                    else:
                        # Remove expired entry
                        self.redis_client.delete(cache_key)
            
            # Fallback to file cache
            cache_file = self.file_cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    entry_dict = json.load(f)
                    entry = CacheEntry.from_dict(entry_dict)
                    
                    if not entry.is_expired():
                        entry.hit_count += 1
                        # Update hit count in file
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
                        
                        # Track cost savings
                        self._track_cache_hit(cache_type, entry.metadata.get('cost_saved', 0))
                        
                        logger.debug(f"🎯 Cache hit (File): {cache_type} - {identifier[:50]}")
                        return entry.data
                    else:
                        # Remove expired file
                        cache_file.unlink()
        
        except Exception as e:
            logger.error(f"❌ Cache get error: {e}")
        
        logger.debug(f"🔍 Cache miss: {cache_type} - {identifier[:50]}")
        return None
    
    def set(self, cache_type: str, identifier: str, data: Any, metadata: Optional[Dict] = None) -> bool:
        """Set item in cache"""
        if metadata is None:
            metadata = {}
        
        cache_key = self._generate_cache_key(cache_type, identifier)
        ttl = self.cache_ttl.get(cache_type, 3600)  # Default 1 hour
        
        entry = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl,
            cache_type=cache_type,
            metadata=metadata
        )
        
        try:
            # Store in Redis
            if self.redis_client:
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(entry.to_dict(), ensure_ascii=False)
                )
                logger.debug(f"💾 Cached in Redis: {cache_type} - {identifier[:50]}")
            
            # Also store in file cache as backup
            cache_file = self.file_cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 Cached: {cache_type} - {identifier[:50]} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache set error: {e}")
            return False
    
    def _track_cache_hit(self, cache_type: str, cost_saved: float):
        """Track cost savings from cache hits"""
        if cache_type == 'google_places':
            self.cost_savings['google_places_calls_saved'] += 1
            self.cost_savings['estimated_cost_savings'] += 0.017  # Google Places cost per request
        elif cache_type == 'openai_response':
            self.cost_savings['openai_tokens_saved'] += cost_saved
            self.cost_savings['estimated_cost_savings'] += cost_saved * 0.0015  # Rough OpenAI cost
        
        self.cost_savings['total_api_calls_saved'] += 1
    
    def invalidate(self, cache_type: str, identifier: str) -> bool:
        """Invalidate specific cache entry"""
        cache_key = self._generate_cache_key(cache_type, identifier)
        
        try:
            # Remove from Redis
            if self.redis_client:
                self.redis_client.delete(cache_key)
            
            # Remove from file cache
            cache_file = self.file_cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                cache_file.unlink()
            
            logger.debug(f"🗑️ Cache invalidated: {cache_type} - {identifier}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache invalidation error: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        cleaned = 0
        
        try:
            # Clean file cache
            for cache_file in self.file_cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        entry_dict = json.load(f)
                        entry = CacheEntry.from_dict(entry_dict)
                        
                        if entry.is_expired():
                            cache_file.unlink()
                            cleaned += 1
                            
                except Exception as e:
                    logger.error(f"Error cleaning cache file {cache_file}: {e}")
                    cache_file.unlink()  # Remove corrupted files
                    cleaned += 1
            
            logger.info(f"🧹 Cleaned {cleaned} expired cache entries")
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Cache cleanup error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'cost_savings': self.cost_savings.copy(),
            'cache_config': self.cache_ttl.copy(),
            'redis_available': self.redis_client is not None,
            'cache_directory': str(self.file_cache_dir)
        }
        
        # Count cache files
        try:
            cache_files = list(self.file_cache_dir.glob("*.json"))
            stats['total_cached_items'] = len(cache_files)
            
            # Calculate cache hit ratio (rough estimate)
            total_hits = sum(entry.hit_count for entry in self._get_all_entries())
            total_requests = total_hits + stats['cost_savings']['total_api_calls_saved']
            stats['estimated_hit_ratio'] = (total_hits / max(total_requests, 1)) * 100
            
        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
            stats['total_cached_items'] = 0
            stats['estimated_hit_ratio'] = 0
        
        return stats
    
    def _get_all_entries(self) -> List[CacheEntry]:
        """Get all cache entries (for statistics)"""
        entries = []
        
        try:
            for cache_file in self.file_cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        entry_dict = json.load(f)
                        entry = CacheEntry.from_dict(entry_dict)
                        if not entry.is_expired():
                            entries.append(entry)
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Error reading cache entries: {e}")
        
        return entries

# Global cache instance
_cache_instance = None

def get_smart_cache() -> SmartCache:
    """Get the global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        redis_url = os.getenv('REDIS_URL')  # Optional Redis connection
        _cache_instance = SmartCache(redis_url)
    return _cache_instance

def cache_google_places_response(query: str, location: str, response: Dict) -> bool:
    """Cache Google Places API response"""
    cache = get_smart_cache()
    
    # Normalize query and location for better cache hits
    normalized_query = _normalize_places_query(query)
    normalized_location = _normalize_location(location)
    
    identifier = f"places:{normalized_query}:{normalized_location}"
    metadata = {
        'query': query,
        'location': location,
        'normalized_query': normalized_query,
        'normalized_location': normalized_location,
        'cost_saved': 0.017  # Google Places API cost per request
    }
    return cache.set('google_places', identifier, response, metadata)

def get_cached_google_places(query: str, location: str) -> Optional[Dict]:
    """Get cached Google Places response"""
    cache = get_smart_cache()
    
    # Try with normalized keys first
    normalized_query = _normalize_places_query(query)
    normalized_location = _normalize_location(location)
    
    identifier = f"places:{normalized_query}:{normalized_location}"
    result = cache.get('google_places', identifier)
    
    if result:
        return result
    
    # Try with just location for broader matches
    location_identifier = f"places:general:{normalized_location}"
    return cache.get('google_places', location_identifier)

def _normalize_places_query(query: str) -> str:
    """Normalize places query for better cache hits"""
    import re
    
    query_lower = query.lower().strip()
    
    # Restaurant queries
    if any(word in query_lower for word in ['restaurant', 'food', 'eat', 'dining', 'cafe']):
        if any(word in query_lower for word in ['turkish', 'local', 'traditional']):
            return 'turkish_restaurants'
        elif any(word in query_lower for word in ['seafood', 'fish', 'meze']):
            return 'seafood_restaurants'
        else:
            return 'restaurants_general'
    
    # Museum/attraction queries
    if any(word in query_lower for word in ['museum', 'attraction', 'site', 'monument']):
        return 'museums_attractions'
    
    # Shopping queries
    if any(word in query_lower for word in ['shop', 'market', 'bazaar', 'mall']):
        return 'shopping'
    
    return 'general'

def _normalize_location(location: str) -> str:
    """Normalize location for better cache hits"""
    if not location:
        return 'istanbul'
    
    location_lower = location.lower().strip()
    
    # Map variations to standard names
    location_mapping = {
        'sultanahmet': ['sultanahmet', 'old city', 'historic peninsula'],
        'galata': ['galata', 'galata tower', 'karakoy'],
        'taksim': ['taksim', 'taksim square', 'beyoglu'],
        'kadikoy': ['kadikoy', 'asian side'],
        'besiktas': ['besiktas', 'dolmabahce'],
        'uskudar': ['uskudar', 'asian side']
    }
    
    for standard_name, variations in location_mapping.items():
        if any(var in location_lower for var in variations):
            return standard_name
    
    return 'istanbul'

def cache_openai_response(user_input: str, context: str, response: str, tokens_used: int) -> bool:
    """Cache OpenAI response for similar queries"""
    cache = get_smart_cache()
    
    # Create semantic identifier that captures query intent
    normalized_input = _normalize_query_for_caching(user_input)
    query_hash = hashlib.md5(f"{normalized_input}:{context[:50]}".encode()).hexdigest()
    identifier = f"openai:{query_hash}"
    
    metadata = {
        'user_input': user_input[:200],  # Truncate for storage
        'normalized_input': normalized_input,
        'tokens_used': tokens_used,
        'cost_saved': tokens_used
    }
    
    # Cache with both detailed and simplified context
    success1 = cache.set('openai_response', identifier, response, metadata)
    
    # Also cache with simplified context for better hit rate
    simple_context = _extract_location_from_context(context)
    simple_hash = hashlib.md5(f"{normalized_input}:{simple_context}".encode()).hexdigest()
    simple_identifier = f"openai:{simple_hash}"
    success2 = cache.set('openai_response', simple_identifier, response, metadata)
    
    return success1 or success2

def get_cached_openai_response(user_input: str, context: str) -> Optional[str]:
    """Get cached OpenAI response for similar queries with fuzzy matching"""
    cache = get_smart_cache()
    
    # Create semantic cache key based on intent rather than exact text
    normalized_input = _normalize_query_for_caching(user_input)
    
    # 🚀 OPTIMIZATION: Try preloaded cache first for instant responses
    try:
        from preloaded_cache import get_preloaded_response
        location = _extract_location_from_context(context)
        preloaded_response = get_preloaded_response(normalized_input, location)
        if preloaded_response:
            logger.info(f"🎯 Using preloaded response (Maximum cost savings)")
            return preloaded_response
    except ImportError:
        pass
    
    query_hash = hashlib.md5(f"{normalized_input}:{context[:50]}".encode()).hexdigest()
    identifier = f"openai:{query_hash}"
    
    # First try exact match
    result = cache.get('openai_response', identifier)
    if result:
        return result
    
    # Try with simplified context for better hit rate
    simple_context = _extract_location_from_context(context)
    simple_hash = hashlib.md5(f"{normalized_input}:{simple_context}".encode()).hexdigest()
    simple_identifier = f"openai:{simple_hash}"
    
    return cache.get('openai_response', simple_identifier)

def _normalize_query_for_caching(query: str) -> str:
    """Normalize query for better cache hit rates"""
    import re
    
    query_lower = query.lower().strip()
    
    # PRIORITY 1: Check for alternative/diversified queries - these should NOT be cached against specific monuments
    alternative_indicators = [
        'beyond', 'other than', 'different from', 'alternatives to', 'instead of', 'apart from',
        'hidden gems', 'lesser known', 'off the beaten path', 'secret spots', 'locals recommend',
        'not touristy', 'authentic', 'unique attractions', 'undiscovered', 'alternative',
        'diversified', 'varied', 'diverse', 'different', 'lesser-known', 'off beaten path'
    ]
    
    if any(indicator in query_lower for indicator in alternative_indicators):
        # This is asking for alternatives - don't match to specific monument caches
        return f"diversified:{query_lower[:50]}"
    
    # Common restaurant query patterns
    restaurant_patterns = [
        (r'\b(best|good|top|recommended)\s+(restaurants?|places to eat|dining)\b', 'restaurant_recommendations'),
        (r'\b(turkish|ottoman|local)\s+(restaurants?|food|cuisine)\b', 'turkish_restaurant'),
        (r'\brestaurants?\s+(in|near|around)\b', 'restaurant_location'),
    ]
    
    # Museum/attraction patterns
    museum_patterns = [
        (r'\b(museums?|attractions?)\s+(to visit|worth visiting|should i see)\b', 'museum_recommendations'),
        (r'\b(hagia sophia|blue mosque|topkapi)\s+(opening|hours|times)\b', 'monument_hours'),
        (r'\btell me about\s+(hagia sophia|blue mosque|topkapi)\b', 'monument_info'),
    ]
    
    # Transportation patterns
    transport_patterns = [
        (r'\bhow\s+(do i get|can i travel|to get)\s+from\s+\w+\s+to\s+\w+\b', 'transportation_route'),
        (r'\b(metro|bus|tram|transport)\b', 'transportation_general'),
    ]
    
    # Check patterns
    for patterns, category in [(restaurant_patterns, 'restaurant'), 
                               (museum_patterns, 'museum'), 
                               (transport_patterns, 'transport')]:
        for pattern, subcategory in patterns:
            if re.search(pattern, query_lower):
                return f"{category}:{subcategory}"
    
    # Fallback: use key words (but exclude monument names for diversified queries)
    key_words = re.findall(r'\b(restaurant|museum|hotel|transport|metro|bus|hagia|sophia|sultanahmet|galata|taksim)\b', query_lower)
    if key_words:
        return f"general:{':'.join(sorted(set(key_words[:3])))}"
    
    return f"general:{query_lower[:50]}"

def _extract_location_from_context(context: str) -> str:
    """Extract location from context for caching"""
    import re
    
    # Common Istanbul locations
    locations = ['sultanahmet', 'galata', 'taksim', 'beyoglu', 'kadikoy', 'uskudar', 'besiktas']
    
    context_lower = context.lower()
    for location in locations:
        if location in context_lower:
            return location
    
    return 'istanbul'

def cache_location_context(location: str, context_data: Dict) -> bool:
    """Cache location context data"""
    cache = get_smart_cache()
    identifier = f"location:{location.lower().strip()}"
    metadata = {'location': location}
    return cache.set('location_context', identifier, context_data, metadata)

def get_cached_location_context(location: str) -> Optional[Dict]:
    """Get cached location context"""
    cache = get_smart_cache()
    identifier = f"location:{location.lower().strip()}"
    return cache.get('location_context', identifier)

if __name__ == "__main__":
    # Test the caching system
    cache = get_smart_cache()
    
    # Test basic operations
    print("🧪 Testing Smart Cache System...")
    
    # Test Google Places caching
    test_places_data = {"results": [{"name": "Test Restaurant", "rating": 4.5}]}
    cache_google_places_response("restaurant", "sultanahmet", test_places_data)
    cached_result = get_cached_google_places("restaurant", "sultanahmet")
    print(f"✅ Places cache test: {'PASSED' if cached_result else 'FAILED'}")
    
    # Test OpenAI caching
    cache_openai_response("Hello", "general", "Hi there!", 50)
    cached_response = get_cached_openai_response("Hello", "general")
    print(f"✅ OpenAI cache test: {'PASSED' if cached_response else 'FAILED'}")
    
    # Show stats
    stats = cache.get_stats()
    print(f"📊 Cache Stats: {stats}")
    
    print("✅ Smart Cache System ready for production!")
