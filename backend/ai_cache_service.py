#!/usr/bin/env python3
"""
AI Rate Limiting and Caching Service for Istanbul AI - Working Version

This module implements rate limiting and caching for GPT API calls.
"""

import os
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

@dataclass
class RateLimitConfig:
    """Configuration for AI API rate limiting"""
    requests_per_minute: int = 30
    requests_per_hour: int = 500
    requests_per_day: int = 2000
    cache_ttl_seconds: int = 3600
    similarity_threshold: float = 0.85

# In-memory storage for rate limiting and caching
rate_limit_store = defaultdict(lambda: defaultdict(int))
cache_store = OrderedDict()
cache_timestamps = {}

class SimpleAIRateLimiter:
    """Simple in-memory rate limiter for AI API calls"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
    
    def _get_time_window(self, window_type: str) -> str:
        """Get current time window string"""
        now = datetime.utcnow()
        if window_type == "minute":
            return now.strftime("%Y%m%d%H%M")
        elif window_type == "hour":
            return now.strftime("%Y%m%d%H") 
        else:  # day
            return now.strftime("%Y%m%d")
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        limits = {
            "minute": self.config.requests_per_minute,
            "hour": self.config.requests_per_hour,
            "day": self.config.requests_per_day
        }
        
        current_counts = {}
        
        for window, limit in limits.items():
            time_window = self._get_time_window(window)
            key = f"{identifier}:{window}:{time_window}"
            count = rate_limit_store[identifier][key]
            current_counts[window] = count
            
            if count >= limit:
                return False, {
                    "allowed": False,
                    "reason": f"Rate limit exceeded for {window}",
                    "current_count": count,
                    "limit": limit
                }
        
        return True, {
            "allowed": True,
            "current_counts": current_counts,
            "limits": limits
        }
    
    def increment_usage(self, identifier: str):
        """Increment usage counters"""
        windows = ["minute", "hour", "day"]
        
        for window in windows:
            time_window = self._get_time_window(window)
            key = f"{identifier}:{window}:{time_window}"
            rate_limit_store[identifier][key] += 1

class SimpleAICache:
    """Simple in-memory cache for AI responses"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.max_cache_size = 1000  # Prevent memory issues
    
    def _generate_cache_key(self, query: str, language: str = "en") -> str:
        """Generate cache key from query"""
        normalized = query.lower().strip()
        normalized = normalized.replace("?", "").replace("!", "").replace(".", "")
        normalized = " ".join(normalized.split())
        cache_string = f"{language}:{normalized}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, timestamp in cache_timestamps.items():
            if current_time - timestamp > timedelta(seconds=self.config.cache_ttl_seconds):
                expired_keys.append(key)
        
        for key in expired_keys:
            cache_store.pop(key, None)
            cache_timestamps.pop(key, None)
    
    def _ensure_cache_size(self):
        """Ensure cache doesn't exceed max size"""
        if len(cache_store) > self.max_cache_size:
            # Remove oldest entries
            for _ in range(len(cache_store) - self.max_cache_size + 100):
                if cache_store:
                    oldest_key = next(iter(cache_store))
                    cache_store.pop(oldest_key, None)
                    cache_timestamps.pop(oldest_key, None)
    
    def get_cached_response(self, query: str, language: str = "en") -> Optional[str]:
        """Get cached response if available"""
        self._cleanup_expired()
        
        cache_key = self._generate_cache_key(query, language)
        
        if cache_key in cache_store:
            # Move to end (LRU)
            response = cache_store.pop(cache_key)
            cache_store[cache_key] = response
            return response
        
        # Try similarity matching for basic queries
        return self._find_similar_response(query, language)
    
    def _find_similar_response(self, query: str, language: str) -> Optional[str]:
        """Find similar cached response"""
        query_words = set(query.lower().split())
        
        for cached_key, cached_data in cache_store.items():
            if cached_data.get('language') == language:
                cached_query = cached_data.get('original_query', '')
                cached_words = set(cached_query.lower().split())
                
                # Simple similarity check
                intersection = len(query_words.intersection(cached_words))
                union = len(query_words.union(cached_words))
                
                if union > 0:
                    similarity = intersection / union
                    if similarity >= self.config.similarity_threshold:
                        return cached_data.get('response')
        
        return None
    
    def cache_response(self, query: str, response: str, language: str = "en"):
        """Cache AI response"""
        self._cleanup_expired()
        self._ensure_cache_size()
        
        cache_key = self._generate_cache_key(query, language)
        
        cache_store[cache_key] = {
            'response': response,
            'original_query': query,
            'language': language,
            'timestamp': datetime.utcnow().isoformat()
        }
        cache_timestamps[cache_key] = datetime.utcnow()

class AIServiceManager:
    """Main AI service manager"""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.rate_limiter = SimpleAIRateLimiter(self.config)
        self.cache = SimpleAICache(self.config)
        self.usage_stats = defaultdict(int)
    
    def get_user_identifier(self, session_id: str, user_ip: str) -> str:
        """Generate user identifier"""
        if session_id and session_id != "default":
            return f"session:{session_id}"
        return f"ip:{user_ip.replace('.', '_')}"  # Make IP safe for keys
    
    def can_make_ai_request(self, session_id: str, user_ip: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if user can make AI request"""
        identifier = self.get_user_identifier(session_id, user_ip)
        return self.rate_limiter.check_rate_limit(identifier)
    
    def get_cached_response(self, query: str, language: str = "en") -> Optional[str]:
        """Get cached response"""
        return self.cache.get_cached_response(query, language)
    
    def record_ai_request(self, session_id: str, user_ip: str, query: str, 
                         response: str, cached: bool = False, **kwargs):
        """Record AI request"""
        identifier = self.get_user_identifier(session_id, user_ip)
        
        # Update usage stats
        self.usage_stats['total_requests'] += 1
        if cached:
            self.usage_stats['cached_requests'] += 1
        
        # Only count against rate limit if not cached
        if not cached:
            self.rate_limiter.increment_usage(identifier)
            
        # Cache the response for future use
        if not cached and response:
            self.cache.cache_response(query, response, kwargs.get('language', 'en'))
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        total = self.usage_stats['total_requests']
        cached = self.usage_stats['cached_requests']
        cache_hit_rate = (cached / total * 100) if total > 0 else 0
        
        return {
            "total_requests": total,
            "cached_requests": cached,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "cache_size": len(cache_store),
            "rate_limit_store_size": len(rate_limit_store)
        }
    
    def clear_cache(self):
        """Clear all cache"""
        cache_store.clear()
        cache_timestamps.clear()
        print("🧹 AI cache cleared")
    
    def clear_rate_limits(self):
        """Clear rate limit counters"""
        rate_limit_store.clear()
        print("🔄 Rate limit counters cleared")

# Global instance
ai_service_manager = AIServiceManager()

def get_ai_service() -> AIServiceManager:
    """Get the global AI service manager"""
    return ai_service_manager
