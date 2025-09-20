#!/usr/bin/env python3
"""
AI Rate Limiting and Caching Service for Istanbul AI

This module implements intelligent rate limiting and caching for GPT API calls
to optimize performance, reduce costs, and prevent API abuse.
"""

import os
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from database import Base, SessionLocal, engine
from collections import defaultdict

# Simple in-memory cache with Redis fallback
try:
    import redis
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD'),
        decode_responses=True,
        socket_connect_timeout=5
    )
    # Test connection
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("✅ Redis connected for AI caching")
except Exception as e:
    print(f"⚠️ Redis not available, using in-memory cache: {e}")
    REDIS_AVAILABLE = False
    redis_client = None

# In-memory cache systems
memory_cache = {}
rate_limit_cache = defaultdict(lambda: defaultdict(int))
cache_timestamps = {}

@dataclass
class RateLimitConfig:
    """Configuration for AI API rate limiting"""
    requests_per_minute: int = 30
    requests_per_hour: int = 500
    requests_per_day: int = 2000
    burst_limit: int = 5
    cache_ttl_seconds: int = 3600
    similarity_threshold: float = 0.85

@dataclass 
class CacheEntry:
    """Cached AI response entry"""
    query_hash: str
    original_query: str
    response: str
    timestamp: datetime
    hit_count: int = 1
    language: str = "en"
    model_used: str = "gpt-3.5-turbo"
    
class AIUsageTracking(Base):
    """Database model for tracking AI API usage"""
    __tablename__ = "ai_usage_tracking"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_ip = Column(String(50), index=True)
    query_hash = Column(String(64), index=True)
    original_query = Column(Text)
    response_cached = Column(Boolean, default=False)
    model_used = Column(String(50), default="gpt-3.5-turbo")
    tokens_used = Column(Integer)
    response_time_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class AIRateLimiter:
    """Advanced rate limiter for AI API calls"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
    
    def _get_time_window_key(self, identifier: str, window_type: str) -> str:
        """Generate time-based key for rate limiting"""
        now = datetime.utcnow()
        if window_type == "minute":
            time_key = now.strftime("%Y%m%d%H%M")
        elif window_type == "hour":
            time_key = now.strftime("%Y%m%d%H")
        else:  # day
            time_key = now.strftime("%Y%m%d")
        return f"{identifier}:{window_type}:{time_key}"
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        limits = {
            "minute": self.config.requests_per_minute,
            "hour": self.config.requests_per_hour,
            "day": self.config.requests_per_day
        }
        
        current_counts = {}
        
        for window, limit in limits.items():
            key = self._get_time_window_key(identifier, window)
            
            if REDIS_AVAILABLE and redis_client:
                try:
                    count = redis_client.get(key)
                    count = int(count) if count else 0
                except Exception:
                    count = rate_limit_cache[identifier][window]
            else:
                count = rate_limit_cache[identifier][window]
            
            current_counts[window] = count
            
            if count >= limit:
                return False, {
                    "allowed": False,
                    "reason": f"Rate limit exceeded for {window}",
                    "current_count": count,
                    "limit": limit,
                    "reset_time": self._get_reset_time(window)
                }
        
        return True, {
            "allowed": True,
            "current_counts": current_counts,
            "limits": limits
        }
    
    def increment_usage(self, identifier: str):
        """Increment usage counters for all time windows"""
        windows = ["minute", "hour", "day"]
        
        for window in windows:
            key = self._get_time_window_key(identifier, window)
            
            if REDIS_AVAILABLE and redis_client:
                try:
                    redis_client.incr(key)
                    if window == "minute":
                        redis_client.expire(key, 120)
                    elif window == "hour":
                        redis_client.expire(key, 7200)
                    else:  # day
                        redis_client.expire(key, 172800)
                except Exception:
                    rate_limit_cache[identifier][window] += 1
            else:
                rate_limit_cache[identifier][window] += 1
    
    def _get_reset_time(self, window: str) -> str:
        """Get reset time for the given window"""
        now = datetime.utcnow()
        if window == "minute":
            reset = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        elif window == "hour":
            reset = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:  # day
            reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return reset.isoformat()

class AIResponseCache:
    """Intelligent caching system for AI responses"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
    
    def _generate_query_hash(self, query: str, language: str = "en") -> str:
        """Generate hash for query normalization and caching"""
        # Normalize query for better cache hits
        normalized = query.lower().strip()
        normalized = normalized.replace("?", "").replace("!", "").replace(".", "")
        normalized = " ".join(normalized.split())
        
        cache_key = f"{language}:{normalized}"
        return hashlib.sha256(cache_key.encode()).hexdigest()
    
    def _similarity_score(self, query1: str, query2: str) -> float:
        """Calculate similarity score between two queries"""
        try:
            from fuzzywuzzy import fuzz
            return fuzz.ratio(query1.lower(), query2.lower()) / 100.0
        except ImportError:
            # Basic fallback similarity
            q1_words = set(query1.lower().split())
            q2_words = set(query2.lower().split())
            intersection = len(q1_words.intersection(q2_words))
            union = len(q1_words.union(q2_words))
            return intersection / union if union > 0 else 0.0
    
    def get_cached_response(self, query: str, language: str = "en") -> Optional[CacheEntry]:
        """Retrieve cached response if available"""
        query_hash = self._generate_query_hash(query, language)
        cache_key = f"ai_cache:{query_hash}"
        
        # Try Redis first
        if REDIS_AVAILABLE and redis_client:
            try:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    cache_entry = CacheEntry(
                        query_hash=data['query_hash'],
                        original_query=data['original_query'],
                        response=data['response'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        hit_count=data.get('hit_count', 1),
                        language=data.get('language', 'en'),
                        model_used=data.get('model_used', 'gpt-3.5-turbo')
                    )
                    # Update hit count
                    cache_entry.hit_count += 1
                    self._update_cache_entry(cache_entry)
                    return cache_entry
            except Exception as e:
                print(f"Redis cache retrieval error: {e}")
        
        # Try in-memory cache
        if cache_key in memory_cache:
            entry_data = memory_cache[cache_key]
            # Check if not expired
            if datetime.utcnow() - entry_data['timestamp'] < timedelta(seconds=self.config.cache_ttl_seconds):
                entry_data['hit_count'] = entry_data.get('hit_count', 0) + 1
                return CacheEntry(
                    query_hash=entry_data['query_hash'],
                    original_query=entry_data['original_query'],
                    response=entry_data['response'],
                    timestamp=entry_data['timestamp'],
                    hit_count=entry_data['hit_count'],
                    language=entry_data.get('language', 'en'),
                    model_used=entry_data.get('model_used', 'gpt-3.5-turbo')
                )
        
        # Try to find similar queries
        return self._find_similar_cached_response(query, language)
    
    def _find_similar_cached_response(self, query: str, language: str) -> Optional[CacheEntry]:
        """Find cached responses for similar queries"""
        best_match = None
        best_score = 0.0
        
        # Search in-memory cache for similar queries
        for cache_key, entry_data in memory_cache.items():
            if entry_data.get('language', 'en') == language:
                # Check if not expired
                if datetime.utcnow() - entry_data['timestamp'] < timedelta(seconds=self.config.cache_ttl_seconds):
                    score = self._similarity_score(query, entry_data['original_query'])
                    if score > best_score and score >= self.config.similarity_threshold:
                        best_score = score
                        best_match = CacheEntry(
                            query_hash=entry_data['query_hash'],
                            original_query=entry_data['original_query'],
                            response=entry_data['response'],
                            timestamp=entry_data['timestamp'],
                            hit_count=entry_data.get('hit_count', 1),
                            language=entry_data.get('language', 'en'),
                            model_used=entry_data.get('model_used', 'gpt-3.5-turbo')
                        )
        
        return best_match
    
    def cache_response(self, query: str, response: str, language: str = "en", model: str = "gpt-3.5-turbo"):
        """Cache AI response for future use"""
        query_hash = self._generate_query_hash(query, language)
        cache_key = f"ai_cache:{query_hash}"
        
        cache_entry = CacheEntry(
            query_hash=query_hash,
            original_query=query,
            response=response,
            timestamp=datetime.utcnow(),
            language=language,
            model_used=model
        )
        
        cache_data = {
            'query_hash': cache_entry.query_hash,
            'original_query': cache_entry.original_query,
            'response': cache_entry.response,
            'timestamp': cache_entry.timestamp.isoformat(),
            'hit_count': cache_entry.hit_count,
            'language': cache_entry.language,
            'model_used': cache_entry.model_used
        }
        
        # Store in Redis if available
        if REDIS_AVAILABLE and redis_client:
            try:
                redis_client.setex(cache_key, self.config.cache_ttl_seconds, json.dumps(cache_data))
            except Exception as e:
                print(f"Redis cache storage error: {e}")
        
        # Always store in memory as backup
        memory_cache[cache_key] = {
            'query_hash': cache_entry.query_hash,
            'original_query': cache_entry.original_query,
            'response': cache_entry.response,
            'timestamp': cache_entry.timestamp,
            'hit_count': cache_entry.hit_count,
            'language': cache_entry.language,
            'model_used': cache_entry.model_used
        }
    
    def _update_cache_entry(self, cache_entry: CacheEntry):
        """Update cache entry hit count"""
        cache_key = f"ai_cache:{cache_entry.query_hash}"
        
        cache_data = {
            'query_hash': cache_entry.query_hash,
            'original_query': cache_entry.original_query,
            'response': cache_entry.response,
            'timestamp': cache_entry.timestamp.isoformat(),
            'hit_count': cache_entry.hit_count,
            'language': cache_entry.language,
            'model_used': cache_entry.model_used
        }
        
        if REDIS_AVAILABLE and redis_client:
            try:
                redis_client.setex(cache_key, self.config.cache_ttl_seconds, json.dumps(cache_data))
            except Exception:
                pass
        
        # Update in-memory cache
        if cache_key in memory_cache:
            memory_cache[cache_key]['hit_count'] = cache_entry.hit_count

class AIServiceManager:
    """Main service manager for AI rate limiting and caching"""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.rate_limiter = AIRateLimiter(self.config)
        self.cache = AIResponseCache(self.config)
    
    def get_user_identifier(self, session_id: str, user_ip: str) -> str:
        """Generate user identifier for rate limiting"""
        return f"session:{session_id}" if session_id and session_id != "default" else f"ip:{user_ip}"
    
    def can_make_ai_request(self, session_id: str, user_ip: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if user can make an AI request"""
        identifier = self.get_user_identifier(session_id, user_ip)
        return self.rate_limiter.check_rate_limit(identifier)
    
    def record_ai_request(self, session_id: str, user_ip: str, query: str, 
                         response: str, cached: bool = False, 
                         tokens_used: int = 0, response_time_ms: int = 0,
                         language: str = "en", model: str = "gpt-3.5-turbo"):
        """Record AI request for tracking and rate limiting"""
        
        # Increment rate limiting counters only for non-cached requests
        identifier = self.get_user_identifier(session_id, user_ip)
        if not cached:
            self.rate_limiter.increment_usage(identifier)
        
        # Cache the response for future use
        if not cached:
            self.cache.cache_response(query, response, language, model)
        
        # Record usage in database
        try:
            db = SessionLocal()
            query_hash = self.cache._generate_query_hash(query, language)
            
            usage_record = AIUsageTracking(
                session_id=session_id,
                user_ip=user_ip,
                query_hash=query_hash,
                original_query=query,
                response_cached=cached,
                model_used=model,
                tokens_used=tokens_used,
                response_time_ms=response_time_ms
            )
            
            db.add(usage_record)
            db.commit()
            db.close()
        except Exception as e:
            print(f"Error recording AI usage: {e}")
    
    def get_cached_response(self, query: str, language: str = "en") -> Optional[str]:
        """Get cached response if available"""
        cache_entry = self.cache.get_cached_response(query, language)
        return cache_entry.response if cache_entry else None
    
    def get_usage_stats(self, session_id: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """Get AI usage statistics"""
        try:
            db = SessionLocal()
            
            # Build query
            query = db.query(AIUsageTracking)
            if session_id:
                query = query.filter(AIUsageTracking.session_id == session_id)
            
            # Filter by date range
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(AIUsageTracking.timestamp >= cutoff_date)
            
            usage_records = query.all()
            
            # Calculate statistics
            total_requests = len(usage_records)
            cached_requests = sum(1 for r in usage_records if r.response_cached)
            cache_hit_rate = (cached_requests / total_requests * 100) if total_requests > 0 else 0
            total_tokens = sum(r.tokens_used or 0 for r in usage_records)
            avg_response_time = sum(r.response_time_ms or 0 for r in usage_records) / total_requests if total_requests > 0 else 0
            
            db.close()
            
            return {
                "total_requests": total_requests,
                "cached_requests": cached_requests,
                "cache_hit_rate": round(cache_hit_rate, 2),
                "total_tokens_used": total_tokens,
                "average_response_time_ms": round(float(avg_response_time), 2),
                "days_analyzed": days
            }
        except Exception as e:
            print(f"Error getting usage stats: {e}")
            return {"error": str(e)}

# Global AI service manager instance
ai_service_manager = AIServiceManager()

def create_ai_usage_table():
    """Create the AI usage tracking table"""
    try:
        AIUsageTracking.__table__.create(bind=engine, checkfirst=True)
        print("✅ AI usage tracking table created")
    except Exception as e:
        print(f"⚠️ AI usage table creation: {e}")

def cleanup_memory_cache():
    """Clean up expired entries from memory cache"""
    current_time = datetime.utcnow()
    config = RateLimitConfig()
    
    expired_keys = []
    for key, entry in memory_cache.items():
        if current_time - entry.get('timestamp', current_time) > timedelta(seconds=config.cache_ttl_seconds):
            expired_keys.append(key)
    
    for key in expired_keys:
        del memory_cache[key]
    
    if expired_keys:
        print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

# Initialize on import
create_ai_usage_table()
