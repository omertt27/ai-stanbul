#!/usr/bin/env python3
"""
ML Result Caching System for AI Istanbul
Reduces ML inference costs by caching common recommendations and results
"""

import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class MLCacheEntry:
    """Cached ML result entry"""
    query_hash: str
    result_data: Dict[str, Any]
    confidence_score: float
    enhancement_systems: List[str]
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    last_accessed: datetime = None

class MLResultCache:
    """
    Intelligent ML result caching system
    - Caches expensive ML/DL inference results
    - Reduces API costs by avoiding re-computation
    - Smart cache invalidation based on data freshness
    """
    
    def __init__(self, cache_dir: str = "cache/ml_results", max_cache_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.memory_cache: Dict[str, MLCacheEntry] = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache configuration
        self.cache_ttl = {
            'restaurant_discovery': timedelta(hours=6),      # Restaurant data changes moderately
            'attraction_recommendation': timedelta(days=7),  # Attractions are mostly static
            'events_personalization': timedelta(hours=2),    # Events change frequently
            'route_optimizer': timedelta(hours=12),          # Routes change with traffic/weather
            'weather_advisor': timedelta(hours=1),           # Weather changes frequently
            'typo_corrector': timedelta(days=30),           # Typos are consistent
            'neighborhood_matcher': timedelta(days=1),       # Neighborhood info is stable
            'event_predictor': timedelta(hours=4),          # Event predictions change
            'museum_route_planning': timedelta(days=1),     # Museum info is stable
            'gps_route_planning': timedelta(minutes=30),    # GPS routes change frequently
            'real_time_updates': timedelta(minutes=5),      # Real-time data is very volatile
            'transport_optimization': timedelta(hours=2),   # Transport conditions change
            'location_recommendations': timedelta(hours=4), # Location-based suggestions
            'general': timedelta(hours=4)                   # Default TTL
        }
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"🗄️ ML Result Cache initialized: {len(self.memory_cache)} entries loaded")
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any] = None, ml_systems: List[str] = None) -> str:
        """Generate a unique cache key for the query and context"""
        
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()
        
        # Create cache key components
        key_components = [normalized_query]
        
        # Add relevant context (exclude volatile data like timestamps)
        if context:
            stable_context = {}
            for key, value in context.items():
                if key not in ['timestamp', 'session_id', 'request_id']:
                    stable_context[key] = value
            
            if stable_context:
                key_components.append(json.dumps(stable_context, sort_keys=True))
        
        # Add ML systems
        if ml_systems:
            key_components.append("|".join(sorted(ml_systems)))
        
        # Generate hash
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, context: Dict[str, Any] = None, ml_systems: List[str] = None) -> Optional[Dict[str, Any]]:
        """Get cached ML result if available and not expired"""
        
        cache_key = self._generate_cache_key(query, context, ml_systems)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            
            # Check if expired
            if datetime.now() > entry.expires_at:
                del self.memory_cache[cache_key]
                self._remove_disk_cache(cache_key)
                return None
            
            # Update access stats
            entry.hit_count += 1
            entry.last_accessed = datetime.now()
            
            logger.debug(f"🎯 ML Cache HIT: {cache_key[:8]}... (hits: {entry.hit_count})")
            return entry.result_data
        
        # Check disk cache
        disk_result = self._load_disk_cache(cache_key)
        if disk_result:
            # Load into memory cache
            self.memory_cache[cache_key] = disk_result
            logger.debug(f"💾 ML Cache DISK HIT: {cache_key[:8]}...")
            return disk_result.result_data
        
        logger.debug(f"❌ ML Cache MISS: {cache_key[:8]}...")
        return None
    
    def set(self, query: str, result_data: Dict[str, Any], confidence_score: float, 
            enhancement_systems: List[str], context: Dict[str, Any] = None) -> None:
        """Cache ML result with intelligent TTL based on system types"""
        
        cache_key = self._generate_cache_key(query, context, enhancement_systems)
        
        # Determine TTL based on enhancement systems
        ttl = self._calculate_ttl(enhancement_systems)
        
        # Create cache entry
        entry = MLCacheEntry(
            query_hash=cache_key,
            result_data=result_data,
            confidence_score=confidence_score,
            enhancement_systems=enhancement_systems,
            created_at=datetime.now(),
            expires_at=datetime.now() + ttl,
            hit_count=0,
            last_accessed=datetime.now()
        )
        
        # Store in memory cache
        self.memory_cache[cache_key] = entry
        
        # Store on disk for persistence
        self._save_disk_cache(cache_key, entry)
        
        # Cleanup if cache is too large
        self._cleanup_cache()
        
        logger.debug(f"💾 ML Cache SET: {cache_key[:8]}... (TTL: {ttl}, systems: {len(enhancement_systems)})")
    
    def _calculate_ttl(self, enhancement_systems: List[str]) -> timedelta:
        """Calculate TTL based on the most volatile system type"""
        
        if not enhancement_systems:
            return self.cache_ttl['general']
        
        # Find the shortest TTL among the systems (most volatile wins)
        min_ttl = timedelta(days=365)  # Start with max
        
        for system in enhancement_systems:
            system_ttl = self.cache_ttl.get(system, self.cache_ttl['general'])
            if system_ttl < min_ttl:
                min_ttl = system_ttl
        
        return min_ttl
    
    def _cleanup_cache(self) -> None:
        """Remove expired entries and enforce size limits"""
        
        current_time = datetime.now()
        
        # Remove expired entries
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if current_time > entry.expires_at:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            self._remove_disk_cache(key)
        
        # Enforce size limits (remove least recently used)
        if len(self.memory_cache) > self.max_cache_size:
            # Sort by last accessed time
            sorted_entries = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1].last_accessed or x[1].created_at
            )
            
            # Remove oldest entries
            entries_to_remove = len(self.memory_cache) - self.max_cache_size
            for key, _ in sorted_entries[:entries_to_remove]:
                del self.memory_cache[key]
                self._remove_disk_cache(key)
        
        if expired_keys or len(self.memory_cache) > self.max_cache_size:
            logger.info(f"🧹 ML Cache cleanup: {len(expired_keys)} expired, {len(self.memory_cache)} total entries")
    
    def _save_disk_cache(self, cache_key: str, entry: MLCacheEntry) -> None:
        """Save cache entry to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            # Convert entry to dict
            entry_dict = asdict(entry)
            entry_dict['created_at'] = entry.created_at.isoformat()
            entry_dict['expires_at'] = entry.expires_at.isoformat()
            if entry.last_accessed:
                entry_dict['last_accessed'] = entry.last_accessed.isoformat()
            
            with open(cache_file, 'w') as f:
                json.dump(entry_dict, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache entry to disk: {e}")
    
    def _load_disk_cache(self, cache_key: str) -> Optional[MLCacheEntry]:
        """Load cache entry from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'r') as f:
                entry_dict = json.load(f)
            
            # Convert back to MLCacheEntry
            entry = MLCacheEntry(
                query_hash=entry_dict['query_hash'],
                result_data=entry_dict['result_data'],
                confidence_score=entry_dict['confidence_score'],
                enhancement_systems=entry_dict['enhancement_systems'],
                created_at=datetime.fromisoformat(entry_dict['created_at']),
                expires_at=datetime.fromisoformat(entry_dict['expires_at']),
                hit_count=entry_dict.get('hit_count', 0),
                last_accessed=datetime.fromisoformat(entry_dict['last_accessed']) if entry_dict.get('last_accessed') else None
            )
            
            # Check if expired
            if datetime.now() > entry.expires_at:
                self._remove_disk_cache(cache_key)
                return None
            
            return entry
            
        except Exception as e:
            logger.warning(f"Failed to load cache entry from disk: {e}")
            return None
    
    def _remove_disk_cache(self, cache_key: str) -> None:
        """Remove cache entry from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
        except Exception as e:
            logger.warning(f"Failed to remove cache entry from disk: {e}")
    
    def _load_cache(self) -> None:
        """Load existing cache entries from disk into memory"""
        try:
            if not os.path.exists(self.cache_dir):
                return
            
            loaded_count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    cache_key = filename[:-5]  # Remove .json extension
                    entry = self._load_disk_cache(cache_key)
                    if entry:
                        self.memory_cache[cache_key] = entry
                        loaded_count += 1
            
            logger.info(f"📥 Loaded {loaded_count} ML cache entries from disk")
            
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(entry.hit_count for entry in self.memory_cache.values())
        
        # Group by system types
        system_stats = {}
        for entry in self.memory_cache.values():
            for system in entry.enhancement_systems:
                if system not in system_stats:
                    system_stats[system] = {'count': 0, 'hits': 0}
                system_stats[system]['count'] += 1
                system_stats[system]['hits'] += entry.hit_count
        
        return {
            'total_entries': len(self.memory_cache),
            'total_hits': total_hits,
            'cache_size_mb': self._get_cache_size_mb(),
            'system_stats': system_stats,
            'oldest_entry': min((entry.created_at for entry in self.memory_cache.values()), default=None),
            'cache_hit_rate': total_hits / len(self.memory_cache) if self.memory_cache else 0
        }
    
    def _get_cache_size_mb(self) -> float:
        """Calculate cache size in MB"""
        total_size = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(file_path)
            return total_size / (1024 * 1024)
        except:
            return 0.0
    
    def clear_cache(self, system_type: str = None) -> int:
        """Clear cache entries, optionally filtered by system type"""
        removed_count = 0
        
        if system_type:
            # Remove specific system type
            keys_to_remove = []
            for key, entry in self.memory_cache.items():
                if system_type in entry.enhancement_systems:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
                self._remove_disk_cache(key)
                removed_count += 1
        else:
            # Clear all cache
            removed_count = len(self.memory_cache)
            self.memory_cache.clear()
            
            # Clear disk cache
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, filename))
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {e}")
        
        logger.info(f"🗑️ Cleared {removed_count} ML cache entries" + (f" for {system_type}" if system_type else ""))
        return removed_count

# Global ML cache instance
ml_cache = MLResultCache()

def get_ml_cache() -> MLResultCache:
    """Get the global ML cache instance"""
    return ml_cache
