#!/usr/bin/env python3
"""
Edge Caching System for AI Istanbul
Caches static attraction, event, and route data at CDN level
Reduces backend load and improves response times globally
"""

import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import os
import gzip

logger = logging.getLogger(__name__)

@dataclass
class EdgeCacheEntry:
    """Edge cache entry for static data"""
    cache_key: str
    content: Union[Dict, List, str]
    content_type: str
    created_at: datetime
    expires_at: datetime
    etag: str
    compressed: bool = True
    access_count: int = 0

class EdgeCacheManager:
    """
    Edge caching system for static content
    - Caches attractions, events, routes for CDN distribution
    - Generates optimized JSON responses
    - Implements proper HTTP caching headers
    """
    
    def __init__(self, cache_dir: str = "cache/edge", enable_compression: bool = True):
        self.cache_dir = cache_dir
        self.enable_compression = enable_compression
        self.cache_entries: Dict[str, EdgeCacheEntry] = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache TTL for different content types
        self.cache_ttl = {
            'attractions': timedelta(days=7),      # Attractions rarely change
            'events': timedelta(hours=6),          # Events updated regularly
            'routes': timedelta(hours=12),         # Routes change with traffic
            'neighborhoods': timedelta(days=1),    # Stable neighborhood data
            'museums': timedelta(days=3),          # Museum info stable
            'restaurants': timedelta(hours=8),     # Restaurant info changes
            'static_assets': timedelta(days=30),   # Images, CSS, JS
            'api_responses': timedelta(hours=2)    # General API responses
        }
        
        # Load existing cache
        self._load_cache_index()
        
        logger.info(f"🌐 Edge Cache Manager initialized: {len(self.cache_entries)} entries")
    
    def cache_attractions_data(self) -> str:
        """Cache all attractions data for CDN distribution"""
        try:
            # Import and get attractions data
            from istanbul_attractions_system import IstanbulAttractionsSystem
            
            attractions_system = IstanbulAttractionsSystem()
            
            # Get all attractions
            all_attractions = []
            for category in attractions_system.categories:
                category_attractions = attractions_system.get_attractions_by_category(category)
                for attraction in category_attractions:
                    all_attractions.append({
                        'id': getattr(attraction, 'id', hash(attraction.name)),
                        'name': attraction.name,
                        'category': attraction.category.value,
                        'district': attraction.district,
                        'description': attraction.description,
                        'highlights': attraction.highlights,
                        'tags': attraction.tags,
                        'coordinates': {
                            'lat': attraction.coordinates[0],
                            'lng': attraction.coordinates[1]
                        } if attraction.coordinates else None,
                        'opening_hours': getattr(attraction, 'opening_hours', {}),
                        'entry_fee': getattr(attraction, 'entry_fee', {}),
                        'accessibility': getattr(attraction, 'accessibility', []),
                        'family_friendly': getattr(attraction, 'family_friendly', True),
                        'romantic_spot': getattr(attraction, 'romantic_spot', False)
                    })
            
            # Cache the data
            cache_key = self._set_edge_cache(
                'attractions_all',
                all_attractions,
                'attractions',
                content_type='application/json'
            )
            
            logger.info(f"🏛️ Cached {len(all_attractions)} attractions for edge distribution")
            return cache_key
            
        except Exception as e:
            logger.error(f"Failed to cache attractions data: {e}")
            return None
    
    def cache_events_data(self) -> str:
        """Cache current events data for CDN distribution"""
        try:
            # Import and get events data
            from monthly_events_scheduler import get_cached_events, fetch_monthly_events
            
            # Get current events
            events = get_cached_events()
            if not events:
                events = fetch_monthly_events()
            
            if not events:
                events = []
            
            # Format events for edge caching
            formatted_events = []
            for event in events:
                if isinstance(event, dict):
                    formatted_events.append(event)
                else:
                    # Convert object to dict
                    formatted_events.append({
                        'title': getattr(event, 'title', str(event)),
                        'date': getattr(event, 'date', ''),
                        'time': getattr(event, 'time', ''),
                        'venue': getattr(event, 'venue', ''),
                        'description': getattr(event, 'description', ''),
                        'price': getattr(event, 'price', ''),
                        'category': getattr(event, 'category', 'Cultural'),
                        'booking_url': getattr(event, 'booking_url', ''),
                        'image_url': getattr(event, 'image_url', '')
                    })
            
            # Cache the data
            cache_key = self._set_edge_cache(
                'events_current',
                formatted_events,
                'events',
                content_type='application/json'
            )
            
            logger.info(f"🎭 Cached {len(formatted_events)} events for edge distribution")
            return cache_key
            
        except Exception as e:
            logger.error(f"Failed to cache events data: {e}")
            return None
    
    def cache_neighborhoods_data(self) -> str:
        """Cache neighborhood guide data for CDN distribution"""
        
        # Static neighborhood data
        neighborhoods = {
            'sultanahmet': {
                'name': 'Sultanahmet',
                'character': ['historic', 'touristy', 'traditional', 'cultural'],
                'good_for': ['history', 'sightseeing', 'museums', 'first_visit'],
                'vibe': 'historic_traditional',
                'top_attractions': ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace', 'Basilica Cistern'],
                'best_restaurants': ['Pandeli', 'Deraliye', 'Sultanahmet Köftecisi'],
                'transportation': ['Tram T1', 'Metro M2', 'Bus'],
                'coordinates': {'lat': 41.0055, 'lng': 28.9769}
            },
            'beyoglu': {
                'name': 'Beyoğlu',
                'character': ['modern', 'vibrant', 'nightlife', 'artsy'],
                'good_for': ['nightlife', 'art', 'shopping', 'young_crowd'],
                'vibe': 'modern_vibrant',
                'top_attractions': ['Galata Tower', 'Istiklal Street', 'Pera Museum'],
                'best_restaurants': ['Mikla', 'Karakoy Lokantasi', 'Pandeli'],
                'transportation': ['Metro M2', 'Funicular F1', 'Bus'],
                'coordinates': {'lat': 41.0361, 'lng': 28.9769}
            },
            'karakoy': {
                'name': 'Karaköy',
                'character': ['trendy', 'artistic', 'gentrified', 'hipster'],
                'good_for': ['art_galleries', 'design', 'coffee', 'creative'],
                'vibe': 'trendy_artistic',
                'top_attractions': ['Istanbul Modern', 'Galata Bridge', 'Karakoy Port'],
                'best_restaurants': ['Karakoy Lokantasi', 'Under', 'House Cafe'],
                'transportation': ['Metro M2', 'Ferry', 'Bus'],
                'coordinates': {'lat': 41.0255, 'lng': 28.9742}
            },
            'kadikoy': {
                'name': 'Kadıköy',
                'character': ['local', 'authentic', 'diverse', 'foodie'],
                'good_for': ['local_experience', 'food', 'nightlife', 'authentic'],
                'vibe': 'local_authentic',
                'top_attractions': ['Moda', 'Kadikoy Market', 'Bagdat Street'],
                'best_restaurants': ['Ciya Sofrasi', 'Kanaat Lokantasi', 'Pandeli'],
                'transportation': ['Metro M4', 'Ferry', 'Bus'],
                'coordinates': {'lat': 40.9833, 'lng': 29.0333}
            },
            'besiktas': {
                'name': 'Beşiktaş',
                'character': ['sporty', 'energetic', 'local', 'waterfront'],
                'good_for': ['sports', 'local_life', 'bosphorus', 'energy'],
                'vibe': 'energetic_local',
                'top_attractions': ['Dolmabahce Palace', 'Besiktas Stadium', 'Ortakoy'],
                'best_restaurants': ['Tugra', 'Feriye Palace', 'House Cafe'],
                'transportation': ['Metro M6', 'Ferry', 'Bus'],
                'coordinates': {'lat': 41.0433, 'lng': 29.0092}
            },
            'ortakoy': {
                'name': 'Ortaköy',
                'character': ['picturesque', 'bosphorus', 'romantic', 'weekend'],
                'good_for': ['romantic', 'photos', 'weekend', 'water_views'],
                'vibe': 'romantic_scenic',
                'top_attractions': ['Ortakoy Mosque', 'Bosphorus Bridge', 'Art Market'],
                'best_restaurants': ['House Cafe', 'Reina', 'Sortie'],
                'transportation': ['Bus', 'Ferry', 'Car'],
                'coordinates': {'lat': 41.0547, 'lng': 29.0267}
            }
        }
        
        # Cache the data
        cache_key = self._set_edge_cache(
            'neighborhoods_guide',
            neighborhoods,
            'neighborhoods',
            content_type='application/json'
        )
        
        logger.info(f"🏘️ Cached {len(neighborhoods)} neighborhoods for edge distribution")
        return cache_key
    
    def cache_api_response(self, endpoint: str, params: Dict[str, Any], response_data: Any) -> str:
        """Cache API response for CDN distribution"""
        
        # Create cache key from endpoint and params
        cache_id = f"{endpoint}_{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]}"
        
        cache_key = self._set_edge_cache(
            cache_id,
            response_data,
            'api_responses',
            content_type='application/json'
        )
        
        return cache_key
    
    def _set_edge_cache(self, cache_id: str, content: Any, content_category: str, 
                       content_type: str = 'application/json') -> str:
        """Set edge cache entry"""
        
        # Generate cache key and ETag
        cache_key = f"{content_category}_{cache_id}"
        content_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
        etag = hashlib.md5(content_str.encode()).hexdigest()[:16]
        
        # Compress content if enabled
        final_content = content
        compressed = False
        
        if self.enable_compression and content_type == 'application/json':
            try:
                compressed_content = gzip.compress(content_str.encode())
                if len(compressed_content) < len(content_str.encode()) * 0.8:  # Only if significant compression
                    final_content = compressed_content
                    compressed = True
            except Exception as e:
                logger.warning(f"Compression failed for {cache_key}: {e}")
        
        # Calculate TTL
        ttl = self.cache_ttl.get(content_category, self.cache_ttl['api_responses'])
        
        # Create cache entry
        entry = EdgeCacheEntry(
            cache_key=cache_key,
            content=final_content,
            content_type=content_type,
            created_at=datetime.now(),
            expires_at=datetime.now() + ttl,
            etag=etag,
            compressed=compressed,
            access_count=0
        )
        
        # Store in memory
        self.cache_entries[cache_key] = entry
        
        # Save to disk
        self._save_cache_entry(entry)
        
        logger.debug(f"🌐 Edge cache SET: {cache_key} (compressed: {compressed}, TTL: {ttl})")
        return cache_key
    
    def get_edge_cache(self, cache_key: str) -> Optional[EdgeCacheEntry]:
        """Get edge cache entry with proper HTTP headers"""
        
        # Check memory first
        if cache_key in self.cache_entries:
            entry = self.cache_entries[cache_key]
            
            # Check if expired
            if datetime.now() > entry.expires_at:
                del self.cache_entries[cache_key]
                self._remove_cache_file(cache_key)
                return None
            
            # Update access count
            entry.access_count += 1
            return entry
        
        # Load from disk
        entry = self._load_cache_entry(cache_key)
        if entry and datetime.now() <= entry.expires_at:
            self.cache_entries[cache_key] = entry
            return entry
        
        return None
    
    def get_cache_headers(self, cache_key: str) -> Dict[str, str]:
        """Get HTTP cache headers for edge distribution"""
        
        entry = self.get_edge_cache(cache_key)
        if not entry:
            return {}
        
        headers = {
            'Cache-Control': f'public, max-age={int((entry.expires_at - datetime.now()).total_seconds())}',
            'ETag': f'"{entry.etag}"',
            'Last-Modified': entry.created_at.strftime('%a, %d %b %Y %H:%M:%S GMT'),
            'Content-Type': entry.content_type,
            'Vary': 'Accept-Encoding'
        }
        
        if entry.compressed:
            headers['Content-Encoding'] = 'gzip'
        
        return headers
    
    def _save_cache_entry(self, entry: EdgeCacheEntry) -> None:
        """Save cache entry to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{entry.cache_key}.cache")
            
            # Prepare entry data
            entry_data = {
                'cache_key': entry.cache_key,
                'content_type': entry.content_type,
                'created_at': entry.created_at.isoformat(),
                'expires_at': entry.expires_at.isoformat(),
                'etag': entry.etag,
                'compressed': entry.compressed,
                'access_count': entry.access_count
            }
            
            # Save content separately if it's compressed binary data
            if entry.compressed and isinstance(entry.content, bytes):
                content_file = os.path.join(self.cache_dir, f"{entry.cache_key}.content")
                with open(content_file, 'wb') as f:
                    f.write(entry.content)
                entry_data['content_file'] = f"{entry.cache_key}.content"
            else:
                entry_data['content'] = entry.content
            
            # Save metadata
            with open(cache_file, 'w') as f:
                json.dump(entry_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save edge cache entry: {e}")
    
    def _load_cache_entry(self, cache_key: str) -> Optional[EdgeCacheEntry]:
        """Load cache entry from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.cache")
            
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'r') as f:
                entry_data = json.load(f)
            
            # Load content
            content = None
            if 'content_file' in entry_data:
                content_file = os.path.join(self.cache_dir, entry_data['content_file'])
                with open(content_file, 'rb') as f:
                    content = f.read()
            else:
                content = entry_data['content']
            
            # Create entry
            entry = EdgeCacheEntry(
                cache_key=entry_data['cache_key'],
                content=content,
                content_type=entry_data['content_type'],
                created_at=datetime.fromisoformat(entry_data['created_at']),
                expires_at=datetime.fromisoformat(entry_data['expires_at']),
                etag=entry_data['etag'],
                compressed=entry_data.get('compressed', False),
                access_count=entry_data.get('access_count', 0)
            )
            
            return entry
            
        except Exception as e:
            logger.warning(f"Failed to load edge cache entry: {e}")
            return None
    
    def _remove_cache_file(self, cache_key: str) -> None:
        """Remove cache files from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.cache")
            content_file = os.path.join(self.cache_dir, f"{cache_key}.content")
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(content_file):
                os.remove(content_file)
                
        except Exception as e:
            logger.warning(f"Failed to remove cache files: {e}")
    
    def _load_cache_index(self) -> None:
        """Load cache index from disk"""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    cache_key = filename[:-6]  # Remove .cache extension
                    entry = self._load_cache_entry(cache_key)
                    if entry and datetime.now() <= entry.expires_at:
                        self.cache_entries[cache_key] = entry
                        
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
    
    def refresh_all_static_data(self) -> Dict[str, str]:
        """Refresh all static data caches"""
        results = {}
        
        try:
            results['attractions'] = self.cache_attractions_data()
        except Exception as e:
            logger.error(f"Failed to refresh attractions cache: {e}")
            results['attractions'] = None
        
        try:
            results['events'] = self.cache_events_data()
        except Exception as e:
            logger.error(f"Failed to refresh events cache: {e}")
            results['events'] = None
        
        try:
            results['neighborhoods'] = self.cache_neighborhoods_data()
        except Exception as e:
            logger.error(f"Failed to refresh neighborhoods cache: {e}")
            results['neighborhoods'] = None
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get edge cache statistics"""
        
        total_size = 0
        category_stats = {}
        
        for entry in self.cache_entries.values():
            category = entry.cache_key.split('_')[0]
            
            if category not in category_stats:
                category_stats[category] = {
                    'count': 0,
                    'total_access': 0,
                    'avg_access': 0
                }
            
            category_stats[category]['count'] += 1
            category_stats[category]['total_access'] += entry.access_count
            
            # Estimate size
            if isinstance(entry.content, bytes):
                total_size += len(entry.content)
            elif isinstance(entry.content, str):
                total_size += len(entry.content.encode())
            else:
                total_size += len(json.dumps(entry.content).encode())
        
        # Calculate averages
        for stats in category_stats.values():
            if stats['count'] > 0:
                stats['avg_access'] = stats['total_access'] / stats['count']
        
        return {
            'total_entries': len(self.cache_entries),
            'total_size_mb': total_size / (1024 * 1024),
            'category_stats': category_stats,
            'compression_enabled': self.enable_compression
        }

# Global edge cache instance
edge_cache = EdgeCacheManager()

def get_edge_cache() -> EdgeCacheManager:
    """Get the global edge cache instance"""
    return edge_cache
