#!/usr/bin/env python3
"""
Transport API Resilience Layer
===============================

Provides retry logic, fallback mechanisms, and GTFS integration for robust
offline-capable public transport data in Istanbul AI.

Features:
- Automatic retry with exponential backoff
- GTFS fallback when real-time APIs fail
- Intelligent caching (Redis + SQLite + memory)
- Offline support with cached/static data
- Connection pooling and circuit breaker pattern
"""

import asyncio
import aiohttp
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import redis
import os

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # API failing, use fallback
    HALF_OPEN = "half_open"  # Testing if API recovered


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CacheConfig:
    """Cache configuration"""
    redis_url: Optional[str] = None
    sqlite_path: str = "./data/transport_cache.db"
    memory_cache_size: int = 1000
    default_ttl: int = 300  # 5 minutes
    offline_ttl: int = 86400  # 24 hours for offline data


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def record_success(self):
        """Record successful API call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        
    def record_failure(self):
        """Record failed API call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
    def can_attempt(self) -> bool:
        """Check if we can attempt API call"""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).seconds
                if elapsed >= self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False
            
        # HALF_OPEN state - allow one attempt
        return True


class TransportAPIResilience:
    """
    Resilience layer for transport APIs with fallback to GTFS
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        cache_config: Optional[CacheConfig] = None
    ):
        self.retry_config = retry_config or RetryConfig()
        self.cache_config = cache_config or CacheConfig()
        
        # Initialize caching layers
        self._init_cache()
        
        # Circuit breakers per endpoint
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # GTFS service for fallback
        self.gtfs_service = None
        self._init_gtfs_service()
        
        logger.info("✅ Transport API resilience layer initialized")
        
    def _init_cache(self):
        """Initialize multi-layer caching"""
        # Redis cache (fastest, for real-time data)
        self.redis_client = None
        if self.cache_config.redis_url:
            try:
                self.redis_client = redis.from_url(self.cache_config.redis_url)
                self.redis_client.ping()
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")
        
        # SQLite cache (persistent, for offline support)
        self.sqlite_conn = None
        try:
            os.makedirs(os.path.dirname(self.cache_config.sqlite_path), exist_ok=True)
            self.sqlite_conn = sqlite3.connect(
                self.cache_config.sqlite_path,
                check_same_thread=False
            )
            self._init_sqlite_schema()
            logger.info("✅ SQLite cache initialized")
        except Exception as e:
            logger.error(f"SQLite cache failed: {e}")
        
        # Memory cache (fallback)
        self.memory_cache: Dict[str, Tuple[datetime, Any]] = {}
        
    def _init_sqlite_schema(self):
        """Create SQLite cache tables"""
        if not self.sqlite_conn:
            return
            
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                cached_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                source TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gtfs_schedules (
                route_id TEXT NOT NULL,
                stop_id TEXT NOT NULL,
                trip_id TEXT,
                arrival_time TEXT NOT NULL,
                departure_time TEXT NOT NULL,
                service_days TEXT NOT NULL,
                cached_at TIMESTAMP NOT NULL,
                PRIMARY KEY (route_id, stop_id, arrival_time)
            )
        """)
        
        self.sqlite_conn.commit()
        
    def _init_gtfs_service(self):
        """Initialize GTFS service for fallback"""
        try:
            from backend.services.gtfs_service import GTFSDataService
            self.gtfs_service = GTFSDataService()
            logger.info("✅ GTFS service loaded for fallback")
        except Exception as e:
            logger.error(f"Failed to load GTFS service: {e}")
            
    def get_circuit_breaker(self, endpoint: str) -> CircuitBreaker:
        """Get or create circuit breaker for endpoint"""
        if endpoint not in self.circuit_breakers:
            self.circuit_breakers[endpoint] = CircuitBreaker()
        return self.circuit_breakers[endpoint]
        
    async def with_retry_and_fallback(
        self,
        api_call: Callable,
        fallback_call: Callable,
        cache_key: str,
        endpoint: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute API call with retry logic and fallback to GTFS
        
        Args:
            api_call: Async function to call the real-time API
            fallback_call: Function to call for fallback data (GTFS)
            cache_key: Key for caching
            endpoint: API endpoint name for circuit breaker
            **kwargs: Arguments to pass to api_call
        """
        circuit_breaker = self.get_circuit_breaker(endpoint)
        
        # Try to get from cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Cache hit for {cache_key}")
            return cached_data
            
        # Check circuit breaker
        if not circuit_breaker.can_attempt():
            logger.info(f"Circuit breaker open for {endpoint}, using fallback")
            return await self._use_fallback(fallback_call, cache_key, **kwargs)
        
        # Try API call with retry logic
        for attempt in range(self.retry_config.max_retries):
            try:
                result = await api_call(**kwargs)
                
                if result and isinstance(result, dict):
                    # Success - cache and return
                    circuit_breaker.record_success()
                    self._store_in_cache(cache_key, result, self.cache_config.default_ttl)
                    return result
                    
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.retry_config.max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.retry_config.initial_delay * 
                        (self.retry_config.exponential_base ** attempt),
                        self.retry_config.max_delay
                    )
                    
                    if self.retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random())
                    
                    logger.info(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    
        # All retries failed
        circuit_breaker.record_failure()
        logger.warning(f"API call failed after {self.retry_config.max_retries} attempts, using fallback")
        
        return await self._use_fallback(fallback_call, cache_key, **kwargs)
        
    async def _use_fallback(
        self,
        fallback_call: Callable,
        cache_key: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Use fallback data source (GTFS)"""
        try:
            # Check if fallback_call is async
            if asyncio.iscoroutinefunction(fallback_call):
                result = await fallback_call(**kwargs)
            else:
                result = fallback_call(**kwargs)
                
            if result:
                # Cache fallback data with longer TTL for offline support
                self._store_in_cache(cache_key, result, self.cache_config.offline_ttl)
                result['data_source'] = 'gtfs_fallback'
                return result
                
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            
        # Return empty result with error indicator
        return {
            'error': 'All data sources unavailable',
            'data_source': 'none',
            'timestamp': datetime.now().isoformat()
        }
        
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from multi-layer cache"""
        # Try Redis first (fastest)
        if self.redis_client:
            try:
                data = self.redis_client.get(cache_key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")
        
        # Try SQLite (persistent)
        if self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    "SELECT data, expires_at FROM api_cache WHERE cache_key = ?",
                    (cache_key,)
                )
                row = cursor.fetchone()
                if row:
                    data, expires_at = row
                    if datetime.fromisoformat(expires_at) > datetime.now():
                        return json.loads(data)
                    else:
                        # Expired, delete it
                        cursor.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
                        self.sqlite_conn.commit()
            except Exception as e:
                logger.debug(f"SQLite get failed: {e}")
        
        # Try memory cache (last resort)
        if cache_key in self.memory_cache:
            cached_time, data = self.memory_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_config.default_ttl:
                return data
            else:
                del self.memory_cache[cache_key]
        
        return None
        
    def _store_in_cache(self, cache_key: str, data: Dict[str, Any], ttl: int):
        """Store data in multi-layer cache"""
        data_json = json.dumps(data)
        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl)
        
        # Store in Redis (with TTL)
        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, ttl, data_json)
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")
        
        # Store in SQLite (persistent)
        if self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO api_cache 
                    (cache_key, data, cached_at, expires_at, source)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (cache_key, data_json, now.isoformat(), expires_at.isoformat(), 
                     data.get('data_source', 'api'))
                )
                self.sqlite_conn.commit()
            except Exception as e:
                logger.debug(f"SQLite set failed: {e}")
        
        # Store in memory cache (always succeeds)
        self.memory_cache[cache_key] = (now, data)
        
        # Limit memory cache size
        if len(self.memory_cache) > self.cache_config.memory_cache_size:
            # Remove oldest entries
            oldest_keys = sorted(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k][0]
            )[:len(self.memory_cache) - self.cache_config.memory_cache_size]
            for key in oldest_keys:
                del self.memory_cache[key]
                
    def get_gtfs_bus_data(self, route_id: Optional[str] = None, stop_id: Optional[str] = None) -> Dict[str, Any]:
        """Get bus data from GTFS feed"""
        if not self.gtfs_service:
            return {
                'error': 'GTFS service not available',
                'timestamp': datetime.now().isoformat(),
                'data_source': 'gtfs_fallback'
            }
            
        try:
            result = {
                'routes': [],
                'stops': [],
                'schedules': [],
                'data_source': 'gtfs',
                'timestamp': datetime.now().isoformat()
            }
            
            # Get route information
            if route_id:
                route = self.gtfs_service.routes.get(route_id)
                if route:
                    result['routes'].append({
                        'route_id': route.route_id,
                        'route_name': route.route_short_name,
                        'route_long_name': route.route_long_name,
                        'route_type': 'bus' if route.route_type == 3 else 'other'
                    })
                    
                    # Get stops for this route
                    if route_id in self.gtfs_service.route_stops:
                        stop_ids = self.gtfs_service.route_stops[route_id]
                        for sid in stop_ids:
                            stop = self.gtfs_service.stops.get(sid)
                            if stop:
                                result['stops'].append({
                                    'stop_id': stop.stop_id,
                                    'stop_name': stop.stop_name,
                                    'lat': stop.stop_lat,
                                    'lon': stop.stop_lon
                                })
            else:
                # Get all bus routes
                for route in self.gtfs_service.routes.values():
                    if route.route_type == 3:  # Bus
                        result['routes'].append({
                            'route_id': route.route_id,
                            'route_name': route.route_short_name,
                            'route_long_name': route.route_long_name
                        })
            
            # Get schedule information
            if stop_id and stop_id in self.gtfs_service.stop_times:
                stop_times = self.gtfs_service.stop_times[stop_id]
                current_time = datetime.now().time()
                
                # Get next arrivals
                for st in stop_times[:10]:  # Limit to next 10
                    result['schedules'].append({
                        'trip_id': st.trip_id,
                        'arrival_time': st.arrival_time,
                        'departure_time': st.departure_time,
                        'stop_sequence': st.stop_sequence
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get GTFS bus data: {e}")
            return {'error': str(e), 'data_source': 'gtfs'}
            
    def get_gtfs_metro_data(self, line_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metro data from GTFS feed"""
        if not self.gtfs_service:
            return {'error': 'GTFS service not available'}
            
        try:
            result = {
                'lines': [],
                'data_source': 'gtfs',
                'timestamp': datetime.now().isoformat()
            }
            
            # Get metro routes (type 1)
            for route in self.gtfs_service.routes.values():
                if route.route_type == 1:  # Metro
                    if not line_id or route.route_id == line_id:
                        result['lines'].append({
                            'line_id': route.route_id,
                            'line_name': route.route_short_name,
                            'status': 'operational',
                            'color': route.route_color,
                            'stops': []
                        })
                        
                        # Get stops for this line
                        if route.route_id in self.gtfs_service.route_stops:
                            stop_ids = self.gtfs_service.route_stops[route.route_id]
                            for sid in stop_ids:
                                stop = self.gtfs_service.stops.get(sid)
                                if stop:
                                    result['lines'][-1]['stops'].append({
                                        'stop_id': stop.stop_id,
                                        'stop_name': stop.stop_name,
                                        'lat': stop.stop_lat,
                                        'lon': stop.stop_lon
                                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get GTFS metro data: {e}")
            return {'error': str(e), 'data_source': 'gtfs'}
            
    def close(self):
        """Clean up resources"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.redis_client:
            self.redis_client.close()


# Decorator for automatic retry and fallback
def resilient_transport_api(
    fallback_function: str,
    cache_key_pattern: str,
    endpoint: str = "default"
):
    """
    Decorator to make transport API calls resilient with automatic retry and GTFS fallback
    
    Args:
        fallback_function: Name of the fallback method to call (e.g., 'get_gtfs_bus_data')
        cache_key_pattern: Pattern for cache key (e.g., 'bus:{route_id}')
        endpoint: Name of the endpoint for circuit breaker
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get resilience layer
            if not hasattr(self, '_resilience_layer'):
                self._resilience_layer = TransportAPIResilience()
            
            resilience = self._resilience_layer
            
            # Generate cache key
            cache_key = cache_key_pattern.format(**kwargs) if kwargs else cache_key_pattern
            
            # Get fallback function
            fallback = getattr(resilience, fallback_function)
            
            # Execute with retry and fallback
            return await resilience.with_retry_and_fallback(
                api_call=func,
                fallback_call=fallback,
                cache_key=cache_key,
                endpoint=endpoint,
                self=self,
                *args,
                **kwargs
            )
        return wrapper
    return decorator
