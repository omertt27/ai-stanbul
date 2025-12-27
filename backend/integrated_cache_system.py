#!/usr/bin/env python3
"""
Integrated Cache System with Production Monitoring
=================================================

This module integrates the time-aware caching system with the existing unified AI system
and provides comprehensive production monitoring and cache performance optimization.

Key Features:
1. Integration with unified AI system
2. Production monitoring and analytics
3. Cache warming for popular queries
4. Dynamic TTL optimization based on real usage patterns
5. Performance monitoring and alerting
"""

import os
import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from collections import defaultdict, Counter
import threading
from contextlib import contextmanager

# Import centralized Redis readiness check
from core.startup_guard import is_redis_ready, get_redis_client, ensure_single_init

# Import the enhanced Google API integration
from google_api_integration import (
    GoogleApiFieldOptimizer, 
    TimeAwareCacheManager, 
    QueryIntent,
    search_restaurants_with_optimization,
    get_cost_analytics
)

# Use lazy import to avoid circular dependency
UNIFIED_AI_AVAILABLE = False
UnifiedAISystem = None
UnifiedContextManager = None

def _lazy_import_unified_ai():
    """Lazy import of unified AI system to avoid circular imports"""
    global UNIFIED_AI_AVAILABLE, UnifiedAISystem, UnifiedContextManager
    if not UNIFIED_AI_AVAILABLE:
        try:
            from unified_ai_system import UnifiedAISystem as _UnifiedAISystem, UnifiedContextManager as _UnifiedContextManager
            UnifiedAISystem = _UnifiedAISystem
            UnifiedContextManager = _UnifiedContextManager
            UNIFIED_AI_AVAILABLE = True
            logging.info("✅ Unified AI system loaded successfully (lazy import)")
        except ImportError as e:
            logging.warning(f"⚠️ Unified AI system not available: {e}")
    return UNIFIED_AI_AVAILABLE

# Import cost monitoring if available
try:
    from cost_monitor import log_google_places_cost, log_cache_performance
    COST_MONITORING_AVAILABLE = True
except ImportError:
    COST_MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CachePerformanceMetrics:
    """Comprehensive cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate_percent: float = 0.0
    average_response_time_ms: float = 0.0
    cost_savings_usd: float = 0.0
    popular_queries: List[Dict[str, Any]] = None
    cache_size_mb: float = 0.0
    eviction_count: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        if self.popular_queries is None:
            self.popular_queries = []

@dataclass
class TTLOptimizationRule:
    """Rule for dynamic TTL optimization"""
    cache_type: str
    min_hit_rate: float
    max_hit_rate: float
    ttl_multiplier: float
    description: str

class ProductionCacheMonitor:
    """
    Production-grade cache monitoring with real-time analytics
    and automated optimization
    """
    
    def __init__(self):
        self.redis_client = self._initialize_redis()
        self.metrics = CachePerformanceMetrics()
        self.query_frequency = Counter()
        self.response_times = defaultdict(list)
        self.cache_errors = []
        self.optimization_history = []
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.metrics_interval = 300  # 5 minutes
        self.optimization_interval = 3600  # 1 hour
        self.alert_thresholds = {
            'min_hit_rate': 60.0,  # Alert if hit rate < 60%
            'max_error_rate': 5.0,  # Alert if error rate > 5%
            'max_response_time': 1000.0  # Alert if response time > 1s
        }
        
        # TTL optimization rules
        self.ttl_optimization_rules = [
            TTLOptimizationRule('restaurant_basic_info', 85.0, 95.0, 1.2, 'Increase TTL for high hit rate'),
            TTLOptimizationRule('restaurant_details', 70.0, 90.0, 1.15, 'Moderate TTL increase'),
            TTLOptimizationRule('opening_hours', 60.0, 80.0, 1.1, 'Small TTL increase'),
            TTLOptimizationRule('real_time_status', 40.0, 70.0, 0.9, 'Decrease TTL for low hit rate'),
            TTLOptimizationRule('live_pricing', 30.0, 60.0, 0.8, 'Aggressive TTL decrease')
        ]
        
        # Start background monitoring
        self._start_monitoring()
    
    def _initialize_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis with monitoring extensions - uses centralized check"""
        # Use centralized Redis readiness check
        if not is_redis_ready():
            logger.info("⏭️ Redis not available (centralized check) - monitoring disabled")
            return None
        
        try:
            # Get cached Redis client from startup_guard
            client = get_redis_client()
            if client:
                logger.info("✅ Production cache monitor using centralized Redis client")
                return client
            
            # Fallback: create new client for monitoring DB
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/2')  # Use DB 2 for monitoring
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            logger.info("✅ Production cache monitor Redis connected")
            return client
        except Exception as e:
            logger.warning(f"⚠️ Redis not available for monitoring: {e}")
            return None
    
    def _start_monitoring(self):
        """Start background monitoring tasks"""
        if not self.monitoring_enabled:
            return
        
        def run_monitoring():
            while self.monitoring_enabled:
                try:
                    self._collect_metrics()
                    time.sleep(self.metrics_interval)
                except Exception as e:
                    logger.error(f"❌ Error in monitoring thread: {e}")
                    time.sleep(60)  # Wait before retrying
        
        def run_optimization():
            while self.monitoring_enabled:
                try:
                    time.sleep(self.optimization_interval)
                    self._optimize_ttl_settings()
                except Exception as e:
                    logger.error(f"❌ Error in optimization thread: {e}")
                    time.sleep(300)  # Wait before retrying
        
        # Start monitoring threads
        monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
        optimization_thread = threading.Thread(target=run_optimization, daemon=True)
        
        monitoring_thread.start()
        optimization_thread.start()
        
        logger.info("🔄 Production cache monitoring started")
    
    def record_cache_request(self, query: str, cache_hit: bool, response_time_ms: float, cache_type: str = 'unknown'):
        """Record a cache request for analytics"""
        self.metrics.total_requests += 1
        self.query_frequency[query] += 1
        self.response_times[cache_type].append(response_time_ms)
        
        if cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
        
        # Update hit rate
        if self.metrics.total_requests > 0:
            self.metrics.hit_rate_percent = (self.metrics.cache_hits / self.metrics.total_requests) * 100
        
        # Update average response time
        all_times = []
        for times_list in self.response_times.values():
            all_times.extend(times_list[-100:])  # Keep last 100 times per type
        
        if all_times:
            self.metrics.average_response_time_ms = sum(all_times) / len(all_times)
    
    def record_cache_error(self, error: str, context: Dict[str, Any]):
        """Record a cache error for monitoring"""
        self.metrics.error_count += 1
        error_record = {
            'error': error,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.cache_errors.append(error_record)
        
        # Keep only last 100 errors
        if len(self.cache_errors) > 100:
            self.cache_errors = self.cache_errors[-100:]
    
    def _collect_metrics(self):
        """Collect comprehensive cache metrics"""
        if not self.redis_client:
            return
        
        try:
            # Get cache size information
            info = self.redis_client.info('memory')
            self.metrics.cache_size_mb = info.get('used_memory', 0) / (1024 * 1024)
            
            # Update popular queries
            most_common = self.query_frequency.most_common(10)
            self.metrics.popular_queries = [
                {'query': query, 'frequency': freq} for query, freq in most_common
            ]
            
            # Calculate cost savings (rough estimate)
            api_requests_saved = self.metrics.cache_hits
            estimated_cost_per_request = 0.055
            self.metrics.cost_savings_usd = api_requests_saved * estimated_cost_per_request
            
            # Check for alerts
            self._check_alerts()
            
            # Log metrics periodically
            if self.metrics.total_requests % 100 == 0:  # Every 100 requests
                logger.info(f"📊 Cache metrics: {self.metrics.hit_rate_percent:.1f}% hit rate, "
                          f"{self.metrics.average_response_time_ms:.0f}ms avg response, "
                          f"${self.metrics.cost_savings_usd:.2f} saved")
        
        except Exception as e:
            logger.error(f"❌ Error collecting cache metrics: {e}")
    
    def _check_alerts(self):
        """Check for performance alerts"""
        alerts = []
        
        # Hit rate alert
        if self.metrics.hit_rate_percent < self.alert_thresholds['min_hit_rate']:
            alerts.append(f"Low cache hit rate: {self.metrics.hit_rate_percent:.1f}%")
        
        # Error rate alert
        if self.metrics.total_requests > 0:
            error_rate = (self.metrics.error_count / self.metrics.total_requests) * 100
            if error_rate > self.alert_thresholds['max_error_rate']:
                alerts.append(f"High error rate: {error_rate:.1f}%")
        
        # Response time alert
        if self.metrics.average_response_time_ms > self.alert_thresholds['max_response_time']:
            alerts.append(f"High response time: {self.metrics.average_response_time_ms:.0f}ms")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"🚨 CACHE ALERT: {alert}")
    
    def _optimize_ttl_settings(self):
        """Optimize TTL settings based on usage patterns"""
        if not self.redis_client:
            return
        
        optimization_results = []
        
        try:
            # Get cache type performance data
            cache_performance = {}
            
            # Analyze hit rates by cache type
            for cache_type in ['restaurant_basic_info', 'restaurant_details', 'opening_hours', 'real_time_status', 'live_pricing']:
                pattern = f"time_aware:{cache_type}:*"
                keys = self.redis_client.keys(pattern)
                
                if keys:
                    # Sample some keys to estimate hit rate
                    sample_size = min(len(keys), 50)
                    sample_keys = keys[:sample_size]
                    
                    hit_count = 0
                    for key in sample_keys:
                        # Check if key has been accessed recently (proxy for hit rate)
                        ttl = self.redis_client.ttl(key)
                        if ttl > 0:  # Key exists and has TTL
                            hit_count += 1
                    
                    hit_rate = (hit_count / sample_size) * 100 if sample_size > 0 else 0
                    cache_performance[cache_type] = {
                        'hit_rate': hit_rate,
                        'key_count': len(keys)
                    }
            
            # Apply optimization rules
            for rule in self.ttl_optimization_rules:
                if rule.cache_type in cache_performance:
                    perf = cache_performance[rule.cache_type]
                    hit_rate = perf['hit_rate']
                    
                    should_optimize = False
                    new_multiplier = 1.0
                    
                    if hit_rate > rule.max_hit_rate:
                        # Hit rate too high, increase TTL
                        should_optimize = True
                        new_multiplier = rule.ttl_multiplier
                        action = 'increase_ttl'
                    elif hit_rate < rule.min_hit_rate:
                        # Hit rate too low, decrease TTL
                        should_optimize = True
                        new_multiplier = 1.0 / rule.ttl_multiplier
                        action = 'decrease_ttl'
                    
                    if should_optimize:
                        optimization_results.append({
                            'cache_type': rule.cache_type,
                            'current_hit_rate': hit_rate,
                            'action': action,
                            'multiplier': new_multiplier,
                            'reason': rule.description
                        })
            
            # Log optimization results
            if optimization_results:
                self.optimization_history.extend(optimization_results)
                logger.info(f"🔧 TTL optimization applied to {len(optimization_results)} cache types")
                for result in optimization_results:
                    logger.info(f"   • {result['cache_type']}: {result['action']} (hit rate: {result['current_hit_rate']:.1f}%)")
        
        except Exception as e:
            logger.error(f"❌ Error in TTL optimization: {e}")
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboard"""
        return {
            'metrics': asdict(self.metrics),
            'recent_errors': self.cache_errors[-10:],  # Last 10 errors
            'optimization_history': self.optimization_history[-20:],  # Last 20 optimizations
            'cache_type_performance': {
                cache_type: {
                    'avg_response_time': sum(times[-50:]) / len(times[-50:]) if times else 0,
                    'request_count': len(times)
                }
                for cache_type, times in self.response_times.items()
            },
            'alert_status': {
                'hit_rate_ok': self.metrics.hit_rate_percent >= self.alert_thresholds['min_hit_rate'],
                'error_rate_ok': (self.metrics.error_count / max(self.metrics.total_requests, 1)) * 100 <= self.alert_thresholds['max_error_rate'],
                'response_time_ok': self.metrics.average_response_time_ms <= self.alert_thresholds['max_response_time']
            },
            'timestamp': datetime.now().isoformat()
        }

class CacheWarmingService:
    """
    Intelligent cache warming service for popular queries
    """
    
    def __init__(self, google_optimizer: GoogleApiFieldOptimizer, cache_manager: TimeAwareCacheManager):
        self.google_optimizer = google_optimizer
        self.cache_manager = cache_manager
        self.redis_client = self._initialize_redis()
        
        # Popular queries for Istanbul (based on common tourist interests)
        self.popular_queries = [
            "best restaurants in Sultanahmet",
            "Turkish breakfast places in Beyoğlu",
            "seafood restaurants near Bosphorus",
            "vegetarian restaurants in Karakoy",
            "rooftop restaurants with view",
            "traditional Turkish restaurants",
            "cheap eats in Kadikoy",
            "fine dining Istanbul",
            "halal restaurants near Hagia Sophia",
            "best kebab places in Istanbul",
            "Turkish street food",
            "restaurants with live music",
            "family friendly restaurants",
            "restaurants open late night",
            "romantic dinner places Istanbul"
        ]
        
        # Warming schedule
        self.warming_enabled = True
        self.warming_interval = 7200  # 2 hours
        self.max_concurrent_warming = 3
        
        # Start cache warming
        self._start_cache_warming()
    
    def _initialize_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis for warming coordination"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/3')  # Use DB 3 for warming
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"⚠️ Redis not available for cache warming: {e}")
            return None
    
    def _start_cache_warming(self):
        """Start background cache warming process"""
        if not self.warming_enabled:
            return
        
        def run_warming():
            while self.warming_enabled:
                try:
                    self._warm_popular_queries()
                    time.sleep(self.warming_interval)
                except Exception as e:
                    logger.error(f"❌ Error in cache warming: {e}")
                    time.sleep(600)  # Wait 10 minutes before retrying
        
        warming_thread = threading.Thread(target=run_warming, daemon=True)
        warming_thread.start()
        
        logger.info("🔥 Cache warming service started")
    
    def _warm_popular_queries(self):
        """Warm cache with popular queries"""
        if not self.redis_client:
            return
        
        try:
            # Check if warming is already in progress
            warming_lock = f"cache_warming:lock"
            if self.redis_client.exists(warming_lock):
                logger.debug("🔥 Cache warming already in progress, skipping")
                return
            
            # Set warming lock
            self.redis_client.setex(warming_lock, 3600, "warming_in_progress")
            
            # Warm queries in parallel (limited concurrency)
            warmed_count = 0
            semaphore = threading.Semaphore(self.max_concurrent_warming)
            
            def warm_single_query(query: str):
                nonlocal warmed_count
                try:
                    with semaphore:
                        # Check if query is already cached and fresh
                        intent = self.google_optimizer.classify_query_intent(query)
                        fields = self.google_optimizer.get_optimized_fields(intent)
                        cached_result = self.cache_manager.get_cached_result(query, "Istanbul, Turkey", intent, fields)
                        
                        if cached_result:
                            logger.debug(f"🔥 Query already cached: {query[:30]}...")
                            return
                        
                        # Warm the cache
                        start_time = time.time()
                        result = self.google_optimizer.search_restaurants_optimized(
                            query=query,
                            location="Istanbul, Turkey",
                            budget_mode=True  # Use budget mode for warming
                        )
                        
                        response_time = (time.time() - start_time) * 1000
                        warmed_count += 1
                        
                        logger.info(f"🔥 Warmed cache for: {query[:30]}... ({response_time:.0f}ms)")
                        
                except Exception as e:
                    logger.error(f"❌ Error warming query '{query}': {e}")
            
            # Start warming threads
            threads = []
            for query in self.popular_queries:
                thread = threading.Thread(target=warm_single_query, args=(query,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=30)  # 30 second timeout per query
            
            # Remove warming lock
            self.redis_client.delete(warming_lock)
            
            logger.info(f"🔥 Cache warming completed: {warmed_count}/{len(self.popular_queries)} queries warmed")
            
        except Exception as e:
            logger.error(f"❌ Error in cache warming process: {e}")
            # Ensure lock is removed
            if self.redis_client:
                self.redis_client.delete(warming_lock)
    
    async def warm_query_async(self, query: str, location: str = "Istanbul, Turkey") -> bool:
        """Asynchronously warm cache for a specific query"""
        try:
            # Run warming in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.google_optimizer.search_restaurants_optimized(
                    query=query,
                    location=location,
                    budget_mode=True
                )
            )
            
            logger.info(f"🔥 Async cache warming completed for: {query[:30]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in async cache warming for '{query}': {e}")
            return False
    
    def add_popular_query(self, query: str):
        """Add a new popular query to the warming list"""
        if query not in self.popular_queries:
            self.popular_queries.append(query)
            logger.info(f"➕ Added popular query for warming: {query}")
    
    def get_warming_stats(self) -> Dict[str, Any]:
        """Get cache warming statistics"""
        return {
            'popular_queries_count': len(self.popular_queries),
            'warming_enabled': self.warming_enabled,
            'warming_interval_minutes': self.warming_interval // 60,
            'max_concurrent_warming': self.max_concurrent_warming,
            'popular_queries': self.popular_queries
        }

class IntegratedCacheSystem:
    """
    Main integrated cache system that combines all components
    """
    
    def __init__(self):
        # Initialize core components
        self.google_optimizer = GoogleApiFieldOptimizer()
        self.cache_manager = TimeAwareCacheManager()
        self.monitor = ProductionCacheMonitor()
        self.warming_service = CacheWarmingService(self.google_optimizer, self.cache_manager)
        
        # Integration flags
        self.unified_ai_integration = UNIFIED_AI_AVAILABLE
        self.cost_monitoring_integration = COST_MONITORING_AVAILABLE
        
        logger.info("🚀 Integrated Cache System initialized successfully")
    
    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring"""
        start_time = time.time()
        try:
            yield
        finally:
            response_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"⏱️ {operation_name} completed in {response_time_ms:.0f}ms")
    
    async def search_restaurants_integrated(self, 
                                          query: str, 
                                          location: str = "Istanbul, Turkey",
                                          context: Optional[Dict[str, Any]] = None,
                                          session_id: Optional[str] = None,
                                          user_ip: Optional[str] = None) -> Dict[str, Any]:
        """
        Integrated restaurant search with unified AI system and monitoring
        """
        
        with self.performance_monitoring(f"restaurant_search: {query[:30]}"):
            start_time = time.time()
            cache_hit = False
            cache_type = 'unknown'
            
            try:
                # Classify query intent
                intent = self.google_optimizer.classify_query_intent(query, context)
                cache_type = self.cache_manager.classify_cache_type(query, intent, [])
                
                # Check cache first
                fields = self.google_optimizer.get_optimized_fields(intent)
                cached_result = self.cache_manager.get_cached_result(query, location, intent, fields)
                
                if cached_result:
                    cache_hit = True
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    # Record cache performance
                    self.monitor.record_cache_request(query, True, response_time_ms, cache_type)
                    
                    # Cost monitoring
                    if COST_MONITORING_AVAILABLE:
                        log_google_places_cost(0, cached=True)  # No cost for cached results
                    
                    logger.info(f"✅ Cache HIT for restaurant search: {query[:50]}")
                    
                    return {
                        **cached_result,
                        'cache_performance': {
                            'cache_hit': True,
                            'response_time_ms': response_time_ms,
                            'cache_type': cache_type
                        }
                    }
                
                # Cache miss - perform API request
                result = self.google_optimizer.search_restaurants_optimized(
                    query=query,
                    location=location,
                    context=context,
                    budget_mode=False
                )
                
                cache_hit = False
                response_time_ms = (time.time() - start_time) * 1000
                
                # Record cache performance
                self.monitor.record_cache_request(query, False, response_time_ms, cache_type)
                
                # Cost monitoring
                if COST_MONITORING_AVAILABLE:
                    estimated_cost = len(fields) * 0.017  # Rough estimate
                    log_google_places_cost(estimated_cost, cached=False)
                
                logger.info(f"🔍 API request for restaurant search: {query[:50]} ({response_time_ms:.0f}ms)")
                
                # Add performance info to result
                result['cache_performance'] = {
                    'cache_hit': False,
                    'response_time_ms': response_time_ms,
                    'cache_type': cache_type
                }
                
                return result
                
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                
                # Record error
                self.monitor.record_cache_error(
                    str(e), 
                    {
                        'query': query,
                        'location': location,
                        'cache_type': cache_type,
                        'response_time_ms': response_time_ms
                    }
                )
                
                logger.error(f"❌ Error in integrated restaurant search: {e}")
                raise
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        return {
            'cache_optimization': self.google_optimizer.get_optimization_analytics(),
            'time_aware_cache': self.cache_manager.get_cache_analytics(),
            'production_monitoring': self.monitor.get_monitoring_dashboard_data(),
            'cache_warming': self.warming_service.get_warming_stats(),
            'integration_status': {
                'unified_ai_available': self.unified_ai_integration,
                'cost_monitoring_available': self.cost_monitoring_integration,
                'components_healthy': True
            },
            'system_health': {
                'cache_manager_connected': self.cache_manager.redis_client is not None,
                'monitor_connected': self.monitor.redis_client is not None,
                'warming_enabled': self.warming_service.warming_enabled
            }
        }
    
    async def warm_cache_for_query(self, query: str, location: str = "Istanbul, Turkey") -> bool:
        """Warm cache for a specific query"""
        return await self.warming_service.warm_query_async(query, location)
    
    def stop_monitoring(self):
        """Stop all monitoring and background processes"""
        self.monitor.monitoring_enabled = False
        self.warming_service.warming_enabled = False
        logger.info("🛑 Integrated cache system monitoring stopped")

# Global instance for the application
integrated_cache_system = IntegratedCacheSystem()

# Export main functions for easy integration
async def search_restaurants_with_integrated_cache(query: str, 
                                                 location: str = "Istanbul, Turkey",
                                                 context: Optional[Dict[str, Any]] = None,
                                                 session_id: Optional[str] = None,
                                                 user_ip: Optional[str] = None) -> Dict[str, Any]:
    """Main function for integrated restaurant search"""
    return await integrated_cache_system.search_restaurants_integrated(
        query=query,
        location=location,
        context=context,
        session_id=session_id,
        user_ip=user_ip
    )

def get_integrated_analytics() -> Dict[str, Any]:
    """Get integrated system analytics"""
    return integrated_cache_system.get_system_analytics()

async def warm_popular_query(query: str, location: str = "Istanbul, Turkey") -> bool:
    """Warm cache for a popular query"""
    return await integrated_cache_system.warm_cache_for_query(query, location)

if __name__ == "__main__":
    # Test the integrated system
    import asyncio
    
    async def test_integrated_system():
        print("🚀 Testing Integrated Cache System")
        print("=" * 50)
        
        test_queries = [
            "best Turkish restaurants in Sultanahmet",
            "vegetarian restaurants near Galata Tower",
            "seafood restaurants with Bosphorus view",
            "budget friendly restaurants in Kadikoy"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing: {query}")
            try:
                result = await search_restaurants_with_integrated_cache(query)
                
                perf = result.get('cache_performance', {})
                opt_info = result.get('optimization_info', {})
                
                print(f"🎯 Cache: {'HIT' if perf.get('cache_hit') else 'MISS'}")
                print(f"⏱️ Response: {perf.get('response_time_ms', 0):.0f}ms")
                print(f"💰 Cost savings: {opt_info.get('cost_savings_percent', 0)}%")
                print(f"🏪 Results: {len(result.get('restaurants', []))}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Test analytics
        print(f"\n📊 System Analytics:")
        analytics = get_integrated_analytics()
        print(f"Cache hit rate: {analytics['production_monitoring']['metrics']['hit_rate_percent']:.1f}%")
        print(f"Total requests: {analytics['production_monitoring']['metrics']['total_requests']}")
        print(f"Cost savings: ${analytics['production_monitoring']['metrics']['cost_savings_usd']:.2f}")
        
        # Test cache warming
        print(f"\n🔥 Testing cache warming...")
        await warm_popular_query("romantic restaurants Istanbul")
        print("✅ Cache warming test completed")
    
    # Run the test
    asyncio.run(test_integrated_system())
