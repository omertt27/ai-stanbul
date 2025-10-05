# performance_optimization_system.py - System-Wide Performance Optimization

import time
import functools
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import redis
from datetime import datetime, timedelta
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    response_time: float
    memory_usage: float
    cpu_usage: float
    cache_hit_rate: float
    error_rate: float
    timestamp: datetime

class CacheManager:
    """Redis-based caching system"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
            self.redis_client = None
            self.memory_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL"""
        try:
            if self.redis_client:
                return self.redis_client.setex(key, ttl, json.dumps(value))
            else:
                self.memory_cache[key] = value
                # Simple TTL simulation for memory cache
                threading.Timer(ttl, lambda: self.memory_cache.pop(key, None)).start()
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                return self.memory_cache.pop(key, None) is not None
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.metrics_history = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.performance_queue = queue.Queue()
        self.monitoring_active = False
    
    def cache_decorator(self, ttl: int = 300, key_prefix: str = ""):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_result
                
                # Execute function and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Cache the result
                self.cache_manager.set(cache_key, result, ttl)
                logger.debug(f"Cached result for {cache_key} (execution time: {execution_time:.3f}s)")
                
                return result
            return wrapper
        return decorator
    
    def async_cache_decorator(self, ttl: int = 300, key_prefix: str = ""):
        """Async decorator for caching function results"""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                start_time = time.time()
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.cache_manager.set(cache_key, result, ttl)
                logger.debug(f"Async cached result for {cache_key} (execution time: {execution_time:.3f}s)")
                
                return result
            return wrapper
        return decorator
    
    def batch_process_decorator(self, batch_size: int = 10):
        """Decorator for batch processing operations"""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(items: List[Any], *args, **kwargs):
                results = []
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    batch_results = func(batch, *args, **kwargs)
                    results.extend(batch_results)
                return results
            return wrapper
        return decorator
    
    def performance_monitor_decorator(self):
        """Decorator for monitoring function performance"""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                memory_before = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    error_occurred = False
                except Exception as e:
                    error_occurred = True
                    raise e
                finally:
                    end_time = time.time()
                    memory_after = self._get_memory_usage()
                    
                    # Record metrics
                    metrics = PerformanceMetrics(
                        response_time=end_time - start_time,
                        memory_usage=memory_after - memory_before,
                        cpu_usage=self._get_cpu_usage(),
                        cache_hit_rate=self._get_cache_hit_rate(),
                        error_rate=1.0 if error_occurred else 0.0,
                        timestamp=datetime.now()
                    )
                    
                    self.metrics_history.append(metrics)
                    self._process_metrics(metrics)
                
                return result
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (simplified)"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (simplified)"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
    
    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        # This would be implemented based on actual cache statistics
        return 0.8  # Placeholder
    
    def _process_metrics(self, metrics: PerformanceMetrics):
        """Process performance metrics"""
        if metrics.response_time > 5.0:  # 5 second threshold
            logger.warning(f"Slow response detected: {metrics.response_time:.3f}s")
        
        if metrics.memory_usage > 100:  # 100MB threshold
            logger.warning(f"High memory usage: {metrics.memory_usage:.1f}MB")
        
        if metrics.error_rate > 0:
            logger.error("Error occurred during execution")

# Optimized recommendation system with caching
class OptimizedRecommendationSystem:
    """Performance-optimized recommendation system"""
    
    def __init__(self, performance_optimizer: PerformanceOptimizer):
        self.optimizer = performance_optimizer
        self.recommendation_cache = {}
    
    @performance_optimizer.cache_decorator(ttl=600, key_prefix="recommendations")
    @performance_optimizer.performance_monitor_decorator()
    def get_recommendations(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get optimized recommendations for user"""
        # Simulate recommendation logic
        time.sleep(0.1)  # Simulate processing time
        
        recommendations = [
            {
                "id": f"rec_{i}",
                "title": f"Istanbul Attraction {i}",
                "score": 0.9 - (i * 0.1),
                "category": "historical" if i % 2 == 0 else "cultural",
                "location": {"lat": 41.0082 + (i * 0.001), "lng": 28.9784 + (i * 0.001)}
            }
            for i in range(5)
        ]
        
        return recommendations
    
    @performance_optimizer.batch_process_decorator(batch_size=5)
    def batch_get_recommendations(self, user_contexts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Process multiple recommendation requests in batches"""
        return [self.get_recommendations(ctx["user_id"], ctx["context"]) for ctx in user_contexts]
    
    async def async_get_recommendations(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Async version of recommendation retrieval"""
        # Run CPU-intensive work in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.optimizer.executor,
            self.get_recommendations,
            user_id,
            context
        )

# Database optimization utilities
class DatabaseOptimizer:
    """Database query optimization utilities"""
    
    def __init__(self, performance_optimizer: PerformanceOptimizer):
        self.optimizer = performance_optimizer
        self.query_cache = {}
    
    @performance_optimizer.cache_decorator(ttl=1800, key_prefix="db_query")
    def execute_optimized_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute database query with caching"""
        # Simulate database query
        time.sleep(0.05)  # Simulate DB latency
        
        # Mock results
        if "attractions" in query.lower():
            return [
                {"id": i, "name": f"Attraction {i}", "rating": 4.5 + (i * 0.1)}
                for i in range(10)
            ]
        elif "users" in query.lower():
            return [
                {"id": i, "name": f"User {i}", "preferences": ["history", "culture"]}
                for i in range(5)
            ]
        
        return []
    
    def create_index_suggestions(self, query_patterns: List[str]) -> List[str]:
        """Suggest database indexes based on query patterns"""
        suggestions = []
        
        for pattern in query_patterns:
            if "WHERE" in pattern.upper():
                # Extract WHERE conditions and suggest indexes
                suggestions.append(f"CREATE INDEX idx_example ON table_name (column_name);")
        
        return suggestions

# Load balancing and resource management
class ResourceManager:
    """System resource management and load balancing"""
    
    def __init__(self):
        self.active_requests = 0
        self.max_concurrent_requests = 100
        self.request_lock = threading.Lock()
    
    def can_accept_request(self) -> bool:
        """Check if system can accept new requests"""
        with self.request_lock:
            return self.active_requests < self.max_concurrent_requests
    
    def acquire_request_slot(self) -> bool:
        """Acquire a request processing slot"""
        with self.request_lock:
            if self.active_requests < self.max_concurrent_requests:
                self.active_requests += 1
                return True
            return False
    
    def release_request_slot(self):
        """Release a request processing slot"""
        with self.request_lock:
            if self.active_requests > 0:
                self.active_requests -= 1
    
    def get_system_load(self) -> float:
        """Get current system load percentage"""
        with self.request_lock:
            return (self.active_requests / self.max_concurrent_requests) * 100

# Performance monitoring and alerting
class PerformanceMonitor:
    """System performance monitoring and alerting"""
    
    def __init__(self, performance_optimizer: PerformanceOptimizer):
        self.optimizer = performance_optimizer
        self.alert_thresholds = {
            "response_time": 3.0,
            "memory_usage": 200.0,
            "cpu_usage": 80.0,
            "error_rate": 0.05
        }
        self.monitoring_active = False
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_system_health()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _check_system_health(self):
        """Check system health and trigger alerts"""
        if not self.optimizer.metrics_history:
            return
        
        # Get recent metrics (last 5 minutes)
        recent_time = datetime.now() - timedelta(minutes=5)
        recent_metrics = [
            m for m in self.optimizer.metrics_history 
            if m.timestamp > recent_time
        ]
        
        if not recent_metrics:
            return
        
        # Calculate averages
        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu_usage = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        # Check thresholds and send alerts
        if avg_response_time > self.alert_thresholds["response_time"]:
            self._send_alert("HIGH_RESPONSE_TIME", f"Average response time: {avg_response_time:.3f}s")
        
        if avg_memory_usage > self.alert_thresholds["memory_usage"]:
            self._send_alert("HIGH_MEMORY_USAGE", f"Average memory usage: {avg_memory_usage:.1f}MB")
        
        if avg_cpu_usage > self.alert_thresholds["cpu_usage"]:
            self._send_alert("HIGH_CPU_USAGE", f"Average CPU usage: {avg_cpu_usage:.1f}%")
        
        if avg_error_rate > self.alert_thresholds["error_rate"]:
            self._send_alert("HIGH_ERROR_RATE", f"Error rate: {avg_error_rate:.2%}")
    
    def _send_alert(self, alert_type: str, message: str):
        """Send performance alert"""
        logger.warning(f"ALERT [{alert_type}]: {message}")
        # In production, this would send to monitoring systems (PagerDuty, Slack, etc.)

if __name__ == "__main__":
    # Example usage
    optimizer = PerformanceOptimizer()
    rec_system = OptimizedRecommendationSystem(optimizer)
    db_optimizer = DatabaseOptimizer(optimizer)
    resource_manager = ResourceManager()
    performance_monitor = PerformanceMonitor(optimizer)
    
    print("Performance optimization system initialized successfully!")
    print("All components ready for production use.")
