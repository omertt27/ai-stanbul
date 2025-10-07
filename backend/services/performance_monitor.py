"""
Performance Monitor for Route Maker Service
Phase 3: Performance optimization and monitoring utilities
"""

import time
import psutil
import gc
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, field
import threading
import json

@dataclass
class PerformanceMetric:
    """Individual performance measurement"""
    timestamp: datetime
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    cache_hit: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """
    Comprehensive performance monitoring for route maker service
    Tracks execution times, memory usage, cache performance
    """
    
    def __init__(self, max_metrics: int = 1000):
        self.metrics: List[PerformanceMetric] = []
        self.max_metrics = max_metrics
        self._lock = threading.Lock()
        self.start_time = datetime.now()
        
    def record_metric(self, 
                     operation: str,
                     duration_ms: float,
                     cache_hit: bool = False,
                     error: Optional[str] = None,
                     **metadata):
        """Record a performance metric"""
        try:
            # Get system metrics
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                cache_hit=cache_hit,
                error=error,
                metadata=metadata
            )
            
            with self._lock:
                self.metrics.append(metric)
                # Keep only recent metrics
                if len(self.metrics) > self.max_metrics:
                    self.metrics = self.metrics[-self.max_metrics:]
            
        except Exception as e:
            print(f"Warning: Failed to record performance metric: {e}")
    
    def get_stats(self, operation: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get performance statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            # Filter metrics
            filtered_metrics = [
                m for m in self.metrics 
                if m.timestamp >= cutoff_time and (not operation or m.operation == operation)
            ]
        
        if not filtered_metrics:
            return {"message": "No metrics available for the specified period"}
        
        # Calculate statistics
        durations = [m.duration_ms for m in filtered_metrics]
        memory_usage = [m.memory_mb for m in filtered_metrics]
        errors = [m for m in filtered_metrics if m.error]
        cache_hits = [m for m in filtered_metrics if m.cache_hit]
        
        stats = {
            "period_hours": hours,
            "total_operations": len(filtered_metrics),
            "operations_per_hour": len(filtered_metrics) / max(hours, 1),
            "performance": {
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p95_duration_ms": self._percentile(durations, 95),
                "p99_duration_ms": self._percentile(durations, 99)
            },
            "memory": {
                "avg_mb": sum(memory_usage) / len(memory_usage),
                "min_mb": min(memory_usage),
                "max_mb": max(memory_usage),
                "current_mb": memory_usage[-1] if memory_usage else 0
            },
            "errors": {
                "count": len(errors),
                "rate": len(errors) / len(filtered_metrics) if filtered_metrics else 0,
                "recent_errors": [
                    {"operation": e.operation, "error": e.error, "timestamp": e.timestamp.isoformat()}
                    for e in errors[-5:]  # Last 5 errors
                ]
            },
            "cache": {
                "hit_count": len(cache_hits),
                "hit_rate": len(cache_hits) / len(filtered_metrics) if filtered_metrics else 0
            },
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }
        
        # Operation breakdown
        operation_stats = {}
        for metric in filtered_metrics:
            op = metric.operation
            if op not in operation_stats:
                operation_stats[op] = {"count": 0, "total_duration": 0, "errors": 0}
            operation_stats[op]["count"] += 1
            operation_stats[op]["total_duration"] += metric.duration_ms
            if metric.error:
                operation_stats[op]["errors"] += 1
        
        for op, data in operation_stats.items():
            data["avg_duration_ms"] = data["total_duration"] / data["count"]
            data["error_rate"] = data["errors"] / data["count"]
        
        stats["operations"] = operation_stats
        
        # System info
        stats["system"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        return stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a list"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_slow_operations(self, threshold_ms: float = 1000, limit: int = 10) -> List[Dict]:
        """Get slowest operations above threshold"""
        with self._lock:
            slow_ops = [
                {
                    "operation": m.operation,
                    "duration_ms": m.duration_ms,
                    "timestamp": m.timestamp.isoformat(),
                    "memory_mb": m.memory_mb,
                    "error": m.error,
                    "metadata": m.metadata
                }
                for m in self.metrics 
                if m.duration_ms >= threshold_ms
            ]
        
        # Sort by duration, descending
        slow_ops.sort(key=lambda x: x["duration_ms"], reverse=True)
        return slow_ops[:limit]
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        with self._lock:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_metrics": len(self.metrics),
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "operation": m.operation,
                        "duration_ms": m.duration_ms,
                        "memory_mb": m.memory_mb,
                        "cpu_percent": m.cpu_percent,
                        "cache_hit": m.cache_hit,
                        "error": m.error,
                        "metadata": m.metadata
                    }
                    for m in self.metrics
                ]
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported {len(self.metrics)} metrics to {filepath}")
    
    def clear_metrics(self):
        """Clear all stored metrics"""
        with self._lock:
            cleared_count = len(self.metrics)
            self.metrics.clear()
        return cleared_count

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str = None, track_cache: bool = False):
    """
    Decorator to monitor function performance
    
    Args:
        operation_name: Custom name for the operation (defaults to function name)
        track_cache: Whether to check for cache hit indicators in return value
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            error = None
            cache_hit = False
            
            try:
                result = func(*args, **kwargs)
                
                # Check for cache hit if enabled
                if track_cache:
                    if hasattr(result, 'from_cache'):
                        cache_hit = result.from_cache
                    elif isinstance(result, dict) and 'from_cache' in result:
                        cache_hit = result['from_cache']
                
                return result
                
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                performance_monitor.record_metric(
                    operation=op_name,
                    duration_ms=duration_ms,
                    cache_hit=cache_hit,
                    error=error,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
        
        return wrapper
    return decorator

def optimize_memory():
    """Force garbage collection and return memory stats"""
    before_mb = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Force garbage collection
    collected = gc.collect()
    
    after_mb = psutil.Process().memory_info().rss / 1024 / 1024
    freed_mb = before_mb - after_mb
    
    return {
        "objects_collected": collected,
        "memory_before_mb": round(before_mb, 2),
        "memory_after_mb": round(after_mb, 2),
        "memory_freed_mb": round(freed_mb, 2)
    }

def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health metrics"""
    try:
        process = psutil.Process()
        
        # Memory info
        memory = psutil.virtual_memory()
        process_memory = process.memory_info()
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        process_cpu = process.cpu_percent()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        health = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / 1024 / 1024 / 1024, 2),
                "disk_free_percent": round((disk.free / disk.total) * 100, 2),
                "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 2)
            },
            "process": {
                "cpu_percent": process_cpu,
                "memory_mb": round(process_memory.rss / 1024 / 1024, 2),
                "memory_percent": round(process_memory.rss / memory.total * 100, 2),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            },
            "health_status": "healthy"
        }
        
        # Determine health status
        if (cpu_percent > 90 or memory.percent > 90 or 
            process_memory.rss / memory.total > 0.5):
            health["health_status"] = "warning"
        
        if (cpu_percent > 95 or memory.percent > 95 or 
            disk.free / disk.total < 0.1):
            health["health_status"] = "critical"
        
        return health
        
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "health_status": "error",
            "error": str(e)
        }

# Utility functions for route optimization
def profile_route_generation(func):
    """Profile route generation specifically"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            result = None
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
            # Extract metadata from result if available
            metadata = {}
            if result and hasattr(result, 'points'):
                metadata['num_points'] = len(result.points)
                metadata['total_distance'] = getattr(result, 'total_distance_km', 0)
                metadata['num_attractions'] = len([p for p in result.points if p.attraction_id])
            
            performance_monitor.record_metric(
                operation="route_generation",
                duration_ms=duration_ms,
                error=error,
                memory_delta_mb=memory_delta,
                success=success,
                **metadata
            )
        
        return result
    return wrapper

# Export the monitoring functions
__all__ = [
    'PerformanceMonitor',
    'performance_monitor', 
    'monitor_performance',
    'optimize_memory',
    'get_system_health',
    'profile_route_generation'
]
