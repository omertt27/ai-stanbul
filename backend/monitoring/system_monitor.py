"""
System Monitor for health checks, metrics, and observability
Provides real-time monitoring of system health and performance
"""

import logging
import psutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Real-time system monitoring and health checks"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize system monitor
        
        Args:
            window_size: Number of recent metrics to keep in memory
        """
        self.window_size = window_size
        self.metrics = {
            "request_times": deque(maxlen=window_size),
            "errors": deque(maxlen=window_size),
            "cache_hits": deque(maxlen=window_size),
        }
        self.counters = defaultdict(int)
        self.service_health = {}
        self.start_time = datetime.now()
        logger.info("✅ SystemMonitor initialized")
    
    def record_request(self, duration: float, success: bool = True, 
                      intent: str = None, service: str = None):
        """
        Record request metrics
        
        Args:
            duration: Request duration in seconds
            success: Whether request succeeded
            intent: Intent type (optional)
            service: Service name (optional)
        """
        self.metrics["request_times"].append({
            "duration": duration,
            "success": success,
            "timestamp": datetime.now(),
            "intent": intent,
            "service": service
        })
        
        self.counters["total_requests"] += 1
        if success:
            self.counters["successful_requests"] += 1
        else:
            self.counters["failed_requests"] += 1
            self.metrics["errors"].append({
                "timestamp": datetime.now(),
                "intent": intent,
                "service": service
            })
    
    def record_cache_access(self, hit: bool, intent: str = None):
        """
        Record cache access metrics
        
        Args:
            hit: Whether cache hit occurred
            intent: Intent type (optional)
        """
        self.metrics["cache_hits"].append({
            "hit": hit,
            "timestamp": datetime.now(),
            "intent": intent
        })
        
        if hit:
            self.counters["cache_hits"] += 1
        else:
            self.counters["cache_misses"] += 1
    
    def record_service_health(self, service: str, healthy: bool, 
                             response_time: float = None, message: str = None):
        """
        Record service health status
        
        Args:
            service: Service name
            healthy: Whether service is healthy
            response_time: Service response time (optional)
            message: Health check message (optional)
        """
        self.service_health[service] = {
            "healthy": healthy,
            "response_time": response_time,
            "message": message,
            "last_check": datetime.now()
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system resource metrics
        
        Returns:
            Dictionary with CPU, memory, disk metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent": memory.percent,
                    "used_gb": round(memory.used / (1024**3), 2)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": disk.percent
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting system metrics: {e}")
            return {}
    
    def get_request_metrics(self) -> Dict[str, Any]:
        """
        Get request performance metrics
        
        Returns:
            Dictionary with request statistics
        """
        if not self.metrics["request_times"]:
            return {
                "total": 0,
                "avg_duration_ms": 0,
                "min_duration_ms": 0,
                "max_duration_ms": 0,
                "success_rate": 0
            }
        
        durations = [r["duration"] for r in self.metrics["request_times"]]
        successes = [r["success"] for r in self.metrics["request_times"]]
        
        return {
            "total": self.counters["total_requests"],
            "success": self.counters["successful_requests"],
            "failed": self.counters["failed_requests"],
            "avg_duration_ms": round(sum(durations) / len(durations) * 1000, 2),
            "min_duration_ms": round(min(durations) * 1000, 2),
            "max_duration_ms": round(max(durations) * 1000, 2),
            "success_rate": f"{sum(successes) / len(successes) * 100:.1f}%",
            "requests_per_minute": self._calculate_rate(self.metrics["request_times"])
        }
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics
        
        Returns:
            Dictionary with cache statistics
        """
        total = self.counters["cache_hits"] + self.counters["cache_misses"]
        hit_rate = (self.counters["cache_hits"] / total * 100) if total > 0 else 0
        
        return {
            "hits": self.counters["cache_hits"],
            "misses": self.counters["cache_misses"],
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%"
        }
    
    def get_error_rate(self, window_minutes: int = 5) -> float:
        """
        Calculate error rate in recent time window
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Error rate as percentage
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_requests = [
            r for r in self.metrics["request_times"]
            if r["timestamp"] > cutoff
        ]
        
        if not recent_requests:
            return 0.0
        
        errors = sum(1 for r in recent_requests if not r["success"])
        return (errors / len(recent_requests)) * 100
    
    def _calculate_rate(self, metrics: deque, window_minutes: int = 1) -> float:
        """
        Calculate rate of events per minute
        
        Args:
            metrics: Deque of metric entries with timestamps
            window_minutes: Time window in minutes
            
        Returns:
            Events per minute
        """
        if not metrics:
            return 0.0
        
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent = [m for m in metrics if m["timestamp"] > cutoff]
        
        return len(recent) / window_minutes if recent else 0.0
    
    def get_uptime(self) -> Dict[str, Any]:
        """
        Get system uptime
        
        Returns:
            Dictionary with uptime information
        """
        uptime_delta = datetime.now() - self.start_time
        
        days = uptime_delta.days
        hours, remainder = divmod(uptime_delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return {
            "start_time": self.start_time.isoformat(),
            "uptime_seconds": uptime_delta.total_seconds(),
            "uptime_formatted": f"{days}d {hours}h {minutes}m {seconds}s"
        }
    
    def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check
        
        Returns:
            Dictionary with health status and details
        """
        health_status = "healthy"
        issues = []
        
        # Check system resources
        system_metrics = self.get_system_metrics()
        
        if system_metrics.get("cpu", {}).get("percent", 0) > 90:
            health_status = "degraded"
            issues.append("High CPU usage")
        
        if system_metrics.get("memory", {}).get("percent", 0) > 90:
            health_status = "degraded"
            issues.append("High memory usage")
        
        if system_metrics.get("disk", {}).get("percent", 0) > 90:
            health_status = "degraded"
            issues.append("Low disk space")
        
        # Check error rate
        error_rate = self.get_error_rate(window_minutes=5)
        if error_rate > 10:
            health_status = "degraded"
            issues.append(f"High error rate: {error_rate:.1f}%")
        elif error_rate > 50:
            health_status = "unhealthy"
            issues.append(f"Critical error rate: {error_rate:.1f}%")
        
        # Check service health
        unhealthy_services = [
            name for name, status in self.service_health.items()
            if not status.get("healthy", False)
        ]
        
        if unhealthy_services:
            health_status = "degraded"
            issues.append(f"Unhealthy services: {', '.join(unhealthy_services)}")
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "system": system_metrics,
            "services": self.service_health,
            "uptime": self.get_uptime()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring dashboard data
        
        Returns:
            Dictionary with all monitoring metrics
        """
        return {
            "health": self.check_health(),
            "system": self.get_system_metrics(),
            "requests": self.get_request_metrics(),
            "cache": self.get_cache_metrics(),
            "error_rate_5min": f"{self.get_error_rate(5):.1f}%",
            "uptime": self.get_uptime(),
            "services": self.service_health
        }
    
    def reset_counters(self):
        """Reset all counters and metrics"""
        self.counters.clear()
        for metric in self.metrics.values():
            metric.clear()
        self.service_health.clear()
        self.start_time = datetime.now()
        logger.info("🔄 Monitor counters reset")


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing SystemMonitor...")
    
    monitor = SystemMonitor(window_size=10)
    
    # Test request recording
    monitor.record_request(0.5, success=True, intent="restaurant")
    monitor.record_request(1.2, success=True, intent="museum")
    monitor.record_request(0.8, success=False, intent="event")
    print("✅ Request recording test passed")
    
    # Test cache recording
    monitor.record_cache_access(hit=True, intent="restaurant")
    monitor.record_cache_access(hit=False, intent="museum")
    print("✅ Cache recording test passed")
    
    # Test service health
    monitor.record_service_health("openai", True, 0.3, "OK")
    monitor.record_service_health("redis", False, None, "Connection failed")
    print("✅ Service health test passed")
    
    # Test metrics
    system_metrics = monitor.get_system_metrics()
    assert "cpu" in system_metrics, "Should have CPU metrics"
    assert "memory" in system_metrics, "Should have memory metrics"
    print("✅ System metrics test passed")
    print(f"📊 CPU: {system_metrics['cpu']['percent']}%")
    print(f"📊 Memory: {system_metrics['memory']['percent']}%")
    
    request_metrics = monitor.get_request_metrics()
    assert request_metrics["total"] == 3, "Should have 3 requests"
    print("✅ Request metrics test passed")
    print(f"📊 {request_metrics}")
    
    cache_metrics = monitor.get_cache_metrics()
    assert cache_metrics["hits"] == 1, "Should have 1 cache hit"
    assert cache_metrics["misses"] == 1, "Should have 1 cache miss"
    print("✅ Cache metrics test passed")
    
    # Test health check
    health = monitor.check_health()
    assert "status" in health, "Should have health status"
    print("✅ Health check test passed")
    print(f"📊 Health: {health['status']}")
    
    # Test dashboard data
    dashboard = monitor.get_dashboard_data()
    assert "health" in dashboard, "Should have health data"
    assert "requests" in dashboard, "Should have request data"
    print("✅ Dashboard data test passed")
    
    print("\n✅ All SystemMonitor tests passed!")
