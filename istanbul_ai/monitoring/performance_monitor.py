"""
Performance Monitor for Istanbul AI
Real-time performance monitoring and metrics collection
"""

import time
import psutil
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    request_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    memory_usage: float = 0.0
    cache_hit_rate: float = 0.0

class PerformanceMonitor:
    """
    Real-time performance monitoring
    Tracks system health and performance metrics
    """
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Try to import GPU monitoring
        self.gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
            logger.info("✅ GPU monitoring enabled")
        except:
            logger.warning("⚠️ GPU monitoring not available")
    
    async def start_monitoring(self):
        """Start background monitoring"""
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(10)  # Every 10 seconds
                
                # Update CPU and memory
                self.metrics.cpu_utilization = psutil.cpu_percent(interval=1)
                self.metrics.memory_usage = psutil.virtual_memory().percent
                
                # Update GPU if available
                if self.gpu_available:
                    try:
                        import pynvml
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        self.metrics.gpu_utilization = util.gpu
                    except:
                        pass
                
                # Log metrics
                logger.info(f"📊 System Metrics: "
                          f"CPU={self.metrics.cpu_utilization:.1f}% "
                          f"Memory={self.metrics.memory_usage:.1f}% "
                          f"GPU={self.metrics.gpu_utilization:.1f}%")
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    def record_request(self, latency: float, success: bool = True):
        """Record a request"""
        self.metrics.request_count += 1
        if not success:
            self.metrics.error_count += 1
        
        self.metrics.total_latency += latency
        self.metrics.response_times.append(latency)
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        uptime = time.time() - self.start_time
        
        # Calculate percentiles
        sorted_times = sorted(self.metrics.response_times)
        n = len(sorted_times)
        
        if n > 0:
            p50 = sorted_times[int(n * 0.50)]
            p95 = sorted_times[int(n * 0.95)]
            p99 = sorted_times[int(n * 0.99)]
            avg = sum(sorted_times) / n
        else:
            p50 = p95 = p99 = avg = 0
        
        return {
            "uptime_seconds": int(uptime),
            "total_requests": self.metrics.request_count,
            "total_errors": self.metrics.error_count,
            "error_rate": (self.metrics.error_count / 
                          max(self.metrics.request_count, 1)) * 100,
            "requests_per_second": self.metrics.request_count / uptime,
            "avg_latency_ms": avg * 1000,
            "p50_latency_ms": p50 * 1000,
            "p95_latency_ms": p95 * 1000,
            "p99_latency_ms": p99 * 1000,
            "cpu_percent": self.metrics.cpu_utilization,
            "memory_percent": self.metrics.memory_usage,
            "gpu_percent": self.metrics.gpu_utilization,
            "cache_hit_rate": self.metrics.cache_hit_rate
        }


# Global monitor
_performance_monitor: Optional[PerformanceMonitor] = None

async def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        await _performance_monitor.start_monitoring()
    return _performance_monitor
