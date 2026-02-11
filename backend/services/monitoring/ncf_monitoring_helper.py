"""
NCF-Specific Monitoring Helpers

Extension of the main metrics collector for Phase 2 NCF features.

Author: AI Istanbul Team
Date: February 10, 2026
"""

import logging
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)


class NCFMonitoringHelper:
    """
    Helper class for NCF-specific monitoring tasks.
    
    Works alongside the main MetricsCollector to provide
    NCF-specific analytics and reporting.
    """
    
    def __init__(self, metrics_collector):
        """
        Initialize NCF monitoring helper.
        
        Args:
            metrics_collector: Main NCFMetricsCollector instance
        """
        self.metrics = metrics_collector
        self.inference_history = []
        self.max_history_size = 1000
    
    def track_onnx_inference(self, duration_ms: float, batch_size: int = 1):
        """Track ONNX model inference."""
        self.metrics.onnx_inference_time.labels(
            batch_size=str(batch_size)
        ).observe(duration_ms / 1000.0)
        
        # Keep history for charts
        self.inference_history.append({
            'timestamp': time.time(),
            'duration_ms': duration_ms,
            'model_type': 'onnx',
            'batch_size': batch_size
        })
        
        # Trim history
        if len(self.inference_history) > self.max_history_size:
            self.inference_history = self.inference_history[-self.max_history_size:]
    
    def track_pytorch_inference(self, duration_ms: float, batch_size: int = 1):
        """Track PyTorch model inference."""
        self.metrics.pytorch_inference_time.labels(
            batch_size=str(batch_size)
        ).observe(duration_ms / 1000.0)
    
    def update_speedup_ratio(self, onnx_ms: float, pytorch_ms: float):
        """Calculate and update ONNX speedup ratio."""
        if onnx_ms > 0:
            speedup = pytorch_ms / onnx_ms
            self.metrics.onnx_speedup_ratio.set(speedup)
            logger.info(f"ONNX speedup: {speedup:.2f}x")
    
    def update_model_quality(
        self,
        accuracy: Optional[float] = None,
        diversity: Optional[float] = None,
        novelty: Optional[float] = None,
        coverage: Optional[float] = None
    ):
        """
        Update model quality metrics.
        
        Args:
            accuracy: Model accuracy (0-1)
            diversity: Recommendation diversity (0-1)
            novelty: Recommendation novelty (0-1)
            coverage: Catalog coverage (0-1)
        """
        if accuracy is not None:
            self.metrics.recommendation_accuracy.set(accuracy)
        
        if diversity is not None:
            self.metrics.recommendation_diversity.set(diversity)
        
        if novelty is not None:
            self.metrics.recommendation_novelty.set(novelty)
        
        if coverage is not None:
            self.metrics.recommendation_coverage.set(coverage)
    
    def track_fallback(self, reason: str):
        """Track fallback recommendation usage."""
        self.metrics.fallback_recommendations.labels(reason=reason).inc()
        logger.warning(f"Fallback recommendation triggered: {reason}")
    
    def track_model_error(self, error_type: str):
        """Track model inference error."""
        self.metrics.model_errors.labels(error_type=error_type).inc()
        logger.error(f"Model error: {error_type}")
    
    def get_latency_stats(self, model_type: str = "onnx", hours: int = 1) -> Dict[str, Any]:
        """
        Get latency statistics for a time window.
        
        Args:
            model_type: 'onnx' or 'pytorch'
            hours: Number of hours to look back
            
        Returns:
            Dictionary with latency statistics
        """
        cutoff_time = time.time() - (hours * 3600)
        
        # Filter history
        relevant = [
            h for h in self.inference_history
            if h['timestamp'] > cutoff_time and h['model_type'] == model_type
        ]
        
        if not relevant:
            return {
                'count': 0,
                'avg_ms': 0,
                'p50_ms': 0,
                'p95_ms': 0,
                'p99_ms': 0
            }
        
        # Calculate statistics
        durations = sorted([h['duration_ms'] for h in relevant])
        count = len(durations)
        
        return {
            'count': count,
            'avg_ms': sum(durations) / count,
            'p50_ms': durations[int(count * 0.5)],
            'p95_ms': durations[int(count * 0.95)],
            'p99_ms': durations[int(count * 0.99)],
            'min_ms': min(durations),
            'max_ms': max(durations)
        }
    
    def get_latency_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get latency history for charting.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of latency data points
        """
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            h for h in self.inference_history
            if h['timestamp'] > cutoff_time
        ]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache stats
        """
        # Note: In production, you'd query actual Prometheus metrics
        # This is a placeholder implementation
        return {
            'hit_rate': 0.65,  # 65% cache hit rate (example)
            'miss_rate': 0.35,
            'total_requests': 1000,
            'hits': 650,
            'misses': 350,
            'avg_hit_latency_ms': 0.8,
            'avg_miss_latency_ms': 12.5
        }
    
    def calculate_speedup_from_history(self, hours: int = 1) -> Optional[float]:
        """
        Calculate average speedup ratio from recent history.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Average speedup ratio or None if insufficient data
        """
        onnx_stats = self.get_latency_stats('onnx', hours)
        pytorch_stats = self.get_latency_stats('pytorch', hours)
        
        if onnx_stats['count'] == 0 or pytorch_stats['count'] == 0:
            return None
        
        speedup = pytorch_stats['avg_ms'] / onnx_stats['avg_ms']
        return speedup


# Global helper instance
_ncf_monitoring_helper: Optional[NCFMonitoringHelper] = None


def get_ncf_monitoring_helper():
    """Get or create global NCF monitoring helper."""
    global _ncf_monitoring_helper
    
    if _ncf_monitoring_helper is None:
        from backend.services.monitoring.metrics_collector import get_metrics_collector
        metrics = get_metrics_collector()
        _ncf_monitoring_helper = NCFMonitoringHelper(metrics)
        logger.info("Created NCF monitoring helper")
    
    return _ncf_monitoring_helper
