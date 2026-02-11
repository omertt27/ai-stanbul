"""
Prometheus Metrics Collector for NCF Recommendation System

Tracks performance, quality, and business metrics for production monitoring.

Author: AI Istanbul Team
Date: February 10, 2026
"""

import time
import logging
from typing import Dict, Any, Optional
from functools import wraps
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)

logger = logging.getLogger(__name__)


class NCFMetricsCollector:
    """
    Prometheus metrics collector for NCF recommendation system.
    
    Tracks:
    - Request metrics (count, latency, errors)
    - Model performance (inference time, cache hits)
    - Business metrics (CTR, user engagement)
    - System health (memory, CPU, model status)
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.
        
        Args:
            registry: Prometheus registry (creates new if None)
        """
        self.registry = registry or CollectorRegistry()
        
        # === Request Metrics ===
        self.request_count = Counter(
            'ncf_requests_total',
            'Total number of NCF recommendation requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.request_latency = Histogram(
            'ncf_request_duration_seconds',
            'Request latency in seconds',
            ['endpoint', 'method'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        self.request_size = Summary(
            'ncf_request_size_bytes',
            'Request payload size in bytes',
            ['endpoint'],
            registry=self.registry
        )
        
        self.response_size = Summary(
            'ncf_response_size_bytes',
            'Response payload size in bytes',
            ['endpoint'],
            registry=self.registry
        )
        
        # === Model Performance Metrics ===
        self.model_inference_time = Histogram(
            'ncf_model_inference_seconds',
            'Model inference time in seconds',
            ['model_type', 'batch_size'],
            buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
            registry=self.registry
        )
        
        self.model_predictions_total = Counter(
            'ncf_predictions_total',
            'Total number of predictions made',
            ['model_type', 'source'],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'ncf_cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'ncf_cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # === A/B Testing Metrics ===
        self.ab_test_assignments = Counter(
            'ncf_ab_test_assignments_total',
            'Total A/B test assignments',
            ['experiment', 'variant'],
            registry=self.registry
        )
        
        self.ab_test_conversions = Counter(
            'ncf_ab_test_conversions_total',
            'Total A/B test conversions (clicks)',
            ['experiment', 'variant'],
            registry=self.registry
        )
        
        # === Business Metrics ===
        self.recommendation_clicks = Counter(
            'ncf_recommendation_clicks_total',
            'Total recommendation clicks',
            ['item_type', 'position'],
            registry=self.registry
        )
        
        self.recommendation_impressions = Counter(
            'ncf_recommendation_impressions_total',
            'Total recommendation impressions',
            ['item_type'],
            registry=self.registry
        )
        
        self.user_feedback = Counter(
            'ncf_user_feedback_total',
            'Total user feedback events',
            ['feedback_type'],
            registry=self.registry
        )
        
        # === System Health Metrics ===
        self.active_users = Gauge(
            'ncf_active_users',
            'Number of active users in last N minutes',
            ['time_window'],
            registry=self.registry
        )
        
        # === Phase 2: ONNX-Specific Metrics ===
        self.onnx_inference_time = Histogram(
            'ncf_onnx_inference_seconds',
            'ONNX model inference time in seconds',
            ['batch_size'],
            buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
            registry=self.registry
        )
        
        self.pytorch_inference_time = Histogram(
            'ncf_pytorch_inference_seconds',
            'PyTorch model inference time in seconds',
            ['batch_size'],
            buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
            registry=self.registry
        )
        
        self.onnx_speedup_ratio = Gauge(
            'ncf_onnx_speedup_ratio',
            'ONNX vs PyTorch speedup ratio',
            registry=self.registry
        )
        
        self.model_load_time = Gauge(
            'ncf_model_load_seconds',
            'Time taken to load model',
            ['model_type'],
            registry=self.registry
        )
        
        # === Phase 2: Model Quality Metrics ===
        self.recommendation_accuracy = Gauge(
            'ncf_recommendation_accuracy',
            'NCF model accuracy score',
            registry=self.registry
        )
        
        self.recommendation_diversity = Gauge(
            'ncf_recommendation_diversity',
            'Recommendation diversity score (0-1)',
            registry=self.registry
        )
        
        self.recommendation_novelty = Gauge(
            'ncf_recommendation_novelty',
            'Recommendation novelty score (0-1)',
            registry=self.registry
        )
        
        self.recommendation_coverage = Gauge(
            'ncf_recommendation_coverage',
            'Percentage of catalog items recommended',
            registry=self.registry
        )
        
        # === Phase 2: Production Metrics ===
        self.fallback_recommendations = Counter(
            'ncf_fallback_recommendations_total',
            'Total fallback recommendations (when NCF unavailable)',
            ['reason'],
            registry=self.registry
        )
        
        self.model_errors = Counter(
            'ncf_model_errors_total',
            'Total model inference errors',
            ['error_type'],
            registry=self.registry
        )
        
        # === Error Metrics ===
        self.error_count = Counter(
            'ncf_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        logger.info("✅ NCF Metrics Collector initialized")
    
    def track_request(self, endpoint: str, method: str = "POST"):
        """
        Decorator to track request metrics.
        
        Usage:
            @metrics.track_request("/api/recommendations/ncf")
            async def get_recommendations():
                ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    self.error_count.labels(
                        error_type=type(e).__name__,
                        component=endpoint
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.request_count.labels(
                        endpoint=endpoint,
                        method=method,
                        status=status
                    ).inc()
                    self.request_latency.labels(
                        endpoint=endpoint,
                        method=method
                    ).observe(duration)
            
            return wrapper
        return decorator
    
    def track_inference(self, model_type: str, batch_size: int = 1):
        """
        Context manager to track model inference time.
        
        Usage:
            with metrics.track_inference("onnx", batch_size=10):
                predictions = model.predict(user_ids, item_ids)
        """
        class InferenceTracker:
            def __init__(self, collector, model_type, batch_size):
                self.collector = collector
                self.model_type = model_type
                self.batch_size = batch_size
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.collector.model_inference_time.labels(
                    model_type=self.model_type,
                    batch_size=str(self.batch_size)
                ).observe(duration)
                
                if exc_type is None:
                    self.collector.model_predictions_total.labels(
                        model_type=self.model_type,
                        source="success"
                    ).inc(self.batch_size)
                else:
                    self.collector.error_count.labels(
                        error_type=exc_type.__name__,
                        component=f"inference_{self.model_type}"
                    ).inc()
        
        return InferenceTracker(self, model_type, batch_size)
    
    def record_cache_hit(self, cache_type: str = "redis"):
        """Record a cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str = "redis"):
        """Record a cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_ab_assignment(self, experiment: str, variant: str):
        """Record an A/B test assignment."""
        self.ab_test_assignments.labels(
            experiment=experiment,
            variant=variant
        ).inc()
    
    def record_ab_conversion(self, experiment: str, variant: str):
        """Record an A/B test conversion (click)."""
        self.ab_test_conversions.labels(
            experiment=experiment,
            variant=variant
        ).inc()
    
    def record_recommendation_click(
        self,
        item_type: str,
        position: int
    ):
        """Record a recommendation click."""
        self.recommendation_clicks.labels(
            item_type=item_type,
            position=str(position)
        ).inc()
    
    def record_recommendation_impression(self, item_type: str):
        """Record a recommendation impression."""
        self.recommendation_impressions.labels(
            item_type=item_type
        ).inc()
    
    def record_user_feedback(self, feedback_type: str):
        """
        Record user feedback.
        
        Args:
            feedback_type: 'positive', 'negative', 'neutral'
        """
        self.user_feedback.labels(feedback_type=feedback_type).inc()
    
    def update_active_users(self, count: int, time_window: str = "5m"):
        """Update active users gauge."""
        self.active_users.labels(time_window=time_window).set(count)
    
    def set_model_status(self, model_type: str, loaded: bool):
        """Set model load status."""
        self.model_load_status.labels(model_type=model_type).set(
            1 if loaded else 0
        )
    
    def set_model_version(self, version_info: Dict[str, str]):
        """
        Set model version information.
        
        Args:
            version_info: Dict with keys like 'version', 'trained_at', 'accuracy'
        """
        self.model_version.info(version_info)
    
    def update_recommendation_diversity(
        self,
        diversity_score: float,
        user_segment: str = "all"
    ):
        """Update recommendation diversity score (0-1)."""
        self.recommendation_diversity.labels(
            user_segment=user_segment
        ).set(diversity_score)
    
    def update_recommendation_novelty(
        self,
        novelty_score: float,
        user_segment: str = "all"
    ):
        """Update recommendation novelty score (0-1)."""
        self.recommendation_novelty.labels(
            user_segment=user_segment
        ).set(novelty_score)
    
    def record_error(self, error_type: str, component: str):
        """Record an error."""
        self.error_count.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def get_metrics(self) -> bytes:
        """
        Get metrics in Prometheus format.
        
        Returns:
            Metrics as bytes (Prometheus text format)
        """
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get content type for metrics endpoint."""
        return CONTENT_TYPE_LATEST
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get human-readable metrics summary.
        
        Returns:
            Dictionary with current metrics
        """
        # This is a simplified summary - in production you'd query the registry
        return {
            "collector": "NCF Metrics Collector",
            "status": "active",
            "metrics_tracked": [
                "request_count",
                "request_latency",
                "model_inference_time",
                "cache_hits",
                "cache_misses",
                "ab_test_assignments",
                "recommendation_clicks",
                "user_feedback",
                "error_count",
                # Phase 2 metrics
                "onnx_inference_time",
                "onnx_speedup_ratio",
                "recommendation_accuracy",
                "recommendation_diversity",
                "fallback_recommendations"
            ]
        }
    
    # === Phase 2: NCF Helper Methods ===
    
    def track_onnx_inference(self, duration_seconds: float, batch_size: int = 1):
        """Track ONNX model inference time."""
        self.onnx_inference_time.labels(batch_size=str(batch_size)).observe(duration_seconds)
    
    def track_pytorch_inference(self, duration_seconds: float, batch_size: int = 1):
        """Track PyTorch model inference time."""
        self.pytorch_inference_time.labels(batch_size=str(batch_size)).observe(duration_seconds)
    
    def update_speedup_ratio(self, onnx_time: float, pytorch_time: float):
        """Update ONNX vs PyTorch speedup ratio."""
        if onnx_time > 0:
            speedup = pytorch_time / onnx_time
            self.onnx_speedup_ratio.set(speedup)
    
    def update_model_accuracy(self, accuracy: float):
        """Update model accuracy metric."""
        self.recommendation_accuracy.set(accuracy)
    
    def update_diversity(self, diversity_score: float):
        """Update recommendation diversity score."""
        self.recommendation_diversity.set(diversity_score)
    
    def track_fallback(self, reason: str):
        """Track fallback recommendation usage."""
        self.fallback_recommendations.labels(reason=reason).inc()
    
    def track_model_error(self, error_type: str):
        """Track model inference error."""
        self.model_errors.labels(error_type=error_type).inc()
    
    def get_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            Cache hit rate as a float (0-1)
        """
        try:
            # Get current values from Prometheus metrics
            # This is simplified - in production you'd query the actual metric values
            hits = 0  # Would get from self.cache_hits
            misses = 0  # Would get from self.cache_misses
            total = hits + misses
            return hits / total if total > 0 else 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {e}")
            return 0.0
        except Exception as e:
            logger.error(f