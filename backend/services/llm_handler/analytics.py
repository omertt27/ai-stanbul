"""
Analytics Manager
Performance metrics, monitoring, and error tracking

Responsibilities:
- Track performance metrics (latency, throughput)
- Monitor errors and failures
- Collect user analytics
- Track signal detection accuracy
- Monitor service usage
- Quality metrics

Extracted from: pure_llm_handler.py (_init_advanced_analytics, _track_performance, _track_error, get_analytics_summary)

Author: Istanbul AI Team
Date: November 14, 2025
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AnalyticsManager:
    """
    Manages all analytics, monitoring, and performance tracking
    
    Tracks:
    - Performance metrics (latency, throughput)
    - Error tracking and patterns
    - User behavior analytics
    - Signal detection accuracy
    - Service usage patterns
    - Quality metrics
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize analytics manager
        
        Args:
            redis_client: Redis client for persistence (optional)
        """
        self.redis = redis_client
        self._init_metrics()
        
        logger.info("📊 Analytics manager initialized")
    
    def _init_metrics(self):
        """
        Initialize all metric collectors
        
        Extracted from pure_llm_handler._init_advanced_analytics()
        """
        # Performance metrics
        self.performance_metrics = {
            "query_latencies": deque(maxlen=1000),  # Last 1000 query times
            "llm_latencies": deque(maxlen=1000),
            "cache_latencies": deque(maxlen=1000),
            "service_latencies": defaultdict(lambda: deque(maxlen=100)),
        }
        
        # Error tracking
        self.error_tracker = {
            "total_errors": 0,
            "error_by_type": defaultdict(int),
            "error_by_service": defaultdict(int),
            "recent_errors": deque(maxlen=50),
            "error_recovery_count": 0
        }
        
        # User analytics
        self.user_analytics = {
            "queries_by_language": defaultdict(int),
            "queries_by_intent": defaultdict(int),
            "multi_intent_patterns": defaultdict(int),
            "user_locations_used": 0,
            "unique_users": set(),
            "queries_per_user": defaultdict(int)
        }
        
        # Signal detection analytics
        self.signal_analytics = {
            "detections_by_signal": defaultdict(int),
            "detection_confidence_scores": defaultdict(list),
            "false_positive_reports": defaultdict(int),
            "semantic_vs_keyword": {"semantic": 0, "keyword": 0, "both": 0},
            "language_specific_accuracy": defaultdict(lambda: {"correct": 0, "incorrect": 0})
        }
        
        # Service usage analytics
        self.service_analytics = {
            "map_generation_success": 0,
            "map_generation_failure": 0,
            "weather_service_calls": 0,
            "events_service_calls": 0,
            "hidden_gems_calls": 0,
            "rag_usage": 0,
            "cache_efficiency": {"hits": 0, "misses": 0}
        }
        
        # Quality metrics
        self.quality_metrics = {
            "responses_validated": 0,
            "validation_failures": 0,
            "response_lengths": deque(maxlen=100),
            "empty_responses": 0,
            "context_usage_rate": deque(maxlen=100)
        }
    
    def track_performance(self, metric_name: str, latency: float):
        """
        Track performance metric
        
        Extracted from pure_llm_handler._track_performance()
        
        Args:
            metric_name: Name of the metric (e.g., 'query', 'llm', 'cache')
            latency: Time taken in seconds
        """
        if metric_name == "query":
            self.performance_metrics["query_latencies"].append(latency)
        elif metric_name == "llm":
            self.performance_metrics["llm_latencies"].append(latency)
        elif metric_name == "cache":
            self.performance_metrics["cache_latencies"].append(latency)
        else:
            self.performance_metrics["service_latencies"][metric_name].append(latency)
    
    def track_error(
        self,
        error_type: str,
        service: str,
        error_message: str,
        query: Optional[str] = None
    ):
        """
        Track error for monitoring and debugging
        
        Extracted from pure_llm_handler._track_error()
        
        Args:
            error_type: Type of error (e.g., 'llm_failure', 'service_timeout')
            service: Service that failed (e.g., 'runpod', 'weather')
            error_message: Error description
            query: Optional user query for context
        """
        self.error_tracker["total_errors"] += 1
        self.error_tracker["error_by_type"][error_type] += 1
        self.error_tracker["error_by_service"][service] += 1
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "service": service,
            "message": error_message[:200],  # Truncate long messages
            "query": query[:100] if query else None
        }
        
        self.error_tracker["recent_errors"].append(error_entry)
        
        logger.error(f"   ❌ Error tracked: {error_type} in {service} - {error_message[:100]}")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics summary
        
        Extracted from pure_llm_handler.get_analytics_summary()
        
        Returns:
            Dictionary with all analytics data for monitoring/dashboards
        """
        # Calculate performance statistics
        def calc_stats(latencies):
            if not latencies:
                return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            return {
                "count": n,
                "avg": sum(sorted_lat) / n,
                "p50": sorted_lat[int(n * 0.5)],
                "p95": sorted_lat[int(n * 0.95)] if n > 20 else sorted_lat[-1],
                "p99": sorted_lat[int(n * 0.99)] if n > 100 else sorted_lat[-1]
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "query_latency": calc_stats(self.performance_metrics["query_latencies"]),
                "llm_latency": calc_stats(self.performance_metrics["llm_latencies"]),
                "cache_latency": calc_stats(self.performance_metrics["cache_latencies"])
            },
            "errors": {
                "total": self.error_tracker["total_errors"],
                "by_type": dict(self.error_tracker["error_by_type"]),
                "by_service": dict(self.error_tracker["error_by_service"]),
                "recovery_count": self.error_tracker["error_recovery_count"],
                "recent": list(self.error_tracker["recent_errors"])[-10:]  # Last 10
            },
            "users": {
                "queries_by_language": dict(self.user_analytics["queries_by_language"]),
                "queries_by_intent": dict(self.user_analytics["queries_by_intent"]),
                "unique_users": len(self.user_analytics["unique_users"]),
                "multi_intent_patterns": dict(self.user_analytics["multi_intent_patterns"]),
                "user_locations_used": self.user_analytics["user_locations_used"]
            },
            "signals": {
                "detections": dict(self.signal_analytics["detections_by_signal"]),
                "semantic_vs_keyword": self.signal_analytics["semantic_vs_keyword"],
                "false_positives": dict(self.signal_analytics["false_positive_reports"])
            },
            "services": {
                "map_success_rate": (
                    self.service_analytics["map_generation_success"] / 
                    max(1, self.service_analytics["map_generation_success"] + 
                        self.service_analytics["map_generation_failure"])
                ),
                "cache_hit_rate": (
                    self.service_analytics["cache_efficiency"]["hits"] /
                    max(1, self.service_analytics["cache_efficiency"]["hits"] + 
                        self.service_analytics["cache_efficiency"]["misses"])
                ),
                "service_usage": {
                    "weather": self.service_analytics["weather_service_calls"],
                    "events": self.service_analytics["events_service_calls"],
                    "hidden_gems": self.service_analytics["hidden_gems_calls"],
                    "rag": self.service_analytics["rag_usage"]
                }
            },
            "quality": {
                "responses_validated": self.quality_metrics["responses_validated"],
                "validation_failures": self.quality_metrics["validation_failures"],
                "validation_success_rate": (
                    (self.quality_metrics["responses_validated"] - 
                     self.quality_metrics["validation_failures"]) /
                    max(1, self.quality_metrics["responses_validated"])
                ),
                "empty_responses": self.quality_metrics["empty_responses"],
                "avg_response_length": (
                    sum(self.quality_metrics["response_lengths"]) / 
                    len(self.quality_metrics["response_lengths"])
                    if self.quality_metrics["response_lengths"] else 0
                )
            }
        }
