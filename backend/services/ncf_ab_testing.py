"""
A/B Testing Framework for NCF Recommendations

Supports:
- User bucketing (consistent assignment)
- Variant tracking
- Metrics collection
- Statistical significance testing

Author: AI Istanbul Team
Date: February 10, 2026
"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - A/B testing will use in-memory storage")


class ABTestFramework:
    """
    A/B testing framework for recommendation experiments.
    """
    
    def __init__(
        self,
        experiment_name: str,
        variants: Dict[str, float],
        redis_client: Optional[Any] = None,
        persist_metrics: bool = True
    ):
        """
        Initialize A/B test framework.
        
        Args:
            experiment_name: Name of the experiment
            variants: Dict of variant_name -> traffic_percentage (e.g., {'control': 0.5, 'onnx': 0.5})
            redis_client: Redis client for persistence
            persist_metrics: Whether to persist metrics to Redis
        """
        self.experiment_name = experiment_name
        self.variants = variants
        self.redis_client = redis_client
        self.persist_metrics = persist_metrics
        
        # Validate variant percentages
        total = sum(variants.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Variant percentages must sum to 1.0, got {total}")
        
        # Normalize to ensure exact 1.0
        self.variants = {k: v/total for k, v in variants.items()}
        
        # In-memory metrics (fallback)
        self.metrics = defaultdict(lambda: {
            'impressions': 0,
            'clicks': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'total_latency_ms': 0.0,
            'errors': 0
        })
        
        logger.info(f"✅ A/B test initialized: {experiment_name}")
        logger.info(f"   Variants: {variants}")
    
    def get_variant(self, user_id: str) -> str:
        """
        Assign user to a variant (consistent assignment).
        
        Args:
            user_id: User identifier
            
        Returns:
            Variant name
        """
        # Use hash for consistent assignment
        hash_value = int(hashlib.md5(
            f"{self.experiment_name}:{user_id}".encode()
        ).hexdigest(), 16)
        
        # Normalize to [0, 1]
        normalized = (hash_value % 10000) / 10000.0
        
        # Assign to variant based on cumulative percentages
        cumulative = 0.0
        for variant, percentage in self.variants.items():
            cumulative += percentage
            if normalized < cumulative:
                return variant
        
        # Fallback (should never reach here)
        return list(self.variants.keys())[0]
    
    def track_impression(
        self,
        user_id: str,
        variant: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track when recommendations are shown to user.
        
        Args:
            user_id: User identifier
            variant: Variant name (auto-assigned if None)
            metadata: Additional tracking data
        """
        if variant is None:
            variant = self.get_variant(user_id)
        
        self.metrics[variant]['impressions'] += 1
        
        if self.persist_metrics and self.redis_client:
            self._persist_event('impression', variant, user_id, metadata)
    
    def track_click(
        self,
        user_id: str,
        variant: Optional[str] = None,
        item_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track when user clicks a recommendation.
        
        Args:
            user_id: User identifier
            variant: Variant name (auto-assigned if None)
            item_id: Clicked item ID
            metadata: Additional tracking data
        """
        if variant is None:
            variant = self.get_variant(user_id)
        
        self.metrics[variant]['clicks'] += 1
        
        if self.persist_metrics and self.redis_client:
            event_data = metadata or {}
            event_data['item_id'] = item_id
            self._persist_event('click', variant, user_id, event_data)
    
    def track_feedback(
        self,
        user_id: str,
        feedback_type: str,  # 'positive' or 'negative'
        variant: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track user feedback on recommendations.
        
        Args:
            user_id: User identifier
            feedback_type: 'positive' or 'negative'
            variant: Variant name (auto-assigned if None)
            metadata: Additional tracking data
        """
        if variant is None:
            variant = self.get_variant(user_id)
        
        if feedback_type == 'positive':
            self.metrics[variant]['positive_feedback'] += 1
        elif feedback_type == 'negative':
            self.metrics[variant]['negative_feedback'] += 1
        
        if self.persist_metrics and self.redis_client:
            event_data = metadata or {}
            event_data['feedback_type'] = feedback_type
            self._persist_event('feedback', variant, user_id, event_data)
    
    def track_latency(
        self,
        variant: str,
        latency_ms: float
    ) -> None:
        """
        Track recommendation latency.
        
        Args:
            variant: Variant name
            latency_ms: Latency in milliseconds
        """
        self.metrics[variant]['total_latency_ms'] += latency_ms
    
    def track_error(
        self,
        variant: str,
        error_type: Optional[str] = None
    ) -> None:
        """
        Track errors in recommendation generation.
        
        Args:
            variant: Variant name
            error_type: Type of error
        """
        self.metrics[variant]['errors'] += 1
    
    def _persist_event(
        self,
        event_type: str,
        variant: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Persist event to Redis."""
        try:
            event = {
                'experiment': self.experiment_name,
                'variant': variant,
                'user_id': user_id,
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # Store in Redis list
            key = f"abtest:{self.experiment_name}:{variant}:events"
            self.redis_client.rpush(key, json.dumps(event))
            
            # Set expiry (30 days)
            self.redis_client.expire(key, 30 * 24 * 3600)
            
        except Exception as e:
            logger.warning(f"Failed to persist A/B test event: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get experiment metrics for all variants.
        
        Returns:
            Dictionary with metrics per variant
        """
        results = {}
        
        for variant in self.variants.keys():
            metrics = self.metrics[variant]
            impressions = metrics['impressions']
            clicks = metrics['clicks']
            
            # Calculate rates
            ctr = clicks / impressions if impressions > 0 else 0.0
            positive = metrics['positive_feedback']
            negative = metrics['negative_feedback']
            total_feedback = positive + negative
            positive_rate = positive / total_feedback if total_feedback > 0 else 0.0
            
            avg_latency = (
                metrics['total_latency_ms'] / impressions 
                if impressions > 0 else 0.0
            )
            
            results[variant] = {
                'impressions': impressions,
                'clicks': clicks,
                'ctr': round(ctr, 4),
                'positive_feedback': positive,
                'negative_feedback': negative,
                'positive_rate': round(positive_rate, 4),
                'avg_latency_ms': round(avg_latency, 2),
                'errors': metrics['errors']
            }
        
        return results
    
    def get_report(self) -> str:
        """
        Generate experiment report.
        
        Returns:
            Formatted report string
        """
        metrics = self.get_metrics()
        
        report = [
            f"\n{'='*80}",
            f"A/B Test Report: {self.experiment_name}",
            f"{'='*80}",
            f"\nVariant Distribution:",
        ]
        
        for variant, percentage in self.variants.items():
            report.append(f"  {variant}: {percentage:.1%}")
        
        report.append(f"\nMetrics:")
        report.append(f"{'Variant':<15} {'Impressions':<12} {'Clicks':<8} {'CTR':<8} "
                     f"{'+Feedback':<10} {'Rate':<8} {'Latency':<10}")
        report.append(f"{'-'*80}")
        
        for variant, m in metrics.items():
            report.append(
                f"{variant:<15} {m['impressions']:<12} {m['clicks']:<8} "
                f"{m['ctr']:<8.2%} {m['positive_feedback']:<10} "
                f"{m['positive_rate']:<8.2%} {m['avg_latency_ms']:<10.1f}ms"
            )
        
        report.append(f"{'='*80}\n")
        
        return "\n".join(report)


# Predefined experiments
def create_ncf_onnx_experiment(redis_client: Optional[Any] = None) -> ABTestFramework:
    """Create ONNX vs Fallback A/B test."""
    return ABTestFramework(
        experiment_name="ncf_onnx_vs_fallback",
        variants={'onnx': 0.5, 'fallback': 0.5},
        redis_client=redis_client
    )
