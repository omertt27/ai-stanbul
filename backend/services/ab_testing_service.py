"""
A/B Testing Service for NCF Recommendations

Production A/B testing framework with:
- Consistent user assignment (hash-based)
- Multiple concurrent tests
- Real-time metrics tracking
- Statistical significance testing
- Test result analysis

Author: AI Istanbul Team
Date: February 12, 2026
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import redis

logger = logging.getLogger(__name__)

# Global A/B testing service instance
_ab_testing_service = None


class ABTestConfig:
    """Configuration for an A/B test."""
    
    def __init__(
        self,
        test_id: str,
        name: str,
        variants: Dict[str, float],  # variant_name -> traffic_percentage
        start_date: datetime,
        end_date: Optional[datetime] = None,
        description: str = "",
        metadata: Optional[Dict] = None
    ):
        self.test_id = test_id
        self.name = name
        self.variants = variants
        self.start_date = start_date
        self.end_date = end_date
        self.description = description
        self.metadata = metadata or {}
        
        # Validate variants sum to 100%
        total = sum(variants.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Variant percentages must sum to 100%, got {total * 100}%")


class ABTestingService:
    """
    Production A/B testing service.
    
    Features:
    - Consistent user assignment (deterministic hashing)
    - Multiple concurrent tests
    - Real-time metrics tracking
    - Conversion tracking
    - Statistical analysis
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        enable_redis: bool = True
    ):
        """
        Initialize A/B testing service.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            enable_redis: Enable Redis for persistence
        """
        self.enable_redis = enable_redis
        self.redis_client = None
        
        # In-memory test registry
        self.active_tests: Dict[str, ABTestConfig] = {}
        
        # Metrics storage (Redis or in-memory)
        self.metrics = defaultdict(lambda: defaultdict(lambda: {
            "impressions": 0,
            "conversions": 0,
            "sum_metric_values": defaultdict(float),
            "count_metric_values": defaultdict(int)
        }))
        
        # Initialize Redis
        if enable_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info(f"✅ A/B Testing Service connected to Redis: {redis_host}:{redis_port}")
                
                # Load active tests from Redis
                self._load_tests_from_redis()
                
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to Redis for A/B testing: {e}")
                logger.warning("📝 Using in-memory A/B testing (tests will not persist)")
                self.redis_client = None
        
        logger.info("✅ A/B Testing Service initialized")
    
    def create_test(
        self,
        test_id: str,
        name: str,
        variants: Dict[str, float],
        duration_days: int = 14,
        description: str = "",
        metadata: Optional[Dict] = None
    ) -> ABTestConfig:
        """
        Create and activate a new A/B test.
        
        Args:
            test_id: Unique test identifier
            name: Human-readable test name
            variants: Dict of variant_name -> traffic_percentage (must sum to 1.0)
            duration_days: Test duration in days
            description: Test description
            metadata: Additional test metadata
            
        Returns:
            ABTestConfig object
            
        Example:
            >>> ab_service.create_test(
            ...     test_id="ncf_vs_popular",
            ...     name="NCF vs Popular Items",
            ...     variants={"control": 0.5, "treatment": 0.5},
            ...     duration_days=14
            ... )
        """
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=duration_days)
        
        config = ABTestConfig(
            test_id=test_id,
            name=name,
            variants=variants,
            start_date=start_date,
            end_date=end_date,
            description=description,
            metadata=metadata
        )
        
        self.active_tests[test_id] = config
        
        # Persist to Redis
        if self.redis_client:
            self._save_test_to_redis(config)
        
        logger.info(f"✅ Created A/B test: {test_id} ({name})")
        logger.info(f"   Variants: {variants}")
        logger.info(f"   Duration: {duration_days} days (ends {end_date.date()})")
        
        return config
    
    def assign_user(
        self,
        user_id: int,
        test_id: str
    ) -> Dict[str, Any]:
        """
        Assign user to A/B test variant using consistent hashing.
        
        Args:
            user_id: User identifier
            test_id: Test identifier
            
        Returns:
            Dict with test_id, variant, and assignment metadata
        """
        if test_id not in self.active_tests:
            logger.warning(f"Test {test_id} not found")
            return {"test_id": None, "variant": None, "error": "test_not_found"}
        
        config = self.active_tests[test_id]
        
        # Check if test is active
        now = datetime.utcnow()
        if now < config.start_date or (config.end_date and now > config.end_date):
            logger.warning(f"Test {test_id} is not currently active")
            return {"test_id": test_id, "variant": None, "error": "test_not_active"}
        
        # Deterministic assignment using hash
        variant = self._hash_assignment(user_id, test_id, config.variants)
        
        # Track impression
        self._track_impression(test_id, variant)
        
        return {
            "test_id": test_id,
            "variant": variant,
            "timestamp": now.isoformat()
        }
    
    def track_conversion(
        self,
        test_id: str,
        variant: str,
        value: float = 1.0
    ):
        """
        Track a conversion event for a variant.
        
        Args:
            test_id: Test identifier
            variant: Variant name
            value: Conversion value (default 1.0)
        """
        if test_id not in self.active_tests:
            logger.warning(f"Cannot track conversion for unknown test: {test_id}")
            return
        
        # Update in-memory metrics
        self.metrics[test_id][variant]["conversions"] += value
        
        # Update Redis
        if self.redis_client:
            key = f"ab:metrics:{test_id}:{variant}"
            self.redis_client.hincrby(key, "conversions", int(value))
        
        logger.debug(f"Tracked conversion: {test_id}/{variant} = {value}")
    
    def track_metric(
        self,
        test_id: str,
        variant: str,
        metric_name: str,
        value: float
    ):
        """
        Track a custom metric for a variant.
        
        Args:
            test_id: Test identifier
            variant: Variant name
            metric_name: Metric name (e.g., "latency_ms", "revenue")
            value: Metric value
        """
        if test_id not in self.active_tests:
            return
        
        # Update in-memory metrics
        metrics = self.metrics[test_id][variant]
        metrics["sum_metric_values"][metric_name] += value
        metrics["count_metric_values"][metric_name] += 1
        
        # Update Redis
        if self.redis_client:
            key = f"ab:metrics:{test_id}:{variant}:{metric_name}"
            self.redis_client.hincrbyfloat(key, "sum", value)
            self.redis_client.hincrby(key, "count", 1)
        
        logger.debug(f"Tracked metric: {test_id}/{variant}/{metric_name} = {value}")
    
    def get_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get A/B test results with statistical analysis.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Dict with test results and statistical analysis
        """
        if test_id not in self.active_tests:
            return {"error": "test_not_found"}
        
        config = self.active_tests[test_id]
        
        # Collect metrics for all variants
        results = {
            "test_id": test_id,
            "name": config.name,
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat() if config.end_date else None,
            "variants": {}
        }
        
        for variant in config.variants.keys():
            metrics = self._get_variant_metrics(test_id, variant)
            
            # Calculate conversion rate
            impressions = metrics["impressions"]
            conversions = metrics["conversions"]
            conversion_rate = conversions / impressions if impressions > 0 else 0
            
            # Calculate average metrics
            avg_metrics = {}
            for metric_name, total in metrics["sum_metric_values"].items():
                count = metrics["count_metric_values"][metric_name]
                avg_metrics[metric_name] = total / count if count > 0 else 0
            
            results["variants"][variant] = {
                "impressions": impressions,
                "conversions": conversions,
                "conversion_rate": conversion_rate,
                "avg_metrics": avg_metrics
            }
        
        # Calculate statistical significance (simplified)
        results["analysis"] = self._calculate_significance(results["variants"])
        
        return results
    
    def stop_test(self, test_id: str) -> Dict[str, Any]:
        """
        Stop an active A/B test and get final results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Final test results
        """
        if test_id not in self.active_tests:
            return {"error": "test_not_found"}
        
        # Get final results
        results = self.get_results(test_id)
        
        # Mark as stopped
        config = self.active_tests[test_id]
        config.end_date = datetime.utcnow()
        
        # Update Redis
        if self.redis_client:
            self._save_test_to_redis(config)
        
        logger.info(f"🛑 Stopped A/B test: {test_id}")
        
        return results
    
    def has_active_tests(self) -> bool:
        """Check if there are any active tests."""
        now = datetime.utcnow()
        for config in self.active_tests.values():
            if now >= config.start_date and (not config.end_date or now <= config.end_date):
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get A/B testing service statistics."""
        active_count = sum(
            1 for config in self.active_tests.values()
            if datetime.utcnow() >= config.start_date and 
            (not config.end_date or datetime.utcnow() <= config.end_date)
        )
        
        return {
            "total_tests": len(self.active_tests),
            "active_tests": active_count,
            "redis_enabled": self.redis_client is not None,
            "test_ids": list(self.active_tests.keys())
        }
    
    # ==================== Private Methods ====================
    
    def _hash_assignment(
        self,
        user_id: int,
        test_id: str,
        variants: Dict[str, float]
    ) -> str:
        """
        Deterministically assign user to variant using consistent hashing.
        
        Args:
            user_id: User identifier
            test_id: Test identifier
            variants: Dict of variant -> traffic_percentage
            
        Returns:
            Assigned variant name
        """
        # Create deterministic hash
        hash_input = f"{user_id}:{test_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Normalize to 0-1
        normalized = (hash_value % 10000) / 10000.0
        
        # Assign to variant based on traffic split
        cumulative = 0.0
        for variant, percentage in variants.items():
            cumulative += percentage
            if normalized < cumulative:
                return variant
        
        # Fallback (should never happen)
        return list(variants.keys())[0]
    
    def _track_impression(self, test_id: str, variant: str):
        """Track an impression for a variant."""
        # Update in-memory
        self.metrics[test_id][variant]["impressions"] += 1
        
        # Update Redis
        if self.redis_client:
            key = f"ab:metrics:{test_id}:{variant}"
            self.redis_client.hincrby(key, "impressions", 1)
    
    def _get_variant_metrics(self, test_id: str, variant: str) -> Dict:
        """Get metrics for a variant (from Redis or in-memory)."""
        if self.redis_client:
            key = f"ab:metrics:{test_id}:{variant}"
            data = self.redis_client.hgetall(key)
            
            return {
                "impressions": int(data.get("impressions", 0)),
                "conversions": int(data.get("conversions", 0)),
                "sum_metric_values": defaultdict(float),
                "count_metric_values": defaultdict(int)
            }
        else:
            return self.metrics[test_id][variant]
    
    def _calculate_significance(self, variants: Dict) -> Dict:
        """
        Calculate statistical significance (simplified).
        
        For production, use scipy.stats or similar for proper statistical tests.
        """
        # Find control and treatment
        if "control" not in variants or "treatment" not in variants:
            return {"error": "Requires 'control' and 'treatment' variants"}
        
        control = variants["control"]
        treatment = variants["treatment"]
        
        control_rate = control["conversion_rate"]
        treatment_rate = treatment["conversion_rate"]
        
        # Calculate lift
        lift = ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0
        
        # Simplified significance test (sample size heuristic)
        min_sample_size = 100
        is_significant = (
            control["impressions"] >= min_sample_size and
            treatment["impressions"] >= min_sample_size and
            abs(lift) > 5  # At least 5% lift
        )
        
        return {
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "lift_percent": lift,
            "is_significant": is_significant,
            "note": "Simplified significance test. Use proper statistical testing for production."
        }
    
    def _save_test_to_redis(self, config: ABTestConfig):
        """Save test configuration to Redis."""
        if not self.redis_client:
            return
        
        key = f"ab:test:{config.test_id}"
        data = {
            "name": config.name,
            "variants": json.dumps(config.variants),
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat() if config.end_date else "",
            "description": config.description,
            "metadata": json.dumps(config.metadata)
        }
        
        self.redis_client.hset(key, mapping=data)
    
    def _load_tests_from_redis(self):
        """Load active tests from Redis."""
        if not self.redis_client:
            return
        
        # Scan for test keys
        for key in self.redis_client.scan_iter("ab:test:*"):
            try:
                data = self.redis_client.hgetall(key)
                test_id = key.split(":")[-1]
                
                config = ABTestConfig(
                    test_id=test_id,
                    name=data["name"],
                    variants=json.loads(data["variants"]),
                    start_date=datetime.fromisoformat(data["start_date"]),
                    end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
                    description=data.get("description", ""),
                    metadata=json.loads(data.get("metadata", "{}"))
                )
                
                self.active_tests[test_id] = config
                logger.info(f"📊 Loaded A/B test from Redis: {test_id}")
                
            except Exception as e:
                logger.error(f"Error loading test {key}: {e}")


def get_ab_testing_service() -> ABTestingService:
    """Get or create global A/B testing service instance."""
    global _ab_testing_service
    
    if _ab_testing_service is None:
        # Get Redis config from environment
        import os
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        _ab_testing_service = ABTestingService(
            redis_host=redis_host,
            redis_port=redis_port,
            enable_redis=True
        )
    
    return _ab_testing_service


# ==================== Convenience Functions ====================

def create_ncf_test(duration_days: int = 14) -> ABTestConfig:
    """
    Create a standard NCF vs Popular test.
    
    Args:
        duration_days: Test duration in days
        
    Returns:
        ABTestConfig
    """
    service = get_ab_testing_service()
    
    return service.create_test(
        test_id="ncf_vs_popular",
        name="NCF Personalization vs Popular Items",
        variants={"control": 0.5, "treatment": 0.5},
        duration_days=duration_days,
        description="Test impact of NCF personalization vs showing popular items",
        metadata={
            "control_algorithm": "popular",
            "treatment_algorithm": "ncf",
            "primary_metric": "conversion_rate"
        }
    )


logger.info("✅ A/B Testing Service loaded")
