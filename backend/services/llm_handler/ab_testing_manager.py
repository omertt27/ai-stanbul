"""
A/B Testing Manager
Experiment management and analysis

Responsibilities:
- Experiment management
- Variant assignment
- Metric recording
- Analysis and reporting

Author: Istanbul AI Team
Date: November 14, 2025
"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ABTestingManager:
    """
    Manages A/B testing experiments
    
    Features:
    - Traffic splitting
    - Variant assignment
    - Metric tracking
    - Statistical analysis
    - Auto winner selection
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize A/B testing manager
        
        Args:
            redis_client: Redis client for persistence
        """
        self.redis = redis_client
        self.framework = None
        self.active_experiments = {}
        
        # Initialize A/B testing framework if available
        self._init_ab_testing_framework()
        
        # Statistics
        self.stats = {
            "experiments_created": 0,
            "variant_assignments": 0,
            "metrics_recorded": 0,
            "experiments_analyzed": 0,
            "errors": 0
        }
        
        logger.info("✅ A/B testing manager initialized")
        logger.info(f"   Framework: {'✅ Enabled' if self.framework else '❌ Disabled'}")
    
    def _init_ab_testing_framework(self):
        """Initialize A/B testing framework if available"""
        try:
            from backend.services.ab_testing import ABTestingFramework
            
            self.framework = ABTestingFramework(
                redis_client=self.redis
            )
            
            logger.info("   🧪 A/B testing framework loaded")
        except ImportError:
            logger.warning("   ⚠️ ABTestingFramework not available")
            self.framework = None
        except Exception as e:
            logger.error(f"   ❌ Failed to initialize A/B testing framework: {e}")
            self.framework = None
    
    def create_experiment(
        self,
        name: str,
        variants: Dict[str, Any],
        traffic_split: Optional[Dict[str, float]] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create new A/B test experiment.
        
        Args:
            name: Experiment name (must be unique)
            variants: Dict of variant_id -> variant_config
            traffic_split: Dict of variant_id -> traffic percentage (must sum to 1.0)
            description: Optional description of the experiment
            
        Returns:
            Dict with experiment details
        """
        if not self.framework:
            return {
                "status": "error",
                "message": "A/B testing framework not available"
            }
        
        try:
            # Validate traffic split
            if traffic_split:
                total = sum(traffic_split.values())
                if abs(total - 1.0) > 0.01:
                    return {
                        "status": "error",
                        "message": f"Traffic split must sum to 1.0 (got {total})"
                    }
            else:
                # Equal split by default
                split_value = 1.0 / len(variants)
                traffic_split = {vid: split_value for vid in variants.keys()}
            
            # Create experiment
            experiment_id = f"exp_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            experiment = {
                "id": experiment_id,
                "name": name,
                "description": description,
                "variants": variants,
                "traffic_split": traffic_split,
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "metrics": {}
            }
            
            self.active_experiments[experiment_id] = experiment
            self.stats["experiments_created"] += 1
            
            logger.info(f"✅ Created experiment: {name} (ID: {experiment_id})")
            logger.info(f"   Variants: {list(variants.keys())}")
            logger.info(f"   Traffic split: {traffic_split}")
            
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "experiment": experiment
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create experiment: {e}")
            self.stats["errors"] += 1
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_variant_for_user(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[str]:
        """
        Get variant assignment for a user.
        
        Uses consistent hashing to ensure same user always gets same variant.
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            
        Returns:
            Variant ID or None if experiment not found
        """
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        traffic_split = experiment["traffic_split"]
        
        # Use consistent hashing for stable assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        hash_normalized = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        # Assign variant based on traffic split
        cumulative = 0.0
        for variant_id, percentage in traffic_split.items():
            cumulative += percentage
            if hash_normalized < cumulative:
                self.stats["variant_assignments"] += 1
                return variant_id
        
        # Fallback to first variant
        return list(traffic_split.keys())[0]
    
    def get_threshold_for_experiment(
        self,
        signal_name: str,
        language: str,
        user_id: str,
        base_threshold: float
    ) -> float:
        """
        Get threshold considering active experiments.
        
        If user is in an experiment that tests thresholds for this signal,
        return the experimental threshold; otherwise return base threshold.
        
        Args:
            signal_name: Name of the signal
            language: Language code
            user_id: User identifier
            base_threshold: Base threshold value
            
        Returns:
            Threshold value (may be experimental or base)
        """
        # Check for threshold experiments
        for exp_id, experiment in self.active_experiments.items():
            if experiment.get("status") != "active":
                continue
            
            # Check if this experiment tests thresholds
            exp_type = experiment.get("type")
            if exp_type != "threshold":
                continue
            
            # Check if it's for this signal
            exp_signal = experiment.get("signal_name")
            if exp_signal != signal_name:
                continue
            
            # Get user's variant
            variant_id = self.get_variant_for_user(exp_id, user_id)
            if not variant_id:
                continue
            
            # Get variant threshold
            variants = experiment.get("variants", {})
            variant_config = variants.get(variant_id, {})
            threshold = variant_config.get("threshold")
            
            if threshold is not None:
                logger.debug(f"🧪 Using experimental threshold for {signal_name}: {threshold} (variant: {variant_id})")
                return threshold
        
        # No active experiment for this signal
        return base_threshold
    
    def record_metric(
        self,
        experiment_id: str,
        variant_id: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record metric for experiment variant.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_name: Name of the metric (e.g., 'accuracy', 'latency')
            value: Metric value
            metadata: Optional additional data
        """
        if experiment_id not in self.active_experiments:
            logger.warning(f"⚠️ Experiment not found: {experiment_id}")
            return
        
        try:
            experiment = self.active_experiments[experiment_id]
            
            # Initialize metrics structure if needed
            if "metrics" not in experiment:
                experiment["metrics"] = {}
            
            if variant_id not in experiment["metrics"]:
                experiment["metrics"][variant_id] = {}
            
            if metric_name not in experiment["metrics"][variant_id]:
                experiment["metrics"][variant_id][metric_name] = []
            
            # Record metric
            metric_record = {
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
            
            if metadata:
                metric_record["metadata"] = metadata
            
            experiment["metrics"][variant_id][metric_name].append(metric_record)
            self.stats["metrics_recorded"] += 1
            
            logger.debug(f"📊 Recorded metric: {metric_name}={value} for variant {variant_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record metric: {e}")
            self.stats["errors"] += 1
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze experiment results.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dict with analysis results
        """
        if experiment_id not in self.active_experiments:
            return {
                "status": "error",
                "message": "Experiment not found"
            }
        
        try:
            experiment = self.active_experiments[experiment_id]
            metrics = experiment.get("metrics", {})
            
            # Compute statistics for each variant
            variant_stats = {}
            
            for variant_id, variant_metrics in metrics.items():
                stats = {}
                
                for metric_name, values in variant_metrics.items():
                    numeric_values = [v["value"] for v in values]
                    
                    if numeric_values:
                        stats[metric_name] = {
                            "count": len(numeric_values),
                            "mean": sum(numeric_values) / len(numeric_values),
                            "min": min(numeric_values),
                            "max": max(numeric_values)
                        }
                
                variant_stats[variant_id] = stats
            
            self.stats["experiments_analyzed"] += 1
            
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "experiment_name": experiment["name"],
                "variant_stats": variant_stats,
                "analyzed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze experiment: {e}")
            self.stats["errors"] += 1
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """
        Get list of active experiments.
        
        Returns:
            List of active experiment dicts
        """
        return [
            exp for exp in self.active_experiments.values()
            if exp.get("status") == "active"
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get A/B testing manager statistics.
        
        Returns:
            Dict with performance metrics
        """
        return {
            **self.stats,
            "active_experiments": len([
                e for e in self.active_experiments.values()
                if e.get("status") == "active"
            ]),
            "framework_available": self.framework is not None
        }
        return {}
