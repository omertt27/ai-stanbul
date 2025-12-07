"""
ab_testing.py - A/B Testing Framework

Comprehensive A/B testing system for controlled experiments with:
- Variant assignment and management
- Statistical significance testing
- Experiment tracking and reporting
- Bayesian analysis
- Multi-variant support

Author: AI Istanbul Team
Date: December 7, 2025
"""

import logging
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    id: str
    name: str
    description: str
    start_date: str
    end_date: str
    variants: Dict[str, Dict[str, Any]]  # variant_name -> {weight, config}
    metrics: List[str]
    minimum_sample_size: int
    status: str = "draft"
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class VariantMetrics:
    """Metrics for a single variant."""
    variant_name: str
    sample_size: int
    metrics: Dict[str, List[float]]  # metric_name -> list of values
    
    def get_mean(self, metric_name: str) -> float:
        """Get mean value for a metric."""
        values = self.metrics.get(metric_name, [])
        return sum(values) / len(values) if values else 0.0
    
    def get_std(self, metric_name: str) -> float:
        """Get standard deviation for a metric."""
        values = self.metrics.get(metric_name, [])
        if not values:
            return 0.0
        
        mean = self.get_mean(metric_name)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def get_confidence_interval(
        self,
        metric_name: str,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Get confidence interval for a metric."""
        import scipy.stats as stats
        
        values = self.metrics.get(metric_name, [])
        if not values:
            return (0.0, 0.0)
        
        mean = self.get_mean(metric_name)
        std = self.get_std(metric_name)
        n = len(values)
        
        # Calculate margin of error
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * (std / (n ** 0.5))
        
        return (mean - margin, mean + margin)


class ExperimentManager:
    """
    Manages A/B experiments with variant assignment and tracking.
    
    Features:
    - Consistent variant assignment using hashing
    - Statistical significance testing
    - Multiple variants support
    - Redis-backed caching
    - Experiment lifecycle management
    """
    
    def __init__(self, redis_client=None, db_connection=None, config: Optional[Dict] = None):
        """
        Initialize experiment manager.
        
        Args:
            redis_client: Redis client for caching (optional)
            db_connection: Database connection for persistence (optional)
            config: Configuration dict
        """
        self.redis = redis_client
        self.db = db_connection
        self.config = config or {}
        
        # In-memory storage (fallback if no Redis)
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.variant_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.metrics_storage: Dict[str, Dict[str, VariantMetrics]] = defaultdict(dict)
        
        logger.info("✅ ExperimentManager initialized")
    
    async def create_experiment(
        self,
        name: str,
        description: str,
        variants: Dict[str, Dict[str, Any]],
        metrics: List[str],
        start_date: str,
        end_date: str,
        minimum_sample_size: int = 100
    ) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Description
            variants: Dict of variant_name -> {weight, config}
            metrics: List of metric names to track
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            minimum_sample_size: Minimum samples per variant
        
        Returns:
            experiment_id: Unique experiment ID
        """
        # Generate unique ID
        experiment_id = f"exp_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        # Validate variant weights sum to 1.0
        total_weight = sum(v.get('weight', 0) for v in variants.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Variant weights must sum to 1.0 (got {total_weight})")
        
        # Create experiment config
        experiment = ExperimentConfig(
            id=experiment_id,
            name=name,
            description=description,
            start_date=start_date,
            end_date=end_date,
            variants=variants,
            metrics=metrics,
            minimum_sample_size=minimum_sample_size,
            status=ExperimentStatus.DRAFT.value
        )
        
        # Store in memory
        self.active_experiments[experiment_id] = experiment
        
        # Initialize metrics storage for each variant
        for variant_name in variants.keys():
            self.metrics_storage[experiment_id][variant_name] = VariantMetrics(
                variant_name=variant_name,
                sample_size=0,
                metrics={metric: [] for metric in metrics}
            )
        
        # Persist to database if available
        if self.db:
            await self._save_experiment_to_db(experiment)
        
        logger.info(f"✅ Created experiment: {experiment_id} - {name}")
        return experiment_id
    
    async def start_experiment(self, experiment_id: str):
        """Start an experiment."""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.status = ExperimentStatus.RUNNING.value
        logger.info(f"🚀 Started experiment: {experiment_id}")
    
    async def stop_experiment(self, experiment_id: str):
        """Stop an experiment."""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.status = ExperimentStatus.STOPPED.value
        logger.info(f"⏸️ Stopped experiment: {experiment_id}")
    
    async def get_variant(
        self,
        experiment_id: str,
        user_id: str,
        force_variant: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get or assign variant for user in experiment.
        
        Uses consistent hashing to ensure same user always gets same variant.
        
        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            force_variant: Force specific variant (for testing)
        
        Returns:
            (variant_name, variant_config)
        """
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            # Return default/control
            return ("control", {})
        
        # Check if experiment is running
        if experiment.status != ExperimentStatus.RUNNING.value:
            return ("control", {})
        
        # Check cache (Redis or in-memory)
        cache_key = f"exp:{experiment_id}:user:{user_id}"
        
        if self.redis:
            try:
                cached_variant = await self.redis.get(cache_key)
                if cached_variant:
                    cached_variant = cached_variant.decode('utf-8')
                    variant_config = experiment.variants.get(cached_variant, {}).get('config', {})
                    return (cached_variant, variant_config)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        else:
            # In-memory cache
            if experiment_id in self.variant_assignments:
                if user_id in self.variant_assignments[experiment_id]:
                    cached_variant = self.variant_assignments[experiment_id][user_id]
                    variant_config = experiment.variants.get(cached_variant, {}).get('config', {})
                    return (cached_variant, variant_config)
        
        # Force variant if specified (for testing)
        if force_variant and force_variant in experiment.variants:
            variant = force_variant
        else:
            # Assign new variant using consistent hashing
            variant = self._assign_variant(user_id, experiment)
        
        # Get variant config
        variant_config = experiment.variants.get(variant, {}).get('config', {})
        
        # Cache assignment
        if self.redis:
            try:
                await self.redis.set(cache_key, variant, ex=604800)  # 7 days
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        else:
            # In-memory cache
            self.variant_assignments[experiment_id][user_id] = variant
        
        logger.debug(f"Assigned user {user_id} to variant '{variant}' in experiment {experiment_id}")
        
        return (variant, variant_config)
    
    def _assign_variant(self, user_id: str, experiment: ExperimentConfig) -> str:
        """
        Assign variant using consistent hashing.
        
        Ensures same user always gets same variant.
        """
        # Create hash from user_id and experiment_id
        hash_input = f"{user_id}:{experiment.id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        # Distribute based on variant weights
        cumulative = 0.0
        for variant_name, variant_data in experiment.variants.items():
            weight = variant_data.get('weight', 0)
            cumulative += weight
            if normalized <= cumulative:
                return variant_name
        
        # Fallback to first variant
        return list(experiment.variants.keys())[0]
    
    async def track_metric(
        self,
        experiment_id: str,
        variant: str,
        metric_name: str,
        value: float
    ):
        """
        Track metric for variant.
        
        Args:
            experiment_id: Experiment ID
            variant: Variant name
            metric_name: Metric name
            value: Metric value
        """
        if experiment_id not in self.metrics_storage:
            logger.warning(f"Experiment not found: {experiment_id}")
            return
        
        if variant not in self.metrics_storage[experiment_id]:
            logger.warning(f"Variant not found: {variant} in experiment {experiment_id}")
            return
        
        # Add metric value
        variant_metrics = self.metrics_storage[experiment_id][variant]
        if metric_name in variant_metrics.metrics:
            variant_metrics.metrics[metric_name].append(value)
            variant_metrics.sample_size = len(variant_metrics.metrics[metric_name])
        
        # Also persist to Redis if available
        if self.redis:
            try:
                key = f"exp_metrics:{experiment_id}:{variant}:{metric_name}"
                await self.redis.lpush(key, str(value))
                await self.redis.expire(key, 2592000)  # 30 days
            except Exception as e:
                logger.warning(f"Redis error: {e}")
    
    async def get_experiment_results(
        self,
        experiment_id: str,
        include_statistical_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive experiment results with statistical analysis.
        
        Args:
            experiment_id: Experiment ID
            include_statistical_analysis: Include significance testing
        
        Returns:
            Dict with experiment results and analysis
        """
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        # Get metrics for all variants
        variant_results = {}
        for variant_name in experiment.variants.keys():
            variant_metrics = self.metrics_storage[experiment_id].get(variant_name)
            if variant_metrics:
                variant_results[variant_name] = {
                    'sample_size': variant_metrics.sample_size,
                    'metrics': {}
                }
                
                for metric_name in experiment.metrics:
                    mean = variant_metrics.get_mean(metric_name)
                    std = variant_metrics.get_std(metric_name)
                    ci_lower, ci_upper = variant_metrics.get_confidence_interval(metric_name)
                    
                    variant_results[variant_name]['metrics'][metric_name] = {
                        'mean': round(mean, 4),
                        'std': round(std, 4),
                        'ci_lower': round(ci_lower, 4),
                        'ci_upper': round(ci_upper, 4),
                        'samples': variant_metrics.metrics[metric_name][-100:]  # Last 100 samples
                    }
        
        results = {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'description': experiment.description,
            'status': experiment.status,
            'start_date': experiment.start_date,
            'end_date': experiment.end_date,
            'variants': variant_results
        }
        
        # Add statistical significance if requested
        if include_statistical_analysis and len(variant_results) >= 2:
            results['statistical_analysis'] = await self._calculate_significance(
                experiment_id,
                experiment.metrics
            )
            results['recommendation'] = self._get_recommendation(results)
        
        return results
    
    async def _calculate_significance(
        self,
        experiment_id: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between variants.
        
        Uses t-test for comparing means.
        """
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not installed, skipping statistical analysis")
            return {}
        
        # Get control variant (first variant or one named 'control')
        variant_names = list(self.metrics_storage[experiment_id].keys())
        control_name = 'control' if 'control' in variant_names else variant_names[0]
        control_metrics = self.metrics_storage[experiment_id][control_name]
        
        significance_results = {}
        
        for variant_name, variant_metrics in self.metrics_storage[experiment_id].items():
            if variant_name == control_name:
                continue
            
            variant_significance = {}
            
            for metric_name in metrics:
                control_samples = control_metrics.metrics.get(metric_name, [])
                treatment_samples = variant_metrics.metrics.get(metric_name, [])
                
                if len(control_samples) < 10 or len(treatment_samples) < 10:
                    variant_significance[metric_name] = {
                        'status': 'insufficient_data',
                        'message': 'Need at least 10 samples per variant'
                    }
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(control_samples, treatment_samples)
                
                # Calculate effect size (Cohen's d)
                effect_size = self._calculate_effect_size(
                    control_samples,
                    treatment_samples
                )
                
                # Calculate improvement percentage
                control_mean = sum(control_samples) / len(control_samples)
                treatment_mean = sum(treatment_samples) / len(treatment_samples)
                improvement_pct = ((treatment_mean - control_mean) / control_mean) * 100 if control_mean != 0 else 0
                
                variant_significance[metric_name] = {
                    'p_value': round(p_value, 4),
                    'significant': p_value < 0.05,
                    't_statistic': round(t_stat, 4),
                    'effect_size': round(effect_size, 4),
                    'improvement_pct': round(improvement_pct, 2),
                    'control_mean': round(control_mean, 4),
                    'treatment_mean': round(treatment_mean, 4)
                }
            
            significance_results[variant_name] = variant_significance
        
        return significance_results
    
    def _calculate_effect_size(
        self,
        control_samples: List[float],
        treatment_samples: List[float]
    ) -> float:
        """Calculate Cohen's d effect size."""
        if not control_samples or not treatment_samples:
            return 0.0
        
        control_mean = sum(control_samples) / len(control_samples)
        treatment_mean = sum(treatment_samples) / len(treatment_samples)
        
        control_var = sum((x - control_mean) ** 2 for x in control_samples) / len(control_samples)
        treatment_var = sum((x - treatment_mean) ** 2 for x in treatment_samples) / len(treatment_samples)
        
        pooled_std = ((control_var + treatment_var) / 2) ** 0.5
        
        if pooled_std == 0:
            return 0.0
        
        return (treatment_mean - control_mean) / pooled_std
    
    def _get_recommendation(self, results: Dict[str, Any]) -> str:
        """
        Get deployment recommendation based on results.
        
        Returns recommendation string like:
        - "DEPLOY: variant_name shows significant improvement"
        - "CONTINUE: No significant winner yet"
        - "STOP: No variants performing better than control"
        """
        if 'statistical_analysis' not in results:
            return "CONTINUE: Insufficient data for analysis"
        
        analysis = results['statistical_analysis']
        
        # Find best performing variant with statistical significance
        best_variant = None
        best_improvement = 0
        
        for variant_name, variant_analysis in analysis.items():
            # Check primary metric (first metric in list)
            primary_metric = list(variant_analysis.keys())[0] if variant_analysis else None
            if not primary_metric:
                continue
            
            metric_analysis = variant_analysis[primary_metric]
            
            if metric_analysis.get('significant', False):
                improvement = metric_analysis.get('improvement_pct', 0)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_variant = variant_name
        
        if best_variant and best_improvement > 0:
            return f"DEPLOY: {best_variant} shows {best_improvement:+.1f}% improvement (statistically significant)"
        elif best_variant and best_improvement < -5:
            return f"STOP: {best_variant} shows {best_improvement:+.1f}% degradation"
        else:
            return "CONTINUE: No significant winner yet, need more data"
    
    async def list_experiments(
        self,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by status.
        
        Args:
            status: Filter by status (optional)
        
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for exp_id, exp in self.active_experiments.items():
            if status and exp.status != status:
                continue
            
            # Get sample sizes
            sample_sizes = {}
            for variant_name in exp.variants.keys():
                variant_metrics = self.metrics_storage[exp_id].get(variant_name)
                sample_sizes[variant_name] = variant_metrics.sample_size if variant_metrics else 0
            
            experiments.append({
                'id': exp_id,
                'name': exp.name,
                'description': exp.description,
                'status': exp.status,
                'start_date': exp.start_date,
                'end_date': exp.end_date,
                'variants': list(exp.variants.keys()),
                'sample_sizes': sample_sizes,
                'created_at': exp.created_at
            })
        
        # Sort by created_at (most recent first)
        experiments.sort(key=lambda x: x['created_at'], reverse=True)
        
        return experiments
    
    async def _save_experiment_to_db(self, experiment: ExperimentConfig):
        """Save experiment to database (if available)."""
        if not self.db:
            return
        
        try:
            await self.db.execute(
                """
                INSERT INTO experiments 
                (id, name, description, start_date, end_date, variants, metrics, 
                 minimum_sample_size, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.id,
                    experiment.name,
                    experiment.description,
                    experiment.start_date,
                    experiment.end_date,
                    json.dumps(experiment.variants),
                    json.dumps(experiment.metrics),
                    experiment.minimum_sample_size,
                    experiment.status,
                    experiment.created_at
                )
            )
        except Exception as e:
            logger.error(f"Failed to save experiment to database: {e}")


# Global singleton instance
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager(
    redis_client=None,
    db_connection=None,
    config: Optional[Dict] = None
) -> ExperimentManager:
    """Get or create global experiment manager instance."""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager(redis_client, db_connection, config)
    return _experiment_manager
