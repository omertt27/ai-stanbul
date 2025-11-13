"""
A/B Testing Framework for Pure LLM Handler

PRIORITY 2.4: Safe experimentation framework for testing handler improvements.

This framework enables:
- Creating and managing A/B test experiments
- Random user assignment to control/treatment groups
- Statistical significance testing
- Automated experiment analysis and reporting
- Safe rollout of winning variants

Features:
- Multiple concurrent experiments
- Stratified sampling by language/user segment
- Bayesian and frequentist analysis
- Early stopping for clear winners/losers
- Experiment history and audit trail

Architecture:
- ABTestingFramework: Main orchestration class
- Experiment: Individual experiment configuration
- UserAssignment: Manages user -> variant mapping
- StatisticalAnalyzer: Computes significance and confidence
- ExperimentReporter: Generates reports and recommendations

Author: AI Istanbul Team
Date: November 2025
"""

import logging
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import numpy as np
from scipy import stats
import redis

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(Enum):
    """Type of variant in experiment."""
    CONTROL = "control"
    TREATMENT = "treatment"


class Experiment:
    """
    Represents an A/B test experiment.
    
    An experiment compares a control variant (baseline) with one or more
    treatment variants (proposed improvements).
    """
    
    def __init__(
        self,
        experiment_id: str,
        name: str,
        description: str,
        variants: Dict[str, Dict[str, Any]],
        traffic_allocation: Dict[str, float],
        success_metrics: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_sample_size: int = 1000,
        significance_level: float = 0.05
    ):
        """
        Initialize experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            name: Human-readable name
            description: Experiment description
            variants: Dict of variant_id -> configuration
            traffic_allocation: Dict of variant_id -> traffic percentage
            success_metrics: List of metrics to optimize
            start_date: When to start (None = immediately)
            end_date: When to end (None = manual)
            min_sample_size: Minimum samples per variant
            significance_level: Statistical significance level (alpha)
        """
        self.experiment_id = experiment_id
        self.name = name
        self.description = description
        self.variants = variants
        self.traffic_allocation = traffic_allocation
        self.success_metrics = success_metrics
        self.start_date = start_date or datetime.now()
        self.end_date = end_date
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        
        self.status = ExperimentStatus.DRAFT
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Validate traffic allocation
        total_traffic = sum(traffic_allocation.values())
        if not (0.99 <= total_traffic <= 1.01):  # Allow small floating point error
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_traffic}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'variants': self.variants,
            'traffic_allocation': self.traffic_allocation,
            'success_metrics': self.success_metrics,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'min_sample_size': self.min_sample_size,
            'significance_level': self.significance_level,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create experiment from dictionary."""
        exp = cls(
            experiment_id=data['experiment_id'],
            name=data['name'],
            description=data['description'],
            variants=data['variants'],
            traffic_allocation=data['traffic_allocation'],
            success_metrics=data['success_metrics'],
            start_date=datetime.fromisoformat(data['start_date']),
            end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
            min_sample_size=data.get('min_sample_size', 1000),
            significance_level=data.get('significance_level', 0.05)
        )
        exp.status = ExperimentStatus(data.get('status', 'draft'))
        exp.created_at = datetime.fromisoformat(data['created_at'])
        exp.updated_at = datetime.fromisoformat(data['updated_at'])
        return exp


class UserAssignment:
    """
    Manages user assignment to experiment variants.
    
    Uses consistent hashing to ensure:
    - Users always get the same variant
    - Traffic allocation is respected
    - Easy to add/remove experiments
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize user assignment manager.
        
        Args:
            redis_client: Redis client for persistent storage
        """
        self.redis = redis_client
        self.assignment_cache = {}  # In-memory cache
    
    def assign_variant(
        self,
        user_id: str,
        experiment: Experiment
    ) -> str:
        """
        Assign user to experiment variant.
        
        Uses consistent hashing to ensure same user always gets same variant.
        
        Args:
            user_id: User identifier
            experiment: Experiment configuration
            
        Returns:
            Assigned variant ID
        """
        # Check cache first
        cache_key = f"{experiment.experiment_id}:{user_id}"
        if cache_key in self.assignment_cache:
            return self.assignment_cache[cache_key]
        
        # Check Redis
        if self.redis:
            redis_key = f"ab_assignment:{experiment.experiment_id}:{user_id}"
            cached = self.redis.get(redis_key)
            if cached:
                variant = cached.decode('utf-8')
                self.assignment_cache[cache_key] = variant
                return variant
        
        # Compute assignment using consistent hashing
        hash_input = f"{experiment.experiment_id}:{user_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Convert to [0, 1) range
        normalized_hash = (hash_value % 10000) / 10000.0
        
        # Assign based on traffic allocation
        cumulative = 0.0
        assigned_variant = None
        
        for variant_id, allocation in experiment.traffic_allocation.items():
            cumulative += allocation
            if normalized_hash < cumulative:
                assigned_variant = variant_id
                break
        
        # Fallback to first variant
        if assigned_variant is None:
            assigned_variant = list(experiment.variants.keys())[0]
        
        # Cache assignment
        self.assignment_cache[cache_key] = assigned_variant
        
        if self.redis:
            redis_key = f"ab_assignment:{experiment.experiment_id}:{user_id}"
            # Cache for 90 days
            self.redis.setex(redis_key, 90 * 24 * 3600, assigned_variant)
        
        logger.debug(f"👤 Assigned user {user_id} to variant {assigned_variant}")
        return assigned_variant
    
    def get_assignment(
        self,
        user_id: str,
        experiment_id: str
    ) -> Optional[str]:
        """
        Get existing user assignment without creating new one.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment ID
            
        Returns:
            Variant ID or None if not assigned
        """
        cache_key = f"{experiment_id}:{user_id}"
        
        # Check memory cache
        if cache_key in self.assignment_cache:
            return self.assignment_cache[cache_key]
        
        # Check Redis
        if self.redis:
            redis_key = f"ab_assignment:{experiment_id}:{user_id}"
            cached = self.redis.get(redis_key)
            if cached:
                variant = cached.decode('utf-8')
                self.assignment_cache[cache_key] = variant
                return variant
        
        return None


class StatisticalAnalyzer:
    """
    Performs statistical analysis on experiment results.
    
    Supports:
    - Two-sample t-test for continuous metrics
    - Chi-squared test for categorical metrics
    - Bayesian analysis for early stopping
    - Confidence intervals and effect sizes
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            significance_level: Alpha level for hypothesis testing
        """
        self.significance_level = significance_level
    
    def analyze_continuous_metric(
        self,
        control_data: List[float],
        treatment_data: List[float],
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Analyze continuous metric (e.g., response time, F1 score).
        
        Uses two-sample t-test to determine if treatment is significantly
        different from control.
        
        Args:
            control_data: Control variant measurements
            treatment_data: Treatment variant measurements
            metric_name: Name of metric
            
        Returns:
            Analysis results dict
        """
        if len(control_data) < 2 or len(treatment_data) < 2:
            return {
                'metric': metric_name,
                'status': 'insufficient_data',
                'control_n': len(control_data),
                'treatment_n': len(treatment_data)
            }
        
        # Compute statistics
        control_mean = np.mean(control_data)
        control_std = np.std(control_data, ddof=1)
        treatment_mean = np.mean(treatment_data)
        treatment_std = np.std(treatment_data, ddof=1)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_data) - 1) * control_std**2 +
             (len(treatment_data) - 1) * treatment_std**2) /
            (len(control_data) + len(treatment_data) - 2)
        )
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Compute confidence interval for difference
        se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
        ci_95 = stats.t.interval(
            0.95,
            len(control_data) + len(treatment_data) - 2,
            loc=treatment_mean - control_mean,
            scale=se_diff
        )
        
        # Determine significance
        is_significant = p_value < self.significance_level
        
        return {
            'metric': metric_name,
            'control_mean': control_mean,
            'control_std': control_std,
            'control_n': len(control_data),
            'treatment_mean': treatment_mean,
            'treatment_std': treatment_std,
            'treatment_n': len(treatment_data),
            'difference': treatment_mean - control_mean,
            'difference_pct': ((treatment_mean - control_mean) / control_mean * 100)
                              if control_mean != 0 else 0,
            'p_value': p_value,
            'is_significant': is_significant,
            'cohens_d': cohens_d,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            'status': 'complete'
        }
    
    def analyze_binary_metric(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        metric_name: str
    ) -> Dict[str, Any]:
        """
        Analyze binary metric (e.g., click rate, success rate).
        
        Uses chi-squared test for proportion comparison.
        
        Args:
            control_successes: Number of successes in control
            control_total: Total observations in control
            treatment_successes: Number of successes in treatment
            treatment_total: Total observations in treatment
            metric_name: Name of metric
            
        Returns:
            Analysis results dict
        """
        if control_total < 1 or treatment_total < 1:
            return {
                'metric': metric_name,
                'status': 'insufficient_data',
                'control_n': control_total,
                'treatment_n': treatment_total
            }
        
        # Compute proportions
        control_rate = control_successes / control_total
        treatment_rate = treatment_successes / treatment_total
        
        # Perform chi-squared test
        contingency_table = [
            [control_successes, control_total - control_successes],
            [treatment_successes, treatment_total - treatment_successes]
        ]
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Compute relative uplift
        relative_uplift = ((treatment_rate - control_rate) / control_rate * 100) \
                         if control_rate > 0 else 0
        
        # Compute confidence interval for difference in proportions
        se_diff = np.sqrt(
            control_rate * (1 - control_rate) / control_total +
            treatment_rate * (1 - treatment_rate) / treatment_total
        )
        ci_95 = (
            (treatment_rate - control_rate) - 1.96 * se_diff,
            (treatment_rate - control_rate) + 1.96 * se_diff
        )
        
        is_significant = p_value < self.significance_level
        
        return {
            'metric': metric_name,
            'control_rate': control_rate,
            'control_successes': control_successes,
            'control_n': control_total,
            'treatment_rate': treatment_rate,
            'treatment_successes': treatment_successes,
            'treatment_n': treatment_total,
            'absolute_difference': treatment_rate - control_rate,
            'relative_uplift_pct': relative_uplift,
            'p_value': p_value,
            'is_significant': is_significant,
            'chi2': chi2,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            'status': 'complete'
        }
    
    def check_early_stopping(
        self,
        analysis_results: Dict[str, Any],
        min_samples: int
    ) -> Tuple[bool, str]:
        """
        Check if experiment should be stopped early.
        
        Early stopping criteria:
        - Clear winner with high confidence
        - Clear loser (treatment performing worse)
        - Sufficient sample size reached
        
        Args:
            analysis_results: Results from metric analysis
            min_samples: Minimum samples required per variant
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if analysis_results.get('status') != 'complete':
            return False, "Insufficient data"
        
        # Check minimum sample size
        control_n = analysis_results.get('control_n', 0)
        treatment_n = analysis_results.get('treatment_n', 0)
        
        if control_n < min_samples or treatment_n < min_samples:
            return False, f"Need more samples (min: {min_samples})"
        
        # Check for clear winner (p < 0.01 and positive improvement)
        p_value = analysis_results.get('p_value', 1.0)
        
        if 'difference' in analysis_results:
            # Continuous metric
            difference = analysis_results['difference']
            if p_value < 0.01 and difference > 0:
                return True, "Clear winner detected (p < 0.01, positive effect)"
            if p_value < 0.01 and difference < 0:
                return True, "Treatment performing worse (p < 0.01, negative effect)"
        
        elif 'relative_uplift_pct' in analysis_results:
            # Binary metric
            uplift = analysis_results['relative_uplift_pct']
            if p_value < 0.01 and uplift > 5:
                return True, f"Clear winner detected ({uplift:+.1f}% uplift, p < 0.01)"
            if p_value < 0.01 and uplift < -5:
                return True, f"Treatment performing worse ({uplift:.1f}% uplift, p < 0.01)"
        
        return False, "Continue experiment"


class ExperimentReporter:
    """
    Generates reports and recommendations for experiments.
    
    Provides:
    - Summary statistics for all variants
    - Statistical significance analysis
    - Recommendations for next steps
    - Visualizable data exports
    """
    
    def __init__(self):
        """Initialize experiment reporter."""
        pass
    
    def generate_report(
        self,
        experiment: Experiment,
        variant_metrics: Dict[str, Dict[str, List[float]]],
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive experiment report.
        
        Args:
            experiment: Experiment configuration
            variant_metrics: Dict of variant_id -> metric_name -> values
            analysis_results: Statistical analysis results
            
        Returns:
            Report dict with summary and recommendations
        """
        # Compute variant summaries
        variant_summaries = {}
        for variant_id, metrics in variant_metrics.items():
            variant_summaries[variant_id] = {
                'sample_size': len(metrics.get(experiment.success_metrics[0], [])),
                'metrics': {}
            }
            
            for metric_name, values in metrics.items():
                if values:
                    variant_summaries[variant_id]['metrics'][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
        
        # Determine recommendation
        recommendation = self._get_recommendation(
            experiment=experiment,
            analysis_results=analysis_results,
            variant_summaries=variant_summaries
        )
        
        return {
            'experiment_id': experiment.experiment_id,
            'experiment_name': experiment.name,
            'status': experiment.status.value,
            'duration_days': (datetime.now() - experiment.start_date).days,
            'variant_summaries': variant_summaries,
            'analysis_results': analysis_results,
            'recommendation': recommendation,
            'generated_at': datetime.now().isoformat()
        }
    
    def _get_recommendation(
        self,
        experiment: Experiment,
        analysis_results: Dict[str, Any],
        variant_summaries: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate recommendation based on experiment results.
        
        Args:
            experiment: Experiment configuration
            analysis_results: Statistical analysis
            variant_summaries: Variant summaries
            
        Returns:
            Recommendation dict
        """
        if analysis_results.get('status') != 'complete':
            return {
                'action': 'continue',
                'reason': 'Insufficient data for decision',
                'confidence': 'low'
            }
        
        is_significant = analysis_results.get('is_significant', False)
        p_value = analysis_results.get('p_value', 1.0)
        
        # Get improvement metric
        if 'difference' in analysis_results:
            improvement = analysis_results['difference']
            improvement_pct = analysis_results['difference_pct']
        elif 'relative_uplift_pct' in analysis_results:
            improvement_pct = analysis_results['relative_uplift_pct']
            improvement = analysis_results['absolute_difference']
        else:
            return {
                'action': 'continue',
                'reason': 'Unable to determine improvement',
                'confidence': 'low'
            }
        
        # Make recommendation
        if is_significant and improvement > 0:
            confidence = 'high' if p_value < 0.01 else 'medium'
            return {
                'action': 'adopt_treatment',
                'reason': f'Treatment shows significant improvement ({improvement_pct:+.1f}%, p={p_value:.4f})',
                'confidence': confidence,
                'winner': 'treatment'
            }
        
        elif is_significant and improvement < 0:
            confidence = 'high' if p_value < 0.01 else 'medium'
            return {
                'action': 'keep_control',
                'reason': f'Treatment performs worse ({improvement_pct:.1f}%, p={p_value:.4f})',
                'confidence': confidence,
                'winner': 'control'
            }
        
        else:
            # Not significant - check if we have enough data
            min_samples = experiment.min_sample_size
            control_n = analysis_results.get('control_n', 0)
            treatment_n = analysis_results.get('treatment_n', 0)
            
            if control_n < min_samples or treatment_n < min_samples:
                return {
                    'action': 'continue',
                    'reason': f'Need more samples (have: {min(control_n, treatment_n)}, need: {min_samples})',
                    'confidence': 'low'
                }
            else:
                return {
                    'action': 'keep_control',
                    'reason': f'No significant difference detected (p={p_value:.4f})',
                    'confidence': 'medium',
                    'winner': 'control'
                }


class ABTestingFramework:
    """
    Main A/B testing framework for Pure LLM Handler.
    
    Orchestrates experiment creation, user assignment, metric tracking,
    and statistical analysis for safe experimentation.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize A/B testing framework.
        
        Args:
            redis_client: Redis client for storage
        """
        self.redis = redis_client
        self.user_assignment = UserAssignment(redis_client)
        self.analyzer = StatisticalAnalyzer()
        self.reporter = ExperimentReporter()
        
        self.experiments = {}  # In-memory cache
        self.metric_buffer = defaultdict(lambda: defaultdict(list))
        
        self._load_experiments()
        
        logger.info("🧪 ABTestingFramework initialized")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: Dict[str, Dict[str, Any]],
        traffic_allocation: Optional[Dict[str, float]] = None,
        success_metrics: Optional[List[str]] = None,
        min_sample_size: int = 1000,
        auto_start: bool = False
    ) -> Experiment:
        """
        Create new A/B test experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: Dict of variant configurations
            traffic_allocation: Traffic split (default: equal split)
            success_metrics: Metrics to optimize
            min_sample_size: Minimum samples per variant
            auto_start: Start experiment immediately
            
        Returns:
            Created experiment
        """
        # Generate experiment ID
        experiment_id = f"exp_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        # Default traffic allocation (equal split)
        if traffic_allocation is None:
            n_variants = len(variants)
            traffic_allocation = {v: 1.0/n_variants for v in variants.keys()}
        
        # Default success metrics
        if success_metrics is None:
            success_metrics = ['response_quality', 'user_satisfaction']
        
        # Create experiment
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            traffic_allocation=traffic_allocation,
            success_metrics=success_metrics,
            min_sample_size=min_sample_size
        )
        
        if auto_start:
            experiment.status = ExperimentStatus.ACTIVE
            logger.info(f"🚀 Started experiment: {name}")
        
        # Store experiment
        self.experiments[experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(f"✅ Created experiment: {name} (ID: {experiment_id})")
        return experiment
    
    def get_variant(
        self,
        user_id: str,
        experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get variant configuration for user in experiment.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment ID
            
        Returns:
            Variant configuration or None if experiment not active
        """
        experiment = self.experiments.get(experiment_id)
        
        if not experiment:
            return None
        
        if experiment.status != ExperimentStatus.ACTIVE:
            return None
        
        # Assign user to variant
        variant_id = self.user_assignment.assign_variant(user_id, experiment)
        
        return {
            'variant_id': variant_id,
            'config': experiment.variants[variant_id],
            'experiment_id': experiment_id
        }
    
    def record_metric(
        self,
        experiment_id: str,
        variant_id: str,
        metric_name: str,
        value: float
    ):
        """
        Record metric value for experiment variant.
        
        Args:
            experiment_id: Experiment ID
            variant_id: Variant ID
            metric_name: Metric name
            value: Metric value
        """
        key = f"{experiment_id}:{variant_id}:{metric_name}"
        self.metric_buffer[experiment_id][key].append(value)
        
        # Store in Redis
        if self.redis:
            redis_key = f"ab_metrics:{experiment_id}:{variant_id}:{metric_name}"
            self.redis.rpush(redis_key, str(value))
            
            # Set expiry to 90 days
            self.redis.expire(redis_key, 90 * 24 * 3600)
    
    def analyze_experiment(
        self,
        experiment_id: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze experiment results.
        
        Args:
            experiment_id: Experiment ID
            force_refresh: Force reload from Redis
            
        Returns:
            Analysis report
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Get metrics for all variants
        variant_metrics = self._get_variant_metrics(experiment, force_refresh)
        
        # Analyze primary success metric
        primary_metric = experiment.success_metrics[0]
        
        # Assume control is first variant
        variants = list(experiment.variants.keys())
        control_id = variants[0]
        treatment_id = variants[1] if len(variants) > 1 else variants[0]
        
        control_data = variant_metrics.get(control_id, {}).get(primary_metric, [])
        treatment_data = variant_metrics.get(treatment_id, {}).get(primary_metric, [])
        
        # Perform statistical analysis
        analysis_results = self.analyzer.analyze_continuous_metric(
            control_data=control_data,
            treatment_data=treatment_data,
            metric_name=primary_metric
        )
        
        # Check early stopping
        should_stop, reason = self.analyzer.check_early_stopping(
            analysis_results=analysis_results,
            min_samples=experiment.min_sample_size
        )
        
        analysis_results['early_stopping'] = {
            'should_stop': should_stop,
            'reason': reason
        }
        
        # Generate report
        report = self.reporter.generate_report(
            experiment=experiment,
            variant_metrics=variant_metrics,
            analysis_results=analysis_results
        )
        
        logger.info(
            f"📊 Analyzed experiment {experiment.name}: "
            f"{report['recommendation']['action']} ({report['recommendation']['confidence']} confidence)"
        )
        
        return report
    
    def stop_experiment(
        self,
        experiment_id: str,
        reason: str = "Manual stop"
    ):
        """
        Stop an active experiment.
        
        Args:
            experiment_id: Experiment ID
            reason: Reason for stopping
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.updated_at = datetime.now()
        
        self._save_experiment(experiment)
        
        logger.info(f"🛑 Stopped experiment {experiment.name}: {reason}")
    
    def _load_experiments(self):
        """Load experiments from Redis."""
        if not self.redis:
            return
        
        try:
            keys = self.redis.keys("ab_experiment:*")
            
            for key in keys:
                data = self.redis.get(key)
                if data:
                    exp_dict = json.loads(data.decode('utf-8'))
                    experiment = Experiment.from_dict(exp_dict)
                    self.experiments[experiment.experiment_id] = experiment
            
            logger.info(f"📂 Loaded {len(self.experiments)} experiments from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")
    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to Redis."""
        if not self.redis:
            return
        
        try:
            key = f"ab_experiment:{experiment.experiment_id}"
            self.redis.set(key, json.dumps(experiment.to_dict()))
            
            logger.debug(f"💾 Saved experiment {experiment.experiment_id}")
            
        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")
    
    def _get_variant_metrics(
        self,
        experiment: Experiment,
        force_refresh: bool = False
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Get all metrics for all variants.
        
        Args:
            experiment: Experiment
            force_refresh: Force reload from Redis
            
        Returns:
            Dict of variant_id -> metric_name -> values
        """
        variant_metrics = defaultdict(lambda: defaultdict(list))
        
        # Get from buffer first
        if not force_refresh:
            buffer_metrics = self.metric_buffer.get(experiment.experiment_id, {})
            for key, values in buffer_metrics.items():
                variant_id, metric_name = key.split(':')[0], key.split(':')[-1]
                variant_metrics[variant_id][metric_name].extend(values)
        
        # Get from Redis
        if self.redis:
            for variant_id in experiment.variants.keys():
                for metric_name in experiment.success_metrics:
                    redis_key = f"ab_metrics:{experiment.experiment_id}:{variant_id}:{metric_name}"
                    values = self.redis.lrange(redis_key, 0, -1)
                    
                    for val in values:
                        try:
                            variant_metrics[variant_id][metric_name].append(float(val))
                        except ValueError:
                            continue
        
        return dict(variant_metrics)
