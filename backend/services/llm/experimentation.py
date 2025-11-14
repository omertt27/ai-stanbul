"""
experimentation.py - Experimentation System

A/B testing and threshold learning for continuous improvement.

Features:
- A/B test framework
- Threshold learning from user feedback
- Auto-tuning system
- Statistical analysis
- Winner determination
- Gradual rollout

Author: AI Istanbul Team
Date: November 2025
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class ExperimentationManager:
    """
    Experimentation system for A/B testing and threshold learning.
    
    Features:
    - Run A/B tests on system parameters
    - Learn optimal thresholds from user feedback
    - Automatic tuning and rollout
    """
    
    def __init__(
        self,
        enable_ab_testing: bool = False,
        enable_threshold_learning: bool = True,
        auto_tune_interval_hours: int = 24
    ):
        """
        Initialize experimentation manager.
        
        Args:
            enable_ab_testing: Enable A/B testing
            enable_threshold_learning: Enable threshold learning
            auto_tune_interval_hours: Hours between auto-tuning runs
        """
        self.enable_ab_testing = enable_ab_testing
        self.enable_threshold_learning = enable_threshold_learning
        self.auto_tune_interval = auto_tune_interval_hours
        
        # A/B test storage
        self.experiments = {}
        self.user_assignments = {}  # user_id -> experiment_id -> variant
        
        # Threshold learning storage
        self.feedback_data = defaultdict(list)  # signal_name -> feedback list
        self.last_auto_tune = {}  # language -> timestamp
        
        logger.info(f"✅ Experimentation Manager initialized (A/B={'on' if enable_ab_testing else 'off'})")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: Dict[str, Any],
        traffic_allocation: Optional[Dict[str, float]] = None,
        success_metrics: Optional[List[str]] = None,
        min_sample_size: int = 100,
        auto_start: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            description: Description
            variants: Dict of variant_id -> config
            traffic_allocation: Dict of variant_id -> percentage (0-1)
            success_metrics: List of metric names to track
            min_sample_size: Minimum samples before analysis
            auto_start: Start experiment immediately
            
        Returns:
            Experiment dict
        """
        if not self.enable_ab_testing:
            logger.warning("A/B testing disabled")
            return {}
        
        experiment_id = self._generate_experiment_id(name)
        
        # Default traffic allocation (equal split)
        if not traffic_allocation:
            num_variants = len(variants)
            traffic_allocation = {
                variant_id: 1.0 / num_variants
                for variant_id in variants.keys()
            }
        
        experiment = {
            'experiment_id': experiment_id,
            'name': name,
            'description': description,
            'variants': variants,
            'traffic_allocation': traffic_allocation,
            'success_metrics': success_metrics or [],
            'min_sample_size': min_sample_size,
            'status': 'running' if auto_start else 'draft',
            'created_at': datetime.now().isoformat(),
            'started_at': datetime.now().isoformat() if auto_start else None,
            'ended_at': None,
            'results': defaultdict(lambda: defaultdict(list))  # variant_id -> metric -> values
        }
        
        self.experiments[experiment_id] = experiment
        
        logger.info(f"🧪 Created experiment: {name} (ID: {experiment_id})")
        
        return experiment
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().isoformat()
        unique_str = f"{name}:{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def get_variant(
        self,
        user_id: str,
        experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get variant assignment for a user.
        
        Args:
            user_id: User identifier
            experiment_id: Experiment ID
            
        Returns:
            Variant dict or None
        """
        if not self.enable_ab_testing:
            return None
        
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != 'running':
            return None
        
        # Check if user already assigned
        if user_id in self.user_assignments:
            if experiment_id in self.user_assignments[user_id]:
                variant_id = self.user_assignments[user_id][experiment_id]
                return {
                    'variant_id': variant_id,
                    'config': experiment['variants'][variant_id]
                }
        
        # Assign variant based on traffic allocation
        variant_id = self._assign_variant(
            user_id,
            experiment['traffic_allocation']
        )
        
        # Store assignment
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = variant_id
        
        return {
            'variant_id': variant_id,
            'config': experiment['variants'][variant_id]
        }
    
    def _assign_variant(
        self,
        user_id: str,
        traffic_allocation: Dict[str, float]
    ) -> str:
        """Assign variant based on traffic allocation."""
        # Hash user_id for consistent assignment
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (hash_val % 100) / 100.0  # 0.00 to 0.99
        
        # Assign based on traffic allocation
        cumulative = 0.0
        for variant_id, allocation in traffic_allocation.items():
            cumulative += allocation
            if bucket < cumulative:
                return variant_id
        
        # Fallback to first variant
        return list(traffic_allocation.keys())[0]
    
    def record_metric(
        self,
        experiment_id: str,
        variant_id: str,
        metric_name: str,
        value: float
    ):
        """
        Record a metric value for an experiment variant.
        
        Args:
            experiment_id: Experiment ID
            variant_id: Variant ID
            metric_name: Metric name
            value: Metric value
        """
        if not self.enable_ab_testing:
            return
        
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        experiment['results'][variant_id][metric_name].append(value)
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze experiment results.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Analysis results
        """
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiments[experiment_id]
        results = experiment['results']
        
        # Calculate statistics for each variant
        variant_stats = {}
        
        for variant_id, metrics in results.items():
            variant_stats[variant_id] = {}
            
            for metric_name, values in metrics.items():
                if not values:
                    continue
                
                stats = self._calculate_stats(values)
                variant_stats[variant_id][metric_name] = stats
        
        # Determine winner
        winner = self._determine_winner(variant_stats, experiment['success_metrics'])
        
        # Check if we have enough data
        min_samples = experiment['min_sample_size']
        has_enough_data = all(
            len(metrics.get(experiment['success_metrics'][0], [])) >= min_samples
            for metrics in results.values()
        ) if experiment['success_metrics'] else False
        
        return {
            'experiment': experiment,
            'variant_stats': variant_stats,
            'winner': winner,
            'has_enough_data': has_enough_data,
            'recommendation': self._get_recommendation(winner, has_enough_data)
        }
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {}
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        return {
            'count': n,
            'mean': sum(sorted_vals) / n,
            'min': sorted_vals[0],
            'max': sorted_vals[-1],
            'median': sorted_vals[n // 2],
            'p95': sorted_vals[int(n * 0.95)] if n > 20 else sorted_vals[-1]
        }
    
    def _determine_winner(
        self,
        variant_stats: Dict[str, Dict[str, Dict[str, float]]],
        success_metrics: List[str]
    ) -> Optional[str]:
        """Determine winning variant based on success metrics."""
        if not success_metrics or not variant_stats:
            return None
        
        primary_metric = success_metrics[0]
        
        # Compare variants on primary metric
        best_variant = None
        best_value = -float('inf')
        
        for variant_id, stats in variant_stats.items():
            if primary_metric in stats:
                mean_value = stats[primary_metric].get('mean', 0)
                if mean_value > best_value:
                    best_value = mean_value
                    best_variant = variant_id
        
        return best_variant
    
    def _get_recommendation(
        self,
        winner: Optional[str],
        has_enough_data: bool
    ) -> Dict[str, Any]:
        """Get experiment recommendation."""
        if not has_enough_data:
            return {
                'action': 'continue',
                'reason': 'Not enough data collected yet'
            }
        
        if winner:
            return {
                'action': 'rollout',
                'winner': winner,
                'reason': f'Variant {winner} shows best performance'
            }
        
        return {
            'action': 'inconclusive',
            'reason': 'No clear winner detected'
        }
    
    def stop_experiment(self, experiment_id: str, reason: str):
        """
        Stop a running experiment.
        
        Args:
            experiment_id: Experiment ID
            reason: Reason for stopping
        """
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['status'] = 'stopped'
            self.experiments[experiment_id]['ended_at'] = datetime.now().isoformat()
            self.experiments[experiment_id]['stop_reason'] = reason
            
            logger.info(f"🛑 Stopped experiment {experiment_id}: {reason}")
    
    def record_feedback(
        self,
        query: str,
        detected_signals: Dict[str, bool],
        confidence_scores: Dict[str, float],
        feedback_type: str,
        feedback_data: Dict[str, Any],
        language: str = "en"
    ):
        """
        Record user feedback for threshold learning.
        
        Args:
            query: User query
            detected_signals: Detected signals
            confidence_scores: Confidence scores
            feedback_type: 'implicit' or 'explicit'
            feedback_data: Feedback data
            language: Language code
        """
        if not self.enable_threshold_learning:
            return
        
        feedback_entry = {
            'query': query,
            'detected_signals': detected_signals,
            'confidence_scores': confidence_scores,
            'feedback_type': feedback_type,
            'feedback_data': feedback_data,
            'language': language,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store feedback for each signal
        for signal_name in detected_signals.keys():
            self.feedback_data[signal_name].append(feedback_entry)
        
        logger.debug(f"📝 Recorded {feedback_type} feedback for threshold learning")
    
    async def auto_tune_thresholds(
        self,
        language: str = "en",
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Automatically tune thresholds based on feedback.
        
        Args:
            language: Language to tune
            force: Force tuning even if interval not met
            
        Returns:
            Tuning results
        """
        if not self.enable_threshold_learning:
            return {'status': 'disabled'}
        
        # Check if tuning needed
        if not force:
            last_tune = self.last_auto_tune.get(language)
            if last_tune:
                hours_since = (datetime.now() - last_tune).total_seconds() / 3600
                if hours_since < self.auto_tune_interval:
                    return {
                        'status': 'skipped',
                        'reason': f'Last tuned {hours_since:.1f} hours ago'
                    }
        
        # Analyze feedback for each signal
        recommendations = {}
        
        for signal_name, feedback_list in self.feedback_data.items():
            if not feedback_list:
                continue
            
            # Analyze feedback
            analysis = self._analyze_feedback(feedback_list, signal_name, language)
            
            if analysis.get('should_tune'):
                recommendations[signal_name] = analysis
        
        # Update last tune time
        self.last_auto_tune[language] = datetime.now()
        
        return {
            'status': 'complete',
            'language': language,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_feedback(
        self,
        feedback_list: List[Dict[str, Any]],
        signal_name: str,
        language: str
    ) -> Dict[str, Any]:
        """Analyze feedback to recommend threshold changes."""
        # Filter for language
        relevant_feedback = [
            fb for fb in feedback_list
            if fb.get('language') == language
        ]
        
        if len(relevant_feedback) < 20:
            return {
                'should_tune': False,
                'reason': 'Insufficient feedback data'
            }
        
        # Analyze positive vs negative feedback
        positive = 0
        negative = 0
        
        for fb in relevant_feedback:
            feedback_data = fb.get('feedback_data', {})
            
            if feedback_data.get('type') == 'thumbs_up':
                positive += 1
            elif feedback_data.get('type') == 'thumbs_down':
                negative += 1
        
        total_feedback = positive + negative
        if total_feedback < 10:
            return {
                'should_tune': False,
                'reason': 'Insufficient explicit feedback'
            }
        
        positive_rate = positive / total_feedback
        
        # Recommend threshold adjustment
        current_threshold = 0.35  # Default, should come from config
        
        if positive_rate < 0.6:
            # Too many false positives, increase threshold
            recommended = min(0.50, current_threshold + 0.05)
            return {
                'should_tune': True,
                'current': current_threshold,
                'recommended': recommended,
                'reason': f'Low positive rate ({positive_rate:.2f}), reducing false positives',
                'confidence': 0.7
            }
        elif positive_rate > 0.9:
            # Very high accuracy, can try lower threshold to catch more
            recommended = max(0.25, current_threshold - 0.05)
            return {
                'should_tune': True,
                'current': current_threshold,
                'recommended': recommended,
                'reason': f'High positive rate ({positive_rate:.2f}), increasing sensitivity',
                'confidence': 0.6
            }
        
        return {
            'should_tune': False,
            'reason': f'Threshold performing well (positive rate: {positive_rate:.2f})'
        }
    
    def get_threshold_for_experiment(
        self,
        signal_name: str,
        language: str,
        user_id: str
    ) -> Optional[float]:
        """
        Get experimental threshold value if user is in A/B test.
        
        Args:
            signal_name: Signal name
            language: Language code
            user_id: User ID
            
        Returns:
            Threshold value or None
        """
        if not self.enable_ab_testing:
            return None
        
        # Find active threshold experiment for this signal
        for experiment_id, experiment in self.experiments.items():
            if experiment['status'] != 'running':
                continue
            
            # Check if this experiment is for the signal
            if signal_name not in experiment.get('name', ''):
                continue
            
            # Get user's variant
            variant = self.get_variant(user_id, experiment_id)
            
            if variant:
                config = variant.get('config', {})
                if config.get('signal') == signal_name and config.get('language') == language:
                    return config.get('threshold')
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get experimentation statistics."""
        active_experiments = sum(
            1 for exp in self.experiments.values()
            if exp['status'] == 'running'
        )
        
        total_feedback = sum(
            len(feedback_list)
            for feedback_list in self.feedback_data.values()
        )
        
        return {
            'ab_testing_enabled': self.enable_ab_testing,
            'threshold_learning_enabled': self.enable_threshold_learning,
            'total_experiments': len(self.experiments),
            'active_experiments': active_experiments,
            'total_feedback_entries': total_feedback,
            'signals_with_feedback': len(self.feedback_data)
        }
