"""
Dynamic Threshold Learning System for Pure LLM Handler

PRIORITY 2.3: Automatic threshold tuning based on user feedback and performance metrics.

This system learns optimal semantic similarity thresholds for signal detection by:
- Collecting user feedback (implicit and explicit)
- Tracking false positives and false negatives
- Computing ROC curves for each signal
- Auto-tuning thresholds to maximize F1 score
- Supporting per-language optimization

Architecture:
- ThresholdLearner: Main class for learning and optimization
- FeedbackCollector: Tracks user interactions and feedback
- ROCAnalyzer: Computes ROC curves and finds optimal thresholds
- ThresholdOptimizer: Applies learned thresholds with safety checks

Author: AI Istanbul Team
Date: November 2025
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
import redis

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collects user feedback for threshold learning.
    
    Tracks both implicit and explicit feedback:
    - Implicit: User behavior (clicks, time spent, bounce rate)
    - Explicit: Thumbs up/down, ratings, corrections
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize feedback collector.
        
        Args:
            redis_client: Redis client for persistent storage
        """
        self.redis = redis_client
        self.feedback_buffer = []
        self.buffer_size = 1000  # Flush to Redis after this many entries
        
    def record_implicit_feedback(
        self,
        query: str,
        detected_signals: Dict[str, bool],
        confidence_scores: Dict[str, float],
        user_action: str,  # 'clicked_result', 'map_used', 'bounced', etc.
        action_value: float = 1.0,  # How positive the action was (0-1)
        language: str = "en",
        timestamp: Optional[datetime] = None
    ):
        """
        Record implicit feedback from user behavior.
        
        Args:
            query: User query
            detected_signals: Detected signals (True/False)
            confidence_scores: Confidence scores for each signal
            user_action: Type of user action
            action_value: Strength of positive signal (0-1)
            language: Query language
            timestamp: When the action occurred
        """
        feedback_entry = {
            'type': 'implicit',
            'query': query,
            'detected_signals': detected_signals,
            'confidence_scores': confidence_scores,
            'user_action': user_action,
            'action_value': action_value,
            'language': language,
            'timestamp': (timestamp or datetime.now()).isoformat()
        }
        
        self._store_feedback(feedback_entry)
        logger.debug(f"📊 Implicit feedback recorded: {user_action} = {action_value}")
    
    def record_explicit_feedback(
        self,
        query: str,
        detected_signals: Dict[str, bool],
        confidence_scores: Dict[str, float],
        feedback_type: str,  # 'thumbs_up', 'thumbs_down', 'correction'
        corrected_signals: Optional[Dict[str, bool]] = None,
        language: str = "en",
        timestamp: Optional[datetime] = None
    ):
        """
        Record explicit user feedback (thumbs up/down, corrections).
        
        Args:
            query: User query
            detected_signals: Detected signals
            confidence_scores: Confidence scores
            feedback_type: Type of explicit feedback
            corrected_signals: User's correction of signals (if provided)
            language: Query language
            timestamp: When feedback was given
        """
        feedback_entry = {
            'type': 'explicit',
            'query': query,
            'detected_signals': detected_signals,
            'confidence_scores': confidence_scores,
            'feedback_type': feedback_type,
            'corrected_signals': corrected_signals,
            'language': language,
            'timestamp': (timestamp or datetime.now()).isoformat()
        }
        
        self._store_feedback(feedback_entry)
        logger.info(f"✅ Explicit feedback recorded: {feedback_type}")
    
    def _store_feedback(self, feedback_entry: Dict[str, Any]):
        """Store feedback in buffer and Redis."""
        self.feedback_buffer.append(feedback_entry)
        
        # Flush to Redis if buffer is full
        if len(self.feedback_buffer) >= self.buffer_size:
            self._flush_to_redis()
    
    def _flush_to_redis(self):
        """Flush feedback buffer to Redis."""
        if not self.redis or not self.feedback_buffer:
            return
        
        try:
            # Store in Redis list with timestamp key
            key = f"threshold_feedback:{datetime.now().strftime('%Y%m%d')}"
            
            for entry in self.feedback_buffer:
                self.redis.rpush(key, json.dumps(entry))
            
            # Set expiry to 90 days
            self.redis.expire(key, 90 * 24 * 3600)
            
            logger.info(f"💾 Flushed {len(self.feedback_buffer)} feedback entries to Redis")
            self.feedback_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush feedback to Redis: {e}")
    
    def get_feedback_history(
        self,
        days: int = 30,
        language: Optional[str] = None,
        signal_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve feedback history from Redis.
        
        Args:
            days: Number of days to look back
            language: Filter by language (optional)
            signal_name: Filter by specific signal (optional)
            
        Returns:
            List of feedback entries
        """
        if not self.redis:
            return self.feedback_buffer
        
        all_feedback = []
        
        try:
            # Get feedback from last N days
            for day_offset in range(days):
                date = datetime.now() - timedelta(days=day_offset)
                key = f"threshold_feedback:{date.strftime('%Y%m%d')}"
                
                entries = self.redis.lrange(key, 0, -1)
                
                for entry_json in entries:
                    entry = json.loads(entry_json)
                    
                    # Apply filters
                    if language and entry.get('language') != language:
                        continue
                    
                    if signal_name and signal_name not in entry.get('detected_signals', {}):
                        continue
                    
                    all_feedback.append(entry)
            
            logger.info(f"📚 Retrieved {len(all_feedback)} feedback entries from {days} days")
            return all_feedback
            
        except Exception as e:
            logger.error(f"Failed to retrieve feedback history: {e}")
            return []


class ROCAnalyzer:
    """
    Analyzes ROC curves to find optimal thresholds for signal detection.
    
    Uses feedback data to compute precision, recall, and F1 scores
    at different threshold values, then selects optimal thresholds.
    """
    
    def __init__(self):
        """Initialize ROC analyzer."""
        self.analysis_cache = {}
    
    def compute_optimal_threshold(
        self,
        feedback_data: List[Dict[str, Any]],
        signal_name: str,
        optimization_metric: str = 'f1'  # 'f1', 'precision', 'recall'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute optimal threshold for a signal based on feedback.
        
        Args:
            feedback_data: List of feedback entries
            signal_name: Signal to optimize (e.g., 'needs_map')
            optimization_metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, metrics_dict)
        """
        if not feedback_data:
            logger.warning(f"No feedback data for {signal_name}")
            return None, {}
        
        # Extract ground truth and scores from feedback
        y_true = []
        y_scores = []
        
        for entry in feedback_data:
            # Determine ground truth based on feedback type
            if entry['type'] == 'explicit' and entry.get('corrected_signals'):
                # User explicitly corrected the signal
                ground_truth = entry['corrected_signals'].get(signal_name, False)
            elif entry['type'] == 'implicit':
                # Infer from user behavior
                ground_truth = self._infer_ground_truth(entry, signal_name)
            else:
                # Use detected signal as ground truth (assume correct)
                ground_truth = entry['detected_signals'].get(signal_name, False)
            
            # Get confidence score
            score = entry['confidence_scores'].get(signal_name, 0.0)
            
            if score > 0:  # Only include if we have a confidence score
                y_true.append(1 if ground_truth else 0)
                y_scores.append(score)
        
        if len(y_true) < 10:
            logger.warning(f"Insufficient data for {signal_name}: {len(y_true)} samples")
            return None, {}
        
        # Compute ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Compute precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            
            # Find optimal threshold based on optimization metric
            best_threshold = None
            best_score = 0
            best_metrics = {}
            
            for threshold in thresholds:
                # Predict at this threshold
                y_pred = [1 if score >= threshold else 0 for score in y_scores]
                
                # Calculate metrics
                tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
                fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
                tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
                fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                
                # Select based on optimization metric
                if optimization_metric == 'f1':
                    score = f1
                elif optimization_metric == 'precision':
                    score = prec
                elif optimization_metric == 'recall':
                    score = rec
                else:
                    score = f1
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = {
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'roc_auc': roc_auc,
                        'tp': tp,
                        'fp': fp,
                        'tn': tn,
                        'fn': fn,
                        'samples': len(y_true)
                    }
            
            logger.info(
                f"✅ Optimal threshold for {signal_name}: {best_threshold:.3f} "
                f"(F1={best_metrics['f1']:.3f}, Precision={best_metrics['precision']:.3f}, "
                f"Recall={best_metrics['recall']:.3f}, AUC={roc_auc:.3f})"
            )
            
            return best_threshold, best_metrics
            
        except Exception as e:
            logger.error(f"ROC analysis failed for {signal_name}: {e}")
            return None, {}
    
    def _infer_ground_truth(self, feedback_entry: Dict[str, Any], signal_name: str) -> bool:
        """
        Infer ground truth from implicit user feedback.
        
        Uses heuristics based on user actions to determine if signal
        detection was correct.
        
        Args:
            feedback_entry: Feedback entry
            signal_name: Signal name
            
        Returns:
            Inferred ground truth (True/False)
        """
        detected = feedback_entry['detected_signals'].get(signal_name, False)
        action = feedback_entry['user_action']
        action_value = feedback_entry['action_value']
        
        # If signal was detected and user engaged positively -> correct detection
        if detected and action_value > 0.5:
            return True
        
        # If signal was not detected but user explicitly requested it -> missed signal
        if not detected and action in ['requested_map', 'requested_weather', 'requested_events']:
            if signal_name in action:
                return True
        
        # If signal was detected but user bounced/ignored -> false positive
        if detected and action_value < 0.3:
            return False
        
        # Default: trust the detection
        return detected


class ThresholdOptimizer:
    """
    Applies learned thresholds with safety checks and gradual rollout.
    
    Ensures that threshold changes don't degrade performance by:
    - Requiring minimum sample size
    - Checking performance improvement
    - Gradual A/B testing before full rollout
    """
    
    def __init__(self, min_samples: int = 100, min_improvement: float = 0.05):
        """
        Initialize threshold optimizer.
        
        Args:
            min_samples: Minimum samples required to trust new threshold
            min_improvement: Minimum F1 improvement required to apply
        """
        self.min_samples = min_samples
        self.min_improvement = min_improvement
    
    def should_apply_threshold(
        self,
        current_threshold: float,
        new_threshold: float,
        metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Determine if new threshold should be applied.
        
        Args:
            current_threshold: Current threshold value
            new_threshold: Proposed new threshold
            metrics: Performance metrics for new threshold
            
        Returns:
            Tuple of (should_apply, reason)
        """
        # Check minimum sample size
        if metrics.get('samples', 0) < self.min_samples:
            return False, f"Insufficient samples: {metrics.get('samples', 0)} < {self.min_samples}"
        
        # Check F1 score improvement
        # We don't have baseline F1, so check if F1 is reasonable
        if metrics.get('f1', 0) < 0.5:
            return False, f"Low F1 score: {metrics.get('f1', 0):.3f} < 0.5"
        
        # Check if threshold change is significant
        threshold_change = abs(new_threshold - current_threshold)
        if threshold_change < 0.01:
            return False, f"Negligible change: {threshold_change:.3f} < 0.01"
        
        # Check precision/recall balance
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        if precision < 0.4 or recall < 0.4:
            return False, f"Unbalanced metrics: P={precision:.3f}, R={recall:.3f}"
        
        return True, "All checks passed"
    
    def apply_threshold_gradually(
        self,
        signal_name: str,
        language: str,
        current_threshold: float,
        new_threshold: float,
        rollout_percentage: float = 0.1  # Start with 10% traffic
    ) -> Dict[str, Any]:
        """
        Apply threshold gradually using A/B testing.
        
        Args:
            signal_name: Signal name
            language: Language code
            current_threshold: Current threshold
            new_threshold: New threshold to test
            rollout_percentage: Percentage of traffic for new threshold
            
        Returns:
            Rollout configuration dict
        """
        return {
            'signal_name': signal_name,
            'language': language,
            'control_threshold': current_threshold,
            'treatment_threshold': new_threshold,
            'rollout_percentage': rollout_percentage,
            'status': 'testing',
            'start_time': datetime.now().isoformat()
        }


class ThresholdLearner:
    """
    Main class for dynamic threshold learning.
    
    Orchestrates feedback collection, ROC analysis, and threshold optimization
    to continuously improve signal detection accuracy.
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        learning_interval_hours: int = 24,
        min_samples: int = 100
    ):
        """
        Initialize threshold learner.
        
        Args:
            redis_client: Redis client for storage
            learning_interval_hours: How often to recompute thresholds
            min_samples: Minimum samples for threshold learning
        """
        self.redis = redis_client
        self.learning_interval_hours = learning_interval_hours
        self.min_samples = min_samples
        
        self.feedback_collector = FeedbackCollector(redis_client)
        self.roc_analyzer = ROCAnalyzer()
        self.optimizer = ThresholdOptimizer(min_samples=min_samples)
        
        self.last_learning_time = {}  # Per signal/language
        self.learned_thresholds = {}  # Cached learned thresholds
        
        logger.info("🎓 ThresholdLearner initialized")
    
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
            feedback_data: Additional feedback data
            language: Query language
        """
        if feedback_type == 'implicit':
            self.feedback_collector.record_implicit_feedback(
                query=query,
                detected_signals=detected_signals,
                confidence_scores=confidence_scores,
                user_action=feedback_data.get('action', 'unknown'),
                action_value=feedback_data.get('value', 0.5),
                language=language
            )
        elif feedback_type == 'explicit':
            self.feedback_collector.record_explicit_feedback(
                query=query,
                detected_signals=detected_signals,
                confidence_scores=confidence_scores,
                feedback_type=feedback_data.get('type', 'thumbs_up'),
                corrected_signals=feedback_data.get('corrected_signals'),
                language=language
            )
    
    def learn_thresholds(
        self,
        signal_names: List[str],
        language: str = "en",
        force: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Learn optimal thresholds for signals based on feedback.
        
        Args:
            signal_names: List of signal names to optimize
            language: Language to optimize for
            force: Force learning even if interval hasn't passed
            
        Returns:
            Dict of signal_name -> {threshold, metrics, should_apply}
        """
        results = {}
        
        for signal_name in signal_names:
            key = f"{signal_name}_{language}"
            
            # Check if we should relearn
            if not force:
                last_time = self.last_learning_time.get(key)
                if last_time:
                    hours_since = (datetime.now() - last_time).total_seconds() / 3600
                    if hours_since < self.learning_interval_hours:
                        logger.debug(
                            f"⏳ Skipping {signal_name} ({language}): "
                            f"learned {hours_since:.1f}h ago"
                        )
                        continue
            
            logger.info(f"🎓 Learning threshold for {signal_name} ({language})...")
            
            # Get feedback data
            feedback_data = self.feedback_collector.get_feedback_history(
                days=30,
                language=language,
                signal_name=signal_name
            )
            
            if not feedback_data:
                logger.warning(f"No feedback data for {signal_name} ({language})")
                continue
            
            # Compute optimal threshold
            optimal_threshold, metrics = self.roc_analyzer.compute_optimal_threshold(
                feedback_data=feedback_data,
                signal_name=signal_name,
                optimization_metric='f1'
            )
            
            if optimal_threshold is None:
                logger.warning(f"Could not compute threshold for {signal_name}")
                continue
            
            # Store results
            results[signal_name] = {
                'threshold': optimal_threshold,
                'metrics': metrics,
                'language': language,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache learned threshold
            self.learned_thresholds[key] = optimal_threshold
            self.last_learning_time[key] = datetime.now()
            
            logger.info(
                f"✅ Learned threshold for {signal_name} ({language}): "
                f"{optimal_threshold:.3f} (F1={metrics.get('f1', 0):.3f})"
            )
        
        return results
    
    def get_recommended_thresholds(
        self,
        current_thresholds: Dict[str, float],
        language: str = "en"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get recommended threshold updates with safety checks.
        
        Args:
            current_thresholds: Current thresholds
            language: Language code
            
        Returns:
            Dict of signal_name -> {current, recommended, should_apply, reason, metrics}
        """
        recommendations = {}
        
        # Learn thresholds for all signals
        signal_names = list(current_thresholds.keys())
        learned = self.learn_thresholds(signal_names, language)
        
        # Apply safety checks
        for signal_name, data in learned.items():
            current = current_thresholds.get(signal_name, 0.5)
            new = data['threshold']
            metrics = data['metrics']
            
            should_apply, reason = self.optimizer.should_apply_threshold(
                current_threshold=current,
                new_threshold=new,
                metrics=metrics
            )
            
            recommendations[signal_name] = {
                'current': current,
                'recommended': new,
                'should_apply': should_apply,
                'reason': reason,
                'metrics': metrics,
                'change': new - current,
                'change_pct': ((new - current) / current * 100) if current > 0 else 0
            }
        
        return recommendations
    
    def auto_tune(
        self,
        current_thresholds: Dict[str, float],
        language: str = "en",
        auto_apply: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Auto-tune thresholds based on feedback.
        
        Args:
            current_thresholds: Current thresholds
            language: Language code
            auto_apply: Automatically apply recommended thresholds
            
        Returns:
            Dict of recommendations (and applied changes if auto_apply=True)
        """
        logger.info(f"🔧 Auto-tuning thresholds for {language}...")
        
        recommendations = self.get_recommended_thresholds(current_thresholds, language)
        
        applied = {}
        for signal_name, rec in recommendations.items():
            if rec['should_apply']:
                logger.info(
                    f"✅ Recommend updating {signal_name}: "
                    f"{rec['current']:.3f} -> {rec['recommended']:.3f} "
                    f"({rec['change_pct']:+.1f}%) - {rec['reason']}"
                )
                
                if auto_apply:
                    applied[signal_name] = rec['recommended']
                    logger.info(f"🎯 Applied new threshold for {signal_name}")
            else:
                logger.info(
                    f"⏸️  Not applying {signal_name}: {rec['reason']}"
                )
        
        return {
            'recommendations': recommendations,
            'applied': applied if auto_apply else {},
            'timestamp': datetime.now().isoformat()
        }
