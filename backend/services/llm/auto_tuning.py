"""
Auto-Tuning Module for Signal Detection

This module provides:
1. Automatic threshold adjustment based on feedback
2. Precision/Recall/F1 optimization
3. Scheduled tuning (daily/weekly)
4. A/B testing for threshold variants

Author: AI Istanbul Team
Date: January 2025
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class SignalMetrics:
    """Metrics for a specific signal"""
    signal_name: str
    
    # Confusion matrix
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Current threshold
    current_threshold: float = 0.7
    
    # Calculated metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    samples_count: int = 0
    
    def calculate_metrics(self):
        """Calculate precision, recall, and F1 score"""
        # Precision: TP / (TP + FP)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0
        
        # Recall: TP / (TP + FN)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0
        
        # F1: 2 * (precision * recall) / (precision + recall)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0
        
        self.samples_count = (
            self.true_positives + self.false_positives +
            self.true_negatives + self.false_negatives
        )


@dataclass
class ThresholdAdjustment:
    """Record of threshold adjustment"""
    signal_name: str
    old_threshold: float
    new_threshold: float
    reason: str
    metrics_before: SignalMetrics
    timestamp: datetime = field(default_factory=datetime.now)


class AutoTuner:
    """
    Automatic threshold tuning for signal detection.
    
    Features:
    - Monitor signal accuracy from feedback
    - Adjust thresholds to optimize F1 score
    - Scheduled tuning runs
    - A/B testing for threshold variants
    """
    
    def __init__(
        self,
        signal_detector,
        personalization_engine,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize auto-tuner.
        
        Args:
            signal_detector: Signal detector instance
            personalization_engine: Personalization engine for feedback
            config: Configuration dictionary
        """
        self.signal_detector = signal_detector
        self.personalization_engine = personalization_engine
        self.config = config or {}
        
        # Metrics tracking
        self.signal_metrics: Dict[str, SignalMetrics] = {}
        self.adjustment_history: List[ThresholdAdjustment] = []
        
        # Tuning configuration
        self.min_samples_for_tuning = self.config.get('min_samples', 50)
        self.target_f1_score = self.config.get('target_f1', 0.8)
        self.max_threshold_change = self.config.get('max_threshold_change', 0.1)
        self.tuning_schedule = self.config.get('tuning_schedule', 'daily')  # daily or weekly
        
        # Tuning state
        self.is_tuning_enabled = self.config.get('enable_tuning', True)
        self.last_tuning_run: Optional[datetime] = None
        
        logger.info("✅ Auto-Tuner initialized")
    
    async def update_metrics_from_feedback(
        self,
        signal: str,
        was_detected: bool,
        should_be_detected: bool
    ):
        """
        Update signal metrics based on feedback.
        
        Args:
            signal: Signal name
            was_detected: Whether signal was detected by system
            should_be_detected: Ground truth from feedback
        """
        if signal not in self.signal_metrics:
            self.signal_metrics[signal] = SignalMetrics(signal_name=signal)
        
        metrics = self.signal_metrics[signal]
        
        # Update confusion matrix
        if was_detected and should_be_detected:
            metrics.true_positives += 1
        elif was_detected and not should_be_detected:
            metrics.false_positives += 1
        elif not was_detected and should_be_detected:
            metrics.false_negatives += 1
        else:  # not was_detected and not should_be_detected
            metrics.true_negatives += 1
        
        # Recalculate metrics
        metrics.calculate_metrics()
        metrics.last_updated = datetime.now()
        
        logger.debug(
            f"Updated metrics for {signal}: "
            f"P={metrics.precision:.2f}, R={metrics.recall:.2f}, F1={metrics.f1_score:.2f}"
        )
    
    async def run_auto_tuning(
        self,
        signals_to_tune: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run automatic threshold tuning.
        
        Args:
            signals_to_tune: List of signals to tune (None = all signals)
            
        Returns:
            Tuning report with adjustments made
        """
        if not self.is_tuning_enabled:
            logger.info("Auto-tuning is disabled")
            return {'status': 'disabled'}
        
        logger.info("🎯 Starting auto-tuning run...")
        
        # Determine which signals to tune
        if signals_to_tune is None:
            signals_to_tune = list(self.signal_metrics.keys())
        
        adjustments_made = []
        
        for signal in signals_to_tune:
            if signal not in self.signal_metrics:
                continue
            
            metrics = self.signal_metrics[signal]
            
            # Skip if not enough samples
            if metrics.samples_count < self.min_samples_for_tuning:
                logger.debug(
                    f"Skipping {signal}: only {metrics.samples_count} samples "
                    f"(need {self.min_samples_for_tuning})"
                )
                continue
            
            # Decide if adjustment is needed
            adjustment = self._calculate_threshold_adjustment(metrics)
            
            if adjustment:
                # Apply adjustment
                old_threshold = metrics.current_threshold
                new_threshold = adjustment
                
                # Update signal detector threshold
                await self._apply_threshold_change(signal, new_threshold)
                
                # Record adjustment
                adjustment_record = ThresholdAdjustment(
                    signal_name=signal,
                    old_threshold=old_threshold,
                    new_threshold=new_threshold,
                    reason=self._get_adjustment_reason(metrics),
                    metrics_before=metrics
                )
                
                self.adjustment_history.append(adjustment_record)
                adjustments_made.append(adjustment_record)
                
                # Update metrics threshold
                metrics.current_threshold = new_threshold
                
                logger.info(
                    f"✅ Adjusted {signal} threshold: "
                    f"{old_threshold:.3f} → {new_threshold:.3f} "
                    f"(F1: {metrics.f1_score:.3f})"
                )
        
        self.last_tuning_run = datetime.now()
        
        return {
            'status': 'complete',
            'timestamp': self.last_tuning_run.isoformat(),
            'signals_evaluated': len(signals_to_tune),
            'adjustments_made': len(adjustments_made),
            'adjustments': [
                {
                    'signal': adj.signal_name,
                    'old_threshold': adj.old_threshold,
                    'new_threshold': adj.new_threshold,
                    'reason': adj.reason
                }
                for adj in adjustments_made
            ]
        }
    
    def _calculate_threshold_adjustment(
        self,
        metrics: SignalMetrics
    ) -> Optional[float]:
        """
        Calculate new threshold based on metrics.
        
        Strategy:
        - If F1 < target and precision < 0.5: increase threshold (reduce FP)
        - If F1 < target and recall < 0.5: decrease threshold (reduce FN)
        - If F1 >= target: no change
        
        Args:
            metrics: Current signal metrics
            
        Returns:
            New threshold or None if no change needed
        """
        current = metrics.current_threshold
        
        # No change if F1 is good enough
        if metrics.f1_score >= self.target_f1_score:
            return None
        
        # Calculate adjustment
        adjustment_size = 0.05  # 5% change
        
        if metrics.precision < 0.5:
            # Too many false positives - increase threshold
            new_threshold = min(current + adjustment_size, 0.95)
        elif metrics.recall < 0.5:
            # Too many false negatives - decrease threshold
            new_threshold = max(current - adjustment_size, 0.3)
        elif metrics.precision < metrics.recall:
            # Precision lower than recall - increase threshold slightly
            new_threshold = current + (adjustment_size / 2)
        elif metrics.recall < metrics.precision:
            # Recall lower than precision - decrease threshold slightly
            new_threshold = current - (adjustment_size / 2)
        else:
            # Balanced but low F1 - try small decrease to improve recall
            new_threshold = current - (adjustment_size / 3)
        
        # Ensure change is not too large
        if abs(new_threshold - current) > self.max_threshold_change:
            new_threshold = current + (
                self.max_threshold_change if new_threshold > current
                else -self.max_threshold_change
            )
        
        # Only return if actually different
        if abs(new_threshold - current) < 0.01:
            return None
        
        return round(new_threshold, 3)
    
    def _get_adjustment_reason(self, metrics: SignalMetrics) -> str:
        """Generate human-readable reason for adjustment"""
        if metrics.f1_score >= self.target_f1_score:
            return "F1 score above target"
        elif metrics.precision < 0.5:
            return f"Low precision ({metrics.precision:.2f}) - reducing false positives"
        elif metrics.recall < 0.5:
            return f"Low recall ({metrics.recall:.2f}) - reducing false negatives"
        elif metrics.precision < metrics.recall:
            return "Optimizing precision"
        else:
            return "Optimizing recall"
    
    async def _apply_threshold_change(self, signal: str, new_threshold: float):
        """Apply threshold change to signal detector"""
        # Update threshold in signal detector
        if hasattr(self.signal_detector, 'language_thresholds'):
            for lang in self.signal_detector.language_thresholds:
                if signal in self.signal_detector.language_thresholds[lang]:
                    self.signal_detector.language_thresholds[lang][signal] = new_threshold
        
        logger.debug(f"Applied threshold {new_threshold} to {signal}")
    
    async def start_scheduled_tuning(self):
        """Start scheduled auto-tuning based on configuration"""
        logger.info(f"📅 Starting scheduled auto-tuning ({self.tuning_schedule})")
        
        while self.is_tuning_enabled:
            try:
                # Run tuning
                result = await self.run_auto_tuning()
                logger.info(f"Scheduled tuning complete: {result.get('adjustments_made', 0)} adjustments")
                
                # Wait for next run
                if self.tuning_schedule == 'daily':
                    await asyncio.sleep(86400)  # 24 hours
                elif self.tuning_schedule == 'weekly':
                    await asyncio.sleep(86400 * 7)  # 7 days
                else:
                    # Default to daily
                    await asyncio.sleep(86400)
                    
            except asyncio.CancelledError:
                logger.info("Scheduled tuning cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scheduled tuning: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def get_tuning_report(self) -> Dict[str, Any]:
        """
        Get comprehensive tuning report.
        
        Returns:
            Report with metrics and adjustment history
        """
        # Calculate aggregate metrics
        total_signals = len(self.signal_metrics)
        signals_meeting_target = sum(
            1 for m in self.signal_metrics.values()
            if m.f1_score >= self.target_f1_score
        )
        
        avg_f1 = (
            statistics.mean(m.f1_score for m in self.signal_metrics.values())
            if self.signal_metrics else 0.0
        )
        
        avg_precision = (
            statistics.mean(m.precision for m in self.signal_metrics.values())
            if self.signal_metrics else 0.0
        )
        
        avg_recall = (
            statistics.mean(m.recall for m in self.signal_metrics.values())
            if self.signal_metrics else 0.0
        )
        
        # Recent adjustments
        recent_adjustments = [
            adj for adj in self.adjustment_history
            if adj.timestamp >= datetime.now() - timedelta(days=7)
        ]
        
        return {
            'overall_metrics': {
                'total_signals': total_signals,
                'signals_meeting_target': signals_meeting_target,
                'target_achievement_rate': signals_meeting_target / total_signals if total_signals > 0 else 0.0,
                'avg_f1_score': avg_f1,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'target_f1': self.target_f1_score
            },
            'signal_metrics': {
                name: {
                    'threshold': metrics.current_threshold,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'samples': metrics.samples_count
                }
                for name, metrics in self.signal_metrics.items()
            },
            'tuning_status': {
                'enabled': self.is_tuning_enabled,
                'last_run': self.last_tuning_run.isoformat() if self.last_tuning_run else None,
                'schedule': self.tuning_schedule,
                'total_adjustments': len(self.adjustment_history),
                'recent_adjustments': len(recent_adjustments)
            },
            'recent_adjustments': [
                {
                    'signal': adj.signal_name,
                    'timestamp': adj.timestamp.isoformat(),
                    'old_threshold': adj.old_threshold,
                    'new_threshold': adj.new_threshold,
                    'reason': adj.reason,
                    'f1_before': adj.metrics_before.f1_score
                }
                for adj in recent_adjustments[-10:]  # Last 10 adjustments
            ]
        }
    
    def reset_metrics(self, signal: Optional[str] = None):
        """Reset metrics for a signal or all signals"""
        if signal:
            if signal in self.signal_metrics:
                self.signal_metrics[signal] = SignalMetrics(signal_name=signal)
                logger.info(f"Reset metrics for {signal}")
        else:
            self.signal_metrics.clear()
            logger.info("Reset all signal metrics")
    
    def enable_tuning(self):
        """Enable auto-tuning"""
        self.is_tuning_enabled = True
        logger.info("Auto-tuning enabled")
    
    def disable_tuning(self):
        """Disable auto-tuning"""
        self.is_tuning_enabled = False
        logger.info("Auto-tuning disabled")
