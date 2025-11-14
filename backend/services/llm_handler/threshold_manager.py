"""
Threshold Manager
Dynamic threshold learning and management

Responsibilities:
- Threshold learning integration
- Auto-tuning logic
- Feedback recording
- Per-language threshold management

Author: Istanbul AI Team
Date: November 14, 2025
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ThresholdManager:
    """
    Manages threshold learning and optimization
    
    Features:
    - Dynamic threshold learning
    - Feedback recording
    - Auto-tuning
    - Per-language optimization
    """
    
    def __init__(
        self,
        redis_client=None,
        language_thresholds=None,
        enable_auto_tuning: bool = True,
        auto_tune_interval_hours: int = 24,
        min_samples: int = 100
    ):
        """
        Initialize threshold manager
        
        Args:
            redis_client: Redis client for persistence
            language_thresholds: Initial per-language thresholds
            enable_auto_tuning: Enable automatic threshold tuning
            auto_tune_interval_hours: Hours between auto-tune runs
            min_samples: Minimum samples required for tuning
        """
        self.redis = redis_client
        self.thresholds = language_thresholds or {}
        self.enable_auto_tuning = enable_auto_tuning
        self.auto_tune_interval_hours = auto_tune_interval_hours
        self.min_samples = min_samples
        
        # Initialize threshold learner (optional dependency)
        self.learner = None
        self._init_threshold_learner()
        
        # Track last auto-tune time per language
        self.last_auto_tune = {}
        
        # Statistics
        self.stats = {
            "feedback_recorded": 0,
            "auto_tunes_performed": 0,
            "thresholds_updated": 0,
            "errors": 0
        }
        
        logger.info("✅ Threshold manager initialized")
        logger.info(f"   Auto-tuning: {'✅ Enabled' if self.enable_auto_tuning else '❌ Disabled'}")
        logger.info(f"   Interval: {auto_tune_interval_hours} hours")
        logger.info(f"   Min samples: {min_samples}")
    
    def _init_threshold_learner(self):
        """Initialize threshold learner if available"""
        try:
            from backend.services.threshold_learner import ThresholdLearner
            
            self.learner = ThresholdLearner(
                redis_client=self.redis,
                learning_interval_hours=self.auto_tune_interval_hours,
                min_samples=self.min_samples
            )
            
            logger.info("   🎓 Threshold learner initialized")
        except ImportError:
            logger.warning("   ⚠️ ThresholdLearner not available - using static thresholds")
            self.learner = None
        except Exception as e:
            logger.error(f"   ❌ Failed to initialize threshold learner: {e}")
            self.learner = None
    
    def get_threshold(
        self,
        signal_name: str,
        language: str = "en"
    ) -> float:
        """
        Get threshold for a signal and language.
        
        Args:
            signal_name: Name of the signal (e.g., 'needs_map')
            language: Language code (en, tr, etc.)
            
        Returns:
            Threshold value (float between 0 and 1)
        """
        # Try to get language-specific threshold
        if language in self.thresholds:
            lang_thresholds = self.thresholds[language]
            if signal_name in lang_thresholds:
                return lang_thresholds[signal_name]
        
        # Fall back to English defaults
        if "en" in self.thresholds:
            en_thresholds = self.thresholds["en"]
            if signal_name in en_thresholds:
                return en_thresholds[signal_name]
        
        # Ultimate fallback
        default_thresholds = {
            "needs_map": 0.35,
            "needs_gps_routing": 0.48,
            "needs_weather": 0.33,
            "needs_events": 0.38,
            "needs_hidden_gems": 0.30,
            "has_budget_constraint": 0.38,
            "likely_restaurant": 0.33,
            "likely_attraction": 0.28
        }
        
        return default_thresholds.get(signal_name, 0.35)
    
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
            query: Original user query
            detected_signals: Dict of signals that were detected
            confidence_scores: Dict of confidence scores for each signal
            feedback_type: Type of feedback (click, thumbs_up, thumbs_down, etc.)
            feedback_data: Additional feedback data
            language: Query language
        """
        if not self.learner:
            return
        
        try:
            self.stats["feedback_recorded"] += 1
            
            # Record feedback for each detected signal
            for signal_name, detected in detected_signals.items():
                confidence = confidence_scores.get(signal_name, 0.0)
                
                # Determine if feedback is positive or negative
                is_positive = feedback_type in ["click", "thumbs_up", "engagement"]
                
                # Record to learner
                self.learner.record_feedback(
                    signal_name=signal_name,
                    confidence_score=confidence,
                    was_correct=is_positive,
                    language=language,
                    metadata={
                        "query": query,
                        "feedback_type": feedback_type,
                        **feedback_data
                    }
                )
            
            logger.debug(f"✅ Recorded feedback for {len(detected_signals)} signals")
            
        except Exception as e:
            logger.error(f"❌ Failed to record feedback: {e}")
            self.stats["errors"] += 1
    
    async def auto_tune(
        self,
        language: str = "en",
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Auto-tune thresholds based on collected feedback.
        
        Args:
            language: Language to tune thresholds for
            force: Force tuning even if interval hasn't passed
            
        Returns:
            Dict with tuning results
        """
        if not self.learner or not self.enable_auto_tuning:
            return {
                "status": "disabled",
                "message": "Auto-tuning not available"
            }
        
        try:
            # Check if enough time has passed since last tune
            if not force:
                last_tune = self.last_auto_tune.get(language)
                if last_tune:
                    hours_since = (datetime.now() - last_tune).total_seconds() / 3600
                    if hours_since < self.auto_tune_interval_hours:
                        return {
                            "status": "skipped",
                            "message": f"Too soon since last tune ({hours_since:.1f}h ago)",
                            "next_tune_in_hours": self.auto_tune_interval_hours - hours_since
                        }
            
            # Perform tuning
            logger.info(f"🎓 Auto-tuning thresholds for language: {language}")
            
            tuning_result = await self.learner.tune_thresholds(language=language)
            
            if tuning_result.get("status") == "success":
                # Update thresholds
                new_thresholds = tuning_result.get("thresholds", {})
                
                if language not in self.thresholds:
                    self.thresholds[language] = {}
                
                self.thresholds[language].update(new_thresholds)
                
                # Update last tune time
                self.last_auto_tune[language] = datetime.now()
                
                # Update stats
                self.stats["auto_tunes_performed"] += 1
                self.stats["thresholds_updated"] += len(new_thresholds)
                
                logger.info(f"✅ Auto-tuning complete: updated {len(new_thresholds)} thresholds")
                
                return {
                    "status": "success",
                    "language": language,
                    "thresholds_updated": len(new_thresholds),
                    "new_thresholds": new_thresholds,
                    "metrics": tuning_result.get("metrics", {})
                }
            else:
                return tuning_result
            
        except Exception as e:
            logger.error(f"❌ Auto-tuning failed: {e}")
            self.stats["errors"] += 1
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get threshold manager statistics.
        
        Returns:
            Dict with performance metrics
        """
        return {
            **self.stats,
            "languages_configured": len(self.thresholds),
            "learner_available": self.learner is not None,
            "auto_tuning_enabled": self.enable_auto_tuning
        }
