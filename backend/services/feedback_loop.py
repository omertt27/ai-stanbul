"""
Real-time Feedback Loop System
Implements user satisfaction ratings, ML model retraining pipeline, and continuous improvement
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collects and manages user feedback and satisfaction ratings
    """
    
    def __init__(self):
        """Initialize feedback collector"""
        self.feedback_queue = deque(maxlen=10000)  # Recent feedback
        self.user_feedback = {}  # user_id -> feedback history
        self.interaction_feedback = {}  # interaction_id -> feedback
        self.aggregate_metrics = {
            'total_ratings': 0,
            'avg_satisfaction': 0.0,
            'satisfaction_by_intent': {},
            'satisfaction_by_feature': {},
            'last_updated': datetime.now().isoformat()
        }
        
        logger.info("✅ FeedbackCollector initialized")
    
    def submit_feedback(self, user_id: str, interaction_id: str, feedback_data: Dict[str, Any]):
        """
        Submit user feedback for an interaction
        
        Args:
            user_id: User identifier
            interaction_id: Unique interaction identifier
            feedback_data: Dict containing:
                - satisfaction_score: float (1-5)
                - was_helpful: bool
                - response_quality: float (1-5)
                - speed_rating: float (1-5)
                - intent: str (the intent type)
                - feature: str (the feature used)
                - comments: str (optional user comments)
                - issues: List[str] (optional list of issues)
        """
        # Validate satisfaction score
        satisfaction_score = feedback_data.get('satisfaction_score', 3.0)
        if not (1.0 <= satisfaction_score <= 5.0):
            logger.warning(f"Invalid satisfaction score: {satisfaction_score}")
            satisfaction_score = max(1.0, min(5.0, satisfaction_score))
        
        # Create feedback entry
        feedback_entry = {
            'user_id': user_id,
            'interaction_id': interaction_id,
            'timestamp': datetime.now().isoformat(),
            'satisfaction_score': satisfaction_score,
            'was_helpful': feedback_data.get('was_helpful', True),
            'response_quality': feedback_data.get('response_quality', 3.0),
            'speed_rating': feedback_data.get('speed_rating', 3.0),
            'intent': feedback_data.get('intent', 'unknown'),
            'feature': feedback_data.get('feature', 'general'),
            'comments': feedback_data.get('comments', ''),
            'issues': feedback_data.get('issues', [])
        }
        
        # Store in queue
        self.feedback_queue.append(feedback_entry)
        
        # Store by user
        if user_id not in self.user_feedback:
            self.user_feedback[user_id] = []
        self.user_feedback[user_id].append(feedback_entry)
        
        # Store by interaction
        self.interaction_feedback[interaction_id] = feedback_entry
        
        # Update aggregate metrics
        self._update_aggregate_metrics(feedback_entry)
        
        logger.info(f"Feedback recorded: user={user_id}, satisfaction={satisfaction_score}, intent={feedback_entry['intent']}")
    
    def _update_aggregate_metrics(self, feedback_entry: Dict[str, Any]):
        """Update aggregate metrics with new feedback"""
        # Update total ratings count
        self.aggregate_metrics['total_ratings'] += 1
        
        # Update average satisfaction (incremental calculation)
        old_avg = self.aggregate_metrics['avg_satisfaction']
        old_count = self.aggregate_metrics['total_ratings'] - 1
        new_avg = (old_avg * old_count + feedback_entry['satisfaction_score']) / self.aggregate_metrics['total_ratings']
        self.aggregate_metrics['avg_satisfaction'] = new_avg
        
        # Update satisfaction by intent
        intent = feedback_entry['intent']
        if intent not in self.aggregate_metrics['satisfaction_by_intent']:
            self.aggregate_metrics['satisfaction_by_intent'][intent] = {
                'count': 0,
                'avg_satisfaction': 0.0,
                'total_score': 0.0
            }
        
        intent_metrics = self.aggregate_metrics['satisfaction_by_intent'][intent]
        intent_metrics['count'] += 1
        intent_metrics['total_score'] += feedback_entry['satisfaction_score']
        intent_metrics['avg_satisfaction'] = intent_metrics['total_score'] / intent_metrics['count']
        
        # Update satisfaction by feature
        feature = feedback_entry['feature']
        if feature not in self.aggregate_metrics['satisfaction_by_feature']:
            self.aggregate_metrics['satisfaction_by_feature'][feature] = {
                'count': 0,
                'avg_satisfaction': 0.0,
                'total_score': 0.0
            }
        
        feature_metrics = self.aggregate_metrics['satisfaction_by_feature'][feature]
        feature_metrics['count'] += 1
        feature_metrics['total_score'] += feedback_entry['satisfaction_score']
        feature_metrics['avg_satisfaction'] = feature_metrics['total_score'] / feature_metrics['count']
        
        self.aggregate_metrics['last_updated'] = datetime.now().isoformat()
    
    def get_user_satisfaction_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get satisfaction history for a user"""
        return self.user_feedback.get(user_id, [])
    
    def get_interaction_feedback(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a specific interaction"""
        return self.interaction_feedback.get(interaction_id)
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate feedback metrics"""
        return self.aggregate_metrics.copy()
    
    def get_recent_feedback(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get most recent feedback entries"""
        return list(self.feedback_queue)[-count:]
    
    def get_low_satisfaction_intents(self, threshold: float = 3.0) -> List[Tuple[str, float]]:
        """
        Get intents with satisfaction below threshold
        
        Returns:
            List of (intent, avg_satisfaction) tuples
        """
        low_satisfaction = []
        
        for intent, metrics in self.aggregate_metrics['satisfaction_by_intent'].items():
            if metrics['avg_satisfaction'] < threshold and metrics['count'] >= 10:
                low_satisfaction.append((intent, metrics['avg_satisfaction']))
        
        # Sort by satisfaction (ascending)
        low_satisfaction.sort(key=lambda x: x[1])
        
        return low_satisfaction
    
    def get_improvement_opportunities(self) -> Dict[str, Any]:
        """Identify areas for improvement based on feedback"""
        opportunities = {
            'low_satisfaction_intents': [],
            'common_issues': defaultdict(int),
            'feature_improvements': [],
            'recommendations': []
        }
        
        # Analyze low satisfaction intents
        opportunities['low_satisfaction_intents'] = self.get_low_satisfaction_intents(threshold=3.5)
        
        # Analyze common issues
        for feedback in list(self.feedback_queue):
            for issue in feedback.get('issues', []):
                opportunities['common_issues'][issue] += 1
        
        # Sort common issues by frequency
        opportunities['common_issues'] = sorted(
            opportunities['common_issues'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Analyze feature improvements needed
        for feature, metrics in self.aggregate_metrics['satisfaction_by_feature'].items():
            if metrics['count'] >= 10 and metrics['avg_satisfaction'] < 3.5:
                opportunities['feature_improvements'].append({
                    'feature': feature,
                    'satisfaction': metrics['avg_satisfaction'],
                    'count': metrics['count']
                })
        
        # Sort by satisfaction (ascending)
        opportunities['feature_improvements'].sort(key=lambda x: x['satisfaction'])
        
        # Generate recommendations
        if opportunities['low_satisfaction_intents']:
            opportunities['recommendations'].append(
                f"Focus on improving {opportunities['low_satisfaction_intents'][0][0]} intent " +
                f"(satisfaction: {opportunities['low_satisfaction_intents'][0][1]:.2f})"
            )
        
        if opportunities['common_issues']:
            opportunities['recommendations'].append(
                f"Address most common issue: {opportunities['common_issues'][0][0]} " +
                f"({opportunities['common_issues'][0][1]} reports)"
            )
        
        return opportunities


class ModelRetrainingPipeline:
    """
    Manages ML model retraining based on feedback and performance
    """
    
    def __init__(self):
        """Initialize retraining pipeline"""
        self.retraining_queue = []  # Queue of retraining tasks
        self.model_versions = {}  # model_name -> versions list
        self.retraining_history = []  # History of retraining events
        self.performance_tracking = {}  # model_name -> performance metrics
        
        # Retraining thresholds
        self.thresholds = {
            'min_feedback_count': 100,  # Minimum feedback before retraining
            'satisfaction_drop_threshold': 0.5,  # Retrain if satisfaction drops by this much
            'accuracy_drop_threshold': 0.05,  # Retrain if accuracy drops by 5%
            'days_since_last_training': 7  # Retrain every week minimum
        }
        
        logger.info("✅ ModelRetrainingPipeline initialized")
    
    def check_retraining_needed(self, model_name: str, current_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if a model needs retraining
        
        Args:
            model_name: Name of the model
            current_metrics: Current performance metrics
        
        Returns:
            (needs_retraining, reason)
        """
        # Check if we have baseline performance
        if model_name not in self.performance_tracking:
            self.performance_tracking[model_name] = {
                'baseline': current_metrics,
                'history': [current_metrics],
                'last_training': datetime.now(),
                'feedback_since_training': 0
            }
            return (False, "Baseline established")
        
        tracking = self.performance_tracking[model_name]
        baseline = tracking['baseline']
        
        # Check satisfaction drop
        if 'satisfaction' in current_metrics and 'satisfaction' in baseline:
            satisfaction_drop = baseline['satisfaction'] - current_metrics['satisfaction']
            if satisfaction_drop > self.thresholds['satisfaction_drop_threshold']:
                return (True, f"Satisfaction dropped by {satisfaction_drop:.2f}")
        
        # Check accuracy drop
        if 'accuracy' in current_metrics and 'accuracy' in baseline:
            accuracy_drop = baseline['accuracy'] - current_metrics['accuracy']
            if accuracy_drop > self.thresholds['accuracy_drop_threshold']:
                return (True, f"Accuracy dropped by {accuracy_drop:.2%}")
        
        # Check time since last training
        days_since_training = (datetime.now() - tracking['last_training']).days
        if days_since_training >= self.thresholds['days_since_last_training']:
            return (True, f"Scheduled retraining ({days_since_training} days since last training)")
        
        # Check feedback count
        if tracking['feedback_since_training'] >= self.thresholds['min_feedback_count']:
            return (True, f"Sufficient feedback collected ({tracking['feedback_since_training']} samples)")
        
        return (False, "No retraining needed")
    
    def schedule_retraining(self, model_name: str, reason: str, priority: str = 'normal'):
        """
        Schedule a model for retraining
        
        Args:
            model_name: Name of the model to retrain
            reason: Reason for retraining
            priority: 'low', 'normal', or 'high'
        """
        task = {
            'model_name': model_name,
            'reason': reason,
            'priority': priority,
            'scheduled_at': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        self.retraining_queue.append(task)
        logger.info(f"Scheduled retraining for {model_name}: {reason} (priority: {priority})")
    
    def record_retraining_completion(self, model_name: str, new_metrics: Dict[str, float]):
        """Record that a model was retrained"""
        # Update performance tracking
        if model_name in self.performance_tracking:
            tracking = self.performance_tracking[model_name]
            tracking['baseline'] = new_metrics
            tracking['history'].append(new_metrics)
            tracking['last_training'] = datetime.now()
            tracking['feedback_since_training'] = 0
        
        # Record in history
        self.retraining_history.append({
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': new_metrics
        })
        
        logger.info(f"Retraining completed for {model_name}")
    
    def get_retraining_queue(self) -> List[Dict[str, Any]]:
        """Get current retraining queue"""
        return self.retraining_queue.copy()
    
    def get_retraining_history(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get retraining history, optionally filtered by model"""
        if model_name:
            return [h for h in self.retraining_history if h['model_name'] == model_name]
        return self.retraining_history.copy()


class ContinuousImprovementSystem:
    """
    Orchestrates continuous improvement based on feedback and retraining
    """
    
    def __init__(self):
        """Initialize continuous improvement system"""
        self.feedback_collector = FeedbackCollector()
        self.retraining_pipeline = ModelRetrainingPipeline()
        
        # Improvement tracking
        self.improvements = []  # List of implemented improvements
        self.experiment_results = {}  # A/B test results
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("🔄 Continuous Improvement System initialized")
    
    def submit_feedback(self, user_id: str, interaction_id: str, feedback_data: Dict[str, Any]):
        """Submit feedback and trigger improvement checks"""
        self.feedback_collector.submit_feedback(user_id, interaction_id, feedback_data)
        
        # Update model feedback count
        intent = feedback_data.get('intent', 'unknown')
        if intent in self.retraining_pipeline.performance_tracking:
            self.retraining_pipeline.performance_tracking[intent]['feedback_since_training'] += 1
    
    def analyze_feedback_and_schedule_improvements(self):
        """Analyze feedback and schedule improvements"""
        # Get improvement opportunities
        opportunities = self.feedback_collector.get_improvement_opportunities()
        
        # Check if retraining is needed for low-performing intents
        for intent, satisfaction in opportunities['low_satisfaction_intents']:
            needs_retraining, reason = self.retraining_pipeline.check_retraining_needed(
                intent,
                {'satisfaction': satisfaction}
            )
            
            if needs_retraining:
                self.retraining_pipeline.schedule_retraining(
                    intent,
                    reason,
                    priority='high'
                )
        
        return opportunities
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        metrics = self.feedback_collector.get_aggregate_metrics()
        opportunities = self.feedback_collector.get_improvement_opportunities()
        retraining_queue = self.retraining_pipeline.get_retraining_queue()
        
        # Calculate health score (0-100)
        health_score = 100.0
        
        # Deduct for low satisfaction
        if metrics['avg_satisfaction'] < 4.0:
            health_score -= (4.0 - metrics['avg_satisfaction']) * 20
        
        # Deduct for pending retraining tasks
        health_score -= len(retraining_queue) * 5
        
        # Deduct for common issues
        health_score -= len(opportunities.get('common_issues', [])) * 2
        
        health_score = max(0.0, min(100.0, health_score))
        
        return {
            'health_score': health_score,
            'avg_satisfaction': metrics['avg_satisfaction'],
            'total_ratings': metrics['total_ratings'],
            'pending_retraining': len(retraining_queue),
            'improvement_opportunities': len(opportunities['low_satisfaction_intents']),
            'status': 'healthy' if health_score >= 80 else 'needs_attention' if health_score >= 60 else 'critical',
            'last_updated': datetime.now().isoformat()
        }
    
    def start_monitoring(self, interval_seconds: int = 3600):
        """Start background monitoring thread"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Analyze feedback and schedule improvements
                    self.analyze_feedback_and_schedule_improvements()
                    
                    # Check system health
                    health = self.get_system_health()
                    logger.info(f"System Health: {health['health_score']:.1f}/100 ({health['status']})")
                    
                    # Sleep until next check
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait a minute before retrying
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Started continuous monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped continuous monitoring")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            'system_health': self.get_system_health(),
            'feedback_metrics': self.feedback_collector.get_aggregate_metrics(),
            'improvement_opportunities': self.feedback_collector.get_improvement_opportunities(),
            'retraining_queue': self.retraining_pipeline.get_retraining_queue(),
            'recent_feedback': self.feedback_collector.get_recent_feedback(count=50),
            'retraining_history': self.retraining_pipeline.get_retraining_history()[-10:]
        }


# Singleton instance
_improvement_system = None


def get_improvement_system() -> ContinuousImprovementSystem:
    """Get or create singleton improvement system"""
    global _improvement_system
    if _improvement_system is None:
        _improvement_system = ContinuousImprovementSystem()
    return _improvement_system


class FeedbackLoopSystem:
    """
    Integrated feedback loop system combining all components
    """
    
    def __init__(self):
        """Initialize the feedback loop system"""
        self.feedback_collector = FeedbackCollector()
        self.retraining_pipeline = ModelRetrainingPipeline()
        self.improvement_system = ContinuousImprovementSystem()
        
        logger.info("🔄 FeedbackLoopSystem initialized")
    
    def submit_feedback(self, user_id: str, interaction_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit feedback through the system"""
        self.feedback_collector.submit_feedback(user_id, interaction_id, feedback_data)
        return {
            'status': 'success',
            'feedback_id': interaction_id,
            'message': 'Feedback recorded successfully'
        }
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate feedback metrics"""
        return self.feedback_collector.get_aggregate_metrics()
    
    def get_feedback_by_intent(self) -> Dict[str, Any]:
        """Get feedback metrics grouped by intent"""
        return self.feedback_collector.aggregate_metrics.get('satisfaction_by_intent', {})
    
    def get_feedback_by_feature(self) -> Dict[str, Any]:
        """Get feedback metrics grouped by feature"""
        return self.feedback_collector.aggregate_metrics.get('satisfaction_by_feature', {})
    
    def get_improvement_status(self) -> Dict[str, Any]:
        """Get improvement system status"""
        return self.improvement_system.get_system_health()
    
    def start_background_monitoring(self, interval_seconds: int = 3600):
        """Start background monitoring and improvement"""
        self.improvement_system.start_monitoring(interval_seconds)
    
    def stop_background_monitoring(self):
        """Stop background monitoring"""
        self.improvement_system.stop_monitoring()


# Singleton instance for integrated system
_feedback_loop_system = None


def get_feedback_loop_system() -> FeedbackLoopSystem:
    """Get or create singleton feedback loop system"""
    global _feedback_loop_system
    if _feedback_loop_system is None:
        _feedback_loop_system = FeedbackLoopSystem()
    return _feedback_loop_system
