"""
analytics.py - Analytics & Monitoring System

Comprehensive analytics and monitoring for the LLM system.

Tracks:
- Query patterns and trends
- Performance metrics
- Error rates and types
- Signal detection accuracy
- Context usage
- User behavior
- Service health
- Cache efficiency
- Intent discrepancies and feedback loop (PHASE 2)
- Production monitoring with real-time metrics (PHASE 4 PRIORITY 4)

Author: AI Istanbul Team
Date: December 7, 2025
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

# Import Phase 4 Priority 4: Production Monitoring
try:
    from .monitoring import get_monitor, track_query as monitor_track_query
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("⚠️ Production monitoring module not available")

logger = logging.getLogger(__name__)


class AnalyticsManager:
    """
    Comprehensive analytics and monitoring system.
    
    Provides real-time and historical analytics for:
    - Performance optimization
    - Error tracking and debugging
    - User behavior analysis
    - System health monitoring
    - Feedback loop training (PHASE 2)
    """
    
    def __init__(
        self, 
        enable_detailed_tracking: bool = True,
        enable_feedback_loop: bool = True,
        feedback_trainer = None
    ):
        """
        Initialize analytics manager.
        
        Args:
            enable_detailed_tracking: Enable detailed metrics (may impact performance)
            enable_feedback_loop: Enable feedback loop training (PHASE 2)
            feedback_trainer: FeedbackTrainer instance (optional, will create if None)
        """
        self.enable_detailed = enable_detailed_tracking
        self.enable_feedback_loop = enable_feedback_loop
        
        # Initialize feedback trainer (PHASE 2)
        self.feedback_trainer = None
        if self.enable_feedback_loop:
            try:
                if feedback_trainer is not None:
                    self.feedback_trainer = feedback_trainer
                else:
                    from .feedback_trainer import FeedbackTrainer
                    self.feedback_trainer = FeedbackTrainer()
                logger.info("✅ Feedback loop enabled for continuous improvement")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize feedback trainer: {e}")
                self.enable_feedback_loop = False
        
        # Basic counters
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'validation_failures': 0,
            'fallback_calls': 0
        }
        
        # Performance metrics
        self.performance = {
            'query_latencies': deque(maxlen=1000),
            'llm_latencies': deque(maxlen=1000),
            'cache_latencies': deque(maxlen=100)
        }
        
        # Error tracking
        self.errors = {
            'total_errors': 0,
            'by_type': defaultdict(int),
            'by_service': defaultdict(int),
            'recent_errors': deque(maxlen=50)
        }
        
        # User analytics
        self.users = {
            'unique_users': set(),
            'queries_by_language': defaultdict(int),
            'queries_by_user': defaultdict(int)
        }
        
        # Signal analytics
        self.signals = {
            'detections_by_signal': defaultdict(int),
            'confidence_scores': defaultdict(list),
            'multi_intent_queries': 0
        }
        
        # Context analytics
        self.context = {
            'database_usage': 0,
            'rag_usage': 0,
            'service_calls': defaultdict(int)
        }
        
        # Time-based tracking (hourly buckets)
        self.hourly_stats = defaultdict(lambda: {
            'queries': 0,
            'errors': 0,
            'avg_latency': []
        })
        
        logger.info("✅ Analytics Manager initialized")
    
    def track_query(self, user_id: str, language: str, query: str):
        """
        Track a new query.
        
        Args:
            user_id: User identifier
            language: Query language
            query: Query text
        """
        self.stats['total_queries'] += 1
        self.users['unique_users'].add(user_id)
        self.users['queries_by_language'][language] += 1
        self.users['queries_by_user'][user_id] += 1
        
        # Track hourly
        hour_key = datetime.now().strftime('%Y-%m-%d-%H')
        self.hourly_stats[hour_key]['queries'] += 1
        
        if self.enable_detailed:
            logger.debug(f"📊 Query tracked: user={user_id}, lang={language}")
    
    def track_cache_hit(self):
        """Track a cache hit."""
        self.stats['cache_hits'] += 1
    
    def track_cache_miss(self):
        """Track a cache miss."""
        self.stats['cache_misses'] += 1
    
    def track_signals(self, signals: Dict[str, bool]):
        """
        Track detected signals.
        
        Args:
            signals: Dict of signal_name -> detected (bool)
        """
        active_count = 0
        
        for signal_name, detected in signals.items():
            if detected:
                self.signals['detections_by_signal'][signal_name] += 1
                active_count += 1
        
        # Track multi-intent
        if active_count > 2:
            self.signals['multi_intent_queries'] += 1
    
    async def track_intent_comparison(
        self,
        regex_intents: Dict[str, bool],
        llm_intents: Dict[str, bool],
        query: str,
        user_id: str = "anonymous",
        language: str = "en",
        user_location: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track comparison between regex-detected and LLM-detected intents (PRIORITY 2).
        
        This helps identify when signal detection fails and LLM compensates.
        Also logs training samples for feedback loop (PHASE 2).
        
        Args:
            regex_intents: Intents detected by regex patterns
            llm_intents: Intents detected by LLM
            query: Original query
            user_id: User identifier
            language: Query language
            user_location: User's GPS location (if available)
            metadata: Additional metadata (confidence scores, etc.)
        """
        # Find discrepancies
        false_positives = [
            signal for signal in regex_intents
            if regex_intents.get(signal, False) and not llm_intents.get(signal, False)
        ]
        
        false_negatives = [
            signal for signal in llm_intents
            if llm_intents.get(signal, False) and not regex_intents.get(signal, False)
        ]
        
        # Track discrepancies
        if not hasattr(self, 'intent_discrepancies'):
            self.intent_discrepancies = {
                'false_positives': defaultdict(int),
                'false_negatives': defaultdict(int),
                'total_comparisons': 0,
                'perfect_matches': 0,
                'recent_mismatches': deque(maxlen=100)
            }
        
        self.intent_discrepancies['total_comparisons'] += 1
        
        # PHASE 2: Log training sample to feedback trainer
        if self.enable_feedback_loop and self.feedback_trainer:
            try:
                self.feedback_trainer.log_training_sample(
                    query=query,
                    regex_intents=regex_intents,
                    llm_intents=llm_intents,
                    language=language,
                    user_location=user_location,
                    metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Failed to log training sample: {e}")
        
        if false_positives or false_negatives:
            # Record discrepancies
            for signal in false_positives:
                self.intent_discrepancies['false_positives'][signal] += 1
            
            for signal in false_negatives:
                self.intent_discrepancies['false_negatives'][signal] += 1
            
            # Store recent mismatch for analysis
            self.intent_discrepancies['recent_mismatches'].append({
                'timestamp': datetime.now(),
                'query': query,
                'user_id': user_id,
                'regex_intents': [k for k, v in regex_intents.items() if v],
                'llm_intents': [k for k, v in llm_intents.items() if v],
                'false_positives': false_positives,
                'false_negatives': false_negatives
            })
            
            logger.info(
                f"🎯 Intent mismatch detected:\n"
                f"   Query: {query}\n"
                f"   Regex detected: {[k for k, v in regex_intents.items() if v]}\n"
                f"   LLM detected: {[k for k, v in llm_intents.items() if v]}\n"
                f"   False positives: {false_positives}\n"
                f"   False negatives: {false_negatives}"
            )
        else:
            # Perfect match
            self.intent_discrepancies['perfect_matches'] += 1
    
    def get_intent_comparison_report(self) -> Dict[str, Any]:
        """
        Generate report on intent detection accuracy (PRIORITY 2).
        
        Returns:
            Report with accuracy metrics and common discrepancies
        """
        if not hasattr(self, 'intent_discrepancies'):
            return {'error': 'No intent comparison data available'}
        
        disc = self.intent_discrepancies
        
        # Calculate accuracy
        total = disc['total_comparisons']
        perfect = disc['perfect_matches']
        accuracy = (perfect / total * 100) if total > 0 else 0
        
        # Find most common false positives/negatives
        top_fp = sorted(
            disc['false_positives'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_fn = sorted(
            disc['false_negatives'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_comparisons': total,
            'perfect_matches': perfect,
            'accuracy_percentage': accuracy,
            'total_discrepancies': total - perfect,
            'top_false_positives': [
                {'signal': sig, 'count': count, 'percentage': count/total*100}
                for sig, count in top_fp
            ],
            'top_false_negatives': [
                {'signal': sig, 'count': count, 'percentage': count/total*100}
                for sig, count in top_fn
            ],
            'recent_mismatches': list(disc['recent_mismatches'])[-10:]
        }
    
    def track_context(self, context: Dict[str, Any]):
        """
        Track context usage.
        
        Args:
            context: Built context dict
        """
        if context.get('database'):
            self.context['database_usage'] += 1
        
        if context.get('rag'):
            self.context['rag_usage'] += 1
        
        # Track service calls
        for service_name in context.get('services', {}).keys():
            self.context['service_calls'][service_name] += 1
    
    def track_response(
        self,
        latency: float,
        llm_latency: float,
        signals: Dict[str, bool],
        context: Dict[str, Any]
    ):
        """
        Track a completed response.
        
        Args:
            latency: Total query latency (seconds)
            llm_latency: LLM generation latency (seconds)
            signals: Detected signals
            context: Used context
        """
        self.performance['query_latencies'].append(latency)
        self.performance['llm_latencies'].append(llm_latency)
        
        # Track hourly avg latency
        hour_key = datetime.now().strftime('%Y-%m-%d-%H')
        self.hourly_stats[hour_key]['avg_latency'].append(latency)
    
    def track_error(self, error_type: str, error_message: str):
        """
        Track an error.
        
        Args:
            error_type: Error type/category
            error_message: Error description
        """
        self.errors['total_errors'] += 1
        self.errors['by_type'][error_type] += 1
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message[:200]
        }
        
        self.errors['recent_errors'].append(error_entry)
        
        # Track hourly
        hour_key = datetime.now().strftime('%Y-%m-%d-%H')
        self.hourly_stats[hour_key]['errors'] += 1
        
        logger.error(f"❌ Error tracked: {error_type} - {error_message[:100]}")
    
    def track_validation_failure(self, reason: str):
        """Track a validation failure."""
        self.stats['validation_failures'] += 1
        self.track_error('validation_failure', reason)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics summary.
        
        Returns:
            Dict with all analytics data
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'basic_stats': dict(self.stats),
            'performance': self._get_performance_stats(),
            'errors': self._get_error_stats(),
            'users': self._get_user_stats(),
            'signals': self._get_signal_stats(),
            'context': self._get_context_stats(),
            'hourly_trends': self._get_hourly_trends()
        }
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        def calc_stats(values):
            if not values:
                return {'count': 0, 'min': 0, 'max': 0, 'avg': 0, 'p50': 0, 'p95': 0, 'p99': 0}
            
            sorted_vals = sorted(values)
            count = len(sorted_vals)
            
            return {
                'count': count,
                'min': sorted_vals[0],
                'max': sorted_vals[-1],
                'avg': sum(sorted_vals) / count,
                'p50': sorted_vals[int(count * 0.5)],
                'p95': sorted_vals[int(count * 0.95)] if count > 20 else sorted_vals[-1],
                'p99': sorted_vals[int(count * 0.99)] if count > 100 else sorted_vals[-1]
            }
        
        return {
            'query_latency': calc_stats(self.performance['query_latencies']),
            'llm_latency': calc_stats(self.performance['llm_latencies']),
            'cache_latency': calc_stats(self.performance['cache_latencies'])
        }
    
    def _get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_queries = max(1, self.stats['total_queries'])
        
        return {
            'total': self.errors['total_errors'],
            'error_rate': self.errors['total_errors'] / total_queries,
            'by_type': dict(self.errors['by_type']),
            'by_service': dict(self.errors['by_service']),
            'recent': list(self.errors['recent_errors'])[-10:]
        }
    
    def _get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics."""
        return {
            'unique_users': len(self.users['unique_users']),
            'queries_by_language': dict(self.users['queries_by_language']),
            'avg_queries_per_user': (
                self.stats['total_queries'] / max(1, len(self.users['unique_users']))
            )
        }
    
    def _get_signal_stats(self) -> Dict[str, Any]:
        """Get signal detection statistics."""
        total_queries = max(1, self.stats['total_queries'])
        
        return {
            'detections_by_signal': dict(self.signals['detections_by_signal']),
            'multi_intent_queries': self.signals['multi_intent_queries'],
            'multi_intent_rate': self.signals['multi_intent_queries'] / total_queries
        }
    
    def _get_context_stats(self) -> Dict[str, Any]:
        """Get context usage statistics."""
        total_queries = max(1, self.stats['total_queries'])
        
        return {
            'database_usage': self.context['database_usage'],
            'database_usage_rate': self.context['database_usage'] / total_queries,
            'rag_usage': self.context['rag_usage'],
            'rag_usage_rate': self.context['rag_usage'] / total_queries,
            'service_calls': dict(self.context['service_calls'])
        }
    
    def _get_hourly_trends(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get hourly trends for the last N hours.
        
        Args:
            hours: Number of hours to include
            
        Returns:
            List of hourly stats
        """
        trends = []
        now = datetime.now()
        
        for i in range(hours):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime('%Y-%m-%d-%H')
            
            stats = self.hourly_stats.get(hour_key, {
                'queries': 0,
                'errors': 0,
                'avg_latency': []
            })
            
            avg_latency = (
                sum(stats['avg_latency']) / len(stats['avg_latency'])
                if stats['avg_latency'] else 0
            )
            
            trends.append({
                'hour': hour.strftime('%Y-%m-%d %H:00'),
                'queries': stats['queries'],
                'errors': stats['errors'],
                'error_rate': stats['errors'] / max(1, stats['queries']),
                'avg_latency': avg_latency
            })
        
        return list(reversed(trends))  # Oldest to newest
    
    def get_cache_efficiency(self) -> Dict[str, Any]:
        """Get cache efficiency metrics."""
        total_cache_checks = self.stats['cache_hits'] + self.stats['cache_misses']
        
        if total_cache_checks == 0:
            return {
                'hit_rate': 0.0,
                'miss_rate': 0.0,
                'total_checks': 0
            }
        
        return {
            'hit_rate': self.stats['cache_hits'] / total_cache_checks,
            'miss_rate': self.stats['cache_misses'] / total_cache_checks,
            'total_checks': total_cache_checks,
            'hits': self.stats['cache_hits'],
            'misses': self.stats['cache_misses']
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Health status with indicators
        """
        # Calculate health indicators
        error_rate = (
            self.errors['total_errors'] / max(1, self.stats['total_queries'])
        )
        
        cache_hit_rate = self.get_cache_efficiency()['hit_rate']
        
        avg_latency = (
            sum(self.performance['query_latencies']) / len(self.performance['query_latencies'])
            if self.performance['query_latencies'] else 0
        )
        
        # Determine health status
        health_score = 100.0
        
        if error_rate > 0.1:
            health_score -= 30  # High error rate
        elif error_rate > 0.05:
            health_score -= 15
        
        if avg_latency > 5.0:
            health_score -= 20  # Slow responses
        elif avg_latency > 3.0:
            health_score -= 10
        
        if cache_hit_rate < 0.3:
            health_score -= 10  # Low cache efficiency
        
        # Status
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'status': status,
            'health_score': health_score,
            'indicators': {
                'error_rate': error_rate,
                'avg_latency': avg_latency,
                'cache_hit_rate': cache_hit_rate
            },
            'recommendations': self._get_health_recommendations(
                error_rate, avg_latency, cache_hit_rate
            )
        }
    
    def get_feedback_loop_status(self) -> Dict[str, Any]:
        """
        Get feedback loop training status (PHASE 2).
        
        Returns:
            Dict with training statistics and retraining readiness
        """
        if not self.enable_feedback_loop or not self.feedback_trainer:
            return {
                'enabled': False,
                'message': 'Feedback loop not enabled'
            }
        
        try:
            stats = self.feedback_trainer.get_statistics()
            readiness = self.feedback_trainer.get_retraining_readiness()
            
            return {
                'enabled': True,
                'statistics': stats,
                'readiness': readiness,
                'can_retrain': readiness['is_ready']
            }
        except Exception as e:
            logger.error(f"Error getting feedback loop status: {e}")
            return {
                'enabled': True,
                'error': str(e)
            }
    
    def trigger_retraining(self) -> Dict[str, Any]:
        """
        Trigger pattern retraining from feedback loop (PHASE 2).
        
        Returns:
            Dict with retraining results
        """
        if not self.enable_feedback_loop or not self.feedback_trainer:
            return {
                'success': False,
                'message': 'Feedback loop not enabled'
            }
        
        try:
            # Check readiness
            readiness = self.feedback_trainer.get_retraining_readiness()
            if not readiness['is_ready']:
                return {
                    'success': False,
                    'message': readiness['recommendation'],
                    'readiness': readiness
                }
            
            # Generate report
            report = self.feedback_trainer.generate_retraining_report()
            
            # Export learned patterns
            success = self.feedback_trainer.export_learned_patterns()
            
            return {
                'success': success,
                'message': 'Retraining completed successfully' if success else 'Retraining failed',
                'report_summary': {
                    'total_samples': report['statistics']['total_samples'],
                    'intents_analyzed': len(report['intent_analyses']),
                    'patterns_suggested': self.feedback_trainer.stats['patterns_suggested']
                },
                'report': report
            }
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return {
                'success': False,
                'message': f'Retraining error: {str(e)}'
            }
    
    def _get_health_recommendations(
        self,
        error_rate: float,
        avg_latency: float,
        cache_hit_rate: float
    ) -> List[str]:
        """Get health improvement recommendations."""
        recommendations = []
        
        if error_rate > 0.1:
            recommendations.append("High error rate detected. Review error logs and address common failures.")
        
        if avg_latency > 5.0:
            recommendations.append("High latency detected. Consider optimizing LLM calls or context building.")
        
        if cache_hit_rate < 0.3:
            recommendations.append("Low cache hit rate. Review cache TTL and semantic similarity thresholds.")
        
        if not recommendations:
            recommendations.append("System is performing well. Continue monitoring.")
        
        return recommendations
    
    def reset_stats(self):
        """Reset all statistics (use carefully!)."""
        logger.warning("🔄 Resetting all analytics statistics")
        
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'validation_failures': 0,
            'fallback_calls': 0
        }
        
        self.performance['query_latencies'].clear()
        self.performance['llm_latencies'].clear()
        self.performance['cache_latencies'].clear()
        
        self.errors['total_errors'] = 0
        self.errors['by_type'].clear()
        self.errors['by_service'].clear()
        self.errors['recent_errors'].clear()
        
        # Keep user tracking
        # self.users['unique_users'].clear()
        # self.users['queries_by_language'].clear()
