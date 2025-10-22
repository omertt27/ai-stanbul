"""
Automatic Intent Feedback Tracker
Integrates with the main query processing to automatically track classifications
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AutomaticFeedbackTracker:
    """
    Automatically track intent classifications for feedback collection
    Integrates with main query processing pipeline
    """
    
    def __init__(self, db_session=None, enable_tracking=True):
        """
        Initialize automatic feedback tracker
        
        Args:
            db_session: SQLAlchemy database session
            enable_tracking: Whether to enable automatic tracking
        """
        self.db_session = db_session
        self.enable_tracking = enable_tracking
        self.feedback_buffer = []
        
    def track_classification(
        self,
        session_id: str,
        query: str,
        intent: str,
        confidence: float,
        method: str = "neural",
        language: Optional[str] = None,
        latency_ms: Optional[float] = None,
        user_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Track an intent classification for potential feedback
        
        This creates a pending feedback entry that can be updated
        when user feedback is received.
        
        Args:
            session_id: User session ID
            query: Original query
            intent: Predicted intent
            confidence: Prediction confidence (0-1)
            method: Classification method ('neural' or 'fallback')
            language: Detected language ('tr', 'en', 'mixed')
            latency_ms: Classification latency in milliseconds
            user_id: Optional user ID
            context_data: Additional context as dictionary
            
        Returns:
            Feedback record ID if tracking is enabled, None otherwise
        """
        
        if not self.enable_tracking:
            return None
        
        if not self.db_session:
            logger.warning("No database session provided, buffering feedback")
            self.feedback_buffer.append({
                'session_id': session_id,
                'query': query,
                'intent': intent,
                'confidence': confidence,
                'method': method,
                'language': language,
                'latency_ms': latency_ms,
                'user_id': user_id,
                'context_data': context_data,
                'timestamp': datetime.utcnow()
            })
            return None
        
        try:
            from models.intent_feedback import IntentFeedback
            
            feedback = IntentFeedback(
                session_id=session_id,
                user_id=user_id,
                original_query=query,
                language=language,
                predicted_intent=intent,
                predicted_confidence=confidence,
                classification_method=method,
                latency_ms=latency_ms,
                feedback_type='pending',  # Will be updated when user provides feedback
                review_status='pending',
                context_data=context_data
            )
            
            self.db_session.add(feedback)
            self.db_session.commit()
            self.db_session.refresh(feedback)
            
            logger.debug(f"📊 Tracked classification: {query[:30]}... -> {intent} ({confidence:.2f})")
            
            return feedback.id
            
        except Exception as e:
            logger.error(f"Failed to track classification: {e}")
            self.db_session.rollback()
            return None
    
    def track_implicit_feedback(
        self,
        session_id: str,
        user_action: str,
        time_spent_seconds: Optional[float] = None
    ):
        """
        Track implicit feedback based on user behavior
        
        Args:
            session_id: User session ID
            user_action: Action taken by user
            time_spent_seconds: Time spent on result
        """
        
        if not self.enable_tracking or not self.db_session:
            return
        
        try:
            from models.intent_feedback import IntentFeedback
            
            # Get the most recent classification for this session
            recent = self.db_session.query(IntentFeedback).filter(
                IntentFeedback.session_id == session_id,
                IntentFeedback.feedback_type == 'pending'
            ).order_by(IntentFeedback.timestamp.desc()).first()
            
            if not recent:
                logger.debug(f"No pending classification found for session {session_id}")
                return
            
            # Infer correctness from user action
            is_correct = None
            if user_action == 'clicked_result':
                is_correct = True
            elif user_action in ['refined_query', 'ignored']:
                is_correct = False
            
            # Update the feedback record
            recent.feedback_type = 'implicit'
            recent.user_action = user_action
            recent.is_correct = is_correct
            
            if time_spent_seconds:
                import json
                context = json.loads(recent.context_data) if recent.context_data else {}
                context['time_spent'] = time_spent_seconds
                recent.context_data = json.dumps(context)
            
            # Auto-approve high confidence positive feedback
            if is_correct and recent.predicted_confidence > 0.8:
                recent.review_status = 'approved'
            
            self.db_session.commit()
            
            logger.info(f"📊 Implicit feedback: {user_action} -> correct={is_correct}")
            
        except Exception as e:
            logger.error(f"Failed to track implicit feedback: {e}")
            self.db_session.rollback()
    
    def get_feedback_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get feedback statistics for the last N days
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with feedback statistics
        """
        
        if not self.db_session:
            return {
                'error': 'No database session',
                'buffered': len(self.feedback_buffer)
            }
        
        try:
            from models.intent_feedback import FeedbackStatistics
            from datetime import timedelta
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            accuracy_stats = FeedbackStatistics.calculate_accuracy(
                self.db_session,
                start_date=start_date
            )
            
            intent_dist = FeedbackStatistics.get_intent_distribution(self.db_session)
            training_ready = FeedbackStatistics.get_training_ready_count(self.db_session)
            
            return {
                'total_feedback': accuracy_stats['total'],
                'accuracy': accuracy_stats['accuracy'],
                'correct': accuracy_stats['correct'],
                'incorrect': accuracy_stats['incorrect'],
                'by_intent': intent_dist,
                'training_ready': training_ready
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {'error': str(e)}
    
    def flush_buffer(self):
        """
        Flush buffered feedback to database
        """
        
        if not self.db_session or not self.feedback_buffer:
            return
        
        try:
            from models.intent_feedback import IntentFeedback
            
            for item in self.feedback_buffer:
                feedback = IntentFeedback(
                    session_id=item['session_id'],
                    user_id=item['user_id'],
                    original_query=item['query'],
                    language=item['language'],
                    predicted_intent=item['intent'],
                    predicted_confidence=item['confidence'],
                    classification_method=item['method'],
                    latency_ms=item['latency_ms'],
                    feedback_type='pending',
                    review_status='pending',
                    context_data=item['context_data'],
                    timestamp=item['timestamp']
                )
                self.db_session.add(feedback)
            
            self.db_session.commit()
            
            logger.info(f"✅ Flushed {len(self.feedback_buffer)} buffered feedback items")
            self.feedback_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush buffer: {e}")
            self.db_session.rollback()


# Decorator for automatic tracking
def track_intent_classification(tracker: AutomaticFeedbackTracker):
    """
    Decorator to automatically track intent classification
    
    Usage:
        @track_intent_classification(tracker)
        def classify_query(query, session_id):
            # ... classification logic ...
            return {
                'intent': intent,
                'confidence': confidence,
                'method': 'neural'
            }
    """
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get session_id from args or kwargs
            session_id = kwargs.get('session_id') or (args[1] if len(args) > 1 else None)
            query = kwargs.get('query') or (args[0] if len(args) > 0 else None)
            
            # Measure latency
            start_time = time.time()
            
            # Call original function
            result = func(*args, **kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Track classification
            if isinstance(result, dict) and 'intent' in result:
                tracker.track_classification(
                    session_id=session_id or 'unknown',
                    query=query or '',
                    intent=result['intent'],
                    confidence=result.get('confidence', 0.0),
                    method=result.get('method', 'unknown'),
                    language=result.get('language'),
                    latency_ms=latency_ms,
                    user_id=kwargs.get('user_id'),
                    context_data=result.get('context')
                )
            
            return result
        
        return wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Automatic Intent Feedback Tracker")
    print("=" * 60)
    
    # Test without database
    tracker = AutomaticFeedbackTracker(enable_tracking=True)
    
    # Track some classifications
    tracker.track_classification(
        session_id='test_session_1',
        query='Sultanahmet\'te restoran öner',
        intent='restaurant',
        confidence=0.85,
        method='neural',
        language='tr',
        latency_ms=12.5
    )
    
    print(f"\n📊 Buffered feedback items: {len(tracker.feedback_buffer)}")
    
    # Simulate implicit feedback
    tracker.track_implicit_feedback(
        session_id='test_session_1',
        user_action='clicked_result',
        time_spent_seconds=15.3
    )
    
    print("\n✅ Automatic tracking test completed")
