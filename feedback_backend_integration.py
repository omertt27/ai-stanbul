#!/usr/bin/env python3
"""
Backend Integration for User Feedback Collection
Middleware to automatically collect predictions and user corrections

Usage in backend/main.py:
    from feedback_backend_integration import FeedbackMiddleware
    
    # Initialize middleware
    feedback_middleware = FeedbackMiddleware()
    
    # Record predictions
    feedback_middleware.record_prediction(
        query=user_message,
        predicted_intent=intent,
        confidence=confidence,
        user_function=function_used
    )
"""

from typing import Dict, Optional
from user_feedback_collection_system import get_feedback_collector
import logging

logger = logging.getLogger(__name__)


class FeedbackMiddleware:
    """
    Middleware for seamless feedback collection in the backend
    Integrates with AI Istanbul's 8 core functions
    """
    
    # Map endpoint/handler to user function
    ENDPOINT_TO_FUNCTION = {
        '/chat': 'daily_talks',
        '/attractions': 'places_attractions',
        '/neighborhoods': 'neighborhood_guides',
        '/transportation': 'transportation',
        '/events': 'events_advising',
        '/route': 'route_planner',
        '/weather': 'weather_system',
        '/hidden-gems': 'local_tips_hidden_gems',
        '/local-tips': 'local_tips_hidden_gems',
    }
    
    def __init__(self):
        """Initialize feedback middleware"""
        self.collector = get_feedback_collector()
        self.feedback_system = self.collector  # Alias for backend compatibility
        self.enabled = True
        logger.info("✅ Feedback middleware initialized")
    
    def record_prediction(
        self,
        query: str,
        predicted_intent: str,
        confidence: float,
        user_function: Optional[str] = None,
        endpoint: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Record a prediction automatically
        
        Args:
            query: User query
            predicted_intent: Predicted intent
            confidence: Prediction confidence
            user_function: Core function being used
            endpoint: API endpoint (for auto-detection)
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional data
            
        Returns:
            Feedback ID or None if disabled
        """
        if not self.enabled:
            return None
        
        # Auto-detect function from endpoint
        if user_function is None and endpoint is not None:
            user_function = self.ENDPOINT_TO_FUNCTION.get(endpoint)
        
        try:
            feedback_id = self.collector.record_prediction(
                query=query,
                predicted_intent=predicted_intent,
                confidence=confidence,
                user_function=user_function,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata
            )
            return feedback_id
        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")
            return None
    
    def record_user_correction(
        self,
        query: str,
        wrong_intent: str,
        correct_intent: str,
        user_function: str,
        language: str = "auto",
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Record user correction for misclassification
        
        Call this when user indicates prediction was wrong
        
        Args:
            query: The query
            wrong_intent: What was predicted
            correct_intent: What it should be
            user_function: Core function
            language: Query language
            user_id: User identifier
            
        Returns:
            Feedback ID or None
        """
        if not self.enabled:
            return None
        
        try:
            feedback_id = self.collector.record_correction(
                query=query,
                wrong_intent=wrong_intent,
                correct_intent=correct_intent,
                user_function=user_function,
                language=language,
                user_id=user_id
            )
            logger.info(f"✏️ User correction recorded: {wrong_intent} → {correct_intent}")
            return feedback_id
        except Exception as e:
            logger.error(f"Failed to record correction: {e}")
            return None
    
    def record_thumbs_down(
        self,
        query: str,
        predicted_intent: str,
        confidence: float,
        user_function: str,
        reason: Optional[str] = None
    ) -> Optional[str]:
        """
        Record negative feedback (thumbs down)
        
        Args:
            query: The query
            predicted_intent: What was predicted
            confidence: Confidence score
            user_function: Core function
            reason: Optional reason for thumbs down
            
        Returns:
            Feedback ID or None
        """
        if not self.enabled:
            return None
        
        metadata = {'feedback_type': 'thumbs_down'}
        if reason:
            metadata['reason'] = reason
        
        return self.record_prediction(
            query=query,
            predicted_intent=predicted_intent,
            confidence=confidence,
            user_function=user_function,
            metadata=metadata
        )
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics"""
        return self.collector.get_feedback_summary(days=30)
    
    def generate_retraining_data(
        self,
        min_corrections: int = 5,
        output_file: str = "data/retraining_data.json"
    ) -> tuple:
        """
        Generate retraining data from feedback
        
        Args:
            min_corrections: Minimum corrections per intent
            output_file: Output file path
            
        Returns:
            (count, file_path)
        """
        return self.collector.generate_retraining_data(
            min_corrections=min_corrections,
            output_file=output_file
        )
    
    def enable(self):
        """Enable feedback collection"""
        self.enabled = True
        logger.info("✅ Feedback collection enabled")
    
    def disable(self):
        """Disable feedback collection"""
        self.enabled = False
        logger.info("⏸️ Feedback collection disabled")


# Alias for backward compatibility with backend/main.py
FeedbackIntegration = FeedbackMiddleware


# Example FastAPI integration
def create_feedback_endpoints(app, feedback_middleware: FeedbackMiddleware):
    """
    Add feedback endpoints to FastAPI app
    
    Usage:
        from fastapi import FastAPI
        from feedback_backend_integration import FeedbackMiddleware, create_feedback_endpoints
        
        app = FastAPI()
        feedback_middleware = FeedbackMiddleware()
        create_feedback_endpoints(app, feedback_middleware)
    """
    from fastapi import HTTPException
    from pydantic import BaseModel
    
    class CorrectionRequest(BaseModel):
        query: str
        wrong_intent: str
        correct_intent: str
        user_function: str
        language: str = "auto"
    
    class ThumbsDownRequest(BaseModel):
        query: str
        predicted_intent: str
        confidence: float
        user_function: str
        reason: Optional[str] = None
    
    @app.post("/api/feedback/correction")
    async def submit_correction(request: CorrectionRequest):
        """Submit a correction for misclassified query"""
        feedback_id = feedback_middleware.record_user_correction(
            query=request.query,
            wrong_intent=request.wrong_intent,
            correct_intent=request.correct_intent,
            user_function=request.user_function,
            language=request.language
        )
        
        if feedback_id:
            return {
                'success': True,
                'feedback_id': feedback_id,
                'message': 'Thank you for the correction!'
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to record correction")
    
    @app.post("/api/feedback/thumbs-down")
    async def submit_thumbs_down(request: ThumbsDownRequest):
        """Submit negative feedback"""
        feedback_id = feedback_middleware.record_thumbs_down(
            query=request.query,
            predicted_intent=request.predicted_intent,
            confidence=request.confidence,
            user_function=request.user_function,
            reason=request.reason
        )
        
        if feedback_id:
            return {
                'success': True,
                'feedback_id': feedback_id,
                'message': 'Thank you for your feedback!'
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to record feedback")
    
    @app.get("/api/feedback/statistics")
    async def get_feedback_stats():
        """Get feedback statistics (admin only)"""
        stats = feedback_middleware.get_statistics()
        return {
            'success': True,
            'statistics': stats
        }
    
    logger.info("✅ Feedback API endpoints created")


# Example usage decorator
def collect_feedback(user_function: str):
    """
    Decorator to automatically collect feedback
    
    Usage:
        @collect_feedback("transportation")
        def handle_transportation_query(query):
            result = intent_classifier.route_query(query)
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Extract query and result
            if isinstance(result, dict):
                query = kwargs.get('query') or (args[0] if args else None)
                
                if query and 'intent' in result:
                    feedback_middleware = FeedbackMiddleware()
                    feedback_middleware.record_prediction(
                        query=query,
                        predicted_intent=result['intent'],
                        confidence=result.get('confidence', 0.0),
                        user_function=user_function
                    )
            
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    print("=" * 80)
    print("FEEDBACK BACKEND INTEGRATION TEST")
    print("=" * 80)
    print()
    
    # Test middleware
    middleware = FeedbackMiddleware()
    
    # Simulate predictions from different functions
    print("📝 Simulating predictions from core functions...\n")
    
    # Daily talks
    middleware.record_prediction(
        query="Hello, how are you?",
        predicted_intent="greeting",
        confidence=0.95,
        user_function="daily_talks"
    )
    
    # Transportation
    middleware.record_prediction(
        query="How to get to Taksim?",
        predicted_intent="transportation",
        confidence=0.88,
        user_function="transportation"
    )
    
    # Places/Attractions (low confidence)
    middleware.record_prediction(
        query="Best museums",
        predicted_intent="attraction",  # Should be "museum"
        confidence=0.55,
        user_function="places_attractions"
    )
    
    # User correction
    middleware.record_user_correction(
        query="Best museums",
        wrong_intent="attraction",
        correct_intent="museum",
        user_function="places_attractions",
        language="en"
    )
    
    print("\n✅ Predictions recorded\n")
    
    # Show statistics
    stats = middleware.get_statistics()
    print("📊 Statistics:")
    print(f"   Total feedback: {stats['total']}")
    print(f"   By function: {dict(stats['by_function'])}")
    print(f"   By language: {dict(stats['by_language'])}")
    print()
