"""
Intent Feedback API Endpoints
Collect user feedback on intent classification
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

# Import database
try:
    from backend.database import get_db
except:
    from database import get_db

# Import models
try:
    from backend.models.intent_feedback import IntentFeedback, FeedbackStatistics
except:
    from models.intent_feedback import IntentFeedback, FeedbackStatistics

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/feedback",
    tags=["Intent Feedback"]
)


# Pydantic models for request/response
class IntentFeedbackRequest(BaseModel):
    """Request model for submitting intent feedback"""
    session_id: str = Field(..., description="User session ID")
    user_id: Optional[str] = Field(None, description="User ID if logged in")
    query: str = Field(..., description="Original query")
    language: Optional[str] = Field(None, description="Query language (tr/en/mixed)")
    predicted_intent: str = Field(..., description="Predicted intent")
    predicted_confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    classification_method: Optional[str] = Field(None, description="neural/fallback")
    latency_ms: Optional[float] = Field(None, description="Classification latency")
    is_correct: Optional[bool] = Field(None, description="Was classification correct?")
    actual_intent: Optional[str] = Field(None, description="Correct intent if is_correct=False")
    feedback_type: str = Field("explicit", description="explicit/implicit/click")
    user_action: Optional[str] = Field(None, description="User action after classification")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "abc123",
                "query": "Sultanahmet'te restoran",
                "predicted_intent": "restaurant",
                "predicted_confidence": 0.85,
                "is_correct": False,
                "actual_intent": "attraction",
                "feedback_type": "explicit"
            }
        }


class ImplicitFeedbackRequest(BaseModel):
    """Request model for implicit feedback"""
    session_id: str
    query: str
    predicted_intent: str
    predicted_confidence: float
    user_action: str  # 'clicked_result', 'refined_query', 'ignored', 'engaged'
    time_spent_seconds: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "abc123",
                "query": "Best restaurants",
                "predicted_intent": "restaurant",
                "predicted_confidence": 0.92,
                "user_action": "clicked_result",
                "time_spent_seconds": 15.3
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    success: bool
    message: str
    feedback_id: Optional[int] = None


class FeedbackStatsResponse(BaseModel):
    """Response model for feedback statistics"""
    total_feedback: int
    accuracy: float
    by_intent: Dict[str, int]
    by_feedback_type: Dict[str, int]
    training_ready: int


# Endpoints

@router.post("/intent", response_model=FeedbackResponse)
async def submit_intent_feedback(
    feedback: IntentFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit explicit or implicit feedback on intent classification
    
    This endpoint stores user feedback for:
    - Active learning and model improvement
    - Quality monitoring
    - Analytics
    """
    try:
        # Validate feedback
        if feedback.is_correct is False and not feedback.actual_intent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="actual_intent required when is_correct=False"
            )
        
        # Create feedback record
        feedback_record = IntentFeedback(
            session_id=feedback.session_id,
            user_id=feedback.user_id,
            original_query=feedback.query,
            language=feedback.language,
            predicted_intent=feedback.predicted_intent,
            predicted_confidence=feedback.predicted_confidence,
            classification_method=feedback.classification_method,
            latency_ms=feedback.latency_ms,
            is_correct=feedback.is_correct,
            actual_intent=feedback.actual_intent,
            feedback_type=feedback.feedback_type,
            user_action=feedback.user_action,
            context_data=feedback.context_data
        )
        
        # Auto-approve high-quality feedback
        if feedback_record.is_high_quality():
            feedback_record.review_status = 'approved'
        
        db.add(feedback_record)
        db.commit()
        db.refresh(feedback_record)
        
        logger.info(f"✅ Feedback received: {feedback.query[:30]}... -> {feedback.predicted_intent} (correct: {feedback.is_correct})")
        
        # Check if we should trigger retraining
        training_ready = FeedbackStatistics.get_training_ready_count(db)
        if training_ready >= 500:
            logger.info(f"🔄 {training_ready} training samples ready - consider triggering retraining")
        
        return FeedbackResponse(
            success=True,
            message="Feedback received successfully",
            feedback_id=feedback_record.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save feedback: {str(e)}"
        )


@router.post("/intent/implicit", response_model=FeedbackResponse)
async def submit_implicit_feedback(
    feedback: ImplicitFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit implicit feedback based on user behavior
    
    User actions indicate classification quality:
    - clicked_result: Likely correct (high confidence)
    - engaged: User spent time (likely correct)
    - refined_query: Likely incorrect (user reformulated)
    - ignored: Likely incorrect or irrelevant
    """
    try:
        # Infer correctness from user action
        is_correct = None
        inferred_confidence = feedback.predicted_confidence
        
        if feedback.user_action == 'clicked_result':
            is_correct = True
            inferred_confidence = min(0.95, feedback.predicted_confidence + 0.1)
        elif feedback.user_action == 'engaged' and feedback.time_spent_seconds and feedback.time_spent_seconds > 10:
            is_correct = True
            inferred_confidence = min(0.90, feedback.predicted_confidence + 0.05)
        elif feedback.user_action == 'refined_query':
            is_correct = False
            inferred_confidence = max(0.3, feedback.predicted_confidence - 0.2)
        elif feedback.user_action == 'ignored':
            is_correct = False
            inferred_confidence = max(0.4, feedback.predicted_confidence - 0.1)
        
        # Create implicit feedback record
        feedback_record = IntentFeedback(
            session_id=feedback.session_id,
            original_query=feedback.query,
            predicted_intent=feedback.predicted_intent,
            predicted_confidence=inferred_confidence,
            is_correct=is_correct,
            feedback_type='implicit',
            user_action=feedback.user_action,
            context_data={'time_spent': feedback.time_spent_seconds} if feedback.time_spent_seconds else None
        )
        
        # Implicit feedback requires review before training
        feedback_record.review_status = 'pending'
        
        db.add(feedback_record)
        db.commit()
        db.refresh(feedback_record)
        
        logger.info(f"📊 Implicit feedback: {feedback.query[:30]}... -> {feedback.user_action} (inferred: {is_correct})")
        
        return FeedbackResponse(
            success=True,
            message="Implicit feedback recorded",
            feedback_id=feedback_record.id
        )
        
    except Exception as e:
        logger.error(f"Error saving implicit feedback: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save implicit feedback: {str(e)}"
        )


@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get feedback statistics for the last N days
    
    Returns:
    - Total feedback count
    - Overall accuracy
    - Distribution by intent
    - Distribution by feedback type
    - Training-ready samples count
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Calculate accuracy
        accuracy_stats = FeedbackStatistics.calculate_accuracy(db, start_date=start_date)
        
        # Get distributions
        intent_dist = FeedbackStatistics.get_intent_distribution(db)
        
        # Count by feedback type
        feedback_types = {}
        for feedback_type in ['explicit', 'implicit', 'click']:
            count = db.query(IntentFeedback).filter(
                IntentFeedback.feedback_type == feedback_type,
                IntentFeedback.timestamp >= start_date
            ).count()
            feedback_types[feedback_type] = count
        
        # Training ready count
        training_ready = FeedbackStatistics.get_training_ready_count(db)
        
        return FeedbackStatsResponse(
            total_feedback=accuracy_stats['total'],
            accuracy=accuracy_stats['accuracy'],
            by_intent=intent_dist,
            by_feedback_type=feedback_types,
            training_ready=training_ready
        )
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feedback stats: {str(e)}"
        )


@router.get("/recent")
async def get_recent_feedback(
    limit: int = 50,
    intent: Optional[str] = None,
    feedback_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get recent feedback items
    
    Useful for:
    - Monitoring classification quality
    - Reviewing misclassifications
    - Quality control
    """
    try:
        query = db.query(IntentFeedback).order_by(
            IntentFeedback.timestamp.desc()
        )
        
        if intent:
            query = query.filter(IntentFeedback.predicted_intent == intent)
        
        if feedback_type:
            query = query.filter(IntentFeedback.feedback_type == feedback_type)
        
        feedback_items = query.limit(limit).all()
        
        return {
            "count": len(feedback_items),
            "items": [item.to_dict() for item in feedback_items]
        }
        
    except Exception as e:
        logger.error(f"Error getting recent feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recent feedback: {str(e)}"
        )


@router.get("/misclassifications")
async def get_misclassifications(
    limit: int = 50,
    min_confidence: float = 0.0,
    db: Session = Depends(get_db)
):
    """
    Get misclassified queries for analysis
    
    Useful for:
    - Understanding model weaknesses
    - Identifying patterns in errors
    - Improving training data
    """
    try:
        misclassifications = db.query(IntentFeedback).filter(
            IntentFeedback.is_correct == False,
            IntentFeedback.predicted_confidence >= min_confidence
        ).order_by(
            IntentFeedback.predicted_confidence.desc()
        ).limit(limit).all()
        
        return {
            "count": len(misclassifications),
            "items": [{
                "query": item.original_query,
                "predicted": item.predicted_intent,
                "actual": item.actual_intent,
                "confidence": item.predicted_confidence,
                "method": item.classification_method,
                "timestamp": item.timestamp.isoformat()
            } for item in misclassifications]
        }
        
    except Exception as e:
        logger.error(f"Error getting misclassifications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get misclassifications: {str(e)}"
        )


# Health check
@router.get("/health")
async def feedback_health_check():
    """Health check for feedback system"""
    return {
        "status": "healthy",
        "service": "intent_feedback",
        "timestamp": datetime.utcnow().isoformat()
    }
