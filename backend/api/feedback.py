"""
Feedback API Endpoints
======================
Endpoints for collecting user feedback on LLM responses
Used for model fine-tuning and quality improvement
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from services.data_collection import log_user_feedback, get_collection_stats, export_training_dataset

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


class FeedbackRequest(BaseModel):
    """User feedback on a chat response"""
    interaction_id: str
    feedback_type: str  # 'thumbs_up', 'thumbs_down', 'rating', 'comment'
    feedback_value: Optional[str] = None
    comment: Optional[str] = None


class StatsResponse(BaseModel):
    """Data collection statistics"""
    total_interactions: int
    positive_feedback: int
    negative_feedback: int
    feedback_rate: float
    positive_rate: float
    languages: dict
    intents: dict


@router.post("/submit")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on a chat response
    
    Feedback types:
    - thumbs_up: User liked the response
    - thumbs_down: User disliked the response  
    - rating: User rated 1-5 stars
    - comment: User provided text feedback
    """
    try:
        log_user_feedback(
            interaction_id=request.interaction_id,
            feedback_type=request.feedback_type,
            feedback_value=request.feedback_value,
            comment=request.comment
        )
        
        logger.info(f"✅ Feedback received: {request.feedback_type} for {request.interaction_id}")
        
        return {
            "status": "success",
            "message": "Feedback recorded",
            "interaction_id": request.interaction_id
        }
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get data collection statistics
    Shows how much training data has been collected
    """
    try:
        stats = get_collection_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.post("/export")
async def export_dataset(
    filter_positive_only: bool = False,
    min_response_length: int = 20,
    max_response_length: int = 500
):
    """
    Export collected data to training dataset format
    Admin endpoint for preparing fine-tuning data
    
    Args:
        filter_positive_only: Only include positively-rated responses
        min_response_length: Minimum response length in characters
        max_response_length: Maximum response length in characters
    """
    try:
        count = export_training_dataset(
            filter_positive_only=filter_positive_only,
            min_response_length=min_response_length,
            max_response_length=max_response_length
        )
        
        logger.info(f"✅ Exported {count} training examples")
        
        return {
            "status": "success",
            "examples_exported": count,
            "message": f"Exported {count} training examples to training_data/training_dataset.jsonl"
        }
        
    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to export dataset")
