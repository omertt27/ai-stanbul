"""
Real-Time Feedback API Endpoints
Provides endpoints for tracking user interactions and getting personalized recommendations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from backend.services.realtime_feedback_loop import get_realtime_feedback_loop
from backend.services.hidden_gems_handler import HiddenGemsHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedback", tags=["Real-Time Feedback"])


# Request models
class ViewEventRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    item_type: str = Field(..., description="Type of item (hidden_gem, restaurant, etc.)")
    dwell_time: Optional[float] = Field(None, description="Time spent viewing (seconds)")
    session_id: Optional[str] = Field(None, description="Session identifier")


class ClickEventRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    item_type: str = Field(..., description="Type of item")
    position: Optional[int] = Field(None, description="Position in results list")
    session_id: Optional[str] = Field(None, description="Session identifier")


class RatingEventRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    item_type: str = Field(..., description="Type of item")
    rating: float = Field(..., ge=0, le=5, description="Rating value (0-5)")
    session_id: Optional[str] = Field(None, description="Session identifier")


class SaveEventRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    item_type: str = Field(..., description="Type of item")
    session_id: Optional[str] = Field(None, description="Session identifier")


class ConversionEventRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    item_type: str = Field(..., description="Type of item")
    session_id: Optional[str] = Field(None, description="Session identifier")


class PersonalizedRecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    location: Optional[str] = Field(None, description="Location filter")
    gem_type: Optional[str] = Field(None, description="Type of gem")
    limit: int = Field(10, ge=1, le=50, description="Number of recommendations")
    session_id: Optional[str] = Field(None, description="Session identifier")


# Response models
class FeedbackResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    personalized: bool
    count: int


class MetricsResponse(BaseModel):
    feedback_collector: Dict[str, Any]
    learning_engine: Dict[str, Any]
    status: str


@router.post("/track/view", response_model=FeedbackResponse)
async def track_view(
    request: ViewEventRequest,
    background_tasks: BackgroundTasks
):
    """
    Track a view event
    Records when a user views an item
    """
    try:
        feedback_loop = get_realtime_feedback_loop()
        
        # Track asynchronously
        background_tasks.add_task(
            feedback_loop.track_view,
            request.user_id,
            request.item_id,
            request.item_type,
            request.dwell_time,
            request.session_id
        )
        
        return FeedbackResponse(
            success=True,
            message="View event tracked successfully",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error tracking view: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track/click", response_model=FeedbackResponse)
async def track_click(
    request: ClickEventRequest,
    background_tasks: BackgroundTasks
):
    """
    Track a click event
    Records when a user clicks on an item
    """
    try:
        feedback_loop = get_realtime_feedback_loop()
        
        background_tasks.add_task(
            feedback_loop.track_click,
            request.user_id,
            request.item_id,
            request.item_type,
            request.position,
            request.session_id
        )
        
        return FeedbackResponse(
            success=True,
            message="Click event tracked successfully",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error tracking click: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track/rating", response_model=FeedbackResponse)
async def track_rating(
    request: RatingEventRequest,
    background_tasks: BackgroundTasks
):
    """
    Track a rating event
    Records when a user rates an item
    """
    try:
        feedback_loop = get_realtime_feedback_loop()
        
        background_tasks.add_task(
            feedback_loop.track_rating,
            request.user_id,
            request.item_id,
            request.item_type,
            request.rating,
            request.session_id
        )
        
        return FeedbackResponse(
            success=True,
            message="Rating event tracked successfully",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error tracking rating: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track/save", response_model=FeedbackResponse)
async def track_save(
    request: SaveEventRequest,
    background_tasks: BackgroundTasks
):
    """
    Track a save/bookmark event
    Records when a user saves an item for later
    """
    try:
        feedback_loop = get_realtime_feedback_loop()
        
        background_tasks.add_task(
            feedback_loop.track_save,
            request.user_id,
            request.item_id,
            request.item_type,
            request.session_id
        )
        
        return FeedbackResponse(
            success=True,
            message="Save event tracked successfully",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error tracking save: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track/conversion", response_model=FeedbackResponse)
async def track_conversion(
    request: ConversionEventRequest,
    background_tasks: BackgroundTasks
):
    """
    Track a conversion event
    Records when a user actually visits the recommended place
    """
    try:
        feedback_loop = get_realtime_feedback_loop()
        
        background_tasks.add_task(
            feedback_loop.track_conversion,
            request.user_id,
            request.item_id,
            request.item_type,
            request.session_id
        )
        
        return FeedbackResponse(
            success=True,
            message="Conversion event tracked successfully",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error tracking conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(request: PersonalizedRecommendationRequest):
    """
    Get personalized hidden gem recommendations
    Uses real-time learning to personalize results based on user history
    """
    try:
        handler = HiddenGemsHandler(enable_realtime_learning=True)
        
        recommendations = handler.get_personalized_recommendations(
            user_id=request.user_id,
            location=request.location,
            gem_type=request.gem_type,
            limit=request.limit,
            session_id=request.session_id
        )
        
        # Check if results are personalized
        is_personalized = any(rec.get('_personalized', False) for rec in recommendations)
        
        return RecommendationResponse(
            recommendations=recommendations,
            personalized=is_personalized,
            count=len(recommendations)
        )
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_feedback_metrics():
    """
    Get metrics about the feedback collection and learning system
    """
    try:
        feedback_loop = get_realtime_feedback_loop()
        metrics = feedback_loop.get_metrics()
        
        return MetricsResponse(
            feedback_collector=metrics.get('collector_metrics', {}),
            learning_engine=metrics.get('learning_engine_metrics', {}),
            status="running" if metrics.get('is_running') else "stopped"
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_feedback_loop():
    """Start the real-time feedback loop"""
    try:
        feedback_loop = get_realtime_feedback_loop()
        await feedback_loop.start()
        return {"message": "Feedback loop started successfully"}
    except Exception as e:
        logger.error(f"Error starting feedback loop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_feedback_loop():
    """Stop the real-time feedback loop"""
    try:
        feedback_loop = get_realtime_feedback_loop()
        await feedback_loop.stop()
        return {"message": "Feedback loop stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping feedback loop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
