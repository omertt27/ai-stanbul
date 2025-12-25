"""
Feedback API Endpoints
======================
Endpoints for collecting user feedback on LLM responses
Used for model fine-tuning and quality improvement

Implements smart feedback processing:
- Behavioral signals dominate
- 👍/👎 only adjusts confidence
- Auto-buckets into Gold/Fail/Noise
- Generates preference pairs for DPO training

Production Features:
- Input validation with length limits
- Rate limiting protection
- Graceful error handling
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum
import logging
import time
from collections import defaultdict

from services.data_collection import log_user_feedback, get_collection_stats, export_training_dataset

logger = logging.getLogger(__name__)

# Import smart feedback processor
try:
    from services.smart_feedback_processor import (
        process_smart_feedback,
        get_feedback_insights,
        generate_dpo_pairs,
        get_smart_feedback_stats,
        DislikeReason,
        BehavioralSignals
    )
    SMART_FEEDBACK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Smart feedback processor not available: {e}")
    SMART_FEEDBACK_AVAILABLE = False

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


# ============================================
# PRODUCTION RATE LIMITING
# ============================================
class RateLimiter:
    """Simple in-memory rate limiter for feedback submissions"""
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit"""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        
        if len(self._requests[key]) >= self.max_requests:
            return False
        
        self._requests[key].append(now)
        return True

# 10 feedback submissions per minute per session
feedback_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)


class DislikeReasonEnum(str, Enum):
    """One-click reasons for 👎 feedback - high value signal!"""
    INCORRECT = "incorrect"
    UNCLEAR = "unclear"
    TOO_GENERIC = "too_generic"
    OUTDATED = "outdated"
    OFF_TOPIC = "off_topic"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    NONE = "none"


class FeedbackRequest(BaseModel):
    """User feedback on a chat response"""
    interaction_id: str
    feedback_type: str  # 'thumbs_up', 'thumbs_down', 'rating', 'comment'
    feedback_value: Optional[str] = None
    comment: Optional[str] = None
    dislike_reason: Optional[DislikeReasonEnum] = DislikeReasonEnum.NONE
    
    # Optional: behavioral signals (collected from frontend)
    session_continued: Optional[bool] = None
    time_on_response_seconds: Optional[float] = None
    copied_response: Optional[bool] = None


class SmartFeedbackRequest(BaseModel):
    """Enhanced feedback with behavioral signals and production validation"""
    interaction_id: str = Field(..., min_length=1, max_length=100)
    user_query: str = Field(..., max_length=2000)  # Limit query size
    llm_response: str = Field(..., max_length=10000)  # Limit response size
    thumbs_up: bool = False
    thumbs_down: bool = False
    dislike_reason: DislikeReasonEnum = DislikeReasonEnum.NONE
    
    # Behavioral signals
    session_continued: bool = False
    rephrase_within_30s: bool = False
    rephrase_within_10s: bool = False
    tool_or_link_used: bool = False
    time_on_response_seconds: float = Field(default=0.0, ge=0, le=3600)  # Max 1 hour
    follow_up_question: bool = False
    copied_response: bool = False
    shared_response: bool = False
    contradiction_detected: bool = False
    
    # Metadata
    language: str = Field(default="en", max_length=10)
    intent: str = Field(default="general", max_length=50)
    session_id: Optional[str] = Field(default=None, max_length=100)
    
    @validator('interaction_id', 'session_id', pre=True)
    def sanitize_ids(cls, v):
        """Sanitize ID fields to prevent injection"""
        if v is None:
            return v
        # Remove any potentially dangerous characters
        return ''.join(c for c in str(v) if c.isalnum() or c in '-_')


class StatsResponse(BaseModel):
    """Data collection statistics"""
    total_interactions: int
    positive_feedback: int
    negative_feedback: int
    feedback_rate: float
    positive_rate: float
    languages: dict
    intents: dict


class SmartStatsResponse(BaseModel):
    """Smart feedback statistics"""
    total_processed: int
    gold_count: int
    fail_count: int
    noise_count: int
    gold_rate: float
    fail_rate: float
    preference_pairs_generated: int
    dislike_reasons: dict


@router.post("/submit")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on a chat response
    
    Feedback types:
    - thumbs_up: User liked the response
    - thumbs_down: User disliked the response  
    - rating: User rated 1-5 stars
    - comment: User provided text feedback
    
    NEW: Include dislike_reason for 👎 to multiply feedback value!
    Options: incorrect, unclear, too_generic, outdated, off_topic, too_long, too_short
    """
    try:
        # Log to basic feedback system
        log_user_feedback(
            interaction_id=request.interaction_id,
            feedback_type=request.feedback_type,
            feedback_value=request.feedback_value,
            comment=request.comment,
            dislike_reason=request.dislike_reason.value if request.dislike_reason else None
        )
        
        logger.info(
            f"✅ Feedback received: {request.feedback_type} for {request.interaction_id}"
            f"{f' (reason: {request.dislike_reason.value})' if request.dislike_reason and request.dislike_reason != DislikeReasonEnum.NONE else ''}"
        )
        
        return {
            "status": "success",
            "message": "Feedback recorded",
            "interaction_id": request.interaction_id,
            "dislike_reason": request.dislike_reason.value if request.dislike_reason else None
        }
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")


@router.post("/submit-smart")
async def submit_smart_feedback(request: SmartFeedbackRequest, http_request: Request):
    """
    Submit enhanced feedback with behavioral signals.
    
    This endpoint uses the smart feedback processor that:
    1. Treats 👍/👎 as weak signals
    2. Uses behavioral signals as primary quality indicator
    3. Auto-buckets into Gold/Fail/Noise
    4. Only actionable feedback goes to training
    
    Production features:
    - Rate limited (10 requests/minute per session)
    - Input validation with size limits
    - Graceful error handling
    """
    if not SMART_FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Smart feedback processor not available"
        )
    
    # Rate limiting check
    rate_limit_key = request.session_id or http_request.client.host if http_request.client else 'unknown'
    if not feedback_rate_limiter.is_allowed(rate_limit_key):
        logger.warning(f"⚠️ Rate limit exceeded for {rate_limit_key}")
        raise HTTPException(
            status_code=429,
            detail="Too many feedback submissions. Please wait before trying again."
        )
    
    try:
        # Create behavioral signals object
        behavior = BehavioralSignals(
            session_continued=request.session_continued,
            rephrase_within_30s=request.rephrase_within_30s,
            rephrase_within_10s=request.rephrase_within_10s,
            tool_or_link_used=request.tool_or_link_used,
            time_on_response_seconds=request.time_on_response_seconds,
            follow_up_question=request.follow_up_question,
            copied_response=request.copied_response,
            shared_response=request.shared_response,
            contradiction_detected=request.contradiction_detected
        )
        
        # Process with smart feedback system
        entry = process_smart_feedback(
            interaction_id=request.interaction_id,
            user_query=request.user_query,
            llm_response=request.llm_response,
            thumbs_up=request.thumbs_up,
            thumbs_down=request.thumbs_down,
            dislike_reason=request.dislike_reason.value,
            behavior=behavior,
            language=request.language,
            intent=request.intent,
            session_id=request.session_id
        )
        
        return {
            "status": "success",
            "message": "Smart feedback processed",
            "interaction_id": request.interaction_id,
            "base_score": entry.base_quality_score,
            "adjusted_score": entry.adjusted_score,
            "quality_bucket": entry.quality_bucket,
            "is_actionable": entry.quality_bucket in ["gold", "fail"]
        }
        
    except Exception as e:
        logger.error(f"Failed to process smart feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process feedback")


@router.get("/dislike-reasons")
async def get_dislike_reasons():
    """
    Get available dislike reasons for the UI.
    
    Display these as one-click options when user clicks 👎
    This multiplies feedback value without manual review!
    """
    return {
        "reasons": [
            {"id": "incorrect", "label": "❌ Incorrect", "description": "The answer was wrong or inaccurate"},
            {"id": "unclear", "label": "😕 Unclear", "description": "Hard to understand or confusing"},
            {"id": "too_generic", "label": "📋 Too Generic", "description": "Not specific enough to Istanbul"},
            {"id": "outdated", "label": "📅 Outdated", "description": "Information is no longer current"},
            {"id": "off_topic", "label": "🎯 Off Topic", "description": "Didn't answer my question"},
            {"id": "too_long", "label": "📏 Too Long", "description": "Response was too verbose"},
            {"id": "too_short", "label": "📝 Too Short", "description": "Needed more detail"},
        ]
    }


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


@router.get("/smart-stats")
async def get_smart_stats():
    """
    Get smart feedback statistics.
    
    Shows:
    - Gold/Fail/Noise bucket distribution
    - Dislike reason breakdown
    - Preference pairs generated for DPO
    """
    if not SMART_FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Smart feedback processor not available"
        )
    
    try:
        stats = get_smart_feedback_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get smart stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.get("/insights")
async def get_insights():
    """
    Get actionable insights from feedback analysis.
    
    Identifies:
    - Intents with high failure rates (need prompt/RAG fixes)
    - Common dislike reasons (need specific improvements)  
    - Recommendations for automatic fixes
    
    Use this to:
    1. Adjust RAG retrieval ranking
    2. Fix prompt constraints
    3. Update knowledge base
    
    NO TRAINING NEEDED for these fixes!
    """
    if not SMART_FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Smart feedback processor not available"
        )
    
    try:
        insights = get_feedback_insights()
        return insights
    except Exception as e:
        logger.error(f"Failed to get insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get insights")


@router.post("/generate-dpo-pairs")
async def generate_preference_pairs():
    """
    Generate preference pairs for DPO (Direct Preference Optimization) training.
    
    Creates pairs of:
    - (prompt, chosen=👍 high-score answer, rejected=👎 low-score answer)
    
    Requirements:
    - Need repeated 👍 answers (Gold bucket)
    - Need repeated 👎 answers (Fail bucket)
    - Same intent/language pattern
    
    Output: training_data/preference_pairs.jsonl
    
    This is perfect for preference tuning your LLM!
    """
    if not SMART_FEEDBACK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Smart feedback processor not available"
        )
    
    try:
        count = generate_dpo_pairs()
        return {
            "status": "success",
            "pairs_generated": count,
            "output_file": "training_data/preference_pairs.jsonl",
            "message": f"Generated {count} preference pairs for DPO training"
        }
    except Exception as e:
        logger.error(f"Failed to generate DPO pairs: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate pairs")


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
