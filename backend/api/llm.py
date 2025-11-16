"""
LLM Test Endpoints Module

RunPod LLM testing and diagnostic endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["RunPod LLM"])


# Request/Response Models
class LLMTestRequest(BaseModel):
    """Request model for LLM testing"""
    prompt: str = Field(..., description="Prompt for LLM generation")
    max_tokens: Optional[int] = Field(250, description="Maximum tokens to generate")


class LLMTestResponse(BaseModel):
    """Response model for LLM testing"""
    success: bool
    generated_text: Optional[str] = None
    error: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None


@router.get("/health")
async def llm_health_check():
    """Check RunPod LLM service health"""
    try:
        from backend.services.runpod_llm_client import get_llm_client
        
        llm_client = get_llm_client()
        health = await llm_client.health_check()
        return health
        
    except ImportError:
        return {
            "status": "unavailable",
            "message": "RunPod LLM client not loaded",
            "endpoint": os.getenv("LLM_API_URL", "Not configured")
        }
    except Exception as e:
        logger.error(f"LLM health check error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/generate", response_model=LLMTestResponse)
async def llm_generate_test(request: LLMTestRequest):
    """Test RunPod LLM generation"""
    try:
        from backend.services.runpod_llm_client import get_llm_client
        
        llm_client = get_llm_client()
        result = await llm_client.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )
        
        if result and 'generated_text' in result:
            return LLMTestResponse(
                success=True,
                generated_text=result['generated_text'],
                model="Llama 3.1 8B (4-bit)",
                endpoint=llm_client.api_url
            )
        else:
            return LLMTestResponse(
                success=False,
                error="No response from LLM"
            )
            
    except ImportError:
        return LLMTestResponse(
            success=False,
            error="RunPod LLM client not available"
        )
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return LLMTestResponse(
            success=False,
            error=str(e)
        )


@router.post("/istanbul-query", response_model=LLMTestResponse)
async def llm_istanbul_query(request: LLMTestRequest):
    """Generate Istanbul-specific response using RunPod LLM"""
    try:
        from backend.services.runpod_llm_client import generate_llm_response
        
        response_text = await generate_llm_response(
            query=request.prompt,
            context=None,
            intent="general"
        )
        
        if response_text:
            return LLMTestResponse(
                success=True,
                generated_text=response_text,
                model="Llama 3.1 8B (4-bit)"
            )
        else:
            return LLMTestResponse(
                success=False,
                error="No response generated"
            )
            
    except ImportError:
        return LLMTestResponse(
            success=False,
            error="RunPod LLM client not available"
        )
    except Exception as e:
        logger.error(f"Istanbul LLM query error: {e}")
        return LLMTestResponse(
            success=False,
            error=str(e)
        )


# ===================================================================
# PHASE 2: FEEDBACK & PERSONALIZATION ENDPOINTS
# ===================================================================

class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="System response")
    feedback_type: str = Field(..., description="Type: positive, negative, or correction")
    detected_signals: list[str] = Field(default=[], description="Signals that were detected")
    signal_scores: dict[str, float] = Field(default={}, description="Signal confidence scores")
    feedback_details: Optional[dict] = Field(None, description="Additional feedback details")


class InteractionRequest(BaseModel):
    """Request model for user interaction tracking"""
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., description="User's query")
    selected_items: list[dict] = Field(..., description="Items user interacted with")
    signals: list[str] = Field(default=[], description="Detected signals")


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for continuous improvement (Phase 2).
    
    This endpoint processes user feedback to:
    - Update user preferences
    - Adjust signal detection thresholds
    - Improve personalization
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core:
            raise HTTPException(status_code=503, detail="Pure LLM Core not initialized")
        
        result = await pure_llm_core.process_user_feedback(
            user_id=request.user_id,
            query=request.query,
            response=request.response,
            feedback_type=request.feedback_type,
            detected_signals=request.detected_signals,
            signal_scores=request.signal_scores,
            feedback_details=request.feedback_details
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Feedback processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interaction")
async def record_interaction(request: InteractionRequest):
    """
    Record user interaction for preference learning (Phase 2).
    
    This endpoint tracks which items users select/click to:
    - Learn cuisine preferences
    - Learn location preferences
    - Learn activity preferences
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core:
            raise HTTPException(status_code=503, detail="Pure LLM Core not initialized")
        
        result = await pure_llm_core.record_user_interaction(
            user_id=request.user_id,
            query=request.query,
            selected_items=request.selected_items,
            signals=request.signals
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profile/{user_id}")
async def get_user_profile(user_id: str):
    """
    Get user profile and preferences (Phase 2).
    
    Returns learned preferences including:
    - Preferred cuisines
    - Preferred districts
    - Interest categories
    - Interaction statistics
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core:
            raise HTTPException(status_code=503, detail="Pure LLM Core not initialized")
        
        profile = await pure_llm_core.get_user_profile(user_id)
        
        return {
            'status': 'success',
            'user_id': user_id,
            'profile': profile
        }
        
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tuning/run")
async def run_auto_tuning(signals: Optional[list[str]] = None):
    """
    Run auto-tuning for signal thresholds (Phase 2).
    
    This endpoint triggers automatic threshold adjustment based on
    accumulated feedback to optimize F1 scores.
    
    Args:
        signals: Specific signals to tune (optional, default = all)
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core:
            raise HTTPException(status_code=503, detail="Pure LLM Core not initialized")
        
        result = await pure_llm_core.run_auto_tuning(signals=signals)
        
        return result
        
    except Exception as e:
        logger.error(f"Auto-tuning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tuning/report")
async def get_tuning_report():
    """
    Get comprehensive tuning report (Phase 2).
    
    Returns:
    - Signal metrics (precision, recall, F1)
    - Threshold adjustment history
    - Overall tuning status
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core:
            raise HTTPException(status_code=503, detail="Pure LLM Core not initialized")
        
        report = await pure_llm_core.get_tuning_report()
        
        return report
        
    except Exception as e:
        logger.error(f"Tuning report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/personalization/metrics")
async def get_personalization_metrics():
    """
    Get personalization system metrics (Phase 2).
    
    Returns statistics about:
    - Total users
    - Personalized users
    - Feedback records
    - Satisfaction rates
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core:
            raise HTTPException(status_code=503, detail="Pure LLM Core not initialized")
        
        metrics = await pure_llm_core.personalization.get_personalization_metrics()
        feedback_summary = await pure_llm_core.personalization.get_feedback_summary(days=7)
        
        return {
            'status': 'success',
            'personalization': metrics,
            'feedback_summary': feedback_summary
        }
        
    except Exception as e:
        logger.error(f"Metrics retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
