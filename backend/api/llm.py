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

router = APIRouter(prefix="/api/v1/llm", tags=["RunPod LLM"])


# Request/Response Models
class LLMTestRequest(BaseModel):
    """Request model for LLM testing"""
    prompt: str = Field(..., description="Prompt for LLM generation")
    max_tokens: Optional[int] = Field(2500, description="Maximum tokens to generate")


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
        from services.runpod_llm_client import get_llm_client
        
        llm_client = get_llm_client()
        
        if not llm_client or not llm_client.enabled:
            return {
                "status": "unavailable",
                "message": "LLM client disabled or not configured",
                "endpoint": os.getenv("LLM_API_URL", "Not configured")
            }
        
        health = await llm_client.health_check()
        return health
        
    except ImportError as ie:
        logger.error(f"LLM import error: {ie}")
        return {
            "status": "unavailable",
            "message": f"RunPod LLM client import failed: {str(ie)}",
            "endpoint": os.getenv("LLM_API_URL", "Not configured")
        }
    except Exception as e:
        logger.error(f"LLM health check error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "endpoint": os.getenv("LLM_API_URL", "Not configured")
        }


@router.post("/generate", response_model=LLMTestResponse)
async def llm_generate_test(request: LLMTestRequest):
    """Test RunPod LLM generation"""
    try:
        from services.runpod_llm_client import get_llm_client
        
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
        from services.runpod_llm_client import generate_llm_response
        
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


# ===== ANALYTICS ENDPOINTS FOR ADMIN DASHBOARD =====

@router.get("/stats")
async def get_general_stats():
    """
    Get general LLM statistics for admin dashboard.
    Returns: total queries, cache performance, LLM calls, etc.
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core or not pure_llm_core.analytics:
            return {
                "status": "unavailable",
                "message": "Analytics system not initialized",
                "total_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "llm_calls": 0,
                "cache_hit_rate": 0.0,
                "unique_users": 0,
                "error_rate": 0.0
            }
        
        # Get comprehensive summary from analytics
        summary = pure_llm_core.analytics.get_summary()
        basic_stats = summary['basic_stats']
        user_stats = summary['users']
        error_stats = summary['errors']
        cache_efficiency = pure_llm_core.analytics.get_cache_efficiency()
        
        return {
            "status": "success",
            "total_queries": basic_stats['total_queries'],
            "cache_hits": basic_stats['cache_hits'],
            "cache_misses": basic_stats['cache_misses'],
            "llm_calls": basic_stats['llm_calls'],
            "cache_hit_rate": round(cache_efficiency['hit_rate'] * 100, 2),
            "unique_users": user_stats['unique_users'],
            "error_rate": round(error_stats['error_rate'] * 100, 2),
            "validation_failures": basic_stats.get('validation_failures', 0),
            "avg_queries_per_user": round(user_stats['avg_queries_per_user'], 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting general stats: {e}")
        return {
            "status": "error",
            "message": str(e),
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0,
            "cache_hit_rate": 0.0
        }


@router.get("/stats/signals")
async def get_signal_stats(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    signal_type: Optional[str] = None,
    language: Optional[str] = None
):
    """
    Get signal detection statistics.
    Returns: signal detection counts, confidence scores, accuracy
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core or not pure_llm_core.analytics:
            return {
                "status": "unavailable",
                "total_detections": 0,
                "by_signal": {},
                "multi_intent_queries": 0,
                "multi_intent_rate": 0.0
            }
        
        summary = pure_llm_core.analytics.get_summary()
        signal_stats = summary['signals']
        
        return {
            "status": "success",
            "total_detections": sum(signal_stats['detections_by_signal'].values()),
            "by_signal": signal_stats['detections_by_signal'],
            "multi_intent_queries": signal_stats['multi_intent_queries'],
            "multi_intent_rate": round(signal_stats['multi_intent_rate'] * 100, 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting signal stats: {e}")
        return {
            "status": "error",
            "message": str(e),
            "total_detections": 0,
            "by_signal": {}
        }


@router.get("/stats/performance")
async def get_performance_stats(hours: int = 24):
    """
    Get performance statistics (response times, latencies).
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core or not pure_llm_core.analytics:
            return {
                "status": "unavailable",
                "avg_query_latency": 0.0,
                "avg_llm_latency": 0.0,
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0
            }
        
        summary = pure_llm_core.analytics.get_summary()
        performance = summary['performance']
        
        query_stats = performance['query_latency']
        llm_stats = performance['llm_latency']
        
        return {
            "status": "success",
            "query_latency": {
                "avg": round(query_stats['avg'] * 1000, 2) if query_stats['avg'] > 0 else 0,  # Convert to ms
                "p50": round(query_stats['p50'] * 1000, 2) if query_stats['p50'] > 0 else 0,
                "p95": round(query_stats['p95'] * 1000, 2) if query_stats['p95'] > 0 else 0,
                "p99": round(query_stats['p99'] * 1000, 2) if query_stats['p99'] > 0 else 0,
                "min": round(query_stats['min'] * 1000, 2) if query_stats['min'] > 0 else 0,
                "max": round(query_stats['max'] * 1000, 2) if query_stats['max'] > 0 else 0
            },
            "llm_latency": {
                "avg": round(llm_stats['avg'] * 1000, 2) if llm_stats['avg'] > 0 else 0,
                "p50": round(llm_stats['p50'] * 1000, 2) if llm_stats['p50'] > 0 else 0,
                "p95": round(llm_stats['p95'] * 1000, 2) if llm_stats['p95'] > 0 else 0,
                "p99": round(llm_stats['p99'] * 1000, 2) if llm_stats['p99'] > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        return {
            "status": "error",
            "message": str(e),
            "avg_query_latency": 0.0
        }


@router.get("/stats/cache")
async def get_cache_stats():
    """
    Get cache performance statistics.
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core or not pure_llm_core.analytics:
            return {
                "status": "unavailable",
                "total_hits": 0,
                "total_misses": 0,
                "hit_rate": 0.0,
                "total_entries": 0
            }
        
        cache_efficiency = pure_llm_core.analytics.get_cache_efficiency()
        
        return {
            "status": "success",
            "total_hits": cache_efficiency['hits'],
            "total_misses": cache_efficiency['misses'],
            "hit_rate": round(cache_efficiency['hit_rate'] * 100, 2),
            "miss_rate": round(cache_efficiency['miss_rate'] * 100, 2),
            "total_checks": cache_efficiency['total_checks']
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {
            "status": "error",
            "message": str(e),
            "total_hits": 0,
            "total_misses": 0
        }


@router.get("/stats/users")
async def get_user_stats(days: int = 7):
    """
    Get user behavior statistics.
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core or not pure_llm_core.analytics:
            return {
                "status": "unavailable",
                "total_users": 0,
                "active_users": 0,
                "queries_by_language": {},
                "avg_queries_per_user": 0.0
            }
        
        summary = pure_llm_core.analytics.get_summary()
        user_stats = summary['users']
        
        return {
            "status": "success",
            "total_users": user_stats['unique_users'],
            "active_users": user_stats['unique_users'],  # All are active in current session
            "queries_by_language": user_stats['queries_by_language'],
            "avg_queries_per_user": round(user_stats['avg_queries_per_user'], 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return {
            "status": "error",
            "message": str(e),
            "total_users": 0
        }


@router.get("/stats/errors")
async def get_error_stats():
    """
    Get error tracking statistics.
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core or not pure_llm_core.analytics:
            return {
                "status": "unavailable",
                "total_errors": 0,
                "error_rate": 0.0,
                "by_type": {},
                "recent_errors": []
            }
        
        summary = pure_llm_core.analytics.get_summary()
        error_stats = summary['errors']
        
        return {
            "status": "success",
            "total_errors": error_stats['total'],
            "error_rate": round(error_stats['error_rate'] * 100, 2),
            "by_type": error_stats['by_type'],
            "by_service": error_stats.get('by_service', {}),
            "recent_errors": error_stats['recent']
        }
        
    except Exception as e:
        logger.error(f"Error getting error stats: {e}")
        return {
            "status": "error",
            "message": str(e),
            "total_errors": 0
        }


@router.get("/stats/hourly")
async def get_hourly_trends(hours: int = 24):
    """
    Get hourly trend data for charts.
    """
    try:
        from core.startup import startup_manager
        from datetime import datetime, timedelta
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core or not pure_llm_core.analytics:
            # Return empty but properly formatted data
            now = datetime.now()
            return {
                "status": "unavailable",
                "labels": [
                    (now - timedelta(hours=hours-i-1)).strftime("%H:%M") 
                    for i in range(hours)
                ],
                "queries": [0] * hours,
                "errors": [0] * hours,
                "avg_latency": [0.0] * hours
            }
        
        summary = pure_llm_core.analytics.get_summary()
        hourly_trends = summary['hourly_trends']
        
        # Extract data for charts
        labels = [trend['hour'].split(' ')[1] for trend in hourly_trends]  # Get just time part
        queries = [trend['queries'] for trend in hourly_trends]
        errors = [trend['errors'] for trend in hourly_trends]
        latencies = [round(trend['avg_latency'] * 1000, 2) for trend in hourly_trends]  # Convert to ms
        
        return {
            "status": "success",
            "labels": labels,
            "queries": queries,
            "errors": errors,
            "avg_latency": latencies,
            "error_rates": [round(trend['error_rate'] * 100, 2) for trend in hourly_trends]
        }
        
    except Exception as e:
        logger.error(f"Error getting hourly trends: {e}")
        from datetime import datetime, timedelta
        now = datetime.now()
        return {
            "status": "error",
            "message": str(e),
            "labels": [
                (now - timedelta(hours=hours-i-1)).strftime("%H:%M") 
                for i in range(hours)
            ],
            "queries": [0] * hours,
            "errors": [0] * hours
        }


@router.get("/stats/export")
async def export_stats(format: str = "json"):
    """
    Export statistics in JSON or CSV format.
    """
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core or not pure_llm_core.analytics:
            return {
                "status": "unavailable",
                "message": "Analytics not available"
            }
        
        stats = pure_llm_core.analytics.get_analytics()
        
        if format.lower() == "csv":
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow(["Metric", "Value"])
            for key, value in stats.items():
                if isinstance(value, (int, float, str)):
                    writer.writerow([key, value])
            
            return {
                "status": "success",
                "format": "csv",
                "data": output.getvalue()
            }
        else:
            return {
                "status": "success",
                "format": "json",
                "data": stats
            }
        
    except Exception as e:
        logger.error(f"Error exporting stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
