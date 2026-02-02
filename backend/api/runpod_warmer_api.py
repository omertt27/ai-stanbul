"""
RunPod Warmer API Endpoints

Provides monitoring and control for the RunPod warming service.

Author: AI Istanbul Team
Date: February 2025
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runpod", tags=["RunPod Warmer"])


@router.get("/warmer/status")
async def get_warmer_status() -> Dict[str, Any]:
    """Get the current status of the RunPod warmer service"""
    try:
        from services.runpod_warmer import get_runpod_warmer
        
        warmer = get_runpod_warmer()
        status = warmer.get_status()
        
        # Add additional context
        status["description"] = "RunPod warmer prevents cold starts by sending periodic requests"
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting warmer status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warmer/warmup")
async def manual_warmup() -> Dict[str, Any]:
    """Manually trigger a warmup request (for testing)"""
    try:
        from services.runpod_warmer import get_runpod_warmer
        
        warmer = get_runpod_warmer()
        result = await warmer.manual_warmup()
        
        return result
        
    except Exception as e:
        logger.error(f"Error during manual warmup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/warmer/health")
async def check_runpod_health() -> Dict[str, Any]:
    """Check RunPod service health without warming"""
    try:
        from services.runpod_llm_client import RunPodLLMClient
        
        client = RunPodLLMClient()
        
        if not client.enabled:
            return {
                "status": "disabled",
                "message": "RunPod client not configured"
            }
        
        health_result = await client.health_check()
        
        return {
            "runpod_health": health_result,
            "timestamp": health_result.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"Error checking RunPod health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/warmer/is_warm")
async def is_runpod_warm() -> Dict[str, Any]:
    """Check if RunPod is currently warm (recently warmed)"""
    try:
        from services.runpod_warmer import is_runpod_warm, get_runpod_warmer
        
        warm = is_runpod_warm()
        warmer = get_runpod_warmer()
        
        return {
            "is_warm": warm,
            "last_warmup": warmer.last_warmup.isoformat() if warmer.last_warmup else None,
            "threshold": "60 seconds",
            "recommendation": "Cold start likely" if not warm else "Fast response expected"
        }
        
    except Exception as e:
        logger.error(f"Error checking warm status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
