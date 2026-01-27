"""
Unified Chat Endpoint - Routes to Pure LLM Endpoint
====================================================

This endpoint routes all /api/chat requests to the fully-featured
/api/v1/chat/pure-llm endpoint which includes:

- Transportation RAG with route_data and map_data
- Multi-route alternatives
- Proper pattern-based location extraction
- Full context building and enhancement
- Response sanitization
- Caching and rate limiting

Created: October 30, 2025
Updated: January 22, 2026 - Simplified to route to pure-llm
Author: ML Systems Integration Team
"""

import sys
import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Initialize router
router = APIRouter(prefix="/api", tags=["Chat"])
logger = logging.getLogger(__name__)

# Request model for this endpoint
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    gps_location: Optional[dict] = None
    language: Optional[str] = "en"  # Response language (en/tr/etc.)


@router.post("/chat")
async def unified_chat(request: ChatRequest):
    """
    Unified chat endpoint - ROUTES TO PURE LLM ENDPOINT
    
    This endpoint routes to /api/v1/chat/pure-llm which has the complete
    implementation including:
    - Transportation RAG with route_data and map_data
    - Multi-route alternatives
    - Proper pattern-based location extraction
    - Full context building and enhancement
    
    Args:
        request (ChatRequest): User message with optional metadata
        
    Returns:
        Full ChatResponse from pure-llm endpoint with route_data support
    """
    # Import the pure-llm endpoint
    from backend.api.chat import pure_llm_chat, ChatRequest as PureLLMRequest
    
    # Convert request to pure-llm format
    # Pass the language from the request (frontend sends this)
    pure_llm_request = PureLLMRequest(
        message=request.message,
        session_id=request.session_id,
        user_id=request.user_id,
        user_location=request.gps_location,
        language=request.language or 'en'  # Use language from request, default to 'en'
    )
    
    logger.info(f"🔀 Routing /api/chat to /api/v1/chat/pure-llm for query: {request.message[:50]}... (language: {request.language})")
    
    # Call the pure-llm endpoint directly
    return await pure_llm_chat(pure_llm_request)


@router.get("/chat/health")
async def chat_health():
    """
    Health check for chat system
    
    Returns system status and available features
    """
    # Check if LLM Core is available from startup manager
    llm_core_ready = False
    try:
        from core.startup_fixed import fast_startup_manager
        llm_core = fast_startup_manager.get_pure_llm_core()
        llm_core_ready = llm_core is not None
    except Exception:
        pass
    
    return {
        "status": "healthy" if llm_core_ready else "degraded",
        "features": {
            "llm_core": llm_core_ready,
            "transportation_rag": llm_core_ready,
            "routing_to_pure_llm": True
        },
        "timestamp": datetime.now().isoformat()
    }
