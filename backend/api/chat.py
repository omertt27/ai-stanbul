"""
Chat Endpoints Module

All chat-related endpoints including ML chat, Pure LLM chat, and legacy chat
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import time
import logging

from database import get_db
from core.startup import startup_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat endpoints"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_location: Optional[Dict[str, float]] = Field(None, description="User GPS location")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")


class ChatResponse(BaseModel):
    """Response model for chat endpoints"""
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session identifier")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: Optional[float] = Field(None, description="Confidence score")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
    map_data: Optional[Dict[str, Any]] = Field(None, description="Map visualization data for routes")
    navigation_active: Optional[bool] = Field(None, description="Whether GPS navigation is active")
    navigation_data: Optional[Dict[str, Any]] = Field(None, description="GPS navigation state and instructions")


class MLChatRequest(BaseModel):
    """Request model for ML-powered chat"""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    user_location: Optional[Dict[str, float]] = Field(None, description="User location {lat, lon}")
    use_llm: Optional[bool] = Field(None, description="Override: Use LLM (None=use default)")
    language: str = Field(default="en", description="Response language (en/tr)")
    user_id: Optional[str] = Field(None, description="User ID for personalization")


class MLChatResponse(BaseModel):
    """Response model for ML-powered chat"""
    response: str = Field(..., description="Bot response text")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Confidence score")
    method: str = Field(..., description="Response generation method")
    context: List[Dict] = Field(default=[], description="Context items used")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    response_time: float = Field(..., description="Response time in seconds")
    ml_service_used: bool = Field(..., description="Whether ML service was used")


@router.post("/pure-llm", response_model=ChatResponse)
async def pure_llm_chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Pure LLM chat endpoint - uses only LLM for responses
    Now with GPS navigation and route planning support!
    """
    # Prepare user context with GPS location
    user_context = {
        'preferences': request.preferences or {},
    }
    
    # Add GPS location to context if available
    if request.user_location:
        user_context['gps'] = request.user_location
        user_context['location'] = request.user_location
        logger.info(f"📍 User GPS location: {request.user_location}")
    
    # First check if this is a GPS navigation command
    try:
        from services.ai_chat_route_integration import get_chat_route_handler
        
        handler = get_chat_route_handler()
        
        # Try to handle as GPS navigation command
        nav_result = handler.handle_gps_navigation_command(
            message=request.message,
            session_id=request.session_id or 'new',
            user_location=request.user_location
        )
        
        if nav_result:
            # This was a navigation command
            return ChatResponse(
                response=nav_result.get('message', ''),
                session_id=request.session_id or 'new',
                intent='gps_navigation',
                confidence=1.0,
                suggestions=_get_navigation_suggestions(nav_result),
                map_data=nav_result.get('navigation_data', {}).get('map_data'),
                navigation_active=nav_result.get('navigation_active', False),
                navigation_data=nav_result.get('navigation_data')
            )
        
        # Try to handle as route request (e.g., "how can I go to Taksim")
        route_result = handler.handle_route_request(
            message=request.message,
            user_context=user_context
        )
        
        if route_result:
            # This was a route request
            response_type = route_result.get('type', '')
            
            # Check if GPS permission is needed
            if response_type == 'gps_permission_required':
                return ChatResponse(
                    response=route_result.get('message', ''),
                    session_id=request.session_id or 'new',
                    intent='route_planning',
                    confidence=1.0,
                    suggestions=[
                        "Enable GPS and try again",
                        "Specify start location manually",
                        "Show me restaurants nearby"
                    ],
                    map_data={'request_gps': True, 'destination': route_result.get('destination')}
                )
            
            # Return route response
            return ChatResponse(
                response=route_result.get('message', ''),
                session_id=request.session_id or 'new',
                intent='route_planning',
                confidence=route_result.get('confidence', 1.0),
                suggestions=route_result.get('suggestions', []),
                map_data=route_result.get('map_data'),
                navigation_active=False
            )
            
    except Exception as e:
        logger.warning(f"Route/Navigation check failed: {e}")
    
    # Not a navigation command, proceed with normal LLM chat
    pure_llm_core = startup_manager.get_pure_llm_core()
    
    if not pure_llm_core:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pure LLM Handler not available"
        )
    
    try:
        start_time = time.time()
        
        # Process query through Pure LLM
        result = await pure_llm_core.process_query(
            query=request.message,
            user_location=request.user_location,
            session_id=request.session_id,
            language="en"
        )
        
        response_time = time.time() - start_time
        
        logger.info(f"Pure LLM response generated in {response_time:.2f}s")
        
        return ChatResponse(
            response=result.get('response', ''),
            session_id=result.get('session_id', request.session_id or 'new'),
            intent=result.get('intent'),
            confidence=result.get('confidence'),
            suggestions=result.get('suggestions', []),
            map_data=result.get('map_data'),  # Include map data for visualization
            navigation_active=result.get('navigation_active', False),
            navigation_data=result.get('navigation_data')
        )
        
    except Exception as e:
        logger.error(f"Pure LLM chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post("/ml", response_model=MLChatResponse)
async def ml_chat(
    request: MLChatRequest,
    db: Session = Depends(get_db)
):
    """
    ML-powered chat endpoint
    """
    try:
        from backend.ml_service_client import get_ml_answer
        
        start_time = time.time()
        
        # Call ML service
        ml_result = await get_ml_answer(
            query=request.message,
            use_llm=request.use_llm,
            language=request.language,
            user_id=request.user_id,
            user_location=request.user_location
        )
        
        response_time = time.time() - start_time
        
        return MLChatResponse(
            response=ml_result.get('answer', ''),
            intent=ml_result.get('intent', 'unknown'),
            confidence=ml_result.get('confidence', 0.0),
            method=ml_result.get('method', 'ml_service'),
            context=ml_result.get('context', []),
            suggestions=ml_result.get('suggestions', []),
            response_time=response_time,
            ml_service_used=True
        )
        
    except Exception as e:
        logger.error(f"ML chat error: {e}")
        
        # Fallback response
        return MLChatResponse(
            response="I'm here to help you explore Istanbul! What would you like to know?",
            intent="general",
            confidence=0.5,
            method="fallback",
            context=[],
            suggestions=[
                "Show me restaurants in Sultanahmet",
                "What are the must-see attractions?",
                "How do I get around Istanbul?"
            ],
            response_time=0.1,
            ml_service_used=False
        )


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Main chat endpoint - routes to appropriate handler
    Now with GPS navigation support!
    """
    # First check if this is a GPS navigation command
    try:
        from services.ai_chat_route_integration import get_chat_route_handler
        
        handler = get_chat_route_handler()
        
        # Try to handle as GPS navigation command
        nav_result = handler.handle_gps_navigation_command(
            message=request.message,
            session_id=request.session_id or 'new',
            user_location=request.user_location
        )
        
        if nav_result:
            # This was a navigation command
            return ChatResponse(
                response=nav_result.get('message', ''),
                session_id=request.session_id or 'new',
                intent='gps_navigation',
                confidence=1.0,
                suggestions=_get_navigation_suggestions(nav_result),
                map_data=nav_result.get('navigation_data', {}).get('map_data'),
                navigation_active=nav_result.get('navigation_active', False),
                navigation_data=nav_result.get('navigation_data')
            )
    except Exception as e:
        logger.warning(f"GPS navigation check failed: {e}")
    
    # Check if Pure LLM is enabled
    pure_llm_core = startup_manager.get_pure_llm_core()
    
    logger.info(f"🔍 Chat endpoint called - pure_llm_core exists: {pure_llm_core is not None}")
    if pure_llm_core:
        logger.info(f"🔍 LLM client exists: {hasattr(pure_llm_core, 'llm_client') and pure_llm_core.llm_client is not None}")
    
    if pure_llm_core:
        # Use Pure LLM
        return await pure_llm_chat(request, db)
    else:
        # Fallback with basic intent detection
        logger.warning("⚠️ Pure LLM Core not available, using basic fallback")
        
        query_lower = request.message.lower()
        
        # Try to detect transportation queries
        transportation_keywords = ['how', 'get', 'go', 'travel', 'kadikoy', 'taksim', 'sultanahmet', 
                                  'besiktas', 'uskudar', 'directions', 'way', 'route']
        is_transport = any(keyword in query_lower for keyword in transportation_keywords)
        
        if is_transport:
            response_text = (
                "I can help you with directions! However, the AI service is currently unavailable. "
                "Please try:\n\n"
                "1. Use the interactive map on our homepage\n"
                "2. Check Istanbul Metro/Tram routes\n"
                "3. Use Google Maps for real-time directions\n\n"
                "The AI service will be back shortly. Thank you for your patience!"
            )
            intent = "transportation"
        else:
            response_text = (
                "Welcome to Istanbul! The AI assistant is temporarily unavailable, "
                "but you can still explore our website for information about:\n\n"
                "• Restaurants and dining\n"
                "• Tourist attractions\n"
                "• Neighborhoods and districts\n"
                "• Transportation options\n\n"
                "Please try again in a moment!"
            )
            intent = "greeting"
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id or "new",
            intent=intent,
            confidence=0.3,
            suggestions=[
                "Show me restaurants near me",
                "What are the best attractions?",
                "How do I use public transport?"
            ]
        )


def _get_navigation_suggestions(nav_result: Dict[str, Any]) -> List[str]:
    """Generate contextual suggestions based on navigation state"""
    nav_type = nav_result.get('type', '')
    
    if nav_type == 'navigation_started':
        return [
            "What's next?",
            "Where am I?",
            "Stop navigation"
        ]
    elif nav_type == 'navigation_update':
        return [
            "What's next?",
            "Repeat instruction",
            "Stop navigation"
        ]
    elif nav_type == 'navigation_stopped' or nav_type == 'navigation_complete':
        return [
            "Navigate to Galata Tower",
            "Show nearby attractions",
            "Find restaurants nearby"
        ]
    elif nav_result.get('navigation_active'):
        return [
            "What's next?",
            "Navigation status",
            "Stop navigation"
        ]
    else:
        return [
            "Navigate to Blue Mosque",
            "Navigate to Taksim Square",
            "Where am I?"
        ]
