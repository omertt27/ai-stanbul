"""
Chat Endpoints Module

All chat-related endpoints including ML chat, Pure LLM chat, and legacy chat
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import time
import logging
import json
import asyncio

from database import get_db
from core.startup import startup_manager
from services.data_collection import log_chat_interaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ==========================================
# RAG Service Integration
# ==========================================
_rag_service = None

def get_rag_service(db: Session = None):
    """Get or create RAG service singleton"""
    global _rag_service
    if _rag_service is None:
        try:
            from services.database_rag_service import get_rag_service as create_rag_service
            _rag_service = create_rag_service(db=db)
            logger.info("✅ RAG Service initialized")
        except Exception as e:
            logger.warning(f"⚠️  RAG Service not available: {e}")
            _rag_service = None
    return _rag_service


# ==========================================
# Phase 3: Response Enhancer Integration
# ==========================================
_response_enhancer = None

def get_response_enhancer():
    """Get or create Response Enhancer singleton"""
    global _response_enhancer
    if _response_enhancer is None:
        try:
            from services.llm import get_response_enhancer
            _response_enhancer = get_response_enhancer()
            logger.info("✅ Response Enhancer initialized")
        except Exception as e:
            logger.warning(f"⚠️  Response Enhancer not available: {e}")
            _response_enhancer = None
    return _response_enhancer


async def enhance_chat_response(
    base_response: str,
    original_query: str,
    user_context: Optional[Dict[str, Any]] = None,
    route_data: Optional[Dict[str, Any]] = None,
    response_type: str = "general"
) -> str:
    """
    Enhance response with LLM-generated contextual insights.
    
    This is Phase 3 of LLM Enhancement - adds intelligent tips to ALL responses.
    Falls back to original response if enhancer unavailable.
    """
    enhancer = get_response_enhancer()
    if not enhancer:
        return base_response
    
    try:
        result = await enhancer.enhance_response(
            base_response=base_response,
            original_query=original_query,
            user_context=user_context,
            route_data=route_data,
            response_type=response_type
        )
        
        # Extract enhanced response
        if hasattr(result, 'enhanced_response'):
            return result.enhanced_response
        elif hasattr(result, 'response'):
            return result.response
        else:
            return base_response
            
    except Exception as e:
        logger.warning(f"Response enhancement failed: {e}")
        return base_response
# ==========================================


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat endpoints"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_location: Optional[Dict[str, float]] = Field(None, description="User GPS location")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    user_id: Optional[str] = Field(None, description="User ID for personalization")


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
    interaction_id: Optional[str] = Field(None, description="Interaction ID for feedback tracking")


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
    Pure LLM chat endpoint - LLM-First Architecture with Full Pipeline
    
    Phase 4.2 Enhancement: Context Resolution runs FIRST to understand conversation flow
    
    Flow:
    0. LLM Conversation Context (Phase 4.2!) - Resolve references & context
    1. LLM Intent Classification (Phase 1) - Understand user intent
    2. LLM Location Resolution (Phase 2) - Resolve locations
    3. LLM Route Preferences (Phase 4.1) - Detect preferences
    4. Specialized handlers (routes, gems, info)
    5. LLM Response Enhancement (Phase 3) - Enhance final response
    """
    
    # Generate or use provided session_id
    session_id = request.session_id or f"session_{hash(request.message)}"
    
    # === PHASE 4.2: LLM CONVERSATION CONTEXT RESOLUTION (NEW!) ===
    # This runs FIRST to resolve pronouns, references, and conversation flow
    logger.info(f"💬 Phase 4.2: Conversation Context Resolution for session: {session_id}")
    
    resolved_context = None
    original_query = request.message
    
    try:
        from services.llm import get_context_manager
        
        # Get or create context manager (uses singleton pattern)
        pure_llm_core = startup_manager.get_pure_llm_core()
        if pure_llm_core:
            llm_client = pure_llm_core.llm_client if hasattr(pure_llm_core, 'llm_client') else None
            
            # DISABLED FOR SPEED: Context resolution adds 20-30s per request
            # Re-enable when we have a faster LLM or can run it in parallel
            ENABLE_CONTEXT_RESOLUTION = False
            
            if ENABLE_CONTEXT_RESOLUTION and llm_client:
                context_manager = get_context_manager(
                    llm_client=llm_client,
                    config={
                        'enable_llm': False,  # DISABLED: Too slow, using rule-based fallback
                        'fallback_to_rules': True,
                        'timeout_seconds': 2,
                        'max_history_turns': 10
                    }
                )
                
                # Resolve context using LLM
                resolved_context = await context_manager.resolve_context(
                    current_query=request.message,
                    session_id=session_id,
                    user_id=getattr(request, 'user_id', None) or session_id,
                    user_location=request.user_location
                )
                
                logger.info(
                    f"✅ Context Resolution complete:\n"
                    f"   - Has References: {resolved_context.get('has_references')}\n"
                    f"   - Confidence: {resolved_context.get('confidence', 0):.2f}\n"
                    f"   - Resolved Refs: {list(resolved_context.get('resolved_references', {}).keys())}\n"
                    f"   - Needs Clarification: {resolved_context.get('needs_clarification')}\n"
                    f"   - Source: {resolved_context.get('source')}"
                )
                
                # If we have a resolved query, use it for downstream processing
                if resolved_context.get('resolved_query'):
                    logger.info(f"   Original: '{original_query}'")
                    logger.info(f"   Resolved: '{resolved_context['resolved_query']}'")
                    request.message = resolved_context['resolved_query']
                
                # If clarification is needed, return early with question
                if resolved_context.get('needs_clarification') and resolved_context.get('clarification_question'):
                    logger.info(f"   ⚠️ Clarification needed: {resolved_context['clarification_question']}")
                    return ChatResponse(
                        response=resolved_context['clarification_question'],
                        intent="clarification",
                        confidence=resolved_context.get('confidence', 0.8),
                        method="context_clarification",
                        suggestions=[],
                        response_time=0.1,
                        session_id=session_id
                    )
        else:
            logger.warning("⚠️ Pure LLM Core not available, skipping context resolution")
            
    except Exception as e:
        logger.error(f"Context resolution error: {e}", exc_info=True)
        # Continue without context resolution - non-blocking
    
    # === PHASE 4.3: MULTI-INTENT DETECTION - DISABLED FOR PERFORMANCE ===
    # REMOVED: Multi-intent detection was taking 12-35 seconds per request
    # The Pure LLM Core is smart enough to handle multi-intent queries naturally
    # If needed in future, can be re-enabled with proper optimization
    logger.info(f"⚡ Skipping multi-intent detection - using Pure LLM for fast response")
    
    # === PHASE 1: LLM INTENT CLASSIFICATION - DISABLED FOR PERFORMANCE ===
    # REMOVED: Intent classification was taking 15-22 seconds per request
    # The Pure LLM Core already understands intent internally - no need for separate layer
    logger.info(f"⚡ Skipping intent classification - Pure LLM handles this naturally")
    
    # Prepare user context for downstream handlers
    user_context = {
        'preferences': request.preferences or {},
    }
    
    if request.user_location:
        user_context['gps'] = request.user_location
        user_context['location'] = request.user_location
        logger.info(f"📍 User GPS available: lat={request.user_location.get('lat')}, lon={request.user_location.get('lon')}")
    
    # Add resolved context if available
    if resolved_context:
        user_context['resolved_context'] = resolved_context.get('implicit_context', {})
    
    # No LLM intent classification - Pure LLM will handle everything
    llm_intent = None
    location_resolution = None
    
    # === PHASE 3: SPECIALIZED HANDLERS ===
    # Keep these - they're fast and useful for specific intents
    # Try hidden gems GPS request first
    try:
        from services.hidden_gems_gps_integration import get_hidden_gems_gps_integration
        
        gems_handler = get_hidden_gems_gps_integration(db)
        
        # Try to handle as hidden gem request
        gems_result = gems_handler.handle_hidden_gem_chat_request(
            message=request.message,
            user_location=request.user_location,
            session_id=request.session_id or 'new'
        )
        
        if gems_result:
            # This was a hidden gems request
            if gems_result.get('error'):
                # Error occurred (no enhancement for errors)
                return ChatResponse(
                    response=gems_result.get('message', 'Sorry, something went wrong with hidden gems.'),
                    session_id=request.session_id or 'new',
                    intent='hidden_gems',
                    confidence=0.8,
                    suggestions=["Show me restaurants", "What are popular attractions?"]
                )
            
            # Check if navigation was started
            if gems_result.get('navigation_active'):
                # Enhance navigation response
                enhanced_msg = await enhance_chat_response(
                    base_response=gems_result.get('message', ''),
                    original_query=request.message,
                    user_context=user_context,
                    route_data=gems_result.get('navigation_data'),
                    response_type="navigation"
                )
                
                return ChatResponse(
                    response=enhanced_msg,
                    session_id=request.session_id or 'new',
                    intent='hidden_gems_navigation',
                    confidence=1.0,
                    suggestions=["What's next?", "Stop navigation", "Show nearby hidden gems"],
                    map_data=gems_result.get('map_data'),
                    navigation_active=True,
                    navigation_data=gems_result.get('navigation_data')
                )
            
            # Return gems discovery response with enhancement
            gems = gems_result.get('gems', [])
            response_text = _format_hidden_gems_response(gems, request.user_location)
            
            # Phase 3: Enhance hidden gems response
            enhanced_response = await enhance_chat_response(
                base_response=response_text,
                original_query=request.message,
                user_context=user_context,
                response_type="hidden_gems"
            )
            
            return ChatResponse(
                response=enhanced_response,
                session_id=request.session_id or 'new',
                intent='hidden_gems',
                confidence=1.0,
                suggestions=_get_hidden_gems_suggestions(gems),
                map_data=gems_result.get('map_data'),
                navigation_active=False
            )
            
    except Exception as e:
        logger.warning(f"Hidden gems GPS check failed: {e}")
        
        # Check if this is a GPS navigation command
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
                # This was a navigation command - enhance with contextual tips
                enhanced_msg = await enhance_chat_response(
                    base_response=nav_result.get('message', ''),
                    original_query=request.message,
                    user_context=user_context,
                    route_data=nav_result.get('navigation_data'),
                    response_type="gps_navigation"
                )
                
                return ChatResponse(
                    response=enhanced_msg,
                    session_id=request.session_id or 'new',
                    intent='gps_navigation',
                    confidence=1.0,
                    suggestions=_get_navigation_suggestions(nav_result),
                    map_data=nav_result.get('navigation_data', {}).get('map_data'),
                    navigation_active=nav_result.get('navigation_active', False),
                    navigation_data=nav_result.get('navigation_data')
                )
            
            # Try to handle as route request (e.g., "how can I go to Taksim")
            logger.info(f"🔍 Checking if message is a route request: '{request.message}'")
            
            try:
                route_result = handler.handle_route_request(
                    message=request.message,
                    user_context=user_context
                )
                
                if route_result:
                    logger.info(f"✅ Route request detected! Result type: {route_result.get('type', 'unknown')}")
                    # This was a route request
                    response_type = route_result.get('type', '')
                    
                    # Check for errors
                    if response_type == 'error':
                        error_msg = route_result.get('message', 'Route planning error')
                        logger.error(f"❌ Route planning error: {error_msg}")
                        # Don't return error, fall through to Pure LLM for better UX
                    
                    # Check if GPS permission is needed
                    elif response_type == 'gps_permission_required':
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
                    
                    # Success - return route response with enhancement
                    elif response_type in ['route', 'multi_stop_itinerary']:
                        # Phase 3: Enhance route response with contextual tips
                        enhanced_msg = await enhance_chat_response(
                            base_response=route_result.get('message', ''),
                            original_query=request.message,
                            user_context=user_context,
                            route_data=route_result.get('route_data'),
                            response_type="route"
                        )
                        
                        return ChatResponse(
                            response=enhanced_msg,
                            session_id=request.session_id or 'new',
                            intent='route_planning',
                            confidence=route_result.get('confidence', 1.0),
                            suggestions=route_result.get('suggestions', []),
                            map_data=route_result.get('route_data'),  # Fixed: use route_data, not map_data
                            navigation_active=False
                        )
                else:
                    logger.info(f"❌ Not detected as a route request, will use Pure LLM")
                    
            except Exception as route_error:
                logger.error(f"Route handler error: {route_error}", exc_info=True)
                # Fall through to Pure LLM on error
                
        except Exception as e:
            logger.warning(f"Route/Navigation check failed: {e}", exc_info=True)
    
    # Not a navigation command, proceed with normal LLM chat
    pure_llm_core = startup_manager.get_pure_llm_core()
    
    if not pure_llm_core:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pure LLM Handler not available"
        )
    
    try:
        start_time = time.time()
        
        # === RAG ENHANCEMENT: Retrieve relevant context from database ===
        rag_context = None
        rag_used = False
        rag_metadata = {}
        try:
            rag_service = get_rag_service(db=db)
            if rag_service:
                logger.info(f"🔍 RAG: Searching for relevant context...")
                rag_results = rag_service.search(request.message, top_k=3)
                if rag_results:
                    rag_context = rag_service.get_context_for_llm(request.message, top_k=3)
                    rag_used = True
                    
                    # Store metadata about RAG results
                    rag_metadata = {
                        'count': len(rag_results),
                        'top_result': {
                            'type': rag_results[0]['metadata']['type'],
                            'name': rag_results[0]['metadata'].get('name', 'N/A'),
                            'score': rag_results[0]['relevance_score']
                        }
                    }
                    
                    logger.info(f"✅ RAG: Retrieved {len(rag_results)} relevant items")
                    logger.info(f"   Top result: {rag_metadata['top_result']['name']} ({rag_metadata['top_result']['type']}) [Score: {rag_metadata['top_result']['score']:.3f}]")
                    
                    # Store RAG context in user_context for downstream use
                    user_context['rag_context'] = rag_context
                    user_context['rag_results'] = rag_results
                else:
                    logger.info(f"ℹ️  RAG: No relevant results found")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            rag_context = None
        
        # Process query through Pure LLM
        logger.info("🚀 Processing query through Pure LLM")
        result = await pure_llm_core.process_query(
            query=request.message,
            user_location=request.user_location,
            session_id=request.session_id,
            language="en"
        )
        
        response_time = time.time() - start_time
        
        # If RAG was used, post-process the response to ensure it's grounded in the retrieved data
        if rag_used and rag_context:
            logger.info(f"� RAG: Post-processing response to ensure factual grounding")
            try:
                # Optionally re-prompt the LLM to be more specific with the RAG context
                # For now, just log that RAG was used
                logger.info(f"Pure LLM response generated in {response_time:.2f}s (RAG: ✓ {rag_metadata['count']} items)")
            except Exception as e:
                logger.warning(f"RAG post-processing failed: {e}")
        else:
            logger.info(f"Pure LLM response generated in {response_time:.2f}s (RAG: ✗)")
        
        # Phase 3: Enhance Pure LLM response with contextual intelligence
        enhanced_response = await enhance_chat_response(
            base_response=result.get('response', ''),
            original_query=request.message,
            user_context=user_context,
            route_data=result.get('map_data'),  # May contain route info
            response_type=result.get('intent', 'general')
        )
        
        # Extract locations from result for conversation tracking
        locations = []
        if llm_intent:
            if llm_intent.origin:
                locations.append(llm_intent.origin)
            if llm_intent.destination:
                locations.append(llm_intent.destination)
        
        # Record conversation turn for future context
        await record_conversation_turn(
            session_id=session_id,
            user_query=original_query,  # Use original query (before resolution)
            bot_response=enhanced_response,
            intent=result.get('intent'),
            locations=locations
        )
        
        # === PHASE 4.4: GENERATE PROACTIVE SUGGESTIONS ===
        # Generate intelligent suggestions for the user's next steps
        # Protected with timeout to prevent frontend hangs
        proactive_suggestions = None
        try:
            # Extract entities from LLM intent
            entities = {}
            if llm_intent:
                if llm_intent.origin:
                    entities['origin'] = llm_intent.origin
                if llm_intent.destination:
                    entities['destination'] = llm_intent.destination
                if llm_intent.extracted_locations:
                    entities['locations'] = llm_intent.extracted_locations
            
            # Get conversation history for context
            conversation_history = []
            if resolved_context and resolved_context.get('conversation_context'):
                conv_ctx = resolved_context['conversation_context']
                if conv_ctx.get('history'):
                    conversation_history = conv_ctx['history'][-5:]  # Last 5 turns
            
            # Generate suggestions with timeout protection (5 seconds max)
            try:
                proactive_suggestions = await asyncio.wait_for(
                    generate_proactive_suggestions(
                        query=original_query,
                        response=enhanced_response,
                        intent=result.get('intent'),
                        entities=entities,
                        conversation_history=conversation_history,
                        user_location=llm_intent.origin if llm_intent else None,
                        session_id=session_id
                    ),
                    timeout=5.0  # 5 second timeout for suggestion generation
                )
            except asyncio.TimeoutError:
                logger.warning("Suggestion generation timed out after 5s, using fallback")
                proactive_suggestions = None
            
            if proactive_suggestions:
                logger.info(f"✨ Added {len(proactive_suggestions)} proactive suggestions")
        except Exception as e:
            logger.warning(f"Proactive suggestion generation failed: {e}")
            proactive_suggestions = None
        
        # Use proactive suggestions if available, fallback to original or defaults
        final_suggestions = proactive_suggestions if proactive_suggestions else result.get('suggestions', [])
        
        # If no suggestions at all, provide helpful defaults
        if not final_suggestions:
            final_suggestions = [
                "What are the top attractions in Istanbul?",
                "Show me popular restaurants nearby",
                "How do I get around the city?",
                "Tell me about hidden gems",
                "What's the weather like today?"
            ]
        
        # If proactive suggestions are dict format, extract text
        if final_suggestions and isinstance(final_suggestions[0], dict):
            final_suggestions = [s.get('text', str(s)) for s in final_suggestions]
        
        # === DATA COLLECTION FOR MODEL FINE-TUNING ===
        try:
            interaction_id = log_chat_interaction(
                user_query=original_query,
                llm_response=enhanced_response,
                language=request.language or 'en',
                intent=result.get('intent'),
                response_time=int((time.time() - start_time) * 1000) if 'start_time' in locals() else None,
                cached=result.get('cached', False),
                method='pure_llm',
                has_map_data=result.get('map_data') is not None,
                session_id=session_id,
                confidence=result.get('confidence')
            )
            logger.debug(f"📝 Logged interaction {interaction_id} for training data")
        except Exception as e:
            logger.warning(f"Failed to log interaction for training: {e}")
            interaction_id = None
        
        return ChatResponse(
            response=enhanced_response,
            session_id=result.get('session_id', request.session_id or 'new'),
            intent=result.get('intent'),
            confidence=result.get('confidence'),
            suggestions=final_suggestions,
            map_data=result.get('map_data'),  # Include map data for visualization
            navigation_active=result.get('navigation_active', False),
            navigation_data=result.get('navigation_data'),
            interaction_id=interaction_id  # Include for frontend feedback tracking
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Pure LLM chat error: {e}")
        logger.error(f"Pure LLM chat error traceback: {traceback.format_exc()}")
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
        # Emergency Fallback: Pure LLM Core not available
        # No pattern matching - just provide helpful generic response
        logger.error("❌ Pure LLM Core not available - returning emergency fallback")
        
        response_text = (
            "🤖 I'm the Istanbul AI Assistant, but I'm currently offline for maintenance.\n\n"
            "While I'm away, you can:\n\n"
            "• 🗺️ Use the interactive map on our homepage\n"
            "• 🍽️ Browse restaurants and attractions\n"
            "• 🚇 Check transportation options\n"
            "• 📱 Enable GPS for location-based features\n\n"
            "I'll be back online shortly! Thank you for your patience. 🙏"
        )
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id or "new",
            intent="error_fallback",
            confidence=0.0,
            suggestions=[
                "Show me the map",
                "Browse restaurants",
                "View attractions",
                "Check back later"
            ]
        )


# ==============================================
# Hidden Gems Helper Functions
# ==============================================

def _format_hidden_gems_response(gems: List[Dict], user_location: Optional[Dict] = None) -> str:
    """
    Format hidden gems discovery response
    
    Args:
        gems: List of hidden gem dictionaries
        user_location: Optional user GPS location
        
    Returns:
        Formatted response text
    """
    if not gems:
        return "I couldn't find any hidden gems nearby. Try a different area or ask me about popular attractions!"
    
    response = f"🗺️ I found {len(gems)} amazing hidden gems for you:\n\n"
    
    for i, gem in enumerate(gems[:5], 1):  # Show top 5
        name = gem.get('name', 'Unknown')
        category = gem.get('category', 'attraction')
        description = gem.get('description', '')
        distance = gem.get('distance')
        
        response += f"{i}. **{name}**"
        
        # Add category emoji
        if 'cafe' in category.lower() or 'coffee' in category.lower():
            response += " ☕"
        elif 'restaurant' in category.lower() or 'food' in category.lower():
            response += " 🍽️"
        elif 'park' in category.lower() or 'garden' in category.lower():
            response += " 🌳"
        elif 'view' in category.lower():
            response += " 🌆"
        elif 'art' in category.lower() or 'gallery' in category.lower():
            response += " 🎨"
        else:
            response += " 💎"
        
        response += f" ({category})\n"
        
        if distance:
            response += f"   📍 {distance:.1f}km away\n"
        
        if description:
            # Truncate description
            desc_short = description[:100] + "..." if len(description) > 100 else description
            response += f"   {desc_short}\n"
        
        response += "\n"
    
    if user_location:
        response += "\n💡 Want to navigate to any of these? Just say \"Navigate to [name]\" or click the location on the map!"
    else:
        response += "\n💡 Enable GPS to see distances and get turn-by-turn navigation!"
    
    return response


def _get_hidden_gems_suggestions(gems: List[Dict]) -> List[str]:
    """
    Generate context-aware suggestions for hidden gems
    
    Args:
        gems: List of hidden gem dictionaries
        
    Returns:
        List of suggestion strings
    """
    suggestions = []
    
    # Add navigation suggestions for top gems
    if gems and len(gems) > 0:
        first_gem = gems[0].get('name', '')
        if first_gem:
            suggestions.append(f"Navigate to {first_gem}")
    
    if gems and len(gems) > 1:
        second_gem = gems[1].get('name', '')
        if second_gem:
            suggestions.append(f"Tell me about {second_gem}")
    
    # Add general suggestions
    suggestions.extend([
        "Show me more hidden gems",
        "Find nearby restaurants",
        "What else is around here?"
    ])
    
    return suggestions[:5]  # Return max 5 suggestions


def _check_hidden_gem_intent(message: str) -> bool:
    """
    Check if message is asking about hidden gems
    
    Args:
        message: User's message
        
    Returns:
        True if message is about hidden gems
    """
    message_lower = message.lower()
    
    hidden_gem_keywords = [
        'hidden gem', 'secret spot', 'local spot', 'off the beaten',
        'undiscovered', 'secret place', 'hidden place', 'local favorite',
        'insider tip', 'secret cafe', 'hidden cafe', 'secret restaurant',
        'gizli', 'saklı', 'yerel', 'bilinmeyen'  # Turkish keywords
    ]
    
    return any(keyword in message_lower for keyword in hidden_gem_keywords)


def _extract_hidden_gem_name_from_message(message: str, gems: List[Dict]) -> Optional[str]:
    """
    Extract hidden gem name from navigation request
    
    Args:
        message: User's message
        gems: List of available gems to match against
        
    Returns:
        Gem name if found, None otherwise
    """
    message_lower = message.lower()
    
    # Check each gem name
    for gem in gems:
        name = gem.get('name', '')
        if name and name.lower() in message_lower:
            return name
    
    return None


def _get_navigation_suggestions(nav_result: Dict) -> List[str]:
    """
    Generate context-aware suggestions for navigation
    
    Args:
        nav_result: Navigation result dictionary
        
    Returns:
        List of suggestion strings
    """
    suggestions = []
    
    # Check if navigation is active
    is_active = nav_result.get('navigation_active', False)
    nav_data = nav_result.get('navigation_data', {})
    
    if is_active:
        # Active navigation suggestions
        suggestions.extend([
            "What's the next turn?",
            "How much longer?",
            "Stop navigation",
            "Show alternative routes"
        ])
    else:
        # Route planning suggestions
        destination = nav_data.get('destination', '')
        if destination:
            suggestions.append(f"Start navigation to {destination}")
        
        suggestions.extend([
            "Show me nearby restaurants",
            "Find hidden gems nearby",
            "What else is around here?"
        ])
    
    return suggestions[:5]  # Return max 5 suggestions


async def record_conversation_turn(
    session_id: str,
    user_query: str,
    bot_response: str,
    intent: Optional[str] = None,
    locations: Optional[List[str]] = None
):
    """
    Record a conversation turn for context tracking.
    
    Args:
        session_id: Session identifier
        user_query: User's query
        bot_response: Bot's response
        intent: Detected intent
        locations: Mentioned locations
    """
    try:
        from services.llm import get_context_manager
        
        # Get context manager
        pure_llm_core = startup_manager.get_pure_llm_core()
        if pure_llm_core:
            llm_client = pure_llm_core.llm_client if hasattr(pure_llm_core, 'llm_client') else None
            
            if llm_client:
                context_manager = get_context_manager(llm_client=llm_client)
                
                # Record the turn
                context_manager.record_turn(
                    session_id=session_id,
                    user_query=user_query,
                    bot_response=bot_response,
                    intent=intent,
                    locations=locations or []
                )
                
                logger.debug(f"📝 Recorded conversation turn for session {session_id}")
    except Exception as e:
        logger.warning(f"Failed to record conversation turn: {e}")
        # Non-blocking - continue even if recording fails


# ==========================================
# Phase 4.4: Proactive Suggestions Integration
# ==========================================
_suggestion_analyzer = None
_suggestion_generator = None
_suggestion_presenter = None

def get_suggestion_components():
    """Get or create Suggestion system singletons"""
    global _suggestion_analyzer, _suggestion_generator, _suggestion_presenter
    
    if _suggestion_analyzer is None or _suggestion_generator is None or _suggestion_presenter is None:
        try:
            from services.llm import (
                get_suggestion_analyzer,
                get_suggestion_generator,
                get_suggestion_presenter
            )
            
            # Get LLM client
            pure_llm_core = startup_manager.get_pure_llm_core()
            llm_client = None
            if pure_llm_core and hasattr(pure_llm_core, 'llm_client'):
                llm_client = pure_llm_core.llm_client
            
            if llm_client:
                _suggestion_analyzer = get_suggestion_analyzer(
                    llm_client=llm_client,
                    config={
                        'use_llm': False,  # DISABLED: Adds 21 seconds!
                        'fallback_enabled': True
                    }
                )
                _suggestion_generator = get_suggestion_generator(
                    llm_client=llm_client,
                    config={
                        'use_llm': False,  # DISABLED: Adds 47 seconds!
                        'timeout_seconds': 2,
                        'fallback_enabled': True
                    }
                )
                _suggestion_presenter = get_suggestion_presenter()
                logger.info("✅ Phase 4.4 Proactive Suggestions initialized (heuristics-only mode)")
            else:
                logger.warning("⚠️ LLM client not available for suggestions")
                
        except Exception as e:
            logger.warning(f"⚠️ Proactive Suggestions not available: {e}")
    
    return _suggestion_analyzer, _suggestion_generator, _suggestion_presenter


async def generate_proactive_suggestions(
    query: str,
    response: str,
    intent: Optional[str] = None,
    entities: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict]] = None,
    user_location: Optional[str] = None,
    session_id: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Generate proactive suggestions for the user.
    
    This is Phase 4.4 of LLM Enhancement - intelligently suggests next steps.
    Falls back gracefully if suggestion system unavailable.
    
    Args:
        query: User's original query
        response: Bot's response
        intent: Detected intent type
        entities: Extracted entities
        conversation_history: Recent conversation turns
        user_location: User's location
        session_id: Session identifier
        
    Returns:
        List of suggestion dictionaries or None if unavailable
    """
    try:
        analyzer, generator, presenter = get_suggestion_components()
        
        if not (analyzer and generator and presenter):
            logger.debug("Suggestion system not available, skipping")
            return None
        
        # Analyze context to see if we should show suggestions
        context = await analyzer.analyze_context(
            query=query,
            response=response,
            conversation_history=conversation_history or [],
            detected_intents=[intent] if intent else [],
            entities=entities or {},
            response_type=intent or "general",
            user_location=user_location
        )
        
        # Check if we should show suggestions
        should_suggest, confidence = await analyzer.should_suggest(context)
        
        if not should_suggest:
            logger.debug(f"Not showing suggestions (confidence: {confidence:.2f})")
            return None
        
        logger.info(f"💡 Generating proactive suggestions (confidence: {confidence:.2f})")
        
        # Generate suggestions
        suggestion_response = await generator.generate_with_response(
            context=context,
            max_suggestions=5
        )
        
        # Format for chat API
        formatted = presenter.format_for_chat(suggestion_response)
        
        logger.info(
            f"✅ Generated {len(formatted['suggestions'])} suggestions "
            f"(method: {formatted['metadata']['generation_method']}, "
            f"time: {formatted['metadata']['generation_time_ms']:.0f}ms)"
        )
        
        return formatted['suggestions']
        
    except Exception as e:
        logger.error(f"Proactive suggestion generation failed: {e}", exc_info=True)
        return None


@router.post("/pure-llm/stream")
async def pure_llm_chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Pure LLM chat endpoint with STREAMING support for real-time UX.
    
    This endpoint provides Server-Sent Events (SSE) streaming for:
    - Real-time progress updates (enhancement, cache, signals, context)
    - Token-by-token response streaming
    - Improved perceived performance (TTFB < 1s)
    
    Response Format (SSE):
    - Each event is a JSON object with 'type' and 'data'/'message' fields
    - Event types: 'progress', 'enhancement', 'cache_hit', 'signals', 'context', 'token', 'complete', 'error'
    
    Usage:
    ```javascript
    const response = await fetch('/api/chat/pure-llm/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: "Hello" })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const event = JSON.parse(line.slice(6));
                
                if (event.type === 'token') {
                    // Append token to response
                    responseText += event.data;
                } else if (event.type === 'complete') {
                    // Final metadata
                    metadata = event.data.metadata;
                }
            }
        }
    }
    ```
    """
    
    # Generate or use provided session_id
    session_id = request.session_id or f"session_{hash(request.message)}"
    
    async def event_generator():
        """Generate SSE events from Pure LLM Core streaming."""
        try:
            # Get Pure LLM Core instance
            pure_llm_core = startup_manager.get_pure_llm_core()
            
            if not pure_llm_core:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Pure LLM Core not available'})}\n\n"
                return
            
            # Prepare user context
            user_context = {
                'preferences': request.preferences or {},
            }
            
            if request.user_location:
                user_context['gps'] = request.user_location
                user_context['location'] = request.user_location
            
            # Stream from Pure LLM Core
            async for event in pure_llm_core.process_query_stream(
                query=request.message,
                user_id=request.user_id or session_id,
                session_id=session_id,
                user_location=request.user_location,
                language="en",  # TODO: Get from request
                max_tokens=500,
                enable_conversation=True
            ):
                # Forward event to client in SSE format
                yield f"data: {json.dumps(event)}\n\n"
        
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    # Return StreamingResponse with SSE format
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
