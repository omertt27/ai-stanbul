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
    Pure LLM chat endpoint - LLM-First Architecture with Intent Classification
    
    Phase 1 Enhancement: LLM Intent Classifier runs FIRST to understand user intent,
    extract locations, detect preferences, and provide routing recommendations.
    
    Flow:
    1. LLM Intent Classification (NEW!) - Always first
    2. Smart routing based on LLM intent
    3. Specialized handlers (routes, gems, info)
    4. Pure LLM fallback
    """
    
    # === PHASE 1: LLM INTENT CLASSIFICATION (NEW!) ===
    # This gives the LLM the primary role in understanding user intent
    logger.info(f"🤖 Phase 1: LLM Intent Classification for query: '{request.message[:60]}...'")
    
    try:
        from services.llm import get_intent_classifier
        
        # Prepare context for intent classification
        user_context = {
            'preferences': request.preferences or {},
        }
        
        if request.user_location:
            user_context['gps'] = request.user_location
            user_context['location'] = request.user_location
            logger.info(f"📍 User GPS available: lat={request.user_location.get('lat')}, lon={request.user_location.get('lon')}")
        
        # Get or create intent classifier (uses singleton pattern)
        pure_llm_core = startup_manager.get_pure_llm_core()
        if not pure_llm_core:
            logger.warning("⚠️ Pure LLM Core not available, skipping LLM intent classification")
            llm_intent = None
        else:
            # Get the LLM client from pure_llm_core
            llm_client = pure_llm_core.llm_client if hasattr(pure_llm_core, 'llm_client') else None
            
            if llm_client:
                intent_classifier = get_intent_classifier(
                    llm_client=llm_client,
                    db_connection=db,
                    cache_manager=None,  # TODO: Add cache manager when available
                    config={'enable_caching': False}  # Disable for initial testing
                )
                
                # Classify intent using LLM
                llm_intent = await intent_classifier.classify_intent(
                    query=request.message,
                    user_context=user_context,
                    use_cache=True
                )
                
                logger.info(
                    f"✅ LLM Intent Classification complete:\n"
                    f"   - Primary Intent: {llm_intent.primary_intent}\n"
                    f"   - Confidence: {llm_intent.confidence:.2f}\n"
                    f"   - Origin: {llm_intent.origin}\n"
                    f"   - Destination: {llm_intent.destination}\n"
                    f"   - Method: {llm_intent.classification_method}\n"
                    f"   - Time: {llm_intent.processing_time_ms:.0f}ms"
                )
                
                # Log preferences if detected
                if llm_intent.user_preferences:
                    logger.info(f"   - Detected Preferences: {llm_intent.user_preferences}")
                
                # Log ambiguities if any
                if llm_intent.ambiguities:
                    logger.warning(f"   - Ambiguities: {llm_intent.ambiguities}")
        
                # === PHASE 2: LLM LOCATION RESOLUTION ===
                # If locations detected or route intent, use LLM location resolver
                location_resolution = None
                if llm_intent.primary_intent in ["route", "hidden_gems", "information", "transport"]:
                    try:
                        from backend.services.llm import get_location_resolver
                        
                        location_resolver = get_location_resolver()
                        
                        # Prepare user context for location resolver
                        loc_context = {
                            'gps': user_context.get('gps'),
                            'previous_locations': []  # TODO: Track in conversation history
                        }
                        
                        # Resolve locations using LLM
                        location_resolution = location_resolver.resolve_locations(
                            query=request.message,
                            user_context=loc_context
                        )
                        
                        logger.info(
                            f"✅ LLM Location Resolution complete:\n"
                            f"   - Pattern: {location_resolution.pattern}\n"
                            f"   - Locations Found: {len(location_resolution.locations)}\n"
                            f"   - Confidence: {location_resolution.confidence:.2f}\n"
                            f"   - Used LLM: {location_resolution.used_llm}\n"
                            f"   - Fallback: {location_resolution.fallback_used}"
                        )
                        
                        # Log each location
                        for i, loc in enumerate(location_resolution.locations, 1):
                            coords_str = f"{loc.coordinates[0]:.4f}, {loc.coordinates[1]:.4f}" if loc.coordinates else "None"
                            logger.info(
                                f"   Location {i}: {loc.name} -> {loc.matched_name} "
                                f"({coords_str}, confidence={loc.confidence:.2f})"
                            )
                        
                        # Log ambiguities
                        if location_resolution.ambiguities:
                            logger.warning(f"   - Location Ambiguities: {location_resolution.ambiguities}")
                        
                        # Update llm_intent with resolved locations for downstream handlers
                        if location_resolution.locations:
                            if len(location_resolution.locations) >= 2 and location_resolution.pattern == "from_to":
                                # Two locations: origin and destination
                                llm_intent.origin = location_resolution.locations[0].matched_name
                                llm_intent.destination = location_resolution.locations[1].matched_name
                                logger.info(f"   ✓ Updated intent: origin={llm_intent.origin}, dest={llm_intent.destination}")
                            elif len(location_resolution.locations) == 1:
                                # Single location: destination only
                                llm_intent.destination = location_resolution.locations[0].matched_name
                                logger.info(f"   ✓ Updated intent: dest={llm_intent.destination}")
                            elif len(location_resolution.locations) > 2:
                                # Multi-stop journey
                                llm_intent.destination = location_resolution.locations[-1].matched_name
                                logger.info(f"   ✓ Multi-stop journey detected, final dest={llm_intent.destination}")
                    
                    except Exception as e:
                        logger.error(f"❌ LLM Location Resolution failed: {e}", exc_info=True)
                        location_resolution = None
            else:
                logger.warning("⚠️ LLM client not available, skipping intent classification")
                llm_intent = None
                
    except Exception as e:
        logger.error(f"❌ LLM Intent Classification failed: {e}", exc_info=True)
        llm_intent = None
    
    # === PHASE 2: SMART ROUTING BASED ON LLM INTENT ===
    # Use LLM intent to make smart routing decisions
    if llm_intent and llm_intent.is_high_confidence(threshold=0.7):
        logger.info(f"🎯 High confidence LLM intent: {llm_intent.primary_intent}")
        
        # Route based on LLM intent classification
        if llm_intent.primary_intent == "information":
            logger.info("ℹ️ LLM detected information request, skipping GPS/route handlers")
            skip_routing = True
        elif llm_intent.primary_intent in ["route", "hidden_gems", "transport"]:
            logger.info(f"🗺️ LLM detected {llm_intent.primary_intent} intent, will check specialized handlers")
            skip_routing = False
        else:
            logger.info(f"💬 LLM detected {llm_intent.primary_intent} intent, will use general flow")
            skip_routing = False
    else:
        # Low confidence or no LLM intent, fallback to regex-based detection
        if llm_intent:
            logger.warning(f"⚠️ Low confidence LLM intent ({llm_intent.confidence:.2f}), falling back to regex")
        else:
            logger.warning(f"⚠️ No LLM intent available, falling back to regex")
        skip_routing = False
        
        # Fallback: Old regex-based information request detection
        def is_information_request(message: str) -> bool:
        """Check if asking for information about POIs, not directions to them"""
        msg_lower = message.lower()
        
        # Information request keywords
        info_keywords = ['what are', 'show me the', 'tell me about', 'recommend', 
                        'best', 'top', 'list', 'which', 'where can i find',
                        'what is', 'what\'s', 'tell me', 'can you recommend']
        has_info_keywords = any(kw in msg_lower for kw in info_keywords)
        
        # POI/Attraction keywords (information target)
        poi_keywords = ['attractions', 'landmarks', 'museums', 'places to visit', 
                       'sights', 'historical', 'monuments', 'palaces', 'mosques',
                       'restaurants', 'cafes', 'things to do', 'things to see',
                       'places to see', 'must see', 'must-see', 'worth visiting']
        about_pois = any(kw in msg_lower for kw in poi_keywords)
        
        # Direction keywords (should NOT be present for info requests)
        direction_keywords = ['from', ' to ', 'route', 'directions', 'how to get', 
                            'how do i get', 'how can i get', 'take me', 'navigate',
                            'navigation', 'way to', 'go to']
        asking_directions = any(kw in msg_lower for kw in direction_keywords)
        
        # Special case: "show me" + location name without info keywords = might be directions
        # But "show me the best/top" = information request
        if 'show me' in msg_lower and not any(word in msg_lower for word in ['best', 'top', 'all', 'some', 'good']):
        # Fallback: Old regex-based information request detection
        def is_information_request(message: str) -> bool:
            """Check if asking for information about POIs, not directions to them"""
            msg_lower = message.lower()
            
            # Information request keywords
            info_keywords = ['what are', 'show me the', 'tell me about', 'recommend', 
                            'best', 'top', 'list', 'which', 'where can i find',
                            'what is', 'what\'s', 'tell me', 'can you recommend']
            has_info_keywords = any(kw in msg_lower for kw in info_keywords)
            
            # POI/Attraction keywords (information target)
            poi_keywords = ['attractions', 'landmarks', 'museums', 'places to visit', 
                           'sights', 'historical', 'monuments', 'palaces', 'mosques',
                           'restaurants', 'cafes', 'things to do', 'things to see',
                           'places to see', 'must see', 'must-see', 'worth visiting']
            about_pois = any(kw in msg_lower for kw in poi_keywords)
            
            # Direction keywords (should NOT be present for info requests)
            direction_keywords = ['from', ' to ', 'route', 'directions', 'how to get', 
                                'how do i get', 'how can i get', 'take me', 'navigate',
                                'navigation', 'way to', 'go to']
            asking_directions = any(kw in msg_lower for kw in direction_keywords)
            
            # Special case: "show me" + location name without info keywords = might be directions
            # But "show me the best/top" = information request
            if 'show me' in msg_lower and not any(word in msg_lower for word in ['best', 'top', 'all', 'some', 'good']):
                return False
            
            # It's an info request if: has info keywords + about POIs + NOT asking directions
            return has_info_keywords and about_pois and not asking_directions
        
        skip_routing = is_information_request(request.message)
        if skip_routing:
            logger.info(f"ℹ️ Fallback regex detected information request: '{request.message[:50]}...'")
    
    # === PHASE 3: SPECIALIZED HANDLERS (if not skipping) ===
    if not skip_routing:
        # First check if this is a hidden gems GPS request
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
        
        # Process query through Pure LLM
        result = await pure_llm_core.process_query(
            query=request.message,
            user_location=request.user_location,
            session_id=request.session_id,
            language="en"
        )
        
        response_time = time.time() - start_time
        
        logger.info(f"Pure LLM response generated in {response_time:.2f}s")
        
        # Phase 3: Enhance Pure LLM response with contextual intelligence
        enhanced_response = await enhance_chat_response(
            base_response=result.get('response', ''),
            original_query=request.message,
            user_context=user_context,
            route_data=result.get('map_data'),  # May contain route info
            response_type=result.get('intent', 'general')
        )
        
        return ChatResponse(
            response=enhanced_response,
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
