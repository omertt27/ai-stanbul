"""
Chat Endpoints Module

All chat-related endpoints including ML chat, Pure LLM chat, and legacy chat
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import time
import logging
import json
import asyncio

from database import get_db
from core.startup_fixed import fast_startup_manager as startup_manager
from services.data_collection import log_chat_interaction
from utils.response_sanitizer import ResponseSanitizer
from utils.place_name_corrector import correct_place_names

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])

# ==========================================
# UnifiedLLMService Dependency Injection
# ==========================================
async def get_unified_llm(request: Request):
    """
    Dependency to inject UnifiedLLMService into endpoints.
    
    This provides access to the centralized LLM service with:
    - Automatic response caching (90% faster for cached queries)
    - Circuit breaker protection (automatic fallback to Groq)
    - Performance metrics tracking
    - Multi-backend support (vLLM + Groq)
    
    Returns:
        UnifiedLLMService instance
        
    Raises:
        HTTPException: If UnifiedLLMService not initialized
    """
    if not hasattr(request.app.state, 'unified_llm') or request.app.state.unified_llm is None:
        logger.warning("⚠️ UnifiedLLMService not available, endpoint will use legacy LLM")
        # Return None to allow graceful degradation
        return None
    return request.app.state.unified_llm


# ==========================================
# Response Sanitizer
# ==========================================
_response_sanitizer = ResponseSanitizer()

# ==========================================
# Translation Service Integration (Priority #3)
# ==========================================
_i18n_service = None

def get_i18n_service():
    """Get or create i18n service singleton"""
    global _i18n_service
    if _i18n_service is None:
        try:
            from i18n_service import i18n_service
            _i18n_service = i18n_service
            logger.info("✅ I18n Service initialized")
        except Exception as e:
            logger.warning(f"⚠️ I18n Service not available: {e}")
            _i18n_service = None
    return _i18n_service


def translate_if_needed(response_text: str, target_language: str) -> str:
    """
    Translate response to target language if needed.
    Priority #3 implementation - ensures all responses match user's language preference.
    
    Also applies place name spelling corrections automatically.
    """
    # STEP 1: Fix place name spellings FIRST (before translation)
    response_text = fix_place_name_spellings(response_text)
    
    # STEP 2: Translate if needed
    if not target_language or target_language == 'en':
        return response_text
    
    try:
        i18n = get_i18n_service()
        if i18n and target_language in i18n.supported_languages:
            # Translate from English to target language
            translated = i18n.translate_openai_response(response_text, target_language)
            logger.info(f"🌍 Translated response: en → {target_language}")
            return translated
    except Exception as e:
        logger.warning(f"⚠️ Translation failed ({target_language}): {e}")
    
    # Fallback: return original if translation fails
    return response_text


def fix_place_name_spellings(response_text: str) -> str:
    """
    Fix common place name misspellings in LLM responses.
    
    Ensures consistent, correct spelling of Istanbul landmarks and districts
    regardless of LLM output quality (e.g., "Galatport" → "Galataport").
    
    Args:
        response_text: Raw LLM response text
        
    Returns:
        Response text with corrected place names
    """
    try:
        return correct_place_names(response_text)
    except Exception as e:
        logger.warning(f"⚠️ Place name correction failed: {e}")
        return response_text  # Fallback: return original


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


# ===========================================
# Import Validated Schemas
# ===========================================
try:
    from schemas.chat import (
        PureLLMChatRequest,
        PureLLMChatResponse,
        MLChatRequest as MLChatRequestSchema,
        MLChatResponse as MLChatResponseSchema,
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    logger.warning("⚠️ Validated schemas not available, using legacy models")


# ===========================================
# Request/Response Models (Legacy + Enhanced)
# ===========================================
class ChatRequest(BaseModel):
    """Request model for chat endpoints (enhanced with validation)"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, max_length=128, description="Session identifier")
    user_location: Optional[Dict[str, float]] = Field(None, description="User GPS location")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    user_id: Optional[str] = Field(None, max_length=128, description="User ID for personalization")
    language: Optional[str] = Field("en", description="Response language (en/tr)")
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Response model for chat endpoints"""
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session identifier")
    llm_mode: str = Field(default="general", description="Mode: 'explain' | 'clarify' | 'general' | 'error'")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: Optional[float] = Field(None, description="Confidence score")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
    map_data: Optional[Dict[str, Any]] = Field(None, description="Map visualization data for routes")
    route_data: Optional[Dict[str, Any]] = Field(None, description="Transportation route data for visual display")
    navigation_active: Optional[bool] = Field(None, description="Whether GPS navigation is active")
    navigation_data: Optional[Dict[str, Any]] = Field(None, description="GPS navigation state and instructions")
    interaction_id: Optional[str] = Field(None, description="Interaction ID for feedback tracking")
    
    # Phase 5B: UnifiedLLMService Metadata (NEW!)
    cached: Optional[bool] = Field(None, description="Whether response was served from cache")
    backend_used: Optional[str] = Field(None, description="LLM backend: 'vllm', 'groq', or 'fallback'")
    latency_ms: Optional[int] = Field(None, description="Response latency in milliseconds")
    circuit_breaker_state: Optional[str] = Field(None, description="Circuit breaker state: 'closed', 'open', 'half_open'")
    tokens_used: Optional[int] = Field(None, description="Tokens used for generation")
    
    @field_validator('response')
    @classmethod
    def correct_place_names_in_response(cls, v):
        """
        Automatically correct common place name misspellings in responses.
        Ensures "Galatport" → "Galataport", "Karakoy" → "Karaköy", etc.
        """
        if v:
            return fix_place_name_spellings(v)
        return v
    
    @field_validator('suggestions')
    @classmethod
    def correct_place_names_in_suggestions(cls, v):
        """Correct place names in suggestion list"""
        if v:
            return [fix_place_name_spellings(suggestion) for suggestion in v]
        return v


class MLChatRequest(BaseModel):
    """Request model for ML-powered chat (enhanced with validation)"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    user_location: Optional[Dict[str, float]] = Field(None, description="User location {lat, lon}")
    use_llm: Optional[bool] = Field(None, description="Override: Use LLM (None=use default)")
    language: str = Field(default="en", description="Response language (en/tr)")
    user_id: Optional[str] = Field(None, max_length=128, description="User ID for personalization")
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


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
    db: Session = Depends(get_db),
    unified_llm = Depends(get_unified_llm)
):
    """
    Pure LLM chat endpoint - LLM-First Architecture with Full Pipeline
    
    Phase 5B Enhancement: Uses UnifiedLLMService for caching, circuit breaker, and metrics
    
    Flow:
    0. LLM Conversation Context (Phase 4.2!) - Resolve references & context
    1. LLM Intent Classification (Phase 1) - Understand user intent
    2. LLM Location Resolution (Phase 2) - Resolve locations
    3. LLM Route Preferences (Phase 4.1) - Detect preferences
    4. Specialized handlers (routes, gems, info)
    5. LLM Response Enhancement (Phase 3) - Enhance final response
    6. Extract UnifiedLLMService metadata for frontend display (Phase 5B!)
    """
    
    # Generate or use provided session_id
    session_id = request.session_id or f"session_{hash(request.message)}"
    
    # === STEP 0: FIX PLACE NAME MISSPELLINGS IN USER INPUT ===
    # Correct common misspellings before LLM processing
    # e.g., "galatport" → "Galataport", "karakoy" → "Karaköy"
    original_message = request.message
    request.message = fix_place_name_spellings(request.message)
    
    if original_message != request.message:
        logger.info(f"✏️ Corrected user input: '{original_message}' → '{request.message}'")
    
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
            
            # ENABLED: Fast rule-based context resolution (~10-50ms overhead)
            # Using rule-based only (no LLM) for speed
            ENABLE_CONTEXT_RESOLUTION = True  # ✅ Priority #4 implementation
            
            if ENABLE_CONTEXT_RESOLUTION and llm_client:
                context_manager = get_context_manager(
                    llm_client=llm_client,
                    config={
                        'enable_llm': False,  # Use rule-based only (fast!)
                        'fallback_to_rules': True,
                        'timeout_seconds': 0.5,  # Very fast timeout
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
                    logger.info(f"   ⚠️ Clarification needed: {resolved_context['clarification_question']}'")
                    
                    # 🌍 Translate clarification question
                    clarification_text = translate_if_needed(
                        resolved_context['clarification_question'], 
                        request.language or "en"
                    )
                    
                    return ChatResponse(
                        response=clarification_text,
                        session_id=session_id,
                        llm_mode="clarify",
                        intent="clarification",
                        confidence=resolved_context.get('confidence', 0.8),
                        suggestions=[]
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
    
    # === AUTO-DETECT LANGUAGE FROM QUERY (Using LLM's Detection) ===
    # Use the multilingual intent system's language detection
    # This is more accurate than character-based detection
    detected_language = request.language or "en"
    
    try:
        from services.multilingual_intent_keywords import detect_intent_multilingual
        
        # Detect intent and language together (fast, ~10ms)
        # Returns: (intent, confidence, matched_keywords, detected_language)
        intent_result, confidence, matched_keywords, lang_detected = detect_intent_multilingual(request.message)
        
        if lang_detected and lang_detected != "en":
            detected_language = lang_detected
            logger.info(f"🌍 LLM detected language: {detected_language} from query")
        
    except Exception as e:
        logger.warning(f"Language detection failed, using default: {e}")
    
    # Update request language for downstream use
    effective_language = detected_language
    user_context['language'] = effective_language
    
    # No LLM intent classification - Pure LLM will handle everything
    llm_intent = None
    location_resolution = None
    
    # === UNIFIED INTENT ROUTER (NEW - Multilingual, Fast) ===
    # This replaces scattered if/else blocks with a clean, centralized router
    # Handles all 9 features in 5 languages with ~10ms response time
    try:
        from services.unified_intent_router import get_intent_router
        
        intent_router = get_intent_router(db)
        router_result = await intent_router.route(
            query=request.message,
            user_location=request.user_location,
            session_id=session_id,
            user_context=user_context
        )
        
        if router_result and router_result.success:
            logger.info(f"✅ Unified Router handled: {router_result.intent} (fast path)")
            
            # Enhance response if needed
            enhanced_response = await enhance_chat_response(
                base_response=router_result.response,
                original_query=request.message,
                user_context=user_context,
                route_data=router_result.navigation_data,
                response_type=router_result.intent
            )
            
            # 🌍 Priority #3: Translate response to user's language
            final_response = translate_if_needed(enhanced_response, effective_language)
            
            # Return unified router result
            return ChatResponse(
                response=final_response,
                session_id=session_id,
                llm_mode="general",
                intent=router_result.intent,
                confidence=0.85,  # Default confidence for unified router results
                suggestions=router_result.suggestions or [],
                map_data=router_result.map_data,
                route_data=router_result.data,
                navigation_active=router_result.navigation_data is not None,
                navigation_data=router_result.navigation_data
            )
    except Exception as e:
        logger.warning(f"Unified router failed: {e}", exc_info=True)
        # Fall through to legacy handlers
    
    # === LEGACY HANDLERS (Fallback if unified router doesn't handle) ===
    try:
        # Hidden Gems GPS Navigation Check
        from services.hidden_gems_gps import hidden_gems_handler
        
        gems_result = hidden_gems_handler.handle_chat_message(
            message=request.message,
            user_location=request.user_location,
            session_id=request.session_id or 'new'
        )
        
        if gems_result and gems_result.get('type') != 'not_handled':
            # This was a hidden gems request
            if gems_result.get('type') == 'error':
                return ChatResponse(
                    response=gems_result.get('message', ''),
                    session_id=request.session_id or 'new',
                    llm_mode="general",
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
                    llm_mode="general",  # Default mode
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
                llm_mode="general",  # Default mode
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
            nav_result = await handler.handle_gps_navigation_command(
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
                    llm_mode="general",  # Default mode
                    intent='gps_navigation',
                    confidence=1.0,
                    suggestions=_get_navigation_suggestions(nav_result),
                    map_data=nav_result.get('navigation_data', {}).get('map_data'),
                    navigation_active=nav_result.get('navigation_active', False),
                    navigation_data=nav_result.get('navigation_data')
                )
            
            # REMOVED: Duplicate route handler call - unified router handles this now
            # The code below was causing double execution of route planning
            # If unified router fails, Pure LLM will handle the query
            logger.info(f"ℹ️ Skipping legacy route handler - unified router handles this")
                    
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
        
        # 🔥 FIX #1: GATE - Quick check if this is a transportation query
        # Do NOT skip Pure LLM (it generates the response), but flag for structured mode
        force_structured_mode = False
        force_intent = None
        force_confidence = None
        
        # Quick pattern check for transportation keywords
        query_lower = request.message.lower()
        transportation_keywords = ['how do i get', 'how can i get', 'how to get', 'route to', 'way to', 
                                   'from', 'directions to', 'navigate to', 'take me to', 'go to']
        if any(keyword in query_lower for keyword in transportation_keywords):
            logger.info("🚦 GATE: Transportation query detected - forcing structured mode")
            force_structured_mode = True
            force_intent = "transportation"
            force_confidence = 0.80  # High confidence from pattern match
        
        # Process query through Pure LLM
        # Use auto-detected language (effective_language) instead of request.language
        logger.info(f"🚀 Processing query through Pure LLM (language: {effective_language})" + (" [STRUCTURED MODE]" if force_structured_mode else ""))
        result = await pure_llm_core.process_query(
            query=request.message,
            user_location=request.user_location,
            session_id=request.session_id,
            language=effective_language  # Use auto-detected language!
        )
        
        # 🔥 CRITICAL FIX: Handle early return of map_data from routing visualization
        # The Pure LLM Core sometimes returns raw map_data instead of proper result dict
        if isinstance(result, dict) and 'type' in result and result.get('type') in ['route', 'marker'] and 'response' not in result:
            logger.warning(f"⚠️ EARLY ROUTING RETURN: Pure LLM returned map_data instead of result dict, wrapping it")
            # Wrap map_data in a proper result structure
            map_data_raw = result
            
            # Generate a simple explanation based on the route data
            route_info = map_data_raw.get('route_data', {})
            origin = map_data_raw.get('origin_name', 'your location')
            dest = map_data_raw.get('destination_name', 'your destination')
            duration = route_info.get('duration_min', 0)
            distance = route_info.get('distance_km', 0)
            transfers = route_info.get('transfers', 0)
            lines = route_info.get('lines', [])
            
            # Create a simple response
            simple_response = f"To get from {origin} to {dest}:\n\n"
            if lines:
                simple_response += f"Take the {', '.join(lines[:3])} line{'s' if len(lines) > 1 else ''}.\n"
            simple_response += f"Approximate travel time: {duration} minutes\n"
            simple_response += f"Distance: {distance:.1f} km\n"
            if transfers > 0:
                simple_response += f"{transfers} transfer{'s' if transfers > 1 else ''} required\n"
            
            result = {
                "status": "success",
                "response": simple_response,
                "map_data": map_data_raw,
                "signals": {"needs_transportation": True},
                "intent": "transportation",
                "confidence": 0.85,
                "metadata": {
                    "source": "routing_early_return",
                    "processing_time": 0
                }
            }
            # Force structured mode since this is clearly a transportation query
            force_structured_mode = True
            force_intent = "transportation"
            force_confidence = 0.85
        
        response_time = time.time() - start_time
        
        # === EXTRACT MAPDATA FROM TRANSPORTATION RAG ===
        # 🔥 CRITICAL FIX: Extract route AFTER LLM processing (when last_route is set)
        # The transportation RAG is called during pure_llm_core.process_query() above
        # So last_route is NOW available (previously it was null)
        map_data_from_transport = None
        route_data_from_transport = None
        try:
            from services.transportation_rag_system import get_transportation_rag
            transport_rag = get_transportation_rag()
            
            logger.info(f"🔍 Checking transport_rag.last_route: {transport_rag.last_route is not None if transport_rag else 'transport_rag is None'}")
            
            if transport_rag:
                # Get map data for visualization
                map_data_from_transport = transport_rag.get_map_data_for_last_route()
                
                # 🔥 DIRECT FIX: Enrich route_data directly from last_route
                if transport_rag.last_route:
                    logger.info(f"✅ Found last_route: {transport_rag.last_route.origin} → {transport_rag.last_route.destination}")
                    
                    # Build basic route_data from last_route
                    basic_route_data = {
                        'origin': transport_rag.last_route.origin,
                        'destination': transport_rag.last_route.destination,
                        'steps': transport_rag.last_route.steps,
                        'total_time': transport_rag.last_route.total_time,
                        'total_distance': transport_rag.last_route.total_distance,
                        'transfers': transport_rag.last_route.transfers,
                        'lines_used': transport_rag.last_route.lines_used
                    }
                    
                    # Enrich it directly with canonical IDs
                    route_data_from_transport = transport_rag.station_normalizer.enrich_route_data(basic_route_data)
                    
                    logger.info(f"✅ Directly enriched route_data:")
                    logger.info(f"   Origin: {route_data_from_transport.get('origin')} (ID: {route_data_from_transport.get('origin_station_id')})")
                    logger.info(f"   Dest: {route_data_from_transport.get('destination')} (ID: {route_data_from_transport.get('destination_station_id')})")
                    logger.info(f"   Steps: {len(route_data_from_transport.get('steps', []))}")
                    
                    # Also update map_data metadata with enriched route_data
                    if map_data_from_transport and 'metadata' in map_data_from_transport:
                        map_data_from_transport['metadata']['route_data'] = route_data_from_transport
                        logger.info(f"   ✅ Updated map_data.metadata with enriched route_data")
                else:
                    logger.warning("⚠️  No last_route available for enrichment")
                    # Fallback to basic route_data if no last_route
                    if map_data_from_transport and 'metadata' in map_data_from_transport:
                        route_data_from_transport = {
                            'origin': 'Unknown',
                            'destination': 'Unknown',
                            'steps': [],
                            'total_time': map_data_from_transport.get('metadata', {}).get('total_time', 0),
                            'total_distance': map_data_from_transport.get('metadata', {}).get('total_distance', 0),
                            'transfers': map_data_from_transport.get('metadata', {}).get('transfers', 0),
                            'lines_used': map_data_from_transport.get('metadata', {}).get('lines_used', [])
                        }
        except Exception as e:
            logger.warning(f"Failed to extract mapData from transportation RAG: {e}")
        
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
        
        # 🔥 CRITICAL FIX: Re-extract route_data AFTER Pure LLM processing
        # The Pure LLM call triggers transportation RAG which sets last_route
        # We need to extract it here, AFTER the LLM has processed the query
        logger.info(f"🔍 POST-LLM Check: route_data_from_transport is {'None' if not route_data_from_transport else 'SET'}")
        
        if not route_data_from_transport:
            logger.info("🔥 POST-LLM: Attempting to extract route data...")
            try:
                from services.transportation_rag_system import get_transportation_rag
                transport_rag_post = get_transportation_rag()
                
                logger.info(f"🔍 POST-LLM: transport_rag exists: {transport_rag_post is not None}")
                if transport_rag_post:
                    logger.info(f"🔍 POST-LLM: last_route exists: {transport_rag_post.last_route is not None}")
                
                if transport_rag_post and transport_rag_post.last_route:
                    logger.info(f"🔥 POST-LLM: Extracting route from last_route: {transport_rag_post.last_route.origin} → {transport_rag_post.last_route.destination}")
                    
                    # Build route_data from last_route
                    basic_route_data = {
                        'origin': transport_rag_post.last_route.origin,
                        'destination': transport_rag_post.last_route.destination,
                        'steps': transport_rag_post.last_route.steps,
                        'total_time': transport_rag_post.last_route.total_time,
                        'total_distance': transport_rag_post.last_route.total_distance,
                        'transfers': transport_rag_post.last_route.transfers,
                        'lines_used': transport_rag_post.last_route.lines_used
                    }
                    
                    # Enrich with canonical IDs
                    route_data_from_transport = transport_rag_post.station_normalizer.enrich_route_data(basic_route_data)
                    logger.info(f"✅ POST-LLM: Route data extracted successfully")
                    logger.info(f"✅ POST-LLM: Route: {route_data_from_transport.get('origin')} → {route_data_from_transport.get('destination')}")
                    
                    # Also get map data
                    if not map_data_from_transport:
                        map_data_from_transport = transport_rag_post.get_map_data_for_last_route()
                        logger.info(f"✅ POST-LLM: Map data extracted: {map_data_from_transport is not None}")
                else:
                    logger.warning("⚠️ POST-LLM: last_route is None - transportation RAG may not have been called")
            except Exception as e:
                logger.error(f"POST-LLM route extraction failed: {e}", exc_info=True)
        
        # Extract route_data from map_data if present
        route_data = None
        map_data = result.get('map_data')
        if map_data and isinstance(map_data, dict):
            # If map_data contains route_data, extract it
            if 'route_data' in map_data:
                route_data = map_data['route_data']
            # If map_data itself looks like route data (has steps, origin, destination)
            elif 'steps' in map_data and 'origin' in map_data and 'destination' in map_data:
                route_data = map_data
        
        # === LLM ROUTE EXTRACTION: For landmarks not in database ===
        # Try to extract route data from LLM response if we don't have complete data
        # This handles landmarks (Galataport, Grand Bazaar, etc.) that aren't transit stations
        if not map_data or not route_data:
            try:
                from services.llm_route_extractor import get_llm_route_extractor
                
                extractor = get_llm_route_extractor()
                extraction_result = extractor.extract_route_from_llm_response(
                    llm_response=result.get('response', ''),
                    user_query=request.message,
                    user_location=request.user_location
                )
                
                if extraction_result:
                    # Use extracted data if we don't already have it
                    if not route_data:
                        route_data = extraction_result.get('route_data')
                    if not map_data:
                        map_data = extraction_result.get('map_data')
                    
                    logger.info(f"✅ LLM Route Extractor: Generated route card for landmark")
                    if route_data:
                        logger.info(f"   Origin: {route_data.get('origin')}")
                        logger.info(f"   Destination: {route_data.get('final_destination')}")
                        logger.info(f"   Via: {route_data.get('destination')}")
                else:
                    logger.info(f"   LLM Route Extractor: No route data could be extracted")
            
            except Exception as e:
                logger.warning(f"LLM route extraction failed: {e}")
                # Continue without extracted route data - non-blocking
        
        # Phase 3: Enhance Pure LLM response with contextual intelligence
        enhanced_response = await enhance_chat_response(
            base_response=result.get('response', ''),
            original_query=request.message,
            user_context=user_context,
            route_data=route_data,  # Pass extracted route data
            response_type=result.get('intent', 'general')
        )
        
        # === RESPONSE SANITIZATION: Clean up LLM artifacts ===
        # Remove system prompt leakage, ensure language consistency
        sanitized_response = _response_sanitizer.sanitize(
            response=enhanced_response,
            expected_language=effective_language,  # Use auto-detected language
            strict_language_check=True
        )
        
        if sanitized_response != enhanced_response:
            logger.info(f"🧹 Response sanitized - removed {len(enhanced_response) - len(sanitized_response)} characters of artifacts")
        
        enhanced_response = sanitized_response
        
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
                language=effective_language,  # Use auto-detected language
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
        
        # Determine LLM mode (Week 1 Improvement #2)
        # 🔥 FIX #2: Force llm_mode explicitly for routes (BEFORE enhancers)
        llm_mode = "general"
        final_intent = result.get('intent')
        final_confidence = result.get('confidence')
        
        # Override with forced values if this is a transportation query
        if force_structured_mode:
            llm_mode = "explain"  # Explaining a verified route
            final_intent = "transportation"
            final_confidence = 0.80  # 0.80
            logger.info(f"🔒 FORCED: llm_mode={llm_mode}, intent={final_intent}, confidence={final_confidence}")
        elif route_data_from_transport or route_data:
            llm_mode = "explain"  # Explaining a verified route
            final_intent = "transportation"
            final_confidence = 0.85
            logger.info(f"🔒 ROUTE DETECTED: llm_mode={llm_mode}, intent={final_intent}")
        elif result.get('signals', {}).get('needs_transportation'):
            llm_mode = "explain"  # Explaining a verified route
            final_intent = "transportation"
            final_confidence = 0.70
        elif result.get('requires_clarification'):
            llm_mode = "clarify"  # Asking for more info
        elif result.get('metadata', {}).get('error') or result.get('metadata', {}).get('degraded_mode'):
            llm_mode = "error"  # Error/fallback response
        # else: stays "general"
        
        # 🔥 FIX #4: Safety net - protect against empty response with silent retry
        final_response = enhanced_response
        
        # Check if result is actually just map_data (routing visualization returned early)
        if isinstance(result, dict) and 'type' in result and result.get('type') in ['route', 'marker']:
            logger.warning(f"⚠️ ROUTING EARLY RETURN: Result is map_data, not full response object")
            # Result IS the map_data, we need to construct a response
            llm_raw_response = result.get('description', '') or "Here's the route for you."
            final_response = llm_raw_response
            # map_data_from_transport should already be set
        elif not final_response or final_response.strip() == "":
            if route_data_from_transport or route_data:
                # If we have route data but empty response, try LLM's raw response first
                llm_raw_response = result.get('response', '')
                if llm_raw_response and llm_raw_response.strip():
                    logger.warning(f"⚠️ SAFETY NET: Empty response detected, using LLM raw response ({len(llm_raw_response)} chars)")
                    final_response = llm_raw_response
                else:
                    # 🔥 WEEK 2: Silent retry with strict template
                    logger.warning(f"⚠️ SAFETY NET: Empty LLM response, attempting silent retry with strict template")
                    retry_response = await _retry_with_strict_template(
                        route_data=route_data_from_transport or route_data,
                        origin=request.message,  # Approximate origin from query
                        destination=request.message,  # Approximate destination from query
                        pure_llm_core=pure_llm_core
                    )
                    if retry_response:
                        logger.info(f"✅ RETRY SUCCESS: Got {len(retry_response)} char response")
                        final_response = retry_response
                    else:
                        logger.error(f"❌ CRITICAL: Empty response AND retry failed! Result keys: {result.keys()}")
                        final_response = "I found a route for you. Please check the map for details."
        
        # Use route_data_from_transport if available, otherwise fall back to extracted route_data
        final_route_data = route_data_from_transport or route_data
        
        # 🌍 Priority #3: Translate final response to user's language
        translated_response = translate_if_needed(final_response, effective_language)
        
        # CRITICAL: Merge multi-route data into map_data for frontend
        # Priority: transport RAG > result > LLM extractor
        final_map_data = map_data_from_transport or result.get('map_data') or map_data
        
        # Try to get transport alternatives from context builder
        # The context builder stores it but we need to access it from the result
        try:
            # Check if context was built and has transport alternatives
            if pure_llm_core and hasattr(pure_llm_core, 'context_builder'):
                ctx_builder = pure_llm_core.context_builder
                if hasattr(ctx_builder, '_transport_alternatives') and ctx_builder._transport_alternatives:
                    transport_alts = ctx_builder._transport_alternatives
                    
                    if transport_alts and (transport_alts.get('alternatives') or transport_alts.get('primary_route')):
                        logger.info(f"🗺️ Merging multi-route data into map_data for frontend")
                        
                        if not final_map_data:
                            final_map_data = {}
                        
                        # Add multi-route data
                        final_map_data.update({
                            'type': 'multi_route',
                            'multi_routes': transport_alts.get('alternatives', []),
                            'primary_route': transport_alts.get('primary_route'),
                            'route_comparison': transport_alts.get('route_comparison', {})
                        })
                        
                        # Preserve existing map_data if present
                        if transport_alts.get('map_data'):
                            final_map_data.update(transport_alts['map_data'])
                        
                        logger.info(f"✅ Added {len(transport_alts.get('alternatives', []))} route alternatives to map_data")
        except Exception as e:
            logger.warning(f"Failed to merge multi-route data: {e}")
        
        # === PHASE 5B: EXTRACT UNIFIEDLLMSERVICE METADATA ===
        # Extract caching, backend, and performance metrics from UnifiedLLMService
        cached = False
        backend_used = None
        latency_ms = None
        circuit_breaker_state = None
        tokens_used = None
        
        try:
            # Get metadata from UnifiedLLMService via PureLLMCore
            if unified_llm:
                # Get current circuit breaker state
                circuit_breaker_state = "open" if unified_llm.circuit_breaker_open else "closed"
                backend_used = "groq" if unified_llm.circuit_breaker_open else "vllm"
                
                # Get metrics
                metrics = unified_llm.get_metrics()
                
                # Check if this specific request was cached
                # The result dict may contain 'cached' flag from PureLLMCore
                cached = result.get('cached', False) or result.get('metadata', {}).get('cached', False)
                
                # Get latency (convert to ms)
                latency_ms = int(response_time * 1000) if response_time else None
                
                # Get tokens (if available in metadata)
                tokens_used = result.get('metadata', {}).get('tokens_used')
                
                logger.info(f"📊 UnifiedLLMService Metadata: cached={cached}, backend={backend_used}, latency={latency_ms}ms, cb_state={circuit_breaker_state}")
            
            elif pure_llm_core and hasattr(pure_llm_core, 'llm_service'):
                # Fallback: Extract from PureLLMCore's llm_service
                llm_service = pure_llm_core.llm_service
                circuit_breaker_state = "open" if llm_service.circuit_breaker_open else "closed"
                backend_used = "groq" if llm_service.circuit_breaker_open else "vllm"
                cached = result.get('cached', False)
                latency_ms = int(response_time * 1000) if response_time else None
                logger.info(f"📊 Metadata from PureLLMCore: cached={cached}, backend={backend_used}")
        
        except Exception as e:
            logger.warning(f"Failed to extract UnifiedLLMService metadata: {e}")
            # Continue without metadata - non-blocking
        
        return ChatResponse(
            response=translated_response,  # 🔥 Translated + safety-netted response
            session_id=result.get('session_id', request.session_id or 'new'),
            llm_mode=llm_mode,  # 🔥 FIX #2: Forced mode
            intent=final_intent,  # 🔥 FIX #2: Forced intent
            confidence=final_confidence,  # 🔥 FIX #2: Forced confidence
            suggestions=final_suggestions,
            map_data=final_map_data,  # 🗺️ Include multi-route data
            route_data=final_route_data,  # 🔥 FIX #3: Early-extracted route data
            navigation_active=result.get('navigation_active', False),
            navigation_data=result.get('navigation_data'),
            interaction_id=interaction_id,  # Include for frontend feedback tracking
            # Phase 5B: UnifiedLLMService metadata (NEW!)
            cached=cached,
            backend_used=backend_used,
            latency_ms=latency_ms,
            circuit_breaker_state=circuit_breaker_state,
            tokens_used=tokens_used
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
        
        # === FIX PLACE NAME MISSPELLINGS IN USER INPUT ===
        original_message = request.message
        request.message = fix_place_name_spellings(request.message)
        
        if original_message != request.message:
            logger.info(f"✏️ Corrected ML chat input: '{original_message}' → '{request.message}'")
        
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
        nav_result = await handler.handle_gps_navigation_command(
            message=request.message,
            session_id=request.session_id or 'new',
            user_location=request.user_location
        )
        
        if nav_result:
            # This was a navigation command
            return ChatResponse(
                response=nav_result.get('message', ''),
                session_id=request.session_id or 'new',
                llm_mode="general",  # Default mode
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
        
        # 🌍 Translate emergency fallback (use request.language since we don't have effective_language here)
        translated_fallback = translate_if_needed(response_text, request.language or "en")
        
        return ChatResponse(
            response=translated_fallback,
            session_id=request.session_id or "new",
            llm_mode="general",  # Default mode
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
        
        response += f"{i}. {name}"
        
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


# ==========================================
# 🔥 WEEK 2 ENHANCEMENT: Silent Retry Safety Net
# ==========================================

async def _retry_with_strict_template(
    route_data: Dict[str, Any],
    origin: str,
    destination: str,
    pure_llm_core
) -> str:
    """
    Silently retry LLM with strict template when initial response is empty/invalid.
    This prevents users from seeing generic fallback messages.
    
    Args:
        route_data: Verified route data from Transportation RAG
        origin: Origin location name
        destination: Destination location name
        pure_llm_core: Pure LLM core instance
        
    Returns:
        Strict templated response explaining the route
    """
    try:
        # Build strict template from route_data
        steps_text = []
        for i, step in enumerate(route_data.get('steps', []), 1):
            steps_text.append(
                f"{i}. Take {step.get('line', 'transit')} from {step.get('from_station', {}).get('name', 'station')} "
                f"to {step.get('to_station', {}).get('name', 'station')} ({step.get('duration', 0):.1f} min)"
            )
        
        strict_prompt = f"""
You are explaining a verified route from {origin} to {destination}.

Here are the exact steps:
{chr(10).join(steps_text)}

Total time: {route_data.get('total_time', 0)} minutes
Total distance: {route_data.get('total_distance', 0):.1f} km
Transfers: {route_data.get('transfers', 0)}

Please provide a clear, natural explanation of this route in 2-3 sentences.
Do NOT modify any station names, lines, or times.
"""
        
        logger.info(f"🔄 RETRY: Calling LLM with strict template ({len(strict_prompt)} chars)")
        
        # Call LLM with strict template
        retry_result = await pure_llm_core.process_query(
            query=strict_prompt,
            user_location=None,
            session_id=None,
            language="en"
        )
        
        retry_response = retry_result.get('response', '')
        if retry_response and len(retry_response) > 50:
            logger.info(f"✅ RETRY SUCCESS: Generated {len(retry_response)} chars")
            return retry_response
        else:
            logger.warning(f"⚠️ RETRY FAILED: Response too short ({len(retry_response)} chars)")
            return None
            
    except Exception as e:
        logger.error(f"❌ RETRY ERROR: {e}")
        return None
