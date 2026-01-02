"""
Unified Chat Endpoint - ML-Powered AI Chat System
==================================================

This endpoint connects all ML systems with IstanbulDailyTalkAI for intelligent
response generation. It replaces the simple OpenAI-only endpoint with a full
ML pipeline that includes:

- Neural Intent Classification
- Query Preprocessing (typo correction, entity extraction)
- Context-Aware Classification
- Response Caching
- Rate Limiting
- System Monitoring

Flow:
1. Rate limit check
2. Cache lookup
3. Query preprocessing
4. Intent classification (neural + context-aware)
5. Route to IstanbulDailyTalkAI with detected intent
6. Cache response
7. Return result with metadata

Created: October 30, 2025
Author: ML Systems Integration Team
"""

import sys
import os
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import time

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import response sanitizer to prevent data leakage
from backend.services.response_sanitizer import sanitize_response

# Import ML systems from backend.main
try:
    from backend.main import (
        process_enhanced_query,
        INTENT_CLASSIFIER_AVAILABLE,
        ENHANCED_QUERY_UNDERSTANDING_ENABLED,
        # Infrastructure
        system_monitor,
        response_cache,
        rate_limiter,
        INFRASTRUCTURE_AVAILABLE
    )
    ML_SYSTEMS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ ML systems not available: {e}")
    ML_SYSTEMS_AVAILABLE = False
    INTENT_CLASSIFIER_AVAILABLE = False
    ENHANCED_QUERY_UNDERSTANDING_ENABLED = False
    INFRASTRUCTURE_AVAILABLE = False
    system_monitor = None
    response_cache = None
    rate_limiter = None

# Import LLM Core from startup manager (already initialized with all dependencies)
try:
    from core.startup_fixed import fast_startup_manager
    LLM_CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Startup manager not available: {e}")
    LLM_CORE_AVAILABLE = False

# Import AI system (legacy - only used if LLM Core not ready)
try:
    from istanbul_ai.main_system import IstanbulDailyTalkAI
    AI_SYSTEM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ IstanbulDailyTalkAI not available: {e}")
    AI_SYSTEM_AVAILABLE = False

# Initialize router
router = APIRouter(prefix="/api", tags=["Chat"])
logger = logging.getLogger(__name__)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    gps_location: Optional[dict] = None

class MapVisualization(BaseModel):
    """Map visualization data for transportation and route planning"""
    type: Optional[str] = None  # 'route', 'locations', 'area'
    coordinates: Optional[list] = None  # [[lat, lon], ...] for route/polyline
    markers: Optional[list] = None  # [{"lat": x, "lon": y, "label": "..."}]
    center: Optional[dict] = None  # {"lat": x, "lon": y}
    zoom: Optional[int] = None  # Suggested zoom level
    route_data: Optional[dict] = None  # Detailed route information
    transport_lines: Optional[list] = None  # Metro/tram/bus lines

class ChatResponse(BaseModel):
    response: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    session_id: str
    timestamp: str
    cache_hit: bool = False
    processing_time_ms: float = 0
    ml_enabled: bool = False
    method: Optional[str] = None  # 'neural', 'fallback', or 'openai'
    map_data: Optional[MapVisualization] = None  # Map visualization data

@router.post("/chat", response_model=ChatResponse)
async def unified_chat(request: ChatRequest):
    """
    Unified chat endpoint with full ML pipeline
    
    This endpoint integrates all ML systems for intelligent response generation:
    - Query Preprocessing: Typo correction, entity extraction
    - Intent Classification: Neural classifier with context awareness
    - Smart Caching: Cache frequently asked questions
    - Rate Limiting: Protect API from abuse
    - System Monitoring: Track performance and errors
    
    Args:
        request (ChatRequest): User message with optional metadata
        
    Returns:
        ChatResponse: AI-generated response with metadata
        
    Raises:
        HTTPException: Rate limit exceeded, system unavailable, or internal error
    """
    start_time = time.time()
    
    # Validate input
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Generate session ID if not provided
    session_id = request.session_id or f"session_{int(time.time())}"
    user_id = request.user_id or "anonymous"
    
    # Start monitoring
    request_id = f"chat_{session_id}_{int(time.time())}"
    metric = None
    if INFRASTRUCTURE_AVAILABLE and system_monitor:
        try:
            metric = system_monitor.start_request(
                request_id=request_id,
                query=request.message,
                user_id=user_id
            )
        except Exception as e:
            logger.warning(f"Failed to start monitoring: {e}")
    
    try:
        # 1. CHECK RATE LIMIT
        if INFRASTRUCTURE_AVAILABLE and rate_limiter:
            try:
                if not rate_limiter.is_allowed("user", user_id):
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded. Please wait before sending more requests."
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Rate limiter error: {e}, allowing request")
        
        # 2. CHECK CACHE
        cached_response = None
        if INFRASTRUCTURE_AVAILABLE and response_cache:
            try:
                cached_response = response_cache.get(
                    query=request.message,
                    intent="unknown",  # We don't know intent yet
                    location=None
                )
                
                if cached_response:
                    logger.info(f"✅ Cache hit for query: {request.message[:50]}...")
                    processing_time = (time.time() - start_time) * 1000
                    
                    # 🔒 SANITIZE cached response too
                    cached_text = sanitize_response(cached_response['response'])
                    
                    if metric:
                        metric.cache_hit = True
                        metric.processing_time_ms = processing_time
                        system_monitor.end_request(metric, success=True)
                    
                    return ChatResponse(
                        response=cached_text,
                        intent=cached_response.get('intent'),
                        confidence=cached_response.get('confidence'),
                        session_id=session_id,
                        timestamp=datetime.now().isoformat(),
                        cache_hit=True,
                        processing_time_ms=processing_time,
                        ml_enabled=True,
                        method=cached_response.get('method', 'cached')
                    )
            except Exception as e:
                logger.warning(f"Cache lookup error: {e}, proceeding without cache")
        
        # 3. PROCESS WITH ML INTENT CLASSIFICATION
        intent_result = None
        method = 'unknown'
        
        if ML_SYSTEMS_AVAILABLE and (ENHANCED_QUERY_UNDERSTANDING_ENABLED or INTENT_CLASSIFIER_AVAILABLE):
            try:
                intent_result = process_enhanced_query(request.message, session_id)
                logger.info(f"🎯 Intent detected: {intent_result.get('intent')} "
                          f"(confidence: {intent_result.get('confidence', 0):.2f}, "
                          f"method: {intent_result.get('method', 'unknown')})")
                method = intent_result.get('method', 'ml')
            except Exception as e:
                logger.warning(f"⚠️ ML intent classification failed: {e}, falling back to AI system")
                intent_result = None
                method = 'fallback'
        else:
            logger.info("⚠️ ML systems not available, using AI system directly")
            method = 'ai_direct'
        
        # 4. ROUTE TO LLM CORE (preferred) or ISTANBULDAILYTALK AI (fallback)
        llm_core = None
        if LLM_CORE_AVAILABLE:
            # Get LLM Core from startup manager (already initialized with all dependencies)
            llm_core = fast_startup_manager.get_pure_llm_core()
            if llm_core:
                logger.info("✅ Using LLM Core from startup manager (with Transportation RAG)")
            else:
                logger.warning("⚠️ LLM Core not ready yet, using fallback")
        
        if llm_core:
            logger.info("🎯 Using LLM Core for response generation (with Transportation RAG)")
            
            # Convert GPS location format if provided
            user_location_dict = None
            if request.gps_location and isinstance(request.gps_location, dict):
                # Support both 'lat'/'lon' and 'latitude'/'longitude' formats
                lat = request.gps_location.get('lat') or request.gps_location.get('latitude')
                lon = request.gps_location.get('lon') or request.gps_location.get('longitude')
                if lat and lon:
                    user_location_dict = {'lat': lat, 'lon': lon}
                    logger.info(f"📍 GPS location provided: {lat:.4f}, {lon:.4f}")
            
            # Process with LLM Core
            llm_result = await llm_core.process_query(
                query=request.message,
                user_id=user_id,
                session_id=session_id,
                user_location=user_location_dict,
                language='en',  # TODO: detect language from query
                max_tokens=500,
                enable_conversation=True
            )
            
            # Extract response and map data from LLM Core result
            ai_response = llm_result.get('response', str(llm_result))
            raw_map_data = llm_result.get('map_data', {})
            detected_intent = llm_result.get('signals', {}).get('needs_transportation', False)
            if detected_intent:
                detected_intent = 'transportation'
            else:
                detected_intent = intent_result.get('intent', 'unknown') if intent_result else 'unknown'
            
            logger.info(f"✅ LLM Core generated response ({len(ai_response)} chars)")
            
        else:
            # Fallback to legacy AI system
            ai_system = None
            if AI_SYSTEM_AVAILABLE:
                try:
                    ai_system = IstanbulDailyTalkAI()
                    logger.info("✅ IstanbulDailyTalkAI initialized as fallback")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize IstanbulDailyTalkAI: {e}")
                    ai_system = None
            
            if ai_system:
                logger.info("🎯 Using IstanbulDailyTalkAI for response generation (fallback mode)")
                # Process message with IstanbulDailyTalkAI
                # For transportation and route planning, request structured response with map data
                detected_intent = intent_result.get('intent', 'unknown') if intent_result else 'unknown'
                needs_map_data = detected_intent in [
                    'transportation', 'route_planning', 'gps_route_planning', 
                    'museum_route_planning', 'airport_transport', 'neighborhood'
                ]
                
                ai_result = ai_system.process_message(
                    user_input=request.message,
                    user_id=user_id,
                    gps_location=request.gps_location,
                    return_structured=needs_map_data
                )
                
                # Extract response and map data
                if isinstance(ai_result, dict):
                    ai_response = ai_result.get('response', str(ai_result))
                    raw_map_data = ai_result.get('map_data', {})
                else:
                    ai_response = str(ai_result)
                    raw_map_data = {}
            else:
                raise HTTPException(
                    status_code=503,
                    detail="No AI system available. Please try again later."
                )
        
        # 🔒 SANITIZE RESPONSE to prevent data leakage
        ai_response = sanitize_response(ai_response)
        logger.debug(f"Response sanitized and ready to return")
        
        # Convert map data to MapVisualization model if available
        map_visualization = None
        if raw_map_data and isinstance(raw_map_data, dict):
            try:
                map_visualization = MapVisualization(
                    type=raw_map_data.get('type'),
                    coordinates=raw_map_data.get('coordinates'),
                    markers=raw_map_data.get('markers'),
                    center=raw_map_data.get('center'),
                    zoom=raw_map_data.get('zoom'),
                    route_data=raw_map_data.get('route_data'),
                    transport_lines=raw_map_data.get('transport_lines')
                )
                logger.info(f"🗺️ Map visualization data generated for {detected_intent}")
            except Exception as e:
                logger.warning(f"Failed to parse map data: {e}")
        
        confidence = intent_result.get('confidence', 0.0) if intent_result else 0.0
        
        # 5. CACHE RESPONSE
        if INFRASTRUCTURE_AVAILABLE and response_cache and confidence > 0.6:
            try:
                response_cache.set(
                    query=request.message,
                    intent=detected_intent,
                    response={
                        'response': ai_response,
                        'intent': detected_intent,
                        'confidence': confidence,
                        'method': method
                    },
                    location=None,
                    ttl=3600  # 1 hour TTL
                )
                logger.info(f"💾 Response cached for intent '{detected_intent}'")
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
        
        # 6. RETURN RESULT
        processing_time = (time.time() - start_time) * 1000
        
        if metric:
            metric.intent = detected_intent
            metric.confidence = confidence
            metric.processing_time_ms = processing_time
            system_monitor.end_request(metric, success=True)
        
        return ChatResponse(
            response=ai_response,
            intent=detected_intent,
            confidence=confidence,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            cache_hit=False,
            processing_time_ms=processing_time,
            ml_enabled=ML_SYSTEMS_AVAILABLE,
            method=method,
            map_data=map_visualization
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}", exc_info=True)
        
        if metric:
            metric.errors.append(str(e))
            system_monitor.end_request(metric, success=False)
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/chat/health")
async def chat_health():
    """
    Health check for chat system
    
    Returns system status and available features
    """
    # Check if LLM Core is available from startup manager
    llm_core_ready = False
    if LLM_CORE_AVAILABLE:
        llm_core = fast_startup_manager.get_pure_llm_core()
        llm_core_ready = llm_core is not None
    
    return {
        "status": "healthy" if llm_core_ready else "degraded",
        "features": {
            "llm_core": llm_core_ready,
            "ai_system": AI_SYSTEM_AVAILABLE,
            "ml_systems": ML_SYSTEMS_AVAILABLE,
            "intent_classification": INTENT_CLASSIFIER_AVAILABLE,
            "enhanced_understanding": ENHANCED_QUERY_UNDERSTANDING_ENABLED,
            "infrastructure": INFRASTRUCTURE_AVAILABLE,
            "caching": INFRASTRUCTURE_AVAILABLE and response_cache is not None,
            "rate_limiting": INFRASTRUCTURE_AVAILABLE and rate_limiter is not None,
            "monitoring": INFRASTRUCTURE_AVAILABLE and system_monitor is not None,
            "transportation_rag": llm_core_ready  # LLM Core includes Transportation RAG
        },
        "timestamp": datetime.now().isoformat()
    }
