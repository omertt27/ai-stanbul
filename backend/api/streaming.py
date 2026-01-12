"""
Streaming Chat API Endpoints

Provides real-time streaming responses via:
- Server-Sent Events (SSE) - For simple HTTP streaming
- WebSocket - For bidirectional real-time communication

Author: AI Istanbul Team
Date: December 2024
"""

import asyncio
import json
import logging
import time
import re
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import data collection for feedback loop
from services.data_collection import log_chat_interaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stream", tags=["Streaming"])


def clean_llm_response(text: str) -> str:
    """
    Clean LLM response to remove any prompt leakage.
    
    This removes:
    - System instruction fragments
    - Language instruction markers
    - Internal notes and warnings
    - Map/visualization mentions
    - Conversation history leakage
    - Context metadata
    """
    if not text:
        return text
    
    # Patterns that indicate prompt leakage (case insensitive)
    leakage_patterns = [
        r'\*\*Map:\*\*.*?(?=\n|$)',  # **Map:** ...
        r'Map:.*will be shown.*?(?=\n|$)',  # Map will be shown
        r'⚠️\s*CRITICAL.*?(?=\n\n|\Z)',  # ⚠️ CRITICAL warnings
        r'CRITICAL:.*?(?=\n\n|\Z)',  # CRITICAL: ...
        r'\[Respond in \w+\]',  # [Respond in Turkish]
        r'\[.*instruction.*\]',  # [instruction...]
        r'---\s*⚠️.*',  # --- ⚠️ ...
        r'Write in \w+ \(\w+\) only\.?',  # Write in TURKISH (Türkçe) only
        r'User Question:.*',  # User Question: ...
        r'❌ DO NOT.*?(?=\n|$)',  # ❌ DO NOT ...
        r'MUST be written ONLY in.*?(?=\n|$)',  # MUST be written ONLY in ...
        r'Never use.*other language.*?(?=\n|$)',  # Never use other language
        r'A map will be shown to the user\.?',  # Map mention
        r'The map shows.*?(?=\n|$)',  # Map description
        r'As shown on the map.*?(?=\n|$)',  # Map reference
        # NEW: Conversation history leakage patterns
        r'---\s*User:.*?(?=---|$)',  # --- User: ... pattern
        r'Response:.*?(?=---|$)',  # Response: ... pattern
        r'Turn \d+:.*?(?=Turn \d+:|$)',  # Turn numbering
        r'Bot:.*?(?=\n|$)',  # Bot: labels
        r'Intent:.*?(?=\n|$)',  # Intent: labels
        r'Locations:.*?(?=\n|$)',  # Locations: labels
        r'Session Context:.*?(?=\n\n|\Z)',  # Session context
        r'Last Mentioned.*?(?=\n|$)',  # Last Mentioned metadata
        r"User's GPS Location.*?(?=\n|$)",  # GPS metadata
        r'Active Task:.*?(?=\n|$)',  # Task tracking
        r'User Preferences:.*?(?=\n|$)',  # Preference data
        r'Conversation Age:.*?(?=\n|$)',  # Conversation stats
        r'CONVERSATION HISTORY:.*?(?=\n\n|\Z)',  # History section
        r'CURRENT QUERY:.*?(?=\n\n|\Z)',  # Query markers
        r'YOUR TASK:.*?(?=\n\n|\Z)',  # Task instructions
        r'RETURN FORMAT.*?(?=\n\n|\Z)',  # Format instructions
        r'"has_references".*?(?=\n|$)',  # JSON analysis
        r'"resolved_references".*?(?=\n|$)',  # Reference resolution
        r'"implicit_context".*?(?=\n|$)',  # Context analysis
        r'"needs_clarification".*?(?=\n|$)',  # Clarification flags
    ]
    
    cleaned = text
    for pattern in leakage_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up multiple newlines and trailing whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    # If we removed significant content, log it
    if len(cleaned) < len(text) * 0.8:
        logger.warning(f"Cleaned significant prompt leakage from response ({len(text)} -> {len(cleaned)} chars)")
    
    return cleaned


router = APIRouter(prefix="/api/stream", tags=["Streaming"])


# Request/Response Models
class StreamChatRequest(BaseModel):
    """Request model for streaming chat"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    language: str = Field(default="en", description="Response language")
    user_location: Optional[Dict[str, float]] = Field(None, description="User GPS location")
    include_context: bool = Field(default=True, description="Include conversation context")


# ==========================================
# Server-Sent Events (SSE) Endpoint
# ==========================================

@router.post("/chat")
async def stream_chat_sse(request: StreamChatRequest):
    """
    Stream chat response using Server-Sent Events (SSE).
    
    This is the recommended method for most use cases as it:
    - Works through HTTP (no special protocol)
    - Automatically reconnects
    - Works well with proxies and load balancers
    
    Response format (SSE):
    ```
    event: start
    data: {"timestamp": 1234567890}
    
    event: token
    data: {"content": "Hello"}
    
    event: token
    data: {"content": " there"}
    
    event: complete
    data: {"content": "Hello there!", "metadata": {...}}
    ```
    """
    # ==========================================
    # 🔥 ENHANCED DEBUG LOGGING - START
    # ==========================================
    logger.info("=" * 80)
    logger.info("🚀 NEW STREAMING CHAT REQUEST RECEIVED")
    logger.info("=" * 80)
    logger.info(f"📝 Query: '{request.message}'")
    logger.info(f"🌍 Language: {request.language}")
    logger.info(f"🆔 Session ID: {request.session_id}")
    logger.info(f"📍 GPS Location Received: {request.user_location is not None}")
    
    if request.user_location:
        logger.info(f"   📍 GPS Data: {request.user_location}")
        lat = request.user_location.get('latitude') or request.user_location.get('lat')
        lon = request.user_location.get('longitude') or request.user_location.get('lon')
        logger.info(f"   🗺️ Coordinates: lat={lat}, lon={lon}")
    else:
        logger.warning("   ⚠️ NO GPS LOCATION PROVIDED IN REQUEST")
    
    # Analyze query type
    query_lower = request.message.lower()
    logger.info(f"🔍 Query Analysis:")
    logger.info(f"   - Contains 'how': {'how' in query_lower}")
    logger.info(f"   - Contains 'get to': {'get to' in query_lower}")
    logger.info(f"   - Contains 'go to': {'go to' in query_lower}")
    logger.info(f"   - Contains 'taksim': {'taksim' in query_lower}")
    logger.info(f"   - Contains 'route': {'route' in query_lower}")
    logger.info(f"   - Contains 'metro': {'metro' in query_lower}")
    logger.info("=" * 80)
    # ==========================================
    # 🔥 ENHANCED DEBUG LOGGING - END
    # ==========================================
    
    async def generate_stream():
        try:
            # Import services
            from services.streaming_llm_service import get_streaming_llm_service
            from services.conversation_history_service import get_conversation_history_service
            from services.advanced_nlp_service import get_nlp_service
            
            streaming_service = get_streaming_llm_service()
            history_service = get_conversation_history_service()
            nlp_service = get_nlp_service()
            
            # Process with NLP
            nlp_result = nlp_service.process(request.message)
            intent_value = nlp_result.intent.value if hasattr(nlp_result.intent, 'value') else str(nlp_result.intent)
            
            # === RAG ENHANCEMENT: Retrieve relevant context ===
            rag_context = None
            map_data = None
            route_data = None
            transport_alternatives = None  # NEW: Multi-route data
            
            # Check if this is a transportation query
            query_lower = request.message.lower()
            transportation_keywords = ['how do i get', 'how can i get', 'how to get', 'route to', 
                                       'from', 'directions to', 'navigate to', 'take me to', 'go to',
                                       'way to', 'best way to', 'how can i reach', 'how do i reach',
                                       'taksim', 'kadikoy', 'kadıköy', 'sultanahmet', 'metro', 'tram']
            is_transportation = (
                intent_value in ['transportation', 'directions', 'route', 'navigate'] or
                any(keyword in query_lower for keyword in transportation_keywords)
            )
            
            # Check if query asks for route alternatives/options
            use_multi_route = is_transportation and any(keyword in query_lower for keyword in [
                'how to get', 'how do i get', 'route to', 'way to',
                'directions to', 'how can i reach', 'best way to',
                'options', 'alternatives', 'routes', 'ways'
            ])
            
            # Check if this is a trip planning query
            trip_planning_keywords = [
                '1 day', '2 day', '3 day', '4 day', '5 day',
                '1-day', '2-day', '3-day', '4-day', '5-day',
                'one day', 'two day', 'three day', 'four day', 'five day',
                'itinerary', 'trip plan', 'trip itinerary', 'day trip',
                'plan my trip', 'plan a trip', 'plan trip',
                'günlük', 'gün', 'gezi planı', 'gezi plan',
            ]
            is_trip_planning = any(keyword in query_lower for keyword in trip_planning_keywords)
            trip_duration = None
            if is_trip_planning:
                # Extract number of days
                import re
                day_patterns = [
                    r'(\d+)\s*(?:day|gün)',
                    r'(one|two|three|four|five)\s*day',
                    r'(bir|iki|üç|dört|beş)\s*gün'
                ]
                day_words = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'bir': 1, 'iki': 2, 'üç': 3, 'dört': 4, 'beş': 5
                }
                for pattern in day_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        val = match.group(1)
                        if val.isdigit():
                            trip_duration = int(val)
                        elif val in day_words:
                            trip_duration = day_words[val]
                        break
                # Default to 1 day if just "itinerary" without number
                if trip_duration is None:
                    trip_duration = 1
                logger.info(f"🗓️ Trip planning query detected - {trip_duration} day(s)")
            
            # Get RAG context based on query type
            trip_plan_data = None
            try:
                if is_trip_planning:
                    # Use Trip Planner for multi-day itinerary queries
                    logger.info(f"🗓️ Trip planning query - using Trip Planner ({trip_duration} days)")
                    from services.trip_planner import get_trip_planner
                    trip_planner = get_trip_planner()
                    
                    # Parse user preferences from the query
                    preferences = trip_planner.parse_user_preferences(request.message)
                    trip_duration = preferences.get('duration', trip_duration)
                    
                    # Map duration to template key
                    if trip_duration == 1:
                        trip_id = "1_day_highlights"
                    elif trip_duration <= 3:
                        trip_id = "3_day_classic"
                    else:
                        trip_id = "5_day_complete"
                    
                    # Get trip plan map data
                    trip_map_data = trip_planner.get_trip_map_data(trip_id)
                    
                    if trip_map_data:
                        map_data = trip_map_data
                        trip_plan_data = trip_map_data
                        logger.info(f"✅ Got trip plan: {trip_map_data.get('name')} ({trip_duration} days)")
                        
                        # Get detailed RAG context for LLM
                        rag_context = trip_planner.get_trip_rag_context(trip_id, language=request.language)
                        
                        # Add user preferences info
                        if preferences.get('interests'):
                            rag_context += f"\n\n**User Interests:** {', '.join(preferences['interests'])}"
                        if preferences.get('style'):
                            rag_context += f"\n**Travel Style:** {preferences['style'].value}"
                        
                        logger.info(f"✅ Built detailed trip planning RAG context ({len(rag_context)} chars)")
                    
                elif is_transportation:
                    # Use Moovit-style enhanced multi-route system for transportation queries
                    logger.info("🚇 Transportation query detected - using ENHANCED MULTI-ROUTE system")
                    
                    if use_multi_route:
                        # MOOVIT-LEVEL: Use route optimizer for multi-route with comfort scores
                        logger.info("🗺️ Using MOOVIT-STYLE multi-route optimizer with comfort scoring")
                        try:
                            from services.transportation_route_integration import get_route_integration
                            
                            # Extract locations from query
                            import re
                            
                            # Get user location
                            user_loc = None
                            if request.user_location:
                                user_loc = {
                                    'lat': request.user_location.get('latitude') or request.user_location.get('lat'),
                                    'lon': request.user_location.get('longitude') or request.user_location.get('lon')
                                }
                                logger.info(f"📍 GPS RECEIVED FROM REQUEST: {user_loc}")
                            else:
                                logger.warning("⚠️ NO GPS LOCATION IN REQUEST - user_location is None")
                            
                            # Parse origin and destination
                            logger.info(f"🔍 EXTRACTING LOCATIONS FROM QUERY: '{request.message}'")
                            locations = _extract_transportation_locations(request.message, user_loc)
                            
                            # DEBUG: Log extraction result
                            if locations:
                                logger.info(f"✅ LOCATIONS EXTRACTED SUCCESSFULLY:")
                                logger.info(f"   📍 Origin: {locations.get('origin')}")
                                logger.info(f"   🎯 Destination: {locations.get('destination')}")
                                logger.info(f"   🗺️ Origin GPS: {locations.get('origin_gps')}")
                                logger.info(f"   🗺️ Destination GPS: {locations.get('destination_gps')}")
                            else:
                                logger.error("❌ LOCATION EXTRACTION FAILED - locations is None")
                                logger.error(f"   Query was: '{request.message}'")
                                logger.error(f"   GPS available: {user_loc is not None}")
                            
                            if locations and locations.get('origin') and locations.get('destination'):
                                route_integration = get_route_integration()
                                
                                # Get multi-route alternatives with comfort scoring
                                result = route_integration.get_route_alternatives(
                                    origin=locations['origin'],
                                    destination=locations['destination'],
                                    origin_gps=locations.get('origin_gps'),
                                    destination_gps=locations.get('destination_gps'),
                                    num_alternatives=3,  # Generate 3 route options
                                    generate_llm_summaries=False,  # LLM will generate its own
                                    user_language=request.language
                                )
                                
                                if result['success']:
                                    logger.info(f"✅ Got {len(result.get('alternatives', []))} route alternatives with comfort scores")
                                    
                                    # Log each route alternative
                                    for i, alt in enumerate(result.get('alternatives', [])):
                                        logger.info(f"   Route {i+1}: {alt.get('duration_minutes')}min, {alt.get('num_transfers')} transfers, comfort: {alt.get('comfort_score', {}).get('overall_comfort', 0):.0f}/100")
                                    
                                    # Store multi-route data
                                    transport_alternatives = {
                                        'primary_route': result.get('primary_route'),
                                        'alternatives': result.get('alternatives', []),
                                        'route_comparison': result.get('route_comparison', {}),
                                        'map_data': result.get('map_data')
                                    }
                                    
                                    # Build rich context for LLM
                                    rag_context = _build_multi_route_context(result, request.language)
                                    
                                    # Set map_data for visualization
                                    if result.get('map_data'):
                                        map_data = result['map_data']
                                        # Add multi-route fields
                                        map_data.update({
                                            'type': 'multi_route',
                                            'multi_routes': result.get('alternatives', []),
                                            'primary_route': result.get('primary_route'),
                                            'route_comparison': result.get('route_comparison', {})
                                        })
                                    
                                    logger.info("✅ Multi-route context built with comfort scores and comparison")
                                else:
                                    logger.warning(f"Multi-route generation failed: {result.get('error')}")
                                    # Fallback to standard RAG
                                    use_multi_route = False
                            else:
                                logger.warning("Could not extract origin/destination - falling back to standard RAG")
                                use_multi_route = False
                                
                        except Exception as multi_err:
                            logger.error(f"Multi-route system failed: {multi_err}")
                            use_multi_route = False
                    
                    # Fallback or standard transportation RAG
                    if not use_multi_route or not rag_context:
                        logger.info("🚇 Using standard Transportation RAG system")
                        from services.transportation_rag_system import get_transportation_rag
                        transport_rag = get_transportation_rag()
                        
                        if transport_rag:
                            # Get route context from Transportation RAG
                            user_loc = None
                            if request.user_location:
                                user_loc = {
                                    'lat': request.user_location.get('latitude') or request.user_location.get('lat'),
                                    'lon': request.user_location.get('longitude') or request.user_location.get('lon')
                                }
                            
                            route_context = transport_rag.get_rag_context_for_query(request.message, user_loc)
                            if route_context:
                                rag_context = route_context
                                logger.info(f"✅ Got transportation RAG context")
                            
                            # Get map data for visualization
                            if not map_data:
                                map_data = transport_rag.get_map_data_for_last_route()
                                if map_data:
                                    logger.info(f"🗺️ Got map_data for route visualization")
                            
                            # Get enriched route data
                            if transport_rag.last_route and not route_data:
                                basic_route_data = {
                                    'origin': transport_rag.last_route.origin,
                                    'destination': transport_rag.last_route.destination,
                                    'steps': transport_rag.last_route.steps,
                                    'total_time': transport_rag.last_route.total_time,
                                    'total_distance': transport_rag.last_route.total_distance,
                                    'transfers': transport_rag.last_route.transfers,
                                    'lines_used': transport_rag.last_route.lines_used
                                }
                                route_data = transport_rag.station_normalizer.enrich_route_data(basic_route_data)
                                logger.info(f"✅ Got enriched route_data: {route_data.get('origin')} → {route_data.get('destination')}")
                else:
                    # Use Database RAG for general queries
                    logger.info("🔍 Using Database RAG for general query")
                    from services.database_rag_service import get_rag_service
                    from database import get_db
                    
                    # Get database session
                    db = next(get_db())
                    try:
                        rag_service = get_rag_service(db=db)
                        if rag_service:
                            rag_context = rag_service.get_context_for_llm(request.message, top_k=3)
                            if rag_context:
                                logger.info(f"✅ Got database RAG context")
                    finally:
                        db.close()
                        
            except Exception as rag_err:
                logger.warning(f"RAG retrieval failed: {rag_err}")
            
            # Get conversation context
            context = {
                "location": request.user_location,
                "intent": intent_value,
                "entities": [e.to_dict() for e in nlp_result.entities],
                "signals": {
                    "needs_trip_planning": is_trip_planning,
                    "needs_transportation": is_transportation
                }
            }
            
            # Add RAG context
            if rag_context:
                context["rag_context"] = rag_context
            
            # Add map_data and route_data to context for prompt builder
            if map_data:
                context["map_data"] = map_data
            if route_data:
                context["route_data"] = route_data
            if trip_plan_data:
                context["trip_plan"] = trip_plan_data
            if transport_alternatives:
                context["transport_alternatives"] = transport_alternatives  # Multi-route data
            
            # Add trip planning signal for LLM prompt
            if is_trip_planning:
                context["needs_trip_planning"] = True
                context["trip_duration"] = trip_duration
            
            # Add conversation history
            if request.include_context and request.session_id:
                conversation_history = history_service.get_conversation_context(
                    request.session_id,
                    max_turns=5
                )
                context["conversation_history"] = conversation_history
            
            # Send start event with trip planning flag
            start_data = {'timestamp': time.time(), 'intent': intent_value}
            if is_trip_planning:
                start_data['is_trip_planning'] = True
                start_data['trip_duration'] = trip_duration
            yield f"event: start\ndata: {json.dumps(start_data)}\n\n"
            
            # Use NLP-detected language (overrides request.language)
            # This ensures we respond in the same language as the query
            detected_language = nlp_result.language if hasattr(nlp_result, 'language') else request.language
            logger.info(f"🌍 Language - Request: {request.language}, Detected: {detected_language}, Using: {detected_language}")
            
            # Stream response
            full_response = ""
            async for chunk in streaming_service.stream_chat_response(
                message=request.message,
                context=context,
                language=detected_language  # Use detected language instead of request.language
            ):
                if chunk["type"] == "token":
                    full_response += chunk["content"]
                    yield f"event: token\ndata: {json.dumps({'content': chunk['content']})}\n\n"
                
                elif chunk["type"] == "complete":
                    # Clean the response to remove any prompt leakage
                    cleaned_response = clean_llm_response(full_response)
                    
                    # Save to history (use cleaned version)
                    if request.session_id:
                        history_service.add_exchange(
                            session_id=request.session_id,
                            user_message=request.message,
                            assistant_response=cleaned_response,
                            metadata={
                                "intent": intent_value,
                                "language": request.language
                            }
                        )
                    
                    # Log interaction for feedback loop (fine-tuning data collection)
                    interaction_id = None
                    try:
                        interaction_id = log_chat_interaction(
                            user_query=request.message,
                            llm_response=cleaned_response,
                            language=request.language or nlp_result.language,
                            intent=intent_value,
                            session_id=request.session_id,
                            has_map_data=bool(map_data or route_data or trip_plan_data),
                            method='streaming'
                        )
                        logger.info(f"📝 Logged interaction {interaction_id} for feedback loop")
                    except Exception as log_err:
                        logger.warning(f"⚠️ Failed to log interaction for feedback: {log_err}")
                    
                    # Build metadata including map_data and route_data if available
                    metadata = {
                        'intent': intent_value,
                        'language': nlp_result.language,
                        'interaction_id': interaction_id  # Include for feedback
                    }
                    if map_data:
                        metadata['map_data'] = map_data
                        logger.info(f"📍 Including map_data in response: {len(str(map_data))} bytes")
                    if route_data:
                        metadata['route_data'] = route_data
                        logger.info(f"🚇 Including route_data in response: {route_data.get('origin')} → {route_data.get('destination')}")
                    if trip_plan_data:
                        metadata['trip_plan'] = trip_plan_data
                        logger.info(f"🗓️ Including trip_plan in response")
                    if transport_alternatives:
                        # Include multi-route data for frontend
                        metadata['transport_alternatives'] = transport_alternatives
                        logger.info(f"🗺️ Including transport_alternatives with {len(transport_alternatives.get('alternatives', []))} routes")
                    
                    # ==========================================
                    # 🔥 FINAL RESPONSE DEBUG LOGGING
                    # ==========================================
                    logger.info("=" * 80)
                    logger.info("✅ RESPONSE READY TO SEND")
                    logger.info("=" * 80)
                    logger.info(f"📝 Response length: {len(cleaned_response)} chars")
                    logger.info(f"🎯 Intent: {intent_value}")
                    logger.info(f"🌍 Language: {nlp_result.language}")
                    logger.info(f"📍 Has map_data: {bool(map_data)}")
                    logger.info(f"🚇 Has route_data: {bool(route_data)}")
                    logger.info(f"🗺️ Has transport_alternatives: {bool(transport_alternatives)}")
                    logger.info(f"🗓️ Has trip_plan: {bool(trip_plan_data)}")
                    
                    if map_data:
                        logger.info(f"   Map type: {map_data.get('type')}")
                        logger.info(f"   Markers: {len(map_data.get('markers', []))}")
                    
                    if route_data:
                        logger.info(f"   Route: {route_data.get('origin')} → {route_data.get('destination')}")
                        logger.info(f"   Duration: {route_data.get('total_time')}")
                    
                    if transport_alternatives:
                        alts = transport_alternatives.get('alternatives', [])
                        logger.info(f"   Alternatives count: {len(alts)}")
                        if alts:
                            logger.info(f"   Best route: {alts[0].get('duration_minutes')}min, {alts[0].get('num_transfers')} transfers")
                    
                    logger.info("=" * 80)
                    # ==========================================
                    
                    # Send completion event with cleaned response
                    yield f"event: complete\ndata: {json.dumps({'content': cleaned_response, 'metadata': metadata})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/chat")
async def stream_chat_sse_get(
    message: str = Query(..., description="User message"),
    session_id: Optional[str] = Query(None, description="Session ID"),
    language: str = Query("en", description="Language")
):
    """
    Stream chat via GET request (for EventSource compatibility).
    
    Usage in JavaScript:
    ```javascript
    const evtSource = new EventSource('/api/stream/chat?message=hello&language=en');
    evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.content);
    };
    ```
    """
    request = StreamChatRequest(
        message=message,
        session_id=session_id,
        language=language
    )
    return await stream_chat_sse(request)


# ==========================================
# WebSocket Endpoint
# ==========================================

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_json(self, session_id: str, data: Dict[str, Any]):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)
    
    async def broadcast(self, data: Dict[str, Any]):
        for connection in self.active_connections.values():
            await connection.send_json(data)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time bidirectional chat.
    
    Protocol:
    1. Client connects to /api/stream/ws/{session_id}
    2. Client sends: {"type": "message", "content": "Hello", "language": "en"}
    3. Server streams: {"type": "token", "content": "Hi"}
    4. Server sends: {"type": "complete", "content": "Hi there!"}
    
    Additional message types:
    - {"type": "ping"} -> {"type": "pong"}
    - {"type": "context"} -> {"type": "context", "history": [...]}
    """
    await manager.connect(websocket, session_id)
    
    try:
        # Import services
        from services.streaming_llm_service import get_streaming_llm_service
        from services.conversation_history_service import get_conversation_history_service
        from services.advanced_nlp_service import get_nlp_service
        
        streaming_service = get_streaming_llm_service()
        history_service = get_conversation_history_service()
        nlp_service = get_nlp_service()
        
        # Send connection acknowledgment
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": time.time()
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            msg_type = data.get("type", "message")
            
            # Handle ping
            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
                continue
            
            # Handle context request
            if msg_type == "context":
                history = history_service.get_conversation_context(session_id)
                await websocket.send_json({
                    "type": "context",
                    "history": history,
                    "timestamp": time.time()
                })
                continue
            
            # Handle chat message
            if msg_type == "message":
                content = data.get("content", "")
                language = data.get("language", "en")
                user_location = data.get("location")
                
                if not content:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Empty message"
                    })
                    continue
                
                # Process with NLP
                nlp_result = nlp_service.process(content)
                
                # Get context
                conversation_history = history_service.get_conversation_context(session_id)
                context = {
                    "conversation_history": conversation_history,
                    "location": user_location,
                    "intent": nlp_result.intent.value
                }
                
                # Send typing indicator
                await websocket.send_json({
                    "type": "typing",
                    "intent": nlp_result.intent.value
                })
                
                # Stream response
                full_response = ""
                try:
                    async for chunk in streaming_service.stream_chat_response(
                        message=content,
                        context=context,
                        language=language
                    ):
                        if chunk["type"] == "token":
                            full_response += chunk["content"]
                            await websocket.send_json({
                                "type": "token",
                                "content": chunk["content"]
                            })
                        
                        elif chunk["type"] == "complete":
                            # Clean response to remove prompt leakage
                            cleaned_response = clean_llm_response(full_response)
                            
                            # Save to history
                            history_service.add_exchange(
                                session_id=session_id,
                                user_message=content,
                                assistant_response=cleaned_response,
                                metadata={
                                    "intent": nlp_result.intent.value,
                                    "language": language
                                }
                            )
                            
                            await websocket.send_json({
                                "type": "complete",
                                "content": cleaned_response,
                                "metadata": {
                                    "intent": nlp_result.intent.value,
                                    "language": nlp_result.language,
                                    "entities": [e.to_dict() for e in nlp_result.entities]
                                }
                            })
                            
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)


# ==========================================
# Utility Endpoints
# ==========================================

@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str, limit: int = Query(10, le=50)):
    """Get conversation history for a session."""
    from services.conversation_history_service import get_conversation_history_service
    
    history_service = get_conversation_history_service()
    context = history_service.get_conversation_context(session_id, max_turns=limit)
    summary = history_service.get_conversation_summary(session_id)
    
    return {
        "session_id": session_id,
        "messages": context,
        "summary": summary
    }


@router.delete("/history/{session_id}")
async def delete_conversation(session_id: str):
    """Delete conversation history for a session."""
    from services.conversation_history_service import get_conversation_history_service
    
    history_service = get_conversation_history_service()
    success = history_service.delete_conversation(session_id)
    
    if success:
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/analyze")
async def analyze_text(
    text: str = Query(..., description="Text to analyze"),
    language: str = Query("auto", description="Language hint")
):
    """
    Analyze text using NLP service.
    
    Returns intent, entities, sentiment, and keywords.
    """
    from services.advanced_nlp_service import get_nlp_service
    
    nlp_service = get_nlp_service()
    result = nlp_service.process(text)
    
    return result.to_dict()


@router.get("/suggestions")
async def get_location_suggestions(
    query: str = Query(..., min_length=2, description="Partial location name")
):
    """Get location suggestions for autocomplete."""
    from services.advanced_nlp_service import get_nlp_service
    
    nlp_service = get_nlp_service()
    suggestions = nlp_service.get_location_suggestions(query)
    
    return {"suggestions": suggestions}


# ==========================================
# Metrics & Health Endpoints
# ==========================================

@router.get("/metrics")
async def get_streaming_metrics():
    """
    Get streaming service metrics.
    
    Returns:
    - Request counts (total, success, failed)
    - Cache hit rate
    - Average latency
    - Circuit breaker state
    - Token counts
    """
    from services.streaming_llm_service import get_streaming_metrics
    
    return get_streaming_metrics()


@router.get("/health")
async def streaming_health_check():
    """
    Check streaming service health.
    
    Returns service status, circuit breaker state, and basic metrics.
    """
    from services.streaming_llm_service import get_streaming_llm_service
    
    service = get_streaming_llm_service()
    return await service.health_check()


@router.post("/cache/clear")
async def clear_streaming_cache():
    """Clear the streaming response cache."""
    from services.streaming_llm_service import get_streaming_llm_service
    
    service = get_streaming_llm_service()
    service.clear_cache()
    
    return {"status": "success", "message": "Streaming cache cleared"}


# ==========================================
# Helper Functions for Multi-Route System
# ==========================================

def _extract_transportation_locations(query: str, user_location: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
    """
    Extract origin and destination from transportation query.
    
    Returns dict with:
    - origin: origin name
    - destination: destination name
    - origin_gps: optional GPS dict
    - destination_gps: optional GPS dict
    """
    import re
    
    # ==========================================
    # 🔥 ENHANCED DEBUG LOGGING - LOCATION EXTRACTION
    # ==========================================
    logger.info("🔍 LOCATION EXTRACTION STARTED")
    logger.info(f"   Query: '{query}'")
    logger.info(f"   GPS Available: {user_location is not None}")
    if user_location:
        logger.info(f"   GPS Data: {user_location}")
    # ==========================================
    
    query_lower = query.lower()
    
    # Check for GPS patterns
    uses_gps_origin = any(pattern in query_lower for pattern in [
        'from my location', 'from here', 'from current location',
        'from where i am', 'from my position', 'starting from here'
    ])
    
    uses_gps_dest = any(pattern in query_lower for pattern in [
        'to my location', 'to here', 'to current location',
        'to where i am', 'back here'
    ])
    
    logger.info(f"   GPS Pattern Detection:")
    logger.info(f"      - Uses GPS origin: {uses_gps_origin}")
    logger.info(f"      - Uses GPS destination: {uses_gps_dest}")
    
    # Extract locations using patterns
    # Pattern 1: "from X to Y" or "X to Y"
    logger.info(f"   Testing Pattern 1: 'from X to Y'")
    match = re.search(r'(?:from\s+)?([a-zğüşöçıİ\s]+?)\s+to\s+([a-zğüşöçıİ\s]+)', query_lower, re.IGNORECASE)
    
    if match:
        origin = match.group(1).strip()
        destination = match.group(2).strip()
        logger.info(f"   ✅ Pattern 1 MATCHED:")
        logger.info(f"      - Raw origin: '{origin}'")
        logger.info(f"      - Raw destination: '{destination}'")
        
        # Replace GPS placeholders with actual location
        result = {}
        
        if 'my location' in origin or 'here' in origin or 'current location' in origin:
            if user_location:
                result['origin'] = 'Current Location'
                result['origin_gps'] = user_location
                logger.info(f"      - Origin set to GPS: {user_location}")
            else:
                logger.warning(f"      ❌ GPS origin requested but no location available")
                return None  # GPS origin but no location available
        else:
            result['origin'] = origin.title()
            logger.info(f"      - Origin set to: '{result['origin']}'")
            
        if 'my location' in destination or 'here' in destination or 'current location' in destination:
            if user_location:
                result['destination'] = 'Current Location'
                result['destination_gps'] = user_location
                logger.info(f"      - Destination set to GPS: {user_location}")
            else:
                logger.warning(f"      ❌ GPS destination requested but no location available")
                return None  # GPS destination but no location available
        else:
            result['destination'] = destination.title()
            logger.info(f"      - Destination set to: '{result['destination']}'")
        
        logger.info(f"   ✅ LOCATION EXTRACTION SUCCESS (Pattern 1): {result}")
        return result
    
    # Pattern 2: "how to get to X" (implies GPS origin if available)
    # Handles: "how can i go to X", "how to get to X", "how do i reach X", etc.
    # Also handles typos like "ow can i go" (missing 'h')
    logger.info(f"   Testing Pattern 2: 'how to get to X'")
    match = re.search(r'(?:h?ow|way)\s+(?:do i |can i |to )?(?:get|go|reach)(?:\s+to)?\s+([a-zğüşöçıİ\s]+)', query_lower, re.IGNORECASE)
    
    if match and user_location:
        destination = match.group(1).strip()
        # Remove "to" from destination if it was captured
        if destination.startswith('to '):
            destination = destination[3:]
        
        result = {
            'origin': 'Current Location',
            'origin_gps': user_location,
            'destination': destination.strip().title()
        }
        logger.info(f"   ✅ Pattern 2 MATCHED:")
        logger.info(f"      - Origin: GPS (Current Location)")
        logger.info(f"      - Destination: '{result['destination']}'")
        logger.info(f"   ✅ LOCATION EXTRACTION SUCCESS (Pattern 2): {result}")
        return result
    elif match and not user_location:
        logger.warning(f"   ⚠️ Pattern 2 matched but NO GPS available - cannot use Current Location")
    
    # Pattern 3: Just "go to X" or "to X" with GPS
    logger.info(f"   Testing Pattern 3: 'go to X' or 'to X'")
    match = re.search(r'(?:go\s+to|to)\s+([a-zğüşöçıİ\s]+)', query_lower, re.IGNORECASE)
    if match and user_location:
        destination = match.group(1).strip()
        result = {
            'origin': 'Current Location',
            'origin_gps': user_location,
            'destination': destination.strip().title()
        }
        logger.info(f"   ✅ Pattern 3 MATCHED:")
        logger.info(f"      - Origin: GPS (Current Location)")
        logger.info(f"      - Destination: '{result['destination']}'")
        logger.info(f"   ✅ LOCATION EXTRACTION SUCCESS (Pattern 3): {result}")
        return result
    elif match and not user_location:
        logger.warning(f"   ⚠️ Pattern 3 matched but NO GPS available - cannot use Current Location")
    
    logger.error(f"   ❌ LOCATION EXTRACTION FAILED - No patterns matched")
    logger.error(f"      Query was: '{query}'")
    logger.error(f"      GPS available: {user_location is not None}")
    return None


def _build_multi_route_context(result: Dict[str, Any], language: str) -> str:
    """Build context string from multi-route result for LLM"""
    lines = []
    
    # Primary route
    pr = result.get('primary_route')
    if pr:
        lines.append(f"PRIMARY ROUTE: {pr['origin']} → {pr['destination']}")
        lines.append(f"Duration: {pr['duration_minutes']} minutes")
        lines.append(f"Distance: {pr.get('distance_km', pr.get('total_distance', 0)):.1f} km")
        lines.append(f"Transfers: {pr['num_transfers']}")
        lines.append(f"Walking: {int(pr['walking_meters'])}m")
        
        # Comfort scores
        if pr.get('comfort_score'):
            cs = pr['comfort_score']
            lines.append(f"Comfort Score: {cs.get('overall_comfort', 0):.0f}/100")
            lines.append(f"  - Crowding: {cs.get('crowding_comfort', 0):.0f}/100")
            lines.append(f"  - Transfers: {cs.get('transfer_comfort', 0):.0f}/100")
            lines.append(f"  - Walking: {cs.get('walking_comfort', 0):.0f}/100")
        
        lines.append("")
    
    # Alternatives
    alternatives = result.get('alternatives', [])
    if alternatives:
        lines.append(f"ALTERNATIVE ROUTES ({len(alternatives)} options):")
        for i, alt in enumerate(alternatives, 1):
            lines.append(f"\n{i}. {alt['preference'].upper().replace('-', ' ')}")
            lines.append(f"   Duration: {alt['duration_minutes']} min")
            lines.append(f"   Transfers: {alt['num_transfers']}")
            lines.append(f"   Walking: {int(alt['walking_meters'])}m")
            
            if alt.get('comfort_score'):
                lines.append(f"   Comfort: {alt['comfort_score']['overall_comfort']:.0f}/100")
            
            if alt.get('overall_score'):
                lines.append(f"   Score: {alt['overall_score']:.1f}/100")
            
            if alt.get('highlights'):
                lines.append(f"   Highlights: {', '.join(alt['highlights'])}")
    
    # Route comparison
    comparison = result.get('route_comparison', {})
    if comparison:
        lines.append("\nQUICK COMPARISON:")
        if comparison.get('fastest_route') is not None:
            lines.append(f"  - Fastest: Route {comparison['fastest_route'] + 1}")
        if comparison.get('fewest_transfers') is not None:
            lines.append(f"  - Fewest Transfers: Route {comparison['fewest_transfers'] + 1}")
        if comparison.get('most_comfortable') is not None:
            lines.append(f"  - Most Comfortable: Route {comparison['most_comfortable'] + 1}")
        if comparison.get('least_walking') is not None:
            lines.append(f"  - Least Walking: Route {comparison['least_walking'] + 1}")
    
    return "\n".join(lines)
