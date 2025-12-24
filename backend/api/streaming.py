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
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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
            
            # Check if this is a transportation query
            query_lower = request.message.lower()
            transportation_keywords = ['how do i get', 'how can i get', 'how to get', 'route to', 
                                       'from', 'directions to', 'navigate to', 'take me to', 'go to',
                                       'taksim', 'kadikoy', 'kadıköy', 'sultanahmet', 'metro', 'tram']
            is_transportation = (
                intent_value in ['transportation', 'directions', 'route', 'navigate'] or
                any(keyword in query_lower for keyword in transportation_keywords)
            )
            
            # Get RAG context based on query type
            try:
                if is_transportation:
                    # Use Transportation RAG for route queries
                    logger.info("🚇 Transportation query detected - using Transportation RAG")
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
                        map_data = transport_rag.get_map_data_for_last_route()
                        if map_data:
                            logger.info(f"🗺️ Got map_data for route visualization")
                        
                        # Get enriched route data
                        if transport_rag.last_route:
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
                "entities": [e.to_dict() for e in nlp_result.entities]
            }
            
            # Add RAG context
            if rag_context:
                context["rag_context"] = rag_context
            
            # Add map_data and route_data to context for prompt builder
            if map_data:
                context["map_data"] = map_data
            if route_data:
                context["route_data"] = route_data
            
            # Add conversation history
            if request.include_context and request.session_id:
                conversation_history = history_service.get_conversation_context(
                    request.session_id,
                    max_turns=5
                )
                context["conversation_history"] = conversation_history
            
            # Send start event
            yield f"event: start\ndata: {json.dumps({'timestamp': time.time(), 'intent': intent_value})}\n\n"
            
            # Stream response
            full_response = ""
            async for chunk in streaming_service.stream_chat_response(
                message=request.message,
                context=context,
                language=request.language
            ):
                if chunk["type"] == "token":
                    full_response += chunk["content"]
                    yield f"event: token\ndata: {json.dumps({'content': chunk['content']})}\n\n"
                
                elif chunk["type"] == "complete":
                    # Save to history
                    if request.session_id:
                        history_service.add_exchange(
                            session_id=request.session_id,
                            user_message=request.message,
                            assistant_response=full_response,
                            metadata={
                                "intent": intent_value,
                                "language": request.language
                            }
                        )
                    
                    # Build metadata including map_data and route_data if available
                    metadata = {
                        'intent': intent_value,
                        'language': nlp_result.language
                    }
                    if map_data:
                        metadata['map_data'] = map_data
                    if route_data:
                        metadata['route_data'] = route_data
                    
                    # Send completion event with all metadata
                    yield f"event: complete\ndata: {json.dumps({'content': full_response, 'metadata': metadata})}\n\n"
            
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
                            # Save to history
                            history_service.add_exchange(
                                session_id=session_id,
                                user_message=content,
                                assistant_response=full_response,
                                metadata={
                                    "intent": nlp_result.intent.value,
                                    "language": language
                                }
                            )
                            
                            await websocket.send_json({
                                "type": "complete",
                                "content": full_response,
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
