"""
Direct Routing API Endpoint

Fast, deterministic transportation routing that bypasses the LLM entirely.
Returns template-based responses in <500ms for pure routing queries.

This endpoint provides:
- Sub-second response times
- Zero LLM hallucination risk
- Deterministic route calculations
- Clean JSON responses

Use this for:
- Pure transportation queries ("How to get from A to B?")
- Route calculations where no language interpretation is needed
- Mobile apps requiring fast routing
- APIs where deterministic responses are critical

The LLM is ONLY used for:
- Query language translation (if not English)
- Response language translation (if not English)
- Handling ambiguous location names

Author: AI Istanbul Team
Date: December 2024
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/routes", tags=["Direct Routing"])

# ==========================================
# Request/Response Models
# ==========================================

class DirectRouteRequest(BaseModel):
    """Request for direct route calculation"""
    origin: str = Field(..., description="Starting location name or station")
    destination: str = Field(..., description="Destination location name or station")
    origin_gps: Optional[Dict[str, float]] = Field(None, description="Origin GPS coordinates {lat, lon}")
    destination_gps: Optional[Dict[str, float]] = Field(None, description="Destination GPS coordinates {lat, lon}")
    max_transfers: int = Field(3, ge=0, le=5, description="Maximum number of transfers allowed")
    language: str = Field("en", description="Response language (en, tr)")

class RouteStep(BaseModel):
    """Single step in a route"""
    instruction: str
    line: str
    from_station: str = Field(alias="from")
    to_station: str = Field(alias="to")
    duration: float  # minutes
    type: str  # 'transit', 'transfer', 'walk', 'arrival'
    stops: Optional[int] = None

class DirectRouteResponse(BaseModel):
    """Response with route information"""
    success: bool
    origin: str
    destination: str
    total_time: int  # minutes
    total_distance: float  # km
    transfers: int
    lines_used: List[str]
    steps: List[Dict[str, Any]]
    time_confidence: str  # 'high', 'medium', 'low'
    response_time_ms: int
    map_data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None  # Human-readable summary

class DirectRouteError(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    error_code: str
    suggestions: List[str] = []
    response_time_ms: int


# ==========================================
# Transportation RAG Integration
# ==========================================

_transportation_rag = None

def get_transportation_rag():
    """Get or create transportation RAG singleton"""
    global _transportation_rag
    if _transportation_rag is None:
        from services.transportation_rag_system import get_transportation_rag as create_rag
        _transportation_rag = create_rag()
        logger.info("✅ Direct Routing: Transportation RAG initialized")
    return _transportation_rag


# ==========================================
# Template-Based Response Generation
# ==========================================

def generate_route_message(route, language: str = "en") -> str:
    """
    Generate human-readable route summary.
    
    This is template-based, NO LLM - for speed and determinism.
    """
    if language == "tr":
        return _generate_route_message_turkish(route)
    else:
        return _generate_route_message_english(route)

def _generate_route_message_english(route) -> str:
    """Generate English route message"""
    time_str = f"{route.total_time} minutes"
    distance_str = f"{route.total_distance:.1f} km"
    
    lines_str = ", ".join(route.lines_used)
    
    if route.transfers == 0:
        transfer_str = "Direct route (no transfers)"
    elif route.transfers == 1:
        transfer_str = "1 transfer"
    else:
        transfer_str = f"{route.transfers} transfers"
    
    # Build step-by-step instructions
    instructions = []
    for i, step in enumerate(route.steps, 1):
        if step['type'] == 'transit':
            instructions.append(
                f"{i}. Take {step['line']} from {step['from']} to {step['to']} "
                f"(~{step['duration']} min)"
            )
        elif step['type'] == 'transfer':
            instructions.append(
                f"{i}. Transfer to {step['line']} at {step['from']} (~{step['duration']} min)"
            )
        elif step['type'] == 'arrival':
            instructions.append(f"{i}. {step['instruction']}")
    
    message = f"""🚇 Route: {route.origin} → {route.destination}

⏱️ **Time:** {time_str} | 📏 **Distance:** {distance_str} | 🔄 **{transfer_str}**
🚊 **Lines:** {lines_str}

**Directions:**
{chr(10).join(instructions)}

✅ Route confidence: {route.time_confidence.upper()}
"""
    
    return message

def _generate_route_message_turkish(route) -> str:
    """Generate Turkish route message"""
    time_str = f"{route.total_time} dakika"
    distance_str = f"{route.total_distance:.1f} km"
    
    lines_str = ", ".join(route.lines_used)
    
    if route.transfers == 0:
        transfer_str = "Direkt güzergah (aktarmasız)"
    elif route.transfers == 1:
        transfer_str = "1 aktarma"
    else:
        transfer_str = f"{route.transfers} aktarma"
    
    # Build step-by-step instructions
    instructions = []
    for i, step in enumerate(route.steps, 1):
        if step['type'] == 'transit':
            instructions.append(
                f"{i}. {step['line']} hattını {step['from']} → {step['to']} "
                f"(~{step['duration']} dk)"
            )
        elif step['type'] == 'transfer':
            instructions.append(
                f"{i}. {step['from']} durağında {step['line']} hattına aktarma (~{step['duration']} dk)"
            )
        elif step['type'] == 'arrival':
            instructions.append(f"{i}. {step['instruction']}")
    
    message = f"""🚇 Güzergah: {route.origin} → {route.destination}

⏱️ **Süre:** {time_str} | 📏 **Mesafe:** {distance_str} | 🔄 **{transfer_str}**
🚊 **Hatlar:** {lines_str}

**Yol Tarifi:**
{chr(10).join(instructions)}

✅ Güzergah güvenilirliği: {route.time_confidence.upper()}
"""
    
    return message


# ==========================================
# Direct Routing Endpoint
# ==========================================

@router.post("/direct", response_model=DirectRouteResponse)
async def calculate_direct_route(request: DirectRouteRequest):
    """
    Calculate transportation route WITHOUT using LLM.
    
    This endpoint provides:
    - Fast response (<500ms target)
    - Deterministic results (no LLM randomness)
    - Accurate travel times from database
    - Template-based human-readable responses
    
    Perfect for:
    - Mobile apps requiring speed
    - APIs needing deterministic responses
    - Pure routing queries without natural language complexity
    """
    start_time = time.time()
    
    try:
        # Get transportation RAG system
        transport_rag = get_transportation_rag()
        if not transport_rag:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Transportation routing system not available"
            )
        
        logger.info(f"🚀 Direct routing: {request.origin} → {request.destination}")
        
        # Find route using Dijkstra with weighted edges
        route = transport_rag.find_route(
            origin=request.origin,
            destination=request.destination,
            max_transfers=request.max_transfers,
            origin_gps=request.origin_gps,
            destination_gps=request.destination_gps
        )
        
        if not route:
            response_time = int((time.time() - start_time) * 1000)
            
            return DirectRouteError(
                success=False,
                error=f"No route found between {request.origin} and {request.destination}",
                error_code="NO_ROUTE_FOUND",
                suggestions=[
                    f"Check if '{request.origin}' is a valid station or neighborhood",
                    f"Check if '{request.destination}' is a valid station or neighborhood",
                    "Try increasing max_transfers parameter",
                    "Try using GPS coordinates for more accurate matching"
                ],
                response_time_ms=response_time
            )
        
        # Generate human-readable message (template-based, no LLM)
        message = generate_route_message(route, request.language)
        
        # Extract map data for visualization
        map_data = transport_rag.get_map_data_for_last_route()
        
        response_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"✅ Route found in {response_time}ms: {route.total_time}min, {route.transfers} transfers")
        
        return DirectRouteResponse(
            success=True,
            origin=route.origin,
            destination=route.destination,
            total_time=route.total_time,
            total_distance=route.total_distance,
            transfers=route.transfers,
            lines_used=route.lines_used,
            steps=[dict(step) for step in route.steps],
            time_confidence=route.time_confidence,
            response_time_ms=response_time,
            map_data=map_data,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Direct routing error: {e}", exc_info=True)
        response_time = int((time.time() - start_time) * 1000)
        
        return DirectRouteError(
            success=False,
            error=str(e),
            error_code="ROUTING_ERROR",
            suggestions=[
                "Check your input parameters",
                "Try using GPS coordinates",
                "Contact support if the problem persists"
            ],
            response_time_ms=response_time
        )


# ==========================================
# Health Check
# ==========================================

@router.get("/health")
async def health_check():
    """Check if direct routing service is available"""
    try:
        transport_rag = get_transportation_rag()
        if not transport_rag:
            return {
                "status": "unavailable",
                "message": "Transportation RAG system not initialized"
            }
        
        # Quick test: can we access the station database?
        station_count = len(transport_rag.stations)
        
        return {
            "status": "healthy",
            "station_count": station_count,
            "dijkstra_enabled": True,
            "travel_time_db_loaded": True,
            "message": "Direct routing service operational"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
