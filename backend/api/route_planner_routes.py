"""
Route Planner API Endpoints
Provides intelligent itinerary planning with map visualization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
from datetime import datetime

from backend.services.route_planner import (
    get_itinerary_planner,
    Itinerary,
    ItineraryLocation
)

router = APIRouter(prefix="/api/routes", tags=["Route Planning"])


class RoutePlanRequest(BaseModel):
    """Request for route planning"""
    query: str = Field(..., description="Natural language query", example="Show me art museums and cafes in Beyoğlu for 4 hours")
    start_location: Optional[Tuple[float, float]] = Field(None, description="Starting lat/lng coordinates")
    max_duration_minutes: int = Field(240, description="Maximum duration in minutes", ge=60, le=720)
    include_meals: bool = Field(True, description="Include meal breaks")


class LocationResponse(BaseModel):
    """Location information in response"""
    id: str
    name: str
    type: str
    position: Tuple[float, float]
    duration: int
    opening_time: str
    closing_time: str
    rating: float
    description: str
    cost: str
    address: str


class SegmentResponse(BaseModel):
    """Route segment information"""
    from_name: str
    to_name: str
    distance_km: float
    duration_minutes: int
    travel_mode: str


class ItineraryResponse(BaseModel):
    """Complete itinerary response"""
    locations: List[LocationResponse]
    segments: List[SegmentResponse]
    total_distance_km: float
    total_duration_minutes: int
    start_time: str
    end_time: str
    description: str
    optimization_score: float
    warnings: List[str]
    advantages: List[str]


class MapDataResponse(BaseModel):
    """Map visualization data"""
    route_polyline: str
    markers: List[Dict]
    bounds: Dict


class RoutePlanResponse(BaseModel):
    """Complete route planning response"""
    success: bool
    itinerary: ItineraryResponse
    map_data: MapDataResponse
    message: str


@router.post("/plan", response_model=RoutePlanResponse)
async def plan_route(request: RoutePlanRequest):
    """
    Plan an optimized route based on user query
    
    Creates an intelligent itinerary combining museums, hidden gems, and dining spots.
    
    Example Request:
    ```json
    {
        "query": "Show me art museums and cafes in Beyoğlu for 4 hours",
        "start_location": [41.0351, 28.9833],
        "max_duration_minutes": 240,
        "include_meals": true
    }
    ```
    
    Returns:
    - Optimized list of locations
    - Route segments with directions
    - Map visualization data
    - Time and distance estimates
    """
    try:
        # Get itinerary planner
        planner = get_itinerary_planner()
        
        # Create itinerary
        itinerary = await planner.create_itinerary(
            user_query=request.query,
            user_location=request.start_location,
            max_duration_minutes=request.max_duration_minutes,
            include_meals=request.include_meals
        )
        
        # Format locations for response
        locations_response = [
            LocationResponse(
                id=loc.id,
                name=loc.name,
                type=loc.type,
                position=(loc.lat, loc.lng),
                duration=loc.duration,
                opening_time=loc.opening_time,
                closing_time=loc.closing_time,
                rating=loc.rating,
                description=loc.description,
                cost=loc.cost,
                address=loc.address
            )
            for loc in itinerary.locations
        ]
        
        # Format segments for response
        segments_response = [
            SegmentResponse(
                from_name=seg.from_location.name,
                to_name=seg.to_location.name,
                distance_km=seg.distance_km,
                duration_minutes=seg.duration_minutes,
                travel_mode=seg.transport_mode.value
            )
            for seg in itinerary.segments
        ]
        
        # Build itinerary response
        itinerary_response = ItineraryResponse(
            locations=locations_response,
            segments=segments_response,
            total_distance_km=itinerary.total_distance_km,
            total_duration_minutes=itinerary.total_duration_minutes,
            start_time=itinerary.start_time.isoformat(),
            end_time=itinerary.end_time.isoformat(),
            description=itinerary.description,
            optimization_score=itinerary.optimization_score,
            warnings=itinerary.warnings,
            advantages=itinerary.advantages
        )
        
        # Build map data
        markers = [
            {
                'id': loc.id,
                'position': [loc.lat, loc.lng],
                'title': loc.name,
                'type': loc.type,
                'icon': _get_marker_icon(loc.type),
                'number': i + 1,
                'rating': loc.rating,
                'cost': loc.cost
            }
            for i, loc in enumerate(itinerary.locations)
        ]
        
        bounds = _calculate_bounds(itinerary.locations) if itinerary.locations else {
            'north': 41.1, 'south': 40.9, 'east': 29.1, 'west': 28.8
        }
        
        map_data_response = MapDataResponse(
            route_polyline=itinerary.route_polyline,
            markers=markers,
            bounds=bounds
        )
        
        return RoutePlanResponse(
            success=True,
            itinerary=itinerary_response,
            map_data=map_data_response,
            message=f"Created itinerary with {len(itinerary.locations)} locations"
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to plan route: {str(e)}"
        )


@router.get("/example")
async def get_example_route():
    """
    Get an example route for testing
    
    Returns a pre-built example itinerary showcasing the route planner capabilities.
    """
    try:
        planner = get_itinerary_planner()
        
        # Create example itinerary
        itinerary = await planner.create_itinerary(
            user_query="Show me historical sites and local cafes in Sultanahmet for 3 hours",
            user_location=(41.0082, 28.9784),
            max_duration_minutes=180,
            include_meals=True
        )
        
        return {
            "success": True,
            "message": "Example itinerary created",
            "locations_count": len(itinerary.locations),
            "total_distance_km": itinerary.total_distance_km,
            "total_duration_minutes": itinerary.total_duration_minutes,
            "description": itinerary.description
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create example: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for route planner service
    """
    return {
        "status": "healthy",
        "service": "route_planner",
        "timestamp": datetime.now().isoformat()
    }


def _get_marker_icon(location_type: str) -> str:
    """Get marker icon emoji based on location type"""
    icons = {
        'museum': '🏛️',
        'hidden_gem': '💎',
        'restaurant': '🍽️',
        'cafe': '☕',
        'park': '🌳',
        'historical': '🏰',
        'shopping': '🛍️',
        'nature': '🌲',
        'culture': '🎭'
    }
    return icons.get(location_type, '📍')


def _calculate_bounds(locations: List[ItineraryLocation]) -> Dict:
    """Calculate map bounds from locations"""
    if not locations:
        return {
            'north': 41.1,
            'south': 40.9,
            'east': 29.1,
            'west': 28.8
        }
    
    lats = [loc.lat for loc in locations]
    lngs = [loc.lng for loc in locations]
    
    padding = 0.01  # Add padding to bounds
    
    return {
        'north': max(lats) + padding,
        'south': min(lats) - padding,
        'east': max(lngs) + padding,
        'west': min(lngs) - padding
    }
