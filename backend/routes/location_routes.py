"""
API Routes for Live Location-Based Routing and POI Recommendations
Integrates algorithmic location services with the existing FastAPI backend

Features:
- Real-time location tracking and session management
- Multi-stop route planning with TSP optimization
- Smart POI recommendations with advanced filtering
- Dynamic route updates based on user movement
- District-to-district navigation
- Offline mode support
- Privacy-safe location handling

Endpoints:
- POST /api/location/start-session - Start live location session
- POST /api/location/update - Update user location  
- POST /api/location/recommendations - Get filtered POI recommendations
- POST /api/location/plan-route - Plan dynamic route with multiple stops
- POST /api/location/multi-stop-route - Advanced multi-stop TSP optimization
- POST /api/location/update-route - Update existing route with new conditions
- POST /api/location/nearby-search - Search POIs near current location
- POST /api/location/districts - Get district navigation info
- POST /api/location/offline - Get offline mode data
- GET /api/location/health - Health check for location services
- GET /api/location/available-pois - List all available POIs
- POST /api/location/cleanup - Clean up inactive sessions
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime
from pydantic import BaseModel, Field

try:
    from services.live_location_integration_service import (
        LiveLocationIntegrationService,
    )
    LOCATION_SERVICE_AVAILABLE = True
except ImportError:
    LOCATION_SERVICE_AVAILABLE = False
    # Create dummy service for fallback
    class LiveLocationIntegrationService:
        def __init__(self): pass
        def start_location_session(self, data): return {"error": "Service not available"}
        def update_location(self, data): return {"error": "Service not available"}
        def get_recommendations(self, data): return {"error": "Service not available"}
        def plan_route(self, data): return {"error": "Service not available"}
        def cleanup_inactive_sessions(self, hours): return 0
        active_sessions = {}

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI Router
router = APIRouter(prefix="/api/location", tags=["Live Location & Routing"])

# Global service instance (will be initialized once)
location_service = None

def get_location_service() -> LiveLocationIntegrationService:
    """Get or create location service instance"""
    global location_service
    if location_service is None:
        location_service = LiveLocationIntegrationService()
        logger.info("Live Location Service initialized")
    return location_service

# Pydantic models for request validation
class LocationSessionRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    accuracy: Optional[float] = Field(None, ge=0, description="Location accuracy in meters")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")

class LocationUpdateRequest(BaseModel):
    session_id: str = Field(..., description="Active session identifier")
    latitude: float = Field(..., ge=-90, le=90, description="Updated latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Updated longitude")
    accuracy: Optional[float] = Field(None, ge=0, description="Location accuracy in meters")

class RecommendationsRequest(BaseModel):
    session_id: str = Field(..., description="Active session identifier")
    categories: Optional[List[str]] = Field(None, description="POI categories to filter")
    cuisine_types: Optional[List[str]] = Field(None, description="Cuisine types for restaurants")
    price_ranges: Optional[List[str]] = Field(None, description="Price range filters")
    open_now: Optional[bool] = Field(None, description="Filter for currently open POIs")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating filter")
    accessibility_required: Optional[List[str]] = Field(None, description="Required accessibility features")
    dietary_requirements: Optional[List[str]] = Field(None, description="Dietary requirement filters")
    max_distance_km: Optional[float] = Field(None, ge=0, description="Maximum distance in kilometers")
    limit: Optional[int] = Field(10, ge=1, le=50, description="Maximum number of results")

class RouteRequest(BaseModel):
    session_id: str = Field(..., description="Active session identifier")
    target_poi_ids: List[str] = Field(..., min_items=1, description="List of POI IDs to visit")
    algorithm: Optional[str] = Field("nearest", description="Routing algorithm to use")
    transport_mode: Optional[str] = Field("mixed", description="Transportation mode preference")
    optimize_for: Optional[str] = Field("time", description="Optimization criteria")

class MultiStopRouteRequest(BaseModel):
    session_id: str = Field(..., description="Active session identifier")
    target_poi_ids: List[str] = Field(..., min_items=2, description="List of POI IDs for multi-stop route")
    algorithm: Optional[str] = Field("tsp_nearest", description="TSP algorithm for optimization")
    transport_mode: Optional[str] = Field("mixed", description="Transportation mode")
    time_constraints: Optional[Dict[str, int]] = Field(None, description="Time constraints per POI")
    start_time: Optional[str] = Field(None, description="Preferred start time (HH:MM)")
    max_total_time_hours: Optional[float] = Field(None, description="Maximum total route time")

class RouteUpdateRequest(BaseModel):
    session_id: str = Field(..., description="Active session identifier")
    route_id: str = Field(..., description="Route identifier to update")
    current_poi_id: Optional[str] = Field(None, description="Current POI if already started route")
    skip_poi_ids: Optional[List[str]] = Field(None, description="POIs to skip in route")
    add_poi_ids: Optional[List[str]] = Field(None, description="POIs to add to route")
    time_delay_minutes: Optional[int] = Field(None, description="Time delay to account for")

class NearbySearchRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID or provide coordinates")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Search center latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Search center longitude")
    radius_km: Optional[float] = Field(1.0, ge=0.1, le=10, description="Search radius in kilometers")
    categories: Optional[List[str]] = Field(None, description="POI categories to search")
    keywords: Optional[List[str]] = Field(None, description="Keywords to search in POI names/descriptions")
    limit: Optional[int] = Field(10, ge=1, le=50, description="Maximum results")

class DistrictNavigationRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID if available")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Starting latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Starting longitude")
    target_districts: Optional[List[str]] = Field(None, description="Target districts")

class OfflineModeRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Center latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Center longitude")
    radius_km: Optional[float] = Field(2.0, ge=0.1, le=20, description="Data radius in kilometers")
    include_categories: Optional[List[str]] = Field(None, description="Categories to include")

class CleanupRequest(BaseModel):
    max_inactive_hours: Optional[int] = Field(24, ge=1, description="Hours before session cleanup")

# Health Check Endpoint
@router.get("/health")
async def health_check():
    """Health check for location services"""
    try:
        service = get_location_service()
        active_sessions = len(service.active_sessions)
        
        if hasattr(service, 'routing_system') and hasattr(service.routing_system, 'route_cache'):
            cached_routes = len(service.routing_system.route_cache)
        else:
            cached_routes = 0
        
        return {
            "status": "healthy",
            "service": "Live Location Routing System",
            "service_available": LOCATION_SERVICE_AVAILABLE,
            "active_sessions": active_sessions,
            "cached_routes": cached_routes,
            "features": [
                "real_time_poi_recommendations",
                "dynamic_route_planning", 
                "multi_stop_tsp_optimization",
                "live_location_updates",
                "offline_mode_support",
                "smart_filtering",
                "district_estimates"
            ],
            "algorithms_available": ["nearest", "dijkstra", "a_star", "tsp_nearest", "tsp_greedy"],
            "poi_categories": ["restaurant", "museum", "landmark", "shopping", "entertainment", "transport", "religious", "viewpoint"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Start Location Session
@router.post("/start-session")
async def start_location_session(request: LocationSessionRequest):
    """Start a new live location session with privacy-safe handling"""
    try:
        if not LOCATION_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Location service not available"
            )
        
        service = get_location_service()
        
        # Convert request to dict for service
        request_data = {
            "user_id": request.user_id,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "accuracy": request.accuracy,
            "preferences": request.preferences or {}
        }
        
        result = service.start_location_session(request_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting location session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to start location session",
                "details": str(e)
            }
        )

# Update Location
@router.post("/update")
async def update_location(request: LocationUpdateRequest):
    """Update user's current location for dynamic recommendations"""
    try:
        if not LOCATION_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Location service not available"
            )
        
        service = get_location_service()
        
        request_data = {
            "session_id": request.session_id,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "accuracy": request.accuracy
        }
        
        result = service.update_location(request_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating location: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to update location",
                "details": str(e)
            }
        )

# Get POI Recommendations
@router.post("/recommendations")
async def get_filtered_recommendations(request: RecommendationsRequest):
    """Get POI recommendations with advanced filtering"""
    try:
        if not LOCATION_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Location service not available"
            )
        
        service = get_location_service()
        
        request_data = {
            "session_id": request.session_id,
            "categories": request.categories,
            "cuisine_types": request.cuisine_types,
            "price_ranges": request.price_ranges,
            "open_now": request.open_now,
            "min_rating": request.min_rating,
            "accessibility_required": request.accessibility_required,
            "dietary_requirements": request.dietary_requirements,
            "max_distance_km": request.max_distance_km,
            "limit": request.limit
        }
        
        result = service.get_recommendations(request_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get recommendations",
                "details": str(e)
            }
        )

# Plan Dynamic Route
@router.post("/plan-route")
async def plan_dynamic_route(request: RouteRequest):
    """Plan a dynamic route with real-time updates"""
    try:
        if not LOCATION_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Location service not available"
            )
        
        service = get_location_service()
        
        request_data = {
            "session_id": request.session_id,
            "target_poi_ids": request.target_poi_ids,
            "algorithm": request.algorithm,
            "transport_mode": request.transport_mode,
            "optimize_for": request.optimize_for
        }
        
        result = service.plan_route(request_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error planning route: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to plan route",
                "details": str(e)
            }
        )

# Multi-Stop Route Planning with TSP
@router.post("/multi-stop-route")
async def plan_multi_stop_route(request: MultiStopRouteRequest):
    """Plan optimized multi-stop route using TSP algorithms"""
    try:
        if not LOCATION_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Location service not available"
            )
        
        service = get_location_service()
        
        request_data = {
            "session_id": request.session_id,
            "target_poi_ids": request.target_poi_ids,
            "algorithm": request.algorithm,
            "transport_mode": request.transport_mode,
            "time_constraints": request.time_constraints,
            "start_time": request.start_time,
            "max_total_time_hours": request.max_total_time_hours,
            "operation": "multi_stop_route"
        }
        
        # Use the plan_route method with multi-stop specific parameters
        result = service.plan_route(request_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error planning multi-stop route: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to plan multi-stop route",
                "details": str(e)
            }
        )

# Update Existing Route
@router.post("/update-route")
async def update_existing_route(request: RouteUpdateRequest):
    """Update an existing route with new conditions or changes"""
    try:
        if not LOCATION_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Location service not available"
            )
        
        service = get_location_service()
        
        request_data = {
            "session_id": request.session_id,
            "route_id": request.route_id,
            "current_poi_id": request.current_poi_id,
            "skip_poi_ids": request.skip_poi_ids,
            "add_poi_ids": request.add_poi_ids,
            "time_delay_minutes": request.time_delay_minutes,
            "operation": "update_route"
        }
        
        # Use specialized route update method if available
        if hasattr(service, 'update_route'):
            result = service.update_route(request_data)
        else:
            # Fallback to plan_route with update parameters
            result = service.plan_route(request_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating route: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to update route",
                "details": str(e)
            }
        )

# Nearby POI Search
@router.post("/nearby-search")
async def search_nearby_pois(request: NearbySearchRequest):
    """Search for POIs near current location or specified coordinates"""
    try:
        if not LOCATION_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Location service not available"
            )
        
        service = get_location_service()
        
        request_data = {
            "session_id": request.session_id,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "radius_km": request.radius_km,
            "categories": request.categories,
            "keywords": request.keywords,
            "limit": request.limit,
            "operation": "nearby_search"
        }
        
        # Use specialized nearby search if available
        if hasattr(service, 'search_nearby'):
            result = service.search_nearby(request_data)
        else:
            # Fallback to recommendations with location override
            result = service.get_recommendations(request_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching nearby POIs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to search nearby POIs",
                "details": str(e)
            }
        )

# District Navigation
@router.post("/districts")
async def get_district_navigation(request: DistrictNavigationRequest):
    """Get navigation info to Istanbul districts"""
    try:
        if not LOCATION_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Location service not available"
            )
        
        service = get_location_service()
        
        request_data = {
            "session_id": request.session_id,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "target_districts": request.target_districts,
            "operation": "district_navigation"
        }
        
        # Use specialized district navigation if available
        if hasattr(service, 'get_district_navigation'):
            result = service.get_district_navigation(request_data)
        else:
            # Fallback logic
            result = {
                "message": "District navigation available",
                "available_districts": ["sultanahmet", "beyoglu", "galata", "kadikoy", "besiktas"],
                "note": "Detailed navigation service under development"
            }
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting district navigation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get district navigation",
                "details": str(e)
            }
        )

# Offline Mode Data
@router.post("/offline")
async def get_offline_mode_data(request: OfflineModeRequest):
    """Get data for offline/minimal AI operation"""
    try:
        if not LOCATION_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Location service not available"
            )
        
        service = get_location_service()
        
        request_data = {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "radius_km": request.radius_km,
            "include_categories": request.include_categories,
            "operation": "offline_mode"
        }
        
        # Use specialized offline mode if available
        if hasattr(service, 'get_offline_data'):
            result = service.get_offline_data(request_data)
        else:
            # Fallback to nearby search for offline data
            result = {
                "offline_data": {
                    "center": {"latitude": request.latitude, "longitude": request.longitude},
                    "radius_km": request.radius_km,
                    "pois": [],
                    "districts": [],
                    "transport_info": {},
                    "generated_at": datetime.now().isoformat()
                },
                "note": "Offline mode service under development"
            }
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting offline mode data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get offline mode data",
                "details": str(e)
            }
        )

# Session Cleanup
@router.post("/cleanup")
async def cleanup_sessions(request: CleanupRequest):
    """Clean up inactive sessions (admin endpoint)"""
    try:
        service = get_location_service()
        
        if hasattr(service, 'cleanup_inactive_sessions'):
            cleaned_count = service.cleanup_inactive_sessions(request.max_inactive_hours)
        else:
            # Fallback cleanup logic
            cleaned_count = 0
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, session_data in service.active_sessions.items():
                if isinstance(session_data, dict) and 'created_at' in session_data:
                    session_time = session_data['created_at']
                    if isinstance(session_time, str):
                        session_time = datetime.fromisoformat(session_time.replace('Z', '+00:00'))
                    
                    hours_inactive = (current_time - session_time).total_seconds() / 3600
                    if hours_inactive > request.max_inactive_hours:
                        sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del service.active_sessions[session_id]
                cleaned_count += 1
        
        return {
            "success": True,
            "cleaned_sessions": cleaned_count,
            "max_inactive_hours": request.max_inactive_hours,
            "active_sessions_remaining": len(service.active_sessions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to clean up sessions",
                "details": str(e)
            }
        )

# Available POIs
@router.get("/available-pois")
async def get_available_pois():
    """Get list of available POIs in the system (for reference)"""
    try:
        service = get_location_service()
        
        if not hasattr(service, 'routing_system') or not hasattr(service.routing_system, 'location_data'):
            return {
                "available_pois": [],
                "available_districts": [],
                "total_pois": 0,
                "total_districts": 0,
                "categories": ["restaurant", "museum", "landmark", "shopping", "entertainment", "transport", "religious", "viewpoint"],
                "filter_options": {
                    "cuisine_types": ["turkish", "ottoman", "mediterranean"],
                    "price_ranges": ["budget-friendly", "mid-range", "luxury"],
                    "accessibility_features": ["wheelchair_accessible", "elevator_available", "audio_guide"],
                    "dietary_options": ["vegetarian", "vegan", "halal", "gluten_free"]
                },
                "note": "POI data service under development"
            }
        
        pois_info = []
        for poi_id, poi in service.routing_system.location_data.pois.items():
            pois_info.append({
                "id": poi.id,
                "name": poi.name,
                "category": poi.category.value if hasattr(poi.category, 'value') else str(poi.category),
                "rating": getattr(poi, 'rating', None),
                "price_range": getattr(poi, 'price_range', None),
                "cuisine_type": getattr(poi, 'cuisine_type', None),
                "coordinates": {
                    "latitude": poi.coordinates.latitude,
                    "longitude": poi.coordinates.longitude
                },
                "features": getattr(poi, 'features', []),
                "accessibility_features": getattr(poi, 'accessibility_features', []),
                "dietary_options": getattr(poi, 'dietary_options', [])
            })
        
        districts_info = [
            {
                "name": district_name,
                "coordinates": {
                    "latitude": coord.latitude,
                    "longitude": coord.longitude
                }
            }
            for district_name, coord in service.routing_system.location_data.districts.items()
        ]
        
        return {
            "available_pois": pois_info,
            "available_districts": districts_info,
            "total_pois": len(pois_info),
            "total_districts": len(districts_info),
            "categories": ["restaurant", "museum", "landmark", "shopping", "entertainment", "transport", "religious", "viewpoint"],
            "filter_options": {
                "cuisine_types": ["turkish", "ottoman", "mediterranean"],
                "price_ranges": ["budget-friendly", "mid-range", "luxury"],
                "accessibility_features": ["wheelchair_accessible", "elevator_available", "audio_guide"],
                "dietary_options": ["vegetarian", "vegan", "halal", "gluten_free"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting available POIs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get available POIs",
                "details": str(e)
            }
        )

# FastAPI automatically handles 404 and 405 errors with proper OpenAPI documentation
