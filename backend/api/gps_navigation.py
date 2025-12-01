"""
GPS Navigation API
==================

FastAPI endpoints for real-time GPS turn-by-turn navigation with map integration.

Features:
- Start/stop navigation sessions
- Real-time location updates
- Turn-by-turn instructions
- Off-route detection & rerouting
- Voice guidance support
- Map visualization data

Endpoints:
- POST /api/gps/navigation/start - Start navigation
- POST /api/gps/navigation/update - Update location
- POST /api/gps/navigation/reroute - Request reroute
- GET /api/gps/navigation/status/{session_id} - Get navigation status
- DELETE /api/gps/navigation/stop/{session_id} - Stop navigation
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GPS navigation
try:
    from services.gps_turn_by_turn_navigation import (
        GPSTurnByTurnNavigator,
        NavigationMode,
        GPSLocation,
        NavigationState,
        convert_osrm_to_steps
    )
    GPS_AVAILABLE = True
except ImportError as e:
    GPS_AVAILABLE = False
    logging.warning(f"GPS navigation not available: {e}")

# Import AI Chat Route Handler for integration
try:
    from services.ai_chat_route_integration import AIChatRouteHandler
    ROUTE_HANDLER_AVAILABLE = True
except ImportError as e:
    ROUTE_HANDLER_AVAILABLE = False
    logging.warning(f"Route handler not available: {e}")

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/gps", tags=["GPS Navigation"])

# Global navigation sessions storage
# In production, use Redis or similar
navigation_sessions: Dict[str, GPSTurnByTurnNavigator] = {}

# Global route handler instance
route_handler = AIChatRouteHandler() if ROUTE_HANDLER_AVAILABLE else None


# ========== Request/Response Models ==========

class GPSLocationModel(BaseModel):
    """GPS location data"""
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    accuracy: float = Field(default=10.0, description="Accuracy in meters")
    speed: Optional[float] = Field(None, description="Speed in m/s")
    bearing: Optional[float] = Field(None, description="Bearing in degrees (0-360)")
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")


class StartNavigationRequest(BaseModel):
    """Request to start navigation"""
    session_id: str = Field(..., description="Unique session identifier")
    route_data: Dict[str, Any] = Field(..., description="Route data from planning")
    current_location: GPSLocationModel = Field(..., description="Current GPS location")
    language: str = Field(default="en", description="Language for instructions (en, tr, etc)")
    mode: str = Field(default="walking", description="Navigation mode: walking, cycling, driving, transit")
    enable_voice: bool = Field(default=True, description="Enable voice guidance")


class UpdateLocationRequest(BaseModel):
    """Request to update GPS location"""
    session_id: str = Field(..., description="Session identifier")
    location: GPSLocationModel = Field(..., description="Current GPS location")


class RerouteRequest(BaseModel):
    """Request to reroute"""
    session_id: str = Field(..., description="Session identifier")
    current_location: GPSLocationModel = Field(..., description="Current location")
    reason: str = Field(default="user_request", description="Reason for reroute")


class NavigationResponse(BaseModel):
    """Navigation response"""
    success: bool
    session_id: str
    navigation_state: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None


# ========== Helper Functions ==========

def location_to_gps_location(location: GPSLocationModel) -> 'GPSLocation':
    """Convert location model to GPSLocation"""
    if not GPS_AVAILABLE:
        raise HTTPException(status_code=503, detail="GPS navigation not available")
    
    timestamp = None
    if location.timestamp:
        try:
            timestamp = datetime.fromisoformat(location.timestamp.replace('Z', '+00:00'))
        except:
            timestamp = datetime.now()
    
    return GPSLocation(
        latitude=location.lat,
        longitude=location.lon,
        accuracy=location.accuracy,
        speed=location.speed,
        bearing=location.bearing,
        altitude=location.altitude,
        timestamp=timestamp or datetime.now()
    )


# ========== API Endpoints ==========

@router.post("/navigation/start", response_model=NavigationResponse)
async def start_navigation(request: StartNavigationRequest):
    """
    Start GPS turn-by-turn navigation
    
    Creates a new navigation session and returns initial navigation state
    with turn-by-turn instructions and map visualization data.
    """
    if not GPS_AVAILABLE:
        raise HTTPException(status_code=503, detail="GPS navigation service not available")
    
    try:
        # Check if session already exists
        if request.session_id in navigation_sessions:
            raise HTTPException(
                status_code=400,
                detail=f"Navigation session {request.session_id} already active. Stop it first."
            )
        
        # Extract route steps from route data
        steps = []
        if 'osrm_route' in request.route_data:
            steps = convert_osrm_to_steps(request.route_data['osrm_route'])
        elif 'steps' in request.route_data:
            steps = request.route_data['steps']
        elif 'route' in request.route_data and 'steps' in request.route_data['route']:
            steps = request.route_data['route']['steps']
        
        if not steps:
            raise HTTPException(
                status_code=400,
                detail="No route steps found in route data"
            )
        
        # Determine navigation mode
        mode_map = {
            'walking': NavigationMode.WALKING,
            'cycling': NavigationMode.CYCLING,
            'driving': NavigationMode.DRIVING,
            'transit': NavigationMode.TRANSIT
        }
        nav_mode = mode_map.get(request.mode, NavigationMode.WALKING)
        
        # Create navigator
        navigator = GPSTurnByTurnNavigator(
            route_steps=steps,
            mode=nav_mode,
            language=request.language
        )
        
        # Store navigator
        navigation_sessions[request.session_id] = navigator
        
        # Create GPS location
        gps_location = location_to_gps_location(request.current_location)
        
        # Start navigation
        state = navigator.start_navigation(gps_location)
        
        logger.info(f"🧭 Started GPS navigation session: {request.session_id}")
        
        return NavigationResponse(
            success=True,
            session_id=request.session_id,
            navigation_state=state.to_dict(),
            message="Navigation started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting navigation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start navigation: {str(e)}")


@router.post("/navigation/update", response_model=NavigationResponse)
async def update_navigation(request: UpdateLocationRequest):
    """
    Update GPS location and get next navigation instruction
    
    Processes new GPS location and returns updated navigation state
    with instructions, distance to next turn, warnings, etc.
    """
    if not GPS_AVAILABLE:
        raise HTTPException(status_code=503, detail="GPS navigation service not available")
    
    try:
        # Check if session exists
        if request.session_id not in navigation_sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Navigation session {request.session_id} not found. Start navigation first."
            )
        
        navigator = navigation_sessions[request.session_id]
        
        # Create GPS location
        gps_location = location_to_gps_location(request.location)
        
        # Update navigation
        state = navigator.update_location(gps_location)
        
        # Check if arrived
        if state.has_arrived:
            logger.info(f"🎯 Navigation completed for session: {request.session_id}")
            # Clean up session
            del navigation_sessions[request.session_id]
            
            return NavigationResponse(
                success=True,
                session_id=request.session_id,
                navigation_state=state.to_dict(),
                message="You have arrived at your destination! 🎉"
            )
        
        # Check if rerouting needed
        message = None
        if state.off_route and state.warnings:
            message = "⚠️ You are off route. Rerouting recommended."
        
        return NavigationResponse(
            success=True,
            session_id=request.session_id,
            navigation_state=state.to_dict(),
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating navigation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update navigation: {str(e)}")


@router.post("/navigation/reroute", response_model=NavigationResponse)
async def reroute_navigation(request: RerouteRequest):
    """
    Request rerouting from current location
    
    Calculates a new route from current location to the original destination.
    """
    if not GPS_AVAILABLE or not ROUTE_HANDLER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Rerouting service not available")
    
    try:
        # Check if session exists
        if request.session_id not in navigation_sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Navigation session {request.session_id} not found"
            )
        
        navigator = navigation_sessions[request.session_id]
        
        # Get destination from navigator
        destination = navigator.route_steps[-1].location
        
        # Request new route using route integration
        current_coords = (request.current_location.lat, request.current_location.lon)
        dest_coords = (destination.latitude, destination.longitude)
        
        # Plan new route
        route_integration = route_handler.route_integration
        if not route_integration:
            raise HTTPException(status_code=503, detail="Route planning service not available")
        
        new_route = route_integration.plan_intelligent_route(
            start=current_coords,
            end=dest_coords,
            transport_mode=navigator.mode.value
        )
        
        # Extract new steps
        if hasattr(new_route, 'osrm_route') and new_route.osrm_route:
            new_steps = convert_osrm_to_steps(new_route.osrm_route)
        else:
            raise HTTPException(status_code=500, detail="Failed to generate new route steps")
        
        # Update navigator with new route
        navigator.route_steps = new_steps
        navigator.current_step_index = 0
        navigator.completed_steps = []
        navigator.is_navigating = True
        
        # Update with current location
        gps_location = location_to_gps_location(request.current_location)
        state = navigator.update_location(gps_location)
        
        logger.info(f"🔄 Rerouted navigation session: {request.session_id}")
        
        return NavigationResponse(
            success=True,
            session_id=request.session_id,
            navigation_state=state.to_dict(),
            message=f"🔄 Route recalculated! New route with {len(new_steps)} steps."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rerouting: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reroute: {str(e)}")


@router.get("/navigation/status/{session_id}", response_model=NavigationResponse)
async def get_navigation_status(session_id: str):
    """
    Get current navigation status
    
    Returns current navigation state without updating location.
    """
    if not GPS_AVAILABLE:
        raise HTTPException(status_code=503, detail="GPS navigation service not available")
    
    try:
        if session_id not in navigation_sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Navigation session {session_id} not found"
            )
        
        navigator = navigation_sessions[session_id]
        
        # Get route overview
        overview = navigator.get_route_overview()
        
        return NavigationResponse(
            success=True,
            session_id=session_id,
            navigation_state={
                'is_navigating': navigator.is_navigating,
                'current_step': navigator.current_step_index,
                'total_steps': len(navigator.route_steps),
                'route_overview': overview,
                'mode': navigator.mode.value,
                'language': navigator.language
            },
            message="Navigation session active"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.delete("/navigation/stop/{session_id}", response_model=NavigationResponse)
async def stop_navigation(session_id: str):
    """
    Stop navigation session
    
    Ends the navigation session and cleans up resources.
    """
    try:
        if session_id not in navigation_sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Navigation session {session_id} not found"
            )
        
        # Remove session
        del navigation_sessions[session_id]
        
        logger.info(f"🛑 Stopped GPS navigation session: {session_id}")
        
        return NavigationResponse(
            success=True,
            session_id=session_id,
            message="Navigation stopped successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping navigation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop navigation: {str(e)}")


@router.get("/navigation/sessions")
async def list_navigation_sessions():
    """
    List all active navigation sessions
    
    Returns list of active session IDs with basic info.
    """
    sessions = []
    for session_id, navigator in navigation_sessions.items():
        sessions.append({
            'session_id': session_id,
            'is_navigating': navigator.is_navigating,
            'mode': navigator.mode.value,
            'language': navigator.language,
            'current_step': navigator.current_step_index,
            'total_steps': len(navigator.route_steps)
        })
    
    return {
        'success': True,
        'count': len(sessions),
        'sessions': sessions
    }


# Health check
@router.get("/health")
async def gps_health_check():
    """GPS navigation service health check"""
    return {
        'status': 'healthy' if GPS_AVAILABLE else 'degraded',
        'gps_available': GPS_AVAILABLE,
        'route_handler_available': ROUTE_HANDLER_AVAILABLE,
        'active_sessions': len(navigation_sessions),
        'timestamp': datetime.now().isoformat()
    }
