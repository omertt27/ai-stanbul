#!/usr/bin/env python3
"""
AI Istanbul - POI-Enhanced Route Planning API
=============================================

Production-ready FastAPI server for Istanbul tourism route optimization.

Features:
- RESTful API with OpenAPI/Swagger docs
- Authentication & rate limiting
- Request validation & error handling
- Monitoring & logging
- CORS support
- Health checks
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import time
import asyncio
from functools import wraps
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_gps_route_planner import get_enhanced_gps_planner, GPSLocation
from services.route_cache_service import get_cache_service
from services.multi_day_itinerary_service import get_multi_day_service, TripPace
from services.crowding_intelligence_service import get_crowding_intelligence_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Istanbul Route Planner API",
    description="Intelligent POI-enhanced route planning for Istanbul tourism",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Rate limiting (simple in-memory implementation)
request_counts = {}
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds


# ============================================================================
# Request/Response Models
# ============================================================================

class LocationModel(BaseModel):
    """GPS location coordinates"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    district: Optional[str] = Field(None, description="Istanbul district name")
    
    @validator('latitude')
    def validate_istanbul_lat(cls, v):
        if not (40.8 <= v <= 41.3):
            logger.warning(f"Location outside Istanbul: {v}")
        return v
    
    @validator('longitude')
    def validate_istanbul_lon(cls, v):
        if not (28.5 <= v <= 29.5):
            logger.warning(f"Location outside Istanbul: {v}")
        return v


class PreferencesModel(BaseModel):
    """User preferences for route planning"""
    interests: List[str] = Field(
        default=["museum", "history"],
        description="Categories of interest (museum, palace, viewpoint, etc.)"
    )
    transport_modes: List[str] = Field(
        default=["walk", "metro"],
        description="Preferred transport modes"
    )
    prefer_diverse_categories: bool = Field(
        default=True,
        description="Prefer POIs from different categories"
    )


class ConstraintsModel(BaseModel):
    """Route constraints"""
    max_pois: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of POIs to include"
    )
    max_detour_minutes: int = Field(
        default=45,
        ge=0,
        le=180,
        description="Maximum detour time per POI (minutes)"
    )
    max_total_detour: int = Field(
        default=120,
        ge=0,
        le=360,
        description="Maximum total detour time (minutes)"
    )
    min_poi_value: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum POI value score (0-1)"
    )


class RouteRequest(BaseModel):
    """Route planning request"""
    user_id: str = Field(..., description="Unique user identifier")
    start_location: LocationModel = Field(..., description="Starting location")
    end_location: LocationModel = Field(..., description="Destination location")
    preferences: Optional[PreferencesModel] = Field(
        default=PreferencesModel(),
        description="User preferences"
    )
    constraints: Optional[ConstraintsModel] = Field(
        default=ConstraintsModel(),
        description="Route constraints"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "start_location": {
                    "latitude": 41.0082,
                    "longitude": 28.9784,
                    "district": "sultanahmet"
                },
                "end_location": {
                    "latitude": 41.0369,
                    "longitude": 28.9857,
                    "district": "beyoglu"
                },
                "preferences": {
                    "interests": ["museum", "history", "palace"],
                    "transport_modes": ["walk", "metro"],
                    "prefer_diverse_categories": True
                },
                "constraints": {
                    "max_pois": 3,
                    "max_detour_minutes": 45,
                    "max_total_detour": 120
                }
            }
        }


class RouteResponse(BaseModel):
    """Route planning response"""
    route_id: str
    user_id: str
    base_route: Dict[str, Any]
    enhanced_route: Dict[str, Any]
    pois_included: List[Dict[str, Any]]
    pois_recommended_not_included: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]
    local_tips_by_district: Dict[str, List[str]]
    recommendation_summary: Dict[str, Any]
    created_at: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    services: Dict[str, bool]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None


# ============================================================================
# Phase 6: Multi-Day Itinerary Models
# ============================================================================

class MultiDayRequest(BaseModel):
    """Multi-day itinerary planning request"""
    user_id: str = Field(..., description="Unique user identifier")
    num_days: int = Field(..., ge=2, le=7, description="Number of days (2-7)")
    accommodation_location: LocationModel = Field(..., description="Hotel/accommodation location")
    start_date: str = Field(..., description="Trip start date (YYYY-MM-DD)")
    preferences: Optional[PreferencesModel] = Field(
        default=PreferencesModel(),
        description="User preferences"
    )
    pace: str = Field(
        default="moderate",
        description="Trip pace: relaxed, moderate, or intensive"
    )
    budget_usd: float = Field(
        default=500.0,
        ge=0,
        description="Total trip budget in USD"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "tourist123",
                "num_days": 3,
                "accommodation_location": {
                    "latitude": 41.0082,
                    "longitude": 28.9784,
                    "district": "sultanahmet"
                },
                "start_date": "2025-06-15",
                "preferences": {
                    "interests": ["museum", "palace", "mosque", "viewpoint"],
                    "transport_modes": ["walk", "metro", "tram", "ferry"]
                },
                "pace": "moderate",
                "budget_usd": 500.0
            }
        }


class CrowdAnalysisRequest(BaseModel):
    """Crowd analysis request for POIs"""
    pois: List[Dict[str, Any]] = Field(..., description="List of POIs to analyze")
    visit_time: str = Field(..., description="Planned visit time (ISO 8601)")
    
    class Config:
        schema_extra = {
            "example": {
                "pois": [
                    {
                        "id": "hagia_sophia",
                        "name": "Hagia Sophia",
                        "category": "museum",
                        "visit_duration_minutes": 60
                    },
                    {
                        "id": "blue_mosque",
                        "name": "Sultan Ahmed Mosque",
                        "category": "mosque",
                        "visit_duration_minutes": 30
                    }
                ],
                "visit_time": "2025-07-15T11:00:00"
            }
        }


# ============================================================================
# Middleware & Dependencies
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """Simple rate limiting middleware"""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    request_counts[client_ip] = [
        t for t in request_counts.get(client_ip, [])
        if current_time - t < RATE_WINDOW
    ]
    
    # Check rate limit
    if len(request_counts.get(client_ip, [])) >= RATE_LIMIT:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per minute.",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # Add request timestamp
    request_counts.setdefault(client_ip, []).append(current_time)
    
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT)
    response.headers["X-RateLimit-Remaining"] = str(RATE_LIMIT - len(request_counts.get(client_ip, [])))
    return response


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify API token (simplified for demo)
    In production: validate against database, JWT, etc.
    """
    token = credentials.credentials
    # For demo: accept any token starting with "ai-istanbul-"
    if not token.startswith("ai-istanbul-"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "name": "AI Istanbul Route Planner API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/api/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check API health and service availability
    
    Returns system status, service availability, and uptime.
    """
    planner = get_enhanced_gps_planner()
    cache = get_cache_service()
    
    services_status = {
        "poi_database": planner.poi_db_service is not None,
        "ml_predictions": planner.ml_prediction_service is not None,
        "poi_optimizer": planner.poi_optimizer is not None,
        "cache_service": cache is not None
    }
    
    all_healthy = all(services_status.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        services=services_status,
        uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    )


@app.get(
    "/api/v1/cache/stats",
    tags=["Monitoring"],
    summary="Get cache statistics"
)
async def get_cache_stats(token: str = Depends(verify_token)):
    """
    Get cache performance statistics
    
    Requires authentication.
    """
    cache = get_cache_service()
    stats = cache.get_cache_stats()
    return {
        "timestamp": datetime.now().isoformat(),
        "cache_stats": stats
    }


@app.post(
    "/api/v1/cache/clear",
    tags=["Monitoring"],
    summary="Clear all caches"
)
async def clear_caches(token: str = Depends(verify_token)):
    """
    Clear all caches (requires authentication)
    
    Use with caution - will impact performance temporarily.
    """
    cache = get_cache_service()
    cache.clear_all_caches()
    logger.info("All caches cleared by admin")
    return {
        "status": "success",
        "message": "All caches cleared",
        "timestamp": datetime.now().isoformat()
    }


@app.post(
    "/api/v1/route/optimize",
    response_model=RouteResponse,
    tags=["Route Planning"],
    summary="Create optimized POI-enhanced route",
    responses={
        200: {"description": "Route successfully created"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def create_optimized_route(
    request: RouteRequest,
    api_request: Request
):
    """
    Create an optimized route with POI recommendations
    
    This endpoint creates an intelligent route between two locations,
    including relevant points of interest based on user preferences.
    
    **Features:**
    - Multi-objective optimization (time, distance, POI value)
    - ML-powered crowding predictions
    - Smart detour calculation
    - Category diversity enforcement
    - Real-time constraint validation
    
    **Process:**
    1. Validate locations and constraints
    2. Query nearby POIs based on interests
    3. Score POIs using ML predictions
    4. Optimize route with selected POIs
    5. Generate turn-by-turn segments
    6. Include local tips and recommendations
    """
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"[{request_id}] Route request from {request.user_id}: {request.start_location.district} → {request.end_location.district}")
        
        # Get planner
        planner = get_enhanced_gps_planner()
        
        # Convert request models to internal format
        start_location = GPSLocation(
            latitude=request.start_location.latitude,
            longitude=request.start_location.longitude,
            district=request.start_location.district
        )
        
        end_location = GPSLocation(
            latitude=request.end_location.latitude,
            longitude=request.end_location.longitude,
            district=request.end_location.district
        )
        
        preferences = request.preferences.dict() if request.preferences else {}
        constraints = request.constraints.dict() if request.constraints else {}
        
        # Create optimized route
        route = await planner.create_poi_optimized_route(
            user_id=request.user_id,
            start_location=start_location,
            end_location=end_location,
            preferences=preferences,
            constraints=constraints
        )
        
        # Add processing time
        processing_time_ms = (time.time() - start_time) * 1000
        route['processing_time_ms'] = processing_time_ms
        
        logger.info(f"[{request_id}] Route created successfully in {processing_time_ms:.0f}ms with {len(route.get('pois_included', []))} POIs")
        
        return RouteResponse(**route)
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"[{request_id}] Error creating route: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while creating route"
        )


@app.get(
    "/api/v1/pois/nearby",
    tags=["POI Discovery"],
    summary="Find nearby POIs"
)
async def get_nearby_pois(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(default=2.0, ge=0.1, le=20.0),
    category: Optional[str] = Query(None),
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Find points of interest near a location
    
    Returns POIs within specified radius, optionally filtered by category.
    """
    try:
        planner = get_enhanced_gps_planner()
        
        # Use POI database to find nearby POIs
        poi_db = planner.poi_db_service
        if not poi_db:
            raise HTTPException(status_code=503, detail="POI service unavailable")
        
        # Get all POIs and filter by distance
        all_pois = poi_db.get_all_pois()
        nearby = []
        
        for poi in all_pois:
            # Calculate distance
            dist = _calculate_distance(
                latitude, longitude,
                poi.location.lat, poi.location.lon
            )
            
            if dist <= radius_km:
                if category is None or poi.category.lower() == category.lower():
                    nearby.append({
                        'id': poi.poi_id,
                        'name': poi.name,
                        'category': poi.category,
                        'rating': poi.rating,
                        'distance_km': round(dist, 2),
                        'coordinates': {
                            'latitude': poi.location.lat,
                            'longitude': poi.location.lon
                        },
                        'district': getattr(poi, 'district', '')
                    })
        
        # Sort by distance and limit
        nearby.sort(key=lambda x: x['distance_km'])
        nearby = nearby[:limit]
        
        return {
            "center": {"latitude": latitude, "longitude": longitude},
            "radius_km": radius_km,
            "count": len(nearby),
            "pois": nearby
        }
        
    except Exception as e:
        logger.error(f"Error finding nearby POIs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error finding nearby POIs")


def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points"""
    import math
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


# ============================================================================
# Phase 6: Multi-Day Itinerary Endpoints
# ============================================================================

@app.post(
    "/api/v1/itinerary/multi-day",
    tags=["Phase 6: Multi-Day Planning"],
    summary="Create multi-day Istanbul itinerary",
    description="""
    Plan a complete multi-day trip to Istanbul (2-7 days).
    
    **Features:**
    - Daily route optimization with POI recommendations
    - Budget tracking and management
    - Energy/fatigue modeling
    - Category diversity across days
    - Accommodation-centered planning
    - Morning, afternoon, and evening activities
    
    **Trip Paces:**
    - **Relaxed**: 2-3 POIs/day, longer visits, more rest
    - **Moderate**: 4-5 POIs/day, balanced schedule
    - **Intensive**: 6+ POIs/day, quick visits, packed schedule
    """
)
async def create_multi_day_itinerary(request: MultiDayRequest):
    """Create a complete multi-day itinerary for Istanbul"""
    start_time = time.time()
    request_id = f"itinerary_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"[{request_id}] Multi-day itinerary request: {request.num_days} days, {request.pace} pace")
        
        # Get multi-day service
        service = get_multi_day_service()
        
        # Parse start date
        from datetime import datetime as dt
        start_date = dt.fromisoformat(request.start_date)
        
        # Convert accommodation location
        accommodation = GPSLocation(
            latitude=request.accommodation_location.latitude,
            longitude=request.accommodation_location.longitude,
            district=request.accommodation_location.district
        )
        
        # Prepare preferences
        preferences = request.preferences.dict() if request.preferences else {}
        preferences['pace'] = request.pace
        
        # Create itinerary
        itinerary = await service.create_multi_day_itinerary(
            user_id=request.user_id,
            num_days=request.num_days,
            accommodation_location=accommodation,
            start_date=start_date,
            preferences=preferences,
            budget_usd=request.budget_usd
        )
        
        # Format response
        response = {
            'trip_id': itinerary.trip_id,
            'user_id': itinerary.user_id,
            'num_days': itinerary.num_days,
            'start_date': itinerary.start_date.isoformat(),
            'accommodation': {
                'latitude': accommodation.latitude,
                'longitude': accommodation.longitude,
                'district': accommodation.district
            },
            'pace': itinerary.pace.value,
            'budget': {
                'total_usd': itinerary.budget_usd,
                'spent_usd': itinerary.total_cost_usd,
                'remaining_usd': itinerary.budget_remaining
            },
            'summary': {
                'total_pois': itinerary.total_pois,
                'total_distance_km': round(itinerary.total_distance_km, 2),
                'total_cost_usd': round(itinerary.total_cost_usd, 2),
                'pois_per_day': round(itinerary.total_pois / itinerary.num_days, 1)
            },
            'daily_plans': []
        }
        
        # Add daily plans
        for day in itinerary.daily_plans:
            day_data = {
                'day_number': day.day_number,
                'date': day.date.isoformat(),
                'stats': {
                    'total_pois': day.total_pois,
                    'distance_km': round(day.total_distance_km, 2),
                    'time_minutes': day.total_time_minutes,
                    'cost_usd': round(day.total_cost_usd, 2),
                    'energy_level': round(day.energy_level, 2)
                },
                'morning_route': day.morning_route,
                'afternoon_route': day.afternoon_route,
                'evening_activities': day.evening_activities,
                'visited_categories': list(day.visited_categories),
                'visited_districts': list(day.visited_districts)
            }
            response['daily_plans'].append(day_data)
        
        # Add text summary
        response['text_summary'] = service.format_itinerary_summary(itinerary)
        
        processing_time_ms = (time.time() - start_time) * 1000
        response['processing_time_ms'] = processing_time_ms
        
        logger.info(f"[{request_id}] Itinerary created in {processing_time_ms:.0f}ms")
        
        return response
        
    except ValueError as e:
        logger.warning(f"[{request_id}] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[{request_id}] Error creating itinerary: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error creating multi-day itinerary")


@app.post(
    "/api/v1/crowding/analyze",
    tags=["Phase 6: Crowd Intelligence"],
    summary="Analyze crowd levels for POIs",
    description="""
    Get real-time crowd predictions for your route.
    
    **Features:**
    - ML-based crowd predictions by time/day/season
    - Wait time estimates
    - Optimal visit time recommendations
    - Alternative time suggestions
    - Route-level crowding analysis
    
    **Crowd Levels:**
    - ⚪ Empty (0-20% capacity)
    - 🟢 Light (20-40%)
    - 🟡 Moderate (40-60%)
    - 🟠 Busy (60-80%)
    - 🔴 Crowded (80-100%)
    - ⛔ Overcrowded (100%+)
    """
)
async def analyze_crowding(request: CrowdAnalysisRequest):
    """Analyze crowding for a list of POIs at specified time"""
    start_time = time.time()
    request_id = f"crowd_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"[{request_id}] Crowd analysis for {len(request.pois)} POIs")
        
        # Get crowding service
        service = get_crowding_intelligence_service()
        
        # Parse visit time
        from datetime import datetime as dt
        visit_time = dt.fromisoformat(request.visit_time)
        
        # Analyze route crowding
        insights = await service.analyze_route_crowding(
            pois=request.pois,
            start_time=visit_time
        )
        
        # Format response
        response = {
            'analysis_time': datetime.now().isoformat(),
            'visit_time': visit_time.isoformat(),
            'num_pois': len(request.pois),
            'overall_crowd_score': round(insights.overall_crowd_score, 3),
            'crowd_level': (
                'optimal' if insights.overall_crowd_score < 0.4 else
                'moderate' if insights.overall_crowd_score < 0.7 else
                'very_crowded'
            ),
            'recommendations': {
                'suggested_adjustment': insights.suggested_route_adjustment,
                'time_shift_minutes': insights.time_shift_recommendation,
                'recommended_start_time': insights.recommended_start_time.isoformat() if insights.recommended_start_time else None
            },
            'crowded_pois': insights.crowded_pois,
            'optimal_pois': insights.optimal_pois,
            'poi_predictions': []
        }
        
        # Add detailed POI predictions
        for poi_id, prediction in insights.poi_crowd_predictions.items():
            poi_data = {
                'poi_id': prediction.poi_id,
                'poi_name': prediction.poi_name,
                'crowd_level': prediction.crowd_level.name,
                'capacity_percentage': round(prediction.capacity_percentage, 1),
                'wait_time_minutes': prediction.wait_time_minutes,
                'recommended_visit': prediction.recommended_visit,
                'best_time_today': prediction.best_time_today.isoformat() if prediction.best_time_today else None,
                'alternative_times': [t.isoformat() for t in prediction.alternative_times] if prediction.alternative_times else [],
                'confidence': prediction.prediction_confidence
            }
            response['poi_predictions'].append(poi_data)
        
        processing_time_ms = (time.time() - start_time) * 1000
        response['processing_time_ms'] = processing_time_ms
        
        logger.info(f"[{request_id}] Crowd analysis completed in {processing_time_ms:.0f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Error analyzing crowding: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error analyzing crowd levels")


@app.get(
    "/api/v1/crowding/poi",
    tags=["Phase 6: Crowd Intelligence"],
    summary="Get crowd prediction for single POI"
)
async def get_poi_crowding(
    poi_id: str,
    poi_name: str,
    category: str,
    visit_time: str,
    district: Optional[str] = None
):
    """Get crowd prediction for a specific POI at given time"""
    try:
        service = get_crowding_intelligence_service()
        
        from datetime import datetime as dt
        visit_dt = dt.fromisoformat(visit_time)
        
        prediction = await service.predict_crowd_for_poi(
            poi_id=poi_id,
            poi_name=poi_name,
            category=category,
            visit_time=visit_dt,
            district=district
        )
        
        return {
            'poi_id': prediction.poi_id,
            'poi_name': prediction.poi_name,
            'visit_time': visit_time,
            'crowd_level': prediction.crowd_level.name,
            'capacity_percentage': round(prediction.capacity_percentage, 1),
            'wait_time_minutes': prediction.wait_time_minutes,
            'recommended_visit': prediction.recommended_visit,
            'best_time_today': prediction.best_time_today.isoformat() if prediction.best_time_today else None,
            'alternative_times': [t.isoformat() for t in prediction.alternative_times] if prediction.alternative_times else [],
            'confidence': prediction.prediction_confidence,
            'data_source': prediction.data_source
        }
        
    except Exception as e:
        logger.error(f"Error getting POI crowding: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting crowd prediction")


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting AI Istanbul Route Planner API (Phase 6)...")
    app.state.start_time = time.time()
    
    # Initialize all services
    planner = get_enhanced_gps_planner()
    cache = get_cache_service()
    multi_day_service = get_multi_day_service()
    crowding_service = get_crowding_intelligence_service()
    
    logger.info("✅ All services initialized successfully")
    logger.info(f"📍 POI Database: {len(planner.poi_db_service.get_all_pois()) if planner.poi_db_service else 0} POIs loaded")
    logger.info("🗓️  Multi-Day Itinerary Planning: Ready")
    logger.info("🌊 Crowd Intelligence System: Ready")
    logger.info("🌐 API ready at http://localhost:8000")
    logger.info("📚 API docs at http://localhost:8000/api/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down AI Istanbul Route Planner API...")
    logger.info("✅ Shutdown complete")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,  # Port 8001 for backend (vLLM tunnel uses 8000)
        reload=True,
        log_level="info"
    )
