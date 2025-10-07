"""
Route Maker API Endpoints
Phase 2: Multi-Stop TSP Optimization with Enhanced User Preferences
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field

from database import get_db
from services.route_maker_service import (
    route_maker, RouteRequest, RouteStyle, TransportMode, 
    GeneratedRoute, RoutePoint
)
from models import Route, RouteWaypoint, EnhancedAttraction, UserRoutePreferences

router = APIRouter(prefix="/api/routes", tags=["Route Maker - Phase 2"])

# Pydantic models for API
class RouteRequestModel(BaseModel):
    start_lat: float = Field(..., ge=-90, le=90, description="Starting latitude")
    start_lng: float = Field(..., ge=-180, le=180, description="Starting longitude")
    end_lat: Optional[float] = Field(None, ge=-90, le=90, description="Ending latitude (optional)")
    end_lng: Optional[float] = Field(None, ge=-180, le=180, description="Ending longitude (optional)")
    max_distance_km: float = Field(5.0, ge=0.1, le=50, description="Maximum route distance in km")
    available_time_hours: float = Field(4.0, ge=0.5, le=24, description="Available time in hours")
    preferred_categories: Optional[List[str]] = Field(None, description="Preferred attraction categories")
    route_style: str = Field("balanced", description="Route style: efficient, scenic, cultural, balanced")
    transport_mode: str = Field("walking", description="Transport: walking, driving, public_transport")
    include_food: bool = Field(True, description="Include food establishments")
    max_attractions: int = Field(6, ge=1, le=15, description="Maximum attractions (3-15 for TSP optimization)")
    optimization_method: Optional[str] = Field("auto", description="TSP method: auto, tsp, heuristic, nearest")

class TSPAnalysisRequest(BaseModel):
    attraction_ids: List[int] = Field(..., description="List of attraction IDs to optimize")
    start_lat: float = Field(..., ge=-90, le=90)
    start_lng: float = Field(..., ge=-180, le=180)
    methods: List[str] = Field(["tsp", "heuristic", "nearest"], description="Methods to compare")

class RoutePointModel(BaseModel):
    lat: float
    lng: float
    attraction_id: Optional[int] = None
    name: str
    category: str
    estimated_duration_minutes: int
    arrival_time: Optional[str] = None
    score: float = 0.0
    notes: str = ""

class GeneratedRouteModel(BaseModel):
    id: Optional[int] = None
    name: str
    description: str
    points: List[RoutePointModel]
    total_distance_km: float
    estimated_duration_hours: float
    overall_score: float
    diversity_score: float
    efficiency_score: float
    map_html: Optional[str] = None
    created_at: str = ""
    metadata: Optional[dict] = Field(default_factory=dict, description="Phase 2 enhancement metadata")

class RoutePreferencesModel(BaseModel):
    session_id: str
    max_walking_distance_km: float = 5.0
    preferred_pace: str = "moderate"
    available_time_hours: float = 4.0
    preferred_categories: Optional[List[str]] = None
    avoided_categories: Optional[List[str]] = None
    min_popularity_score: float = 2.0
    route_style: str = "balanced"
    include_food_stops: bool = True

@router.get("/")
async def get_route_maker_info():
    """Get information about the route maker capabilities"""
    return {
        "service": "Istanbul Route Maker",
        "version": "1.0",
        "description": "Production-ready route generation using OSM data and algorithmic optimization",
        "features": [
            "OSM-based pathfinding",
            "Multi-stop optimization", 
            "Attraction scoring",
            "User preference integration",
            "Interactive map generation"
        ],
        "supported_styles": ["efficient", "scenic", "cultural", "balanced"],
        "supported_transport": ["walking", "driving", "public_transport"],
        "max_attractions": 10,
        "max_distance_km": 20.0,
        "llm_free": True
    }

@router.post("/generate", response_model=GeneratedRouteModel)
async def generate_route(
    request: RouteRequestModel,
    save_to_db: bool = Query(False, description="Save generated route to database"),
    db: Session = Depends(get_db)
):
    """
    Phase 2: Enhanced route generation with TSP optimization
    
    Features:
    - TSP optimization for 3-10 attractions  
    - Intelligent attraction selection with diversity scoring
    - Time constraint optimization
    - Automatic district selection for best coverage
    - Multiple optimization methods (exact TSP, heuristic, nearest-neighbor)
    """
    try:
        # Convert route style string to enum
        try:
            route_style_enum = RouteStyle(request.route_style.upper())
        except ValueError:
            route_style_enum = RouteStyle.BALANCED
        
        # Convert transport mode string to enum
        try:
            transport_mode_enum = TransportMode(request.transport_mode.upper())
        except ValueError:
            transport_mode_enum = TransportMode.WALKING
        
        # Convert to internal request model
        route_request = RouteRequest(
            start_lat=request.start_lat,
            start_lng=request.start_lng,
            end_lat=request.end_lat,
            end_lng=request.end_lng,
            max_distance_km=request.max_distance_km,
            available_time_hours=request.available_time_hours,
            preferred_categories=request.preferred_categories or [],
            route_style=route_style_enum,
            transport_mode=transport_mode_enum,
            include_food=request.include_food,
            max_attractions=request.max_attractions
        )
        
        # Generate the route with enhanced optimization
        generated_route = route_maker.generate_route(route_request, db)
        
        # Determine which optimization method was used
        num_attractions = len([p for p in generated_route.points if p.attraction_id])
        if request.optimization_method == "auto":
            if num_attractions <= 2:
                opt_method_used = "direct"
            elif num_attractions <= 8:
                opt_method_used = "tsp_exact"
            else:
                opt_method_used = "tsp_heuristic"
        else:
            opt_method_used = request.optimization_method
        
        # Save to database if requested
        route_id = None
        if save_to_db and generated_route.points:
            try:
                saved_route = route_maker.save_route_to_db(generated_route, db)
                route_id = saved_route.id
            except Exception as save_error:
                print(f"Warning: Failed to save route to DB: {save_error}")
        
        # Enhanced response with Phase 2 features
        response = GeneratedRouteModel(
            id=route_id,
            name=generated_route.name,
            description=generated_route.description,
            total_distance_km=generated_route.total_distance_km,
            estimated_duration_hours=generated_route.estimated_duration_hours,
            overall_score=generated_route.overall_score,
            diversity_score=generated_route.diversity_score,
            efficiency_score=generated_route.efficiency_score,
            points=[{
                "lat": p.lat,
                "lng": p.lng,
                "attraction_id": p.attraction_id,
                "name": p.name,
                "category": p.category,
                "estimated_duration_minutes": p.estimated_duration_minutes,
                "arrival_time": p.arrival_time,
                "score": p.score,
                "notes": p.notes
            } for p in generated_route.points],
            created_at=generated_route.created_at.isoformat() if generated_route.created_at else "",
            metadata={
                "optimization_method": opt_method_used,
                "districts_covered": getattr(route_maker, 'covered_districts', ['Unknown']),
                "tsp_optimized": opt_method_used in ["tsp_exact", "tsp_heuristic"],
                "num_attractions": num_attractions,
                "primary_district": getattr(route_maker, 'primary_district', 'Unknown'),
                "phase": "2_multi_stop_tsp"
            }
        )
        
        return response
        
        # Generate map HTML
        map_html = route_maker.generate_map_html(generated_route)
        
        # Convert to response model
        response = GeneratedRouteModel(
            name=generated_route.name,
            description=generated_route.description,
            points=[
                RoutePointModel(
                    lat=p.lat,
                    lng=p.lng,
                    attraction_id=p.attraction_id,
                    name=p.name,
                    category=p.category,
                    estimated_duration_minutes=p.estimated_duration_minutes,
                    arrival_time=p.arrival_time,
                    score=p.score,
                    notes=p.notes
                ) for p in generated_route.points
            ],
            total_distance_km=generated_route.total_distance_km,
            estimated_duration_hours=generated_route.estimated_duration_hours,
            overall_score=generated_route.overall_score,
            diversity_score=generated_route.diversity_score,
            efficiency_score=generated_route.efficiency_score,
            map_html=map_html
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route generation failed: {str(e)}")

@router.post("/save")
async def save_route(
    request: RouteRequestModel,
    db: Session = Depends(get_db)
):
    """Generate and save a route to the database"""
    try:
        # Generate route
        route_request = RouteRequest(
            start_lat=request.start_lat,
            start_lng=request.start_lng,
            end_lat=request.end_lat,
            end_lng=request.end_lng,
            max_distance_km=request.max_distance_km,
            available_time_hours=request.available_time_hours,
            preferred_categories=request.preferred_categories or [],
            route_style=RouteStyle(request.route_style),
            transport_mode=TransportMode(request.transport_mode),
            include_food=request.include_food,
            max_attractions=request.max_attractions
        )
        
        generated_route = route_maker.generate_route(route_request, db)
        
        # Save to database
        saved_route = route_maker.save_route_to_db(generated_route, db)
        
        return {
            "success": True,
            "route_id": saved_route.id,
            "message": "Route saved successfully",
            "route": {
                "name": saved_route.name,
                "total_distance_km": saved_route.total_distance_km,
                "estimated_duration_hours": saved_route.estimated_duration_hours,
                "overall_score": saved_route.overall_score
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save route: {str(e)}")

@router.get("/saved")
async def get_saved_routes(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get list of saved routes"""
    routes = db.query(Route).filter(Route.is_saved == True).limit(limit).all()
    
    return {
        "routes": [
            {
                "id": route.id,
                "name": route.name,
                "description": route.description,
                "total_distance_km": route.total_distance_km,
                "estimated_duration_hours": route.estimated_duration_hours,
                "overall_score": route.overall_score,
                "created_at": route.created_at.isoformat() if route.created_at else None,
                "waypoint_count": len(route.waypoints)
            } for route in routes
        ]
    }

@router.get("/saved/{route_id}")
async def get_saved_route(route_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a saved route"""
    route = db.query(Route).filter(Route.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    
    # Get waypoints with attraction details
    waypoints = []
    for waypoint in sorted(route.waypoints, key=lambda x: x.waypoint_order):
        attraction = db.query(EnhancedAttraction).filter(
            EnhancedAttraction.id == waypoint.attraction_id
        ).first()
        
        waypoints.append({
            "order": waypoint.waypoint_order,
            "arrival_time": waypoint.estimated_arrival_time,
            "duration_minutes": waypoint.suggested_duration_minutes,
            "attraction": {
                "id": attraction.id if attraction else None,
                "name": attraction.name if attraction else "Unknown",
                "category": attraction.category if attraction else "unknown",
                "coordinates": {
                    "lat": attraction.coordinates_lat if attraction else 0,
                    "lng": attraction.coordinates_lng if attraction else 0
                }
            } if attraction else None,
            "score": waypoint.attraction_score,
            "notes": waypoint.notes
        })
    
    return {
        "route": {
            "id": route.id,
            "name": route.name,
            "description": route.description,
            "start_coordinates": {
                "lat": route.start_lat,
                "lng": route.start_lng
            },
            "end_coordinates": {
                "lat": route.end_lat,
                "lng": route.end_lng
            } if route.end_lat and route.end_lng else None,
            "total_distance_km": route.total_distance_km,
            "estimated_duration_hours": route.estimated_duration_hours,
            "overall_score": route.overall_score,
            "diversity_score": route.diversity_score,
            "efficiency_score": route.efficiency_score,
            "created_at": route.created_at.isoformat() if route.created_at else None,
            "waypoints": waypoints
        }
    }

@router.get("/{route_id}/map")
async def get_route_map(route_id: int, db: Session = Depends(get_db)):
    """Get interactive HTML map for a saved route"""
    try:
        route = db.query(Route).filter(Route.id == route_id).first()
        if not route:
            raise HTTPException(status_code=404, detail="Route not found")
        
        # Load waypoints
        waypoints = db.query(RouteWaypoint).filter(
            RouteWaypoint.route_id == route_id
        ).order_by(RouteWaypoint.waypoint_order).all()
        
        # Convert to GeneratedRoute format for map generation
        points = []
        for wp in waypoints:
            attraction = db.query(EnhancedAttraction).filter(
                EnhancedAttraction.id == wp.attraction_id
            ).first()
            
            if attraction:
                points.append(RoutePoint(
                    lat=attraction.lat,
                    lng=attraction.lng,
                    attraction_id=attraction.id,
                    name=attraction.name,
                    category=attraction.category,
                    estimated_duration_minutes=wp.suggested_duration_minutes,
                    arrival_time=wp.estimated_arrival_time,
                    score=wp.attraction_score or 0.0,
                    notes=wp.notes or ""
                ))
        
        generated_route = GeneratedRoute(
            id=route.id,
            name=route.name,
            description=route.description,
            points=points,
            total_distance_km=route.total_distance_km,
            estimated_duration_hours=route.estimated_duration_hours,
            overall_score=route.overall_score
        )
        
        map_html = route_maker.generate_map_html(generated_route)
        
        return {
            "route_id": route_id,
            "map_html": map_html,
            "interactive": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Map generation failed: {str(e)}")

@router.get("/cache/stats")
async def get_cache_stats():
    """Get route cache performance statistics"""
    try:
        from backend.services.route_cache import route_cache
        stats = route_cache.get_stats()
        return {
            "cache_enabled": True,
            "cache_type": stats.get("backend", "memory"),
            "total_requests": stats.get("total_requests", 0),
            "cache_hits": stats.get("cache_hits", 0),
            "cache_misses": stats.get("cache_misses", 0),
            "hit_rate": stats.get("hit_rate", 0.0),
            "cached_routes": stats.get("cached_routes", 0),
            "memory_usage_mb": stats.get("memory_usage_mb", 0),
            "last_reset": stats.get("last_reset", "Never")
        }
    except ImportError:
        return {
            "cache_enabled": False,
            "message": "Route caching not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")

@router.post("/cache/clear")
async def clear_route_cache():
    """Clear the route cache"""
    try:
        from backend.services.route_cache import route_cache
        cleared_count = route_cache.clear()
        return {
            "success": True,
            "cleared_routes": cleared_count,
            "message": f"Cleared {cleared_count} cached routes"
        }
    except ImportError:
        raise HTTPException(status_code=404, detail="Route caching not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@router.get("/attractions/nearby")
async def find_nearby_attractions(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(5.0, ge=0.1, le=50),
    categories: List[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    min_score: float = Query(0.0, ge=0, le=10),
    db: Session = Depends(get_db)
):
    """Find attractions near a specific location"""
    try:
        from sqlalchemy import func
        
        # Base query
        query = db.query(EnhancedAttraction).filter(
            EnhancedAttraction.lat.isnot(None),
            EnhancedAttraction.lng.isnot(None),
            EnhancedAttraction.popularity_score >= min_score
        )
        
        # Filter by categories if provided
        if categories:
            query = query.filter(EnhancedAttraction.category.in_(categories))
        
        # Get all attractions (we'll filter by distance in Python for simplicity)
        all_attractions = query.all()
        
        # Calculate distances and filter
        from geopy.distance import geodesic
        nearby_attractions = []
        
        for attraction in all_attractions:
            if attraction.lat is not None and attraction.lng is not None:
                distance = geodesic((lat, lng), (attraction.lat, attraction.lng)).kilometers
                if distance <= radius_km:
                    nearby_attractions.append({
                        "id": attraction.id,
                        "name": attraction.name,
                        "category": attraction.category,
                        "district": attraction.district,
                        "lat": attraction.lat,
                        "lng": attraction.lng,
                        "popularity_score": attraction.popularity_score,
                        "description": attraction.description,
                        "distance_km": round(distance, 2)
                    })
        
        # Sort by distance and limit
        nearby_attractions.sort(key=lambda x: x["distance_km"])
        nearby_attractions = nearby_attractions[:limit]
        
        return {
            "attractions": nearby_attractions,
            "total_found": len(nearby_attractions),
            "search_params": {
                "center": {"lat": lat, "lng": lng},
                "radius_km": radius_km,
                "categories": categories,
                "min_score": min_score
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attraction search failed: {str(e)}")

@router.post("/save")
async def save_route_to_db(
    route_data: GeneratedRouteModel,
    db: Session = Depends(get_db)
):
    """Save a generated route to database for future reference"""
    try:
        # Convert model to internal format
        points = [RoutePoint(
            lat=p.lat,
            lng=p.lng,
            attraction_id=p.attraction_id,
            name=p.name,
            category=p.category,
            estimated_duration_minutes=p.estimated_duration_minutes,
            arrival_time=p.arrival_time,
            score=p.score,
            notes=p.notes
        ) for p in route_data.points]
        
        generated_route = GeneratedRoute(
            name=route_data.name,
            description=route_data.description,
            points=points,
            total_distance_km=route_data.total_distance_km,
            estimated_duration_hours=route_data.estimated_duration_hours,
            overall_score=route_data.overall_score,
            diversity_score=route_data.diversity_score,
            efficiency_score=route_data.efficiency_score
        )
        
        # Save to database
        saved_route = route_maker.save_route_to_db(generated_route, db)
        
        return {
            "success": True,
            "route_id": saved_route.id,
            "message": f"Route '{saved_route.name}' saved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route save failed: {str(e)}")

@router.get("/{route_id}")
async def get_saved_route(route_id: int, db: Session = Depends(get_db)):
    """Load a saved route from database"""
    try:
        route = db.query(Route).filter(Route.id == route_id).first()
        if not route:
            raise HTTPException(status_code=404, detail="Route not found")
        
        # Load waypoints with attractions
        waypoints = db.query(RouteWaypoint).join(EnhancedAttraction).filter(
            RouteWaypoint.route_id == route_id
        ).order_by(RouteWaypoint.waypoint_order).all()
        
        points = []
        for wp in waypoints:
            attraction = wp.attraction
            points.append(RoutePointModel(
                lat=attraction.lat,
                lng=attraction.lng,
                attraction_id=attraction.id,
                name=attraction.name,
                category=attraction.category,
                estimated_duration_minutes=wp.suggested_duration_minutes,
                arrival_time=wp.estimated_arrival_time,
                score=wp.attraction_score or 0.0,
                notes=wp.notes or ""
            ))
        
        return GeneratedRouteModel(
            id=route.id,
            name=route.name,
            description=route.description,
            points=points,
            total_distance_km=route.total_distance_km,
            estimated_duration_hours=route.estimated_duration_hours,
            overall_score=route.overall_score,
            diversity_score=route.diversity_score,
            efficiency_score=route.efficiency_score,
            created_at=route.created_at.isoformat() if route.created_at else ""
        )
        
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Route load failed: {str(e)}")

@router.delete("/{route_id}")
async def delete_saved_route(route_id: int, db: Session = Depends(get_db)):
    """Delete a saved route from database"""
    try:
        route = db.query(Route).filter(Route.id == route_id).first()
        if not route:
            raise HTTPException(status_code=404, detail="Route not found")
        
        # Delete waypoints first (foreign key constraint)
        db.query(RouteWaypoint).filter(RouteWaypoint.route_id == route_id).delete()
        
        # Delete the route
        db.delete(route)
        db.commit()
        
        return {"success": True, "message": f"Route '{route.name}' deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Route deletion failed: {str(e)}")

@router.get("/attractions/nearby")
async def get_nearby_attractions(
    lat: float = Query(..., ge=40.8, le=41.25, description="Latitude"),
    lng: float = Query(..., ge=28.6, le=29.4, description="Longitude"),
    radius_km: float = Query(2.0, ge=0.1, le=10.0, description="Search radius in kilometers"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get attractions near a specific location"""
    try:
        attractions = route_maker.get_attractions_near_point(lat, lng, radius_km, db)
        
        # Filter by category if specified
        if category:
            attractions = [a for a in attractions if a.category.lower() == category.lower()]
        
        # Limit results
        attractions = attractions[:limit]
        
        return {
            "location": {"lat": lat, "lng": lng},
            "radius_km": radius_km,
            "total_found": len(attractions),
            "attractions": [
                {
                    "id": attr.id,
                    "name": attr.name,
                    "category": attr.category,
                    "subcategory": attr.subcategory,
                    "coordinates": {
                        "lat": attr.coordinates_lat,
                        "lng": attr.coordinates_lng
                    },
                    "distance_km": round(getattr(attr, 'distance_from_point', 0), 2),
                    "popularity_score": attr.popularity_score,
                    "estimated_visit_time_minutes": attr.estimated_visit_time_minutes,
                    "price_range": attr.price_range,
                    "district": attr.district,
                    "crowd_level": attr.crowd_level
                } for attr in attractions
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get nearby attractions: {str(e)}")

@router.post("/preferences")
async def save_route_preferences(
    preferences: RoutePreferencesModel,
    db: Session = Depends(get_db)
):
    """Save user route preferences"""
    try:
        # Check if preferences already exist
        existing = db.query(UserRoutePreferences).filter(
            UserRoutePreferences.session_id == preferences.session_id
        ).first()
        
        if existing:
            # Update existing preferences
            existing.max_walking_distance_km = preferences.max_walking_distance_km
            existing.preferred_pace = preferences.preferred_pace
            existing.available_time_hours = preferences.available_time_hours
            existing.preferred_categories = preferences.preferred_categories
            existing.avoided_categories = preferences.avoided_categories
            existing.min_popularity_score = preferences.min_popularity_score
            existing.route_style = preferences.route_style
            existing.include_food_stops = preferences.include_food_stops
            existing.updated_at = db.func.now()
        else:
            # Create new preferences
            new_prefs = UserRoutePreferences(
                session_id=preferences.session_id,
                max_walking_distance_km=preferences.max_walking_distance_km,
                preferred_pace=preferences.preferred_pace,
                available_time_hours=preferences.available_time_hours,
                preferred_categories=preferences.preferred_categories,
                avoided_categories=preferences.avoided_categories,
                min_popularity_score=preferences.min_popularity_score,
                route_style=preferences.route_style,
                include_food_stops=preferences.include_food_stops
            )
            db.add(new_prefs)
        
        db.commit()
        
        return {
            "success": True,
            "message": "Route preferences saved successfully",
            "session_id": preferences.session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save preferences: {str(e)}")

@router.get("/preferences/{session_id}")
async def get_route_preferences(session_id: str, db: Session = Depends(get_db)):
    """Get user route preferences"""
    preferences = db.query(UserRoutePreferences).filter(
        UserRoutePreferences.session_id == session_id
    ).first()
    
    if not preferences:
        # Return default preferences
        return {
            "session_id": session_id,
            "max_walking_distance_km": 5.0,
            "preferred_pace": "moderate",
            "available_time_hours": 4.0,
            "preferred_categories": [],
            "avoided_categories": [],
            "min_popularity_score": 2.0,
            "route_style": "balanced",
            "include_food_stops": True,
            "is_default": True
        }
    
    return {
        "session_id": preferences.session_id,
        "max_walking_distance_km": preferences.max_walking_distance_km,
        "preferred_pace": preferences.preferred_pace,
        "available_time_hours": preferences.available_time_hours,
        "preferred_categories": preferences.preferred_categories,
        "avoided_categories": preferences.avoided_categories,
        "min_popularity_score": preferences.min_popularity_score,
        "route_style": preferences.route_style,
        "include_food_stops": preferences.include_food_stops,
        "created_at": preferences.created_at.isoformat() if preferences.created_at else None,
        "updated_at": preferences.updated_at.isoformat() if preferences.updated_at else None,
        "is_default": False
    }

@router.get("/categories")
async def get_attraction_categories(db: Session = Depends(get_db)):
    """Get list of available attraction categories"""
    try:
        # Get unique categories from the database
        categories = db.query(EnhancedAttraction.category.distinct()).filter(
            EnhancedAttraction.is_active == True
        ).all()
        
        category_list = [cat[0] for cat in categories if cat[0]]
        
        return {
            "categories": sorted(category_list),
            "total_count": len(category_list),
            "description": "Available attraction categories for route generation"
        }
        
    except Exception as e:
        return {
            "categories": [
                "mosque", "museum", "restaurant", "cafe", "bazaar", 
                "park", "viewpoint", "cultural_site", "nightlife", 
                "shopping", "transportation", "historic_site"
            ],
            "total_count": 12,
            "description": "Default attraction categories (database query failed)",
            "error": str(e)
        }

@router.post("/analyze-tsp")
async def analyze_tsp_optimization(
    request: TSPAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Phase 2: Analyze different TSP optimization methods for a set of attractions
    Compare nearest-neighbor, exact TSP, and heuristic approaches
    """
    try:
        # Get attractions from database
        attractions = db.query(EnhancedAttraction).filter(
            EnhancedAttraction.id.in_(request.attraction_ids)
        ).all()
        
        if not attractions:
            raise HTTPException(status_code=404, detail="No attractions found")
        
        if len(attractions) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 attractions for optimization")
        
        # Test different optimization methods
        results = {}
        
        for method in request.methods:
            if method == "tsp" and len(attractions) > 10:
                results[method] = {"error": "TSP exact method limited to 10 attractions"}
                continue
            
            try:
                optimized = route_maker.optimize_route_order(
                    attractions, request.start_lat, request.start_lng, method=method
                )
                
                total_distance = route_maker._calculate_route_distance(
                    optimized, request.start_lat, request.start_lng
                )
                
                results[method] = {
                    "total_distance_km": round(total_distance, 2),
                    "route_order": [a.id for a in optimized],
                    "attraction_names": [a.name for a in optimized],
                    "improvement_over_nearest": 0.0  # Will calculate below
                }
            except Exception as e:
                results[method] = {"error": str(e)}
        
        # Calculate improvements
        if "nearest" in results and "total_distance_km" in results["nearest"]:
            baseline = results["nearest"]["total_distance_km"]
            for method, result in results.items():
                if "total_distance_km" in result:
                    improvement = ((baseline - result["total_distance_km"]) / baseline) * 100
                    result["improvement_over_nearest"] = round(improvement, 1)
        
        # Determine best method
        valid_results = {k: v for k, v in results.items() if "total_distance_km" in v}
        best_method = min(valid_results.keys(), key=lambda k: valid_results[k]["total_distance_km"]) if valid_results else None
        
        return {
            "analysis": results,
            "recommendation": best_method,
            "num_attractions": len(attractions),
            "tsp_feasible": len(attractions) <= 10,
            "summary": {
                "best_distance": valid_results[best_method]["total_distance_km"] if best_method else None,
                "worst_distance": max(r["total_distance_km"] for r in valid_results.values()) if valid_results else None,
                "optimization_benefit": f"{max(r['improvement_over_nearest'] for r in valid_results.values()):.1f}%" if valid_results else "0%"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TSP analysis failed: {str(e)}")

@router.get("/attractions/categories")
async def get_attraction_categories(db: Session = Depends(get_db)):
    """Get all available attraction categories for filtering"""
    try:
        categories = db.query(EnhancedAttraction.category).distinct().all()
        return {
            "categories": [cat[0] for cat in categories if cat[0]],
            "total_categories": len(categories)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

@router.get("/districts/status")
async def get_districts_status():
    """Get current district coverage and routing capabilities"""
    try:
        status = {
            "primary_district": getattr(route_maker, 'primary_district', 'Unknown'),
            "available_districts": list(getattr(route_maker, 'available_districts', {}).keys()),
            "graph_stats": {
                "nodes": len(route_maker.graph.nodes) if route_maker.graph else 0,
                "edges": len(route_maker.graph.edges) if route_maker.graph else 0
            },
            "district_stats": {}
        }
        
        # Add stats for each available district
        if hasattr(route_maker, 'available_districts'):
            for name, graph in route_maker.available_districts.items():
                status["district_stats"][name] = {
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                    "is_primary": name == route_maker.primary_district
                }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/districts/switch")
async def switch_district(
    district_name: str = Query(..., description="District name to switch to"),
):
    """Switch the primary routing district"""
    try:
        success = route_maker.switch_to_district(district_name)
        
        if success:
            return {
                "success": True,
                "message": f"Switched to {district_name} district",
                "new_primary": route_maker.primary_district,
                "graph_size": {
                    "nodes": len(route_maker.graph.nodes),
                    "edges": len(route_maker.graph.edges)
                }
            }
        else:
            available = list(getattr(route_maker, 'available_districts', {}).keys())
            raise HTTPException(
                status_code=400, 
                detail=f"District '{district_name}' not available. Available: {available}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"District switch failed: {str(e)}")

@router.get("/performance/stats")
async def get_performance_stats(
    operation: Optional[str] = Query(None, description="Filter by operation type"),
    hours: int = Query(24, ge=1, le=168, description="Stats period in hours")
):
    """Get performance statistics for route operations"""
    try:
        from backend.services.performance_monitor import performance_monitor
        stats = performance_monitor.get_stats(operation=operation, hours=hours)
        return {
            "success": True,
            "performance_stats": stats,
            "monitoring_enabled": True
        }
    except ImportError:
        return {
            "success": False,
            "message": "Performance monitoring not available",
            "monitoring_enabled": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance stats failed: {str(e)}")

@router.get("/performance/health")
async def get_system_health():
    """Get comprehensive system health metrics"""
    try:
        from backend.services.performance_monitor import get_system_health
        health = get_system_health()
        return health
    except ImportError:
        return {
            "health_status": "unknown",
            "message": "Health monitoring not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/performance/slow-operations")
async def get_slow_operations(
    threshold_ms: float = Query(1000, ge=100, description="Minimum duration in milliseconds"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return")
):
    """Get slowest operations above threshold"""
    try:
        from backend.services.performance_monitor import performance_monitor
        slow_ops = performance_monitor.get_slow_operations(threshold_ms=threshold_ms, limit=limit)
        return {
            "slow_operations": slow_ops,
            "threshold_ms": threshold_ms,
            "total_found": len(slow_ops)
        }
    except ImportError:
        return {
            "slow_operations": [],
            "message": "Performance monitoring not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Slow operations query failed: {str(e)}")

@router.post("/performance/optimize-memory")
async def optimize_memory():
    """Force garbage collection and optimize memory usage"""
    try:
        from backend.services.performance_monitor import optimize_memory
        result = optimize_memory()
        return {
            "success": True,
            "memory_optimization": result
        }
    except ImportError:
        return {
            "success": False,
            "message": "Memory optimization not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory optimization failed: {str(e)}")

@router.post("/performance/clear-metrics")
async def clear_performance_metrics():
    """Clear all stored performance metrics"""
    try:
        from backend.services.performance_monitor import performance_monitor
        cleared_count = performance_monitor.clear_metrics()
        return {
            "success": True,
            "cleared_metrics": cleared_count,
            "message": f"Cleared {cleared_count} performance metrics"
        }
    except ImportError:
        return {
            "success": False,
            "message": "Performance monitoring not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics clear failed: {str(e)}")

@router.get("/user")
async def get_user_routes(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get user's saved routes"""
    try:
        # For now, get all routes (in production, filter by user ID)
        routes = db.query(Route).order_by(Route.created_at.desc()).limit(limit).all()
        
        route_list = []
        for route in routes:
            route_list.append({
                "id": route.id,
                "name": route.name,
                "description": route.description,
                "total_distance_km": route.total_distance_km,
                "estimated_duration_hours": route.estimated_duration_hours,
                "overall_score": route.overall_score,
                "created_at": route.created_at.isoformat() if route.created_at else None,
                "waypoint_count": db.query(RouteWaypoint).filter(RouteWaypoint.route_id == route.id).count()
            })
        
        return {
            "routes": route_list,
            "total_found": len(route_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user routes: {str(e)}")

@router.get("/cache/analytics")
async def get_cache_analytics():
    """Get detailed cache performance analytics"""
    try:
        from backend.services.route_cache import route_cache
        analytics = route_cache.get_cache_analytics()
        return {
            "success": True,
            "analytics": analytics
        }
    except ImportError:
        return {
            "success": False,
            "message": "Cache analytics not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache analytics failed: {str(e)}")

@router.get("/cache/popular-routes")
async def get_popular_routes(limit: int = Query(10, ge=1, le=50)):
    """Get most popular route patterns"""
    try:
        from backend.services.route_cache import route_cache
        popular_routes = route_cache.get_popular_routes(limit=limit)
        return {
            "popular_routes": popular_routes,
            "total_returned": len(popular_routes)
        }
    except ImportError:
        return {
            "popular_routes": [],
            "message": "Popular routes tracking not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Popular routes query failed: {str(e)}")

@router.post("/cache/optimize")
async def optimize_cache():
    """Optimize cache performance and cleanup"""
    try:
        from backend.services.route_cache import route_cache
        optimization_result = route_cache.optimize_cache_performance()
        return {
            "success": True,
            "optimization": optimization_result
        }
    except ImportError:
        return {
            "success": False,
            "message": "Cache optimization not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache optimization failed: {str(e)}")

@router.post("/cache/precompute-popular")
async def precompute_popular_routes():
    """Precompute and cache popular route patterns"""
    try:
        from backend.services.route_cache import route_cache
        result = route_cache.cache_popular_routes()
        return {
            "success": True,
            "precomputation": result
        }
    except ImportError:
        return {
            "success": False,
            "message": "Route precomputation not available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route precomputation failed: {str(e)}")
