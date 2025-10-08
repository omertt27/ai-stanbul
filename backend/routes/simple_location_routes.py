"""
Simple Location Routes for Testing - Minimal Implementation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/location", tags=["Location Services"])

# Request models
class LocationValidationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    location_name: Optional[str] = None

class LocationSessionRequest(BaseModel):
    user_id: Optional[str] = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    preferences: Optional[Dict[str, Any]] = None

@router.get("/health")
async def location_health():
    """Location service health check"""
    return {
        "status": "healthy",
        "service": "location_api",
        "features": ["validation", "sessions", "recommendations"]
    }

@router.post("/validate")
async def validate_location(request: LocationValidationRequest):
    """Validate GPS coordinates and check if in Istanbul"""
    
    try:
        # Istanbul bounds (approximate)
        ISTANBUL_BOUNDS = {
            "north": 41.2,
            "south": 40.8,
            "east": 29.3,
            "west": 28.5
        }
        
        lat, lng = request.latitude, request.longitude
        
        # Basic coordinate validation
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return {
                "valid": False,
                "in_istanbul": False,
                "error": "invalid_coordinates",
                "message": "Coordinates are out of valid range"
            }
        
        # Check if in Istanbul bounds
        in_istanbul = (
            ISTANBUL_BOUNDS["south"] <= lat <= ISTANBUL_BOUNDS["north"] and
            ISTANBUL_BOUNDS["west"] <= lng <= ISTANBUL_BOUNDS["east"]
        )
        
        # Calculate rough accuracy (for demo)
        accuracy_meters = 50 if in_istanbul else 1000
        
        # Determine district based on coordinates
        district = "Unknown"
        if in_istanbul:
            if lat > 41.03 and lng < 28.98:
                district = "Beyoğlu"
            elif lat < 41.01 and lng < 28.98:
                district = "Fatih/Sultanahmet"
            elif lat > 41.03 and lng > 28.98:
                district = "Beşiktaş"
            elif lng > 29.0:
                district = "Kadıköy (Asian Side)"
            else:
                district = "Central Istanbul"
        
        return {
            "valid": True,
            "in_istanbul": in_istanbul,
            "accuracy_meters": accuracy_meters,
            "location_name": request.location_name,
            "coordinates": {"latitude": lat, "longitude": lng},
            "district": district,
            "timestamp": "2025-10-08T19:30:00Z"
        }
        
    except Exception as e:
        logger.error(f"Location validation error: {str(e)}")
        return {
            "valid": False,
            "in_istanbul": False,
            "error": "validation_failed",
            "message": f"Validation failed: {str(e)}"
        }

@router.post("/session")
async def create_location_session(request: LocationSessionRequest):
    """Create a location-based session"""
    
    # Validate coordinates first
    validation_request = LocationValidationRequest(
        latitude=request.latitude,
        longitude=request.longitude
    )
    validation = await validate_location(validation_request)
    
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail="Invalid coordinates")
    
    # Create session
    session_id = f"session_{hash(f'{request.latitude}_{request.longitude}')}"
    
    return {
        "session_id": session_id,
        "user_id": request.user_id or "anonymous",
        "location": {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "in_istanbul": validation["in_istanbul"],
            "district": validation["district"]
        },
        "preferences": request.preferences or {},
        "status": "active",
        "features_available": [
            "nearby_restaurants",
            "route_planning", 
            "location_chat",
            "poi_recommendations"
        ]
    }

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "Location API is working!", "timestamp": "2025-10-08"}

@router.post("/recommendations")
async def get_location_recommendations(request: LocationSessionRequest):
    """Get POI recommendations based on location"""
    
    try:
        # Validate location first
        validation_request = LocationValidationRequest(
            latitude=request.latitude,
            longitude=request.longitude
        )
        validation = await validate_location(validation_request)
        
        if not validation["in_istanbul"]:
            return {
                "recommendations": [],
                "message": "No recommendations available outside Istanbul",
                "location_valid": False
            }
        
        # Mock recommendations based on district
        district = validation["district"]
        
        recommendations = {
            "Fatih/Sultanahmet": [
                {"name": "Blue Mosque", "type": "attraction", "distance": "200m", "rating": 4.8},
                {"name": "Hagia Sophia", "type": "attraction", "distance": "300m", "rating": 4.9},
                {"name": "Sultanahmet Köftecisi", "type": "restaurant", "distance": "150m", "rating": 4.5}
            ],
            "Beyoğlu": [
                {"name": "Galata Tower", "type": "attraction", "distance": "500m", "rating": 4.7},
                {"name": "İstiklal Street", "type": "street", "distance": "100m", "rating": 4.6},
                {"name": "Mikla Restaurant", "type": "restaurant", "distance": "800m", "rating": 4.8}
            ],
            "Beşiktaş": [
                {"name": "Dolmabahçe Palace", "type": "attraction", "distance": "1.2km", "rating": 4.7},
                {"name": "Beşiktaş Fish Market", "type": "market", "distance": "300m", "rating": 4.4},
                {"name": "Karaköy Lokantası", "type": "restaurant", "distance": "600m", "rating": 4.6}
            ]
        }
        
        district_recommendations = recommendations.get(district, [
            {"name": "Generic Istanbul Attraction", "type": "attraction", "distance": "1km", "rating": 4.5}
        ])
        
        return {
            "recommendations": district_recommendations,
            "location": validation["coordinates"],
            "district": district,
            "total_count": len(district_recommendations),
            "location_valid": True
        }
        
    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}")
        return {
            "recommendations": [],
            "error": str(e),
            "location_valid": False
        }
