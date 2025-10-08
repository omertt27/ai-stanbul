#!/usr/bin/env python3
"""
Quick Location API Fix - Standalone location validation service
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from datetime import datetime

app = FastAPI(title="AI Istanbul Location Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LocationValidationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    location_name: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "AI Istanbul Location Service", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "location_api",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/location/validate")
async def validate_location(request: LocationValidationRequest):
    """Validate GPS coordinates and check if in Istanbul"""
    try:
        # Istanbul bounds
        ISTANBUL_BOUNDS = {
            "north": 41.2,
            "south": 40.8,
            "east": 29.3,
            "west": 28.5
        }
        
        lat, lng = request.latitude, request.longitude
        
        # Basic validation
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return {
                "valid": False,
                "in_istanbul": False,
                "error": "invalid_coordinates",
                "message": "Coordinates out of valid range"
            }
        
        # Check Istanbul bounds
        in_istanbul = (
            ISTANBUL_BOUNDS["south"] <= lat <= ISTANBUL_BOUNDS["north"] and
            ISTANBUL_BOUNDS["west"] <= lng <= ISTANBUL_BOUNDS["east"]
        )
        
        # Determine district
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
            "accuracy_meters": 50 if in_istanbul else 1000,
            "location_name": request.location_name,
            "coordinates": {"latitude": lat, "longitude": lng},
            "district": district,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "in_istanbul": False
        }

@app.get("/api/location/health")
async def location_health():
    return {
        "status": "healthy",
        "service": "location_validation",
        "features": ["coordinate_validation", "istanbul_detection", "district_mapping"],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting AI Istanbul Location Service...")
    print("📍 Available endpoints:")
    print("   GET  /health")
    print("   GET  /api/location/health") 
    print("   POST /api/location/validate")
    print("🌍 Starting server on http://localhost:8001")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
