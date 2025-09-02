from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Place
from typing import Optional
import urllib.parse

router = APIRouter(prefix="/places", tags=["Places"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def generate_google_maps_url(place_name: str, district: str = None) -> str:
    """Generate Google Maps search URL for a place"""
    search_query = f"{place_name}"
    if district:
        search_query += f", {district}"
    search_query += ", Istanbul, Turkey"
    
    # URL encode the search query
    encoded_query = urllib.parse.quote_plus(search_query)
    return f"https://www.google.com/maps/search/?api=1&query={encoded_query}"

@router.get("/")
def get_places(
    district: Optional[str] = Query(None, description="Filter by district"),
    limit: Optional[int] = Query(6, description="Limit number of results"),
    db: Session = Depends(get_db)
):
    # Build query
    query = db.query(Place)
    
    # Apply district filter if provided
    if district:
        query = query.filter(Place.district.ilike(f"%{district}%"))
    
    # Apply limit
    places = query.limit(limit).all()
    
    # Enhance places with Google Maps links and additional info
    enhanced_places = []
    for place in places:
        google_maps_url = generate_google_maps_url(place.name, place.district)
        
        enhanced_place = {
            "id": place.id,
            "name": place.name,
            "category": place.category,
            "district": place.district,
            "google_maps_url": google_maps_url,
            "address": f"{place.district}, Istanbul" if place.district else "Istanbul",
            "description": f"A notable {place.category.lower() if place.category else 'attraction'} located in {place.district if place.district else 'Istanbul'}. Click the link to view on Google Maps and get directions.",
        }
        enhanced_places.append(enhanced_place)
    
    return enhanced_places
