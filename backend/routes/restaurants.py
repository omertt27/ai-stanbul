from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from database import SessionLocal
from models import Restaurant
from api_clients.google_places import GooglePlacesClient, get_istanbul_restaurants_with_descriptions

router = APIRouter(prefix="/restaurants", tags=["Restaurants"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_restaurants(db: Session = Depends(get_db)):
    """Get all restaurants from the database."""
    return db.query(Restaurant).all()

@router.get("/search")
def search_restaurants_with_descriptions(
    location: Optional[str] = Query(None, description="Location to search (e.g., 'Beyoğlu, Istanbul')"),
    district: Optional[str] = Query(None, description="Istanbul district (e.g., 'Beyoğlu', 'Sultanahmet')"),
    keyword: Optional[str] = Query(None, description="Keyword to filter restaurants"),
    limit: int = Query(10, ge=1, le=50, description="Number of restaurants to return"),
    radius: int = Query(1500, ge=100, le=5000, description="Search radius in meters")
):
    """
    Search for restaurants with descriptions from Google Maps.
    Returns detailed information including descriptions, reviews, and photos.
    """
    try:
        client = GooglePlacesClient()
        
        if district:
            # Use specific Istanbul district
            search_location = f"{district}, Istanbul, Turkey"
        elif location:
            search_location = location
        else:
            # Default to Istanbul center
            search_location = "Istanbul, Turkey"
        
        restaurants = client.get_restaurants_with_descriptions(
            location=search_location,
            radius=radius,
            limit=limit,
            keyword=keyword
        )
        
        return {
            "status": "success",
            "location_searched": search_location,
            "total_found": len(restaurants),
            "restaurants": restaurants
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching restaurants: {str(e)}")

@router.get("/istanbul/{district}")
def get_istanbul_district_restaurants(
    district: str,
    limit: int = Query(10, ge=1, le=30, description="Number of restaurants to return")
):
    """
    Get restaurants from a specific Istanbul district with descriptions.
    
    Popular districts: Beyoğlu, Sultanahmet, Beşiktaş, Kadıköy, Şişli, Fatih, Üsküdar
    """
    try:
        restaurants = get_istanbul_restaurants_with_descriptions(
            district=district,
            limit=limit
        )
        
        if not restaurants:
            raise HTTPException(
                status_code=404, 
                detail=f"No restaurants found in {district} district"
            )
        
        return {
            "status": "success",
            "district": district,
            "total_found": len(restaurants),
            "restaurants": restaurants
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching restaurants: {str(e)}")

@router.get("/details/{place_id}")
def get_restaurant_details(place_id: str):
    """Get detailed information about a specific restaurant by its Google Places ID."""
    try:
        client = GooglePlacesClient()
        details = client.get_place_details(place_id)
        
        if details.get("status") != "OK":
            raise HTTPException(
                status_code=404, 
                detail=f"Restaurant not found or API error: {details.get('status')}"
            )
        
        return {
            "status": "success",
            "restaurant": details.get("result", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching restaurant details: {str(e)}")

@router.post("/save")
def save_restaurant_to_db(
    place_id: str,
    db: Session = Depends(get_db)
):
    """Save a restaurant from Google Places to the local database."""
    try:
        client = GooglePlacesClient()
        details = client.get_place_details(place_id)
        
        if details.get("status") != "OK":
            raise HTTPException(status_code=404, detail="Restaurant not found")
        
        result = details.get("result", {})
        
        # Check if restaurant already exists
        existing = db.query(Restaurant).filter(
            Restaurant.name == result.get("name")
        ).first()
        
        if existing:
            return {"status": "already_exists", "restaurant": existing}
        
        # Create new restaurant record
        restaurant = Restaurant(
            name=result.get("name"),
            cuisine=client._extract_cuisine_types(result.get("types", [])),
            location=result.get("formatted_address"),
            rating=result.get("rating"),
            source="Google Places"
        )
        
        db.add(restaurant)
        db.commit()
        db.refresh(restaurant)
        
        return {
            "status": "success",
            "message": "Restaurant saved to database",
            "restaurant": restaurant
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving restaurant: {str(e)}")

@router.get("/popular")
def get_popular_restaurants(
    min_rating: float = Query(4.0, ge=1.0, le=5.0, description="Minimum rating"),
    limit: int = Query(15, ge=1, le=30, description="Number of restaurants to return")
):
    """Get popular restaurants in Istanbul with high ratings and descriptions."""
    try:
        client = GooglePlacesClient()
        
        # Search in popular Istanbul areas
        popular_areas = ["Beyoğlu", "Sultanahmet", "Beşiktaş", "Kadıköy"]
        all_restaurants = []
        
        for area in popular_areas:
            restaurants = client.get_restaurants_with_descriptions(
                location=f"{area}, Istanbul, Turkey",
                limit=limit // len(popular_areas) + 2,
                radius=1000
            )
            
            # Filter by rating
            filtered = [r for r in restaurants if r.get("rating", 0) >= min_rating]
            all_restaurants.extend(filtered)
        
        # Sort by rating and remove duplicates
        unique_restaurants = {}
        for restaurant in all_restaurants:
            place_id = restaurant.get("place_id")
            if place_id not in unique_restaurants:
                unique_restaurants[place_id] = restaurant
        
        sorted_restaurants = sorted(
            unique_restaurants.values(),
            key=lambda x: (x.get("rating", 0), x.get("user_ratings_total", 0)),
            reverse=True
        )[:limit]
        
        return {
            "status": "success",
            "min_rating": min_rating,
            "total_found": len(sorted_restaurants),
            "restaurants": sorted_restaurants
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching popular restaurants: {str(e)}")
