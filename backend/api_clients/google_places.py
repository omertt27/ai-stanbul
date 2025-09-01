import requests
import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GooglePlacesClient:
    """Google Places API client for fetching restaurant information including descriptions."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_PLACES_API_KEY")
        if not self.api_key:
            raise ValueError("Google Places API key not found. Please set GOOGLE_MAPS_API_KEY or GOOGLE_PLACES_API_KEY environment variable.")
        self.base_url = "https://maps.googleapis.com/maps/api/place"
        
    def search_restaurants(self, 
                         location: str = None,
                         lat_lng: str = None, 
                         radius: int = 1500,
                         keyword: str = None,
                         min_rating: float = None) -> Dict:
        """
        Search for restaurants using Google Places Nearby Search API.
        
        Args:
            location: Location name (e.g., "Istanbul, Turkey")
            lat_lng: Coordinates in "lat,lng" format (e.g., "41.0082,28.9784")
            radius: Search radius in meters (max 50000)
            keyword: Keyword to filter results
            min_rating: Minimum rating filter
            
        Returns:
            Dictionary containing search results
        """
        url = f"{self.base_url}/nearbysearch/json"
        
        params = {
            "key": self.api_key,
            "type": "restaurant",
            "radius": radius,
        }
        
        # Use coordinates if provided, otherwise geocode location
        if lat_lng:
            params["location"] = lat_lng
        elif location:
            # Convert location name to coordinates
            geocode_result = self._geocode_location(location)
            if geocode_result:
                lat = geocode_result["lat"]
                lng = geocode_result["lng"]
                params["location"] = f"{lat},{lng}"
            else:
                logger.error(f"Could not geocode location: {location}")
                return {"status": "GEOCODING_ERROR", "results": []}
        else:
            # Default to Istanbul center
            params["location"] = "41.0082,28.9784"
            
        if keyword:
            params["keyword"] = keyword
            
        if min_rating:
            params["min_price_level"] = 0  # This doesn't filter by rating, but we'll filter after
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Filter by rating if specified
            if min_rating and "results" in data:
                data["results"] = [
                    place for place in data["results"] 
                    if place.get("rating", 0) >= min_rating
                ]
            
            return data
        except requests.RequestException as e:
            logger.error(f"Error searching restaurants: {e}")
            return {"status": "API_ERROR", "results": []}
    
    def get_place_details(self, place_id: str, fields: List[str] = None) -> Dict:
        """
        Get detailed information about a specific place including description/reviews.
        
        Args:
            place_id: Google Places place_id
            fields: List of fields to retrieve. If None, uses comprehensive default set.
            
        Returns:
            Dictionary containing place details
        """
        url = f"{self.base_url}/details/json"
        
        # Default fields for restaurant details with descriptions
        default_fields = [
            "place_id", "name", "formatted_address", "formatted_phone_number",
            "website", "rating", "user_ratings_total", "price_level",
            "opening_hours", "photos", "reviews", "types", "geometry",
            "business_status", "vicinity", "editorial_summary"
        ]
        
        fields_to_use = fields if fields else default_fields
        
        params = {
            "key": self.api_key,
            "place_id": place_id,
            "fields": ",".join(fields_to_use)
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error getting place details for {place_id}: {e}")
            return {"status": "API_ERROR"}
    
    def get_restaurants_with_descriptions(self, 
                                        location: str = None,
                                        lat_lng: str = None,
                                        radius: int = 1500,
                                        limit: int = 20,
                                        keyword: str = None) -> List[Dict]:
        """
        Get restaurants with detailed descriptions and reviews.
        
        Args:
            location: Location name or coordinates
            radius: Search radius in meters
            limit: Maximum number of restaurants to return
            keyword: Keyword filter
            
        Returns:
            List of restaurants with descriptions
        """
        # First, search for restaurants
        search_results = self.search_restaurants(
            location=location, 
            lat_lng=lat_lng,
            radius=radius, 
            keyword=keyword
        )
        
        if search_results.get("status") != "OK":
            logger.error(f"Restaurant search failed: {search_results.get('status')}")
            return []
        
        restaurants = []
        places = search_results.get("results", [])[:limit]
        
        for place in places:
            place_id = place.get("place_id")
            if not place_id:
                continue
                
            # Get detailed information for each restaurant
            details = self.get_place_details(place_id)
            
            if details.get("status") == "OK":
                result = details.get("result", {})
                
                # Extract description from various sources
                description = self._extract_description(result)
                
                restaurant_info = {
                    "place_id": place_id,
                    "name": result.get("name", "Unknown"),
                    "address": result.get("formatted_address", ""),
                    "phone": result.get("formatted_phone_number", ""),
                    "website": result.get("website", ""),
                    "rating": result.get("rating", 0),
                    "user_ratings_total": result.get("user_ratings_total", 0),
                    "price_level": result.get("price_level"),
                    "description": description,
                    "cuisine_types": self._extract_cuisine_types(result.get("types", [])),
                    "opening_hours": self._format_opening_hours(result.get("opening_hours")),
                    "location": {
                        "lat": result.get("geometry", {}).get("location", {}).get("lat"),
                        "lng": result.get("geometry", {}).get("location", {}).get("lng")
                    },
                    "photos": self._get_photo_urls(result.get("photos", [])[:3]),  # First 3 photos
                    "reviews_summary": self._summarize_reviews(result.get("reviews", [])[:5])
                }
                
                restaurants.append(restaurant_info)
        
        return restaurants
    
    def _geocode_location(self, location: str) -> Optional[Dict]:
        """Convert location name to coordinates using Google Geocoding API."""
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "key": self.api_key,
            "address": location
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "OK" and data.get("results"):
                location_data = data["results"][0]["geometry"]["location"]
                return {"lat": location_data["lat"], "lng": location_data["lng"]}
        except requests.RequestException as e:
            logger.error(f"Geocoding error: {e}")
        
        return None
    
    def _extract_description(self, place_details: Dict) -> str:
        """Extract description from various sources in place details."""
        descriptions = []
        
        # Editorial summary (Google's AI-generated summary)
        if place_details.get("editorial_summary", {}).get("overview"):
            descriptions.append(place_details["editorial_summary"]["overview"])
        
        # Extract from top reviews
        reviews = place_details.get("reviews", [])
        if reviews:
            # Get the most helpful review text
            top_review = max(reviews, key=lambda x: x.get("rating", 0))
            if top_review.get("text") and len(top_review["text"]) > 50:
                descriptions.append(f"Review: {top_review['text'][:200]}...")
        
        # If no description found, create one based on available data
        if not descriptions:
            name = place_details.get("name", "This restaurant")
            rating = place_details.get("rating")
            types = place_details.get("types", [])
            cuisine = self._extract_cuisine_types(types)
            
            desc_parts = [f"{name} is a restaurant"]
            if cuisine:
                desc_parts.append(f"specializing in {cuisine}")
            if rating:
                desc_parts.append(f"with a {rating} star rating")
            
            descriptions.append(" ".join(desc_parts) + ".")
        
        return " | ".join(descriptions)
    
    def _extract_cuisine_types(self, types: List[str]) -> str:
        """Extract cuisine types from Google Places types."""
        cuisine_mapping = {
            "restaurant": "Restaurant",
            "food": "Food",
            "meal_takeaway": "Takeaway",
            "meal_delivery": "Delivery",
            "bakery": "Bakery",
            "cafe": "Cafe",
            "bar": "Bar",
            "night_club": "Night Club",
            "tourist_attraction": "Tourist Attraction"
        }
        
        cuisines = []
        for place_type in types:
            if place_type in cuisine_mapping:
                cuisines.append(cuisine_mapping[place_type])
        
        return ", ".join(cuisines) if cuisines else "Restaurant"
    
    def _format_opening_hours(self, opening_hours: Dict) -> Dict:
        """Format opening hours information."""
        if not opening_hours:
            return {}
        
        return {
            "open_now": opening_hours.get("open_now", False),
            "weekday_text": opening_hours.get("weekday_text", [])
        }
    
    def _get_photo_urls(self, photos: List[Dict], max_width: int = 400) -> List[str]:
        """Convert photo references to URLs."""
        photo_urls = []
        for photo in photos:
            if photo.get("photo_reference"):
                url = f"{self.base_url}/photo"
                params = f"?maxwidth={max_width}&photoreference={photo['photo_reference']}&key={self.api_key}"
                photo_urls.append(url + params)
        
        return photo_urls
    
    def _summarize_reviews(self, reviews: List[Dict]) -> Dict:
        """Summarize reviews information."""
        if not reviews:
            return {}
        
        total_reviews = len(reviews)
        avg_rating = sum(review.get("rating", 0) for review in reviews) / total_reviews
        
        # Get most recent review snippet
        recent_review = ""
        if reviews:
            recent = max(reviews, key=lambda x: x.get("time", 0))
            recent_review = recent.get("text", "")[:150] + "..." if len(recent.get("text", "")) > 150 else recent.get("text", "")
        
        return {
            "total_reviews_shown": total_reviews,
            "average_rating": round(avg_rating, 1),
            "recent_review_snippet": recent_review
        }


# Convenience function for quick usage
def get_istanbul_restaurants_with_descriptions(district: str = None, limit: int = 10) -> List[Dict]:
    """
    Quick function to get Istanbul restaurants with descriptions.
    
    Args:
        district: Specific district in Istanbul (e.g., "Beyoğlu", "Sultanahmet")
        limit: Number of restaurants to return
        
    Returns:
        List of restaurants with descriptions
    """
    client = GooglePlacesClient()
    
    location = "Istanbul, Turkey"
    if district:
        location = f"{district}, Istanbul, Turkey"
    
    return client.get_restaurants_with_descriptions(
        location=location,
        limit=limit,
        radius=2000  # 2km radius
    )
