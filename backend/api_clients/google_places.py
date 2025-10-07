import requests
import os
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GooglePlacesClient:
    """Google Places API client for fetching restaurant information including descriptions."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_PLACES_API_KEY")
        self.has_api_key = bool(self.api_key)
        
        if not self.has_api_key:
            logger.warning("Google Places API key not found. Using fallback mode with mock data.")
        
        self.base_url = "https://maps.googleapis.com/maps/api/place"
        
    def search_restaurants(self, 
                         location: Optional[str] = None,
                         lat_lng: Optional[str] = None, 
                         radius: int = 1500,
                         keyword: Optional[str] = None,
                         min_rating: Optional[float] = None) -> Dict:
        """
        Search for restaurants using Google Places Text Search API for better restaurant filtering.
        
        Args:
            location: Location name (e.g., "Istanbul, Turkey")
            lat_lng: Coordinates in "lat,lng" format (e.g., "41.0082,28.9784")
            radius: Search radius in meters (max 50000)
            keyword: Keyword to filter results
            min_rating: Minimum rating filter
            
        Returns:
            Dictionary containing search results
        """
        # Always use mock data for AI chat system
        return self._get_mock_restaurant_data(location, keyword)
            
        # Use text search for better restaurant filtering
        url = f"{self.base_url}/textsearch/json"
        
        # Build search query
        query_parts = ["restaurant"]
        
        if location:
            query_parts.append(f"in {location}")
        elif lat_lng:
            # Convert coordinates to location if possible
            geocode_result = self._reverse_geocode(lat_lng)
            if geocode_result:
                query_parts.append(f"in {geocode_result}")
        else:
            query_parts.append("in Istanbul, Turkey")
            
        if keyword:
            query_parts.append(keyword)
        
        query = " ".join(query_parts)
        
        params = {
            "key": self.api_key,
            "query": query,
            "type": "restaurant",
        }
        
        if radius and radius <= 50000:
            params["radius"] = radius
        
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
    
    def get_place_details(self, place_id: str, fields: Optional[List[str]] = None) -> Dict:
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
                                        location: Optional[str] = None,
                                        lat_lng: Optional[str] = None,
                                        radius: int = 1500,
                                        limit: int = 5,
                                        keyword: Optional[str] = None) -> List[Dict]:
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
        places = search_results.get("results", [])
        
        # Filter out hotels and lodging establishments - be more strict
        filtered_places = []
        for place in places:
            place_types = place.get("types", [])
            # Exclude places that are primarily lodging/accommodation
            lodging_types = ['lodging', 'hotel', 'motel', 'inn', 'resort', 'hostel', 'guest_house']
            is_lodging = any(lodging_type in place_types for lodging_type in lodging_types)
            
            # Only include if it's NOT lodging at all
            if not is_lodging:
                filtered_places.append(place)
                logger.info(f"Included: {place.get('name', 'Unknown')} - types: {place_types}")
            else:
                logger.info(f"Excluded: {place.get('name', 'Unknown')} - lodging establishment with types: {place_types}")
        
        # Limit after filtering
        filtered_places = filtered_places[:limit]
        
        for place in filtered_places:
            place_id = place.get("place_id")
            if not place_id:
                continue
                
            # Get detailed information for each restaurant
            details = self.get_place_details(place_id)
            
            if details.get("status") == "OK":
                result = details.get("result", {})
                
                # Additional filter: Double-check that this is not primarily a lodging establishment
                result_types = result.get("types", [])
                lodging_types = ['lodging', 'hotel', 'motel', 'inn', 'resort', 'hostel', 'guest_house']
                is_lodging = any(lodging_type in result_types for lodging_type in lodging_types)
                
                # Skip if it's primarily lodging
                if is_lodging:
                    logger.info(f"Skipping in details: {result.get('name', 'Unknown')} - lodging establishment with types: {result_types}")
                    continue
                
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
    
    def _reverse_geocode(self, lat_lng: str) -> Optional[str]:
        """Convert coordinates to location name using Google Reverse Geocoding API."""
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "key": self.api_key,
            "latlng": lat_lng
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "OK" and data.get("results"):
                # Get the first formatted address
                return data["results"][0]["formatted_address"]
        except requests.RequestException as e:
            logger.error(f"Reverse geocoding error: {e}")
        
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
    
    def _get_mock_restaurant_data(self, location: Optional[str] = None, keyword: Optional[str] = None) -> Dict:
        """Return comprehensive mock restaurant data covering all major Istanbul districts"""
        
        # Beyoğlu restaurants
        beyoglu_restaurants = [
            {
                "place_id": "mock_beyoglu_1",
                "name": "Mikla Restaurant",
                "rating": 4.6,
                "price_level": 4,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0369, "lng": 28.9744}},
                "cuisine": "Turkish",
                "budget": "luxury",
                "description": "Modern Turkish cuisine with panoramic Bosphorus views and innovative interpretations of Anatolian flavors."
            },
            {
                "place_id": "mock_beyoglu_2",
                "name": "Karakoy Lokantasi",
                "rating": 4.5,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0276, "lng": 28.9441}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Elegant restaurant serving modern interpretations of Ottoman cuisine in a beautifully restored building."
            },
            {
                "place_id": "mock_beyoglu_3",
                "name": "Neolokal",
                "rating": 4.4,
                "price_level": 4,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0358, "lng": 28.9783}},
                "cuisine": "Turkish",
                "budget": "luxury",
                "description": "Contemporary Turkish cuisine showcasing local ingredients and traditional techniques with a modern twist."
            },
            {
                "place_id": "mock_beyoglu_4",
                "name": "Zubeyir Ocakbaşı",
                "rating": 4.3,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0342, "lng": 28.9756}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Authentic Turkish grill house famous for its kebabs and traditional meze selection."
            },
            {
                "place_id": "mock_beyoglu_5",
                "name": "Galata Mevlevihanesi Cafe",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0356, "lng": 28.9745}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Historic cafe in a former dervish lodge serving traditional Turkish coffee and pastries."
            },
            {
                "place_id": "mock_beyoglu_6",
                "name": "Sushi Mushi",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0334, "lng": 28.9721}},
                "cuisine": "Japanese",
                "budget": "premium",
                "description": "Fresh sushi and Japanese cuisine in the heart of Beyoğlu with authentic flavors."
            },
            {
                "place_id": "mock_beyoglu_7",
                "name": "360 Istanbul",
                "rating": 4.2,
                "price_level": 4,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0285, "lng": 28.9756}},
                "cuisine": "International",
                "budget": "luxury",
                "description": "Rooftop restaurant with 360-degree views of Istanbul and contemporary international cuisine."
            },
            {
                "place_id": "mock_beyoglu_8",
                "name": "Leb-i Derya",
                "rating": 4.1,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0298, "lng": 28.9743}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Stylish rooftop venue with Bosphorus views serving modern Turkish and international dishes."
            },
            {
                "place_id": "mock_beyoglu_9",
                "name": "Cezayir Restaurant",
                "rating": 4.0,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0312, "lng": 28.9734}},
                "cuisine": "French",
                "budget": "premium",
                "description": "Elegant French restaurant in a historic building with classic French cuisine and wine selection."
            },
            {
                "place_id": "mock_beyoglu_10",
                "name": "Ficcin Restaurant",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0327, "lng": 28.9721}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Traditional Turkish restaurant specializing in Southeastern Anatolian cuisine and spicy dishes."
            },
            {
                "place_id": "mock_beyoglu_11",
                "name": "Antiochia Concept",
                "rating": 4.3,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0341, "lng": 28.9712}},
                "cuisine": "Middle Eastern",
                "budget": "mid-range",
                "description": "Authentic Middle Eastern cuisine with specialties from Antakya region and mezze platters."
            },
            {
                "place_id": "mock_beyoglu_12",
                "name": "Galata Kitchen",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0287, "lng": 28.9724}},
                "cuisine": "International",
                "budget": "mid-range",
                "description": "Modern fusion restaurant combining Turkish and international flavors in artistic presentations."
            },
            {
                "place_id": "mock_beyoglu_13",
                "name": "Meze by Bonfilet",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0301, "lng": 28.9756}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Upscale Turkish meze restaurant with creative interpretations of traditional small plates."
            },
            {
                "place_id": "mock_beyoglu_14",
                "name": "Taksim Döner Palace",
                "rating": 4.0,
                "price_level": 1,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0367, "lng": 28.9851}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Popular döner kebab spot in Taksim serving authentic Turkish street food."
            },
            {
                "place_id": "mock_beyoglu_15",
                "name": "Sunset Terrace Galata",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0289, "lng": 28.9743}},
                "cuisine": "International",
                "budget": "premium",
                "description": "Rooftop terrace restaurant with sunset views and Mediterranean-Turkish fusion cuisine."
            },
            {
                "place_id": "mock_beyoglu_16",
                "name": "Istiklal Street Cafe",
                "rating": 3.9,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0346, "lng": 28.9781}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Historic street cafe on Istiklal serving traditional Turkish coffee and pastries."
            },
            {
                "place_id": "mock_beyoglu_17",
                "name": "Seoul Kitchen Galata",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0278, "lng": 28.9712}},
                "cuisine": "Korean",
                "budget": "mid-range",
                "description": "Authentic Korean restaurant serving bibimbap, bulgogi, and Korean BBQ."
            },
            {
                "place_id": "mock_beyoglu_18",
                "name": "Divan Pub Beyoğlu",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0323, "lng": 28.9798}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Traditional Turkish tavern with live music and classic Anatolian dishes."
            },
            {
                "place_id": "mock_beyoglu_19",
                "name": "Pizza Locale Galata",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0293, "lng": 28.9734}},
                "cuisine": "Italian",
                "budget": "mid-range",
                "description": "Authentic Italian pizzeria with wood-fired oven and imported ingredients."
            },
            {
                "place_id": "mock_beyoglu_20",
                "name": "Galata Bridge Fish Sandwich",
                "rating": 4.3,
                "price_level": 1,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0201, "lng": 28.9634}},
                "cuisine": "Seafood",
                "budget": "budget",
                "description": "Famous fish sandwich vendor on Galata Bridge serving fresh grilled fish."
            }
        ]
        
        # Sultanahmet/Fatih restaurants
        fatih_restaurants = [
            {
                "place_id": "mock_fatih_1",
                "name": "Pandeli Restaurant",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0167, "lng": 28.9708}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Historic Ottoman restaurant serving traditional Turkish cuisine since 1901 in the Grand Bazaar area."
            },
            {
                "place_id": "mock_fatih_2",
                "name": "Asitane Restaurant",
                "rating": 4.4,
                "price_level": 4,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0351, "lng": 28.9434}},
                "cuisine": "Turkish",
                "budget": "luxury",
                "description": "Exquisite Ottoman palace cuisine recreated from historical recipes, located near Chora Church."
            },
            {
                "place_id": "mock_fatih_3",
                "name": "Deraliye Ottoman Cuisine",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0290, "lng": 28.9463}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Traditional Ottoman palace cuisine served in an elegant setting near historical monuments."
            },
            {
                "place_id": "mock_fatih_4",
                "name": "Hamdi Restaurant",
                "rating": 4.5,
                "price_level": 2,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0178, "lng": 28.9708}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Famous for its lamb kebabs and traditional Turkish dishes with Golden Horn views."
            },
            {
                "place_id": "mock_fatih_5",
                "name": "Seasons Restaurant",
                "rating": 4.3,
                "price_level": 4,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0084, "lng": 28.9794}},
                "cuisine": "International",
                "budget": "luxury",
                "description": "Fine dining restaurant in Four Seasons Hotel with international and Turkish cuisine options."
            },
            {
                "place_id": "mock_fatih_6",
                "name": "Sultanahmet Köftecisi",
                "rating": 4.1,
                "price_level": 1,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0058, "lng": 28.9784}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Traditional Turkish meatball restaurant serving authentic köfte since 1920."
            },
            {
                "place_id": "mock_fatih_7",
                "name": "Balıkçı Sabahattin",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0067, "lng": 28.9812}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Historic seafood restaurant serving fresh fish and Ottoman-style preparations since 1927."
            },
            {
                "place_id": "mock_fatih_8",
                "name": "Khorasani Restaurant",
                "rating": 4.1,
                "price_level": 3,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0078, "lng": 28.9789}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Traditional Turkish restaurant near Blue Mosque serving classic Ottoman dishes."
            },
            {
                "place_id": "mock_fatih_9",
                "name": "Giritli Restaurant",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0089, "lng": 28.9801}},
                "cuisine": "Greek",
                "budget": "premium",
                "description": "Traditional Greek cuisine from Crete with fresh seafood and mezze in historic setting."
            },
            {
                "place_id": "mock_fatih_10",
                "name": "Rumeli Cafe",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0073, "lng": 28.9796}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Cozy cafe serving Turkish breakfast, coffee, and light meals near Hagia Sophia."
            },
            {
                "place_id": "mock_fatih_11",
                "name": "Sarnıç Restaurant",
                "rating": 4.2,
                "price_level": 4,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0081, "lng": 28.9803}},
                "cuisine": "Turkish",
                "budget": "luxury",
                "description": "Unique dining experience in a restored Byzantine cistern with atmospheric lighting."
            },
            {
                "place_id": "mock_fatih_12",
                "name": "Medusa Restaurant",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0076, "lng": 28.9787}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Traditional Turkish restaurant with rooftop terrace offering views of Blue Mosque."
            },
            {
                "place_id": "mock_fatih_13",
                "name": "Grand Bazaar Spice House",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0104, "lng": 28.9681}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Spice merchant and restaurant in Grand Bazaar serving Ottoman-spiced dishes."
            },
            {
                "place_id": "mock_fatih_14",
                "name": "Byzantine Restaurant",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0087, "lng": 28.9798}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Historic restaurant near Hagia Sophia specializing in Byzantine-era recipes."
            },
            {
                "place_id": "mock_fatih_15",
                "name": "Eminönü Fish Market",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0188, "lng": 28.9723}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Traditional fish market restaurant with daily fresh catch from Bosphorus."
            },
            {
                "place_id": "mock_fatih_16",
                "name": "Topkapi Palace Cafe",
                "rating": 3.9,
                "price_level": 2,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0115, "lng": 28.9834}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Museum cafe serving traditional Ottoman palace refreshments."
            },
            {
                "place_id": "mock_fatih_17",
                "name": "Sultanahmet Meatball House",
                "rating": 4.3,
                "price_level": 1,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0062, "lng": 28.9776}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Famous local meatball restaurant serving handmade köfte since 1950."
            },
            {
                "place_id": "mock_fatih_18",
                "name": "Çemberlitaş Hamam Restaurant",
                "rating": 4.0,
                "price_level": 3,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0098, "lng": 28.9712}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Restaurant adjacent to historic bathhouse serving traditional post-bath meals."
            },
            {
                "place_id": "mock_fatih_19",
                "name": "Golden Horn View Restaurant",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0201, "lng": 28.9756}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Panoramic restaurant overlooking Golden Horn with Ottoman cuisine."
            },
            {
                "place_id": "mock_fatih_20",
                "name": "Historic Peninsula Cafe",
                "rating": 3.8,
                "price_level": 2,
                "vicinity": "Sultanahmet, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0069, "lng": 28.9823}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Cozy cafe in historic peninsula serving Turkish breakfast and light meals."
            }
        ]
        
        # Kadıköy restaurants
        kadikoy_restaurants = [
            {
                "place_id": "mock_kadikoy_1",
                "name": "Çiya Sofrası",
                "rating": 4.5,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9925, "lng": 29.0315}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Authentic Anatolian cuisine with regional specialties from across Turkey and traditional home cooking."
            },
            {
                "place_id": "mock_kadikoy_2",
                "name": "Ciya Kebap",
                "rating": 4.4,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9928, "lng": 29.0318}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Sister restaurant to Çiya Sofrası, specializing in traditional kebabs and grilled meats."
            },
            {
                "place_id": "mock_kadikoy_3",
                "name": "Kanaat Lokantası",
                "rating": 4.3,
                "price_level": 1,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9889, "lng": 29.0301}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Historic local eatery serving traditional Turkish comfort food since 1933."
            },
            {
                "place_id": "mock_kadikoy_4",
                "name": "Kadıköy Fish Market Restaurant",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9898, "lng": 29.0289}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Fresh seafood restaurant in the heart of Kadıköy's famous fish market."
            },
            {
                "place_id": "mock_kadikoy_5",
                "name": "Yanyalı Fehmi Lokantası",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9912, "lng": 29.0298}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Traditional Anatolian restaurant famous for its lamb dishes and regional specialties."
            },
            {
                "place_id": "mock_kadikoy_6",
                "name": "Çiya Mutfak",
                "rating": 4.3,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9931, "lng": 29.0321}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Third branch of famous Çiya restaurants focusing on traditional Turkish desserts and sweets."
            },
            {
                "place_id": "mock_kadikoy_7",
                "name": "Tarihi Moda İskelesi",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9854, "lng": 29.0267}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Waterfront seafood restaurant with Bosphorus views and fresh daily catch."
            },
            {
                "place_id": "mock_kadikoy_8",
                "name": "Pandora Bookstore Cafe",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 40.9903, "lng": 29.0285}},
                "cuisine": "International",
                "budget": "mid-range",
                "description": "Cozy bookstore cafe serving international dishes, coffee, and hosting cultural events."
            },
            {
                "place_id": "mock_kadikoy_9",
                "name": "Kiva Han",
                "rating": 4.1,
                "price_level": 1,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9917, "lng": 29.0293}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Traditional Turkish breakfast and tea house popular with locals and students."
            },
            {
                "place_id": "mock_kadikoy_10",
                "name": "Sade Kahve",
                "rating": 4.3,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 40.9896, "lng": 29.0312}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Specialty coffee roastery and cafe with artisanal Turkish coffee and light meals."
            },
            {
                "place_id": "mock_kadikoy_11",
                "name": "Fenerbahçe Fish Restaurant",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9823, "lng": 29.0234}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Waterfront seafood restaurant in Fenerbahçe with Marmara Sea views."
            },
            {
                "place_id": "mock_kadikoy_12",
                "name": "Moda Park Cafe",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 40.9867, "lng": 29.0289}},
                "cuisine": "International",
                "budget": "mid-range",
                "description": "Park cafe with outdoor seating serving international breakfast and brunch."
            },
            {
                "place_id": "mock_kadikoy_13",
                "name": "Süreyya Opera Cafe",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 40.9889, "lng": 29.0267}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Opera house cafe serving pre-show dinners and cultural event catering."
            },
            {
                "place_id": "mock_kadikoy_14",
                "name": "Asian Side Kebab House",
                "rating": 4.3,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9934, "lng": 29.0298}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Traditional kebab house on Asian side serving Adana and Urfa specialties."
            },
            {
                "place_id": "mock_kadikoy_15",
                "name": "Bahariye Street Bistro",
                "rating": 3.9,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9901, "lng": 29.0278}},
                "cuisine": "International",
                "budget": "mid-range",
                "description": "Trendy bistro on shopping street serving fusion cuisine and craft cocktails."
            },
            {
                "place_id": "mock_kadikoy_16",
                "name": "Haydarpaşa Terminal Restaurant",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9987, "lng": 29.0167}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Historic train station restaurant serving traditional Turkish railway cuisine."
            },
            {
                "place_id": "mock_kadikoy_17",
                "name": "Bostancı Seaside Grill",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9656, "lng": 29.0589}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Seaside grill restaurant serving barbecued meats and seafood with sea views."
            },
            {
                "place_id": "mock_kadikoy_18",
                "name": "Kadıköy Patisserie",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 40.9878, "lng": 29.0295}},
                "cuisine": "French",
                "budget": "mid-range",
                "description": "French patisserie and cafe serving authentic pastries, croissants, and coffee."
            },
            {
                "place_id": "mock_kadikoy_19",
                "name": "Nostaljik Tramvay Cafe",
                "rating": 3.8,
                "price_level": 1,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 40.9845, "lng": 29.0345}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Nostalgic tram-themed cafe serving Turkish tea, coffee, and light snacks."
            }
        ]
        
        # Beşiktaş restaurants
        besiktas_restaurants = [
            {
                "place_id": "mock_besiktas_1",
                "name": "Feriye Palace Restaurant",
                "rating": 4.4,
                "price_level": 4,
                "vicinity": "Beşiktaş, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0425, "lng": 29.0058}},
                "cuisine": "Ottoman",
                "budget": "luxury",
                "description": "Elegant Ottoman cuisine in a restored 19th-century palace with Bosphorus views."
            },
            {
                "place_id": "mock_besiktas_2",
                "name": "Beşiktaş Balık Pazarı",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Beşiktaş, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0427, "lng": 29.0067}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Traditional fish market restaurant serving fresh seafood and meze."
            },
            {
                "place_id": "mock_besiktas_3",
                "name": "Ortaköy Kumpir",
                "rating": 4.0,
                "price_level": 1,
                "vicinity": "Beşiktaş, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0473, "lng": 29.0268}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Famous street food spot serving loaded baked potatoes with various toppings."
            },
            {
                "place_id": "mock_besiktas_4",
                "name": "House Cafe Ortaköy",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Beşiktaş, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0467, "lng": 29.0254}},
                "cuisine": "International",
                "budget": "mid-range",
                "description": "Popular cafe chain with Bosphorus views serving international cuisine and coffee."
            },
            {
                "place_id": "mock_besiktas_5",
                "name": "Dönerci Şahin Usta",
                "rating": 4.2,
                "price_level": 1,
                "vicinity": "Beşiktaş, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0432, "lng": 29.0071}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Local favorite döner kebab shop known for high-quality meat and fresh bread."
            },
            {
                "place_id": "mock_besiktas_6",
                "name": "Muzedechanga",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Beşiktaş, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0429, "lng": 29.0089}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Sophisticated Turkish restaurant in Santralistanbul with modern interpretations of classics."
            },
            {
                "place_id": "mock_besiktas_7",
                "name": "Blackk Coffee",
                "rating": 4.4,
                "price_level": 2,
                "vicinity": "Beşiktaş, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0441, "lng": 29.0078}},
                "cuisine": "International",
                "budget": "mid-range",
                "description": "Specialty coffee shop with third-wave coffee culture and light international meals."
            },
            {
                "place_id": "mock_besiktas_8",
                "name": "Dolmabahçe Sarayı Cafe",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Beşiktaş, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0391, "lng": 29.0007}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Historic palace cafe serving traditional Turkish refreshments and light meals."
            }
        ]
        
        # Üsküdar restaurants
        uskudar_restaurants = [
            {
                "place_id": "mock_uskudar_1",
                "name": "Kanaat Lokantası Üsküdar",
                "rating": 4.3,
                "price_level": 2,
                "vicinity": "Üsküdar, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0214, "lng": 29.0144}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Traditional Turkish home-style cooking in historic Üsküdar with Bosphorus views."
            },
            {
                "place_id": "mock_uskudar_2",
                "name": "Çamlıca Hill Restaurant",
                "rating": 4.4,
                "price_level": 3,
                "vicinity": "Üsküdar, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0086, "lng": 29.0661}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Panoramic restaurant on Çamlıca Hill offering stunning city views and traditional cuisine."
            },
            {
                "place_id": "mock_uskudar_3",
                "name": "Maiden's Tower Restaurant",
                "rating": 4.5,
                "price_level": 4,
                "vicinity": "Üsküdar, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0212, "lng": 29.0044}},
                "cuisine": "International",
                "budget": "luxury",
                "description": "Unique dining experience on the historic Maiden's Tower with 360-degree Bosphorus views."
            },
            {
                "place_id": "mock_uskudar_4",
                "name": "Hidiv Kasrı Restaurant",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Üsküdar, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0798, "lng": 29.0834}},
                "cuisine": "Ottoman",
                "budget": "premium",
                "description": "Historic Ottoman pavilion restaurant with period decor and traditional palace cuisine."
            },
            {
                "place_id": "mock_uskudar_5",
                "name": "İskele Restaurant",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Üsküdar, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0223, "lng": 29.0167}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Waterfront restaurant at Üsküdar ferry terminal with fresh fish and Bosphorus views."
            },
            {
                "place_id": "mock_uskudar_6",
                "name": "Şemsi Paşa Cafe",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Üsküdar, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0201, "lng": 29.0151}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Peaceful cafe next to historic mosque serving Turkish coffee and traditional sweets."
            },
            {
                "place_id": "mock_uskudar_7",
                "name": "Beylerbeyi Balık Restaurant",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Üsküdar, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0412, "lng": 29.0412}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Elegant seafood restaurant near Beylerbeyi Palace with Ottoman-era atmosphere."
            }
        ]
        
        # Sarıyer restaurants
        sariyer_restaurants = [
            {
                "place_id": "mock_sariyer_1",
                "name": "Uskumru Restaurant",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1058, "lng": 29.0534}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Waterfront seafood restaurant in Sarıyer known for fresh fish and Black Sea specialties."
            },
            {
                "place_id": "mock_sariyer_2",
                "name": "Yeniköy Balık Restaurant",
                "rating": 4.4,
                "price_level": 3,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1121, "lng": 29.0498}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Historic fish restaurant in Yeniköy with traditional Turkish seafood dishes and Bosphorus views."
            },
            {
                "place_id": "mock_sariyer_3",
                "name": "Tarabya Bay Restaurant",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1167, "lng": 29.0567}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Casual dining by Tarabya Bay with fresh seafood and traditional Turkish meze."
            },
            {
                "place_id": "mock_sariyer_4",
                "name": "Les Ambassadeurs Restaurant",
                "rating": 4.5,
                "price_level": 4,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1089, "lng": 29.0521}},
                "cuisine": "French",
                "budget": "luxury",
                "description": "Upscale French restaurant in a historic Bosphorus mansion with elegant atmosphere."
            },
            {
                "place_id": "mock_sariyer_5",
                "name": "Kilyos Beach Restaurant",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.2345, "lng": 29.0456}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Beachfront restaurant on Black Sea coast serving fresh seafood and grilled fish."
            },
            {
                "place_id": "mock_sariyer_6",
                "name": "Büyükdere Balık Lokantası",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1234, "lng": 29.0612}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Local fish restaurant popular with families, serving traditional Turkish seafood dishes."
            },
            {
                "place_id": "mock_sariyer_7",
                "name": "Emirgan Park Restaurant",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1098, "lng": 29.0456}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Park restaurant famous for tulip season dining with Ottoman palace cuisine."
            },
            {
                "place_id": "mock_sariyer_8",
                "name": "Rumeli Fortress Cafe",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0834, "lng": 29.0567}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Historic fortress cafe with Bosphorus views serving traditional Turkish refreshments."
            },
            {
                "place_id": "mock_sariyer_9",
                "name": "Sarıyer Börek House",
                "rating": 4.2,
                "price_level": 1,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1123, "lng": 29.0534}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Traditional börek bakery and cafe serving flaky pastries and Turkish tea."
            },
            {
                "place_id": "mock_sariyer_10",
                "name": "Belgrad Forest Restaurant",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1456, "lng": 28.9987}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Forest restaurant popular for weekend getaways serving grilled meats and kebabs."
            },
            {
                "place_id": "mock_sariyer_11",
                "name": "Istinye Bay Seafood",
                "rating": 4.4,
                "price_level": 4,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1087, "lng": 29.0423}},
                "cuisine": "Seafood",
                "budget": "luxury",
                "description": "Upscale seafood restaurant in marina with premium fish and Bosphorus views."
            },
            {
                "place_id": "mock_sariyer_12",
                "name": "Maçka Restaurant",
                "rating": 4.1,
                "price_level": 3,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0956, "lng": 29.0398}},
                "cuisine": "International",
                "budget": "premium",
                "description": "International cuisine restaurant with park views and seasonal menu."
            },
            {
                "place_id": "mock_sariyer_13",
                "name": "Bosphorus Bridge View Restaurant",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0723, "lng": 29.0445}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Panoramic restaurant with bridge views serving modern Turkish cuisine."
            },
            {
                "place_id": "mock_sariyer_14",
                "name": "Çayır Restaurant",
                "rating": 3.9,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1234, "lng": 29.0234}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Family-run restaurant in meadow setting serving home-style Turkish cooking."
            },
            {
                "place_id": "mock_sariyer_15",
                "name": "Fishing Village Restaurant",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1445, "lng": 29.0678}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Authentic fishing village restaurant with daily catch and traditional preparations."
            }
        ]
        
        # Şişli restaurants
        sisli_restaurants = [
            {
                "place_id": "mock_sisli_1",
                "name": "Nusr-Et Steakhouse",
                "rating": 4.3,
                "price_level": 4,
                "vicinity": "Şişli, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0498, "lng": 28.9876}},
                "cuisine": "International",
                "budget": "luxury",
                "description": "Famous steakhouse known for premium cuts and theatrical presentation by celebrity chef."
            },
            {
                "place_id": "mock_sisli_2",
                "name": "Ciya Sofrası Şişli",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Şişli, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0487, "lng": 28.9854}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Branch of famous Anatolian restaurant bringing regional Turkish specialties to Şişli."
            },
            {
                "place_id": "mock_sisli_3",
                "name": "Seasons Restaurant Şişli",
                "rating": 4.1,
                "price_level": 3,
                "vicinity": "Şişli, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0512, "lng": 28.9889}},
                "cuisine": "International",
                "budget": "premium",
                "description": "Upscale international cuisine restaurant in luxury hotel with seasonal menu."
            },
            {
                "place_id": "mock_sisli_4",
                "name": "Hacı Abdullah",
                "rating": 4.4,
                "price_level": 2,
                "vicinity": "Şişli, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0523, "lng": 28.9901}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Historic Ottoman restaurant serving traditional Turkish cuisine since 1888."
            },
            {
                "place_id": "mock_sisli_5",
                "name": "Pera Thai",
                "rating": 4.0,
                "price_level": 3,
                "vicinity": "Şişli, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0467, "lng": 28.9832}},
                "cuisine": "Thai",
                "budget": "premium",
                "description": "Authentic Thai restaurant with traditional recipes and modern presentation."
            },
            {
                "place_id": "mock_sisli_6",
                "name": "İmam Çağdaş",
                "rating": 4.2,
                "price_level": 1,
                "vicinity": "Şişli, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0534, "lng": 28.9876}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Famous kebab restaurant known for Gaziantep-style specialties and baklava."
            }
        ]
        
        # Bakırköy restaurants
        bakirkoy_restaurants = [
            {
                "place_id": "mock_bakirkoy_1",
                "name": "Sunset Grill & Bar",
                "rating": 4.2,
                "price_level": 4,
                "vicinity": "Bakırköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9723, "lng": 28.8634}},
                "cuisine": "International",
                "budget": "luxury",
                "description": "Upscale restaurant with panoramic sea views and international fusion cuisine."
            },
            {
                "place_id": "mock_bakirkoy_2",
                "name": "Florya Balık Restaurant",
                "rating": 4.1,
                "price_level": 3,
                "vicinity": "Bakırköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9798, "lng": 28.7934}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Seaside seafood restaurant with fresh catch and beautiful Marmara Sea views."
            },
            {
                "place_id": "mock_bakirkoy_3",
                "name": "Yeşilköy Marina Restaurant",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Bakırköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9656, "lng": 28.8123}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Marina restaurant serving Turkish cuisine with marina and sea views."
            },
            {
                "place_id": "mock_bakirkoy_4",
                "name": "Capacity Shopping Center Food Court",
                "rating": 3.8,
                "price_level": 1,
                "vicinity": "Bakırköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9823, "lng": 28.8745}},
                "cuisine": "International",
                "budget": "budget",
                "description": "Modern food court with diverse cuisine options and family-friendly atmosphere."
            }
        ]
        
        # Levent/Maslak restaurants
        levent_restaurants = [
            {
                "place_id": "mock_levent_1",
                "name": "Ulus 29",
                "rating": 4.4,
                "price_level": 4,
                "vicinity": "Levent, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0821, "lng": 29.0167}},
                "cuisine": "International",
                "budget": "luxury",
                "description": "Sophisticated restaurant with Bosphorus views and contemporary international cuisine."
            },
            {
                "place_id": "mock_levent_2",
                "name": "Sakhalin Restaurant",
                "rating": 4.2,
                "price_level": 4,
                "vicinity": "Levent, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0834, "lng": 29.0189}},
                "cuisine": "Russian",
                "budget": "luxury",
                "description": "Upscale Russian restaurant with authentic cuisine and elegant atmosphere."
            },
            {
                "place_id": "mock_levent_3",
                "name": "Business Lunch Café",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Levent, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0798, "lng": 29.0143}},
                "cuisine": "International",
                "budget": "mid-range",
                "description": "Modern cafe popular with business professionals, serving quick international meals."
            },
            {
                "place_id": "mock_levent_4",
                "name": "Zuma Istanbul",
                "rating": 4.5,
                "price_level": 4,
                "vicinity": "Levent, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0845, "lng": 29.0198}},
                "cuisine": "Japanese",
                "budget": "luxury",
                "description": "High-end Japanese restaurant with contemporary sushi and robatayaki cuisine."
            }
        ]
        
        # Additional Beyoğlu restaurants for expanded coverage
        additional_beyoglu = [
            {
                "place_id": "mock_beyoglu_extra_1",
                "name": "Pandeli Restaurant",
                "rating": 4.4,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0365, "lng": 28.9751}},
                "cuisine": "Ottoman",
                "budget": "premium",
                "description": "Historic Ottoman cuisine restaurant with traditional recipes and elegant atmosphere since 1901."
            },
            {
                "place_id": "mock_beyoglu_extra_2",
                "name": "Salt Bae Nusr-Et Steakhouse",
                "rating": 4.0,
                "price_level": 4,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0345, "lng": 28.9768}},
                "cuisine": "International",
                "budget": "luxury",
                "description": "Famous steakhouse known for premium cuts and theatrical presentation."
            },
            {
                "place_id": "mock_beyoglu_extra_3",
                "name": "Galata House Restaurant",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0289, "lng": 28.9721}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Traditional Turkish restaurant in a historic Galata building with live music."
            },
            {
                "place_id": "mock_beyoglu_extra_4",
                "name": "Cafe Central",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0378, "lng": 28.9734}},
                "cuisine": "International",
                "budget": "mid-range",
                "description": "Historic European-style cafe serving light meals and excellent coffee since 1950s."
            },
            {
                "place_id": "mock_beyoglu_extra_5",
                "name": "Helvetia Lokantası",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0356, "lng": 28.9789}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Traditional meyhane serving authentic Turkish meze and rakı in historic setting."
            },
            {
                "place_id": "mock_beyoglu_extra_6",
                "name": "Teppanyaki Ginza",
                "rating": 4.4,
                "price_level": 4,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0323, "lng": 28.9756}},
                "cuisine": "Japanese",
                "budget": "luxury",
                "description": "Premium Japanese teppanyaki restaurant with live cooking and fresh ingredients."
            },
            {
                "place_id": "mock_beyoglu_extra_7",
                "name": "Bosphorus Terrace",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0387, "lng": 28.9823}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Rooftop seafood restaurant with stunning Bosphorus views and fresh daily catch."
            },
            {
                "place_id": "mock_beyoglu_extra_8",
                "name": "Bistro Francais",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0367, "lng": 28.9712}},
                "cuisine": "French",
                "budget": "premium",
                "description": "Authentic French bistro with classic dishes and extensive wine selection."
            },
            {
                "place_id": "mock_beyoglu_extra_9",
                "name": "Meyhane Asmalımescit",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0342, "lng": 28.9743}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Lively traditional meyhane with live Turkish folk music and authentic meze."
            },
            {
                "place_id": "mock_beyoglu_extra_10",
                "name": "Koreli Mutfağı",
                "rating": 4.1,
                "price_level": 2,
                "vicinity": "Beyoğlu, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0334, "lng": 28.9756}},
                "cuisine": "Korean",
                "budget": "mid-range",
                "description": "Authentic Korean restaurant with traditional BBQ and kimchi dishes."
            }
        ]
        
        # Additional Fatih/Sultanahmet restaurants
        additional_fatih = [
            {
                "place_id": "mock_fatih_extra_1",
                "name": "Sultanahmet Fish House",
                "rating": 4.5,
                "price_level": 3,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0054, "lng": 28.9768}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Traditional fish restaurant near Blue Mosque with fresh Bosphorus seafood."
            },
            {
                "place_id": "mock_fatih_extra_2",
                "name": "Cooking Alaturka",
                "rating": 4.6,
                "price_level": 2,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0089, "lng": 28.9756}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Traditional Turkish cooking school and restaurant with hands-on dining experience."
            },
            {
                "place_id": "mock_fatih_extra_3",
                "name": "Aya Sofya Kebab House",
                "rating": 4.2,
                "price_level": 2,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0086, "lng": 28.9802}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Family-run kebab restaurant serving authentic grilled meats near Hagia Sophia."
            },
            {
                "place_id": "mock_fatih_extra_4",
                "name": "Ottoman Kitchen",
                "rating": 4.4,
                "price_level": 4,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0076, "lng": 28.9789}},
                "cuisine": "Ottoman",
                "budget": "luxury",
                "description": "Upscale Ottoman cuisine restaurant with historical recipes and royal ambiance."
            },
            {
                "place_id": "mock_fatih_extra_5",
                "name": "Sirkeci Balık Lokantası",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0134, "lng": 28.9743}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Historic fish restaurant near Sirkeci Station with traditional preparation methods."
            },
            {
                "place_id": "mock_fatih_extra_6",
                "name": "Spice Bazaar Cafe",
                "rating": 4.0,
                "price_level": 1,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0167, "lng": 28.9734}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Small cafe inside Spice Bazaar serving Turkish coffee and traditional sweets."
            },
            {
                "place_id": "mock_fatih_extra_7",
                "name": "Baklava Sarayı",
                "rating": 4.1,
                "price_level": 1,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "dessert", "establishment"],
                "geometry": {"location": {"lat": 41.0098, "lng": 28.9723}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Traditional baklava shop and cafe with handmade Ottoman desserts."
            },
            {
                "place_id": "mock_fatih_extra_8",
                "name": "Golden Horn Restaurant",
                "rating": 4.3,
                "price_level": 3,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0145, "lng": 28.9678}},
                "cuisine": "Turkish",
                "budget": "premium",
                "description": "Elegant restaurant overlooking Golden Horn with modern Turkish cuisine."
            },
            {
                "place_id": "mock_fatih_extra_9",
                "name": "Topkapi Palace Cafe",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.0115, "lng": 28.9834}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Museum cafe near Topkapi Palace serving light meals and Turkish tea."
            },
            {
                "place_id": "mock_fatih_extra_10",
                "name": "Byzantine Tavern",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Fatih, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.0067, "lng": 28.9712}},
                "cuisine": "Greek",
                "budget": "premium",
                "description": "Historic tavern serving Greek and Byzantine-inspired dishes with live music."
            }
        ]
        
        # Additional Kadıköy restaurants
        additional_kadikoy = [
            {
                "place_id": "mock_kadikoy_extra_1",
                "name": "Kadıköy Fish Market Restaurant",
                "rating": 4.4,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9910, "lng": 29.0243}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Fresh seafood restaurant in the heart of Kadıköy fish market with daily catches."
            },
            {
                "place_id": "mock_kadikoy_extra_2",
                "name": "Çiya Sofrası",
                "rating": 4.6,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9902, "lng": 29.0267}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Famous restaurant preserving Anatolian cuisine traditions with regional specialties."
            },
            {
                "place_id": "mock_kadikoy_extra_3",
                "name": "Bağdat Caddesi Steakhouse",
                "rating": 4.3,
                "price_level": 4,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9634, "lng": 29.0823}},
                "cuisine": "International",
                "budget": "luxury",
                "description": "Upscale steakhouse on famous Bağdat Avenue with premium cuts and wine selection."
            },
            {
                "place_id": "mock_kadikoy_extra_4",
                "name": "Moda Pier Restaurant",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9834, "lng": 29.0456}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Seaside restaurant on Moda Pier with panoramic sea views and fresh fish."
            },
            {
                "place_id": "mock_kadikoy_extra_5",
                "name": "Haydarpaşa Terminal Cafe",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 40.9945, "lng": 29.0178}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Historic train station cafe with traditional Turkish breakfast and nostalgic atmosphere."
            },
            {
                "place_id": "mock_kadikoy_extra_6",
                "name": "Thai Garden Kadıköy",
                "rating": 4.1,
                "price_level": 3,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9887, "lng": 29.0298}},
                "cuisine": "Thai",
                "budget": "premium",
                "description": "Authentic Thai restaurant with traditional spices and Asian garden atmosphere."
            },
            {
                "place_id": "mock_kadikoy_extra_7",
                "name": "Fenerbahçe Stadyum Restaurant",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9890, "lng": 29.0543}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Sports-themed restaurant near Fenerbahçe Stadium with Turkish grill specialties."
            },
            {
                "place_id": "mock_kadikoy_extra_8",
                "name": "Asian Side Sushi",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9923, "lng": 29.0312}},
                "cuisine": "Japanese",
                "budget": "premium",
                "description": "Contemporary sushi restaurant on Asian side with fresh fish and creative rolls."
            },
            {
                "place_id": "mock_kadikoy_extra_9",
                "name": "Bostancı Marina Restaurant",
                "rating": 4.3,
                "price_level": 4,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9456, "lng": 29.0987}},
                "cuisine": "Seafood",
                "budget": "luxury",
                "description": "Marina restaurant with yacht views and premium seafood in elegant setting."
            },
            {
                "place_id": "mock_kadikoy_extra_10",
                "name": "Street Food Kadıköy",
                "rating": 3.9,
                "price_level": 1,
                "vicinity": "Kadıköy, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 40.9912, "lng": 29.0256}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Popular street food stand with döner, simit, and traditional Turkish snacks."
            }
        ]
        
        # Additional Sarıyer restaurants
        additional_sariyer = [
            {
                "place_id": "mock_sariyer_extra_1",
                "name": "Sarıyer Balıkçısı",
                "rating": 4.5,
                "price_level": 3,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1456, "lng": 29.0534}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Traditional fish restaurant in Sarıyer village with boat-fresh catches and Bosphorus views."
            },
            {
                "place_id": "mock_sariyer_extra_2",
                "name": "Kilyos Beach Restaurant",
                "rating": 4.2,
                "price_level": 3,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.2156, "lng": 29.0423}},
                "cuisine": "Seafood",
                "budget": "premium",
                "description": "Beachside restaurant on Black Sea coast with fresh seafood and summer atmosphere."
            },
            {
                "place_id": "mock_sariyer_extra_3",
                "name": "Tarabya Bay Restaurant",
                "rating": 4.4,
                "price_level": 4,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1234, "lng": 29.0654}},
                "cuisine": "Turkish",
                "budget": "luxury",
                "description": "Elegant Bosphorus-side restaurant in Tarabya with Ottoman palace views and fine dining."
            },
            {
                "place_id": "mock_sariyer_extra_4",
                "name": "Büyükdere Köftecisi",
                "rating": 4.1,
                "price_level": 1,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1123, "lng": 29.0587}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Local favorite for authentic Turkish meatballs and traditional home cooking."
            },
            {
                "place_id": "mock_sariyer_extra_5",
                "name": "Belgrad Forest Cafe",
                "rating": 4.0,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.1890, "lng": 28.9876}},
                "cuisine": "Turkish",
                "budget": "mid-range",
                "description": "Forest cafe popular with hikers serving Turkish breakfast and nature views."
            },
            {
                "place_id": "mock_sariyer_extra_6",
                "name": "Yeniköy Mansion Restaurant",
                "rating": 4.6,
                "price_level": 4,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1298, "lng": 29.0712}},
                "cuisine": "Ottoman",
                "budget": "luxury",
                "description": "Historic waterfront mansion restaurant with authentic Ottoman cuisine and Bosphorus views."
            },
            {
                "place_id": "mock_sariyer_extra_7",
                "name": "Rumeli Kavağı Fish Restaurant",
                "rating": 4.3,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1756, "lng": 29.0823}},
                "cuisine": "Seafood",
                "budget": "mid-range",
                "description": "Village fish restaurant at the mouth of Bosphorus with simple, fresh preparations."
            },
            {
                "place_id": "mock_sariyer_extra_8",
                "name": "Emirgan Park Cafe",
                "rating": 3.9,
                "price_level": 2,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.1087, "lng": 29.0534}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Park cafe famous during tulip season with garden seating and light meals."
            },
            {
                "place_id": "mock_sariyer_extra_9",
                "name": "Istinye Marina Club",
                "rating": 4.4,
                "price_level": 4,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "food", "establishment"],
                "geometry": {"location": {"lat": 41.1156, "lng": 29.0645}},
                "cuisine": "International",
                "budget": "luxury",
                "description": "Exclusive marina club restaurant with yacht harbor views and international cuisine."
            },
            {
                "place_id": "mock_sariyer_extra_10",
                "name": "Anadolu Kavağı Castle Cafe",
                "rating": 4.0,
                "price_level": 1,
                "vicinity": "Sarıyer, Istanbul",
                "types": ["restaurant", "cafe", "establishment"],
                "geometry": {"location": {"lat": 41.1823, "lng": 29.0912}},
                "cuisine": "Turkish",
                "budget": "budget",
                "description": "Historic castle cafe with panoramic Black Sea views and traditional Turkish tea."
            }
        ]
        
        # Compile all restaurants including additional ones
        all_restaurants = (beyoglu_restaurants + fatih_restaurants + kadikoy_restaurants + 
                          besiktas_restaurants + uskudar_restaurants + sariyer_restaurants +
                          sisli_restaurants + bakirkoy_restaurants + levent_restaurants +
                          additional_beyoglu + additional_fatih + additional_kadikoy + additional_sariyer)
        
        # Filter by location if provided
        if location:
            location_lower = location.lower()
            if 'beyoğlu' in location_lower or 'beyoglu' in location_lower or 'galata' in location_lower or 'taksim' in location_lower:
                selected_restaurants = beyoglu_restaurants + additional_beyoglu
            elif 'fatih' in location_lower or 'sultanahmet' in location_lower or 'eminönü' in location_lower:
                selected_restaurants = fatih_restaurants + additional_fatih
            elif 'kadıköy' in location_lower or 'kadikoy' in location_lower:
                selected_restaurants = kadikoy_restaurants + additional_kadikoy
            elif 'beşiktaş' in location_lower or 'besiktas' in location_lower or 'ortaköy' in location_lower or 'ortakoy' in location_lower:
                selected_restaurants = besiktas_restaurants
            elif 'üsküdar' in location_lower or 'uskudar' in location_lower:
                selected_restaurants = uskudar_restaurants
            elif 'sarıyer' in location_lower or 'sariyer' in location_lower:
                selected_restaurants = sariyer_restaurants + additional_sariyer
            elif 'şişli' in location_lower or 'sisli' in location_lower:
                selected_restaurants = sisli_restaurants
            elif 'bakırköy' in location_lower or 'bakirkoy' in location_lower or 'florya' in location_lower:
                selected_restaurants = bakirkoy_restaurants
            elif 'levent' in location_lower or 'maslak' in location_lower:
                selected_restaurants = levent_restaurants
            else:
                selected_restaurants = all_restaurants
        else:
            selected_restaurants = all_restaurants
        
        # Filter by keyword if provided
        if keyword:
            keyword_lower = keyword.lower()
            filtered_restaurants = []
            for r in selected_restaurants:
                # Check name, description, cuisine, and budget
                if (keyword_lower in r["name"].lower() or 
                    keyword_lower in r["description"].lower() or
                    keyword_lower in r.get("cuisine", "").lower() or
                    keyword_lower in r.get("budget", "").lower() or
                    # Check for cuisine types
                    ('turkish' in keyword_lower and r.get("cuisine") == "Turkish") or
                    ('japanese' in keyword_lower and r.get("cuisine") == "Japanese") or
                    ('seafood' in keyword_lower and r.get("cuisine") == "Seafood") or
                    ('ottoman' in keyword_lower and r.get("cuisine") == "Ottoman") or
                    ('french' in keyword_lower and r.get("cuisine") == "French") or
                    ('international' in keyword_lower and r.get("cuisine") == "International") or
                    ('greek' in keyword_lower and r.get("cuisine") == "Greek") or
                    ('thai' in keyword_lower and r.get("cuisine") == "Thai") or
                    ('russian' in keyword_lower and r.get("cuisine") == "Russian") or
                    ('middle eastern' in keyword_lower and r.get("cuisine") == "Middle Eastern") or
                    ('korean' in keyword_lower and r.get("cuisine") == "Korean") or
                    ('italian' in keyword_lower and r.get("cuisine") == "Italian") or
                    # Check for budget types - improved mapping
                    ('cheap' in keyword_lower and r.get("budget") in ["budget", "mid-range"]) or
                    ('expensive' in keyword_lower and r.get("budget") in ["luxury", "premium"]) or
                    ('pricey' in keyword_lower and r.get("budget") in ["luxury", "premium"]) or
                    ('upscale' in keyword_lower and r.get("budget") in ["luxury", "premium"]) or
                    ('high-end' in keyword_lower and r.get("budget") in ["luxury", "premium"]) or
                    ('luxury' in keyword_lower and r.get("budget") == "luxury") or
                    ('premium' in keyword_lower and r.get("budget") == "premium") or
                    ('budget' in keyword_lower and r.get("budget") == "budget") or
                    ('affordable' in keyword_lower and r.get("budget") in ["budget", "mid-range"]) or
                    # Dietary restrictions - improved vegan detection
                    ('vegetarian' in keyword_lower and (r.get("cuisine") in ["Turkish", "International", "Greek"] or 
                     'vegetarian' in r["description"].lower() or 'veggie' in r["description"].lower())) or
                    ('vegan' in keyword_lower and (r.get("cuisine") in ["Turkish", "International"] or 
                     'vegan' in r["description"].lower() or 'plant-based' in r["description"].lower() or
                     ('meze' in r["description"].lower() and r.get("cuisine") in ["Turkish", "Greek"]) or
                     ('salad' in r["description"].lower() and r.get("cuisine") == "International"))) or
                    ('halal' in keyword_lower and r.get("cuisine") in ["Turkish", "Ottoman", "Middle Eastern"]) or
                    ('kosher' in keyword_lower and r.get("cuisine") in ["International"]) or
                    ('gluten-free' in keyword_lower and r.get("cuisine") in ["Turkish", "International", "Seafood"]) or
                    # Food type keywords - improved coffee detection
                    ('fish' in keyword_lower and r.get("cuisine") == "Seafood") or
                    ('kebab' in keyword_lower and 'kebab' in r["description"].lower()) or
                    ('döner' in keyword_lower and 'döner' in r["description"].lower()) or
                    ('pizza' in keyword_lower and 'pizza' in r["description"].lower()) or
                    ('steak' in keyword_lower and 'steak' in r["description"].lower()) or
                    ('coffee' in keyword_lower and ('cafe' in r.get("types", []) or 'coffee' in r["description"].lower() or
                     'coffee' in r["name"].lower() or 'cafe' in r["name"].lower() or 'espresso' in r["description"].lower())) or
                    ('cafe' in keyword_lower and ('cafe' in r.get("types", []) or 'cafe' in r["name"].lower() or
                     'coffee' in r["description"].lower())) or
                    ('breakfast' in keyword_lower and 'breakfast' in r["description"].lower()) or
                    ('street' in keyword_lower and r.get("budget") == "budget") or
                    ('meze' in keyword_lower and 'meze' in r["description"].lower())):
                    filtered_restaurants.append(r)
            selected_restaurants = filtered_restaurants
        
        # Limit to maximum 5 restaurants
        limited_restaurants = selected_restaurants[:5]
        
        return {
            "status": "OK",
            "results": limited_restaurants
        }


# Convenience function for quick usage
def get_istanbul_restaurants_with_descriptions(district: Optional[str] = None, limit: int = 10) -> List[Dict]:
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
