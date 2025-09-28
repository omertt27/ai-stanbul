import requests
import os
import time
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EnhancedGooglePlacesClient:
    """Enhanced Google Places API client with real data priority, caching, and rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or 
                       os.getenv("GOOGLE_PLACES_API_KEY") or 
                       os.getenv("GOOGLE_MAPS_API_KEY") or
                       os.getenv("GOOGLE_WEATHER_API_KEY"))
        self.has_api_key = bool(self.api_key and len(self.api_key) > 20)  # Valid key check
        self.base_url = "https://maps.googleapis.com/maps/api/place"
        
        # Rate limiting and caching configuration
        self.rate_limit = int(os.getenv("GOOGLE_PLACES_RATE_LIMIT", "100"))  # per day
        self.cache_duration = int(os.getenv("CACHE_DURATION_MINUTES", "30"))  # minutes
        self.use_real_apis = os.getenv("USE_REAL_APIS", "true").lower() == "true"
        self.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        
        # Simple in-memory cache (in production, use Redis)
        self._cache = {}
        self._request_count = 0
        self._last_reset = datetime.now()
        
        if not self.has_api_key or not self.use_real_apis:
            logger.warning("Google Places API: Using fallback mode with enhanced mock data.")
        else:
            logger.info("Google Places API: Ready for live data integration!")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        # Reset daily counter
        if datetime.now() - self._last_reset > timedelta(days=1):
            self._request_count = 0
            self._last_reset = datetime.now()
        
        if self._request_count >= self.rate_limit:
            logger.warning(f"Rate limit exceeded: {self._request_count}/{self.rate_limit}")
            return False
        return True
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for request."""
        sorted_params = sorted(kwargs.items())
        return f"{method}:{hash(str(sorted_params))}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if available and not expired."""
        if not self.enable_caching or cache_key not in self._cache:
            return None
        
        cached_data, timestamp = self._cache[cache_key]
        if datetime.now() - timestamp < timedelta(minutes=self.cache_duration):
            logger.info(f"Cache HIT: {cache_key}")
            return cached_data
        else:
            logger.info(f"Cache EXPIRED: {cache_key}")
            del self._cache[cache_key]
            return None
    
    def _cache_response(self, cache_key: str, data: Dict) -> None:
        """Cache API response."""
        if self.enable_caching:
            self._cache[cache_key] = (data, datetime.now())
            logger.info(f"Cache STORED: {cache_key}")
    
    def search_restaurants(self, 
                         location: Optional[str] = None,
                         lat_lng: Optional[str] = None, 
                         radius: int = 1500,
                         keyword: Optional[str] = None,
                         min_rating: Optional[float] = None) -> Dict:
        """
        Search for restaurants with real Google Places API data when available.
        Falls back to enhanced mock data when API is unavailable.
        """
        cache_key = self._get_cache_key("search_restaurants", 
                                      location=location, lat_lng=lat_lng, 
                                      radius=radius, keyword=keyword, min_rating=min_rating)
        
        # Try cache first
        cached_result = self._get_cached_response(cache_key)
        if cached_result:
            return cached_result
        
        # Use real API if available and within limits
        if self.has_api_key and self.use_real_apis and self._check_rate_limit():
            try:
                result = self._search_restaurants_real_api(location, lat_lng, radius, keyword, min_rating)
                self._request_count += 1
                self._cache_response(cache_key, result)
                logger.info(f"✅ REAL DATA: Restaurant search successful ({len(result.get('results', []))} results)")
                return result
            except Exception as e:
                logger.error(f"Real API failed, falling back to mock data: {e}")
        
        # Fallback to enhanced mock data
        result = self._get_enhanced_mock_restaurant_data(location, keyword)
        logger.info(f"📝 MOCK DATA: Using enhanced fallback data ({len(result.get('results', []))} results)")
        return result
    
    def _search_restaurants_real_api(self, location: Optional[str], lat_lng: Optional[str], 
                                   radius: int, keyword: Optional[str], min_rating: Optional[float]) -> Dict:
        """Execute real Google Places API search."""
        url = f"{self.base_url}/textsearch/json"
        
        # Build search query - improved query construction
        if keyword and location:
            query = f"{keyword} restaurant in {location}"
        elif keyword:
            query = f"{keyword} restaurant in Istanbul Turkey"
        elif location:
            query = f"Turkish restaurant in {location} Istanbul"
        else:
            query = "Turkish restaurant in Istanbul Turkey"
        
        params = {
            "key": self.api_key,
            "query": query,
            "type": "restaurant",
        }
        
        logger.info(f"🔍 Google Places API query: {query}")
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"📊 Google Places API returned {len(data.get('results', []))} results with status: {data.get('status')}")
        
        # Filter by rating if specified
        if min_rating and "results" in data:
            original_count = len(data["results"])
            data["results"] = [
                place for place in data["results"] 
                if place.get("rating", 0) >= min_rating
            ]
            logger.info(f"🎯 Filtered by rating {min_rating}+: {original_count} → {len(data['results'])} results")
        
        # Process and structure results for better response
        structured_results = []
        for i, place in enumerate(data.get("results", [])[:8]):  # Limit to 8 for better response
            structured_place = {
                "name": place.get("name", "Unknown Restaurant"),
                "rating": place.get("rating"),
                "address": place.get("formatted_address", "Address not available"),
                "price_level": place.get("price_level"),
                "types": place.get("types", []),
                "place_id": place.get("place_id"),
                "opening_hours": place.get("opening_hours", {}),
                "photos": place.get("photos", []),
                "geometry": place.get("geometry", {}),
                "plus_code": place.get("plus_code", {}),
                "business_status": place.get("business_status", "OPERATIONAL")
            }
            structured_results.append(structured_place)
        
        return {
            "restaurants": structured_results,  # Changed key for clarity
            "status": data.get("status"),
            "data_source": "google_places_api",
            "timestamp": datetime.now().isoformat(),
            "query_used": query,
            "total_results": len(data.get("results", []))
        }
    
    def _enhance_place_data(self, place: Dict) -> Dict:
        """Enhance place data with additional details if API quota allows."""
        if not self._check_rate_limit():
            return place
        
        place_id = place.get("place_id")
        if not place_id:
            return place
        
        try:
            details = self.get_place_details(place_id, ["photos", "reviews", "opening_hours", "website"])
            if details.get("status") == "OK":
                result = details.get("result", {})
                
                # Add enhanced data
                place["photos_count"] = len(result.get("photos", []))
                place["reviews_count"] = len(result.get("reviews", []))
                place["has_website"] = bool(result.get("website"))
                place["opening_hours"] = result.get("opening_hours", {})
                
                # Add sample review if available
                reviews = result.get("reviews", [])
                if reviews:
                    place["sample_review"] = {
                        "text": reviews[0].get("text", "")[:200] + "...",
                        "rating": reviews[0].get("rating"),
                        "author": reviews[0].get("author_name")
                    }
                    
            return place
        except Exception as e:
            logger.warning(f"Failed to enhance place data for {place.get('name')}: {e}")
            return place
    
    def get_place_details(self, place_id: str, fields: Optional[List[str]] = None) -> Dict:
        """Get detailed information about a specific place."""
        if not self.has_api_key or not self.use_real_apis:
            return {"status": "MOCK_MODE", "result": {}}
        
        cache_key = self._get_cache_key("place_details", place_id=place_id, fields=fields)
        cached_result = self._get_cached_response(cache_key)
        if cached_result:
            return cached_result
        
        if not self._check_rate_limit():
            return {"status": "RATE_LIMITED"}
        
        url = f"{self.base_url}/details/json"
        
        # Default fields for restaurant details
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
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            result["data_source"] = "real_api"
            result["timestamp"] = datetime.now().isoformat()
            
            self._request_count += 1
            self._cache_response(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error getting place details for {place_id}: {e}")
            return {"status": "API_ERROR"}
    
    def _get_enhanced_mock_restaurant_data(self, location: Optional[str], keyword: Optional[str]) -> Dict:
        """Return enhanced mock data that closely resembles real Google Places responses."""
        
        # Enhanced mock restaurants with realistic Istanbul data
        mock_restaurants = [
            {
                "place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4",
                "name": "Pandeli Restaurant",
                "rating": 4.3,
                "user_ratings_total": 1247,
                "price_level": 3,
                "vicinity": "Eminönü, Fatih/İstanbul",
                "types": ["restaurant", "food", "establishment"],
                "opening_hours": {"open_now": True, "periods": [], "weekday_text": ["Monday: 12:00 PM – 10:00 PM"]},
                "geometry": {"location": {"lat": 41.0168, "lng": 28.9734}},
                "photos": [{"photo_reference": "mock_photo_1", "height": 1080, "width": 1920}],
                "sample_review": {
                    "text": "Authentic Ottoman cuisine in a historic setting. The lamb dishes are exceptional...",
                    "rating": 5,
                    "author": "Local Food Critic"
                },
                "has_website": True,
                "photos_count": 156,
                "reviews_count": 1247
            },
            {
                "place_id": "ChIJN1t_tDeuEmsRUsoyG83frY5",
                "name": "Hamdi Restaurant",
                "rating": 4.5,
                "user_ratings_total": 3421,
                "price_level": 2,
                "vicinity": "Eminönü, Fatih/İstanbul",
                "types": ["restaurant", "food", "establishment"],
                "opening_hours": {"open_now": True},
                "geometry": {"location": {"lat": 41.0186, "lng": 28.9736}},
                "photos": [{"photo_reference": "mock_photo_2"}],
                "sample_review": {
                    "text": "Best kebab in Istanbul with amazing Bosphorus views. Must try the Beyti...",
                    "rating": 5,
                    "author": "Istanbul Traveler"
                },
                "has_website": True,
                "photos_count": 89,
                "reviews_count": 3421
            },
            {
                "place_id": "ChIJN1t_tDeuEmsRUsoyG83frY6",
                "name": "Çiya Sofrası",
                "rating": 4.4,
                "user_ratings_total": 2156,
                "price_level": 2,
                "vicinity": "Kadıköy, İstanbul",
                "types": ["restaurant", "food", "establishment"],
                "opening_hours": {"open_now": False},
                "geometry": {"location": {"lat": 40.9908, "lng": 29.0299}},
                "photos": [{"photo_reference": "mock_photo_3"}],
                "sample_review": {
                    "text": "Incredible variety of traditional Turkish dishes. Every dish tells a story...",
                    "rating": 4,
                    "author": "Food Enthusiast"
                },
                "has_website": False,
                "photos_count": 234,
                "reviews_count": 2156
            },
            {
                "place_id": "ChIJN1t_tDeuEmsRUsoyG83frY7",
                "name": "Karakoy Lokantasi",
                "rating": 4.2,
                "user_ratings_total": 1834,
                "price_level": 3,
                "vicinity": "Karaköy, Beyoğlu/İstanbul",
                "types": ["restaurant", "food", "establishment"],
                "opening_hours": {"open_now": True},
                "geometry": {"location": {"lat": 41.0242, "lng": 28.9744}},
                "photos": [{"photo_reference": "mock_photo_4"}],
                "sample_review": {
                    "text": "Modern interpretation of Ottoman cuisine. The sea bass is perfectly prepared...",
                    "rating": 5,
                    "author": "Michelin Reviewer"
                },
                "has_website": True,
                "photos_count": 167,
                "reviews_count": 1834
            },
            {
                "place_id": "ChIJN1t_tDeuEmsRUsoyG83frY8",
                "name": "Deraliye Ottoman Cuisine",
                "rating": 4.1,
                "user_ratings_total": 967,
                "price_level": 4,
                "vicinity": "Sultanahmet, Fatih/İstanbul",
                "types": ["restaurant", "food", "establishment"],
                "opening_hours": {"open_now": True},
                "geometry": {"location": {"lat": 41.0058, "lng": 28.9784}},
                "photos": [{"photo_reference": "mock_photo_5"}],
                "sample_review": {
                    "text": "Exquisite Ottoman palace cuisine. The historical recipes are fascinating...",
                    "rating": 4,
                    "author": "History & Food Lover"
                },
                "has_website": True,
                "photos_count": 93,
                "reviews_count": 967
            }
        ]
        
        # Filter by location if specified
        if location and "sultanahmet" in location.lower():
            mock_restaurants = [r for r in mock_restaurants if "sultanahmet" in r["vicinity"].lower()]
        elif location and "kadikoy" in location.lower():
            mock_restaurants = [r for r in mock_restaurants if "kadıköy" in r["vicinity"].lower()]
        
        # Filter by keyword if specified
        if keyword:
            keyword_lower = keyword.lower()
            filtered = []
            for restaurant in mock_restaurants:
                if (keyword_lower in restaurant["name"].lower() or 
                    any(keyword_lower in review_text.lower() 
                        for review_text in [restaurant.get("sample_review", {}).get("text", "")])):
                    filtered.append(restaurant)
            if filtered:
                mock_restaurants = filtered
        
        return {
            "results": mock_restaurants,
            "status": "OK",
            "data_source": "enhanced_mock",
            "timestamp": datetime.now().isoformat(),
            "info_message": "🔄 Using enhanced mock data. Add GOOGLE_PLACES_API_KEY for real-time data."
        }
    
    def _reverse_geocode(self, lat_lng: str) -> Optional[str]:
        """Convert coordinates to location name (simplified)."""
        try:
            lat, lng = map(float, lat_lng.split(','))
            if 40.9 <= lat <= 41.2 and 28.8 <= lng <= 29.2:
                return "Istanbul, Turkey"
            return None
        except:
            return None
    
    def get_api_status(self) -> Dict:
        """Get current API status and usage information."""
        return {
            "has_api_key": self.has_api_key,
            "use_real_apis": self.use_real_apis,
            "enable_caching": self.enable_caching,
            "requests_today": self._request_count,
            "rate_limit": self.rate_limit,
            "cache_entries": len(self._cache),
            "data_source": "real_api" if (self.has_api_key and self.use_real_apis) else "mock_data"
        }
