#!/usr/bin/env python3
"""
Real Google Maps Restaurant Service for AI Istanbul
Provides live restaurant data from Google Places API
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from api_clients.enhanced_google_places import EnhancedGooglePlacesClient

logger = logging.getLogger(__name__)

class GoogleMapsRestaurantService:
    """Service for fetching real restaurant data from Google Maps"""
    
    def __init__(self):
        # Override the API key validation issue by directly checking environment
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = (os.getenv("GOOGLE_PLACES_API_KEY") or 
                   os.getenv("GOOGLE_MAPS_API_KEY") or
                   os.getenv("GOOGLE_WEATHER_API_KEY"))
        
        # Manual validation that works reliably
        self.has_valid_api_key = bool(api_key and len(api_key) > 20)
        self.use_real_apis = os.getenv("USE_REAL_APIS", "true").lower() == "true"
        
        # Initialize client but don't rely on its validation
        self.google_client = EnhancedGooglePlacesClient()
        
        # Override the service enablement logic
        self.is_enabled = self.has_valid_api_key and self.use_real_apis
        
        if self.is_enabled:
            logger.info(f"✅ Google Maps Restaurant Service: ENABLED with real API data (key: {api_key[:10]}...)")
        else:
            logger.warning(f"❌ Google Maps Restaurant Service: DISABLED - API key valid: {self.has_valid_api_key}, Use real APIs: {self.use_real_apis}")
    
    def get_restaurants_for_chat(self, user_query: str, location_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get restaurant recommendations for chat responses
        
        Args:
            user_query: User's restaurant-related query
            location_context: Specific Istanbul district/area
            
        Returns:
            Dictionary with restaurant data formatted for chat responses
        """
        
        if not self.is_enabled:
            return self._get_fallback_response(user_query, location_context)
        
        try:
            # Extract search parameters from query
            search_params = self._parse_restaurant_query(user_query, location_context)
            
            # Search Google Places API - use direct API call if client fails
            try:
                api_response = self.google_client.search_restaurants(
                    location=search_params.get("location"),
                    keyword=search_params.get("keyword"),
                    min_rating=search_params.get("min_rating")
                )
                
                # If client returns mock data but we should have real data, use direct API
                if (api_response.get("data_source") == "enhanced_mock" and 
                    self.has_valid_api_key and self.use_real_apis):
                    logger.info("🔄 Client returned mock data, trying direct API call...")
                    api_response = self._direct_google_places_search(search_params)
                    
            except Exception as client_error:
                logger.warning(f"Google client failed: {client_error}, trying direct API...")
                api_response = self._direct_google_places_search(search_params)
            
            # Format for chat response
            if api_response.get("restaurants"):
                return self._format_restaurant_response(api_response, search_params)
            else:
                logger.warning(f"No restaurants found for query: {user_query}")
                return self._get_fallback_response(user_query, location_context)
                
        except Exception as e:
            logger.error(f"Error fetching restaurants from Google Maps: {e}")
            return self._get_fallback_response(user_query, location_context)
    
    def _parse_restaurant_query(self, user_query: str, location_context: Optional[str]) -> Dict[str, Any]:
        """Parse user query to extract search parameters"""
        
        query_lower = user_query.lower()
        search_params = {}
        
        # Extract location preference - enhanced with more areas and variations  
        istanbul_areas = [
            'sultanahmet', 'beyoglu', 'galata', 'karakoy', 'taksim', 
            'kadikoy', 'besiktas', 'ortakoy', 'eminonu', 'fatih',
            'sisli', 'levent', 'maslak', 'etiler', 'bebek', 'arnavutkoy',
            'sariyer', 'bakirkoy', 'yesilkoy', 'atakoy', 'florya',
            'zeytinburnu', 'aksaray', 'laleli', 'beyazit', 'capa',
            'balat', 'fener', 'eyup', 'kasimpasa', 'haskoy',
            'uskudar', 'cengelkoy', 'beylerbeyi', 'camlica', 'altunizade',
            'bostanci', 'goztepe', 'bagdat caddesi', 'moda', 'caddebostan',
            'sisli', 'bebek', 'arnavutkoy', 'balat', 'fener'
        ]
        
        detected_area = location_context
        for area in istanbul_areas:
            if area in query_lower:
                detected_area = area
                break
        
        if detected_area:
            search_params["location"] = f"{detected_area} Istanbul"
        else:
            search_params["location"] = "Istanbul Turkey"
        
        # Extract cuisine/keyword preferences - enhanced with more options
        cuisine_keywords = {
            'turkish': ['turkish', 'ottoman', 'traditional', 'kebab', 'döner', 'authentic', 'local', 'anatolian'],
            'seafood': ['seafood', 'fish', 'balık', 'deniz', 'marine', 'fresh fish', 'sea food'],
            'italian': ['italian', 'pizza', 'pasta', 'pizzeria'],
            'asian': ['asian', 'sushi', 'japanese', 'chinese', 'thai', 'indian'],
            'vegetarian': ['vegetarian', 'vegan', 'plant-based', 'veggie', 'meat-free'],
            'fine dining': ['fine dining', 'upscale', 'luxury', 'elegant', 'michelin', 'gourmet'],
            'casual': ['casual', 'family', 'budget', 'affordable', 'cheap'],
            'breakfast': ['breakfast', 'kahvaltı', 'morning', 'brunch', 'turkish breakfast'],
            'dessert': ['dessert', 'sweet', 'baklava', 'lokum', 'turkish delight', 'pastry'],
            'meze': ['meze', 'appetizer', 'small plate', 'mezze', 'starter'],
            'coffee': ['coffee', 'cafe', 'kahve', 'turkish coffee', 'espresso'],
            'street food': ['street', 'fast', 'quick', 'street food', 'local food']
        }
        
        detected_keywords = []
        for cuisine, keywords in cuisine_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_keywords.append(cuisine)
        
        if detected_keywords:
            search_params["keyword"] = detected_keywords[0]  # Use first detected
        else:
            search_params["keyword"] = "Turkish"  # Default to Turkish cuisine
        
        # Extract quality preferences - more nuanced rating requirements
        quality_indicators = {
            'excellent': ['excellent', 'amazing', 'outstanding', 'exceptional', 'superb'],
            'very_good': ['very good', 'great', 'fantastic', 'wonderful', 'top', 'best'],
            'good': ['good', 'nice', 'decent', 'quality', 'recommended'],
            'any': ['any', 'whatever', 'anything']
        }
        
        min_rating = None
        for quality_level, indicators in quality_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                if quality_level == 'excellent':
                    min_rating = 4.5
                elif quality_level == 'very_good':
                    min_rating = 4.2
                elif quality_level == 'good':
                    min_rating = 4.0
                else:  # any
                    min_rating = 3.5
                break
        
        if min_rating:
            search_params["min_rating"] = min_rating
        
        # Extract price preferences
        price_indicators = {
            'expensive': ['expensive', 'upscale', 'fine dining', 'luxury', 'high-end'],
            'moderate': ['moderate', 'mid-range', 'reasonable'],
            'cheap': ['cheap', 'budget', 'affordable', 'inexpensive']
        }
        
        for price_level, indicators in price_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                search_params["price_preference"] = price_level
                break

        return search_params
    
    def _direct_google_places_search(self, search_params: Dict) -> Dict[str, Any]:
        """Direct API call to Google Places as fallback"""
        import requests
        import os
        
        api_key = (os.getenv("GOOGLE_PLACES_API_KEY") or 
                   os.getenv("GOOGLE_MAPS_API_KEY") or
                   os.getenv("GOOGLE_WEATHER_API_KEY"))
        
        if not api_key:
            raise Exception("No API key available for direct call")
        
        # Build query string
        location = search_params.get("location", "Istanbul Turkey")
        keyword = search_params.get("keyword", "Turkish")
        query = f"{keyword} restaurants in {location}"
        
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            'query': query,
            'key': api_key,
            'type': 'restaurant'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'OK':
                status = data.get('status')
                error_msg = data.get('error_message', 'Unknown error')
                
                # Handle specific API errors
                if status == 'ZERO_RESULTS':
                    logger.info(f"No restaurants found for query: {query}")
                    return {'success': False, 'restaurants': [], 'message': 'No restaurants found'}
                elif status == 'OVER_QUERY_LIMIT':
                    logger.error("Google Places API quota exceeded")
                    raise Exception("API quota exceeded - try again later")
                elif status == 'REQUEST_DENIED':
                    logger.error("Google Places API request denied - check API key")
                    raise Exception("API access denied - check configuration")
                else:
                    raise Exception(f"Google Places API error: {status} - {error_msg}")
                    
        except requests.RequestException as e:
            logger.error(f"Network error calling Google Places API: {e}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing Google Places API response: {e}")
            raise
        
        # Convert to our expected format with filtering
        restaurants = []
        min_rating = search_params.get('min_rating')
        price_preference = search_params.get('price_preference')
        
        for place in data.get('results', [])[:15]:  # Get more results for better filtering
            # Skip if business is closed permanently
            if place.get('business_status') == 'CLOSED_PERMANENTLY':
                continue
                
            # Apply rating filter
            rating = place.get('rating')
            if min_rating and rating:
                if rating < min_rating:
                    continue
            
            # Apply price filter if specified
            price_level = place.get('price_level')
            if price_preference and price_level is not None:
                if price_preference == 'cheap' and price_level > 2:
                    continue
                elif price_preference == 'expensive' and price_level < 3:
                    continue
                elif price_preference == 'moderate' and (price_level < 2 or price_level > 3):
                    continue
            
            # Enhanced restaurant data
            restaurant = {
                'place_id': place.get('place_id'),
                'name': place.get('name', 'Unknown Restaurant'),
                'rating': rating,
                'user_ratings_total': place.get('user_ratings_total', 0),
                'address': place.get('formatted_address', 'Address not available'),
                'price_level': price_level,
                'types': place.get('types', []),
                'opening_hours': place.get('opening_hours', {}),
                'business_status': place.get('business_status', 'OPERATIONAL'),
                'vicinity': place.get('vicinity', ''),
                'geometry': place.get('geometry', {}),
                # Add enhanced fields
                'rating_summary': self._create_rating_summary(rating, place.get('user_ratings_total', 0)),
                'cuisine_type': self._extract_cuisine_from_types(place.get('types', [])),
                'is_highly_rated': rating and rating >= 4.2,
                'has_many_reviews': place.get('user_ratings_total', 0) >= 100
            }
            restaurants.append(restaurant)
            
            # Stop when we have enough good results
            if len(restaurants) >= 10:
                break
        
        return {
            'success': True,
            'restaurants': restaurants,
            'data_source': 'google_places_direct',
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_restaurant_response(self, api_response: Dict, search_params: Dict) -> Dict[str, Any]:
        """Format Google Places API response for chat"""
        
        restaurants = api_response.get("restaurants", [])
        
        formatted_restaurants = []
        for restaurant in restaurants[:6]:  # Limit to 6 for better chat experience
            
            # Format price level
            price_level = restaurant.get("price_level")
            if price_level == 1:
                price_desc = "Budget-friendly"
            elif price_level == 2:
                price_desc = "Moderate"
            elif price_level == 3:
                price_desc = "Upscale"
            elif price_level == 4:
                price_desc = "Very expensive"
            else:
                price_desc = "Price varies"
            
            # Format restaurant types
            types = restaurant.get("types", [])
            cuisine_types = [t.replace("_", " ").title() for t in types if t not in ['point_of_interest', 'establishment', 'food']]
            
            formatted_restaurant = {
                "name": restaurant.get("name", "Unknown Restaurant"),
                "rating": restaurant.get("rating", "No rating"),
                "address": restaurant.get("address", "Address not available"),
                "price_level": price_desc,
                "cuisine_types": cuisine_types[:3],  # Limit types
                "is_open": restaurant.get("opening_hours", {}).get("open_now"),
                "business_status": restaurant.get("business_status", "OPERATIONAL"),
                "google_maps_link": f"https://www.google.com/maps/place/?q=place_id:{restaurant.get('place_id', '')}"
            }
            formatted_restaurants.append(formatted_restaurant)
        
        return {
            "success": True,
            "data_source": "google_maps_live",
            "restaurants": formatted_restaurants,
            "search_location": search_params.get("location", "Istanbul"),
            "search_keyword": search_params.get("keyword", "Turkish"),
            "timestamp": datetime.now().isoformat(),
            "google_maps_tip": "For the most current information including hours, photos, and reviews, search these restaurants on Google Maps."
        }
    
    def _get_fallback_response(self, user_query: str, location_context: Optional[str]) -> Dict[str, Any]:
        """Fallback response when Google Maps API is unavailable"""
        
        return {
            "success": False,
            "data_source": "fallback",
            "message": "Google Maps restaurant data temporarily unavailable",
            "google_maps_tip": "Search Google Maps directly for 'restaurants in Istanbul' to find current recommendations with live data.",
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_rating_summary(self, rating, user_ratings_total):
        """Create a human-readable rating summary for a restaurant"""
        if rating is None:
            return "No rating available"
        if user_ratings_total is None or user_ratings_total == 0:
            return f"Rated {rating}/5 (no reviews)"
        if rating >= 4.5 and user_ratings_total >= 100:
            return f"Excellent ({rating}/5, {user_ratings_total} reviews)"
        elif rating >= 4.2:
            return f"Very good ({rating}/5, {user_ratings_total} reviews)"
        elif rating >= 4.0:
            return f"Good ({rating}/5, {user_ratings_total} reviews)"
        else:
            return f"Rated {rating}/5 ({user_ratings_total} reviews)"

    def _extract_cuisine_from_types(self, types):
        """Extract cuisine type from Google Places types list"""
        if not types:
            return "Unknown"
        cuisine_map = {
            'turkish': ['turkish', 'kebab', 'doner', 'meze'],
            'seafood': ['seafood', 'fish', 'balik'],
            'italian': ['italian', 'pizza', 'pasta'],
            'asian': ['asian', 'sushi', 'japanese', 'chinese', 'thai', 'indian'],
            'vegetarian': ['vegetarian', 'vegan'],
            'dessert': ['dessert', 'baklava', 'pastry'],
            'coffee': ['coffee', 'cafe', 'kahve'],
            'street food': ['street_food', 'fast_food']
        }
        for cuisine, keywords in cuisine_map.items():
            for t in types:
                t_lower = t.lower().replace('_', ' ')
                if any(keyword in t_lower for keyword in keywords):
                    return cuisine.title()
        # Fallback: use first type that's not generic
        for t in types:
            if t not in ['restaurant', 'food', 'establishment', 'point_of_interest']:
                return t.replace('_', ' ').title()
        return "Restaurant"

# Global instance for easy import
google_restaurant_service = GoogleMapsRestaurantService()

def get_live_restaurant_recommendations(user_query: str, location_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to get live restaurant recommendations
    
    Args:
        user_query: User's restaurant query
        location_context: Specific Istanbul area/district
        
    Returns:
        Formatted restaurant data for chat responses
    """
    return google_restaurant_service.get_restaurants_for_chat(user_query, location_context)
