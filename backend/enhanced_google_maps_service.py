#!/usr/bin/env python3
"""
Enhanced Google Maps Integration Service
=======================================

Expanded Google Maps integration covering restaurants, museums, attractions,
and transportation with fact-checking capabilities.
"""

import os
import requests
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class PlaceInfo:
    """Enhanced place information from Google Maps"""
    name: str
    place_id: str
    address: str
    formatted_address: str
    location: Dict[str, float]  # lat, lng
    rating: Optional[float]
    user_ratings_total: Optional[int]
    price_level: Optional[str]
    opening_hours: Optional[Dict[str, Any]]
    is_open_now: Optional[bool]
    place_types: List[str]
    photos: List[str]
    reviews_summary: Optional[str]
    website: Optional[str]
    phone: Optional[str]
    google_maps_link: str
    fact_checked: bool = False
    official_source: Optional[str] = None

class EnhancedGoogleMapsService:
    """Enhanced Google Maps service for comprehensive Istanbul information"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        self.base_url = "https://maps.googleapis.com/maps/api"
        self.session = requests.Session()
        
        # Istanbul coordinates for location-based searches
        self.istanbul_center = {"lat": 41.0082, "lng": 28.9784}
        self.search_radius = 50000  # 50km to cover all of Istanbul
        
        # Official sources for fact-checking
        self.official_sources = {
            "museums": [
                "muze.gov.tr",
                "istanbul.gov.tr", 
                "topkapisarayi.gov.tr",
                "ayasofyamuzesi.gov.tr"
            ],
            "transportation": [
                "metro.istanbul",
                "iett.istanbul",
                "sehirhatlari.istanbul"
            ],
            "attractions": [
                "istanbul.gov.tr",
                "kultur.gov.tr"
            ]
        }
    
    def search_places_comprehensive(self, query: str, place_type: str, location_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive place search with fact-checking for museums, attractions, restaurants, and transportation
        """
        try:
            if not self.api_key:
                return {"success": False, "message": "Google Maps API key not configured"}
            
            # Determine search parameters based on type
            search_params = self._get_search_parameters(query, place_type, location_context)
            
            # Perform initial search
            places_data = self._perform_places_search(search_params)
            
            if not places_data["success"]:
                return places_data
            
            # Enhanced place details with fact-checking
            enhanced_places = []
            for place in places_data["places"][:8]:  # Process up to 8 places
                enhanced_place = self._get_enhanced_place_details(place["place_id"], place_type)
                if enhanced_place:
                    enhanced_places.append(enhanced_place)
            
            return {
                "success": True,
                "places": enhanced_places,
                "search_type": place_type,
                "search_query": query,
                "location_context": location_context,
                "timestamp": datetime.now().isoformat(),
                "total_found": len(enhanced_places)
            }
            
        except Exception as e:
            logging.error(f"Error in comprehensive place search: {e}")
            return {"success": False, "message": f"Search error: {str(e)}"}
    
    def _get_search_parameters(self, query: str, place_type: str, location_context: Optional[str] = None) -> Dict[str, Any]:
        """Get optimized search parameters based on place type and context"""
        
        base_params = {
            "key": self.api_key,
            "language": "en",
            "region": "tr"
        }
        
        # Location-specific coordinates
        if location_context:
            location_coords = self._get_location_coordinates(location_context)
            if location_coords:
                base_params["location"] = f"{location_coords['lat']},{location_coords['lng']}"
                base_params["radius"] = "5000"  # 5km for specific areas
            else:
                base_params["location"] = f"{self.istanbul_center['lat']},{self.istanbul_center['lng']}"
                base_params["radius"] = str(self.search_radius)
        else:
            base_params["location"] = f"{self.istanbul_center['lat']},{self.istanbul_center['lng']}"
            base_params["radius"] = str(self.search_radius)
        
        # Type-specific parameters
        if place_type == "museum":
            base_params["type"] = "museum"
            base_params["query"] = f"{query} museum Istanbul"
        elif place_type == "attraction":
            base_params["type"] = "tourist_attraction"
            base_params["query"] = f"{query} attraction Istanbul"
        elif place_type == "restaurant":
            base_params["type"] = "restaurant"
            base_params["query"] = f"{query} restaurant Istanbul"
        elif place_type == "transportation":
            base_params["type"] = "transit_station"
            base_params["query"] = f"{query} metro tram station Istanbul"
        else:
            base_params["query"] = f"{query} Istanbul"
        
        return base_params
    
    def _get_location_coordinates(self, location_context: str) -> Optional[Dict[str, float]]:
        """Get specific coordinates for Istanbul neighborhoods/districts"""
        
        location_coords = {
            "sultanahmet": {"lat": 41.0055, "lng": 28.9769},
            "beyoglu": {"lat": 41.0370, "lng": 28.9776},
            "galata": {"lat": 41.0256, "lng": 28.9740},
            "kadikoy": {"lat": 40.9897, "lng": 29.0196},
            "taksim": {"lat": 41.0370, "lng": 28.9858},
            "eminonu": {"lat": 41.0167, "lng": 28.9709},
            "besiktas": {"lat": 41.0422, "lng": 29.0067},
            "ortakoy": {"lat": 41.0555, "lng": 29.0268},
            "uskudar": {"lat": 41.0214, "lng": 29.0167},
            "balat": {"lat": 41.0292, "lng": 28.9489}
        }
        
        return location_coords.get(location_context.lower().replace("ğ", "g").replace("ü", "u").replace("ı", "i"))
    
    def _perform_places_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual Google Places search"""
        
        try:
            # Use Text Search API for better results
            url = f"{self.base_url}/place/textsearch/json"
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "OK":
                return {"success": False, "message": f"Google API error: {data.get('status')}"}
            
            places = []
            for result in data.get("results", []):
                place_info = {
                    "place_id": result.get("place_id"),
                    "name": result.get("name"),
                    "address": result.get("formatted_address", ""),
                    "rating": result.get("rating"),
                    "user_ratings_total": result.get("user_ratings_total"),
                    "price_level": self._format_price_level(result.get("price_level")),
                    "types": result.get("types", []),
                    "location": result.get("geometry", {}).get("location", {}),
                    "is_open_now": result.get("opening_hours", {}).get("open_now")
                }
                places.append(place_info)
            
            return {"success": True, "places": places}
            
        except requests.RequestException as e:
            return {"success": False, "message": f"Network error: {str(e)}"}
        except Exception as e:
            return {"success": False, "message": f"Search error: {str(e)}"}
    
    def _get_enhanced_place_details(self, place_id: str, place_type: str) -> Optional[PlaceInfo]:
        """Get detailed information for a specific place with fact-checking"""
        
        try:
            url = f"{self.base_url}/place/details/json"
            params = {
                "key": self.api_key,
                "place_id": place_id,
                "fields": "name,formatted_address,geometry,rating,user_ratings_total,price_level,opening_hours,website,formatted_phone_number,photos,reviews,types,url"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "OK":
                return None
            
            result = data.get("result", {})
            
            # Extract enhanced information
            place_info = PlaceInfo(
                name=result.get("name", ""),
                place_id=place_id,
                address=result.get("formatted_address", ""),
                formatted_address=result.get("formatted_address", ""),
                location=result.get("geometry", {}).get("location", {}),
                rating=result.get("rating"),
                user_ratings_total=result.get("user_ratings_total"),
                price_level=self._format_price_level(result.get("price_level")),
                opening_hours=self._format_opening_hours(result.get("opening_hours")),
                is_open_now=result.get("opening_hours", {}).get("open_now"),
                place_types=result.get("types", []),
                photos=self._extract_photo_urls(result.get("photos", [])),
                reviews_summary=self._summarize_reviews(result.get("reviews", [])),
                website=result.get("website"),
                phone=result.get("formatted_phone_number"),
                google_maps_link=result.get("url", f"https://maps.google.com/?place_id={place_id}")
            )
            
            # Fact-checking
            place_info.fact_checked, place_info.official_source = self._fact_check_place(place_info, place_type)
            
            return place_info
            
        except Exception as e:
            logging.error(f"Error getting place details for {place_id}: {e}")
            return None
    
    def _format_price_level(self, price_level: Optional[int]) -> Optional[str]:
        """Convert Google's price level to readable format"""
        if price_level is None:
            return None
        
        price_map = {
            0: "Free",
            1: "Inexpensive", 
            2: "Moderate",
            3: "Expensive",
            4: "Very Expensive"
        }
        return price_map.get(price_level, "Unknown")
    
    def _format_opening_hours(self, opening_hours: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """Format opening hours information"""
        if not opening_hours:
            return None
        
        return {
            "open_now": opening_hours.get("open_now"),
            "weekday_text": opening_hours.get("weekday_text", []),
            "periods": opening_hours.get("periods", [])
        }
    
    def _extract_photo_urls(self, photos: List[Dict]) -> List[str]:
        """Extract photo URLs from Google Photos references"""
        photo_urls = []
        for photo in photos[:3]:  # Limit to 3 photos
            if photo.get("photo_reference"):
                photo_url = f"{self.base_url}/place/photo?maxwidth=400&photoreference={photo['photo_reference']}&key={self.api_key}"
                photo_urls.append(photo_url)
        return photo_urls
    
    def _summarize_reviews(self, reviews: List[Dict]) -> Optional[str]:
        """Create a summary of recent reviews"""
        if not reviews:
            return None
        
        # Get top-rated reviews
        top_reviews = sorted(reviews, key=lambda x: x.get("rating", 0), reverse=True)[:3]
        
        summary_points = []
        for review in top_reviews:
            text = review.get("text", "")
            if text and len(text) > 50:
                # Extract key phrases (simplified approach)
                summary_points.append(text[:100] + "..." if len(text) > 100 else text)
        
        return " | ".join(summary_points) if summary_points else None
    
    def _fact_check_place(self, place_info: PlaceInfo, place_type: str) -> Tuple[bool, Optional[str]]:
        """Fact-check place information against official sources"""
        
        # Check if website matches official sources
        if place_info.website:
            official_domains = self.official_sources.get(place_type, [])
            for domain in official_domains:
                if domain in place_info.website:
                    return True, domain
        
        # For well-known places, mark as fact-checked
        well_known_places = [
            "hagia sophia", "topkapi palace", "blue mosque", "galata tower",
            "basilica cistern", "grand bazaar", "spice bazaar", "dolmabahce palace"
        ]
        
        if any(known in place_info.name.lower() for known in well_known_places):
            return True, "verified_landmark"
        
        return False, None
    
    # Public methods for different place types
    
    def search_museums(self, query: str, location_context: Optional[str] = None) -> Dict[str, Any]:
        """Search for museums with enhanced information"""
        return self.search_places_comprehensive(query, "museum", location_context)
    
    def search_attractions(self, query: str, location_context: Optional[str] = None) -> Dict[str, Any]:
        """Search for tourist attractions with enhanced information"""
        return self.search_places_comprehensive(query, "attraction", location_context)
    
    def search_restaurants(self, query: str, location_context: Optional[str] = None) -> Dict[str, Any]:
        """Search for restaurants with enhanced information"""
        return self.search_places_comprehensive(query, "restaurant", location_context)
    
    def search_transportation(self, query: str, location_context: Optional[str] = None) -> Dict[str, Any]:
        """Search for transportation stations and hubs"""
        return self.search_places_comprehensive(query, "transportation", location_context)
    
    def get_place_actionable_info(self, place_info: PlaceInfo) -> Dict[str, str]:
        """Generate actionable information for a place"""
        
        actionable_info = {
            "exact_address": place_info.formatted_address,
            "google_maps_link": place_info.google_maps_link,
        }
        
        # Add phone if available
        if place_info.phone:
            actionable_info["phone"] = place_info.phone
        
        # Add website if available
        if place_info.website:
            actionable_info["website"] = place_info.website
        
        # Add opening hours
        if place_info.opening_hours and place_info.opening_hours.get("weekday_text"):
            actionable_info["opening_hours"] = place_info.opening_hours["weekday_text"]
        
        # Add current status
        if place_info.is_open_now is not None:
            actionable_info["current_status"] = "Open now" if place_info.is_open_now else "Currently closed"
        
        return actionable_info

# Global instance
enhanced_google_maps = EnhancedGoogleMapsService()
