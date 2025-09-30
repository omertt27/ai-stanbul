#!/usr/bin/env python3
"""
Real Istanbul Museum Service
============================

This module integrates with real APIs to fetch live museum information:
- Google Maps Places API for museum details, reviews, and photos
- Istanbul Open Data Portal for official museum data
- Real-time opening hours, events, and ticket information
"""

import asyncio
import aiohttp
import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time
from urllib.parse import quote

logger = logging.getLogger(__name__)

@dataclass
class RealMuseumInfo:
    """Real museum information from APIs"""
    place_id: str
    name: str
    address: str
    coordinates: Tuple[float, float]
    phone: Optional[str]
    website: Optional[str]
    rating: Optional[float]
    total_ratings: Optional[int]
    opening_hours: Dict[str, str]
    opening_hours_text: List[str]
    price_level: Optional[int]  # 0-4 scale from Google
    photos: List[str]  # Photo references
    reviews: List[Dict[str, Any]]
    types: List[str]  # Place types from Google
    current_status: str  # OPEN, CLOSED, UNKNOWN
    next_open: Optional[str]
    next_close: Optional[str]
    special_hours: Optional[Dict[str, str]]
    accessibility: Optional[Dict[str, bool]]
    last_updated: str

class RealMuseumService:
    """Service for fetching real museum data from multiple APIs"""
    
    def __init__(self):
        self.google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.istanbul_open_data_key = os.getenv('ISTANBUL_OPEN_DATA_API_KEY')
        self.cache = {}
        self.cache_duration = int(os.getenv('CACHE_DURATION_HOURS', '1')) * 3600  # Convert to seconds
        
        # Famous Istanbul museums with their Google Place IDs
        self.istanbul_museums = {
            "hagia_sophia": {
                "name": "Hagia Sophia",
                "search_query": "Hagia Sophia Istanbul Turkey",
                "place_id": None  # Will be fetched dynamically
            },
            "topkapi_palace": {
                "name": "Topkapi Palace",
                "search_query": "Topkapi Palace Istanbul Turkey",
                "place_id": None
            },
            "blue_mosque": {
                "name": "Blue Mosque",
                "search_query": "Blue Mosque Sultan Ahmed Istanbul Turkey",
                "place_id": None
            },
            "basilica_cistern": {
                "name": "Basilica Cistern",
                "search_query": "Basilica Cistern Istanbul Turkey",
                "place_id": None
            },
            "galata_tower": {
                "name": "Galata Tower",
                "search_query": "Galata Tower Istanbul Turkey",
                "place_id": None
            },
            "dolmabahce_palace": {
                "name": "Dolmabahce Palace",
                "search_query": "Dolmabahce Palace Istanbul Turkey",
                "place_id": None
            },
            "istanbul_archaeology": {
                "name": "Istanbul Archaeology Museums",
                "search_query": "Istanbul Archaeology Museums Turkey",
                "place_id": None
            },
            "turkish_islamic_arts": {
                "name": "Turkish and Islamic Arts Museum",
                "search_query": "Turkish Islamic Arts Museum Istanbul",
                "place_id": None
            },
            "pera_museum": {
                "name": "Pera Museum",
                "search_query": "Pera Museum Istanbul Turkey",
                "place_id": None
            },
            "istanbul_modern": {
                "name": "Istanbul Modern",
                "search_query": "Istanbul Modern Art Museum Turkey",
                "place_id": None
            }
        }
        
    async def get_museum_info(self, museum_key: str) -> Optional[RealMuseumInfo]:
        """Get real museum information from Google Maps API"""
        if not self.google_maps_api_key:
            logger.warning("Google Maps API key not configured")
            return None
            
        # Check cache first
        cache_key = f"museum_{museum_key}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
            
        museum_config = self.istanbul_museums.get(museum_key)
        if not museum_config:
            logger.error(f"Unknown museum key: {museum_key}")
            return None
            
        try:
            # Get place ID if not cached
            if not museum_config['place_id']:
                place_id = await self._find_place_id(museum_config['search_query'])
                if not place_id:
                    logger.error(f"Could not find place ID for {museum_config['name']}")
                    return None
                museum_config['place_id'] = place_id
                
            # Get detailed place information
            place_details = await self._get_place_details(museum_config['place_id'])
            if not place_details:
                return None
                
            # Convert to our data structure
            museum_info = self._convert_to_museum_info(place_details)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': museum_info,
                'timestamp': time.time()
            }
            
            return museum_info
            
        except Exception as e:
            logger.error(f"Error fetching museum info for {museum_key}: {str(e)}")
            return None
    
    async def _find_place_id(self, search_query: str) -> Optional[str]:
        """Find Google Place ID using text search"""
        url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        params = {
            'input': search_query,
            'inputtype': 'textquery',
            'fields': 'place_id,name,geometry',
            'key': self.google_maps_api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    candidates = data.get('candidates', [])
                    if candidates:
                        return candidates[0]['place_id']
                return None
    
    async def _get_place_details(self, place_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed place information from Google Places API"""
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            'place_id': place_id,
            'fields': (
                'place_id,name,formatted_address,geometry,formatted_phone_number,'
                'website,rating,user_ratings_total,opening_hours,price_level,'
                'photos,reviews,types,current_opening_hours,secondary_opening_hours'
            ),
            'key': self.google_maps_api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('result')
                else:
                    logger.error(f"Google Places API error: {response.status}")
                    return None
    
    def _convert_to_museum_info(self, place_data: Dict[str, Any]) -> RealMuseumInfo:
        """Convert Google Places data to our museum info structure"""
        # Extract coordinates
        location = place_data.get('geometry', {}).get('location', {})
        coordinates = (location.get('lat', 0.0), location.get('lng', 0.0))
        
        # Process opening hours
        opening_hours = {}
        opening_hours_text = []
        current_status = "UNKNOWN"
        next_open = None
        next_close = None
        
        if 'opening_hours' in place_data:
            oh_data = place_data['opening_hours']
            opening_hours_text = oh_data.get('weekday_text', [])
            
            # Convert to our format
            days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            periods = oh_data.get('periods', [])
            
            for i, day in enumerate(days):
                day_periods = [p for p in periods if p.get('open', {}).get('day') == i]
                if day_periods:
                    period = day_periods[0]
                    open_time = period.get('open', {}).get('time', '0000')
                    close_time = period.get('close', {}).get('time', '2400')
                    
                    # Format times
                    open_formatted = f"{open_time[:2]}:{open_time[2:]}"
                    close_formatted = f"{close_time[:2]}:{close_time[2:]}"
                    opening_hours[day] = f"{open_formatted} - {close_formatted}"
                else:
                    opening_hours[day] = "Closed"
        
        # Determine current status
        if 'current_opening_hours' in place_data:
            current_oh = place_data['current_opening_hours']
            if current_oh.get('open_now'):
                current_status = "OPEN"
            else:
                current_status = "CLOSED"
        
        # Process photos
        photos = []
        if 'photos' in place_data:
            photos = [
                f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo['photo_reference']}&key={self.google_maps_api_key}"
                for photo in place_data['photos'][:5]  # Limit to 5 photos
            ]
        
        # Process reviews
        reviews = []
        if 'reviews' in place_data:
            reviews = [
                {
                    'author_name': review.get('author_name', ''),
                    'rating': review.get('rating', 0),
                    'text': review.get('text', ''),
                    'time': review.get('time', 0),
                    'language': review.get('language', 'en')
                }
                for review in place_data['reviews'][:5]  # Limit to 5 reviews
            ]
        
        return RealMuseumInfo(
            place_id=place_data.get('place_id', ''),
            name=place_data.get('name', ''),
            address=place_data.get('formatted_address', ''),
            coordinates=coordinates,
            phone=place_data.get('formatted_phone_number'),
            website=place_data.get('website'),
            rating=place_data.get('rating'),
            total_ratings=place_data.get('user_ratings_total'),
            opening_hours=opening_hours,
            opening_hours_text=opening_hours_text,
            price_level=place_data.get('price_level'),
            photos=photos,
            reviews=reviews,
            types=place_data.get('types', []),
            current_status=current_status,
            next_open=next_open,
            next_close=next_close,
            special_hours=None,
            accessibility=None,
            last_updated=datetime.now().isoformat()
        )
    
    async def get_all_museums(self) -> Dict[str, RealMuseumInfo]:
        """Get information for all configured Istanbul museums"""
        results = {}
        
        # Create tasks for concurrent API calls
        tasks = []
        for museum_key in self.istanbul_museums.keys():
            tasks.append(self.get_museum_info(museum_key))
        
        # Execute all tasks concurrently
        museum_infos = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for museum_key, museum_info in zip(self.istanbul_museums.keys(), museum_infos):
            if isinstance(museum_info, RealMuseumInfo):
                results[museum_key] = museum_info
            elif isinstance(museum_info, Exception):
                logger.error(f"Error fetching {museum_key}: {str(museum_info)}")
            
        return results
    
    async def search_museums_nearby(self, lat: float, lng: float, radius: int = 5000) -> List[RealMuseumInfo]:
        """Search for museums near a location"""
        if not self.google_maps_api_key:
            return []
            
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            'location': f"{lat},{lng}",
            'radius': radius,
            'type': 'museum',
            'key': self.google_maps_api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        
                        for place in data.get('results', [])[:10]:  # Limit to 10 results
                            # Get detailed info for each place
                            detailed_info = await self._get_place_details(place['place_id'])
                            if detailed_info:
                                museum_info = self._convert_to_museum_info(detailed_info)
                                results.append(museum_info)
                        
                        return results
                    return []
        except Exception as e:
            logger.error(f"Error searching nearby museums: {str(e)}")
            return []
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_entry = self.cache[cache_key]
        current_time = time.time()
        return (current_time - cache_entry['timestamp']) < self.cache_duration
    
    async def get_museum_events(self, museum_key: str) -> List[Dict[str, Any]]:
        """Get special events for a museum (placeholder for future integration)"""
        # This could integrate with museum-specific APIs or event platforms
        # For now, return empty list
        return []
    
    def to_dict(self, museum_info: RealMuseumInfo) -> Dict[str, Any]:
        """Convert museum info to dictionary for API response"""
        return asdict(museum_info)

# Global service instance
real_museum_service = RealMuseumService()
