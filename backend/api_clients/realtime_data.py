"""
Real-time Data Integration Module for AI Istanbul Travel Guide
Provides live events, crowd levels, wait times, dynamic pricing, and traffic data
"""

import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import asyncio

# Optional aiohttp import
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    # Create dummy aiohttp for type hints and graceful degradation
    class DummyResponse:
        def __init__(self):
            self.status = 503
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def json(self):
            return {}
    
    class DummyClientSession:
        def __init__(self):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def get(self, *args, **kwargs):
            return DummyResponse()
    
    class DummyAiohttp:
        ClientSession = DummyClientSession
    
    aiohttp = DummyAiohttp()
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EventData:
    """Real-time event information"""
    event_id: str
    name: str
    location: str
    start_time: datetime
    end_time: Optional[datetime]
    category: str
    price_range: Optional[Tuple[float, float]]
    crowd_level: str
    description: str
    venue_capacity: Optional[int]
    tickets_available: bool

@dataclass
class CrowdData:
    """Real-time crowd level information"""
    location_id: str
    location_name: str
    current_crowd_level: str  # low, moderate, high, very_high
    peak_times: List[str]
    wait_time_minutes: Optional[int]
    last_updated: datetime
    confidence_score: float

@dataclass
class TrafficData:
    """Real-time traffic and route information"""
    origin: str
    destination: str
    duration_normal: int  # minutes
    duration_current: int  # minutes
    traffic_level: str  # light, moderate, heavy, severe
    recommended_route: str
    alternative_routes: List[Dict]
    last_updated: datetime

class RealTimeEventClient:
    """Client for fetching real-time event data"""
    
    def __init__(self):
        self.ticketmaster_key = os.getenv("TICKETMASTER_API_KEY")
        self.eventbrite_token = os.getenv("EVENTBRITE_TOKEN")
        self.facebook_token = os.getenv("FACEBOOK_ACCESS_TOKEN")
        
        # Mock data for when APIs are unavailable
        self.mock_events = self._generate_mock_events()
    
    async def get_live_events(self, date_range: int = 7, categories: Optional[List[str]] = None) -> List[EventData]:
        """Get live events in Istanbul for the next specified days"""
        events = []
        
        # Try multiple event sources
        if self.ticketmaster_key:
            try:
                tm_events = await self._fetch_ticketmaster_events(date_range, categories)
                events.extend(tm_events)
            except Exception as e:
                logger.warning(f"Ticketmaster API failed: {e}")
        
        if self.eventbrite_token:
            try:
                eb_events = await self._fetch_eventbrite_events(date_range, categories)
                events.extend(eb_events)
            except Exception as e:
                logger.warning(f"Eventbrite API failed: {e}")
        
        # If no real data available, use intelligent mock data
        if not events:
            events = self._get_intelligent_mock_events(date_range, categories)
        
        return events[:20]  # Limit to 20 events
    
    async def _fetch_ticketmaster_events(self, date_range: int, categories: Optional[List[str]]) -> List[EventData]:
        """Fetch events from Ticketmaster API"""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, using fallback method for Ticketmaster API")
            return self._fetch_ticketmaster_events_sync(date_range, categories)
            
        url = "https://app.ticketmaster.com/discovery/v2/events.json"
        
        end_date = (datetime.now() + timedelta(days=date_range)).strftime("%Y-%m-%dT23:59:59Z")
        
        params = {
            "apikey": self.ticketmaster_key,
            "city": "Istanbul",
            "countryCode": "TR",
            "startDateTime": datetime.now().strftime("%Y-%m-%dT00:00:00Z"),
            "endDateTime": end_date,
            "size": 20,
            "sort": "date,asc"
        }
        
        if categories:
            # Map categories to Ticketmaster classification
            tm_categories = self._map_to_ticketmaster_categories(categories)
            if tm_categories:
                params["classificationName"] = ",".join(tm_categories)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_ticketmaster_events(data)
                    else:
                        logger.error(f"Ticketmaster API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching Ticketmaster events: {e}")
            return []
    
    def _fetch_ticketmaster_events_sync(self, date_range: int, categories: Optional[List[str]]) -> List[EventData]:
        """Fallback synchronous method for Ticketmaster API"""
        url = "https://app.ticketmaster.com/discovery/v2/events.json"
        
        end_date = (datetime.now() + timedelta(days=date_range)).strftime("%Y-%m-%dT23:59:59Z")
        
        params = {
            "apikey": self.ticketmaster_key,
            "city": "Istanbul",
            "countryCode": "TR",
            "startDateTime": datetime.now().strftime("%Y-%m-%dT00:00:00Z"),
            "endDateTime": end_date,
            "size": 20,
            "sort": "date,asc"
        }
        
        if categories:
            # Map categories to Ticketmaster classification
            tm_categories = self._map_to_ticketmaster_categories(categories)
            if tm_categories:
                params["classificationName"] = ",".join(tm_categories)
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_ticketmaster_events(data)
            else:
                logger.error(f"Ticketmaster API error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching Ticketmaster events: {e}")
            return []
    
    async def _fetch_eventbrite_events(self, date_range: int, categories: Optional[List[str]]) -> List[EventData]:
        """Fetch events from Eventbrite API"""
        url = "https://www.eventbriteapi.com/v3/events/search/"
        
        end_date = (datetime.now() + timedelta(days=date_range)).strftime("%Y-%m-%dT23:59:59")
        
        params = {
            "location.address": "Istanbul, Turkey",
            "start_date.range_start": datetime.now().strftime("%Y-%m-%dT00:00:00"),
            "start_date.range_end": end_date,
            "expand": "venue,organizer,ticket_availability",
            "sort_by": "date"
        }
        
        headers = {
            "Authorization": f"Bearer {self.eventbrite_token}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_eventbrite_events(data)
                else:
                    logger.error(f"Eventbrite API error: {response.status}")
                    return []
    
    def _parse_ticketmaster_events(self, data: Dict) -> List[EventData]:
        """Parse Ticketmaster API response"""
        events = []
        
        if "_embedded" in data and "events" in data["_embedded"]:
            for event in data["_embedded"]["events"]:
                try:
                    event_data = EventData(
                        event_id=event["id"],
                        name=event["name"],
                        location=event.get("_embedded", {}).get("venues", [{}])[0].get("name", "Unknown"),
                        start_time=datetime.fromisoformat(event["dates"]["start"]["dateTime"].replace("Z", "+00:00")),
                        end_time=None,
                        category=event.get("classifications", [{}])[0].get("segment", {}).get("name", "Entertainment"),
                        price_range=self._extract_price_range(event.get("priceRanges", [])),
                        crowd_level=self._estimate_crowd_level(event),
                        description=event.get("info", ""),
                        venue_capacity=None,
                        tickets_available=event.get("dates", {}).get("status", {}).get("code") == "onsale"
                    )
                    events.append(event_data)
                except Exception as e:
                    logger.warning(f"Error parsing Ticketmaster event: {e}")
        
        return events
    
    def _parse_eventbrite_events(self, data: Dict) -> List[EventData]:
        """Parse Eventbrite API response"""
        events = []
        
        if "events" in data:
            for event in data["events"]:
                try:
                    start_time = datetime.fromisoformat(event["start"]["utc"].replace("Z", "+00:00"))
                    end_time = datetime.fromisoformat(event["end"]["utc"].replace("Z", "+00:00")) if event.get("end") else None
                    
                    event_data = EventData(
                        event_id=event["id"],
                        name=event["name"]["text"],
                        location=event.get("venue", {}).get("name", "Unknown"),
                        start_time=start_time,
                        end_time=end_time,
                        category=event.get("category", {}).get("name", "Other"),
                        price_range=self._extract_eventbrite_price_range(event),
                        crowd_level=self._estimate_crowd_level_eventbrite(event),
                        description=event.get("description", {}).get("text", ""),
                        venue_capacity=event.get("capacity", None),
                        tickets_available=event.get("ticket_availability", {}).get("has_available_tickets", False)
                    )
                    events.append(event_data)
                except Exception as e:
                    logger.warning(f"Error parsing Eventbrite event: {e}")
        
        return events
    
    def _generate_mock_events(self) -> List[EventData]:
        """Generate realistic mock events for Istanbul"""
        base_time = datetime.now()
        
        mock_events_data = [
            {
                "name": "Traditional Turkish Music Concert",
                "location": "Cemal Reşit Rey Concert Hall",
                "category": "Music",
                "hours_from_now": 24,
                "duration_hours": 2,
                "price_range": (150, 300),
                "crowd_level": "moderate",
                "description": "An enchanting evening of classical Turkish music featuring traditional instruments."
            },
            {
                "name": "Istanbul Art Festival",
                "location": "Taksim Square",
                "category": "Arts",
                "hours_from_now": 48,
                "duration_hours": 8,
                "price_range": (0, 0),
                "crowd_level": "high",
                "description": "Free outdoor art festival showcasing local and international artists."
            },
            {
                "name": "Bosphorus Jazz Night",
                "location": "Nardis Jazz Club",
                "category": "Music",
                "hours_from_now": 72,
                "duration_hours": 3,
                "price_range": (200, 400),
                "crowd_level": "moderate",
                "description": "Jazz performances with stunning Bosphorus views."
            },
            {
                "name": "Turkish Cuisine Workshop",
                "location": "Cooking Alaturka",
                "category": "Food",
                "hours_from_now": 96,
                "duration_hours": 4,
                "price_range": (350, 500),
                "crowd_level": "low",
                "description": "Learn to cook authentic Turkish dishes with a professional chef."
            },
            {
                "name": "Sunset Photography Tour",
                "location": "Galata Bridge",
                "category": "Tours",
                "hours_from_now": 120,
                "duration_hours": 3,
                "price_range": (180, 250),
                "crowd_level": "moderate",
                "description": "Capture Istanbul's golden hour from the best vantage points."
            }
        ]
        
        events = []
        for i, event_data in enumerate(mock_events_data):
            start_time = base_time + timedelta(hours=event_data["hours_from_now"])
            end_time = start_time + timedelta(hours=event_data["duration_hours"])
            
            event = EventData(
                event_id=f"mock_{i}",
                name=event_data["name"],
                location=event_data["location"],
                start_time=start_time,
                end_time=end_time,
                category=event_data["category"],
                price_range=event_data["price_range"] if event_data["price_range"][0] > 0 else None,
                crowd_level=event_data["crowd_level"],
                description=event_data["description"],
                venue_capacity=None,
                tickets_available=True
            )
            events.append(event)
        
        return events
    
    def _get_intelligent_mock_events(self, date_range: int, categories: Optional[List[str]]) -> List[EventData]:
        """Get filtered mock events based on criteria"""
        events = self.mock_events.copy()
        
        # Filter by date range
        max_date = datetime.now() + timedelta(days=date_range)
        events = [e for e in events if e.start_time <= max_date]
        
        # Filter by categories if specified
        if categories:
            category_map = {
                'music': ['Music'],
                'arts': ['Arts'],
                'food': ['Food'],
                'tours': ['Tours'],
                'entertainment': ['Music', 'Arts', 'Entertainment']
            }
            
            allowed_categories = []
            for cat in categories:
                allowed_categories.extend(category_map.get(cat.lower(), [cat]))
            
            events = [e for e in events if e.category in allowed_categories]
        
        return events
    
    def _extract_price_range(self, price_ranges: List[Dict]) -> Optional[Tuple[float, float]]:
        """Extract price range from Ticketmaster data"""
        if not price_ranges:
            return None
        
        min_price = float('inf')
        max_price = 0
        
        for price_range in price_ranges:
            if "min" in price_range:
                min_price = min(min_price, price_range["min"])
            if "max" in price_range:
                max_price = max(max_price, price_range["max"])
        
        if min_price == float('inf'):
            return None
        
        return (min_price, max_price)
    
    def _extract_eventbrite_price_range(self, event: Dict) -> Optional[Tuple[float, float]]:
        """Extract price range from Eventbrite data"""
        if event.get("is_free", False):
            return None
        
        # This would need to be enhanced based on actual Eventbrite API response structure
        return None
    
    def _estimate_crowd_level(self, event: Dict) -> str:
        """Estimate crowd level based on event data"""
        # Simple heuristic based on venue size and event type
        venue_name = event.get("_embedded", {}).get("venues", [{}])[0].get("name", "").lower()
        
        if any(word in venue_name for word in ["stadium", "arena", "festival"]):
            return "very_high"
        elif any(word in venue_name for word in ["theater", "concert hall", "auditorium"]):
            return "high"
        elif any(word in venue_name for word in ["club", "bar", "gallery"]):
            return "moderate"
        else:
            return "low"
    
    def _estimate_crowd_level_eventbrite(self, event: Dict) -> str:
        """Estimate crowd level for Eventbrite events"""
        capacity = event.get("capacity", 0)
        
        if capacity > 5000:
            return "very_high"
        elif capacity > 1000:
            return "high"
        elif capacity > 200:
            return "moderate"
        else:
            return "low"
    
    def _map_to_ticketmaster_categories(self, categories: List[str]) -> List[str]:
        """Map generic categories to Ticketmaster classifications"""
        mapping = {
            "music": ["Music"],
            "sports": ["Sports"],
            "arts": ["Arts & Theatre"],
            "family": ["Family"],
            "entertainment": ["Miscellaneous"]
        }
        
        tm_categories = []
        for category in categories:
            tm_categories.extend(mapping.get(category.lower(), []))
        
        return list(set(tm_categories))

class RealTimeCrowdClient:
    """Client for fetching real-time crowd level data"""
    
    def __init__(self):
        self.google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY")
        self.foursquare_key = os.getenv("FOURSQUARE_API_KEY")
        
        # Popular Istanbul locations for crowd monitoring
        self.monitored_locations = [
            {"id": "hagia_sophia", "name": "Hagia Sophia", "place_id": "ChIJlRJoFMm6yhQRTBbUHBmq4hE"},
            {"id": "blue_mosque", "name": "Blue Mosque", "place_id": "ChIJgxOI7Me6yhQRKJaBa5g-FdM"},
            {"id": "grand_bazaar", "name": "Grand Bazaar", "place_id": "ChIJN_2LCca6yhQR8wKF16rx0Xo"},
            {"id": "galata_tower", "name": "Galata Tower", "place_id": "ChIJ8z7dzdG6yhQRBgDJ1LdVCPo"},
            {"id": "taksim_square", "name": "Taksim Square", "place_id": "ChIJL9n6ute6yhQRr2E2Vs8n-N0"},
            {"id": "kadikoy_ferry", "name": "Kadıköy Ferry Terminal", "place_id": "ChIJNz0vB5rAyhQRhvNZHn7GdZE"},
            {"id": "galata_bridge", "name": "Galata Bridge", "place_id": "ChIJlVXKQs-6yhQRwTGO3V2LMzI"},
            {"id": "spice_bazaar", "name": "Spice Bazaar", "place_id": "ChIJlRJoGMm6yhQRh7Q_GovhPG4"}
        ]
    
    async def get_crowd_levels(self, location_ids: Optional[List[str]] = None) -> List[CrowdData]:
        """Get current crowd levels for specified locations"""
        if not location_ids:
            location_ids = [loc["id"] for loc in self.monitored_locations]
        
        crowd_data = []
        
        for location_id in location_ids:
            location = next((loc for loc in self.monitored_locations if loc["id"] == location_id), None)
            if location:
                crowd_info = await self._get_location_crowd_data(location)
                if crowd_info:
                    crowd_data.append(crowd_info)
        
        return crowd_data
    
    async def _get_location_crowd_data(self, location: Dict) -> Optional[CrowdData]:
        """Get crowd data for a specific location"""
        # Try Google Places API first
        if self.google_maps_key:
            try:
                google_data = await self._fetch_google_crowd_data(location)
                if google_data:
                    return google_data
            except Exception as e:
                logger.warning(f"Google crowd data failed for {location['name']}: {e}")
        
        # Fallback to intelligent mock data
        return self._generate_mock_crowd_data(location)
    
    async def _fetch_google_crowd_data(self, location: Dict) -> Optional[CrowdData]:
        """Fetch crowd data from Google Places API"""
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        
        params = {
            "place_id": location["place_id"],
            "fields": "name,popular_times,current_opening_hours",
            "key": self.google_maps_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "OK":
                        result = data.get("result", {})
                        return self._parse_google_crowd_data(location, result)
        
        return None
    
    def _parse_google_crowd_data(self, location: Dict, data: Dict) -> CrowdData:
        """Parse Google Places crowd data"""
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()  # Monday = 0
        
        # Extract popular times if available
        popular_times = data.get("popular_times", [])
        current_crowd_level = "moderate"
        peak_times = []
        wait_time = None
        
        if popular_times and len(popular_times) > current_day:
            day_data = popular_times[current_day]
            if "data" in day_data and len(day_data["data"]) > current_hour:
                current_busyness = day_data["data"][current_hour]
                
                # Convert busyness percentage to crowd level
                if current_busyness >= 80:
                    current_crowd_level = "very_high"
                    wait_time = 30
                elif current_busyness >= 60:
                    current_crowd_level = "high"
                    wait_time = 15
                elif current_busyness >= 30:
                    current_crowd_level = "moderate"
                    wait_time = 5
                else:
                    current_crowd_level = "low"
                    wait_time = 0
                
                # Find peak hours
                for hour, busyness in enumerate(day_data["data"]):
                    if busyness >= 70:
                        peak_times.append(f"{hour:02d}:00")
        
        return CrowdData(
            location_id=location["id"],
            location_name=location["name"],
            current_crowd_level=current_crowd_level,
            peak_times=peak_times,
            wait_time_minutes=wait_time,
            last_updated=datetime.now(),
            confidence_score=0.8
        )
    
    def _generate_mock_crowd_data(self, location: Dict) -> CrowdData:
        """Generate realistic mock crowd data based on time and location"""
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # Different patterns for different location types
        location_patterns = {
            "hagia_sophia": {"peak_hours": [10, 11, 14, 15], "base_crowd": "high"},
            "blue_mosque": {"peak_hours": [9, 10, 13, 14], "base_crowd": "high"},
            "grand_bazaar": {"peak_hours": [11, 12, 15, 16], "base_crowd": "moderate"},
            "galata_tower": {"peak_hours": [16, 17, 18, 19], "base_crowd": "moderate"},
            "taksim_square": {"peak_hours": [17, 18, 19, 20], "base_crowd": "high"},
            "kadikoy_ferry": {"peak_hours": [8, 9, 17, 18], "base_crowd": "moderate"},
            "galata_bridge": {"peak_hours": [18, 19, 20], "base_crowd": "moderate"},
            "spice_bazaar": {"peak_hours": [10, 11, 14, 15], "base_crowd": "moderate"}
        }
        
        pattern = location_patterns.get(location["id"], {"peak_hours": [12, 15], "base_crowd": "moderate"})
        
        # Determine current crowd level
        if current_hour in pattern["peak_hours"]:
            if current_day >= 5:  # Weekend
                current_crowd_level = "very_high"
                wait_time = 25
            else:
                current_crowd_level = "high"
                wait_time = 15
        elif current_hour in [h+1 for h in pattern["peak_hours"]] or current_hour in [h-1 for h in pattern["peak_hours"]]:
            current_crowd_level = pattern["base_crowd"]
            wait_time = 8
        else:
            current_crowd_level = "low" if current_hour < 9 or current_hour > 20 else "moderate"
            wait_time = 2
        
        peak_times = [f"{hour:02d}:00-{hour+1:02d}:00" for hour in pattern["peak_hours"]]
        
        return CrowdData(
            location_id=location["id"],
            location_name=location["name"],
            current_crowd_level=current_crowd_level,
            peak_times=peak_times,
            wait_time_minutes=wait_time,
            last_updated=datetime.now(),
            confidence_score=0.6  # Lower confidence for mock data
        )

class RealTimeTrafficClient:
    """Client for real-time traffic and route optimization"""
    
    def __init__(self):
        self.google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY")
        self.mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
    
    async def get_traffic_aware_route(self, origin: str, destination: str, mode: str = "driving") -> TrafficData:
        """Get traffic-aware route between two points in Istanbul"""
        if self.google_maps_key:
            try:
                return await self._fetch_google_traffic_data(origin, destination, mode)
            except Exception as e:
                logger.warning(f"Google traffic data failed: {e}")
        
        # Fallback to mock data
        return self._generate_mock_traffic_data(origin, destination, mode)
    
    async def _fetch_google_traffic_data(self, origin: str, destination: str, mode: str) -> TrafficData:
        """Fetch real traffic data from Google Maps"""
        url = "https://maps.googleapis.com/maps/api/directions/json"
        
        params = {
            "origin": f"{origin}, Istanbul, Turkey",
            "destination": f"{destination}, Istanbul, Turkey",
            "mode": mode,
            "departure_time": "now",
            "traffic_model": "best_guess",
            "alternatives": "true",
            "key": self.google_maps_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_google_traffic_data(origin, destination, data)
        
        # Fallback
        return self._generate_mock_traffic_data(origin, destination, mode)
    
    def _parse_google_traffic_data(self, origin: str, destination: str, data: Dict) -> TrafficData:
        """Parse Google Directions API response"""
        if data.get("status") != "OK" or not data.get("routes"):
            return self._generate_mock_traffic_data(origin, destination, "driving")
        
        main_route = data["routes"][0]
        leg = main_route["legs"][0]
        
        # Get duration in traffic vs normal duration
        duration_current = leg["duration_in_traffic"]["value"] // 60  # Convert to minutes
        duration_normal = leg["duration"]["value"] // 60
        
        # Determine traffic level
        delay_factor = duration_current / duration_normal
        if delay_factor >= 2.0:
            traffic_level = "severe"
        elif delay_factor >= 1.5:
            traffic_level = "heavy"
        elif delay_factor >= 1.2:
            traffic_level = "moderate"
        else:
            traffic_level = "light"
        
        # Extract route summary
        route_summary = main_route.get("summary", "via main roads")
        
        # Process alternative routes
        alternative_routes = []
        for route in data["routes"][1:3]:  # Up to 2 alternatives
            alt_leg = route["legs"][0]
            alternative_routes.append({
                "summary": route.get("summary", "alternative route"),
                "duration": alt_leg["duration"]["value"] // 60,
                "duration_in_traffic": alt_leg.get("duration_in_traffic", {}).get("value", alt_leg["duration"]["value"]) // 60,
                "distance": alt_leg["distance"]["text"]
            })
        
        return TrafficData(
            origin=origin,
            destination=destination,
            duration_normal=duration_normal,
            duration_current=duration_current,
            traffic_level=traffic_level,
            recommended_route=route_summary,
            alternative_routes=alternative_routes,
            last_updated=datetime.now()
        )
    
    def _generate_mock_traffic_data(self, origin: str, destination: str, mode: str) -> TrafficData:
        """Generate realistic mock traffic data"""
        current_hour = datetime.now().hour
        
        # Base travel times between major areas (in minutes)
        base_times = {
            ("sultanahmet", "taksim"): 25,
            ("sultanahmet", "kadikoy"): 45,
            ("taksim", "kadikoy"): 35,
            ("beyoglu", "sultanahmet"): 20,
            ("galata", "kadikoy"): 40,
            ("besiktas", "sultanahmet"): 30,
        }
        
        # Normalize location names
        origin_clean = origin.lower().strip()
        destination_clean = destination.lower().strip()
        
        # Try to find base time
        base_time = None
        for (o, d), time in base_times.items():
            if o in origin_clean and d in destination_clean:
                base_time = time
                break
            elif d in origin_clean and o in destination_clean:
                base_time = time
                break
        
        if not base_time:
            # Default time estimation
            base_time = 30
        
        # Apply rush hour multipliers
        if current_hour in [8, 9, 17, 18, 19]:  # Rush hours
            traffic_multiplier = 1.8
            traffic_level = "heavy"
        elif current_hour in [7, 10, 16, 20]:  # Moderate traffic
            traffic_multiplier = 1.3
            traffic_level = "moderate"
        else:
            traffic_multiplier = 1.1
            traffic_level = "light"
        
        duration_current = int(base_time * traffic_multiplier)
        
        # Generate alternative routes
        alternative_routes = [
            {
                "summary": "via Bosphorus Bridge",
                "duration": base_time + 5,
                "duration_in_traffic": int((base_time + 5) * (traffic_multiplier * 0.9)),
                "distance": f"{base_time * 1.2:.1f} km"
            },
            {
                "summary": "via metro and walking",
                "duration": base_time + 10,
                "duration_in_traffic": base_time + 12,  # Public transport less affected
                "distance": f"{base_time * 0.8:.1f} km"
            }
        ]
        
        return TrafficData(
            origin=origin,
            destination=destination,
            duration_normal=base_time,
            duration_current=duration_current,
            traffic_level=traffic_level,
            recommended_route=f"via main roads connecting {origin} to {destination}",
            alternative_routes=alternative_routes,
            last_updated=datetime.now()
        )

class RealTimeDataAggregator:
    """Aggregates and provides unified real-time data"""
    
    def __init__(self):
        self.event_client = RealTimeEventClient()
        self.crowd_client = RealTimeCrowdClient()
        self.traffic_client = RealTimeTrafficClient()
    
    async def get_comprehensive_real_time_data(
        self, 
        include_events: bool = True,
        include_crowds: bool = True,
        include_traffic: bool = False,
        origin: Optional[str] = None,
        destination: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive real-time data for Istanbul"""
        results = {}
        
        try:
            if include_events:
                events = await self.event_client.get_live_events()
                results["events"] = [self._event_to_dict(event) for event in events[:10]]
            
            if include_crowds:
                crowd_data = await self.crowd_client.get_crowd_levels()
                results["crowd_levels"] = [self._crowd_to_dict(crowd) for crowd in crowd_data]
            
            if include_traffic and origin and destination:
                traffic_data = await self.traffic_client.get_traffic_aware_route(origin, destination)
                results["traffic"] = self._traffic_to_dict(traffic_data)
        
        except Exception as e:
            logger.error(f"Error aggregating real-time data: {e}")
        
        return results
    
    def _event_to_dict(self, event: EventData) -> Dict:
        """Convert EventData to dictionary"""
        return {
            "id": event.event_id,
            "name": event.name,
            "location": event.location,
            "start_time": event.start_time.isoformat(),
            "end_time": event.end_time.isoformat() if event.end_time else None,
            "category": event.category,
            "price_range": event.price_range,
            "crowd_level": event.crowd_level,
            "description": event.description[:200] + "..." if len(event.description) > 200 else event.description,
            "tickets_available": event.tickets_available
        }
    
    def _crowd_to_dict(self, crowd: CrowdData) -> Dict:
        """Convert CrowdData to dictionary"""
        return {
            "location_id": crowd.location_id,
            "location_name": crowd.location_name,
            "current_crowd_level": crowd.current_crowd_level,
            "peak_times": crowd.peak_times,
            "wait_time_minutes": crowd.wait_time_minutes,
            "last_updated": crowd.last_updated.isoformat(),
            "confidence_score": crowd.confidence_score
        }
    
    def _traffic_to_dict(self, traffic: TrafficData) -> Dict:
        """Convert TrafficData to dictionary"""
        return {
            "origin": traffic.origin,
            "destination": traffic.destination,
            "duration_normal": traffic.duration_normal,
            "duration_current": traffic.duration_current,
            "traffic_level": traffic.traffic_level,
            "recommended_route": traffic.recommended_route,
            "alternative_routes": traffic.alternative_routes,
            "last_updated": traffic.last_updated.isoformat()
        }

# Global aggregator instance
realtime_data_aggregator = RealTimeDataAggregator()
