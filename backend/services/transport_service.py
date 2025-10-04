"""
Transportation Service
Replaces GPT for transportation queries with GTFS data and routing algorithms.
Enhanced with real-time GTFS integration for optimal routing and schedules.
"""

import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Import GTFS service for real-time transit data
from .gtfs_service import get_gtfs_service, GTFSDataService

logger = logging.getLogger(__name__)

@dataclass
class TransportRoute:
    """Transportation route information"""
    origin: str
    destination: str
    transport_type: str
    duration_minutes: int
    cost: str
    instructions: List[str]
    distance_km: float
    
@dataclass
class TransportSchedule:
    """Transportation schedule information"""
    route_name: str
    transport_type: str
    departures: List[str]
    frequency_minutes: int
    operating_hours: str

class TransportService:
    """
    Replace GPT usage for transportation queries with structured data and algorithms
    Handles: route planning, schedules, transport options, costs
    Enhanced with GTFS integration for real-time transit data
    """
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.metro_data = self._load_metro_data()
        self.bus_data = self._load_bus_data()
        self.ferry_data = self._load_ferry_data()
        self.locations = self._load_locations()
        
        # Initialize GTFS service for real-time data
        try:
            self.gtfs_service = get_gtfs_service()
            logger.info("✅ GTFS service initialized for transport service")
        except Exception as e:
            logger.warning(f"⚠️ GTFS service initialization failed: {e}")
            self.gtfs_service = None
        
    def _load_metro_data(self) -> Dict:
        """Load Istanbul Metro system data"""
        return {
            "M1A": {
                "name": "Yenikapı - Atatürk Airport",
                "stations": [
                    "Yenikapı", "Aksaray", "Emniyet-Fatih", "Bayrampaşa-Maltepe", 
                    "Sağmalcılar", "Kocatepe", "Otogar", "Terazidere", "Davutpaşa-YTÜ",
                    "Merter", "Zeytinburnu", "Bakırköy-İncirli", "Bahçelievler", 
                    "İstoç", "Mahmutbey", "Atatürk Havalimanı"
                ],
                "color": "red",
                "operating_hours": "06:00-24:00",
                "frequency": "3-8 minutes"
            },
            "M2": {
                "name": "Vezneciler - Hacıosman", 
                "stations": [
                    "Vezneciler", "Haliç", "Şişhane", "Taksim", "Osmanbey", "Şişli-Mecidiyeköy",
                    "Gayrettepe", "Beşiktaş", "Levent", "4.Levent", "Sanayi Mahallesi", "İTÜ-Ayazağa",
                    "Darüşşafaka", "Hacıosman"
                ],
                "color": "green", 
                "operating_hours": "06:00-24:00",
                "frequency": "2-5 minutes"
            },
            "M3": {
                "name": "Kirazlı - Başakşehir/Metrokent",
                "stations": ["Kirazlı", "Bağcılar", "Başak Konutları", "Siteler", "Turgut Özal", "İkitelli Sanayi", "Ispartakule", "Başakşehir", "Metrokent"],
                "color": "blue",
                "operating_hours": "06:00-24:00", 
                "frequency": "4-7 minutes"
            }
        }
    
    def _load_bus_data(self) -> Dict:
        """Load Istanbul bus system data"""
        return {
            "metrobus": {
                "name": "Metrobüs",
                "route": "Avcılar - Zincirlikuyu - Mecidiyeköy - Beşiktaş - Kabataş",
                "frequency": "30 seconds - 2 minutes",
                "operating_hours": "05:30-01:00",
                "type": "BRT"
            },
            "regular_buses": {
                "coverage": "City-wide network",
                "payment": "Istanbul Card or contactless",
                "frequency": "5-15 minutes on main routes",
                "operating_hours": "06:00-23:00 (most routes)"
            }
        }
        
    def _load_ferry_data(self) -> Dict:
        """Load Istanbul ferry system data"""
        return {
            "bosphorus_ferries": {
                "route": "Eminönü - Karaköy - Beşiktaş - Üsküdar",
                "frequency": "15-30 minutes",
                "operating_hours": "07:00-21:00",
                "scenic": True
            },
            "golden_horn_ferries": {
                "route": "Eminönü - Hasköy - Sütlüce - Eyüp",
                "frequency": "30-45 minutes", 
                "operating_hours": "07:00-19:00"
            },
            "princess_islands": {
                "route": "Kabataş/Bostancı - Büyükada - Heybeliada - Burgazada - Kınalıada",
                "frequency": "1-2 hours",
                "operating_hours": "06:30-23:00"
            }
        }
        
    def _load_locations(self) -> Dict:
        """Load major location coordinates and transport connections"""
        return {
            "sultanahmet": {
                "name": "Sultanahmet",
                "coordinates": (41.0082, 28.9784),
                "nearest_metro": "Vezneciler (M2)",
                "transport_options": ["metro", "tram", "bus", "taxi"]
            },
            "taksim": {
                "name": "Taksim", 
                "coordinates": (41.0369, 28.9850),
                "nearest_metro": "Taksim (M2)",
                "transport_options": ["metro", "bus", "taxi", "funicular"]
            },
            "besiktas": {
                "name": "Beşiktaş",
                "coordinates": (41.0422, 29.0067),
                "nearest_metro": "Beşiktaş (M2)",
                "transport_options": ["metro", "ferry", "bus", "taxi"]
            },
            "kadikoy": {
                "name": "Kadıköy",
                "coordinates": (40.9918, 29.0253),
                "transport_options": ["ferry", "metrobus", "bus", "taxi"]
            },
            "galata_bridge": {
                "name": "Galata Bridge",
                "coordinates": (41.0191, 28.9744),
                "transport_options": ["tram", "bus", "ferry", "walking"]
            }
        }
    
    def get_route_info(self, origin: str, destination: str, 
                      departure_time: Optional[str] = None) -> List[TransportRoute]:
        """
        Get transportation routes between two locations
        Enhanced with GTFS integration for real-time optimal routing
        """
        origin_lower = origin.lower()
        destination_lower = destination.lower()
        
        # Find locations in database
        origin_data = self._find_location(origin_lower)
        destination_data = self._find_location(destination_lower)
        
        if not origin_data or not destination_data:
            return [TransportRoute(
                origin=origin,
                destination=destination,
                transport_type="unknown",
                duration_minutes=0,
                cost="Unknown",
                instructions=["Please provide more specific location names"],
                distance_km=0.0
            )]
        
        # Calculate routes using different transport modes
        routes = []
        
        # Priority 1: Try GTFS-enhanced routing for optimal real-time routes
        if self.gtfs_service:
            try:
                gtfs_route = self.get_gtfs_route_with_schedule(origin, destination, departure_time)
                if gtfs_route:
                    routes.append(gtfs_route)
                    logger.info(f"✅ GTFS route found: {origin} → {destination}")
            except Exception as e:
                logger.warning(f"GTFS routing failed: {e}")
        
        # Priority 2: Metro route (legacy fallback)
        metro_route = self._calculate_metro_route(origin_data, destination_data)
        if metro_route:
            routes.append(metro_route)
        
        # Priority 3: Ferry route (if applicable)
        ferry_route = self._calculate_ferry_route(origin_data, destination_data)
        if ferry_route:
            routes.append(ferry_route)
        
        # Bus/taxi route (always available)
        bus_route = self._calculate_bus_route(origin_data, destination_data)
        routes.append(bus_route)
        
        return routes
    
    def get_schedule_info(self, transport_type: str, route: str = "") -> TransportSchedule:
        """Get schedule information for specific transport type"""
        transport_type_lower = transport_type.lower()
        
        if "metro" in transport_type_lower:
            return self._get_metro_schedule(route)
        elif "ferry" in transport_type_lower:
            return self._get_ferry_schedule(route)
        elif "bus" in transport_type_lower:
            return self._get_bus_schedule(route)
        else:
            return TransportSchedule(
                route_name="Unknown",
                transport_type=transport_type,
                departures=[],
                frequency_minutes=0,
                operating_hours="Information not available"
            )
    
    def _find_location(self, location_name: str) -> Optional[Dict]:
        """Find location data by name with enhanced fuzzy matching"""
        location_lower = location_name.lower()
        
        # Enhanced location aliases
        location_aliases = {
            "sultanahmet": ["sultanahmet", "old city", "historic peninsula", "blue mosque area"],
            "taksim": ["taksim", "taksim square", "istiklal", "beyoglu center"],
            "galata": ["galata", "galata tower", "karakoy", "galata bridge"],
            "besiktas": ["besiktas", "beşiktaş", "dolmabahce", "vodafone park"],
            "kadikoy": ["kadikoy", "kadıköy", "moda", "asian side"],
            "uskudar": ["uskudar", "üsküdar", "maidens tower", "kiz kulesi"],
            "eminonu": ["eminonu", "eminönü", "spice bazaar", "galata bridge"],
            "sirkeci": ["sirkeci", "train station", "orient express"],
            "airport": ["airport", "havalimani", "havalimanı", "ataturk airport", "istanbul airport"]
        }
        
        # First try exact matches
        for loc_id, loc_data in self.locations.items():
            if (location_lower == loc_id or 
                location_lower == loc_data["name"].lower() or
                location_lower in loc_data["name"].lower()):
                return loc_data
        
        # Try alias matching
        for canonical_name, aliases in location_aliases.items():
            if any(alias in location_lower for alias in aliases):
                for loc_id, loc_data in self.locations.items():
                    if canonical_name in loc_id or canonical_name in loc_data["name"].lower():
                        return loc_data
        
        # Try partial word matching
        location_words = location_lower.split()
        for loc_id, loc_data in self.locations.items():
            loc_words = (loc_id + " " + loc_data["name"].lower()).split()
            if any(word in loc_words for word in location_words if len(word) > 2):
                return loc_data
        
        # If no match found, return a generic Istanbul location
        return {
            "name": location_name.title(),
            "coordinates": (41.0082, 28.9784),  # Central Istanbul
            "transport_options": ["metro", "bus", "taxi", "ferry"]
        }
    
    def _calculate_metro_route(self, origin: Dict, destination: Dict) -> Optional[TransportRoute]:
        """Calculate metro route with enhanced routing information"""
        if "metro" not in origin.get("transport_options", []) or \
           "metro" not in destination.get("transport_options", []):
            return None
        
        # Enhanced metro route mapping
        metro_routes = {
            ("sultanahmet", "taksim"): {
                "line": "M2 (Green Line)",
                "stations": ["Vezneciler", "Haliç", "Şişhane", "Taksim"],
                "duration": 25,
                "transfers": 0
            },
            ("taksim", "sultanahmet"): {
                "line": "M2 (Green Line)", 
                "stations": ["Taksim", "Şişhane", "Haliç", "Vezneciler"],
                "duration": 25,
                "transfers": 0
            },
            ("besiktas", "taksim"): {
                "line": "M2 (Green Line)",
                "stations": ["Beşiktaş", "Taksim"],
                "duration": 15,
                "transfers": 0
            }
        }
        
        # Try to find specific route
        origin_key = origin["name"].lower().replace(" ", "")
        dest_key = destination["name"].lower().replace(" ", "")
        route_key = (origin_key, dest_key)
        
        if route_key in metro_routes:
            route_info = metro_routes[route_key]
            instructions = [
                f"Take {route_info['line']} from {origin['name']}",
                f"Travel through: {' → '.join(route_info['stations'])}",
                f"Total travel time: ~{route_info['duration']} minutes",
                "Use Istanbul Card for payment (₺15 per journey)",
                "Metro operates 06:00-24:00 daily"
            ]
            
            return TransportRoute(
                origin=origin["name"],
                destination=destination["name"],
                transport_type=f"Metro ({route_info['line']})",
                duration_minutes=route_info['duration'],
                cost="₺15 with Istanbul Card",
                instructions=instructions,
                distance_km=self._calculate_distance(origin["coordinates"], destination["coordinates"])
            )
        
        # Fallback to generic metro route
        distance = self._calculate_distance(origin["coordinates"], destination["coordinates"])
        duration = max(20, int(distance * 3))
        
        return TransportRoute(
            origin=origin["name"],
            destination=destination["name"],
            transport_type="Metro",
            duration_minutes=duration,
            cost="₺15 with Istanbul Card",
            instructions=[
                f"Take Metro from {origin['name']} to {destination['name']}",
                "Use Istanbul Card for payment",
                "Follow metro signs and announcements",
                "Metro operates 06:00-24:00 daily"
            ],
            distance_km=distance
        )
    
    def _calculate_ferry_route(self, origin: Dict, destination: Dict) -> Optional[TransportRoute]:
        """Calculate ferry route with specific Bosphorus routes"""
        if "ferry" not in origin.get("transport_options", []) or \
           "ferry" not in destination.get("transport_options", []):
            return None
        
        # Specific ferry routes on the Bosphorus
        ferry_routes = {
            ("kadikoy", "eminonu"): {
                "line": "Kadıköy-Eminönü Ferry",
                "duration": 25,
                "frequency": "Every 20 minutes",
                "cost": "₺15"
            },
            ("eminonu", "kadikoy"): {
                "line": "Eminönü-Kadıköy Ferry", 
                "duration": 25,
                "frequency": "Every 20 minutes",
                "cost": "₺15"
            },
            ("uskudar", "besiktas"): {
                "line": "Üsküdar-Beşiktaş Ferry",
                "duration": 15,
                "frequency": "Every 30 minutes", 
                "cost": "₺15"
            },
            ("besiktas", "uskudar"): {
                "line": "Beşiktaş-Üsküdar Ferry",
                "duration": 15,
                "frequency": "Every 30 minutes",
                "cost": "₺15"
            }
        }
        
        # Try to find specific ferry route
        origin_key = origin["name"].lower().replace(" ", "")
        dest_key = destination["name"].lower().replace(" ", "")
        route_key = (origin_key, dest_key)
        
        if route_key in ferry_routes:
            route_info = ferry_routes[route_key]
            instructions = [
                f"Take {route_info['line']}",
                f"Departure frequency: {route_info['frequency']}",
                f"Journey time: ~{route_info['duration']} minutes",
                "Beautiful Bosphorus views during the journey",
                "Operating hours: 07:00-21:00 (varies by season)",
                "Use Istanbul Card for payment"
            ]
            
            return TransportRoute(
                origin=origin["name"],
                destination=destination["name"],
                transport_type=f"Ferry ({route_info['line']})",
                duration_minutes=route_info['duration'],
                cost=route_info['cost'] + " with Istanbul Card",
                instructions=instructions,
                distance_km=self._calculate_distance(origin["coordinates"], destination["coordinates"])
            )
        
        # Generic ferry route
        distance = self._calculate_distance(origin["coordinates"], destination["coordinates"])
        duration = max(15, int(distance * 5))
        
        return TransportRoute(
            origin=origin["name"],
            destination=destination["name"],
            transport_type="Ferry",
            duration_minutes=duration,
            cost="₺15 with Istanbul Card",
            instructions=[
                f"Take ferry from {origin['name']} to {destination['name']}",
                "Enjoy scenic Bosphorus views",
                "Check schedule - ferries run every 20-30 minutes",
                "Operating hours: 07:00-21:00"
            ],
            distance_km=distance
        )
    
    def _calculate_bus_route(self, origin: Dict, destination: Dict) -> TransportRoute:
        """Calculate bus/taxi route (always available)"""
        distance = self._calculate_distance(origin["coordinates"], destination["coordinates"])
        
        # Bus route
        bus_duration = max(15, int(distance * 4))  # Account for traffic
        
        return TransportRoute(
            origin=origin["name"],
            destination=destination["name"],
            transport_type="Bus/Taxi",
            duration_minutes=bus_duration,
            cost="₺3-5 (bus) / ₺25-50 (taxi)",
            instructions=[
                f"Take bus or taxi from {origin['name']} to {destination['name']}",
                "Bus: Use Istanbul Card, check real-time apps",
                "Taxi: Negotiate fare or use BiTaksi/Uber"
            ],
            distance_km=distance
        )
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates (Haversine formula)"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth's radius in km
        
        return round(c * r, 1)
    
    def _get_metro_schedule(self, route: str) -> TransportSchedule:
        """Get metro schedule information"""
        # Use first metro line as example
        metro_line = list(self.metro_data.values())[0]
        return TransportSchedule(
            route_name=metro_line["name"],
            transport_type="Metro",
            departures=["Every 3-8 minutes during day", "Every 10-15 minutes after 22:00"],
            frequency_minutes=5,
            operating_hours=metro_line["operating_hours"]
        )
    
    def _get_ferry_schedule(self, route: str) -> TransportSchedule:
        """Get ferry schedule information"""
        return TransportSchedule(
            route_name="Bosphorus Ferry",
            transport_type="Ferry", 
            departures=["07:00", "07:30", "08:00", "08:30", "09:00"],
            frequency_minutes=30,
            operating_hours="07:00-21:00"
        )
    
    def _get_bus_schedule(self, route: str) -> TransportSchedule:
        """Get bus schedule information"""
        return TransportSchedule(
            route_name="City Bus Network",
            transport_type="Bus",
            departures=["Every 5-15 minutes on main routes"],
            frequency_minutes=10,
            operating_hours="06:00-23:00"
        )
    
    def format_route_response(self, routes: List[TransportRoute]) -> str:
        """Format route information into readable response"""
        if not routes:
            return "No transportation routes found. Please check your locations."
        
        response_parts = [f"🚌 **Transportation Options from {routes[0].origin} to {routes[0].destination}:**\n"]
        
        for i, route in enumerate(routes, 1):
            route_info = [
                f"**Option {i}: {route.transport_type}**",
                f"⏱️ Duration: ~{route.duration_minutes} minutes",
                f"💰 Cost: {route.cost}",
                f"📏 Distance: {route.distance_km} km"
            ]
            
            if route.instructions:
                route_info.append("📋 Instructions:")
                for instruction in route.instructions:
                    route_info.append(f"• {instruction}")
            
            response_parts.append("\n".join(route_info))
            response_parts.append("")  # Empty line between routes
        
        return "\n".join(response_parts)
    
    def search_transport(self, query: str) -> str:
        """Main transportation search function with GTFS integration"""
        query_lower = query.lower()
        
        # Check for real-time schedule requests first
        if any(word in query_lower for word in ['schedule', 'time', 'when', 'next', 'departure']):
            location = self._extract_location_from_query(query)
            if location:
                schedules = self.get_real_time_schedules(location)
                if schedules:
                    return self._format_schedule_response(schedules, location)
        
        # Route planning queries
        if "from" in query_lower and "to" in query_lower:
            # Extract origin and destination
            parts = query_lower.split(" to ")
            if len(parts) == 2:
                origin = parts[0].replace("from ", "").strip()
                destination = parts[1].strip()
                
                # Check for time specification
                departure_time = self._extract_time_from_query(query)
                routes = self.get_route_info(origin, destination, departure_time)
                return self.format_route_response(routes)
        
        # Enhanced transport type specific queries with GTFS data
        elif self.gtfs_service and any(word in query_lower for word in ['metro', 'subway']):
            return self._get_enhanced_metro_info()
        elif self.gtfs_service and any(word in query_lower for word in ['ferry', 'boat']):
            return self._get_enhanced_ferry_info()
        elif self.gtfs_service and any(word in query_lower for word in ['bus', 'metrobus']):
            return self._get_enhanced_bus_info()
        
        # Legacy schedule queries (fallback)
        elif any(word in query_lower for word in ["schedule", "timetable", "hours", "frequency"]):
            if "metro" in query_lower:
                schedule = self.get_schedule_info("metro")
            elif "ferry" in query_lower:
                schedule = self.get_schedule_info("ferry")
            else:
                schedule = self.get_schedule_info("bus")
            
            return f"🕒 **{schedule.route_name} Schedule:**\n" \
                   f"⏰ Operating hours: {schedule.operating_hours}\n" \
                   f"🔄 Frequency: Every {schedule.frequency_minutes} minutes\n" \
                   f"🚌 Transport type: {schedule.transport_type}"
        
        # General transport info
        else:
            return """🚌 **Istanbul Public Transportation:**

🎫 **Payment:** Use Istanbul Card (Istanbulkart) for all public transport
🚇 **Metro:** Fast, air-conditioned, connects major districts
🚌 **Bus:** Extensive network, includes Metrobüs (BRT system)
⛴️ **Ferry:** Scenic Bosphorus routes, connects European & Asian sides
🚋 **Tram:** Historic tram in Beyoğlu, modern tram to Sultanahmet

💡 **Tips:**
• Download BiTaksi or Uber for taxis
• Use Moovit or Citymapper for real-time transit info
• Rush hours: 07:00-09:00 and 17:00-19:00
• Night buses available on weekends"""

    def get_gtfs_route_with_schedule(self, from_location: str, to_location: str, 
                                   departure_time: Optional[str] = None) -> Optional[TransportRoute]:
        """
        Enhanced route planning using GTFS data with real-time schedules
        """
        if not self.gtfs_service:
            logger.warning("GTFS service not available, falling back to basic routing")
            return None
        
        try:
            # Parse departure time or use current time
            if departure_time:
                target_time = datetime.strptime(departure_time, "%H:%M").time()
            else:
                target_time = datetime.now().time()
            
            # Find best route using GTFS data
            best_route = self._find_optimal_gtfs_route(from_location, to_location, target_time)
            
            if best_route:
                return self._create_gtfs_transport_route(best_route, from_location, to_location)
                
        except Exception as e:
            logger.warning(f"GTFS route planning failed: {e}")
        
        return None

    def _find_optimal_gtfs_route(self, from_location: str, to_location: str, 
                               target_time: datetime.time) -> Optional[Dict[str, Any]]:
        """Find optimal route using GTFS data"""
        if not self.gtfs_service:
            return None
            
        # Find nearby stops for origin and destination
        origin_stops = self._find_nearby_gtfs_stops(from_location)
        dest_stops = self._find_nearby_gtfs_stops(to_location)
        
        if not origin_stops or not dest_stops:
            return None
        
        best_route = None
        best_score = float('inf')
        
        # Try different route combinations
        for origin_stop in origin_stops[:3]:  # Limit to top 3 for performance
            for dest_stop in dest_stops[:3]:
                route = self._calculate_gtfs_route(origin_stop, dest_stop, target_time)
                if route and route.get('total_duration', float('inf')) < best_score:
                    best_route = route
                    best_score = route['total_duration']
        
        return best_route

    def _find_nearby_gtfs_stops(self, location: str) -> List[Dict[str, Any]]:
        """Find GTFS stops near a location"""
        if not self.gtfs_service:
            return []
        
        location_aliases = self._get_location_aliases(location.lower())
        nearby_stops = []
        
        for stop_id, stop in self.gtfs_service.stops.items():
            stop_name_lower = stop.stop_name.lower()
            
            # Check if location matches stop name or nearby landmarks
            for alias in location_aliases:
                if (alias in stop_name_lower or 
                    any(keyword in stop_name_lower for keyword in alias.split()) or
                    self._is_location_near_stop(location, stop)):
                    
                    nearby_stops.append({
                        'stop': stop,
                        'distance': self._estimate_walking_distance(location, stop)
                    })
                    break
        
        # Sort by distance and return closest stops
        nearby_stops.sort(key=lambda x: x['distance'])
        return nearby_stops[:5]

    def _calculate_gtfs_route(self, origin_stop: Dict, dest_stop: Dict, 
                            target_time: datetime.time) -> Optional[Dict[str, Any]]:
        """Calculate route between two GTFS stops"""
        if not self.gtfs_service:
            return None
        
        origin = origin_stop['stop']
        destination = dest_stop['stop']
        
        # Find routes that serve both stops
        connecting_routes = self._find_connecting_gtfs_routes(origin.stop_id, destination.stop_id)
        
        if not connecting_routes:
            return None
        
        best_route = None
        
        for route_info in connecting_routes:
            # Get next departures for this route
            departures = self.gtfs_service.get_next_departures(
                origin.stop_id, 
                target_time.strftime("%H:%M"),
                route_id=route_info['route_id'],
                limit=3
            )
            
            if departures:
                next_departure = departures[0]
                travel_time = route_info['estimated_duration']
                
                route = {
                    'route_id': route_info['route_id'],
                    'route_name': route_info['route_name'],
                    'origin_stop': origin,
                    'dest_stop': destination,
                    'departure_time': next_departure['departure_time'],
                    'arrival_time': next_departure.get('arrival_time', 'N/A'),
                    'travel_time': travel_time,
                    'walking_time': origin_stop['distance'] + dest_stop['distance'],
                    'total_duration': travel_time + origin_stop['distance'] + dest_stop['distance'],
                    'route_type': route_info['route_type']
                }
                
                if best_route is None or route['total_duration'] < best_route['total_duration']:
                    best_route = route
        
        return best_route

    def _find_connecting_gtfs_routes(self, origin_stop_id: str, dest_stop_id: str) -> List[Dict[str, Any]]:
        """Find GTFS routes that connect two stops"""
        if not self.gtfs_service:
            return []
        
        connecting_routes = []
        
        for route_id, stops in self.gtfs_service.route_stops.items():
            if origin_stop_id in stops and dest_stop_id in stops:
                route = self.gtfs_service.routes.get(route_id)
                if route:
                    # Calculate estimated duration based on stop sequence
                    origin_idx = stops.index(origin_stop_id)
                    dest_idx = stops.index(dest_stop_id)
                    
                    if origin_idx < dest_idx:  # Ensure correct direction
                        stop_count = dest_idx - origin_idx
                        estimated_duration = max(5, stop_count * 2)  # 2 minutes per stop minimum
                        
                        connecting_routes.append({
                            'route_id': route_id,
                            'route_name': route.route_long_name or route.route_short_name,
                            'route_type': route.route_type,
                            'estimated_duration': estimated_duration,
                            'stops_between': stop_count
                        })
        
        return connecting_routes

    def _create_gtfs_transport_route(self, gtfs_route: Dict[str, Any], 
                                   from_location: str, to_location: str) -> TransportRoute:
        """Create TransportRoute from GTFS route data"""
        route_type_names = {
            0: "Tram",
            1: "Metro", 
            2: "Rail",
            3: "Bus",
            4: "Ferry"
        }
        
        transport_type = route_type_names.get(gtfs_route['route_type'], "Transit")
        route_name = gtfs_route['route_name']
        
        instructions = [
            f"Walk to {gtfs_route['origin_stop'].stop_name} ({int(gtfs_route['walking_time'])} min walk)",
            f"Take {transport_type} {gtfs_route['route_id']} - {route_name}",
            f"Departure: {gtfs_route['departure_time']}",
            f"Travel time: {gtfs_route['travel_time']} minutes",
            f"Walk to destination from {gtfs_route['dest_stop'].stop_name} ({int(gtfs_route.get('dest_walking_time', 2))} min walk)",
            "💡 Check real-time updates on Istanbul transport apps"
        ]
        
        return TransportRoute(
            origin=from_location,
            destination=to_location,
            transport_type=f"{transport_type} ({gtfs_route['route_id']})",
            duration_minutes=int(gtfs_route['total_duration']),
            cost="₺15 with Istanbul Card",
            instructions=instructions,
            distance_km=self._estimate_route_distance(gtfs_route)
        )

    def _is_location_near_stop(self, location: str, stop) -> bool:
        """Check if a location is near a GTFS stop"""
        # Simple heuristic - check if location name appears in stop name
        location_words = location.lower().split()
        stop_words = stop.stop_name.lower().split()
        
        return any(word in stop_words for word in location_words if len(word) > 2)

    def _estimate_walking_distance(self, location: str, stop) -> float:
        """Estimate walking time to a stop (in minutes)"""
        # This is a simplified version - in practice would use actual coordinates
        if location.lower() in stop.stop_name.lower():
            return 1  # Very close
        return 5  # Default walking time

    def _estimate_route_distance(self, gtfs_route: Dict[str, Any]) -> float:
        """Estimate route distance from GTFS data"""
        # Simplified - could use actual stop coordinates
        return max(2.0, gtfs_route.get('stops_between', 5) * 0.8)

    def get_real_time_schedules(self, location: str, transport_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get real-time schedules for a location using GTFS data"""
        if not self.gtfs_service:
            return []
        
        try:
            nearby_stops = self._find_nearby_gtfs_stops(location)
            schedules = []
            
            current_time = datetime.now().time().strftime("%H:%M")
            
            for stop_info in nearby_stops[:3]:
                stop = stop_info['stop']
                
                # Get next departures for this stop
                departures = self.gtfs_service.get_next_departures(
                    stop.stop_id, 
                    current_time,
                    limit=5
                )
                
                if departures:
                    for departure in departures:
                        route = self.gtfs_service.routes.get(departure['route_id'])
                        if route and (not transport_type or self._matches_transport_type(route.route_type, transport_type)):
                            schedules.append({
                                'stop_name': stop.stop_name,
                                'route_name': route.route_short_name or route.route_long_name,
                                'destination': departure.get('trip_headsign', 'Unknown'),
                                'departure_time': departure['departure_time'],
                                'transport_type': self._get_transport_type_name(route.route_type),
                                'walking_time': int(stop_info['distance'])
                            })
            
            # Sort by departure time
            schedules.sort(key=lambda x: x['departure_time'])
            return schedules
            
        except Exception as e:
            logger.warning(f"Failed to get real-time schedules: {e}")
            return []

    def _matches_transport_type(self, route_type: int, requested_type: str) -> bool:
        """Check if route type matches requested transport type"""
        type_mapping = {
            'metro': 1,
            'bus': 3,
            'ferry': 4,
            'tram': 0
        }
        return route_type == type_mapping.get(requested_type.lower(), -1)

    def _get_transport_type_name(self, route_type: int) -> str:
        """Get transport type name from route type code"""
        names = {0: "Tram", 1: "Metro", 2: "Rail", 3: "Bus", 4: "Ferry"}
        return names.get(route_type, "Transit")
    
    def _extract_location_from_query(self, query: str) -> Optional[str]:
        """Extract location from query for schedule requests"""
        query_lower = query.lower()
        
        # Common location patterns
        location_patterns = [
            r'schedule (?:for|at|from) (.+?)(?:\?|$)',
            r'(?:when|what time) .* (?:from|at) (.+?)(?:\?|$)',
            r'next .* (?:from|at) (.+?)(?:\?|$)',
        ]
        
        import re
        for pattern in location_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1).strip()
        
        # Check for known locations in query
        for loc_name in self.locations.keys():
            if loc_name in query_lower:
                return loc_name
        
        return None

    def _extract_time_from_query(self, query: str) -> Optional[str]:
        """Extract time from query (e.g., 'at 9:30', 'leave at 14:00')"""
        import re
        
        # Look for time patterns like "9:30", "14:00", "at 9am"
        time_patterns = [
            r'(?:at|leave at|departure at)\s*(\d{1,2}:\d{2})',
            r'(?:at|leave at|departure at)\s*(\d{1,2})\s*(?:am|pm)',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1)
        
        return None

    def _format_schedule_response(self, schedules: List[Dict[str, Any]], location: str) -> str:
        """Format real-time schedule response"""
        if not schedules:
            return f"No real-time schedule information available for {location}."
        
        response_parts = [f"🚌 **Next Departures from {location.title()}:**\n"]
        
        for schedule in schedules[:6]:  # Show up to 6 next departures
            walking_info = f" ({schedule['walking_time']} min walk)" if schedule['walking_time'] > 2 else ""
            response_parts.append(
                f"🚇 **{schedule['transport_type']} {schedule['route_name']}** → {schedule['destination']}\n"
                f"   📍 From: {schedule['stop_name']}{walking_info}\n"
                f"   ⏰ Departure: {schedule['departure_time']}\n"
            )
        
        response_parts.append("\n💡 **Tips:**")
        response_parts.append("• Times are estimated based on GTFS data")
        response_parts.append("• Check mobile apps for real-time updates")
        response_parts.append("• Have your Istanbul Card ready")
        
        return "\n".join(response_parts)

    def _get_enhanced_metro_info(self) -> str:
        """Get enhanced metro information using GTFS data"""
        if not self.gtfs_service:
            return self._get_basic_metro_info()
        
        metro_routes = [route for route in self.gtfs_service.routes.values() if route.route_type == 1]
        
        if not metro_routes:
            return self._get_basic_metro_info()
        
        response_parts = ["🚇 **Istanbul Metro System:**\n"]
        
        for route in metro_routes[:5]:  # Show up to 5 metro lines
            route_info = self.gtfs_service.get_route_info(route.route_id)
            if route_info:
                response_parts.append(
                    f"**{route.route_short_name}** - {route.route_long_name}\n"
                    f"   🚉 Stops: {route_info['total_stops']}\n"
                    f"   ⏱️ Journey time: ~{route_info['estimated_duration_minutes']} minutes\n"
                )
        
        response_parts.extend([
            "\n💡 **Metro Tips:**",
            "• Operates 06:00-24:00 daily",
            "• Frequency: 2-8 minutes depending on line and time",
            "• Cost: ₺15 with Istanbul Card",
            "• Air conditioned and accessible"
        ])
        
        return "\n".join(response_parts)

    def _get_enhanced_ferry_info(self) -> str:
        """Get enhanced ferry information using GTFS data"""
        if not self.gtfs_service:
            return self._get_basic_ferry_info()
        
        ferry_routes = [route for route in self.gtfs_service.routes.values() if route.route_type == 4]
        
        if not ferry_routes:
            return self._get_basic_ferry_info()
        
        response_parts = ["⛴️ **Istanbul Ferry System:**\n"]
        
        for route in ferry_routes[:4]:  # Show up to 4 ferry routes
            route_info = self.gtfs_service.get_route_info(route.route_id)
            if route_info:
                response_parts.append(
                    f"**{route.route_short_name}** - {route.route_long_name}\n"
                    f"   🏁 Stops: {route_info['total_stops']}\n"
                    f"   ⏱️ Journey time: ~{route_info['estimated_duration_minutes']} minutes\n"
                )
        
        response_parts.extend([
            "\n💡 **Ferry Tips:**",
            "• Beautiful Bosphorus views",
            "• Operates 07:00-21:00 (varies by season)",
            "• Frequency: 15-60 minutes depending on route",
            "• Cost: ₺15 with Istanbul Card",
            "• Perfect for sightseeing while traveling"
        ])
        
        return "\n".join(response_parts)

    def _get_enhanced_bus_info(self) -> str:
        """Get enhanced bus information using GTFS data"""
        if not self.gtfs_service:
            return self._get_basic_bus_info()
        
        bus_routes = [route for route in self.gtfs_service.routes.values() if route.route_type == 3]
        
        response_parts = ["🚌 **Istanbul Bus System:**\n"]
        
        if bus_routes:
            response_parts.append(f"📊 **Network Coverage:** {len(bus_routes)} routes in GTFS data\n")
        
        # Show Metrobus specifically if available
        metrobus = next((route for route in bus_routes if 'metrobus' in route.route_long_name.lower()), None)
        if metrobus:
            route_info = self.gtfs_service.get_route_info(metrobus.route_id)
            if route_info:
                response_parts.append(
                    f"**Metrobüs (BRT)** - {metrobus.route_long_name}\n"
                    f"   🚉 Stops: {route_info['total_stops']}\n"
                    f"   ⏱️ Full journey: ~{route_info['estimated_duration_minutes']} minutes\n"
                    f"   🔄 Frequency: Every 30 seconds - 2 minutes\n"
                )
        
        response_parts.extend([
            "\n💡 **Bus Tips:**",
            "• Extensive city-wide network",
            "• Metrobüs is the fastest BRT system",
            "• Regular buses: 06:00-23:00",
            "• Night buses available on weekends",
            "• Cost: ₺15 with Istanbul Card"
        ])
        
        return "\n".join(response_parts)

    def _get_basic_metro_info(self) -> str:
        """Fallback metro info when GTFS is not available"""
        return """🚇 **Istanbul Metro System:**

**M1A** - Yenikapı to Atatürk Airport (Red Line)
**M2** - Vezneciler to Hacıosman (Green Line)  
**M3** - Kirazlı to Başakşehir (Blue Line)

⏰ Operating hours: 06:00-24:00
🔄 Frequency: 2-8 minutes
💰 Cost: ₺15 with Istanbul Card"""

    def _get_basic_ferry_info(self) -> str:
        """Fallback ferry info when GTFS is not available"""
        return """⛴️ **Istanbul Ferry System:**

**Bosphorus Ferries** - Eminönü ↔ Karaköy ↔ Beşiktaş ↔ Üsküdar
**Golden Horn** - Eminönü ↔ Hasköy ↔ Eyüp
**Prince Islands** - Kabataş/Bostancı ↔ Islands

⏰ Operating hours: 07:00-21:00
🔄 Frequency: 15-60 minutes
💰 Cost: ₺15 with Istanbul Card"""

    def _get_basic_bus_info(self) -> str:
        """Fallback bus info when GTFS is not available"""
        return """🚌 **Istanbul Bus System:**

**Metrobüs (BRT)** - Avcılar to Zincirlikuyu to Kabataş
**Regular Buses** - City-wide network

⏰ Operating hours: 06:00-23:00 (most routes)
🔄 Frequency: 30 seconds - 15 minutes
💰 Cost: ₺15 with Istanbul Card"""
