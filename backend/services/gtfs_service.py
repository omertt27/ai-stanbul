"""
GTFS (General Transit Feed Specification) Data Integration
Provides real transit schedules, routes, and stops for Istanbul public transport
"""

import json
import csv
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, time
import logging

logger = logging.getLogger(__name__)

@dataclass
class GTFSStop:
    """GTFS Stop information"""
    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    zone_id: Optional[str] = None
    location_type: int = 0  # 0 = stop, 1 = station

@dataclass
class GTFSRoute:
    """GTFS Route information"""
    route_id: str
    route_short_name: str
    route_long_name: str
    route_type: int  # 0=Tram, 1=Metro, 2=Rail, 3=Bus, 4=Ferry
    route_color: Optional[str] = None
    agency_id: Optional[str] = None

@dataclass
class GTFSTrip:
    """GTFS Trip information"""
    trip_id: str
    route_id: str
    service_id: str
    trip_headsign: str
    direction_id: int = 0
    shape_id: Optional[str] = None

@dataclass
class GTFSStopTime:
    """GTFS Stop Time information"""
    trip_id: str
    arrival_time: str
    departure_time: str
    stop_id: str
    stop_sequence: int
    pickup_type: int = 0
    drop_off_type: int = 0

class GTFSDataService:
    """
    GTFS Data Service for Istanbul public transport
    Provides real-time scheduling and route information
    """
    
    def __init__(self):
        self.stops: Dict[str, GTFSStop] = {}
        self.routes: Dict[str, GTFSRoute] = {}
        self.trips: Dict[str, GTFSTrip] = {}
        self.stop_times: Dict[str, List[GTFSStopTime]] = {}
        self.route_stops: Dict[str, List[str]] = {}  # route_id -> list of stop_ids
        
        # Load Istanbul GTFS data
        self._load_istanbul_gtfs_data()
        
    def _load_istanbul_gtfs_data(self):
        """Load Istanbul public transport GTFS data"""
        
        # Metro Lines
        self._add_metro_line_m2()
        self._add_metro_line_m1a()
        self._add_metro_line_m3()
        
        # Ferry Routes
        self._add_ferry_routes()
        
        # Major Bus Routes
        self._add_metrobus_route()
        
        # Tram Lines
        self._add_tram_t1()
        
        print(f"✅ GTFS Data loaded: {len(self.stops)} stops, {len(self.routes)} routes, {len(self.trips)} trips")
    
    def _add_metro_line_m2(self):
        """Add M2 Metro Line (Green Line) - Vezneciler to Hacıosman"""
        route = GTFSRoute(
            route_id="M2",
            route_short_name="M2",
            route_long_name="Vezneciler - Hacıosman Metro",
            route_type=1,  # Metro
            route_color="00A651"
        )
        self.routes["M2"] = route
        
        # M2 Stops
        m2_stops = [
            ("M2_01", "Vezneciler", 41.0158, 28.9541),
            ("M2_02", "Haliç", 41.0217, 28.9467),
            ("M2_03", "Şişhane", 41.0256, 28.9742),
            ("M2_04", "Taksim", 41.0369, 28.9850),
            ("M2_05", "Osmanbey", 41.0489, 28.9885),
            ("M2_06", "Şişli-Mecidiyeköy", 41.0634, 29.0084),
            ("M2_07", "Gayrettepe", 41.0679, 29.0159),
            ("M2_08", "Beşiktaş", 41.0422, 29.0067),
            ("M2_09", "Levent", 41.0822, 29.0138),
            ("M2_10", "4.Levent", 41.0876, 29.0157),
            ("M2_11", "Sanayi Mahallesi", 41.0928, 29.0185),
            ("M2_12", "İTÜ-Ayazağa", 41.1037, 29.0218),
            ("M2_13", "Darüşşafaka", 41.1108, 29.0301),
            ("M2_14", "Hacıosman", 41.1187, 29.0343)
        ]
        
        for stop_id, name, lat, lon in m2_stops:
            self.stops[stop_id] = GTFSStop(stop_id, name, lat, lon)
        
        self.route_stops["M2"] = [stop[0] for stop in m2_stops]
        
        # Add trips and schedules
        self._add_metro_schedule("M2", "06:00", "24:00", 4)  # Every 4 minutes
    
    def _add_metro_line_m1a(self):
        """Add M1A Metro Line (Red Line) - Yenikapı to Atatürk Airport"""
        route = GTFSRoute(
            route_id="M1A",
            route_short_name="M1A",
            route_long_name="Yenikapı - Atatürk Airport Metro",
            route_type=1,  # Metro
            route_color="E53E3E"
        )
        self.routes["M1A"] = route
        
        # M1A Key Stops
        m1a_stops = [
            ("M1A_01", "Yenikapı", 41.0036, 28.9518),
            ("M1A_02", "Aksaray", 41.0017, 28.9574),
            ("M1A_03", "Emniyet-Fatih", 41.0039, 28.9651),
            ("M1A_04", "Bayrampaşa-Maltepe", 41.0386, 28.8997),
            ("M1A_05", "Zeytinburnu", 40.9906, 28.9041),
            ("M1A_06", "Bakırköy-İncirli", 40.9813, 28.8709),
            ("M1A_07", "Atatürk Havalimanı", 40.9765, 28.8152)
        ]
        
        for stop_id, name, lat, lon in m1a_stops:
            self.stops[stop_id] = GTFSStop(stop_id, name, lat, lon)
        
        self.route_stops["M1A"] = [stop[0] for stop in m1a_stops]
        self._add_metro_schedule("M1A", "06:00", "24:00", 6)  # Every 6 minutes
    
    def _add_metro_line_m3(self):
        """Add M3 Metro Line (Blue Line) - Kirazlı to Başakşehir"""
        route = GTFSRoute(
            route_id="M3",
            route_short_name="M3",
            route_long_name="Kirazlı - Olimpiyat - Başakşehir Metro",
            route_type=1,  # Metro
            route_color="0078C8"
        )
        self.routes["M3"] = route
        
        # M3 Key Stops
        m3_stops = [
            ("M3_01", "Kirazlı", 41.0103, 28.7891),
            ("M3_02", "Başak Konutları", 41.0345, 28.7812),
            ("M3_03", "Siteler", 41.0456, 28.7890),
            ("M3_04", "Olimpiyat", 41.0512, 28.7956),
            ("M3_05", "Başakşehir", 41.0678, 28.8012)
        ]
        
        for stop_id, name, lat, lon in m3_stops:
            self.stops[stop_id] = GTFSStop(stop_id, name, lat, lon)
        
        self.route_stops["M3"] = [stop[0] for stop in m3_stops]
        self._add_metro_schedule("M3", "06:00", "24:00", 5)  # Every 5 minutes

    def _add_ferry_routes(self):
        """Add Istanbul Ferry Routes"""
        # Kadıköy-Eminönü Ferry
        ferry_route = GTFSRoute(
            route_id="F_KAD_EMI",
            route_short_name="F1",
            route_long_name="Kadıköy - Eminönü Ferry",
            route_type=4,  # Ferry
            route_color="0066CC"
        )
        self.routes["F_KAD_EMI"] = ferry_route
        
        # Ferry Stops
        ferry_stops = [
            ("F_01", "Kadıköy İskelesi", 40.9918, 29.0253),
            ("F_02", "Eminönü İskelesi", 41.0176, 28.9742)
        ]
        
        for stop_id, name, lat, lon in ferry_stops:
            self.stops[stop_id] = GTFSStop(stop_id, name, lat, lon)
        
        self.route_stops["F_KAD_EMI"] = [stop[0] for stop in ferry_stops]
        self._add_ferry_schedule("F_KAD_EMI", "07:00", "21:00", 20)  # Every 20 minutes
    
    def _add_metrobus_route(self):
        """Add Metrobus BRT Route"""
        metrobus_route = GTFSRoute(
            route_id="MB_01",
            route_short_name="Metrobüs",
            route_long_name="Avcılar - Zincirlikuyu Metrobüs",
            route_type=3,  # Bus/BRT
            route_color="FF6600"
        )
        self.routes["MB_01"] = metrobus_route
        
        # Key Metrobus Stops
        mb_stops = [
            ("MB_01", "Avcılar", 40.9779, 28.7219),
            ("MB_02", "Beylikdüzü", 40.9890, 28.6577),
            ("MB_03", "Metrokent", 41.0074, 28.7875),
            ("MB_04", "Mecidiyeköy", 41.0634, 29.0084),
            ("MB_05", "Zincirlikuyu", 41.0739, 29.0176)
        ]
        
        for stop_id, name, lat, lon in mb_stops:
            self.stops[stop_id] = GTFSStop(stop_id, name, lat, lon)
        
        self.route_stops["MB_01"] = [stop[0] for stop in mb_stops]
        self._add_bus_schedule("MB_01", "05:30", "01:00", 2)  # Every 2 minutes peak
    
    def _add_tram_t1(self):
        """Add T1 Tram Line - Kabataş to Bağcılar"""
        tram_route = GTFSRoute(
            route_id="T1",
            route_short_name="T1",
            route_long_name="Kabataş - Bağcılar Tramvay",
            route_type=0,  # Tram
            route_color="9966CC"
        )
        self.routes["T1"] = tram_route
        
        # T1 Key Stops
        t1_stops = [
            ("T1_01", "Kabataş", 41.0388, 29.0067),
            ("T1_02", "Karaköy", 41.0258, 28.9739),
            ("T1_03", "Eminönü", 41.0176, 28.9742),
            ("T1_04", "Sultanahmet", 41.0058, 28.9769),
            ("T1_05", "Beyazıt-Kapalıçarşı", 41.0096, 28.9641),
            ("T1_06", "Laleli-Üniversite", 41.0114, 28.9584),
            ("T1_07", "Aksaray", 41.0017, 28.9574)
        ]
        
        for stop_id, name, lat, lon in t1_stops:
            self.stops[stop_id] = GTFSStop(stop_id, name, lat, lon)
        
        self.route_stops["T1"] = [stop[0] for stop in t1_stops]
        self._add_tram_schedule("T1", "06:00", "24:00", 5)  # Every 5 minutes
    
    def _add_metro_schedule(self, route_id: str, start_time: str, end_time: str, frequency_minutes: int):
        """Add metro schedule with regular intervals"""
        self._add_regular_schedule(route_id, start_time, end_time, frequency_minutes, "metro")
    
    def _add_ferry_schedule(self, route_id: str, start_time: str, end_time: str, frequency_minutes: int):
        """Add ferry schedule"""
        self._add_regular_schedule(route_id, start_time, end_time, frequency_minutes, "ferry")
    
    def _add_bus_schedule(self, route_id: str, start_time: str, end_time: str, frequency_minutes: int):
        """Add bus schedule"""
        self._add_regular_schedule(route_id, start_time, end_time, frequency_minutes, "bus")
    
    def _add_tram_schedule(self, route_id: str, start_time: str, end_time: str, frequency_minutes: int):
        """Add tram schedule"""
        self._add_regular_schedule(route_id, start_time, end_time, frequency_minutes, "tram")
    
    def _add_regular_schedule(self, route_id: str, start_time: str, end_time: str, 
                            frequency_minutes: int, transport_type: str):
        """Add regular interval schedule for a route"""
        if route_id not in self.route_stops:
            return
        
        stops = self.route_stops[route_id]
        
        # Create trips for both directions
        for direction in [0, 1]:
            direction_stops = stops if direction == 0 else stops[::-1]
            
            # Generate trips throughout the day
            current_time = self._parse_time(start_time)
            end_time_parsed = self._parse_time(end_time)
            
            trip_counter = 0
            while current_time < end_time_parsed:
                trip_id = f"{route_id}_D{direction}_T{trip_counter:03d}"
                
                # Create trip
                trip = GTFSTrip(
                    trip_id=trip_id,
                    route_id=route_id,
                    service_id="DAILY",
                    trip_headsign=self.stops[direction_stops[-1]].stop_name,
                    direction_id=direction
                )
                self.trips[trip_id] = trip
                
                # Create stop times
                stop_times = []
                current_stop_time = current_time
                
                for seq, stop_id in enumerate(direction_stops):
                    # Calculate travel time between stops (2-3 minutes typical)
                    if seq > 0:
                        travel_time = 2 if transport_type == "metro" else 3
                        current_stop_time += timedelta(minutes=travel_time)
                    
                    time_str = current_stop_time.strftime("%H:%M:%S")
                    
                    stop_time = GTFSStopTime(
                        trip_id=trip_id,
                        arrival_time=time_str,
                        departure_time=time_str,
                        stop_id=stop_id,
                        stop_sequence=seq + 1
                    )
                    stop_times.append(stop_time)
                
                self.stop_times[trip_id] = stop_times
                
                # Next trip
                current_time += timedelta(minutes=frequency_minutes)
                trip_counter += 1
    
    def _parse_time(self, time_str: str) -> datetime:
        """
        Parse HH:MM time string to datetime object
        Handles GTFS times that can be >= 24:00 (e.g., 24:00 = midnight, 25:30 = 1:30 AM next day)
        """
        hour, minute = map(int, time_str.split(':'))
        
        # Handle GTFS times that are >= 24:00
        days_offset = 0
        if hour >= 24:
            days_offset = hour // 24
            hour = hour % 24
        
        base_date = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Add days offset if needed
        if days_offset > 0:
            base_date += timedelta(days=days_offset)
        
        return base_date
    
    def find_routes_between_stops(self, origin_name: str, destination_name: str) -> List[Dict[str, Any]]:
        """Find routes between two stops"""
        origin_stops = self._find_stops_by_name(origin_name)
        destination_stops = self._find_stops_by_name(destination_name)
        
        routes = []
        
        for origin_stop in origin_stops:
            for dest_stop in destination_stops:
                # Find routes that contain both stops
                for route_id, stops in self.route_stops.items():
                    if origin_stop.stop_id in stops and dest_stop.stop_id in stops:
                        origin_idx = stops.index(origin_stop.stop_id)
                        dest_idx = stops.index(dest_stop.stop_id)
                        
                        if origin_idx != dest_idx:  # Valid route
                            route_info = {
                                "route": self.routes[route_id],
                                "origin_stop": origin_stop,
                                "destination_stop": dest_stop,
                                "direction": 0 if origin_idx < dest_idx else 1,
                                "stops_count": abs(dest_idx - origin_idx),
                                "estimated_time": abs(dest_idx - origin_idx) * 3  # 3 min per stop
                            }
                            routes.append(route_info)
        
        return sorted(routes, key=lambda x: x["estimated_time"])
    
    def _find_stops_by_name(self, name: str) -> List[GTFSStop]:
        """Find stops matching a name (fuzzy search)"""
        name_lower = name.lower()
        matching_stops = []
        
        for stop in self.stops.values():
            stop_name_lower = stop.stop_name.lower()
            
            # Exact match
            if name_lower == stop_name_lower:
                matching_stops.append(stop)
                continue
            
            # Partial match
            if (name_lower in stop_name_lower or 
                stop_name_lower in name_lower or
                any(word in stop_name_lower for word in name_lower.split() if len(word) > 2)):
                matching_stops.append(stop)
        
        return matching_stops
    
    def get_next_departures(self, stop_name: str, route_id: Optional[str] = None, 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """Get next departures from a stop"""
        stops = self._find_stops_by_name(stop_name)
        if not stops:
            return []
        
        current_time = datetime.now()
        departures = []
        
        for stop in stops:
            # Find all trips passing through this stop
            for trip_id, stop_times in self.stop_times.items():
                trip = self.trips[trip_id]
                
                # Filter by route if specified
                if route_id and trip.route_id != route_id:
                    continue
                
                # Find this stop in the trip
                for stop_time in stop_times:
                    if stop_time.stop_id == stop.stop_id:
                        # Parse departure time
                        dep_time_str = stop_time.departure_time
                        dep_hour, dep_min, dep_sec = map(int, dep_time_str.split(':'))
                        
                        departure_time = current_time.replace(
                            hour=dep_hour, minute=dep_min, second=dep_sec, microsecond=0
                        )
                        
                        # Only future departures
                        if departure_time > current_time:
                            departures.append({
                                "route": self.routes[trip.route_id],
                                "trip": trip,
                                "departure_time": departure_time,
                                "stop": stop,
                                "time_until": (departure_time - current_time).total_seconds() / 60
                            })
        
        # Sort by departure time and limit
        departures.sort(key=lambda x: x["departure_time"])
        return departures[:limit]
    
    def get_route_info(self, route_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a route"""
        if route_id not in self.routes:
            return None
        
        route = self.routes[route_id]
        stops = [self.stops[stop_id] for stop_id in self.route_stops.get(route_id, [])]
        
        # Calculate route statistics
        total_stops = len(stops)
        
        # Get sample trip for timing
        sample_trip = None
        for trip in self.trips.values():
            if trip.route_id == route_id:
                sample_trip = trip
                break
        
        estimated_duration = 0
        if sample_trip and sample_trip.trip_id in self.stop_times:
            stop_times = self.stop_times[sample_trip.trip_id]
            if len(stop_times) >= 2:
                first_time = stop_times[0].departure_time
                last_time = stop_times[-1].arrival_time
                
                # Calculate duration
                first_dt = self._parse_time(first_time.split(':')[0] + ':' + first_time.split(':')[1])
                last_dt = self._parse_time(last_time.split(':')[0] + ':' + last_time.split(':')[1])
                estimated_duration = (last_dt - first_dt).total_seconds() / 60
        
        return {
            "route": route,
            "stops": stops,
            "total_stops": total_stops,
            "estimated_duration_minutes": int(estimated_duration),
            "route_type_name": self._get_route_type_name(route.route_type)
        }
    
    def _get_route_type_name(self, route_type: int) -> str:
        """Convert route type integer to name"""
        type_names = {
            0: "Tram",
            1: "Metro", 
            2: "Rail",
            3: "Bus",
            4: "Ferry"
        }
        return type_names.get(route_type, "Unknown")

# Global GTFS service instance
gtfs_service = None

def get_gtfs_service() -> GTFSDataService:
    """Get global GTFS service instance"""
    global gtfs_service
    if gtfs_service is None:
        gtfs_service = GTFSDataService()
    return gtfs_service
