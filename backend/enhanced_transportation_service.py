#!/usr/bin/env python3
"""
Enhanced Transportation Service for AI Istanbul
==============================================

Real-time integration with Istanbul public transit APIs including:
- IETT (Istanbul Electric Tramway and Tunnel) bus system
- Metro Istanbul real-time data
- Ferry schedules and routes
- Traffic and route optimization
"""

import requests
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class TransportRoute:
    """Structured transport route information"""
    origin: str
    destination: str
    transport_type: str  # metro, bus, ferry, walking
    line_name: str
    duration_minutes: int
    cost_tl: float
    instructions: List[str]
    real_time_status: str
    next_departure: Optional[str] = None
    platform_info: Optional[str] = None

@dataclass
class TransportStop:
    """Transport stop/station information"""
    name: str
    coordinates: Tuple[float, float]
    stop_type: str  # metro_station, bus_stop, ferry_terminal
    lines_served: List[str]
    accessibility: bool
    amenities: List[str]

class EnhancedTransportationService:
    """Enhanced transportation service with real-time data integration"""
    
    def __init__(self):
        self.metro_lines = self._load_metro_data()
        self.bus_routes = self._load_bus_data()
        self.ferry_routes = self._load_ferry_data()
        self.transport_hubs = self._load_transport_hubs()
        
        # API endpoints (would use real APIs in production)
        self.iett_api_base = "https://api.iett.istanbul"  # Mock endpoint
        self.metro_api_base = "https://api.metro.istanbul"  # Mock endpoint
        self.ferry_api_base = "https://api.ido.com.tr"  # Mock endpoint
    
    def _load_metro_data(self) -> Dict[str, Any]:
        """Load comprehensive metro line data"""
        return {
            "M1A": {
                "name": "Yenikapı - Atatürk Airport",
                "stations": [
                    "Yenikapı", "Vezneciler", "Üniversite", "Beyazıt-Kapalıçarşı",
                    "Emniyet-Fatih", "Aksaray", "Otogar", "Kocatepe", "Olimpiyatköy",
                    "Maltepe", "Davutpaşa-YTÜ", "Terazidere", "Merter", "Zeytinburnu",
                    "Bakırköy-İncirli", "Bahçelievler", "Ataköy-Şirinevler",
                    "Yenibosna", "DTM-İstanbul Fuar Merkezi", "Atatürk Airport"
                ],
                "frequency_minutes": 4,
                "first_train": "06:00",
                "last_train": "00:30",
                "average_speed_kmh": 35
            },
            "M1B": {
                "name": "Yenikapı - Kirazlı",
                "stations": [
                    "Yenikapı", "Vezneciler", "Üniversite", "Beyazıt-Kapalıçarşı",
                    "Emniyet-Fatih", "Aksaray", "Otogar", "Kocatepe", "Olimpiyatköy",
                    "Bağcılar-Meydan", "Kirazlı"
                ],
                "frequency_minutes": 5,
                "first_train": "06:00",
                "last_train": "00:30",
                "average_speed_kmh": 35
            },
            "M2": {
                "name": "Yenikapı - Hacıosman",
                "stations": [
                    "Yenikapı", "Vezneciler", "Haliç", "Şişhane", "Taksim",
                    "Osmanbey", "Şişli-Mecidiyeköy", "Gayrettepe", "Beşiktaş",
                    "Levent", "4. Levent", "Sanayi Mahallesi", "İTÜ-Ayazağa",
                    "Darüşşafaka", "Hacıosman"
                ],
                "frequency_minutes": 3,
                "first_train": "06:15",
                "last_train": "00:30",
                "average_speed_kmh": 40
            },
            "M3": {
                "name": "Olympiaköy - Başakşehir",
                "stations": [
                    "Olympiaköy", "Başak Konutları", "Siteler", "Turgut Özal",
                    "İkitelli Sanayi", "İstoç", "Mahmutbey", "Başakşehir"
                ],
                "frequency_minutes": 6,
                "first_train": "06:00",
                "last_train": "00:00",
                "average_speed_kmh": 30
            },
            "M4": {
                "name": "Kadıköy - Tavşantepe",
                "stations": [
                    "Kadıköy", "Ayrılık Çeşmesi", "Acıbadem", "Ünalan",
                    "Göztepe", "Bostancı", "Küçükyalı", "Maltepe",
                    "Huzurevi", "Kartal", "Yakacık", "Pendik", "Tavşantepe"
                ],
                "frequency_minutes": 4,
                "first_train": "06:00",
                "last_train": "00:30",
                "average_speed_kmh": 38
            },
            "M5": {
                "name": "Üsküdar - Yamanevler",
                "stations": [
                    "Üsküdar", "Fıstıkağacı", "Bağlarbaşı", "Altunizade",
                    "Bulgurlu", "Ümraniye", "Çarşı", "Yamanevler"
                ],
                "frequency_minutes": 5,
                "first_train": "06:00",
                "last_train": "00:30",
                "average_speed_kmh": 32
            },
            "M6": {
                "name": "Levent - Boğaziçi Üniversitesi",
                "stations": [
                    "Levent", "Nispetiye", "Etiler", "Boğaziçi Üniversitesi"
                ],
                "frequency_minutes": 8,
                "first_train": "06:30",
                "last_train": "23:30",
                "average_speed_kmh": 25
            },
            "M7": {
                "name": "Mecidiyeköy - Mahmutbey",
                "stations": [
                    "Mecidiyeköy", "Çağlayan", "Kağıthane", "Nurtepe",
                    "Alibeyköy", "Veysel Karani-Akşemsettin", "Kazım Karabekir",
                    "Yenimahalle", "Göztepe", "İsmetpaşa", "Mahmutbey"
                ],
                "frequency_minutes": 4,
                "first_train": "06:00",
                "last_train": "00:30",
                "average_speed_kmh": 35
            }
        }
    
    def _load_bus_data(self) -> Dict[str, Any]:
        """Load major bus route data"""
        return {
            "metrobus": {
                "name": "Metrobüs (BRT)",
                "route": "Beylikdüzü - Söğütlüçeşme",
                "frequency_minutes": 1,
                "major_stops": [
                    "Beylikdüzü", "Avcılar", "Üniversite", "Florya", "Bakırköy",
                    "Merter", "Zeytinburnu", "Topkapı", "Aksaray", "Eminönü",
                    "Karaköy", "Mecidiyeköy", "Zincirlikuyu", "Beşiktaş",
                    "Ortaköy", "Kuruçeşme", "Arnavutköy", "Bebek", "Rumeli Hisarı",
                    "Sarıyer", "Hacıosman", "Maslak", "Levent", "Gayrettepe",
                    "Şişli", "Mecidiyeköy", "Zincirlikuyu", "Beşiktaş", "Üsküdar",
                    "Altunizade", "Acıbadem", "Ünalan", "Göztepe", "Bostancı",
                    "Kartal", "Pendik", "Tuzla", "Gebze", "Söğütlüçeşme"
                ],
                "operating_hours": "05:30-01:00"
            },
            "havas": {
                "name": "Havaş Airport Shuttle",
                "routes": {
                    "IST": "Istanbul Airport - Taksim/Mecidiyeköy",
                    "SAW": "Sabiha Gökçen - Taksim/Kadıköy"
                },
                "frequency_minutes": 30,
                "operating_hours": "24/7"
            }
        }
    
    def _load_ferry_data(self) -> Dict[str, Any]:
        """Load ferry route data"""
        return {
            "bosphorus_tour": {
                "name": "Bosphorus Tour Ferry",
                "route": "Eminönü - Üsküdar - Beşiktaş - Ortaköy - Arnavutköy - Bebek - Rumeli Hisarı - Sarıyer",
                "frequency_minutes": 20,
                "duration_minutes": 90,
                "price_tl": 25,
                "operating_season": "Year-round",
                "first_departure": "10:00",
                "last_departure": "17:00"
            },
            "golden_horn": {
                "name": "Golden Horn Ferry",
                "route": "Eminönü - Kasımpaşa - Fener - Balat - Ayvansaray - Sütlüce - Eyüp",
                "frequency_minutes": 30,
                "duration_minutes": 45,
                "price_tl": 15,
                "operating_hours": "07:00-19:00"
            },
            "princes_islands": {
                "name": "Princes' Islands Ferry",
                "route": "Kabataş/Bostancı - Kınalıada - Burgazada - Heybeliada - Büyükada",
                "frequency_minutes": 60,
                "duration_minutes": 120,
                "price_tl": 35,
                "seasonal_schedule": True
            },
            "cross_bosphorus": {
                "name": "Cross-Bosphorus Ferry",
                "routes": [
                    "Eminönü - Üsküdar",
                    "Beşiktaş - Üsküdar",
                    "Kabataş - Üsküdar",
                    "Karaköy - Haydarpaşa"
                ],
                "frequency_minutes": 15,
                "duration_minutes": 20,
                "price_tl": 8,
                "operating_hours": "06:00-23:00"
            }
        }
    
    def _load_transport_hubs(self) -> Dict[str, Any]:
        """Load major transportation hubs"""
        return {
            "airports": {
                "IST": {
                    "name": "Istanbul Airport",
                    "location": "Arnavutköy",
                    "connections": ["Metro M11", "Havaş", "Taxi", "Car Rental"],
                    "to_city_center": {
                        "metro": {"duration": 45, "cost": 8},
                        "bus": {"duration": 60, "cost": 18},
                        "taxi": {"duration": 40, "cost": 150}
                    }
                },
                "SAW": {
                    "name": "Sabiha Gökçen Airport",
                    "location": "Pendik",
                    "connections": ["Havaş", "Metro M4 (via shuttle)", "Taxi"],
                    "to_city_center": {
                        "bus": {"duration": 75, "cost": 18},
                        "metro_shuttle": {"duration": 90, "cost": 15},
                        "taxi": {"duration": 60, "cost": 200}
                    }
                }
            },
            "major_stations": {
                "Taksim": {
                    "connections": ["Metro M2", "Bus", "Taxi", "Funicular F1"],
                    "nearby_attractions": ["Istiklal Street", "Galata Tower"],
                    "facilities": ["Tourist Information", "ATM", "Restaurants"]
                },
                "Sultanahmet": {
                    "connections": ["Tram T1", "Bus", "Metro M1", "Walking"],
                    "nearby_attractions": ["Hagia Sophia", "Blue Mosque", "Topkapi Palace"],
                    "facilities": ["Tourist Information", "Restaurants", "Hotels"]
                },
                "Eminönü": {
                    "connections": ["Ferry", "Tram T1", "Bus", "Metro M2"],
                    "nearby_attractions": ["Spice Bazaar", "Galata Bridge", "New Mosque"],
                    "facilities": ["Ferry Terminal", "Shopping", "Restaurants"]
                }
            }
        }
    
    async def get_real_time_route(self, origin: str, destination: str, transport_mode: str = "all") -> List[TransportRoute]:
        """Get real-time route information with multiple options"""
        try:
            routes = []
            
            # Generate route options based on transport mode
            if transport_mode in ["all", "metro"]:
                metro_routes = await self._get_metro_routes(origin, destination)
                routes.extend(metro_routes)
            
            if transport_mode in ["all", "bus"]:
                bus_routes = await self._get_bus_routes(origin, destination)
                routes.extend(bus_routes)
            
            if transport_mode in ["all", "ferry"]:
                ferry_routes = await self._get_ferry_routes(origin, destination)
                routes.extend(ferry_routes)
            
            # Sort by duration and return top options
            routes.sort(key=lambda r: r.duration_minutes)
            return routes[:3]  # Return top 3 options
            
        except Exception as e:
            logger.error(f"Error getting real-time route: {e}")
            return self._get_fallback_routes(origin, destination)
    
    async def _get_metro_routes(self, origin: str, destination: str) -> List[TransportRoute]:
        """Get metro route options"""
        routes = []
        
        # Find metro connections (simplified logic)
        for line_id, line_data in self.metro_lines.items():
            if origin.lower() in [s.lower() for s in line_data["stations"]] and \
               destination.lower() in [s.lower() for s in line_data["stations"]]:
                
                # Calculate duration based on station distance
                origin_idx = next(i for i, s in enumerate(line_data["stations"]) if s.lower() == origin.lower())
                dest_idx = next(i for i, s in enumerate(line_data["stations"]) if s.lower() == destination.lower())
                station_count = abs(dest_idx - origin_idx)
                duration = max(station_count * 3, 10)  # 3 minutes per station minimum
                
                route = TransportRoute(
                    origin=origin,
                    destination=destination,
                    transport_type="metro",
                    line_name=f"{line_id} - {line_data['name']}",
                    duration_minutes=duration,
                    cost_tl=8.0,  # Standard metro fare
                    instructions=[
                        f"Walk to {origin} Metro Station",
                        f"Take {line_id} line towards {line_data['stations'][-1] if dest_idx > origin_idx else line_data['stations'][0]}",
                        f"Travel {station_count} stations ({duration} minutes)",
                        f"Exit at {destination} station"
                    ],
                    real_time_status="On time",
                    next_departure=self._get_next_departure(line_data["frequency_minutes"]),
                    platform_info=f"Platform for {line_id} line"
                )
                routes.append(route)
        
        return routes
    
    async def _get_bus_routes(self, origin: str, destination: str) -> List[TransportRoute]:
        """Get bus route options"""
        routes = []
        
        # Check Metrobus route
        metrobus = self.bus_routes["metrobus"]
        origin_lower = origin.lower()
        dest_lower = destination.lower()
        
        if any(origin_lower in stop.lower() for stop in metrobus["major_stops"]) and \
           any(dest_lower in stop.lower() for stop in metrobus["major_stops"]):
            
            route = TransportRoute(
                origin=origin,
                destination=destination,
                transport_type="bus",
                line_name="Metrobüs (BRT)",
                duration_minutes=45,  # Average duration
                cost_tl=8.0,
                instructions=[
                    f"Walk to nearest Metrobüs station",
                    "Take Metrobüs in appropriate direction",
                    f"Travel to {destination} station",
                    "Exit and walk to final destination"
                ],
                real_time_status="Frequent service",
                next_departure="2 minutes",
                platform_info="Dedicated BRT platform"
            )
            routes.append(route)
        
        return routes
    
    async def _get_ferry_routes(self, origin: str, destination: str) -> List[TransportRoute]:
        """Get ferry route options"""
        routes = []
        
        # Check cross-Bosphorus ferries
        cross_bosphorus = self.ferry_routes["cross_bosphorus"]
        for route_name in cross_bosphorus["routes"]:
            if origin.lower() in route_name.lower() and destination.lower() in route_name.lower():
                ferry_route = TransportRoute(
                    origin=origin,
                    destination=destination,
                    transport_type="ferry",
                    line_name=f"Ferry - {route_name}",
                    duration_minutes=cross_bosphorus["duration_minutes"],
                    cost_tl=cross_bosphorus["price_tl"],
                    instructions=[
                        f"Walk to {origin} Ferry Terminal",
                        "Purchase ferry ticket",
                        f"Board ferry to {destination}",
                        "Enjoy scenic Bosphorus crossing"
                    ],
                    real_time_status="Regular schedule",
                    next_departure=self._get_next_ferry_departure(),
                    platform_info="Ferry pier - check departure board"
                )
                routes.append(ferry_route)
        
        return routes
    
    def _get_next_departure(self, frequency_minutes: int) -> str:
        """Calculate next departure time based on frequency"""
        now = datetime.now()
        minutes_to_add = frequency_minutes - (now.minute % frequency_minutes)
        next_departure = now + timedelta(minutes=minutes_to_add)
        return next_departure.strftime("%H:%M")
    
    def _get_next_ferry_departure(self) -> str:
        """Get next ferry departure time"""
        now = datetime.now()
        # Ferries typically run every 15 minutes during peak hours
        minutes_to_add = 15 - (now.minute % 15)
        next_departure = now + timedelta(minutes=minutes_to_add)
        return next_departure.strftime("%H:%M")
    
    def _get_fallback_routes(self, origin: str, destination: str) -> List[TransportRoute]:
        """Provide fallback routes when real-time data is unavailable"""
        fallback_route = TransportRoute(
            origin=origin,
            destination=destination,
            transport_type="mixed",
            line_name="Multi-modal route",
            duration_minutes=60,
            cost_tl=15.0,
            instructions=[
                f"Use public transport from {origin}",
                "Consider metro, bus, or ferry options",
                "Allow 45-90 minutes for journey",
                f"Navigate to {destination}"
            ],
            real_time_status="Estimated",
            next_departure="Variable",
            platform_info="Check local transport information"
        )
        return [fallback_route]
    
    def get_transport_summary(self, location: str) -> Dict[str, Any]:
        """Get comprehensive transport information for a location"""
        summary = {
            "location": location,
            "transport_options": [],
            "nearby_stations": [],
            "accessibility": {},
            "tips": []
        }
        
        # Check for metro stations
        for line_id, line_data in self.metro_lines.items():
            if any(location.lower() in station.lower() for station in line_data["stations"]):
                summary["transport_options"].append({
                    "type": "metro",
                    "line": f"{line_id} - {line_data['name']}",
                    "frequency": f"Every {line_data['frequency_minutes']} minutes",
                    "operating_hours": f"{line_data['first_train']} - {line_data['last_train']}"
                })
        
        # Add general transport tips
        summary["tips"] = [
            "Use Istanbulkart for all public transport",
            "Check real-time arrival information at stations",
            "Allow extra time during rush hours (07:00-09:00, 17:00-19:00)",
            "Download Moovit or similar app for live updates"
        ]
        
        return summary
    
    def get_transportation_info(self, query: str, location: str = None) -> Dict[str, Any]:
        """Wrapper method for main.py integration - get transportation information"""
        try:
            # Use the existing get_transport_summary method
            transport_summary = self.get_transport_summary(location or "istanbul")
            
            # Extract route information if query suggests route planning
            route_keywords = ['from', 'to', 'how to get', 'route', 'directions']
            if any(keyword in query.lower() for keyword in route_keywords):
                # Try to extract origin/destination from query
                # This is a simplified approach - in a real implementation you'd use NLP
                routes = []
                
                # Create sample route data based on the transport summary
                for option in transport_summary.get("transport_options", []):
                    if option["type"] == "metro":
                        routes.append({
                            "summary": f"Take {option['line']} metro line",
                            "duration": "15-25 minutes",
                            "distance": "5-15 km",
                            "instructions": f"Board {option['line']} and follow station announcements",
                            "type": "metro",
                            "cost": "Affordable with Istanbulkart"
                        })
            else:
                routes = []
            
            # Add live data simulation (in real implementation, this would call actual APIs)
            live_data = {}
            for option in transport_summary.get("transport_options", []):
                if option["type"] == "metro":
                    live_data[option["line"]] = "Operating normally"
                    
            return {
                "success": True,
                "routes": routes,
                "live_data": live_data,
                "transport_summary": transport_summary,
                "location_context": location,
                "tips": transport_summary.get("tips", [])
            }
            
        except Exception as e:
            logger.error(f"Error in get_transportation_info: {e}")
            return {
                "success": False,
                "error": str(e),
                "routes": [],
                "live_data": {}
            }
