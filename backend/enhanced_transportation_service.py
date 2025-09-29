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
                "route": "Kabataş/Bostancı - Kınalıada - Burgazada - Heybeliada - Büyükada - Sedefadası - Yassıada - Sivriada - Kaşık Adası",
                "frequency_minutes": 60,
                "duration_minutes": 120,
                "price_tl": 35,
                "seasonal_schedule": True,
                "main_islands": ["Büyükada", "Heybeliada", "Burgazada", "Kınalıada"],
                "additional_islands": ["Sedefadası", "Yassıada", "Sivriada", "Kaşık Adası", "Tavşan Adası", "İncir Adası"],
                "island_descriptions": {
                    "Büyükada": "Largest island, famous for Victorian mansions and horse carriages",
                    "Heybeliada": "Second largest, home to naval academy and pine forests", 
                    "Burgazada": "Peaceful island known for its beaches and Sait Faik Museum",
                    "Kınalıada": "Smallest inhabited island, closest to mainland",
                    "Sedefadası": "Private island with exclusive resort facilities",
                    "Yassıada": "Historic island, former political prison",
                    "Sivriada": "Rocky island popular for swimming and diving",
                    "Kaşık Adası": "Small uninhabited island near Büyükada",
                    "Tavşan Adası": "Rabbit Island - small rocky islet",
                    "İncir Adası": "Fig Island - tiny uninhabited islet"
                }
            },
            "marmara_islands": {
                "name": "Marmara Islands Ferry",
                "route": "Tekirdağ/Erdek - Marmara Adası - Avşa Adası - Paşalimanı - Ekinlik Adası",
                "frequency_minutes": 180,
                "duration_minutes": 240,
                "price_tl": 85,
                "seasonal_schedule": True,
                "operating_season": "April-October",
                "islands": ["Marmara Adası", "Avşa Adası", "Ekinlik Adası", "Paşalimanı"],
                "island_descriptions": {
                    "Marmara Adası": "Famous for marble quarries and ancient Greek ruins",
                    "Avşa Adası": "Popular tourist destination with beautiful beaches and vineyards",
                    "Ekinlik Adası": "Small peaceful island with fishing villages",
                    "Paşalimanı": "Historic harbor town with Ottoman architecture"
                }
            },
            "imrali_restricted": {
                "name": "İmralı Island (Restricted Access)",
                "route": "Special permission required - prison island",
                "frequency_minutes": 0,
                "duration_minutes": 90,
                "price_tl": 0,
                "access_restricted": True,
                "description": "High-security prison island, no public access"
            },
            "bosphorus_islands": {
                "name": "Bosphorus Small Islands Tour",
                "route": "Beşiktaş - Galatasaray Adası - Suada - Various Bosphorus points",
                "frequency_minutes": 120,
                "duration_minutes": 180,
                "price_tl": 50,
                "seasonal_schedule": True,
                "islands": ["Galatasaray Adası", "Suada", "Kuruçeşme Adası"],
                "island_descriptions": {
                    "Galatasaray Adası": "Private island belonging to Galatasaray Sports Club",
                    "Suada": "Club and event venue on artificial island in Bosphorus",
                    "Kuruçeşme Adası": "Small rocky islet used for events and dining"
                }
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
        """Enhanced wrapper method for main.py integration - get comprehensive transportation information"""
        try:
            import asyncio
            
            # Enhanced route extraction with better NLP-like parsing
            query_lower = query.lower()
            routes = []
            live_data = {}
            
            # Enhanced origin/destination extraction
            origin = None
            destination = None
            
            # Ferry-specific destination mapping (expanded with all islands)
            ferry_destinations = {
                # Prince Islands
                'prince islands': 'Prince Islands',
                'princes islands': 'Prince Islands',
                'adalar': 'Prince Islands',
                'büyükada': 'Büyükada',
                'heybeliada': 'Heybeliada', 
                'burgazada': 'Burgazada',
                'kınalıada': 'Kınalıada',
                'sedefadası': 'Sedefadası',
                'yassıada': 'Yassıada',
                'sivriada': 'Sivriada',
                'kaşık adası': 'Kaşık Adası',
                'tavşan adası': 'Tavşan Adası',
                'i̇ncir adası': 'İncir Adası',
                # Marmara Islands
                'marmara islands': 'Marmara Islands',
                'marmara adası': 'Marmara Adası',
                'avşa adası': 'Avşa Adası',
                'avsa adasi': 'Avşa Adası',
                'ekinlik adası': 'Ekinlik Adası',
                'paşalimanı': 'Paşalimanı',
                'pasalimani': 'Paşalimanı',
                # Bosphorus Islands
                'galatasaray adası': 'Galatasaray Adası',
                'suada': 'Suada',
                'kuruçeşme adası': 'Kuruçeşme Adası',
                # Mainland terminals
                'üsküdar': 'Üsküdar',
                'eminönü': 'Eminönü',
                'beşiktaş': 'Beşiktaş',
                'karaköy': 'Karaköy',
                'kabataş': 'Kabataş',
                'bostancı': 'Bostancı',
                'tekirdağ': 'Tekirdağ',
                'erdek': 'Erdek'
            }
            
            # Better origin/destination extraction
            for key, value in ferry_destinations.items():
                if key in query_lower:
                    if not destination:
                        destination = value
                    elif not origin:
                        origin = value
            
            # If ferry/boat mentioned but no specific destinations found, use defaults
            if any(term in query_lower for term in ['ferry', 'ferries', 'boat', 'boats']):
                if not origin:
                    origin = "Kabataş"  # Common ferry terminal
                if not destination and 'prince' in query_lower:
                    destination = "Prince Islands"
                elif not destination:
                    destination = "Üsküdar"  # Common cross-Bosphorus destination
            
            # Generic location extraction for other transport
            if not origin and not destination:
                # Extract from common patterns
                if 'from' in query_lower and 'to' in query_lower:
                    parts = query_lower.split('from')[1].split('to')
                    if len(parts) >= 2:
                        origin = parts[0].strip().title()
                        destination = parts[1].strip().title()
                elif 'get to' in query_lower:
                    destination = query_lower.split('get to')[1].strip().title()
                    origin = location or "Taksim"
                elif 'to' in query_lower and any(term in query_lower for term in ['how', 'way', 'get']):
                    # Handle "How do I get to X?" patterns
                    destination = query_lower.split('to')[1].strip().title()
                    origin = location or "Taksim"
            
            # Use async method to get comprehensive route information
            async def get_enhanced_routes():
                try:
                    if origin and destination:
                        # Use the comprehensive route method
                        precise_routes = await self.get_precise_route_with_platforms(origin, destination)
                        if precise_routes.get('success') and precise_routes.get('route_options'):
                            return precise_routes['route_options']
                    
                    # Ferry-specific handling
                    if any(term in query_lower for term in ['ferry', 'ferries', 'boat', 'prince']):
                        ferry_routes = await self._find_ferry_routes_detailed(
                            origin or "Kabataş", 
                            destination or "Prince Islands"
                        )
                        if ferry_routes:
                            return ferry_routes
                    
                    # Metro-specific handling
                    if any(term in query_lower for term in ['metro', 'subway', 'underground']):
                        metro_routes = await self._find_metro_routes_with_platforms(
                            origin or "Taksim",
                            destination or "Sultanahmet"
                        )
                        if metro_routes:
                            return metro_routes
                    
                    # Bus-specific handling
                    if any(term in query_lower for term in ['bus', 'autobus', 'metrobus']):
                        bus_routes = await self._find_bus_routes_with_stops(
                            origin or "Taksim",
                            destination or "Sultanahmet"
                        )
                        if bus_routes:
                            return bus_routes
                    
                    return []
                    
                except Exception as e:
                    logger.error(f"Error in async route finding: {e}")
                    return []
            
            # Run async route finding
            try:
                # Use a new thread to avoid event loop conflicts
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(get_enhanced_routes())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    enhanced_routes = future.result(timeout=10)  # 10 second timeout
                
                # Convert enhanced routes to simplified format for main.py
                for route in enhanced_routes[:3]:  # Top 3 routes
                    route_summary = {
                        "summary": self._create_route_summary(route),
                        "duration": self._format_duration(route),
                        "distance": self._estimate_distance(route),
                        "instructions": self._format_instructions(route),
                        "type": route.get("transport_type", "mixed"),
                        "cost": f"₺{route.get('cost_tl', 15)}"
                    }
                    routes.append(route_summary)
                    
                    # Add live data for this route
                    if route.get("real_time_info"):
                        line_key = route.get("line_name", route.get("route_name", "Transport"))
                        live_data[line_key] = route["real_time_info"].get("status", "Operating")
                
            except Exception as e:
                logger.error(f"Error running async routes: {e}")
            
            # Fallback: get basic transport summary
            transport_summary = self.get_transport_summary(location or "istanbul")
            
            # Enhanced live data with ferry information
            if any(term in query_lower for term in ['ferry', 'ferries', 'boat']):
                live_data["Ferry Services"] = "Regular departures every 15-20 minutes"
                live_data["Prince Islands Ferry"] = "Seasonal schedule - check İDO website"
                live_data["Cross-Bosphorus Ferries"] = "Frequent service 06:00-23:00"
            
            return {
                "success": True,
                "routes": routes,
                "live_data": live_data,
                "transport_summary": transport_summary,
                "location_context": location,
                "query_analysis": {
                    "origin": origin,
                    "destination": destination,
                    "transport_types_detected": self._detect_transport_types(query_lower)
                },
                "tips": self._get_context_specific_tips(query_lower)
            }
            
        except Exception as e:
            logger.error(f"Error in get_transportation_info: {e}")
            return {
                "success": False,
                "error": str(e),
                "routes": [],
                "live_data": {}
            }
    
    def _create_route_summary(self, route: Dict) -> str:
        """Create a concise summary of the route"""
        transport_type = route.get("transport_type", "transport")
        
        if transport_type == "ferry":
            return f"Take ferry from {route.get('origin_terminal', 'terminal')} to {route.get('destination_terminal', 'destination')} ({route.get('travel_time_minutes', 20)} min crossing)"
        elif transport_type == "metro":
            return f"Take {route.get('line_name', 'metro')} from {route.get('origin_station', 'origin')} to {route.get('destination_station', 'destination')}"
        elif transport_type == "bus":
            return f"Take bus {route.get('route_id', '')} from {route.get('origin_stop', 'origin')} to {route.get('destination_stop', 'destination')}"
        else:
            return f"Multi-modal journey using {transport_type}"
    
    def _format_duration(self, route: Dict) -> str:
        """Format the total duration including walking time"""
        total_duration = route.get("total_duration_minutes", route.get("travel_time_minutes", 30))
        return f"{total_duration} minutes"
    
    def _estimate_distance(self, route: Dict) -> str:
        """Estimate distance based on route type and duration"""
        transport_type = route.get("transport_type", "mixed")
        duration = route.get("total_duration_minutes", 30)
        
        if transport_type == "ferry":
            return "5-15 km (water crossing)"
        elif transport_type == "metro":
            return f"{duration // 4}-{duration // 2} km"
        else:
            return "Variable distance"
    
    def _format_instructions(self, route: Dict) -> str:
        """Format instructions for display"""
        instructions = route.get("detailed_instructions", [])
        if instructions:
            return f"Step-by-step: {len(instructions)} detailed instructions provided"
        else:
            return f"Follow {route.get('transport_type', 'transport')} signs and announcements"
    
    def _detect_transport_types(self, query: str) -> List[str]:
        """Detect what types of transport are mentioned in the query"""
        types = []
        if any(term in query for term in ['ferry', 'ferries', 'boat']):
            types.append('ferry')
        if any(term in query for term in ['metro', 'subway']):
            types.append('metro')
        if any(term in query for term in ['bus', 'autobus', 'metrobus']):
            types.append('bus')
        if any(term in query for term in ['tram', 'tramway']):
            types.append('tram')
        if any(term in query for term in ['taxi', 'cab']):
            types.append('taxi')
        return types or ['general']
    
    def _get_context_specific_tips(self, query: str) -> List[str]:
        """Get tips specific to the query context"""
        tips = [
            "Use Istanbulkart for all public transport",
            "Check real-time arrival information at stations",
            "Allow extra time during rush hours (07:00-09:00, 17:00-19:00)"
        ]
        
        if any(term in query for term in ['ferry', 'ferries', 'boat']):
            tips.extend([
                "Ferry schedules may change due to weather conditions",
                "Prince Islands ferries are seasonal - check İDO website",
                "Marmara Islands ferries operate April-October from Tekirdağ/Erdek",
                "Bosphorus island ferries may require club membership or reservations",
                "Ferries offer scenic views of Istanbul - bring a camera!",
                "Cross-Bosphorus ferries are the cheapest way to see the city from water",
                "Some islands have car restrictions - plan for walking or bicycles",
                "Island ferries are less frequent than city ferries - check schedules carefully"
            ])
        
        if any(term in query for term in ['metro', 'subway']):
            tips.extend([
                "Metro stations have clear directional signage in Turkish and English",
                "Elevators available at all metro stations for accessibility"
            ])
        
        if any(term in query for term in ['bus', 'metrobus']):
            tips.extend([
                "Metrobus is the fastest way to cross long distances",
                "Regular buses show route numbers clearly on front and sides"
            ])
        
        return tips
    
    async def get_real_time_metro_status(self, line_id: str) -> Dict[str, Any]:
        """Get real-time metro line status and delays"""
        try:
            # In production, this would call actual Metro Istanbul API
            current_time = datetime.now()
            
            # Simulate real-time status with realistic variations
            if line_id in self.metro_lines:
                line_data = self.metro_lines[line_id]
                
                # Simulate realistic delays and disruptions
                status_options = [
                    {"status": "normal", "delay_minutes": 0, "message": "Service operating normally"},
                    {"status": "minor_delay", "delay_minutes": 2, "message": "Minor delays due to passenger volume"},
                    {"status": "moderate_delay", "delay_minutes": 5, "message": "Moderate delays due to technical issue"},
                    {"status": "service_alert", "delay_minutes": 1, "message": "Increased passenger volume during peak hours"}
                ]
                
                # Peak hours have higher chance of delays
                is_peak_hour = (7 <= current_time.hour <= 9) or (17 <= current_time.hour <= 19)
                if is_peak_hour:
                    current_status = status_options[1]  # Minor delays during peak
                else:
                    current_status = status_options[0]  # Normal service
                
                return {
                    "line_id": line_id,
                    "line_name": line_data["name"],
                    "status": current_status["status"],
                    "delay_minutes": current_status["delay_minutes"],
                    "message": current_status["message"],
                    "frequency_minutes": line_data["frequency_minutes"] + current_status["delay_minutes"],
                    "last_updated": current_time.isoformat(),
                    "next_departures": self._calculate_next_departures(line_data, current_status["delay_minutes"])
                }
            
            return {"error": f"Line {line_id} not found"}
            
        except Exception as e:
            logger.error(f"Error getting real-time metro status: {e}")
            return {"error": "Real-time data temporarily unavailable"}
    
    def _calculate_next_departures(self, line_data: Dict, delay_minutes: int) -> List[str]:
        """Calculate next 3 departure times accounting for delays"""
        current_time = datetime.now()
        base_frequency = line_data["frequency_minutes"] + delay_minutes
        
        departures = []
        for i in range(3):
            next_departure = current_time + timedelta(minutes=base_frequency * (i + 1))
            departures.append(next_departure.strftime("%H:%M"))
        
        return departures
    
    async def get_precise_route_with_platforms(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get precise route with platform information and walking directions"""
        try:
            route_options = []
            
            # Find metro routes with platform information
            metro_routes = await self._find_metro_routes_with_platforms(origin, destination)
            route_options.extend(metro_routes)
            
            # Find bus routes with stop information
            bus_routes = await self._find_bus_routes_with_stops(origin, destination)
            route_options.extend(bus_routes)
            
            # Find ferry routes if applicable
            ferry_routes = await self._find_ferry_routes_detailed(origin, destination)
            route_options.extend(ferry_routes)
            
            # Sort by total time (including walking)
            route_options.sort(key=lambda x: x.get("total_duration_minutes", 999))
            
            return {
                "success": True,
                "origin": origin,
                "destination": destination,
                "route_options": route_options[:3],  # Top 3 routes
                "generated_at": datetime.now().isoformat(),
                "includes_real_time": True
            }
            
        except Exception as e:
            logger.error(f"Error calculating precise routes: {e}")
            return {
                "success": False,
                "error": "Route calculation temporarily unavailable",
                "fallback_advice": "Use Google Maps or ask locals for directions"
            }
    
    async def _find_metro_routes_with_platforms(self, origin: str, destination: str) -> List[Dict[str, Any]]:
        """Find metro routes with detailed platform and timing information"""
        routes = []
        
        # Enhanced route finding with platform details
        for line_id, line_data in self.metro_lines.items():
            stations = line_data["stations"]
            
            origin_matches = [s for s in stations if origin.lower() in s.lower()]
            dest_matches = [s for s in stations if destination.lower() in s.lower()]
            
            if origin_matches and dest_matches:
                origin_station = origin_matches[0]
                dest_station = dest_matches[0]
                
                origin_idx = stations.index(origin_station)
                dest_idx = stations.index(dest_station)
                
                if origin_idx != dest_idx:
                    # Calculate detailed route information
                    stations_count = abs(dest_idx - origin_idx)
                    travel_time = int(stations_count * 2.5)  # ~2.5 min per station
                    
                    # Get real-time status
                    real_time_status = await self.get_real_time_metro_status(line_id)
                    
                    # Determine platform information
                    platform_info = self._get_platform_info(line_id, origin_station, dest_idx > origin_idx)
                    
                    # Enhanced walking directions
                    walking_directions = self._get_detailed_walking_directions(origin, origin_station, dest_station, destination)
                    
                    route = {
                        "transport_type": "metro",
                        "line_id": line_id,
                        "line_name": line_data["name"],
                        "origin_station": origin_station,
                        "destination_station": dest_station,
                        "stations_count": stations_count,
                        "travel_time_minutes": travel_time,
                        "walking_time_to_station": walking_directions["to_station_minutes"],
                        "walking_time_from_station": walking_directions["from_station_minutes"],
                        "total_duration_minutes": travel_time + walking_directions["to_station_minutes"] + walking_directions["from_station_minutes"],
                        "platform_info": platform_info,
                        "real_time_status": real_time_status,
                        "detailed_instructions": self._generate_detailed_metro_instructions(
                            line_id, origin_station, dest_station, platform_info, walking_directions
                        ),
                        "cost_tl": 15.0,  # Current Istanbul metro fare
                        "accessibility": "Wheelchair accessible",
                        "next_departures": real_time_status.get("next_departures", [])
                    }
                    routes.append(route)
        
        return routes
    
    def _get_platform_info(self, line_id: str, station: str, direction_forward: bool) -> Dict[str, str]:
        """Get detailed platform information for metro stations"""
        platform_data = {
            "M1A": {
                "platform_side": "Island platform (center)",
                "direction_indicators": {
                    True: "Direction: Atatürk Airport (Platform A)",
                    False: "Direction: Yenikapı (Platform B)"
                }
            },
            "M1B": {
                "platform_side": "Island platform (center)", 
                "direction_indicators": {
                    True: "Direction: Kirazlı (Platform A)",
                    False: "Direction: Yenikapı (Platform B)"
                }
            },
            "M2": {
                "platform_side": "Side platforms",
                "direction_indicators": {
                    True: "Direction: Hacıosman (Right platform)",
                    False: "Direction: Yenikapı (Left platform)"
                }
            },
            "M4": {
                "platform_side": "Island platform (center)",
                "direction_indicators": {
                    True: "Direction: Tavşantepe (Platform A)",
                    False: "Direction: Kadıköy (Platform B)"
                }
            }
        }
        
        line_platforms = platform_data.get(line_id, {
            "platform_side": "Follow station signs",
            "direction_indicators": {
                True: f"Check direction signs at {station}",
                False: f"Check direction signs at {station}"
            }
        })
        
        return {
            "platform_type": line_platforms["platform_side"],
            "direction": line_platforms["direction_indicators"][direction_forward],
            "elevator_location": "Near entrance/exit",
            "accessibility_note": "All metro stations have elevator access"
        }
    
    def _get_detailed_walking_directions(self, origin: str, origin_station: str, dest_station: str, destination: str) -> Dict[str, Any]:
        """Generate detailed walking directions with landmarks"""
        
        # Estimate walking times based on typical distances
        to_station_time = self._estimate_walking_time(origin, origin_station)
        from_station_time = self._estimate_walking_time(dest_station, destination)
        
        # Generate specific walking instructions
        to_station_directions = self._generate_walking_instructions(origin, origin_station, "to")
        from_station_directions = self._generate_walking_instructions(dest_station, destination, "from")
        
        return {
            "to_station_minutes": to_station_time,
            "from_station_minutes": from_station_time,
            "total_walking_minutes": to_station_time + from_station_time,
            "to_station_directions": to_station_directions,
            "from_station_directions": from_station_directions,
            "walking_safety_tips": [
                "Use pedestrian crossings and follow traffic signals",
                "Main streets are well-lit and generally safe for walking",
                "Ask locals for directions if needed - most speak some English"
            ]
        }
    
    def _estimate_walking_time(self, location1: str, location2: str) -> int:
        """Estimate walking time between locations"""
        # Enhanced distance estimation based on Istanbul geography
        location_distances = {
            ("sultanahmet", "vezneciler"): 8,
            ("taksim", "şişhane"): 12,
            ("kadıköy", "ayrılık çeşmesi"): 6,
            ("galata", "şişhane"): 5,
            ("beyoğlu", "taksim"): 8,
            ("üsküdar", "üsküdar"): 0,
            ("beşiktaş", "beşiktaş"): 0
        }
        
        # Check for known distances
        key1 = (location1.lower(), location2.lower())
        key2 = (location2.lower(), location1.lower())
        
        if key1 in location_distances:
            return location_distances[key1]
        elif key2 in location_distances:
            return location_distances[key2]
        
        # Default estimate based on typical Istanbul distances
        return 10  # 10 minutes default walking time
    
    def _generate_walking_instructions(self, start: str, end: str, direction: str) -> List[str]:
        """Generate specific walking instructions with landmarks"""
        
        # Location-specific walking directions
        walking_guides = {
            ("sultanahmet", "vezneciler"): [
                "Exit Sultanahmet area heading northwest",
                "Walk along Divan Yolu street (main historic road)",
                "Pass by the Çemberlitaş (Column of Constantine)",
                "Continue straight until you reach Vezneciler Metro Station",
                "Station entrance is on the right side of the street"
            ],
            ("taksim", "şişhane"): [
                "From Taksim Square, head down İstiklal Street",
                "Walk for about 10 minutes down the pedestrian street",
                "Take the historic Tünel funicular down to Karaköy",
                "Şişhane Metro Station is at the bottom of the hill",
                "Look for the 'M' metro sign"
            ],
            ("galata", "şişhane"): [
                "From Galata Tower, walk downhill towards the Golden Horn",
                "Head southeast on Galata Köprüsü Cd.",
                "Follow signs to Karaköy/Metro",
                "Şişhane station entrance is near the Galata Bridge approach"
            ]
        }
        
        key = (start.lower(), end.lower())
        if key in walking_guides:
            return walking_guides[key]
        
        # Generic walking directions
        return [
            f"Head from {start} towards {end}",
            "Follow main streets and pedestrian signs",
            "Ask locals for directions if needed",
            f"Look for metro signs (M) when approaching {end}"
        ]
    
    def _generate_detailed_metro_instructions(self, line_id: str, origin_station: str, dest_station: str, platform_info: Dict, walking_directions: Dict) -> List[str]:
        """Generate comprehensive step-by-step metro instructions"""
        
        instructions = []
        
        # Walking to station
        instructions.extend([
            f"🚶 WALK TO STATION ({walking_directions['to_station_minutes']} min):"
        ] + [f"   • {step}" for step in walking_directions['to_station_directions']])
        
        # Metro boarding
        instructions.extend([
            f"",
            f"🚇 METRO BOARDING at {origin_station}:",
            f"   • {platform_info['platform_type']}",
            f"   • {platform_info['direction']}",
            f"   • Wait for {line_id} train",
            f"   • Board the train (estimated wait: 3-5 minutes)"
        ])
        
        # Journey
        instructions.extend([
            f"",
            f"🚊 DURING JOURNEY:",
            f"   • Stay on {line_id} line",
            f"   • Listen for station announcements",
            f"   • Exit at {dest_station}",
            f"   • Follow exit signs"
        ])
        
        # Walking from station
        instructions.extend([
            f"",
            f"🚶 WALK FROM STATION ({walking_directions['from_station_minutes']} min):"
        ] + [f"   • {step}" for step in walking_directions['from_station_directions']])
        
        return instructions

    async def _find_bus_routes_with_stops(self, origin: str, destination: str) -> List[Dict[str, Any]]:
        """Find bus routes with detailed stop information and real-time data"""
        routes = []
        
        try:
            # Enhanced bus route database with real stops
            bus_routes_data = {
                "28": {
                    "route_name": "Beşiktaş - Edirnekapı",
                    "stops": [
                        "Beşiktaş", "Dolmabahçe", "Kabataş", "Tophane", "Karaköy",
                        "Eminönü", "Beyazıt", "Aksaray", "Topkapı", "Edirnekapı"
                    ],
                    "frequency_minutes": 8,
                    "operates_24h": True,
                    "route_type": "historic_route"
                },
                "25E": {
                    "route_name": "Kabataş - Sarıyer",
                    "stops": [
                        "Kabataş", "Dolmabahçe", "Beşiktaş", "Ortaköy", "Arnavutköy",
                        "Bebek", "Rumeli Hisarı", "Emirgan", "İstinye", "Sarıyer"
                    ],
                    "frequency_minutes": 12,
                    "scenic_route": True,
                    "route_type": "bosphorus_route"
                },
                "99": {
                    "route_name": "Taksim - Şişli - Mecidiyeköy",
                    "stops": [
                        "Taksim", "Harbiye", "Şişli", "Mecidiyeköy", "Gayrettepe"
                    ],
                    "frequency_minutes": 6,
                    "business_route": True,
                    "route_type": "business_district"
                },
                "15": {
                    "route_name": "Kadıköy - Üsküdar",
                    "stops": [
                        "Kadıköy", "Haydarpaşa", "Selimiye", "Üsküdar"
                    ],
                    "frequency_minutes": 10,
                    "asian_side": True,
                    "route_type": "asian_connection"
                }
            }
            
            # Find relevant bus routes
            for route_id, route_data in bus_routes_data.items():
                stops = route_data["stops"]
                
                origin_matches = [s for s in stops if origin.lower() in s.lower()]
                dest_matches = [s for s in stops if destination.lower() in s.lower()]
                
                if origin_matches and dest_matches:
                    origin_stop = origin_matches[0]
                    dest_stop = dest_matches[0]
                    
                    origin_idx = stops.index(origin_stop)
                    dest_idx = stops.index(dest_stop)
                    
                    if origin_idx != dest_idx:
                        # Get real-time bus information
                        real_time_info = await self._get_real_time_bus_info(route_id, origin_stop)
                        
                        # Calculate route details
                        stops_count = abs(dest_idx - origin_idx)
                        travel_time = stops_count * 4 + 10  # ~4 min per stop + buffer
                        
                        # Enhanced walking directions to/from bus stops
                        walking_info = self._get_bus_stop_walking_directions(origin, origin_stop, dest_stop, destination)
                        
                        route = {
                            "transport_type": "bus",
                            "route_id": route_id,
                            "route_name": route_data["route_name"],
                            "origin_stop": origin_stop,
                            "destination_stop": dest_stop,
                            "stops_count": stops_count,
                            "travel_time_minutes": travel_time,
                            "walking_time_to_stop": walking_info["to_stop_minutes"],
                            "walking_time_from_stop": walking_info["from_stop_minutes"],
                            "total_duration_minutes": travel_time + walking_info["to_stop_minutes"] + walking_info["from_stop_minutes"],
                            "frequency_minutes": route_data["frequency_minutes"],
                            "real_time_info": real_time_info,
                            "detailed_instructions": self._generate_bus_instructions(
                                route_id, route_data, origin_stop, dest_stop, walking_info, real_time_info
                            ),
                            "cost_tl": 15.0,  # Current Istanbul bus fare
                            "special_features": self._get_bus_route_features(route_data),
                            "next_arrivals": real_time_info.get("next_arrivals", [])
                        }
                        routes.append(route)
            
            return routes
            
        except Exception as e:
            logger.error(f"Error finding bus routes: {e}")
            return []
    
    async def _get_real_time_bus_info(self, route_id: str, stop_name: str) -> Dict[str, Any]:
        """Get real-time bus arrival information"""
        try:
            current_time = datetime.now()
            
            # Simulate real-time bus data (in production, would call IETT API)
            base_frequency = {
                "28": 8, "25E": 12, "99": 6, "15": 10
            }.get(route_id, 10)
            
            # Account for peak hour variations
            is_peak = (7 <= current_time.hour <= 9) or (17 <= current_time.hour <= 19)
            delay_factor = 1.5 if is_peak else 1.0
            
            adjusted_frequency = int(base_frequency * delay_factor)
            
            # Calculate next arrivals
            next_arrivals = []
            for i in range(3):
                arrival_time = current_time + timedelta(minutes=adjusted_frequency * (i + 1))
                next_arrivals.append(arrival_time.strftime("%H:%M"))
            
            return {
                "route_id": route_id,
                "stop_name": stop_name,
                "status": "on_time" if not is_peak else "minor_delays",
                "next_arrivals": next_arrivals,
                "frequency_minutes": adjusted_frequency,
                "last_updated": current_time.isoformat(),
                "message": "Service operating normally" if not is_peak else "Minor delays due to traffic"
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time bus info: {e}")
            return {"error": "Real-time data temporarily unavailable"}
    
    def _get_bus_stop_walking_directions(self, origin: str, origin_stop: str, dest_stop: str, destination: str) -> Dict[str, Any]:
        """Get detailed walking directions to/from bus stops"""
        
        # Bus stop walking times (typically shorter than metro)
        to_stop_time = max(3, self._estimate_walking_time(origin, origin_stop) - 3)
        from_stop_time = max(3, self._estimate_walking_time(dest_stop, destination) - 3)
        
        # Generate bus stop specific directions
        to_stop_directions = self._generate_bus_stop_directions(origin, origin_stop, "to")
        from_stop_directions = self._generate_bus_stop_directions(dest_stop, destination, "from")
        
        return {
            "to_stop_minutes": to_stop_time,
            "from_stop_minutes": from_stop_time,
            "total_walking_minutes": to_stop_time + from_stop_time,
            "to_stop_directions": to_stop_directions,
            "from_stop_directions": from_stop_directions,
            "bus_stop_features": [
                "Bus stops have digital arrival displays",
                "Look for red IETT bus stop signs",
                "Many stops have covered waiting areas"
            ]
        }
    
    def _generate_bus_stop_directions(self, start: str, end: str, direction: str) -> List[str]:
        """Generate specific directions to bus stops"""
        
        bus_stop_guides = {
            ("sultanahmet", "eminönü"): [
                "Walk north from Sultanahmet Square",
                "Head towards the Golden Horn waterfront",
                "Bus stop is near the Spice Bazaar",
                "Look for the large IETT bus stop sign"
            ],
            ("taksim", "taksim"): [
                "Bus stops are located around Taksim Square",
                "Main stops are on the north side of the square",
                "Look for route number displays",
                "Ask information booth for specific route locations"
            ],
            ("beşiktaş", "beşiktaş"): [
                "Head to Beşiktaş ferry terminal area",
                "Bus stops are along the main coastal road",
                "Multiple stops serve different routes",
                "Check route numbers on stop signs"
            ]
        }
        
        key = (start.lower(), end.lower())
        if key in bus_stop_guides:
            return bus_stop_guides[key]
        
        return [
            f"Walk towards the main road from {start}",
            f"Look for IETT bus stop signs",
            f"Bus stops are typically on major streets",
            f"Check route numbers match your intended bus"
        ]
    
    def _generate_bus_instructions(self, route_id: str, route_data: Dict, origin_stop: str, dest_stop: str, walking_info: Dict, real_time_info: Dict) -> List[str]:
        """Generate comprehensive bus journey instructions"""
        
        instructions = []
        
        # Walking to bus stop
        instructions.extend([
            f"🚶 WALK TO BUS STOP ({walking_info['to_stop_minutes']} min):"
        ] + [f"   • {step}" for step in walking_info['to_stop_directions']])
        
        # Bus boarding
        next_arrival = real_time_info.get("next_arrivals", ["Check display"])[0]
        instructions.extend([
            f"",
            f"🚌 BUS BOARDING at {origin_stop}:",
            f"   • Route: {route_id} - {route_data['route_name']}",
            f"   • Next bus: {next_arrival}",
            f"   • Wait at the bus stop sign",
            f"   • Have your Istanbulkart ready",
            f"   • Board from the front door"
        ])
        
        # Journey
        instructions.extend([
            f"",
            f"🚍 DURING JOURNEY:",
            f"   • Stay on Route {route_id}",
            f"   • Listen for stop announcements",
            f"   • Press stop button before {dest_stop}",
            f"   • Exit from rear door when stopped"
        ])
        
        # Walking from bus stop
        instructions.extend([
            f"",
            f"🚶 WALK FROM BUS STOP ({walking_info['from_stop_minutes']} min):"
        ] + [f"   • {step}" for step in walking_info['from_stop_directions']])
        
        return instructions
    
    def _get_bus_route_features(self, route_data: Dict) -> List[str]:
        """Get special features of bus routes"""
        features = []
        
        if route_data.get("operates_24h"):
            features.append("24-hour service")
        if route_data.get("scenic_route"):
            features.append("Scenic Bosphorus views")
        if route_data.get("business_route"):
            features.append("Business district connection")
        if route_data.get("historic_route"):
            features.append("Historic route through old city")
        if route_data.get("asian_side"):
            features.append("Asian side connection")
            
        return features if features else ["Standard city bus service"]
    
    async def _find_ferry_routes_detailed(self, origin: str, destination: str) -> List[Dict[str, Any]]:
        """Find detailed ferry routes with timing and boarding information"""
        routes = []
        
        try:
            # Enhanced ferry route analysis
            for ferry_type, ferry_data in self.ferry_routes.items():
                
                if ferry_type == "cross_bosphorus":
                    # Check cross-Bosphorus routes
                    for route in ferry_data["routes"]:
                        route_parts = route.split(" - ")
                        if len(route_parts) == 2:
                            route_origin, route_dest = route_parts
                            
                            if (origin.lower() in route_origin.lower() and destination.lower() in route_dest.lower()) or \
                               (origin.lower() in route_dest.lower() and destination.lower() in route_origin.lower()):
                                
                                # Get real-time ferry information
                                real_time_info = await self._get_real_time_ferry_info(route)
                                
                                # Enhanced walking directions to ferry terminals
                                walking_info = self._get_ferry_terminal_walking_directions(origin, route_origin, route_dest, destination)
                                
                                ferry_route = {
                                    "transport_type": "ferry",
                                    "route_name": route,
                                    "ferry_type": "cross_bosphorus",
                                    "origin_terminal": route_origin,
                                    "destination_terminal": route_dest,
                                    "travel_time_minutes": ferry_data["duration_minutes"],
                                    "walking_time_to_terminal": walking_info["to_terminal_minutes"],
                                    "walking_time_from_terminal": walking_info["from_terminal_minutes"],
                                    "total_duration_minutes": ferry_data["duration_minutes"] + walking_info["to_terminal_minutes"] + walking_info["from_terminal_minutes"],
                                    "frequency_minutes": ferry_data["frequency_minutes"],
                                    "cost_tl": ferry_data["price_tl"],
                                    "real_time_info": real_time_info,
                                    "detailed_instructions": self._generate_ferry_instructions(
                                        route, ferry_data, walking_info, real_time_info
                                    ),
                                    "scenic_value": "Excellent Bosphorus views during crossing",
                                    "operating_hours": ferry_data["operating_hours"],
                                    "next_departures": real_time_info.get("next_departures", [])
                                }
                                routes.append(ferry_route)
                
                elif ferry_type == "princes_islands":
                    # Check Prince Islands routes - expanded island detection
                    island_terms = [
                        'prince', 'princes', 'adalar', 'büyükada', 'heybeliada', 'burgazada', 'kınalıada',
                        'sedefadası', 'yassıada', 'sivriada', 'kaşık adası', 'tavşan adası', 'i̇ncir adası',
                        'island', 'islands', 'ada', 'sedef', 'yassı', 'sivri', 'kaşık', 'tavşan', 'i̇ncir'
                    ]
                    if any(term in destination.lower() for term in island_terms) or any(term in origin.lower() for term in island_terms):
                        # Determine best origin terminal
                        origin_terminal = "Kabataş"
                        if any(term in origin.lower() for term in ['bostancı', 'kartal', 'maltepe']):
                            origin_terminal = "Bostancı"
                        
                        # Get real-time ferry information
                        real_time_info = await self._get_real_time_ferry_info(f"{origin_terminal} - Prince Islands")
                        
                        # Enhanced walking directions to ferry terminals
                        walking_info = self._get_ferry_terminal_walking_directions(origin, origin_terminal, "Prince Islands", destination)
                        
                        islands_route = {
                            "transport_type": "ferry",
                            "route_name": f"{origin_terminal} - Prince Islands",
                            "ferry_type": "princes_islands",
                            "origin_terminal": origin_terminal,
                            "destination_terminal": "Prince Islands (Büyükada, Heybeliada, Burgazada, Kınalıada, Sedefadası, Yassıada, Sivriada, Kaşık Adası)",
                            "travel_time_minutes": ferry_data["duration_minutes"],
                            "walking_time_to_terminal": walking_info["to_terminal_minutes"],
                            "walking_time_from_terminal": walking_info["from_terminal_minutes"],
                            "total_duration_minutes": ferry_data["duration_minutes"] + walking_info["to_terminal_minutes"] + walking_info["from_terminal_minutes"],
                            "frequency_minutes": ferry_data["frequency_minutes"],
                            "cost_tl": ferry_data["price_tl"],
                            "real_time_info": real_time_info,
                            "detailed_instructions": self._generate_ferry_instructions(
                                f"{origin_terminal} - Prince Islands", ferry_data, walking_info, real_time_info
                            ),
                            "scenic_value": "Beautiful island hopping experience with sea views",
                            "operating_hours": "Seasonal schedule - typically 07:00-19:00",
                            "special_features": [
                                "Multiple island stops (8+ islands)", 
                                "Seasonal service", 
                                "Car-free main islands", 
                                "Historic wooden mansions",
                                "Swimming and diving spots",
                                "Pine forests and beaches",
                                "Horse carriage tours on Büyükada",
                                "Museums and cultural sites"
                            ],
                            "seasonal_note": "Check İDO website for current seasonal schedule",
                            "next_departures": real_time_info.get("next_departures", [])
                        }
                        routes.append(islands_route)
                
                elif ferry_type == "bosphorus_tour":
                    # Check if route is suitable for Bosphorus tour
                    if any(term in origin.lower() + destination.lower() for term in ['eminönü', 'üsküdar', 'beşiktaş', 'ortaköy']):
                        walking_info = self._get_ferry_terminal_walking_directions(origin, "Eminönü", "Various stops", destination)
                        
                        tour_route = {
                            "transport_type": "ferry",
                            "route_name": "Bosphorus Sightseeing Ferry",
                            "ferry_type": "tourist_ferry",
                            "origin_terminal": "Eminönü Pier",
                            "destination_terminal": "Multiple scenic stops",
                            "travel_time_minutes": ferry_data["duration_minutes"],
                            "walking_time_to_terminal": walking_info["to_terminal_minutes"],
                            "walking_time_from_terminal": walking_info["from_terminal_minutes"],
                            "total_duration_minutes": ferry_data["duration_minutes"] + walking_info["to_terminal_minutes"] + walking_info["from_terminal_minutes"],
                            "frequency_minutes": ferry_data["frequency_minutes"],
                            "cost_tl": ferry_data["price_tl"],
                            "scenic_value": "Premium Bosphorus sightseeing experience",
                            "operating_hours": f"{ferry_data['first_departure']} - {ferry_data['last_departure']}",
                            "special_features": ["Scenic tour", "Multiple photo stops", "Audio guide available"]
                        }
                        routes.append(tour_route)
                
                elif ferry_type == "marmara_islands":
                    # Check Marmara Islands routes
                    marmara_terms = [
                        'marmara', 'avşa', 'avsa', 'ekinlik', 'paşalimanı', 'pasalimani', 'marble'
                    ]
                    if any(term in destination.lower() for term in marmara_terms) or any(term in origin.lower() for term in marmara_terms):
                        # Determine best origin terminal
                        origin_terminal = "Tekirdağ"
                        if any(term in origin.lower() for term in ['erdek', 'bandırma']):
                            origin_terminal = "Erdek"
                        
                        # Get real-time ferry information
                        real_time_info = await self._get_real_time_ferry_info(f"{origin_terminal} - Marmara Islands")
                        
                        # Enhanced walking directions
                        walking_info = self._get_ferry_terminal_walking_directions(origin, origin_terminal, "Marmara Islands", destination)
                        
                        marmara_route = {
                            "transport_type": "ferry",
                            "route_name": f"{origin_terminal} - Marmara Islands",
                            "ferry_type": "marmara_islands",
                            "origin_terminal": origin_terminal,
                            "destination_terminal": "Marmara Islands (Marmara Adası, Avşa Adası, Ekinlik Adası, Paşalimanı)",
                            "travel_time_minutes": ferry_data["duration_minutes"],
                            "walking_time_to_terminal": walking_info["to_terminal_minutes"],
                            "walking_time_from_terminal": walking_info["from_terminal_minutes"],
                            "total_duration_minutes": ferry_data["duration_minutes"] + walking_info["to_terminal_minutes"] + walking_info["from_terminal_minutes"],
                            "frequency_minutes": ferry_data["frequency_minutes"],
                            "cost_tl": ferry_data["price_tl"],
                            "real_time_info": real_time_info,
                            "detailed_instructions": self._generate_ferry_instructions(
                                f"{origin_terminal} - Marmara Islands", ferry_data, walking_info, real_time_info
                            ),
                            "scenic_value": "Beautiful Sea of Marmara crossing with vineyard islands",
                            "operating_hours": f"Seasonal service ({ferry_data['operating_season']})",
                            "special_features": [
                                "Famous marble quarries on Marmara Island",
                                "Vineyards and wine tasting on Avşa Island",
                                "Ancient Greek and Roman ruins",
                                "Fishing villages and traditional cuisine",
                                "Sandy beaches and clear waters",
                                "Seasonal service (April-October)",
                                "Less crowded than Prince Islands"
                            ],
                            "seasonal_note": f"Operating season: {ferry_data['operating_season']}",
                            "next_departures": real_time_info.get("next_departures", [])
                        }
                        routes.append(marmara_route)
                
                elif ferry_type == "bosphorus_islands":
                    # Check Bosphorus small islands
                    bosphorus_island_terms = [
                        'galatasaray adası', 'suada', 'kuruçeşme', 'galatasaray island'
                    ]
                    if any(term in destination.lower() for term in bosphorus_island_terms) or any(term in origin.lower() for term in bosphorus_island_terms):
                        origin_terminal = "Beşiktaş"
                        
                        # Get real-time ferry information
                        real_time_info = await self._get_real_time_ferry_info("Beşiktaş - Bosphorus Islands")
                        
                        # Enhanced walking directions
                        walking_info = self._get_ferry_terminal_walking_directions(origin, origin_terminal, "Bosphorus Islands", destination)
                        
                        bosphorus_islands_route = {
                            "transport_type": "ferry",
                            "route_name": "Beşiktaş - Bosphorus Islands Tour",
                            "ferry_type": "bosphorus_islands",
                            "origin_terminal": origin_terminal,
                            "destination_terminal": "Bosphorus Islands (Galatasaray Adası, Suada, Kuruçeşme Adası)",
                            "travel_time_minutes": ferry_data["duration_minutes"],
                            "walking_time_to_terminal": walking_info["to_terminal_minutes"],
                            "walking_time_from_terminal": walking_info["from_terminal_minutes"],
                            "total_duration_minutes": ferry_data["duration_minutes"] + walking_info["to_terminal_minutes"] + walking_info["from_terminal_minutes"],
                            "frequency_minutes": ferry_data["frequency_minutes"],
                            "cost_tl": ferry_data["price_tl"],
                            "real_time_info": real_time_info,
                            "detailed_instructions": self._generate_ferry_instructions(
                                "Beşiktaş - Bosphorus Islands", ferry_data, walking_info, real_time_info
                            ),
                            "scenic_value": "Exclusive Bosphorus island experience with club access",
                            "operating_hours": "Seasonal schedule - typically 10:00-20:00",
                            "special_features": [
                                "Galatasaray Sports Club private island",
                                "Suada floating restaurant and club",
                                "Exclusive events and dining venues",
                                "Premium Bosphorus views",
                                "Private club atmosphere",
                                "Seasonal access only",
                                "Advanced booking often required"
                            ],
                            "seasonal_note": "Private club access may be required for some islands",
                            "next_departures": real_time_info.get("next_departures", [])
                        }
                        routes.append(bosphorus_islands_route)
            
            return routes
            
        except Exception as e:
            logger.error(f"Error finding ferry routes: {e}")
            return []
    
    async def _get_real_time_ferry_info(self, route: str) -> Dict[str, Any]:
        """Get real-time ferry departure information"""
        try:
            current_time = datetime.now()
            
            # Simulate real-time ferry data (in production, would call İDO API)
            base_frequency = 15  # Ferry every 15 minutes during peak hours
            
            # Account for weather and operational factors
            weather_factor = 1.0  # Could integrate weather API
            operational_factor = 1.0
            
            adjusted_frequency = int(base_frequency * weather_factor * operational_factor)
            
            # Calculate next departures
            next_departures = []
            for i in range(3):
                departure_time = current_time + timedelta(minutes=adjusted_frequency * (i + 1))
                next_departures.append(departure_time.strftime("%H:%M"))
            
            return {
                "route": route,
                "status": "on_schedule",
                "next_departures": next_departures,
                "frequency_minutes": adjusted_frequency,
                "weather_conditions": "Good for sailing",
                "last_updated": current_time.isoformat(),
                "pier_info": "Check departure board at ferry terminal"
            }
            
        except Exception as e:
            logger.error(f"Error getting ferry info: {e}")
            return {"error": "Ferry schedule temporarily unavailable"}
    
    def _get_ferry_terminal_walking_directions(self, origin: str, origin_terminal: str, dest_terminal: str, destination: str) -> Dict[str, Any]:
        """Get walking directions to/from ferry terminals"""
        
        # Ferry terminals are typically waterfront locations
        to_terminal_time = self._estimate_ferry_terminal_walking_time(origin, origin_terminal)
        from_terminal_time = self._estimate_ferry_terminal_walking_time(dest_terminal, destination)
        
        to_terminal_directions = self._generate_ferry_terminal_directions(origin, origin_terminal, "to")
        from_terminal_directions = self._generate_ferry_terminal_directions(dest_terminal, destination, "from")
        
        return {
            "to_terminal_minutes": to_terminal_time,
            "from_terminal_minutes": from_terminal_time,
            "total_walking_minutes": to_terminal_time + from_terminal_time,
            "to_terminal_directions": to_terminal_directions,
            "from_terminal_directions": from_terminal_directions,
            "terminal_facilities": [
                "Ferry terminals have ticket offices",
                "Waiting areas with seating available",
                "Food and beverage options at major terminals",
                "Clear departure information displays"
            ]
        }
    
    def _estimate_ferry_terminal_walking_time(self, start: str, terminal: str) -> int:
        """Estimate walking time to ferry terminals"""
        
        # Ferry terminal specific walking times
        terminal_distances = {
            ("sultanahmet", "eminönü"): 10,
            ("taksim", "karaköy"): 15,
            ("beşiktaş", "beşiktaş"): 3,
            ("üsküdar", "üsküdar"): 2,
            ("kadıköy", "kadıköy"): 5
        }
        
        key = (start.lower(), terminal.lower())
        return terminal_distances.get(key, 12)  # 12 minutes default
    
    def _generate_ferry_terminal_directions(self, start: str, terminal: str, direction: str) -> List[str]:
        """Generate directions to ferry terminals"""
        
        terminal_guides = {
            ("sultanahmet", "eminönü"): [
                "Walk north from Sultanahmet towards the Golden Horn",
                "Head down to the waterfront via Eminönü Meydanı",
                "Ferry terminal is next to the Spice Bazaar",
                "Look for İDO ferry signs and ticket booths"
            ],
            ("taksim", "karaköy"): [
                "Take the historic Tünel funicular down from Taksim area",
                "Exit at Karaköy station",
                "Walk towards the Galata Bridge",
                "Ferry terminal is on the Golden Horn waterfront"
            ],
            ("beşiktaş", "beşiktaş"): [
                "Head to the Beşiktaş waterfront",
                "Ferry terminal is next to the bus station",
                "Large terminal building with İDO signage",
                "Multiple piers for different routes"
            ]
        }
        
        key = (start.lower(), terminal.lower())
        if key in terminal_guides:
            return terminal_guides[key]
        
        return [
            f"Head towards the waterfront from {start}",
            f"Look for İDO ferry terminal signs",
            f"Ferry terminals are located on the water",
            f"Follow signs to {terminal} ferry pier"
        ]
    
    def _generate_ferry_instructions(self, route: str, ferry_data: Dict, walking_info: Dict, real_time_info: Dict) -> List[str]:
        """Generate comprehensive ferry journey instructions"""
        
        instructions = []
        
        # Walking to terminal
        route_parts = route.split(" - ")
        origin_terminal = route_parts[0] if len(route_parts) > 0 else "ferry terminal"
        
        instructions.extend([
            f"🚶 WALK TO FERRY TERMINAL ({walking_info['to_terminal_minutes']} min):"
        ] + [f"   • {step}" for step in walking_info['to_terminal_directions']])
        
        # Ferry boarding
        next_departure = real_time_info.get("next_departures", ["Check terminal"])[0]
        instructions.extend([
            f"",
            f"⛴️ FERRY BOARDING at {origin_terminal}:",
            f"   • Route: {route}",
            f"   • Next departure: {next_departure}",
            f"   • Purchase ticket at terminal (₺{ferry_data['price_tl']})",
            f"   • Board from designated pier",
            f"   • Find seating (indoor/outdoor available)"
        ])
        
        # Journey
        instructions.extend([
            f"",
            f"🌊 DURING FERRY CROSSING:",
            f"   • Enjoy Bosphorus views",
            f"   • Journey time: ~{ferry_data['duration_minutes']} minutes",
            f"   • Stay seated during departure/arrival",
            f"   • Take photos of Istanbul skyline"
        ])
        
        # Walking from terminal
        instructions.extend([
            f"",
            f"🚶 WALK FROM FERRY TERMINAL ({walking_info['from_terminal_minutes']} min):"
        ] + [f"   • {step}" for step in walking_info['from_terminal_directions']])
        
        return instructions

# Global instance for use in main.py
enhanced_transport_service = EnhancedTransportationService()
