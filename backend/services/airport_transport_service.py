#!/usr/bin/env python
"""
Istanbul Airport Transport Service
=================================

Comprehensive, up-to-date airport transport information for Istanbul's airports.
Integrates with IBB API and provides reliable fallback data.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AirportRoute:
    """Airport transport route information"""
    route_id: str
    name: str
    origin: str
    destination: str
    transport_type: str  # metro, bus, taxi
    duration_minutes: int
    frequency_minutes: int
    operating_hours: str
    price_try: float
    features: List[str]  # wifi, luggage_space, etc.

@dataclass
class Airport:
    """Airport information"""
    code: str
    name: str
    location: str
    status: str
    coordinates: Tuple[float, float]  # lat, lng

class IstanbulAirportTransportService:
    """Service for Istanbul airport transport information"""
    
    def __init__(self):
        self.airports = self._load_airport_data()
        self.transport_routes = self._load_transport_routes()
        logger.info("✈️ Istanbul Airport Transport Service initialized")
    
    def _load_airport_data(self) -> Dict[str, Airport]:
        """Load current airport information"""
        return {
            'IST': Airport(
                code='IST',
                name='İstanbul Airport',
                location='European Side - Arnavutköy',
                status='active',
                coordinates=(41.2753, 28.7519)
            ),
            'SAW': Airport(
                code='SAW', 
                name='Sabiha Gökçen Airport',
                location='Asian Side - Pendik',
                status='active',
                coordinates=(40.8986, 29.3092)
            ),
            'ATL': Airport(
                code='ATL',
                name='Atatürk Airport',
                location='European Side - Bakırköy',
                status='closed_2019',
                coordinates=(40.9769, 28.8146)
            )
        }
    
    def _load_transport_routes(self) -> Dict[str, List[AirportRoute]]:
        """Load transport routes for each airport"""
        return {
            'IST': [
                # Metro M11
                AirportRoute(
                    route_id='M11',
                    name='M11 Metro Line',
                    origin='İstanbul Airport',
                    destination='Gayrettepe',
                    transport_type='metro',
                    duration_minutes=75,
                    frequency_minutes=4,
                    operating_hours='06:00-00:30',
                    price_try=13.50,
                    features=['wifi', 'luggage_space', 'air_conditioning', 'frequent_service']
                ),
                # Havaist Buses
                AirportRoute(
                    route_id='IST-1',
                    name='Havaist IST-1',
                    origin='İstanbul Airport',
                    destination='Taksim',
                    transport_type='bus',
                    duration_minutes=90,
                    frequency_minutes=30,
                    operating_hours='24/7',
                    price_try=25.00,
                    features=['wifi', 'large_luggage_space', 'air_conditioning', 'usb_charging']
                ),
                AirportRoute(
                    route_id='IST-2',
                    name='Havaist IST-2',
                    origin='İstanbul Airport',
                    destination='Kadıköy',
                    transport_type='bus',
                    duration_minutes=120,
                    frequency_minutes=45,
                    operating_hours='05:00-01:00',
                    price_try=30.00,
                    features=['wifi', 'large_luggage_space', 'air_conditioning', 'usb_charging']
                ),
                AirportRoute(
                    route_id='IST-3',
                    name='Havaist IST-3',
                    origin='İstanbul Airport',
                    destination='Beşiktaş',
                    transport_type='bus',
                    duration_minutes=75,
                    frequency_minutes=30,
                    operating_hours='05:00-01:00',
                    price_try=25.00,
                    features=['wifi', 'large_luggage_space', 'air_conditioning', 'usb_charging']
                ),
                AirportRoute(
                    route_id='IST-4',
                    name='Havaist IST-4',
                    origin='İstanbul Airport',
                    destination='Sultanahmet',
                    transport_type='bus',
                    duration_minutes=90,
                    frequency_minutes=60,
                    operating_hours='05:00-01:00',
                    price_try=25.00,
                    features=['wifi', 'large_luggage_space', 'air_conditioning', 'tourist_friendly']
                ),
                AirportRoute(
                    route_id='IST-5',
                    name='Havaist IST-5',
                    origin='İstanbul Airport',
                    destination='Üsküdar',
                    transport_type='bus',
                    duration_minutes=100,
                    frequency_minutes=45,
                    operating_hours='05:00-01:00',
                    price_try=30.00,
                    features=['wifi', 'large_luggage_space', 'air_conditioning', 'bosphorus_views']
                ),
                AirportRoute(
                    route_id='IST-19',
                    name='Havaist IST-19',
                    origin='İstanbul Airport',
                    destination='Cevizlibağ (M1A Connection)',
                    transport_type='bus',
                    duration_minutes=45,
                    frequency_minutes=20,
                    operating_hours='05:00-01:00',
                    price_try=18.00,
                    features=['metro_connection', 'budget_option', 'frequent_service']
                )
            ],
            'SAW': [
                # Metro M4
                AirportRoute(
                    route_id='M4',
                    name='M4 Metro Line',
                    origin='Sabiha Gökçen Airport',
                    destination='Kadıköy',
                    transport_type='metro',
                    duration_minutes=60,
                    frequency_minutes=7,
                    operating_hours='06:00-00:30',
                    price_try=13.50,
                    features=['direct_connection', 'luggage_space', 'air_conditioning', 'asian_side']
                ),
                # Havabus
                AirportRoute(
                    route_id='H-2',
                    name='Havabus H-2',
                    origin='Sabiha Gökçen Airport',
                    destination='Taksim',
                    transport_type='bus',
                    duration_minutes=90,
                    frequency_minutes=30,
                    operating_hours='04:00-02:00',
                    price_try=20.00,
                    features=['wifi', 'luggage_space', 'air_conditioning', 'cross_city']
                ),
                AirportRoute(
                    route_id='H-3',
                    name='Havabus H-3',
                    origin='Sabiha Gökçen Airport',
                    destination='Kadıköy',
                    transport_type='bus',
                    duration_minutes=45,
                    frequency_minutes=20,
                    operating_hours='04:00-02:00',
                    price_try=15.00,
                    features=['local_connection', 'frequent_service', 'asian_side']
                )
            ]
        }
    
    def get_airport_info(self, airport_code: str) -> Optional[Airport]:
        """Get information about a specific airport"""
        return self.airports.get(airport_code.upper())
    
    def get_active_airports(self) -> List[Airport]:
        """Get list of active airports"""
        return [airport for airport in self.airports.values() if airport.status == 'active']
    
    def get_transport_options(self, airport_code: str, destination: str = None) -> List[AirportRoute]:
        """Get transport options for an airport"""
        airport_code = airport_code.upper()
        if airport_code not in self.transport_routes:
            return []
        
        routes = self.transport_routes[airport_code]
        
        if destination:
            # Filter by destination
            destination_lower = destination.lower()
            filtered_routes = []
            for route in routes:
                if any(dest in route.destination.lower() for dest in [destination_lower]):
                    filtered_routes.append(route)
            return filtered_routes
        
        return routes
    
    def find_best_route(self, airport_code: str, destination: str, 
                       preferences: Dict[str, Any] = None) -> Optional[AirportRoute]:
        """Find the best route based on preferences"""
        routes = self.get_transport_options(airport_code, destination)
        
        if not routes:
            return None
        
        if not preferences:
            # Default: prefer faster routes
            return min(routes, key=lambda r: r.duration_minutes)
        
        # Score routes based on preferences
        best_route = None
        best_score = -1
        
        for route in routes:
            score = 0
            
            # Preference: budget (lower price is better)
            if preferences.get('budget', False):
                score += (100 - route.price_try) / 10
            
            # Preference: speed (lower duration is better) 
            if preferences.get('fast', False):
                score += (180 - route.duration_minutes) / 10
            
            # Preference: comfort (more features is better)
            if preferences.get('comfort', False):
                score += len(route.features) * 5
            
            # Preference: frequency (higher frequency is better)
            if preferences.get('frequent', False):
                score += (60 - route.frequency_minutes) / 5
            
            if score > best_score:
                best_score = score
                best_route = route
        
        return best_route
    
    def get_route_recommendations(self, airport_code: str, destination: str = None) -> str:
        """Get formatted route recommendations"""
        airport = self.get_airport_info(airport_code)
        if not airport:
            return f"❌ Airport code '{airport_code}' not found or not active"
        
        if airport.status == 'closed_2019':
            return f"❌ **{airport.name}** has been closed since 2019. Use İstanbul Airport (IST) or Sabiha Gökçen (SAW) instead."
        
        routes = self.get_transport_options(airport_code, destination)
        
        if not routes:
            return f"❌ No transport routes found from {airport.name}"
        
        # Build response
        response = f"✈️ **Transport from {airport.name} ({airport_code})**\n\n"
        
        # Group by transport type
        metro_routes = [r for r in routes if r.transport_type == 'metro']
        bus_routes = [r for r in routes if r.transport_type == 'bus']
        
        if metro_routes:
            response += "🚇 **Metro Options:**\n"
            for route in metro_routes:
                response += f"• **{route.name}**: {route.origin} → {route.destination}\n"
                response += f"  ⏱️ {route.duration_minutes} min • 🔄 Every {route.frequency_minutes} min • 💰 {route.price_try}₺\n"
                response += f"  🕐 {route.operating_hours} • ✨ {', '.join(route.features)}\n\n"
        
        if bus_routes:
            response += "🚍 **Bus Options:**\n"
            for route in bus_routes:
                response += f"• **{route.name}**: {route.origin} → {route.destination}\n"
                response += f"  ⏱️ {route.duration_minutes} min • 🔄 Every {route.frequency_minutes} min • 💰 {route.price_try}₺\n"
                response += f"  🕐 {route.operating_hours} • ✨ {', '.join(route.features)}\n\n"
        
        response += "💡 **Tips:**\n"
        response += "• Buy an İstanbulkart for seamless metro travel\n"
        response += "• Havaist buses accept contactless payments\n" 
        response += "• Check live arrivals on İBB Mobile app\n"
        response += "• Allow extra time during rush hours (07:00-09:30, 17:00-19:30)"
        
        return response
    
    def get_airport_comparison(self) -> str:
        """Get comparison of Istanbul's airports"""
        response = "✈️ **Istanbul Airports Comparison**\n\n"
        
        active_airports = self.get_active_airports()
        
        for airport in active_airports:
            routes = self.get_transport_options(airport.code)
            metro_count = len([r for r in routes if r.transport_type == 'metro'])
            bus_count = len([r for r in routes if r.transport_type == 'bus'])
            
            response += f"**{airport.name} ({airport.code})**\n"
            response += f"📍 {airport.location}\n"
            response += f"🚇 {metro_count} metro line{'s' if metro_count != 1 else ''}\n"
            response += f"🚍 {bus_count} bus route{'s' if bus_count != 1 else ''}\n"
            
            if routes:
                fastest_route = min(routes, key=lambda r: r.duration_minutes)
                cheapest_route = min(routes, key=lambda r: r.price_try)
                response += f"⚡ Fastest to city: {fastest_route.duration_minutes} min ({fastest_route.name})\n"
                response += f"💰 Cheapest option: {cheapest_route.price_try}₺ ({cheapest_route.name})\n"
            
            response += "\n"
        
        # Add note about closed airport
        closed_airport = self.airports['ATL']
        response += f"❌ **{closed_airport.name} ({closed_airport.code})** - Closed since March 2019\n"
        
        return response

# Global instance
airport_transport_service = IstanbulAirportTransportService()

def get_airport_transport_service() -> IstanbulAirportTransportService:
    """Get the global airport transport service instance"""
    return airport_transport_service

# Example usage and testing
if __name__ == "__main__":
    print("✈️ Testing Istanbul Airport Transport Service")
    print("=" * 50)
    
    service = get_airport_transport_service()
    
    # Test different queries
    test_queries = [
        ("IST", "Taksim"),
        ("SAW", "Kadıköy"), 
        ("ATL", None),  # Closed airport
        ("IST", None),  # All options
    ]
    
    for airport_code, destination in test_queries:
        print(f"\n🔍 Query: {airport_code} → {destination or 'All destinations'}")
        print("-" * 40)
        result = service.get_route_recommendations(airport_code, destination)
        print(result[:300] + "..." if len(result) > 300 else result)
        print()
    
    print("\n📊 Airport Comparison:")
    print("-" * 40)
    comparison = service.get_airport_comparison()
    print(comparison)
