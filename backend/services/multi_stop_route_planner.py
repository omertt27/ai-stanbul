"""
Multi-Stop Route Planner for Istanbul
======================================

Plan optimized itineraries visiting multiple locations with intelligent routing.

Features:
- Visit 2-10 locations in a single trip
- Automatic route optimization (minimize travel time/distance)
- Transportation suggestions between stops
- Time estimation for each location
- Accessibility-aware routing
- Integration with existing routing infrastructure

Example queries:
- "Plan a day visiting Hagia Sophia, Blue Mosque, and Grand Bazaar"
- "Best route to visit 3 museums in Sultanahmet"
- "Create an itinerary for Topkapi Palace, Basilica Cistern, Spice Market"

Author: Istanbul AI Team
Date: November 30, 2025
"""

import logging
import math
import itertools
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

# Import existing routing infrastructure
try:
    # Try relative import first (when imported as module)
    from .graph_routing_engine import GraphRoutingEngine
    from .route_optimizer import RouteOptimizer, RoutePreference
    from .transportation_directions_service import TransportationDirectionsService
    ROUTING_AVAILABLE = True
    logger.info("✅ Routing infrastructure available for multi-stop planning")
except ImportError:
    try:
        # Try absolute import (when run directly or from test)
        from graph_routing_engine import GraphRoutingEngine
        from route_optimizer import RouteOptimizer, RoutePreference
        from transportation_directions_service import TransportationDirectionsService
        ROUTING_AVAILABLE = True
        logger.info("✅ Routing infrastructure available for multi-stop planning (absolute imports)")
    except ImportError as e:
        ROUTING_AVAILABLE = False
        logger.warning(f"⚠️ Routing infrastructure not available: {e}")
    
    # Define placeholder for type hints when imports fail
    if not ROUTING_AVAILABLE:
        class RoutePreference:
            """Placeholder for type hints when imports fail"""
            pass


class OptimizationStrategy(Enum):
    """Strategy for optimizing multi-stop routes"""
    SHORTEST_TOTAL_TIME = "shortest_time"  # Minimize total travel time
    SHORTEST_TOTAL_DISTANCE = "shortest_distance"  # Minimize total distance
    NEAREST_NEIGHBOR = "nearest_neighbor"  # Visit nearest unvisited location
    CUSTOM_ORDER = "custom_order"  # User-specified order
    ACCESSIBLE_FIRST = "accessible_first"  # Prioritize accessible routes


@dataclass
class PointOfInterest:
    """A location to visit in the itinerary"""
    name: str
    coordinates: Tuple[float, float]  # (lat, lon)
    category: str = "attraction"  # attraction, restaurant, shopping, etc.
    suggested_duration_minutes: int = 60  # How long to spend here
    opening_hours: Optional[str] = None
    entry_fee: Optional[float] = None
    accessibility_level: str = "unknown"  # fully_accessible, partial, limited, unknown
    priority: int = 1  # 1=must-see, 2=recommended, 3=optional
    
    def __hash__(self):
        return hash((self.name, self.coordinates))
    
    def __eq__(self, other):
        if not isinstance(other, PointOfInterest):
            return False
        return self.name == other.name and self.coordinates == other.coordinates


@dataclass
class RouteSegmentMultiStop:
    """A segment between two stops in the itinerary"""
    from_poi: PointOfInterest
    to_poi: PointOfInterest
    transport_route: Any  # TransportRoute from graph_routing_engine
    distance_km: float
    duration_minutes: int
    cost_tl: float
    modes_used: List[str]
    transfers: int
    accessibility_score: float = 0.0


@dataclass
class MultiStopItinerary:
    """Complete multi-stop itinerary"""
    stops: List[PointOfInterest]
    route_segments: List[RouteSegmentMultiStop]
    total_distance_km: float
    total_travel_time_minutes: int
    total_visit_time_minutes: int
    total_time_minutes: int  # travel + visit
    total_cost_tl: float
    optimization_strategy: OptimizationStrategy
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    highlights: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    accessibility_friendly: bool = True
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """Generate timeline of the itinerary"""
        timeline = []
        current_time = self.start_time or datetime.now()
        
        for i, stop in enumerate(self.stops):
            # Add arrival at location
            timeline.append({
                'time': current_time.strftime('%H:%M'),
                'type': 'arrival',
                'location': stop.name,
                'category': stop.category
            })
            
            # Add visit duration
            visit_end = current_time + timedelta(minutes=stop.suggested_duration_minutes)
            timeline.append({
                'time': f"{current_time.strftime('%H:%M')} - {visit_end.strftime('%H:%M')}",
                'type': 'visit',
                'location': stop.name,
                'duration': stop.suggested_duration_minutes,
                'details': f"Explore {stop.name}"
            })
            
            current_time = visit_end
            
            # Add travel to next location
            if i < len(self.route_segments):
                segment = self.route_segments[i]
                travel_end = current_time + timedelta(minutes=segment.duration_minutes)
                
                modes_str = " → ".join(segment.modes_used[:2])  # Show first 2 modes
                if len(segment.modes_used) > 2:
                    modes_str += "..."
                
                timeline.append({
                    'time': f"{current_time.strftime('%H:%M')} - {travel_end.strftime('%H:%M')}",
                    'type': 'travel',
                    'from': segment.from_poi.name,
                    'to': segment.to_poi.name,
                    'duration': segment.duration_minutes,
                    'modes': segment.modes_used,
                    'details': f"Travel by {modes_str} ({segment.distance_km:.1f}km)"
                })
                
                current_time = travel_end
        
        return timeline


class MultiStopRoutePlanner:
    """Plans optimized routes visiting multiple locations"""
    
    def __init__(self):
        """Initialize multi-stop route planner"""
        if not ROUTING_AVAILABLE:
            raise ImportError("Routing infrastructure not available")
        
        # Use TransportationDirectionsService which has the full routing infrastructure
        self.directions_service = TransportationDirectionsService()
        self.route_optimizer = RouteOptimizer()
        
        # Istanbul POI database (major tourist attractions)
        self.poi_database = self._initialize_poi_database()
        
        logger.info("✅ Multi-stop route planner initialized")
    
    def _initialize_poi_database(self) -> Dict[str, PointOfInterest]:
        """Initialize database of known points of interest"""
        pois = {
            # Sultanahmet area
            'hagia_sophia': PointOfInterest(
                name='Hagia Sophia',
                coordinates=(41.0086, 28.9802),
                category='museum',
                suggested_duration_minutes=90,
                opening_hours='09:00-19:00',
                entry_fee=25.0,
                accessibility_level='partial',
                priority=1
            ),
            'blue_mosque': PointOfInterest(
                name='Blue Mosque',
                coordinates=(41.0054, 28.9768),
                category='mosque',
                suggested_duration_minutes=45,
                opening_hours='Prayer times excluded',
                entry_fee=0.0,
                accessibility_level='partial',
                priority=1
            ),
            'topkapi_palace': PointOfInterest(
                name='Topkapi Palace',
                coordinates=(41.0115, 28.9833),
                category='museum',
                suggested_duration_minutes=120,
                opening_hours='09:00-18:00',
                entry_fee=30.0,
                accessibility_level='limited',
                priority=1
            ),
            'basilica_cistern': PointOfInterest(
                name='Basilica Cistern',
                coordinates=(41.0084, 28.9778),
                category='museum',
                suggested_duration_minutes=30,
                opening_hours='09:00-18:30',
                entry_fee=20.0,
                accessibility_level='limited',
                priority=2
            ),
            'grand_bazaar': PointOfInterest(
                name='Grand Bazaar',
                coordinates=(41.0108, 28.9680),
                category='shopping',
                suggested_duration_minutes=90,
                opening_hours='09:00-19:00',
                entry_fee=0.0,
                accessibility_level='partial',
                priority=1
            ),
            'spice_bazaar': PointOfInterest(
                name='Spice Bazaar',
                coordinates=(41.0166, 28.9700),
                category='shopping',
                suggested_duration_minutes=45,
                opening_hours='08:00-19:00',
                entry_fee=0.0,
                accessibility_level='partial',
                priority=2
            ),
            
            # Beyoğlu area
            'galata_tower': PointOfInterest(
                name='Galata Tower',
                coordinates=(41.0256, 28.9742),
                category='attraction',
                suggested_duration_minutes=60,
                opening_hours='09:00-20:00',
                entry_fee=15.0,
                accessibility_level='limited',
                priority=1
            ),
            'istiklal_street': PointOfInterest(
                name='Istiklal Street',
                coordinates=(41.0332, 28.9775),
                category='shopping',
                suggested_duration_minutes=120,
                opening_hours='Always open',
                entry_fee=0.0,
                accessibility_level='fully_accessible',
                priority=1
            ),
            'taksim_square': PointOfInterest(
                name='Taksim Square',
                coordinates=(41.0370, 28.9850),
                category='attraction',
                suggested_duration_minutes=30,
                opening_hours='Always open',
                entry_fee=0.0,
                accessibility_level='fully_accessible',
                priority=2
            ),
            
            # Bosphorus
            'dolmabahce_palace': PointOfInterest(
                name='Dolmabahçe Palace',
                coordinates=(41.0392, 29.0000),
                category='museum',
                suggested_duration_minutes=90,
                opening_hours='09:00-16:00',
                entry_fee=30.0,
                accessibility_level='partial',
                priority=1
            ),
            'maiden_tower': PointOfInterest(
                name='Maiden Tower',
                coordinates=(41.0210, 29.0044),
                category='attraction',
                suggested_duration_minutes=60,
                opening_hours='09:00-19:00',
                entry_fee=15.0,
                accessibility_level='limited',
                priority=2
            ),
            
            # Asian side
            'kadikoy': PointOfInterest(
                name='Kadıköy',
                coordinates=(40.9900, 29.0250),
                category='district',
                suggested_duration_minutes=120,
                opening_hours='Always open',
                entry_fee=0.0,
                accessibility_level='fully_accessible',
                priority=2
            ),
        }
        
        return pois
    
    def find_poi_by_name(self, name: str) -> Optional[PointOfInterest]:
        """Find POI by name (fuzzy matching)"""
        name_lower = name.lower()
        
        # Exact match
        for key, poi in self.poi_database.items():
            if poi.name.lower() == name_lower:
                return poi
        
        # Partial match
        for key, poi in self.poi_database.items():
            if name_lower in poi.name.lower() or poi.name.lower() in name_lower:
                return poi
        
        return None
    
    def plan_multi_stop_route(
        self,
        locations: List[Tuple[float, float]] = None,
        poi_names: List[str] = None,
        start_location: Optional[Tuple[float, float]] = None,
        strategy: OptimizationStrategy = OptimizationStrategy.SHORTEST_TOTAL_TIME,
        preferences: List[RoutePreference] = None,
        start_time: Optional[datetime] = None
    ) -> MultiStopItinerary:
        """
        Plan optimal multi-stop route
        
        Args:
            locations: List of (lat, lon) coordinates to visit
            poi_names: List of POI names to visit (alternative to locations)
            start_location: Starting point (if None, starts at first location)
            strategy: Optimization strategy
            preferences: Route preferences (accessible, fastest, etc.)
            start_time: When to start the itinerary
            
        Returns:
            Complete multi-stop itinerary
        """
        # Build list of POIs
        pois = []
        
        if poi_names:
            for name in poi_names:
                poi = self.find_poi_by_name(name)
                if poi:
                    pois.append(poi)
                else:
                    # Create custom POI
                    logger.warning(f"POI '{name}' not in database, creating custom POI")
                    pois.append(PointOfInterest(
                        name=name,
                        coordinates=(41.0, 29.0),  # Default Istanbul center
                        category='custom',
                        suggested_duration_minutes=60
                    ))
        
        if locations:
            for i, coords in enumerate(locations):
                if not any(poi.coordinates == coords for poi in pois):
                    pois.append(PointOfInterest(
                        name=f"Location {i+1}",
                        coordinates=coords,
                        category='custom',
                        suggested_duration_minutes=60
                    ))
        
        if len(pois) < 2:
            raise ValueError("Need at least 2 locations for multi-stop routing")
        
        # Optimize visit order
        optimized_order = self._optimize_visit_order(pois, start_location, strategy)
        
        # Calculate routes between consecutive stops
        route_segments = []
        total_distance = 0.0
        total_travel_time = 0
        total_cost = 0.0
        
        for i in range(len(optimized_order) - 1):
            from_poi = optimized_order[i]
            to_poi = optimized_order[i + 1]
            
            # Get route between stops
            try:
                # Use directions service to find route
                start_lat, start_lng = from_poi.coordinates
                end_lat, end_lng = to_poi.coordinates
                
                route = self.directions_service.routing_engine.find_route(
                    start_lat=start_lat,
                    start_lng=start_lng,
                    end_lat=end_lat,
                    end_lng=end_lng
                )
                
                if route:
                    segment = RouteSegmentMultiStop(
                        from_poi=from_poi,
                        to_poi=to_poi,
                        transport_route=route,
                        distance_km=route.total_distance / 1000,
                        duration_minutes=route.total_duration,
                        cost_tl=route.total_cost,
                        modes_used=route.modes_used,
                        transfers=route.transfers,
                        accessibility_score=80.0  # Placeholder
                    )
                    
                    route_segments.append(segment)
                    total_distance += segment.distance_km
                    total_travel_time += segment.duration_minutes
                    total_cost += segment.cost_tl
                else:
                    logger.warning(f"No route found between {from_poi.name} and {to_poi.name}")
            
            except Exception as e:
                logger.error(f"Error finding route: {e}")
        
        # Calculate total visit time
        total_visit_time = sum(poi.suggested_duration_minutes for poi in optimized_order)
        
        # Generate highlights and warnings
        highlights, warnings = self._generate_highlights_and_warnings(
            optimized_order, route_segments, strategy
        )
        
        # Create itinerary
        itinerary = MultiStopItinerary(
            stops=optimized_order,
            route_segments=route_segments,
            total_distance_km=total_distance,
            total_travel_time_minutes=total_travel_time,
            total_visit_time_minutes=total_visit_time,
            total_time_minutes=total_travel_time + total_visit_time,
            total_cost_tl=total_cost,
            optimization_strategy=strategy,
            start_time=start_time or datetime.now(),
            highlights=highlights,
            warnings=warnings
        )
        
        # Calculate end time
        if itinerary.start_time:
            itinerary.end_time = itinerary.start_time + timedelta(minutes=itinerary.total_time_minutes)
        
        logger.info(f"✅ Created multi-stop itinerary with {len(optimized_order)} stops")
        
        return itinerary
    
    def _optimize_visit_order(
        self,
        pois: List[PointOfInterest],
        start_location: Optional[Tuple[float, float]],
        strategy: OptimizationStrategy
    ) -> List[PointOfInterest]:
        """Optimize the order of visiting locations"""
        
        if strategy == OptimizationStrategy.CUSTOM_ORDER:
            # Keep user-specified order
            return pois
        
        elif strategy == OptimizationStrategy.NEAREST_NEIGHBOR:
            # Greedy nearest neighbor algorithm
            return self._nearest_neighbor_order(pois, start_location)
        
        elif strategy == OptimizationStrategy.SHORTEST_TOTAL_TIME:
            # Try to minimize total travel time (TSP-like)
            return self._optimize_by_travel_time(pois, start_location)
        
        elif strategy == OptimizationStrategy.ACCESSIBLE_FIRST:
            # Visit most accessible locations first
            sorted_pois = sorted(pois, key=lambda p: (
                0 if p.accessibility_level == 'fully_accessible' else
                1 if p.accessibility_level == 'partial' else
                2 if p.accessibility_level == 'limited' else 3
            ))
            return sorted_pois
        
        else:
            # Default: nearest neighbor
            return self._nearest_neighbor_order(pois, start_location)
    
    def _nearest_neighbor_order(
        self,
        pois: List[PointOfInterest],
        start_location: Optional[Tuple[float, float]]
    ) -> List[PointOfInterest]:
        """Use nearest neighbor heuristic to order visits"""
        if not pois:
            return []
        
        # Find starting point
        if start_location:
            current_pos = start_location
        else:
            current_pos = pois[0].coordinates
        
        unvisited = set(pois)
        route = []
        
        while unvisited:
            # Find nearest unvisited POI
            nearest = min(unvisited, key=lambda p: self._distance(current_pos, p.coordinates))
            route.append(nearest)
            unvisited.remove(nearest)
            current_pos = nearest.coordinates
        
        return route
    
    def _optimize_by_travel_time(
        self,
        pois: List[PointOfInterest],
        start_location: Optional[Tuple[float, float]]
    ) -> List[PointOfInterest]:
        """Optimize order to minimize total travel time (simplified TSP)"""
        # For small numbers of POIs, we can try multiple permutations
        # For larger numbers, fall back to nearest neighbor
        
        if len(pois) <= 6:
            # Try all permutations (feasible for up to 6-7 locations)
            best_order = None
            best_time = float('inf')
            
            for perm in itertools.permutations(pois):
                total_time = 0
                current_pos = start_location or perm[0].coordinates
                
                for poi in perm:
                    # Estimate travel time based on distance
                    dist = self._distance(current_pos, poi.coordinates)
                    # Rough estimate: 3 minutes per km in Istanbul
                    travel_time = dist * 3
                    total_time += travel_time
                    current_pos = poi.coordinates
                
                if total_time < best_time:
                    best_time = total_time
                    best_order = list(perm)
            
            return best_order
        else:
            # Fall back to nearest neighbor for large sets
            return self._nearest_neighbor_order(pois, start_location)
    
    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two coordinates (km)"""
        lat1, lon1 = pos1
        lat2, lon2 = pos2
        
        R = 6371  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon/2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _generate_highlights_and_warnings(
        self,
        pois: List[PointOfInterest],
        segments: List[RouteSegmentMultiStop],
        strategy: OptimizationStrategy
    ) -> Tuple[List[str], List[str]]:
        """Generate highlights and warnings for the itinerary"""
        highlights = []
        warnings = []
        
        # Highlights
        highlights.append(f"🗺️ Visiting {len(pois)} locations")
        
        total_distance = sum(s.distance_km for s in segments)
        highlights.append(f"📏 Total distance: {total_distance:.1f} km")
        
        total_time = sum(s.duration_minutes for s in segments) + sum(p.suggested_duration_minutes for p in pois)
        hours = total_time // 60
        mins = total_time % 60
        highlights.append(f"⏱️ Total time: {hours}h {mins}m")
        
        # Count free attractions
        free_count = sum(1 for p in pois if p.entry_fee == 0.0)
        if free_count > 0:
            highlights.append(f"💰 {free_count} free attractions")
        
        # Check accessibility
        accessible_count = sum(1 for p in pois if p.accessibility_level in ['fully_accessible', 'partial'])
        if accessible_count == len(pois):
            highlights.append("♿ All locations wheelchair accessible")
        elif accessible_count >= len(pois) * 0.7:
            highlights.append(f"♿ Most locations ({accessible_count}/{len(pois)}) wheelchair accessible")
        
        # Strategy highlight
        if strategy == OptimizationStrategy.SHORTEST_TOTAL_TIME:
            highlights.append("⚡ Optimized for shortest travel time")
        elif strategy == OptimizationStrategy.NEAREST_NEIGHBOR:
            highlights.append("🎯 Optimized route order")
        
        # Warnings
        if total_time > 480:  # More than 8 hours
            warnings.append("⚠️ Long itinerary (8+ hours) - consider splitting into 2 days")
        
        if total_distance > 20:
            warnings.append("⚠️ Long total distance - plan for rest breaks")
        
        max_transfers = max((s.transfers for s in segments), default=0)
        if max_transfers > 2:
            warnings.append(f"⚠️ Some routes require multiple transfers ({max_transfers} max)")
        
        # Check for inaccessible locations
        inaccessible = [p.name for p in pois if p.accessibility_level == 'limited']
        if inaccessible:
            warnings.append(f"♿ Limited accessibility at: {', '.join(inaccessible)}")
        
        return highlights, warnings
    
    def format_itinerary_text(self, itinerary: MultiStopItinerary, language: str = 'en') -> str:
        """Format itinerary as readable text"""
        lines = []
        
        lines.append("🗺️ **YOUR ISTANBUL ITINERARY**\n")
        
        # Summary
        lines.append("📊 **SUMMARY**")
        lines.append(f"• Stops: {len(itinerary.stops)}")
        lines.append(f"• Total Distance: {itinerary.total_distance_km:.1f} km")
        
        travel_hours = itinerary.total_travel_time_minutes // 60
        travel_mins = itinerary.total_travel_time_minutes % 60
        lines.append(f"• Travel Time: {travel_hours}h {travel_mins}m")
        
        visit_hours = itinerary.total_visit_time_minutes // 60
        visit_mins = itinerary.total_visit_time_minutes % 60
        lines.append(f"• Visit Time: {visit_hours}h {visit_mins}m")
        
        total_hours = itinerary.total_time_minutes // 60
        total_mins = itinerary.total_time_minutes % 60
        lines.append(f"• Total Time: {total_hours}h {total_mins}m")
        lines.append(f"• Total Cost: ₺{itinerary.total_cost_tl:.2f}\n")
        
        # Highlights
        if itinerary.highlights:
            lines.append("✨ **HIGHLIGHTS**")
            for highlight in itinerary.highlights:
                lines.append(f"  {highlight}")
            lines.append("")
        
        # Timeline
        lines.append("📍 **ITINERARY**\n")
        timeline = itinerary.get_timeline()
        
        for item in timeline:
            if item['type'] == 'arrival':
                lines.append(f"**{item['time']}** - Arrive at **{item['location']}**")
            elif item['type'] == 'visit':
                lines.append(f"  🎯 Visit for {item['duration']} minutes")
            elif item['type'] == 'travel':
                modes_emoji = {
                    'metro': '🚇',
                    'tram': '🚋',
                    'ferry': '⛴️',
                    'walking': '🚶',
                    'funicular': '🚡',
                    'marmaray': '🚆'
                }
                mode_icons = ' '.join(modes_emoji.get(m, '🚌') for m in item['modes'][:3])
                lines.append(f"  {mode_icons} {item['details']}")
                lines.append("")
        
        # Warnings
        if itinerary.warnings:
            lines.append("\n⚠️ **IMPORTANT NOTES**")
            for warning in itinerary.warnings:
                lines.append(f"  {warning}")
        
        return "\n".join(lines)


# Global instance
_multi_stop_planner = None


def get_multi_stop_planner() -> MultiStopRoutePlanner:
    """Get or create multi-stop route planner instance"""
    global _multi_stop_planner
    if _multi_stop_planner is None:
        _multi_stop_planner = MultiStopRoutePlanner()
    return _multi_stop_planner


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("🗺️ Testing Multi-Stop Route Planner")
    print("="*80 + "\n")
    
    if not ROUTING_AVAILABLE:
        print("❌ Routing infrastructure not available")
        exit(1)
    
    planner = get_multi_stop_planner()
    
    # Test 1: Classic Sultanahmet tour
    print("TEST 1: Sultanahmet Tour (3 stops)")
    print("-" * 80)
    
    try:
        itinerary = planner.plan_multi_stop_route(
            poi_names=['Hagia Sophia', 'Blue Mosque', 'Grand Bazaar'],
            strategy=OptimizationStrategy.SHORTEST_TOTAL_TIME,
            start_time=datetime(2025, 11, 30, 9, 0)
        )
        
        print(planner.format_itinerary_text(itinerary))
        print("\n✅ Test 1 passed!\n")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}\n")
    
    # Test 2: Full day tour (5 stops)
    print("\n" + "="*80)
    print("TEST 2: Full Day Tour (5 stops)")
    print("-" * 80)
    
    try:
        itinerary = planner.plan_multi_stop_route(
            poi_names=['Topkapi Palace', 'Hagia Sophia', 'Blue Mosque', 'Grand Bazaar', 'Spice Bazaar'],
            strategy=OptimizationStrategy.NEAREST_NEIGHBOR,
            start_time=datetime(2025, 11, 30, 9, 0)
        )
        
        print(planner.format_itinerary_text(itinerary))
        print("\n✅ Test 2 passed!\n")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}\n")
    
    # Test 3: Accessible tour
    print("\n" + "="*80)
    print("TEST 3: Accessible Tour")
    print("-" * 80)
    
    try:
        from .route_optimizer import RoutePreference
        
        itinerary = planner.plan_multi_stop_route(
            poi_names=['Taksim Square', 'Istiklal Street', 'Galata Tower'],
            strategy=OptimizationStrategy.ACCESSIBLE_FIRST,
            preferences=[RoutePreference.ACCESSIBLE],
            start_time=datetime(2025, 11, 30, 14, 0)
        )
        
        print(planner.format_itinerary_text(itinerary))
        print("\n✅ Test 3 passed!\n")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}\n")
    
    print("="*80)
    print("✅ Multi-Stop Route Planner tests complete!")
    print("="*80)
