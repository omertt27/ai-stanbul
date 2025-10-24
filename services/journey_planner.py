"""
Journey Planner - High-Level Multi-Modal Journey Planning
Integrates route finding, location matching, and user preferences
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

from .intelligent_route_finder import (
    IntelligentRouteFinder, Journey, RoutePreferences
)
from .location_matcher import LocationMatcher, LocationMatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JourneyRequest:
    """User journey request"""
    origin: str  # Location name or coordinates
    destination: str  # Location name or coordinates
    departure_time: Optional[datetime] = None
    preferences: Optional[RoutePreferences] = None
    include_alternatives: bool = True
    max_alternatives: int = 3

@dataclass
class JourneyPlan:
    """Complete journey plan with multiple route options"""
    origin_location: LocationMatch
    destination_location: LocationMatch
    primary_journey: Journey
    alternative_journeys: List[Journey] = field(default_factory=list)
    request_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'request_time': self.request_time.isoformat(),
            'origin': self.origin_location.to_dict(),
            'destination': self.destination_location.to_dict(),
            'primary_route': self.primary_journey.to_dict(),
            'alternative_routes': [j.to_dict() for j in self.alternative_journeys],
            'total_options': 1 + len(self.alternative_journeys)
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary_lines = [
            f"Journey from {self.origin_location.stop_name} to {self.destination_location.stop_name}",
            f"\nPrimary Route:",
            f"  Duration: {self.primary_journey.total_duration_minutes} minutes",
            f"  Transfers: {self.primary_journey.total_transfers}",
            f"  Transport: {', '.join(self.primary_journey.transport_types_used)}",
            f"  Quality: {self.primary_journey.quality_score:.1%}",
            f"  Cost: ₺{self.primary_journey.estimated_cost_tl:.2f}"
        ]
        
        if self.alternative_journeys:
            summary_lines.append(f"\n{len(self.alternative_journeys)} Alternative Routes Available")
            for i, alt in enumerate(self.alternative_journeys, 1):
                summary_lines.append(
                    f"  Option {i}: {alt.total_duration_minutes}min, "
                    f"{alt.total_transfers} transfers, "
                    f"{', '.join(alt.transport_types_used)}"
                )
        
        return '\n'.join(summary_lines)

class JourneyPlanner:
    """
    Industry-level journey planner
    Orchestrates location matching, route finding, and optimization
    """
    
    def __init__(self, network_graph):
        """
        Initialize journey planner
        
        Args:
            network_graph: TransportationNetwork from route_network_builder
        """
        self.network = network_graph
        self.route_finder = IntelligentRouteFinder(network_graph)
        self.location_matcher = LocationMatcher(network_graph)
        
        logger.info("Journey planner initialized with network graph")
    
    def plan_journey(self, request: JourneyRequest) -> Optional[JourneyPlan]:
        """
        Plan a complete journey with multiple route options
        
        Args:
            request: JourneyRequest object
            
        Returns:
            JourneyPlan with primary and alternative routes, or None if no route found
        """
        logger.info(f"Planning journey: {request.origin} -> {request.destination}")
        
        # Step 1: Match origin location
        origin_match = self.location_matcher.match_location(request.origin)
        if not origin_match:
            logger.warning(f"Could not match origin: {request.origin}")
            return None
        
        logger.info(f"Matched origin: {origin_match.stop_name} (confidence: {origin_match.confidence:.2f})")
        
        # Step 2: Match destination location
        dest_match = self.location_matcher.match_location(request.destination)
        if not dest_match:
            logger.warning(f"Could not match destination: {request.destination}")
            return None
        
        logger.info(f"Matched destination: {dest_match.stop_name} (confidence: {dest_match.confidence:.2f})")
        
        # Check if origin and destination are the same
        if origin_match.stop_id == dest_match.stop_id:
            logger.warning("Origin and destination are the same")
            return None
        
        # Step 3: Find primary route
        preferences = request.preferences or RoutePreferences()
        primary_journey = self.route_finder.find_optimal_route(
            origin_match.stop_id,
            dest_match.stop_id,
            preferences=preferences,
            use_astar=True
        )
        
        if not primary_journey:
            logger.warning("No route found")
            return None
        
        logger.info(
            f"Primary route found: {primary_journey.total_duration_minutes}min, "
            f"{primary_journey.total_transfers} transfers"
        )
        
        # Step 4: Find alternative routes if requested
        alternative_journeys = []
        if request.include_alternatives:
            alternative_journeys = self.route_finder.find_alternative_routes(
                origin_match.stop_id,
                dest_match.stop_id,
                primary_journey=primary_journey,
                preferences=preferences,
                max_alternatives=request.max_alternatives
            )
            logger.info(f"Found {len(alternative_journeys)} alternative routes")
        
        # Step 5: Create journey plan
        plan = JourneyPlan(
            origin_location=origin_match,
            destination_location=dest_match,
            primary_journey=primary_journey,
            alternative_journeys=alternative_journeys
        )
        
        return plan
    
    def plan_journey_simple(self, origin: str, destination: str,
                           minimize_transfers: bool = True) -> Optional[JourneyPlan]:
        """
        Simplified journey planning interface
        
        Args:
            origin: Origin location (name or coordinates)
            destination: Destination location (name or coordinates)
            minimize_transfers: Minimize number of transfers
            
        Returns:
            JourneyPlan or None
        """
        preferences = RoutePreferences(
            minimize_transfers=minimize_transfers,
            minimize_time=not minimize_transfers
        )
        
        request = JourneyRequest(
            origin=origin,
            destination=destination,
            preferences=preferences
        )
        
        return self.plan_journey(request)
    
    def get_journey_with_realtime(self, journey: Journey,
                                  departure_time: datetime = None) -> Dict:
        """
        Enhance journey with real-time information
        (Placeholder for real-time integration)
        
        Args:
            journey: Base Journey object
            departure_time: Planned departure time
            
        Returns:
            Enhanced journey data with real-time updates
        """
        if departure_time is None:
            departure_time = datetime.now()
        
        enhanced_data = journey.to_dict()
        enhanced_data['departure_time'] = departure_time.isoformat()
        
        # Calculate estimated arrival time
        arrival_time = departure_time + timedelta(minutes=journey.total_duration_minutes)
        enhanced_data['estimated_arrival'] = arrival_time.isoformat()
        
        # Placeholder for real-time vehicle positions
        enhanced_data['realtime_available'] = False
        enhanced_data['service_alerts'] = []
        
        # TODO: Integrate with real-time İBB data when available
        # - Vehicle positions on each line
        # - Service delays/disruptions
        # - Updated ETAs
        
        return enhanced_data
    
    def compare_routes(self, journeys: List[Journey]) -> Dict:
        """
        Compare multiple journey options
        
        Args:
            journeys: List of Journey objects to compare
            
        Returns:
            Comparison data
        """
        if not journeys:
            return {}
        
        comparison = {
            'total_routes': len(journeys),
            'fastest': None,
            'least_transfers': None,
            'shortest_distance': None,
            'cheapest': None,
            'highest_quality': None,
            'comparison_table': []
        }
        
        # Find best in each category
        fastest = min(journeys, key=lambda j: j.total_duration_minutes)
        least_transfers = min(journeys, key=lambda j: j.total_transfers)
        shortest = min(journeys, key=lambda j: j.total_distance_km)
        cheapest = min(journeys, key=lambda j: j.estimated_cost_tl)
        highest_quality = max(journeys, key=lambda j: j.quality_score)
        
        comparison['fastest'] = {
            'duration': fastest.total_duration_minutes,
            'route_index': journeys.index(fastest)
        }
        comparison['least_transfers'] = {
            'transfers': least_transfers.total_transfers,
            'route_index': journeys.index(least_transfers)
        }
        comparison['shortest_distance'] = {
            'distance_km': shortest.total_distance_km,
            'route_index': journeys.index(shortest)
        }
        comparison['cheapest'] = {
            'cost_tl': cheapest.estimated_cost_tl,
            'route_index': journeys.index(cheapest)
        }
        comparison['highest_quality'] = {
            'quality_score': highest_quality.quality_score,
            'route_index': journeys.index(highest_quality)
        }
        
        # Build comparison table
        for i, journey in enumerate(journeys):
            comparison['comparison_table'].append({
                'route_index': i,
                'duration_min': journey.total_duration_minutes,
                'transfers': journey.total_transfers,
                'distance_km': round(journey.total_distance_km, 2),
                'cost_tl': journey.estimated_cost_tl,
                'quality_score': round(journey.quality_score, 2),
                'transport_types': list(journey.transport_types_used)
            })
        
        return comparison
    
    def get_accessible_route(self, origin: str, destination: str) -> Optional[JourneyPlan]:
        """
        Find wheelchair-accessible route
        
        Args:
            origin: Origin location
            destination: Destination location
            
        Returns:
            Accessible JourneyPlan or None
        """
        preferences = RoutePreferences(
            wheelchair_accessible=True,
            minimize_transfers=True,
            max_walking_meters=500  # Limit walking for accessibility
        )
        
        request = JourneyRequest(
            origin=origin,
            destination=destination,
            preferences=preferences,
            include_alternatives=False  # Focus on primary accessible route
        )
        
        return self.plan_journey(request)
    
    def explore_area(self, location: str, max_distance_km: float = 1.0) -> Dict:
        """
        Explore transport options around a location
        
        Args:
            location: Location name or coordinates
            max_distance_km: Search radius
            
        Returns:
            Dictionary of available transport options
        """
        # Match location
        location_match = self.location_matcher.match_location(location)
        if not location_match:
            return {'error': 'Location not found'}
        
        # Get nearby transport options
        nearby = self.location_matcher.get_nearby_transport_options(
            location_match.latitude,
            location_match.longitude,
            max_distance_km
        )
        
        exploration_data = {
            'center_location': location_match.to_dict(),
            'search_radius_km': max_distance_km,
            'transport_options': {}
        }
        
        for transport_type, stops in nearby.items():
            exploration_data['transport_options'][transport_type] = [
                {
                    'name': stop.stop_name,
                    'distance_km': round(stop.distance_km, 2),
                    'lines': stop.lines
                }
                for stop in stops[:5]  # Top 5 per type
            ]
        
        return exploration_data
    
    def get_multi_destination_plan(self, origin: str, 
                                   destinations: List[str]) -> Dict:
        """
        Plan journeys to multiple destinations from single origin
        
        Args:
            origin: Starting location
            destinations: List of destination locations
            
        Returns:
            Multi-destination journey plans
        """
        plans = {}
        
        for dest in destinations:
            plan = self.plan_journey_simple(origin, dest)
            if plan:
                plans[dest] = {
                    'duration_min': plan.primary_journey.total_duration_minutes,
                    'transfers': plan.primary_journey.total_transfers,
                    'transport_types': list(plan.primary_journey.transport_types_used),
                    'full_plan': plan.to_dict()
                }
        
        # Sort by duration
        sorted_plans = sorted(
            plans.items(),
            key=lambda x: x[1]['duration_min']
        )
        
        return {
            'origin': origin,
            'destinations_count': len(destinations),
            'successful_routes': len(plans),
            'destinations': dict(sorted_plans)
        }
