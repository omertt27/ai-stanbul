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
from .walking_directions import WalkingDirectionsGenerator

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
        self.walking_generator = WalkingDirectionsGenerator()
        
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

    def plan_journey_from_gps(self,
                             gps_lat: float,
                             gps_lng: float,
                             destination: str,
                             max_start_walking_m: int = 1000,
                             preferences: Optional[RoutePreferences] = None) -> Optional[Dict]:
        """
        Plan complete journey from GPS location to destination
        Includes walking to nearest stop, transit journey, and walking to final destination
        
        Args:
            gps_lat: User's GPS latitude
            gps_lng: User's GPS longitude
            destination: Destination location (name or coordinates)
            max_start_walking_m: Maximum walking distance to start stop (meters)
            preferences: Route preferences
            
        Returns:
            Complete journey plan with walking and transit segments
        """
        logger.info(f"Planning GPS journey from ({gps_lat}, {gps_lng}) to {destination}")
        
        # Step 1: Find nearest transit stops from GPS location
        nearest_stops = self.location_matcher.find_nearest_stops(
            gps_lat=gps_lat,
            gps_lng=gps_lng,
            max_distance_km=max_start_walking_m / 1000,
            limit=5
        )
        
        if not nearest_stops:
            logger.warning("No transit stops found near GPS location")
            return None
        
        logger.info(f"Found {len(nearest_stops)} nearby stops")
        
        # Step 2: Match destination location
        dest_match = self.location_matcher.match_location(destination)
        if not dest_match:
            logger.warning(f"Could not match destination: {destination}")
            return None
        
        logger.info(f"Matched destination: {dest_match.stop_name}")
        
        # Step 3: Try to find best journey from each nearby stop
        best_journey = None
        best_start_stop = None
        best_total_time = float('inf')
        
        route_prefs = preferences or RoutePreferences()
        
        for start_stop in nearest_stops:
            # Plan journey from this stop to destination
            journey = self.route_finder.find_optimal_route(
                start_stop['stop_id'],
                dest_match.stop_id,
                preferences=route_prefs,
                use_astar=True
            )
            
            if journey:
                # Calculate total time including initial walking
                total_time = start_stop['walking_time_min'] + journey.total_duration_minutes
                
                logger.info(
                    f"Route via {start_stop['stop_name']}: "
                    f"{start_stop['walking_time_min']}min walk + "
                    f"{journey.total_duration_minutes}min transit = {total_time}min total"
                )
                
                if total_time < best_total_time:
                    best_total_time = total_time
                    best_journey = journey
                    best_start_stop = start_stop
        
        if not best_journey or not best_start_stop:
            logger.warning("No viable journey found from GPS location")
            return None
        
        # Step 4: Generate walking directions to start stop
        walking_to_start = self.walking_generator.generate_walking_directions(
            from_lat=gps_lat,
            from_lon=gps_lng,
            to_lat=best_start_stop['coordinates']['lat'],
            to_lon=best_start_stop['coordinates']['lon'],
            to_name=best_start_stop['stop_name'],
            transport_type=best_start_stop['transport_type']
        )
        
        # Step 5: Check if final destination requires walking from last stop
        last_stop = self.network.stops.get(best_journey.segments[-1].to_stop)
        walking_to_destination = None
        
        if last_stop:
            # Calculate distance from last stop to final destination
            final_walk_distance = self.location_matcher.calculate_distance(
                (last_stop.lat, last_stop.lon),
                (dest_match.latitude, dest_match.longitude)
            )
            
            # If more than 100m, generate walking directions
            if final_walk_distance > 0.1:  # 100m = 0.1km
                walking_to_destination = self.walking_generator.generate_walking_directions(
                    from_lat=last_stop.lat,
                    from_lon=last_stop.lon,
                    to_lat=dest_match.latitude,
                    to_lon=dest_match.longitude,
                    to_name=dest_match.stop_name,
                    transport_type=None  # Final destination
                )
        
        # Step 6: Compile complete journey
        complete_journey = {
            'journey_type': 'gps_to_destination',
            'gps_start': {
                'latitude': gps_lat,
                'longitude': gps_lng
            },
            'destination': dest_match.to_dict(),
            
            # Walking segment 1: GPS to transit stop
            'walking_to_transit': {
                'directions': walking_to_start,
                'start_location': {'lat': gps_lat, 'lon': gps_lng},
                'end_stop': {
                    'stop_id': best_start_stop['stop_id'],
                    'stop_name': best_start_stop['stop_name'],
                    'transport_type': best_start_stop['transport_type'],
                    'coordinates': best_start_stop['coordinates']
                }
            },
            
            # Transit segment
            'transit_journey': best_journey.to_dict(),
            
            # Walking segment 2: Last stop to final destination (if needed)
            'walking_to_destination': walking_to_destination,
            
            # Summary
            'summary': {
                'total_duration_min': (
                    walking_to_start['total_duration_min'] +
                    best_journey.total_duration_minutes +
                    (walking_to_destination['total_duration_min'] if walking_to_destination else 0)
                ),
                'walking_duration_min': (
                    walking_to_start['total_duration_min'] +
                    (walking_to_destination['total_duration_min'] if walking_to_destination else 0)
                ),
                'transit_duration_min': best_journey.total_duration_minutes,
                'total_distance_km': round(
                    walking_to_start['total_distance_km'] +
                    best_journey.total_distance_km +
                    (walking_to_destination['total_distance_km'] if walking_to_destination else 0),
                    2
                ),
                'total_transfers': best_journey.total_transfers,
                'transport_types': list(best_journey.transport_types_used),
                'estimated_cost_tl': best_journey.estimated_cost_tl
            }
        }
        
        logger.info(
            f"Complete GPS journey planned: "
            f"{complete_journey['summary']['total_duration_min']}min total, "
            f"{complete_journey['summary']['total_transfers']} transfers"
        )
        
        return complete_journey
    
    def find_best_gps_start_stops(self,
                                  gps_lat: float,
                                  gps_lng: float,
                                  destination_stop_id: str,
                                  max_walking_m: int = 1000,
                                  limit: int = 3) -> List[Dict]:
        """
        Find best transit stops to start journey from GPS location
        Evaluates based on walking distance + transit time
        
        Args:
            gps_lat: User's GPS latitude
            gps_lng: User's GPS longitude
            destination_stop_id: Destination stop ID
            max_walking_m: Maximum walking distance to consider
            limit: Number of options to return
            
        Returns:
            List of start stop options with journey details
        """
        # Find nearby stops
        nearby_stops = self.location_matcher.find_nearest_stops(
            gps_lat=gps_lat,
            gps_lng=gps_lng,
            max_distance_km=max_walking_m / 1000,
            limit=10  # Get more candidates
        )
        
        if not nearby_stops:
            return []
        
        # Evaluate each stop
        evaluated_options = []
        
        for stop in nearby_stops:
            # Try to find route from this stop
            journey = self.route_finder.find_optimal_route(
                stop['stop_id'],
                destination_stop_id,
                use_astar=True
            )
            
            if journey:
                total_time = stop['walking_time_min'] + journey.total_duration_minutes
                
                evaluated_options.append({
                    'start_stop': stop,
                    'transit_journey': journey.to_dict(),
                    'walking_time_min': stop['walking_time_min'],
                    'transit_time_min': journey.total_duration_minutes,
                    'total_time_min': total_time,
                    'transfers': journey.total_transfers,
                    'quality_score': journey.quality_score
                })
        
        # Sort by total time (walking + transit)
        evaluated_options.sort(key=lambda x: x['total_time_min'])
        
        return evaluated_options[:limit]
