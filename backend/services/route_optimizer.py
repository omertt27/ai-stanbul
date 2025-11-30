"""
Route Optimizer and Alternatives Generator
===========================================

Provides multiple route options with different optimization criteria:
- Fastest route (minimize duration)
- Cheapest route (minimize cost/transfers)
- Least transfers (minimize complexity)
- Scenic route (prefer ferry/tram)

Uses k-shortest paths algorithm to find alternative routes.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RoutePreference(Enum):
    """Route optimization preferences"""
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    LEAST_TRANSFERS = "least_transfers"
    SCENIC = "scenic"
    ACCESSIBLE = "accessible"


@dataclass
class RouteOption:
    """A single route option with scoring"""
    route: any  # TransportRoute object
    preference: RoutePreference
    score: float
    highlights: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """For sorting by score"""
        return self.score < other.score


class RouteOptimizer:
    """Optimizes and scores routes based on different criteria"""
    
    def __init__(self):
        """Initialize route optimizer"""
        self.weights = {
            RoutePreference.FASTEST: {
                'duration': 1.0,
                'transfers': 0.2,
                'cost': 0.1,
                'walking': 0.3
            },
            RoutePreference.CHEAPEST: {
                'duration': 0.3,
                'transfers': 0.8,  # More transfers = more cost
                'cost': 1.0,
                'walking': 0.1
            },
            RoutePreference.LEAST_TRANSFERS: {
                'duration': 0.4,
                'transfers': 1.0,
                'cost': 0.2,
                'walking': 0.3
            },
            RoutePreference.SCENIC: {
                'duration': 0.2,
                'transfers': 0.3,
                'cost': 0.1,
                'ferry_bonus': 1.0,  # Prefer ferry routes
                'tram_bonus': 0.5    # Prefer trams
            }
        }
    
    def calculate_route_score(self, route, preference: RoutePreference) -> float:
        """
        Calculate route score based on preference
        
        Lower score = better route for that preference
        """
        weights = self.weights.get(preference, self.weights[RoutePreference.FASTEST])
        
        score = 0.0
        
        # Duration component (normalized to minutes)
        if 'duration' in weights:
            score += weights['duration'] * route.total_duration
        
        # Transfer component
        if 'transfers' in weights:
            transfers = len([s for s in route.steps if s.mode != 'walk']) - 1
            score += weights['transfers'] * transfers * 10  # Each transfer = 10 points
        
        # Cost component
        if 'cost' in weights:
            score += weights['cost'] * route.estimated_cost
        
        # Walking distance component (meters to minutes conversion)
        if 'walking' in weights:
            walking_meters = sum(s.distance for s in route.steps if s.mode == 'walk')
            walking_minutes = walking_meters / 1000 * 12  # ~5 km/h walking
            score += weights['walking'] * walking_minutes
        
        # Scenic bonuses (negative score = better)
        if preference == RoutePreference.SCENIC:
            if 'ferry' in route.modes_used:
                score -= weights.get('ferry_bonus', 0) * 20  # Big bonus for ferry
            if 'tram' in route.modes_used:
                score -= weights.get('tram_bonus', 0) * 10  # Small bonus for tram
        
        return max(0, score)  # Score can't be negative
    
    def generate_highlights(self, route, preference: RoutePreference) -> List[str]:
        """Generate highlight text for a route based on preference"""
        highlights = []
        
        transfers = len([s for s in route.steps if s.mode != 'walk']) - 1
        
        if preference == RoutePreference.FASTEST:
            highlights.append(f"⚡ Fastest option ({route.total_duration} min)")
            if transfers == 0:
                highlights.append("🎯 Direct route - no transfers!")
        
        elif preference == RoutePreference.CHEAPEST:
            highlights.append(f"💰 Most economical (₺{route.estimated_cost:.2f})")
            if route.estimated_cost == 0:
                highlights.append("🆓 Completely free!")
            elif transfers <= 1:
                highlights.append(f"💵 Low cost - {transfers} transfer" + ("s" if transfers > 1 else ""))
        
        elif preference == RoutePreference.LEAST_TRANSFERS:
            highlights.append(f"🎯 Simplest route ({transfers} transfer" + ("s" if transfers != 1 else "") + ")")
            if transfers == 0:
                highlights.append("✨ Direct connection!")
        
        elif preference == RoutePreference.SCENIC:
            if 'ferry' in route.modes_used:
                highlights.append("⛴️ Scenic ferry ride included")
            if 'tram' in route.modes_used:
                highlights.append("🚊 Nostalgic tram experience")
            if len(route.modes_used) >= 3:
                highlights.append("🎨 Multi-modal adventure")
        
        return highlights
    
    def optimize_routes(self, routes: List, preferences: List[RoutePreference] = None) -> List[RouteOption]:
        """
        Score and optimize a list of routes for different preferences
        
        Args:
            routes: List of TransportRoute objects
            preferences: List of preferences to optimize for (default: all)
            
        Returns:
            List of RouteOption objects, sorted by relevance
        """
        if not routes:
            return []
        
        if preferences is None:
            preferences = [
                RoutePreference.FASTEST,
                RoutePreference.CHEAPEST,
                RoutePreference.LEAST_TRANSFERS
            ]
        
        route_options = []
        
        for preference in preferences:
            # Find best route for this preference
            best_route = None
            best_score = float('inf')
            
            for route in routes:
                score = self.calculate_route_score(route, preference)
                if score < best_score:
                    best_score = score
                    best_route = route
            
            if best_route:
                highlights = self.generate_highlights(best_route, preference)
                route_options.append(RouteOption(
                    route=best_route,
                    preference=preference,
                    score=best_score,
                    highlights=highlights
                ))
        
        # Remove duplicates (same route for different preferences)
        unique_options = []
        seen_routes = set()
        
        for option in sorted(route_options, key=lambda x: x.score):
            route_id = self._get_route_id(option.route)
            if route_id not in seen_routes:
                seen_routes.add(route_id)
                unique_options.append(option)
        
        return unique_options[:3]  # Return top 3 unique routes
    
    def _get_route_id(self, route) -> str:
        """Generate unique ID for a route based on its path characteristics"""
        # Use the actual stations/stops path to identify uniqueness
        # This prevents identical routes from appearing with different scores
        try:
            # Get the sequence of transit steps (excluding walks)
            transit_steps = []
            for step in route.steps:
                if step.mode != 'walk':
                    # Include mode and key info
                    transit_steps.append(f"{step.mode}:{step.line}")
            
            # Create signature from transit sequence
            if transit_steps:
                return "_".join(transit_steps)
            else:
                # Fallback to duration + cost
                return f"duration_{route.total_duration}_cost_{route.estimated_cost:.2f}"
        except:
            # Ultimate fallback
            return f"{route.summary}_{route.total_duration}"


class RouteAlternativesGenerator:
    """Generate alternative routes using different strategies"""
    
    def __init__(self, routing_service):
        """
        Initialize alternatives generator
        
        Args:
            routing_service: TransportationDirectionsService instance
        """
        self.routing_service = routing_service
        self.optimizer = RouteOptimizer()
    
    def generate_alternatives(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str = "Start",
        end_name: str = "Destination",
        count: int = 3
    ) -> List[RouteOption]:
        """
        Generate alternative routes between two points
        
        Args:
            start: Start coordinates (lat, lng)
            end: End coordinates (lat, lng)
            start_name: Name of start location
            end_name: Name of end location
            count: Number of alternatives to generate (default: 3)
            
        Returns:
            List of RouteOption objects with different optimizations
        """
        logger.info(f"Generating {count} route alternatives: {start_name} → {end_name}")
        
        # Get primary route
        primary_route = self.routing_service.get_directions(
            start=start,
            end=end,
            start_name=start_name,
            end_name=end_name
        )
        
        if not primary_route:
            logger.warning("No primary route found")
            return []
        
        routes = [primary_route]
        
        # Try to generate alternative routes
        # Strategy 1: If graph routing available, get alternatives from there
        if self.routing_service.routing_engine:
            try:
                alternative_paths = self._get_graph_alternatives(start, end, count + 2)
                logger.info(f"Received {len(alternative_paths)} paths from graph engine")
                
                for i, path in enumerate(alternative_paths):
                    # Convert all paths (including first for comparison)
                    alt_route = self.routing_service._convert_graph_route_to_transport_route(
                        path, start, end, start_name, end_name
                    )
                    if alt_route and alt_route.total_duration > 0:
                        # Log path characteristics
                        logger.debug(f"Path {i+1}: {alt_route.total_duration}min, "
                                   f"{len([s for s in alt_route.steps if s.mode != 'walk'])} transit steps, "
                                   f"₺{alt_route.estimated_cost:.2f}")
                        routes.append(alt_route)
                        
                logger.info(f"Converted {len(routes)} total routes (including primary)")
            except Exception as e:
                logger.warning(f"Could not generate graph alternatives: {e}", exc_info=True)
        
        # Strategy 2: Generate preference-based variations
        # (For now, we'll optimize the routes we have)
        
        # Optimize and score routes
        route_options = self.optimizer.optimize_routes(routes)
        
        logger.info(f"Generated {len(route_options)} unique route alternatives")
        
        return route_options[:count]
    
    def _get_graph_alternatives(self, start: Tuple[float, float], end: Tuple[float, float], k: int = 3):
        """
        Get k alternative paths from graph routing engine using Yen's algorithm
        
        Uses the k-shortest paths implementation in the graph routing engine
        to find diverse alternative routes.
        
        Args:
            start: Start coordinates (lat, lng)
            end: End coordinates (lat, lng)
            k: Number of alternatives to find
            
        Returns:
            List of RoutePath objects
        """
        try:
            # Use the graph routing engine's find_alternative_routes method
            if hasattr(self.routing_service.routing_engine, 'find_alternative_routes'):
                alternatives = self.routing_service.routing_engine.find_alternative_routes(
                    start[0], start[1], end[0], end[1], num_alternatives=k
                )
                logger.info(f"Found {len(alternatives)} alternative paths from graph engine")
                return alternatives
            else:
                logger.debug("Graph routing engine does not support alternative routes yet")
                return []
        except Exception as e:
            logger.warning(f"Error getting graph alternatives: {e}")
            return []
    
    def get_fastest_route(self, start, end, start_name="Start", end_name="Destination"):
        """Get the fastest possible route"""
        alternatives = self.generate_alternatives(start, end, start_name, end_name, count=1)
        if alternatives:
            return alternatives[0].route
        return None
    
    def get_cheapest_route(self, start, end, start_name="Start", end_name="Destination"):
        """Get the cheapest possible route"""
        alternatives = self.generate_alternatives(start, end, start_name, end_name, count=3)
        # Find the one with lowest cost
        cheapest = min(alternatives, key=lambda x: x.route.estimated_cost)
        return cheapest.route if alternatives else None


# Example usage
if __name__ == "__main__":
    from backend.services.transportation_directions_service import TransportationDirectionsService
    
    # Initialize services
    directions_service = TransportationDirectionsService()
    alt_generator = RouteAlternativesGenerator(directions_service)
    
    # Generate alternatives
    alternatives = alt_generator.generate_alternatives(
        start=(41.0370, 28.9850),  # Taksim
        end=(40.9900, 29.0250),     # Kadıköy
        start_name="Taksim",
        end_name="Kadıköy",
        count=3
    )
    
    print(f"\nFound {len(alternatives)} route alternatives:\n")
    
    for i, option in enumerate(alternatives, 1):
        print(f"{i}. {option.preference.value.upper()}")
        print(f"   Duration: {option.route.total_duration} min")
        print(f"   Cost: ₺{option.route.estimated_cost:.2f}")
        print(f"   Summary: {option.route.summary}")
        for highlight in option.highlights:
            print(f"   {highlight}")
        print()
