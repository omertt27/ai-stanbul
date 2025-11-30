"""
Route Optimizer and Alternatives Generator
===========================================

Provides multiple route options with different optimization criteria:
- Fastest route (minimize duration)
- Cheapest route (minimize cost/transfers)
- Least transfers (minimize complexity)
- Scenic route (prefer ferry/tram)
- Accessible route (wheelchair-friendly)

Uses k-shortest paths algorithm to find alternative routes.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Import accessibility checker
try:
    from .accessibility_features import get_accessibility_checker
    ACCESSIBILITY_AVAILABLE = True
    logger.info("✅ Accessibility features available")
except ImportError as e:
    ACCESSIBILITY_AVAILABLE = False
    logger.warning(f"⚠️ Accessibility features not available: {e}")


class RoutePreference(Enum):
    """Route optimization preferences"""
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    LEAST_TRANSFERS = "least_transfers"
    SCENIC = "scenic"
    ACCESSIBLE = "accessible"  # ♿ Wheelchair-accessible routes


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
            },
            RoutePreference.ACCESSIBLE: {
                'duration': 0.3,
                'transfers': 1.0,  # Minimize transfers for accessibility
                'accessibility': 1.0,  # Prioritize accessible stations
                'cost': 0.1
            }
        }
        
        # Initialize accessibility checker
        self.accessibility_checker = None
        if ACCESSIBILITY_AVAILABLE:
            try:
                self.accessibility_checker = get_accessibility_checker()
                logger.info("✅ Accessibility checker integrated with route optimizer")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize accessibility checker: {e}")
    
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
        
        # Accessibility component (negative score = better)
        if preference == RoutePreference.ACCESSIBLE and self.accessibility_checker:
            # This would need graph path data, which we don't have in TransportRoute
            # For now, give bonus for low transfers and short duration
            if hasattr(route, 'graph_path') and route.graph_path:
                accessibility_score = self.accessibility_checker.get_accessibility_score(route.graph_path)
                # Invert score (100 = best, 0 = worst) to penalty
                score -= weights.get('accessibility', 0) * accessibility_score
            else:
                # Fallback: prefer routes with fewer transfers
                transfers = len([s for s in route.steps if s.mode != 'walk']) - 1
                if transfers <= 1:
                    score -= 20  # Bonus for accessibility-friendly low transfers
        
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
        
        elif preference == RoutePreference.ACCESSIBLE:
            highlights.append(f"♿ Wheelchair-accessible route")
            if transfers <= 1:
                highlights.append("✅ Minimal transfers for easy access")
            
            # Add accessibility info if available
            if self.accessibility_checker and hasattr(route, 'graph_path') and route.graph_path:
                accessibility_highlights = self.accessibility_checker.get_accessibility_highlights(route.graph_path)
                highlights.extend(accessibility_highlights)
            else:
                # Fallback info
                if transfers == 0:
                    highlights.append("✅ Direct route - no transfer navigation needed")
                highlights.append("ℹ️ Please verify elevator availability at stations")
        
        return highlights
    
    def generate_route_explanation(self, route) -> str:
        """
        Generate natural language explanation of what makes this route unique
        
        Explains the route characteristics and why it might be chosen over alternatives.
        
        Args:
            route: TransportRoute object
            
        Returns:
            Human-readable explanation string
        """
        explanations = []
        
        # Analyze route composition
        modes_used = route.modes_used if hasattr(route, 'modes_used') else []
        transfers = len([s for s in route.steps if s.mode != 'walk']) - 1
        
        # Ferry routes are scenic
        if 'ferry' in modes_used:
            explanations.append("This route takes a scenic ferry crossing across the Bosphorus, offering beautiful water views")
        
        # Tram routes have nostalgic appeal
        if 'tram' in modes_used and 'ferry' not in modes_used:
            explanations.append("This route uses Istanbul's historic tram lines for a charming journey")
        
        # Marmaray is fast underwater crossing
        if 'marmaray' in modes_used:
            explanations.append("This route uses the Marmaray undersea tunnel for a fast crossing between continents")
        
        # Multi-modal is adventure
        if len(modes_used) >= 4:
            mode_names = {
                'metro': 'metro',
                'tram': 'tram',
                'ferry': 'ferry',
                'marmaray': 'Marmaray',
                'funicular': 'funicular',
                'bus': 'bus'
            }
            mode_list = ', '.join([mode_names.get(m, m) for m in modes_used if m != 'walk'])
            explanations.append(f"This multi-modal route combines {mode_list} for a comprehensive Istanbul transport experience")
        
        # No transfers is convenient
        if transfers == 0:
            explanations.append("This is a direct route with no transfers - simple and stress-free")
        
        # Many transfers might have other benefits
        elif transfers >= 3:
            explanations.append(f"Though this route requires {transfers} transfers, it may serve specific destinations better")
        
        # Cost considerations
        if route.estimated_cost == 0:
            explanations.append("This route is completely free with the Istanbulkart transfer discount")
        elif route.estimated_cost < 8.0:
            explanations.append(f"This economical route costs only ₺{route.estimated_cost:.2f}")
        
        # Time considerations
        if route.total_duration < 30:
            explanations.append(f"Quick journey of just {route.total_duration} minutes")
        elif route.total_duration > 60:
            explanations.append(f"Longer journey ({route.total_duration} min) but may offer better connections")
        
        # Accessibility information
        if self.accessibility_checker and hasattr(route, 'graph_path') and route.graph_path:
            is_accessible = self.accessibility_checker.is_route_accessible(route.graph_path)
            if is_accessible:
                explanations.append("This route is wheelchair accessible with elevator access at all stations")
            else:
                warnings = self.accessibility_checker.get_accessibility_warnings(route.graph_path)
                if warnings and len(warnings) > 0:
                    # Just mention accessibility might be limited
                    explanations.append("Note: Some stations on this route may have limited accessibility features")
        
        # Combine explanations
        if explanations:
            return ". ".join(explanations) + "."
        else:
            return f"Alternative route via {', '.join(modes_used[:2])} with {transfers} transfer(s)."
    
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
        
        # First pass: Find best route for each preference
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
                # Add route explanation
                explanation = self.generate_route_explanation(best_route)
                if explanation and explanation not in highlights:
                    highlights.append(f"ℹ️ {explanation}")
                
                route_options.append(RouteOption(
                    route=best_route,
                    preference=preference,
                    score=best_score,
                    highlights=highlights
                ))
        
        # Second pass: Add diverse alternatives (different transit sequences)
        # This ensures we show routes with different modes even if they're not "optimal"
        seen_in_options = {self._get_route_id(opt.route) for opt in route_options}
        
        for route in routes:
            route_id = self._get_route_id(route)
            if route_id not in seen_in_options:
                # Add this route as a "diverse" option
                # Use FASTEST preference as default for diverse routes
                score = self.calculate_route_score(route, RoutePreference.FASTEST)
                highlights = [f"🎨 Alternative route"]
                
                # Add route explanation for diverse alternatives
                explanation = self.generate_route_explanation(route)
                if explanation:
                    highlights.append(f"ℹ️ {explanation}")
                
                # Add specific highlights based on route characteristics
                if 'ferry' in route.modes_used and "ferry" not in explanation:
                    highlights.append("⛴️ Scenic ferry option")
                if len(route.modes_used) >= 4 and "multi-modal" not in explanation:
                    highlights.append("🚇 Multi-modal journey")
                
                route_options.append(RouteOption(
                    route=route,
                    preference=RoutePreference.FASTEST,  # Default preference for diverse routes
                    score=score,
                    highlights=highlights
                ))
                seen_in_options.add(route_id)
                logger.debug(f"   Added diverse alternative: {route_id}")
        
        # Remove duplicates (same route for different preferences)
        unique_options = []
        seen_routes = set()
        
        logger.debug(f"🔍 Checking {len(route_options)} route options for duplicates...")
        
        for option in sorted(route_options, key=lambda x: x.score):
            route_id = self._get_route_id(option.route)
            
            logger.debug(f"   Option: {option.preference.value}, score={option.score:.1f}, id={route_id}")
            
            if route_id not in seen_routes:
                seen_routes.add(route_id)
                unique_options.append(option)
                logger.debug(f"      ✅ Added as unique (total unique: {len(unique_options)})")
            else:
                logger.debug(f"      ❌ Duplicate detected - skipping")
        
        logger.info(f"After deduplication: {len(unique_options)} unique routes from {len(route_options)} options")
        
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
                    line_info = step.line_name if hasattr(step, 'line_name') and step.line_name else getattr(step, 'line', 'unknown')
                    transit_steps.append(f"{step.mode}:{line_info}")
            
            # Create signature from transit sequence
            if transit_steps:
                route_id = "_".join(transit_steps)
                logger.debug(f"      Route ID generated: {route_id}")
                return route_id
            else:
                # Fallback to duration + cost
                fallback_id = f"duration_{route.total_duration}_cost_{route.estimated_cost:.2f}"
                logger.debug(f"      Route ID (fallback): {fallback_id}")
                return fallback_id
        except Exception as e:
            # Ultimate fallback
            ultimate_fallback = f"{route.summary}_{route.total_duration}"
            logger.debug(f"      Route ID (error fallback): {ultimate_fallback}, error: {e}")
            return ultimate_fallback


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
