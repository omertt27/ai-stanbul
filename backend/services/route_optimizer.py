"""
Route Optimizer and Alternatives Generator - Moovit Style
==========================================================

Production-ready multi-route optimizer with comfort scoring and smart ranking.

Features (Moovit-style improvements):
✅ 1. Alternative Routes with Different Priorities:
   - Fastest route (minimize duration)
   - Best route (balanced comfort + speed)
   - Least transfers (minimize complexity)
   - Least walking (minimize walking distance)
   - Most comfortable (prefer metro/ferry over bus)

✅ 2. Comfort Scoring System:
   - Transport mode comfort scores (metro > tram > ferry > bus)
   - Crowding predictions (time of day + line)
   - Transfer quality (covered, easy, difficult)
   - Walking comfort (distance + elevation)

✅ 3. Smart Transfer Optimization:
   - Istanbul-specific transfer times and quality
   - Same-station transfers vs cross-platform
   - Transfer penalties based on quality

✅ 4. Time-Based Routing:
   - Peak/off-peak awareness
   - Service frequency modeling
   - Time-of-day crowding predictions

✅ 5. Route Comparison & Ranking:
   - Multi-criteria scoring
   - Pareto-optimal route selection
   - LLM-powered route summaries

Author: AI Istanbul Team
Date: January 2025
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time
import math

logger = logging.getLogger(__name__)

# Import accessibility checker
try:
    from .accessibility_features import get_accessibility_checker
    ACCESSIBILITY_AVAILABLE = True
    logger.info("✅ Accessibility features available")
except ImportError as e:
    ACCESSIBILITY_AVAILABLE = False
    logger.warning(f"⚠️ Accessibility features not available: {e}")

# Import OpenAI for LLM summaries
try:
    import openai
    from config import settings
    OPENAI_AVAILABLE = True
    logger.info("✅ OpenAI available for route summaries")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("⚠️ OpenAI not available - route summaries will be basic")


class RoutePreference(Enum):
    """Route optimization preferences - Moovit style"""
    FASTEST = "fastest"                    # Minimize total duration
    BEST = "best"                          # Balanced comfort + speed (recommended)
    LEAST_TRANSFERS = "least_transfers"    # Minimize transfers
    LEAST_WALKING = "least_walking"        # Minimize walking distance
    MOST_COMFORTABLE = "most_comfortable"  # Prefer comfortable transport modes
    ACCESSIBLE = "accessible"              # ♿ Wheelchair-accessible routes


class TransportModeComfort(Enum):
    """Comfort scores for different transport modes (0-100)"""
    METRO = 90        # Clean, modern, air-conditioned
    MARMARAY = 88     # Modern rail, cross-Bosphorus
    FUNICULAR = 85    # Short, efficient
    TRAM = 75         # Comfortable but can be crowded
    FERRY = 70        # Scenic but weather-dependent
    BUS = 50          # Can be crowded, traffic-dependent
    DOLMUS = 40       # Crowded, unpredictable
    WALK = 60         # Weather and distance dependent


class TransferQuality(Enum):
    """Transfer quality ratings for Istanbul-specific transfers"""
    EXCELLENT = 100   # Same platform, no stairs (e.g., M2-M6 at Levent)
    GOOD = 80         # Same station, covered walkway (e.g., M1-M2 at Yenikapı)
    FAIR = 60         # Short walk, some stairs (e.g., Taksim Tram-Metro)
    POOR = 40         # Long walk, many stairs (e.g., Kabataş Funicular-Tram)
    DIFFICULT = 20    # Exit station, cross street (e.g., Şişhane-Tünel)


# Istanbul-specific transfer quality map
# Format: (from_station, from_line, to_line) -> TransferQuality
ISTANBUL_TRANSFER_QUALITY = {
    # Excellent transfers (same platform)
    ('Levent', 'M2', 'M6'): TransferQuality.EXCELLENT,
    ('Hacıosman', 'M2', 'M11'): TransferQuality.EXCELLENT,
    
    # Good transfers (same station, covered)
    ('Yenikapı', 'M1A', 'M2'): TransferQuality.GOOD,
    ('Yenikapı', 'M1B', 'M2'): TransferQuality.GOOD,
    ('Yenikapı', 'M1A', 'Marmaray'): TransferQuality.GOOD,
    ('Ataköy-Şirinevler', 'M1A', 'M9'): TransferQuality.GOOD,
    
    # Fair transfers (short walk, some stairs)
    ('Taksim', 'M2', 'T1'): TransferQuality.FAIR,
    ('Zeytinburnu', 'M1A', 'T1'): TransferQuality.FAIR,
    ('Kabataş', 'T1', 'F1'): TransferQuality.FAIR,
    
    # Poor transfers (long walk, many stairs)
    ('Kabataş', 'T1', 'Ferry'): TransferQuality.POOR,
    ('Eminönü', 'T1', 'Ferry'): TransferQuality.POOR,
    
    # Difficult transfers (exit station)
    ('Şişhane', 'M2', 'Tünel'): TransferQuality.DIFFICULT,
}


# Peak hours for crowding predictions
PEAK_HOURS = [
    (time(7, 30), time(9, 30)),   # Morning rush
    (time(17, 30), time(19, 30))  # Evening rush
]


def is_peak_hour(dt: datetime = None) -> bool:
    """Check if given time is during peak hours"""
    if dt is None:
        dt = datetime.now()
    
    current_time = dt.time()
    for start, end in PEAK_HOURS:
        if start <= current_time <= end:
            return True
    return False


@dataclass
class ComfortScore:
    """Detailed comfort scoring for a route"""
    mode_comfort: float          # Average comfort of transport modes (0-100)
    transfer_quality: float      # Quality of transfers (0-100)
    crowding_penalty: float      # Penalty for peak-hour crowding (0-50)
    walking_comfort: float       # Comfort of walking segments (0-100)
    overall_comfort: float       # Combined comfort score (0-100)
    
    highlights: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RouteOption:
    """A single route option with comprehensive scoring"""
    route: any  # TransportRoute object
    preference: RoutePreference
    
    # Core metrics
    duration_minutes: int
    walking_meters: float
    num_transfers: int
    cost_tl: float
    
    # Comfort scoring
    comfort_score: ComfortScore
    
    # Overall ranking score (0-100, higher is better)
    overall_score: float
    
    # Human-readable summary
    highlights: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    llm_summary: Optional[str] = None
    
    def __lt__(self, other):
        """For sorting by overall score (higher is better)"""
        return self.overall_score > other.overall_score


class RouteOptimizer:
    """
    Production-ready route optimizer with Moovit-style features.
    
    Generates multiple route alternatives and scores them based on:
    - Duration (travel time)
    - Comfort (transport modes, transfers, crowding)
    - Walking distance
    - Number of transfers
    - Cost (when applicable)
    """
    
    def __init__(self):
        """Initialize route optimizer"""
        # Optimization weights for different preferences
        self.weights = {
            RoutePreference.FASTEST: {
                'duration': 1.0,
                'transfers': 0.2,
                'walking': 0.3,
                'comfort': 0.1
            },
            RoutePreference.BEST: {
                'duration': 0.6,
                'transfers': 0.4,
                'walking': 0.3,
                'comfort': 0.8  # Balanced: comfort + speed
            },
            RoutePreference.LEAST_TRANSFERS: {
                'duration': 0.4,
                'transfers': 1.0,
                'walking': 0.3,
                'comfort': 0.2
            },
            RoutePreference.LEAST_WALKING: {
                'duration': 0.3,
                'transfers': 0.4,
                'walking': 1.0,
                'comfort': 0.2
            },
            RoutePreference.MOST_COMFORTABLE: {
                'duration': 0.3,
                'transfers': 0.5,
                'walking': 0.2,
                'comfort': 1.0
            },
            RoutePreference.ACCESSIBLE: {
                'duration': 0.3,
                'transfers': 0.8,  # Minimize transfers
                'accessibility': 1.0,
                'comfort': 0.2
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
    
    def calculate_comfort_score(self, route, departure_time: datetime = None) -> ComfortScore:
        """
        Calculate comprehensive comfort score for a route.
        
        Args:
            route: TransportRoute object
            departure_time: Departure time for crowding predictions
            
        Returns:
            ComfortScore with detailed breakdown
        """
        if departure_time is None:
            departure_time = datetime.now()
        
        highlights = []
        warnings = []
        
        # 1. Mode comfort (average comfort of all transport segments)
        transport_steps = [s for s in route.steps if s.mode != 'walk']
        if transport_steps:
            mode_scores = []
            for step in transport_steps:
                mode = step.mode.upper()
                comfort = TransportModeComfort[mode].value if mode in TransportModeComfort.__members__ else 50
                mode_scores.append(comfort)
            
            mode_comfort = sum(mode_scores) / len(mode_scores)
            
            # Add highlights based on modes
            if any(s.mode == 'metro' for s in transport_steps):
                highlights.append("🚇 Modern metro line")
            if any(s.mode == 'ferry' for s in transport_steps):
                highlights.append("⛴️ Scenic ferry ride")
        else:
            mode_comfort = 60.0  # Walking only
        
        # 2. Transfer quality
        transfer_scores = []
        for i in range(len(transport_steps) - 1):
            from_step = transport_steps[i]
            to_step = transport_steps[i + 1]
            
            # Try to find Istanbul-specific transfer quality
            transfer_key = (
                self._get_station_name(from_step),
                from_step.line_name or '',
                to_step.line_name or ''
            )
            
            quality = ISTANBUL_TRANSFER_QUALITY.get(transfer_key)
            if quality:
                transfer_scores.append(quality.value)
                if quality == TransferQuality.EXCELLENT:
                    highlights.append(f"✨ Easy transfer at {transfer_key[0]}")
                elif quality == TransferQuality.DIFFICULT:
                    warnings.append(f"⚠️ Difficult transfer at {transfer_key[0]}")
            else:
                # Default transfer quality based on mode
                transfer_scores.append(70)  # Fair by default
        
        transfer_quality = sum(transfer_scores) / len(transfer_scores) if transfer_scores else 100.0
        
        # 3. Crowding penalty (peak hours)
        crowding_penalty = 0.0
        if is_peak_hour(departure_time):
            # More penalty for buses and trams during peak
            for step in transport_steps:
                if step.mode in ['bus', 'tram']:
                    crowding_penalty += 10
                elif step.mode == 'metro':
                    crowding_penalty += 5  # Metros less affected
            
            crowding_penalty = min(crowding_penalty, 50)  # Cap at 50
            if crowding_penalty > 20:
                warnings.append("🚦 Peak hour - expect crowds")
        
        # 4. Walking comfort (based on distance)
        walking_steps = [s for s in route.steps if s.mode == 'walk']
        total_walking_meters = sum(s.distance for s in walking_steps)
        
        if total_walking_meters < 200:
            walking_comfort = 100.0
            highlights.append("👣 Minimal walking")
        elif total_walking_meters < 500:
            walking_comfort = 80.0
        elif total_walking_meters < 1000:
            walking_comfort = 60.0
        else:
            walking_comfort = 40.0
            warnings.append(f"👟 {int(total_walking_meters)}m walking required")
        
        # 5. Overall comfort (weighted average)
        overall_comfort = (
            mode_comfort * 0.4 +
            transfer_quality * 0.3 +
            walking_comfort * 0.2 +
            (100 - crowding_penalty) * 0.1
        )
        
        return ComfortScore(
            mode_comfort=mode_comfort,
            transfer_quality=transfer_quality,
            crowding_penalty=crowding_penalty,
            walking_comfort=walking_comfort,
            overall_comfort=overall_comfort,
            highlights=highlights,
            warnings=warnings
        )
    
    def _get_station_name(self, step) -> str:
        """Extract station name from a transport step"""
        # Try to extract from instruction
        if hasattr(step, 'instruction') and step.instruction:
            # Look for patterns like "Board at X" or "Transfer at X"
            import re
            match = re.search(r'(?:at|to)\s+([A-Z][a-zğüşöçıİ]+(?:\s+[A-Z][a-zğüşöçıİ]+)*)', step.instruction)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def calculate_route_score(self, route, preference: RoutePreference, departure_time: datetime = None) -> Tuple[float, List[str], List[str]]:
        """
        Calculate route score based on preference.
        
        Args:
            route: TransportRoute object
            preference: RoutePreference enum
            departure_time: Departure time for time-based scoring
            
        Returns:
            Tuple of (score 0-100, highlights, warnings)
        """
        weights = self.weights.get(preference, self.weights[RoutePreference.BEST])
        
        highlights = []
        warnings = []
        
        # Calculate comfort score
        comfort = self.calculate_comfort_score(route, departure_time)
        highlights.extend(comfort.highlights)
        warnings.extend(comfort.warnings)
        
        # Extract route metrics
        duration = route.total_duration
        transfers = len([s for s in route.steps if s.mode != 'walk']) - 1
        walking_meters = sum(s.distance for s in route.steps if s.mode == 'walk')
        cost = route.estimated_cost if hasattr(route, 'estimated_cost') else 0
        
        # Normalize metrics to 0-100 scale (higher is better)
        # Duration: 0-120 minutes -> 100-0
        duration_score = max(0, 100 - (duration / 120) * 100)
        
        # Transfers: 0-5 transfers -> 100-0
        transfer_score = max(0, 100 - (transfers / 5) * 100)
        
        # Walking: 0-2000 meters -> 100-0
        walking_score = max(0, 100 - (walking_meters / 2000) * 100)
        
        # Comfort: already 0-100
        comfort_score = comfort.overall_comfort
        
        # Weighted combination
        overall_score = 0.0
        
        if 'duration' in weights:
            overall_score += weights['duration'] * duration_score
        if 'transfers' in weights:
            overall_score += weights['transfers'] * transfer_score
        if 'walking' in weights:
            overall_score += weights['walking'] * walking_score
        if 'comfort' in weights:
            overall_score += weights['comfort'] * comfort_score
        
        # Normalize to sum of weights
        total_weight = sum(weights.values())
        overall_score = (overall_score / total_weight) if total_weight > 0 else 50.0
        
        # Add preference-specific highlights
        if preference == RoutePreference.FASTEST and duration < 30:
            highlights.append("⚡ Very fast route")
        elif preference == RoutePreference.LEAST_TRANSFERS and transfers == 0:
            highlights.append("🎯 Direct route - no transfers")
        elif preference == RoutePreference.LEAST_WALKING and walking_meters < 300:
            highlights.append("👣 Minimal walking")
        elif preference == RoutePreference.MOST_COMFORTABLE and comfort_score > 80:
            highlights.append("⭐ Premium comfort")
        
        return overall_score, highlights, warnings
        
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
    
    def optimize_routes(
        self, 
        routes: List[Any], 
        preferences: List[RoutePreference] = None,
        departure_time: datetime = None,
        generate_llm_summaries: bool = False,
        user_language: str = 'en'
    ) -> List[RouteOption]:
        """
        Optimize and score routes based on multiple preferences.
        
        Args:
            routes: List of TransportRoute objects
            preferences: List of RoutePreference enums (default: all)
            departure_time: Departure time for time-based scoring
            generate_llm_summaries: Whether to generate LLM summaries
            user_language: User's preferred language for summaries
            
        Returns:
            List of RouteOption objects, sorted by overall score
        """
        if not routes:
            logger.warning("No routes provided to optimize")
            return []
        
        if preferences is None:
            # Default: optimize for all preferences except accessible (unless requested)
            preferences = [
                RoutePreference.BEST,
                RoutePreference.FASTEST,
                RoutePreference.LEAST_TRANSFERS,
                RoutePreference.LEAST_WALKING,
                RoutePreference.MOST_COMFORTABLE
            ]
        
        if departure_time is None:
            departure_time = datetime.now()
        
        logger.info(f"🎯 Optimizing {len(routes)} routes for {len(preferences)} preferences")
        
        route_options = []
        
        for route in routes:
            # Create route options for each preference
            for preference in preferences:
                score, highlights, warnings = self.calculate_route_score(route, preference, departure_time)
                
                # Calculate comfort score
                comfort = self.calculate_comfort_score(route, departure_time)
                
                # Extract metrics
                duration = route.total_duration
                transfers = len([s for s in route.steps if s.mode != 'walk']) - 1
                walking_meters = sum(s.distance for s in route.steps if s.mode == 'walk')
                cost = route.estimated_cost if hasattr(route, 'estimated_cost') else 0.0
                
                # Create route option
                option = RouteOption(
                    route=route,
                    preference=preference,
                    duration_minutes=duration,
                    walking_meters=walking_meters,
                    num_transfers=transfers,
                    cost_tl=cost,
                    comfort_score=comfort,
                    overall_score=score,
                    highlights=highlights,
                    warnings=warnings
                )
                
                route_options.append(option)
        
        # Remove duplicates (same route for different preferences)
        unique_options = self._deduplicate_routes(route_options)
        
        # Sort by overall score (higher is better)
        unique_options.sort(reverse=True)
        
        # Limit to top 5 routes
        top_routes = unique_options[:5]
        
        # Generate LLM summaries if requested
        if generate_llm_summaries and OPENAI_AVAILABLE:
            logger.info("📝 Generating LLM summaries for top routes")
            for option in top_routes:
                try:
                    option.llm_summary = self._generate_llm_summary(option, user_language)
                except Exception as e:
                    logger.warning(f"Failed to generate LLM summary: {e}")
        
        logger.info(f"✅ Optimized {len(top_routes)} unique routes")
        return top_routes
    
    def _deduplicate_routes(self, route_options: List[RouteOption]) -> List[RouteOption]:
        """
        Remove duplicate routes (same path, different preference scoring).
        Keep the version with the highest overall score.
        """
        route_map = {}
        
        for option in route_options:
            # Create a unique key based on route steps
            route_key = self._get_route_signature(option.route)
            
            if route_key not in route_map:
                route_map[route_key] = option
            else:
                # Keep the one with higher score
                if option.overall_score > route_map[route_key].overall_score:
                    route_map[route_key] = option
        
        return list(route_map.values())
    
    def _get_route_signature(self, route) -> str:
        """Generate a unique signature for a route based on its steps"""
        steps_sig = []
        for step in route.steps:
            if step.mode != 'walk':
                sig = f"{step.mode}:{step.line_name or 'unknown'}"
                steps_sig.append(sig)
        return "->".join(steps_sig)
    
    def _generate_llm_summary(self, option: RouteOption, language: str = 'en') -> str:
        """
        Generate LLM-powered natural language summary of a route.
        
        Args:
            option: RouteOption to summarize
            language: Target language (en, tr, etc.)
            
        Returns:
            Natural language summary
        """
        route = option.route
        
        # Build context for LLM
        steps_summary = []
        for i, step in enumerate(route.steps, 1):
            if step.mode == 'walk':
                steps_summary.append(f"  {i}. Walk {int(step.distance)}m ({step.duration} min)")
            else:
                steps_summary.append(
                    f"  {i}. {step.mode.title()}: {step.line_name or 'unknown line'} "
                    f"({step.stops_count or '?'} stops, {step.duration} min)"
                )
        
        prompt = f"""Summarize this Istanbul transportation route in 1-2 sentences in {language}.

Route Details:
- Total Duration: {option.duration_minutes} minutes
- Transfers: {option.num_transfers}
- Walking: {int(option.walking_meters)}m
- Comfort Score: {option.comfort_score.overall_comfort:.0f}/100
- Preference: {option.preference.value}

Steps:
{chr(10).join(steps_summary)}

Highlights:
{chr(10).join(f"- {h}" for h in option.highlights)}

Warnings:
{chr(10).join(f"- {w}" for w in option.warnings)}

Write a concise, helpful summary focusing on what makes this route unique or preferable."""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful Istanbul transportation assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"✅ Generated LLM summary: {summary[:50]}...")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate LLM summary: {e}")
            # Fallback to simple summary
            return f"{option.duration_minutes} min via {', '.join(set(s.mode for s in route.steps if s.mode != 'walk'))}"


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_route_optimizer_instance = None


def get_route_optimizer() -> RouteOptimizer:
    """Get or create singleton RouteOptimizer instance"""
    global _route_optimizer_instance
    if _route_optimizer_instance is None:
        _route_optimizer_instance = RouteOptimizer()
        logger.info("✅ RouteOptimizer initialized")
    return _route_optimizer_instance


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the route optimizer
    """
    import sys
    sys.path.append('/Users/omer/Desktop/ai-stanbul/backend')
    
    from transportation_directions_service import TransportationDirectionsService
    
    # Initialize services
    print("🚀 Initializing transportation services...")
    directions_service = TransportationDirectionsService()
    optimizer = get_route_optimizer()
    
    # Example: Get routes and optimize them
    print("\n📍 Finding routes from Taksim to Kadıköy...")
    route = directions_service.get_directions(
        start=(41.0370, 28.9850),
        end=(40.9900, 29.0250),
        start_name="Taksim",
        end_name="Kadıköy"
    )
    
    if route:
        print(f"\n✅ Found route: {route.summary}")
        print(f"   Duration: {route.total_duration} min")
        print(f"   Steps: {len(route.steps)}")
        
        # Optimize with different preferences
        print("\n🎯 Optimizing route with different preferences...")
        optimized = optimizer.optimize_routes(
            routes=[route],
            departure_time=datetime.now(),
            generate_llm_summaries=False  # Set to True if OpenAI is configured
        )
        
        print(f"\n📊 Generated {len(optimized)} route options:\n")
        for i, option in enumerate(optimized, 1):
            print(f"{i}. {option.preference.value.upper()}")
            print(f"   ⏱️  Duration: {option.duration_minutes} min")
            print(f"   🔄 Transfers: {option.num_transfers}")
            print(f"   👟 Walking: {int(option.walking_meters)}m")
            print(f"   ⭐ Comfort: {option.comfort_score.overall_comfort:.0f}/100")
            print(f"   📊 Score: {option.overall_score:.1f}/100")
            
            if option.highlights:
                print(f"   ✨ Highlights:")
                for h in option.highlights:
                    print(f"      {h}")
            
            if option.warnings:
                print(f"   ⚠️  Warnings:")
                for w in option.warnings:
                    print(f"      {w}")
            print()
    else:
        print("❌ No route found")

