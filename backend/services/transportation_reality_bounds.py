"""
Reality bounds validation for transportation routes.
Prevents absurd outputs by enforcing physical constraints.
"""
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RealityBoundsValidator:
    """
    Validates route outputs against physical reality constraints.
    
    This prevents absurd results like:
    - 10+ stops in 2 minutes
    - Ferry routes under 10 minutes
    - Impossibly high speeds
    """
    
    # Minimum times by transport type (minutes)
    MIN_TIMES = {
        'FERRY': 10,  # Ferries need minimum boarding + sailing time
        'MARMARAY': 2,  # Minimum for any rail
        'METRO': 2,
        'TRAM': 2,
        'FUNICULAR': 1,
    }
    
    # Maximum realistic speeds (km/h)
    MAX_SPEEDS = {
        'FERRY': 40,  # Ferries are slow
        'MARMARAY': 100,  # High-speed rail
        'METRO': 80,
        'TRAM': 50,
        'FUNICULAR': 30,
    }
    
    # Minimum time per stop (minutes/stop)
    MIN_TIME_PER_STOP = {
        'FERRY': 2.0,  # Ferries don't really have "stops" in the metro sense
        'MARMARAY': 1.5,  # Fast trains
        'METRO': 1.2,
        'TRAM': 1.5,
        'FUNICULAR': 1.0,
    }
    
    @staticmethod
    def get_transport_type(line: str) -> str:
        """Determine transport type from line ID."""
        line_upper = line.upper()
        if line_upper == 'FERRY':
            return 'FERRY'
        elif line_upper == 'MARMARAY':
            return 'MARMARAY'
        elif line_upper.startswith('M'):
            return 'METRO'
        elif line_upper.startswith('T'):
            return 'TRAM'
        elif line_upper.startswith('F'):
            return 'FUNICULAR'
        else:
            return 'METRO'  # Default
    
    @classmethod
    def validate_segment(
        cls,
        line: str,
        stops: int,
        duration: float,
        distance: float
    ) -> Dict[str, Any]:
        """
        Validate a single route segment against reality bounds.
        
        Returns:
            Dict with 'valid', 'issues', 'warnings'
        """
        transport_type = cls.get_transport_type(line)
        issues = []
        warnings = []
        
        # Special handling for ferries: stops may be None for "Direct ferry crossing"
        if transport_type == 'FERRY' and stops is None:
            stops = 0  # Treat as no intermediate stops for validation
        
        # Check 1: Minimum time constraint
        min_time = cls.MIN_TIMES.get(transport_type, 2)
        if stops > 0 and duration < min_time:
            issues.append(f"{transport_type} segment cannot be under {min_time} min (got {duration} min)")
        
        # Check 2: Time per stop constraint (skip for ferries with no stops)
        if stops > 0:
            time_per_stop = duration / stops
            min_per_stop = cls.MIN_TIME_PER_STOP.get(transport_type, 1.0)
            if time_per_stop < min_per_stop:
                issues.append(
                    f"Impossible: {time_per_stop:.2f} min/stop is too fast for {transport_type} "
                    f"(minimum: {min_per_stop} min/stop)"
                )
        
        # Check 3: Speed constraint
        if duration > 0 and distance > 0:
            speed_kmh = (distance / duration) * 60  # Convert km/min to km/h
            max_speed = cls.MAX_SPEEDS.get(transport_type, 100)
            if speed_kmh > max_speed:
                warnings.append(
                    f"High speed: {speed_kmh:.0f} km/h (max realistic: {max_speed} km/h for {transport_type})"
                )
        
        # Check 4: Ferry distance validation
        if transport_type == 'FERRY' and distance > 10.0:
            warnings.append(
                f"⚠️ FERRY DISTANCE ANOMALY: {distance:.2f}km exceeds 10km threshold. "
                "This may indicate a polyline/Haversine calculation bug."
            )
        
        # Check 5: Many stops must take significant time
        if stops >= 10 and duration < 15:
            issues.append(f"{stops} stops in {duration} min is physically impossible")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'transport_type': transport_type
        }
    
    @classmethod
    def validate_route(cls, route: Any) -> Dict[str, Any]:
        """
        Validate an entire route against reality bounds.
        
        Args:
            route: TransitRoute object
            
        Returns:
            Dict with validation results
        """
        all_issues = []
        all_warnings = []
        valid_segments = 0
        total_segments = 0
        
        for step in route.steps:
            if step.get('type') == 'transit':
                total_segments += 1
                line = step.get('line', '')
                stops = step.get('stops', 0)
                duration = step.get('duration', 0)
                
                # Handle ferry_crossing: stops may be None
                if step.get('ferry_crossing') and stops is None:
                    stops = 0  # No intermediate stops for direct ferry crossing
                
                # Estimate distance if not provided
                # (route.total_distance is overall, not per-segment)
                estimated_distance = duration * 0.5  # Rough estimate: 30 km/h average
                
                result = cls.validate_segment(line, stops, duration, estimated_distance)
                
                if result['valid']:
                    valid_segments += 1
                else:
                    for issue in result['issues']:
                        all_issues.append(f"{line} ({step['from']} → {step['to']}): {issue}")
                
                all_warnings.extend(result['warnings'])
        
        return {
            'valid': len(all_issues) == 0,
            'valid_segments': valid_segments,
            'total_segments': total_segments,
            'issues': all_issues,
            'warnings': all_warnings
        }


def validate_route_reality(route: Any) -> bool:
    """
    Quick validation function.
    
    Returns:
        True if route passes reality checks, False otherwise
    """
    result = RealityBoundsValidator.validate_route(route)
    
    if not result['valid']:
        logger.warning(f"⚠️ Route failed reality checks:")
        for issue in result['issues']:
            logger.warning(f"  - {issue}")
    
    if result['warnings']:
        logger.info(f"ℹ️ Route has warnings:")
        for warning in result['warnings']:
            logger.info(f"  - {warning}")
    
    return result['valid']
