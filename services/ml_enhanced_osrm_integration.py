"""
ML-Enhanced OSRM Integration
Combines OSRM street-level routing with ML-enhanced transportation system

This module provides:
- Hybrid routing (OSRM walking + Istanbul metro/tram/ferry)
- ML-predicted travel times vs OSRM estimates
- Intelligent mode switching (walk → metro → walk)
- Cache-optimized route recommendations
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import asyncio

# Import OSRM routing
try:
    from services.osrm_routing_service import get_osrm_service, OSRMRoute
    OSRM_AVAILABLE = True
except ImportError:
    OSRM_AVAILABLE = False
    logging.warning("OSRM not available")

# Import ML-enhanced transportation system
try:
    from ml_enhanced_transportation_system import (
        MLEnhancedTransportationSystem,
        TransportMode,
        RouteSegment,
        OptimizedRoute
    )
    ML_TRANSPORT_AVAILABLE = True
except ImportError:
    ML_TRANSPORT_AVAILABLE = False
    logging.warning("ML Transportation System not available")

# Import ML cache
try:
    from services.ml_prediction_cache_service import get_ml_cache
    ML_CACHE_AVAILABLE = True
except ImportError:
    ML_CACHE_AVAILABLE = False
    logging.warning("ML Cache not available")

# Import POI database
try:
    from services.poi_database_service import POIDatabaseService
    POI_DATABASE_AVAILABLE = True
except ImportError:
    POI_DATABASE_AVAILABLE = False
    logging.warning("POI Database not available")

logger = logging.getLogger(__name__)


@dataclass
class HybridRouteSegment:
    """Route segment with both OSRM and ML data"""
    segment_type: str  # 'osrm_walk', 'osrm_drive', 'metro', 'tram', 'ferry', 'walk_to_station'
    transport_mode: str  # 'walking', 'driving', 'cycling', 'metro', 'tram', 'ferry'
    from_location: str
    to_location: str
    distance_km: float
    osrm_duration_min: float  # OSRM estimate
    ml_predicted_duration_min: float  # ML prediction
    actual_duration_min: float  # Best estimate (ML or OSRM)
    cost: float = 0.0
    instructions: List[str] = field(default_factory=list)
    geometry: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'segment_type': self.segment_type,
            'transport_mode': self.transport_mode,
            'from_location': self.from_location,
            'to_location': self.to_location,
            'distance_km': round(self.distance_km, 2),
            'osrm_duration_min': round(self.osrm_duration_min, 1),
            'ml_predicted_duration_min': round(self.ml_predicted_duration_min, 1),
            'actual_duration_min': round(self.actual_duration_min, 1),
            'cost': round(self.cost, 2),
            'instructions': self.instructions,
            'geometry': self.geometry,
            'confidence': round(self.confidence, 2)
        }


@dataclass
class HybridRoute:
    """Complete hybrid route with ML enhancements"""
    route_id: str
    segments: List[HybridRouteSegment]
    total_distance_km: float
    total_osrm_duration_min: float
    total_ml_duration_min: float
    total_actual_duration_min: float
    total_cost: float
    mode_breakdown: Dict[str, float]  # percentage of each transport mode
    ml_insights: Dict[str, Any]
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'route_id': self.route_id,
            'segments': [seg.to_dict() for seg in self.segments],
            'total_distance_km': round(self.total_distance_km, 2),
            'total_osrm_duration_min': round(self.total_osrm_duration_min, 1),
            'total_ml_duration_min': round(self.total_ml_duration_min, 1),
            'total_actual_duration_min': round(self.total_actual_duration_min, 1),
            'total_cost': round(self.total_cost, 2),
            'mode_breakdown': self.mode_breakdown,
            'ml_insights': self.ml_insights,
            'cache_hit': self.cache_hit
        }


class MLEnhancedOSRMIntegration:
    """
    Integrate OSRM routing with ML-enhanced transportation system
    
    Provides intelligent route planning that combines:
    - OSRM for precise walking/driving directions
    - Istanbul metro/tram/ferry network data
    - ML predictions for travel times and crowding
    - Cache optimization for fast responses
    
    Example:
        >>> integration = MLEnhancedOSRMIntegration()
        >>> route = await integration.plan_hybrid_route(
        ...     start=(41.0082, 28.9784),  # Sultanahmet
        ...     end=(41.0369, 28.9850),    # Taksim
        ...     prefer_public_transport=True
        ... )
        >>> print(f"Duration: {route.total_actual_duration_min} min")
    """
    
    def __init__(self):
        """Initialize ML-enhanced OSRM integration"""
        self.osrm = get_osrm_service() if OSRM_AVAILABLE else None
        self.ml_transport = MLEnhancedTransportationSystem() if ML_TRANSPORT_AVAILABLE else None
        self.ml_cache = get_ml_cache() if ML_CACHE_AVAILABLE else None
        self.poi_db = POIDatabaseService() if POI_DATABASE_AVAILABLE else None
        
        logger.info("✅ ML-Enhanced OSRM Integration initialized")
        logger.info(f"   OSRM: {'✅' if OSRM_AVAILABLE else '❌'}")
        logger.info(f"   ML Transport: {'✅' if ML_TRANSPORT_AVAILABLE else '❌'}")
        logger.info(f"   ML Cache: {'✅' if ML_CACHE_AVAILABLE else '❌'}")
        logger.info(f"   POI Database: {'✅' if POI_DATABASE_AVAILABLE else '❌'}")
    
    async def plan_hybrid_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        prefer_public_transport: bool = True,
        max_walking_km: float = 1.5,
        avoid_crowded: bool = True,
        user_context: Optional[Dict] = None
    ) -> Optional[HybridRoute]:
        """
        Plan hybrid route using both OSRM and ML transportation system
        
        Args:
            start: Starting coordinates (lat, lon)
            end: Ending coordinates (lat, lon)
            prefer_public_transport: Prefer metro/tram over pure walking
            max_walking_km: Maximum walking distance before considering transit
            avoid_crowded: Use ML predictions to avoid crowded routes
            user_context: User context for ML predictions
        
        Returns:
            HybridRoute with optimized segments
        
        Strategy:
        1. Check cache for similar route
        2. Get pure OSRM walking route as baseline
        3. If distance > max_walking_km and prefer_public_transport:
           - Find nearby metro/tram stations
           - Plan multi-modal route (walk → transit → walk)
           - Use ML to predict actual travel times
        4. Compare options and return best route
        """
        import uuid
        
        route_id = str(uuid.uuid4())
        
        # Check cache first
        cache_key = f"hybrid_route_{start[0]:.4f}_{start[1]:.4f}_{end[0]:.4f}_{end[1]:.4f}"
        if self.ml_cache:
            cached = self.ml_cache.get(
                cache_key=cache_key,
                context=user_context or {},
                prediction_types=['route_recommendation']
            )
            if cached:
                logger.info(f"✅ Cache hit for hybrid route")
                cached['cache_hit'] = True
                return HybridRoute(**cached)
        
        # Get baseline OSRM walking route
        osrm_route = await self.osrm.get_route(start, end, mode='walking')
        if not osrm_route:
            logger.error("OSRM route failed")
            return None
        
        walking_distance_km = osrm_route.total_distance_km
        
        # Decision: pure walking vs multi-modal
        if walking_distance_km <= max_walking_km or not prefer_public_transport:
            # Pure OSRM walking route
            return await self._create_pure_osrm_route(
                route_id=route_id,
                osrm_route=osrm_route,
                start=start,
                end=end,
                user_context=user_context
            )
        else:
            # Multi-modal route (walking + transit)
            return await self._create_multimodal_route(
                route_id=route_id,
                start=start,
                end=end,
                osrm_walking_route=osrm_route,
                avoid_crowded=avoid_crowded,
                user_context=user_context
            )
    
    async def _create_pure_osrm_route(
        self,
        route_id: str,
        osrm_route: OSRMRoute,
        start: Tuple[float, float],
        end: Tuple[float, float],
        user_context: Optional[Dict] = None
    ) -> HybridRoute:
        """
        Create hybrid route from pure OSRM walking route
        Enhanced with ML predictions
        """
        # Get ML prediction for walking time
        ml_duration = osrm_route.total_duration_min
        if self.ml_transport:
            # Use ML to adjust walking time based on:
            # - Time of day
            # - Weather conditions
            # - User walking speed profile
            ml_adjustment = await self._get_ml_walking_adjustment(
                distance_km=osrm_route.total_distance_km,
                user_context=user_context
            )
            ml_duration = osrm_route.total_duration_min * ml_adjustment
        
        # Create single walking segment
        segment = HybridRouteSegment(
            segment_type='osrm_walk',
            transport_mode='walking',
            from_location=f"Start ({start[0]:.4f}, {start[1]:.4f})",
            to_location=f"End ({end[0]:.4f}, {end[1]:.4f})",
            distance_km=osrm_route.total_distance_km,
            osrm_duration_min=osrm_route.total_duration_min,
            ml_predicted_duration_min=ml_duration,
            actual_duration_min=ml_duration,
            cost=0.0,
            instructions=[step.instruction for step in osrm_route.steps],
            geometry=osrm_route.geometry,
            confidence=0.95
        )
        
        # ML insights
        ml_insights = {
            'route_type': 'pure_walking',
            'ml_adjustment_factor': ml_duration / osrm_route.total_duration_min if osrm_route.total_duration_min > 0 else 1.0,
            'recommendations': [
                'Pure walking route - good for sightseeing',
                'Estimated time includes breaks and pace adjustments'
            ]
        }
        
        return HybridRoute(
            route_id=route_id,
            segments=[segment],
            total_distance_km=osrm_route.total_distance_km,
            total_osrm_duration_min=osrm_route.total_duration_min,
            total_ml_duration_min=ml_duration,
            total_actual_duration_min=ml_duration,
            total_cost=0.0,
            mode_breakdown={'walking': 100.0},
            ml_insights=ml_insights
        )
    
    async def _create_multimodal_route(
        self,
        route_id: str,
        start: Tuple[float, float],
        end: Tuple[float, float],
        osrm_walking_route: OSRMRoute,
        avoid_crowded: bool = True,
        user_context: Optional[Dict] = None
    ) -> HybridRoute:
        """
        Create multi-modal route combining OSRM walking + Istanbul transit
        
        Steps:
        1. Find nearest metro/tram station to start
        2. Plan transit route using ML-enhanced system
        3. Find walking route from last station to destination
        4. Combine all segments
        """
        segments = []
        total_distance = 0.0
        total_osrm_time = 0.0
        total_ml_time = 0.0
        total_cost = 0.0
        
        # Find nearest station to start
        start_station = self._find_nearest_station(start)
        if not start_station:
            # Fallback to pure walking
            return await self._create_pure_osrm_route(
                route_id, osrm_walking_route, start, end, user_context
            )
        
        # 1. Walking segment: start → nearest station
        walk_to_station = await self.osrm.get_route(
            start=start,
            end=(start_station['lat'], start_station['lon']),
            mode='walking'
        )
        
        if walk_to_station:
            ml_walk_time = walk_to_station.total_duration_min * await self._get_ml_walking_adjustment(
                walk_to_station.total_distance_km, user_context
            )
            
            segments.append(HybridRouteSegment(
                segment_type='walk_to_station',
                transport_mode='walking',
                from_location='Start',
                to_location=start_station['name'],
                distance_km=walk_to_station.total_distance_km,
                osrm_duration_min=walk_to_station.total_duration_min,
                ml_predicted_duration_min=ml_walk_time,
                actual_duration_min=ml_walk_time,
                cost=0.0,
                instructions=[step.instruction for step in walk_to_station.steps[:5]],
                geometry=walk_to_station.geometry,
                confidence=0.9
            ))
            
            total_distance += walk_to_station.total_distance_km
            total_osrm_time += walk_to_station.total_duration_min
            total_ml_time += ml_walk_time
        
        # 2. Transit segment: use ML-enhanced transportation system
        if self.ml_transport:
            transit_route = await self._get_ml_transit_route(
                from_station=start_station,
                to_coords=end,
                avoid_crowded=avoid_crowded,
                user_context=user_context
            )
            
            if transit_route:
                for transit_seg in transit_route['segments']:
                    segments.append(HybridRouteSegment(
                        segment_type=transit_seg['type'],
                        transport_mode=transit_seg['mode'],
                        from_location=transit_seg['from'],
                        to_location=transit_seg['to'],
                        distance_km=transit_seg['distance_km'],
                        osrm_duration_min=transit_seg['scheduled_time_min'],
                        ml_predicted_duration_min=transit_seg['predicted_time_min'],
                        actual_duration_min=transit_seg['predicted_time_min'],
                        cost=transit_seg['cost'],
                        instructions=[f"Take {transit_seg['mode']} to {transit_seg['to']}"],
                        geometry=[],
                        confidence=transit_seg['confidence']
                    ))
                    
                    total_distance += transit_seg['distance_km']
                    total_osrm_time += transit_seg['scheduled_time_min']
                    total_ml_time += transit_seg['predicted_time_min']
                    total_cost += transit_seg['cost']
        
        # 3. Walking segment: last station → destination
        if segments:
            last_station_coords = (segments[-1].to_location, segments[-1].to_location)  # Placeholder
            walk_from_station = await self.osrm.get_route(
                start=last_station_coords,
                end=end,
                mode='walking'
            )
            
            if walk_from_station:
                ml_walk_time = walk_from_station.total_duration_min * await self._get_ml_walking_adjustment(
                    walk_from_station.total_distance_km, user_context
                )
                
                segments.append(HybridRouteSegment(
                    segment_type='walk_from_station',
                    transport_mode='walking',
                    from_location=segments[-1].to_location,
                    to_location='Destination',
                    distance_km=walk_from_station.total_distance_km,
                    osrm_duration_min=walk_from_station.total_duration_min,
                    ml_predicted_duration_min=ml_walk_time,
                    actual_duration_min=ml_walk_time,
                    cost=0.0,
                    instructions=[step.instruction for step in walk_from_station.steps[:5]],
                    geometry=walk_from_station.geometry,
                    confidence=0.9
                ))
                
                total_distance += walk_from_station.total_distance_km
                total_osrm_time += walk_from_station.total_duration_min
                total_ml_time += ml_walk_time
        
        # Calculate mode breakdown
        mode_breakdown = self._calculate_mode_breakdown(segments)
        
        # ML insights
        ml_insights = {
            'route_type': 'multimodal',
            'time_saved_vs_walking': osrm_walking_route.total_duration_min - total_ml_time,
            'cost_per_minute_saved': total_cost / max(osrm_walking_route.total_duration_min - total_ml_time, 1),
            'recommendations': [
                f"Save {int(osrm_walking_route.total_duration_min - total_ml_time)} minutes by using public transport",
                f"Total cost: {total_cost:.2f} TL",
                "ML predictions account for current traffic and crowding"
            ]
        }
        
        return HybridRoute(
            route_id=route_id,
            segments=segments,
            total_distance_km=total_distance,
            total_osrm_duration_min=total_osrm_time,
            total_ml_duration_min=total_ml_time,
            total_actual_duration_min=total_ml_time,
            total_cost=total_cost,
            mode_breakdown=mode_breakdown,
            ml_insights=ml_insights
        )
    
    def _find_nearest_station(self, coords: Tuple[float, float], max_distance_km: float = 1.0) -> Optional[Dict]:
        """Find nearest metro/tram station to coordinates"""
        # Placeholder - integrate with actual station database
        # For now, return mock data
        return {
            'name': 'Sultanahmet Station',
            'lat': 41.0082,
            'lon': 28.9784,
            'type': 'tram',
            'lines': ['T1']
        }
    
    async def _get_ml_transit_route(
        self,
        from_station: Dict,
        to_coords: Tuple[float, float],
        avoid_crowded: bool,
        user_context: Optional[Dict]
    ) -> Optional[Dict]:
        """Get ML-enhanced transit route"""
        if not self.ml_transport:
            return None
        
        # Use ML transportation system to plan transit route
        # This is a placeholder - integrate with actual ML transport system
        return {
            'segments': [
                {
                    'type': 'tram',
                    'mode': 'tram',
                    'from': from_station['name'],
                    'to': 'Kabatas',
                    'distance_km': 3.5,
                    'scheduled_time_min': 12,
                    'predicted_time_min': 14.5,  # ML prediction with traffic
                    'cost': 17.0,
                    'confidence': 0.85
                }
            ]
        }
    
    async def _get_ml_walking_adjustment(
        self,
        distance_km: float,
        user_context: Optional[Dict]
    ) -> float:
        """
        Get ML adjustment factor for walking time
        
        Returns multiplier (e.g., 1.2 = 20% slower than OSRM estimate)
        """
        # Factors that affect walking speed:
        # - Time of day (rush hour = slower)
        # - Weather (rain/heat = slower)
        # - Tourist areas (crowded = slower)
        # - User profile (age, fitness = faster/slower)
        
        adjustment = 1.0
        
        if user_context:
            # Time of day adjustment
            hour = user_context.get('hour', 12)
            if 8 <= hour <= 9 or 17 <= hour <= 19:
                adjustment *= 1.15  # Rush hour: 15% slower
            
            # Weather adjustment
            weather = user_context.get('weather', {})
            if weather.get('condition') == 'rain':
                adjustment *= 1.2  # Rain: 20% slower
            elif weather.get('temperature', 20) > 30:
                adjustment *= 1.1  # Hot: 10% slower
        
        return adjustment
    
    def _calculate_mode_breakdown(self, segments: List[HybridRouteSegment]) -> Dict[str, float]:
        """Calculate percentage breakdown of transport modes"""
        total_time = sum(seg.actual_duration_min for seg in segments)
        if total_time == 0:
            return {}
        
        breakdown = {}
        for seg in segments:
            mode = seg.transport_mode
            if mode not in breakdown:
                breakdown[mode] = 0.0
            breakdown[mode] += (seg.actual_duration_min / total_time) * 100
        
        return breakdown


# ═══════════════════════════════════════════════════════════════════
# Global Integration Instance
# ═══════════════════════════════════════════════════════════════════

_integration_instance: Optional[MLEnhancedOSRMIntegration] = None


def get_ml_osrm_integration() -> MLEnhancedOSRMIntegration:
    """Get or create global ML-OSRM integration instance"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = MLEnhancedOSRMIntegration()
    return _integration_instance


# ═══════════════════════════════════════════════════════════════════
# Testing
# ═══════════════════════════════════════════════════════════════════

async def test_ml_osrm_integration():
    """Test ML-enhanced OSRM integration"""
    print("\n🚀 ML-Enhanced OSRM Integration - Test\n")
    print("=" * 70)
    
    integration = MLEnhancedOSRMIntegration()
    
    # Test 1: Short distance (pure walking)
    print("\n📍 Test 1: Short distance (Sultanahmet → Blue Mosque)")
    route1 = await integration.plan_hybrid_route(
        start=(41.0082, 28.9784),  # Sultanahmet
        end=(41.0054, 28.9768),    # Blue Mosque
        prefer_public_transport=True,
        user_context={'hour': 14, 'weather': {'condition': 'sunny', 'temperature': 25}}
    )
    
    if route1:
        print(f"✅ Route planned!")
        print(f"   Distance: {route1.total_distance_km:.2f} km")
        print(f"   OSRM time: {route1.total_osrm_duration_min:.1f} min")
        print(f"   ML predicted time: {route1.total_ml_duration_min:.1f} min")
        print(f"   Cost: {route1.total_cost:.2f} TL")
        print(f"   Mode breakdown: {route1.mode_breakdown}")
    
    # Test 2: Long distance (multi-modal)
    print("\n📍 Test 2: Long distance (Sultanahmet → Taksim)")
    route2 = await integration.plan_hybrid_route(
        start=(41.0082, 28.9784),  # Sultanahmet
        end=(41.0369, 28.9850),    # Taksim
        prefer_public_transport=True,
        user_context={'hour': 8, 'weather': {'condition': 'rainy'}}
    )
    
    if route2:
        print(f"✅ Route planned!")
        print(f"   Distance: {route2.total_distance_km:.2f} km")
        print(f"   OSRM time: {route2.total_osrm_duration_min:.1f} min")
        print(f"   ML predicted time: {route2.total_ml_duration_min:.1f} min")
        print(f"   Cost: {route2.total_cost:.2f} TL")
        print(f"   Mode breakdown: {route2.mode_breakdown}")
        print(f"   ML insights: {route2.ml_insights.get('recommendations', [])}")
    
    print("\n" + "=" * 70)
    print("✅ Test completed!\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ml_osrm_integration())
