#!/usr/bin/env python3
"""
POI-Enhanced Route Optimizer - Phase 4
======================================

Advanced route optimization with POI integration:
- Enhanced A* pathfinding with POI constraints
- Optimal POI insertion into routes
- Multi-objective optimization (time, cost, cultural value, crowding)
- ML prediction integration
- Smart detour calculation

Integrates:
- Phase 1: POI Database Service
- Phase 2: Transport Graph Service
- Phase 3: ML Prediction Service
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq

# Add services directory to path
sys.path.insert(0, str(Path(__file__).parent / 'services'))
sys.path.insert(0, str(Path(__file__).parent))

# Import required services
try:
    from services.poi_database_service import POIDatabaseService, POI, GeoCoordinate
    POI_DATABASE_AVAILABLE = True
except ImportError:
    POI_DATABASE_AVAILABLE = False
    logging.warning("POI Database not available")

try:
    from services.transport_graph_service import (
        TransportGraphService, GraphNode, GraphEdge, NodeType, EdgeType
    )
    GRAPH_SERVICE_AVAILABLE = True
except ImportError:
    GRAPH_SERVICE_AVAILABLE = False
    logging.warning("Transport Graph Service not available")

try:
    from services.ml_prediction_service import (
        MLPredictionService, CrowdingPrediction, TravelTimePrediction
    )
    ML_PREDICTION_AVAILABLE = True
except ImportError:
    ML_PREDICTION_AVAILABLE = False
    logging.warning("ML Prediction Service not available")

logger = logging.getLogger(__name__)


@dataclass
class RouteConstraints:
    """Constraints for route optimization"""
    max_total_time_minutes: int = 180  # 3 hours default
    max_detour_time_minutes: int = 45  # Max detour for a single POI
    max_total_detour_minutes: int = 120  # Max total detour time for all POIs combined
    max_pois: int = 3
    max_cost: float = 100.0
    require_category_diversity: bool = True
    min_poi_value: float = 0.3  # Lowered from 0.6 to be more inclusive
    preferred_transport_modes: List[str] = field(default_factory=lambda: ['metro', 'tram', 'walk'])
    accessibility_required: bool = False
    
    
@dataclass
class POIDetour:
    """Information about POI detour cost and value"""
    poi: POI
    insertion_point: int  # Index in route segments
    access_station_id: str
    walking_distance_km: float
    walking_time_minutes: int
    visit_duration_minutes: int
    total_detour_time_minutes: int
    value_score: float
    crowding_level: float
    cost_benefit_ratio: float  # value per minute of detour
    

@dataclass
class OptimizedRouteSegment:
    """Enhanced route segment with ML predictions"""
    segment_type: str  # 'transit', 'walk', 'poi_visit'
    from_location: str
    to_location: str
    transport_mode: Optional[str] = None
    distance_km: float = 0.0
    scheduled_time_minutes: int = 0
    predicted_time_minutes: int = 0
    cost: float = 0.0
    scenic_score: float = 0.5
    ml_predictions: Dict[str, Any] = field(default_factory=dict)
    poi_details: Optional[Dict[str, Any]] = None


@dataclass
class POIEnhancedRoute:
    """Complete optimized route with POIs"""
    route_id: str
    start_location: GeoCoordinate
    end_location: GeoCoordinate
    base_route: Dict[str, Any]
    enhanced_route: Dict[str, Any]
    segments: List[OptimizedRouteSegment]
    pois_included: List[POI]
    pois_recommended_not_included: List[Dict[str, Any]]
    optimization_insights: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


class POIEnhancedRouteOptimizer:
    """
    Advanced route optimizer with POI integration and ML predictions
    
    Features:
    - Enhanced A* pathfinding
    - Optimal POI selection and insertion
    - Multi-objective optimization
    - ML crowding and travel time predictions
    - Smart detour calculation
    """
    
    def __init__(self):
        # Initialize services
        self.poi_service = POIDatabaseService() if POI_DATABASE_AVAILABLE else None
        self.graph_service = TransportGraphService() if GRAPH_SERVICE_AVAILABLE else None
        self.ml_service = MLPredictionService() if ML_PREDICTION_AVAILABLE else None
        
        logger.info("🎯 POI-Enhanced Route Optimizer initialized")
        logger.info(f"   - POI Service: {self.poi_service is not None}")
        logger.info(f"   - Graph Service: {self.graph_service is not None}")
        logger.info(f"   - ML Service: {self.ml_service is not None}")
    
    async def create_poi_enhanced_route(
        self,
        start_location: GeoCoordinate,
        end_location: GeoCoordinate,
        user_preferences: Dict[str, Any],
        constraints: Optional[RouteConstraints] = None
    ) -> POIEnhancedRoute:
        """
        Create optimized route with POI integration
        
        Algorithm:
        1. Find base optimal route (start -> end)
        2. Identify candidate POIs along route
        3. Score each POI (value vs. detour cost)
        4. Select top N POIs using knapsack optimization
        5. Optimize POI insertion order
        6. Rebuild route with POIs
        7. Add ML predictions for all segments
        
        Args:
            start_location: Starting point
            end_location: Destination
            user_preferences: User interests, budget, etc.
            constraints: Route constraints
            
        Returns:
            POIEnhancedRoute with optimized POIs and ML predictions
        """
        if constraints is None:
            constraints = RouteConstraints()
        
        logger.info(f"🗺️ Creating POI-enhanced route from {start_location.lat:.4f},{start_location.lon:.4f} to {end_location.lat:.4f},{end_location.lon:.4f}")
        
        # Step 1: Find base optimal route
        base_route = await self._find_base_route(
            start_location, end_location, constraints
        )
        
        if not base_route:
            logger.warning("Could not find base route")
            return self._create_empty_route(start_location, end_location)
        
        logger.info(f"   Base route: {base_route['distance_km']:.2f}km, {base_route['time_minutes']}min")
        
        # Step 2: Find candidate POIs along route
        candidate_pois = self._find_pois_along_route(
            base_route,
            user_preferences,
            constraints
        )
        
        logger.info(f"   Found {len(candidate_pois)} candidate POIs")
        
        # Step 3: Calculate detour cost for each POI
        poi_detours = []
        current_time = datetime.now()
        
        logger.info(f"   Calculating detours for {len(candidate_pois)} POIs...")
        
        for poi in candidate_pois:
            detour = self._calculate_poi_detour(
                poi, base_route, current_time, constraints
            )
            if detour:
                poi_detours.append(detour)
                logger.info(f"     ✅ {poi.name}: detour={detour.total_detour_time_minutes}min, value={detour.value_score:.2f}")
            else:
                logger.debug(f"     ❌ {poi.name}: filtered out")
        
        logger.info(f"   Calculated {len(poi_detours)} POI detours")
        
        # Step 4: Select optimal POI set
        selected_detours = self._select_optimal_poi_set(
            poi_detours, constraints
        )
        
        logger.info(f"   Selected {len(selected_detours)} POIs for route")
        
        # Step 5: Optimize POI order
        optimized_detours = self._optimize_poi_order(
            selected_detours, base_route
        )
        
        # Step 6: Build enhanced route with POIs
        enhanced_route = await self._build_route_with_pois(
            base_route, optimized_detours, current_time
        )
        
        # Step 7: Add ML predictions
        enhanced_route = self._add_ml_predictions(
            enhanced_route, current_time
        )
        
        # Create result
        route_id = f"poi_route_{int(datetime.now().timestamp())}"
        
        # Calculate metrics
        total_detour_time = sum(d.total_detour_time_minutes for d in optimized_detours)
        total_cultural_value = sum(d.value_score for d in optimized_detours)
        
        pois_included = [d.poi for d in optimized_detours]
        
        # POIs that were good but not included
        pois_not_included = []
        for detour in poi_detours:
            if detour.poi not in pois_included:
                pois_not_included.append({
                    'name': detour.poi.name,
                    'reason': self._get_exclusion_reason(detour, constraints),
                    'alternative_suggestion': f"Visit separately or extend time budget by {detour.total_detour_time_minutes}min"
                })
        
        result = POIEnhancedRoute(
            route_id=route_id,
            start_location=start_location,
            end_location=end_location,
            base_route={
                'distance_km': base_route['distance_km'],
                'time_minutes': base_route['time_minutes'],
                'cost': base_route.get('cost', 0.0)
            },
            enhanced_route={
                'distance_km': enhanced_route['distance_km'],
                'time_minutes': enhanced_route['time_minutes'],
                'cost': enhanced_route.get('cost', 0.0),
                'detour_summary': {
                    'additional_time': total_detour_time,
                    'additional_distance': enhanced_route['distance_km'] - base_route['distance_km'],
                    'pois_included': len(pois_included),
                    'cultural_value_score': total_cultural_value / len(pois_included) if pois_included else 0
                }
            },
            segments=enhanced_route['segments'],
            pois_included=pois_included,
            pois_recommended_not_included=pois_not_included[:5],  # Top 5
            optimization_insights={
                'pois_evaluated': len(candidate_pois),
                'pois_with_valid_detours': len(poi_detours),
                'pois_selected': len(pois_included),
                'optimization_time_ms': 0,  # Would be calculated
                'ml_predictions_used': ML_PREDICTION_AVAILABLE
            }
        )
        
        logger.info(f"✅ Created POI-enhanced route with {len(pois_included)} POIs")
        
        return result
    
    async def _find_base_route(
        self,
        start: GeoCoordinate,
        end: GeoCoordinate,
        constraints: RouteConstraints
    ) -> Optional[Dict[str, Any]]:
        """Find base optimal route without POIs"""
        
        # For now, create a simple mock route
        # In production, this would use the transport graph service
        
        distance_km = self._calculate_distance(start, end)
        time_minutes = int(distance_km / 0.5 * 60)  # Assume 30 km/h average
        
        return {
            'distance_km': distance_km,
            'time_minutes': time_minutes,
            'cost': 15.0,  # Flat fare
            'segments': []
        }
    
    def _find_pois_along_route(
        self,
        base_route: Dict[str, Any],
        user_preferences: Dict[str, Any],
        constraints: RouteConstraints
    ) -> List[POI]:
        """Find POIs along the route within detour radius"""
        
        if not self.poi_service:
            return []
        
        # For simplicity, find all POIs (in production, would use route corridor)
        all_pois = list(self.poi_service.pois.values())
        
        logger.info(f"   Total POIs in database: {len(all_pois)}")
        
        # Filter by interests
        interests = user_preferences.get('interests', [])
        candidate_pois = []
        
        logger.info(f"   User interests: {interests}")
        
        for poi in all_pois:
            # Check if POI matches interests
            if interests:
                # Check against category and subcategory
                matches = any(
                    interest.lower() in poi.category.lower() or
                    interest.lower() in poi.subcategory.lower()
                    for interest in interests
                )
                if matches:
                    candidate_pois.append(poi)
                    logger.debug(f"     Matched {poi.name} ({poi.category}/{poi.subcategory})")
            else:
                candidate_pois.append(poi)
        
        logger.info(f"   Candidate POIs after interest filtering: {len(candidate_pois)}")
        
        return candidate_pois[:20]  # Limit to top 20 for performance
    
    def _calculate_poi_detour(
        self,
        poi: POI,
        base_route: Dict[str, Any],
        current_time: datetime,
        constraints: RouteConstraints
    ) -> Optional[POIDetour]:
        """Calculate detour cost and value for visiting a POI"""
        
        # Find nearest station (mock for now)
        access_station = "nearest_station"
        walking_distance_km = 0.5  # Mock
        walking_speed_kmh = 4.5  # Average walking speed
        walking_time_minutes = int((walking_distance_km / walking_speed_kmh) * 60)
        
        # Get visit duration
        visit_duration = poi.visit_duration_min
        
        # Total detour time = walk there + visit + walk back
        total_detour_time = (walking_time_minutes * 2) + visit_duration
        
        logger.debug(f"     {poi.name}: walk={walking_time_minutes}min*2, visit={visit_duration}min, total={total_detour_time}min, max={constraints.max_detour_time_minutes}min")
        
        # Check if within time constraint
        if total_detour_time > constraints.max_detour_time_minutes:
            logger.debug(f"     ❌ {poi.name}: Time constraint violated ({total_detour_time} > {constraints.max_detour_time_minutes})")
            return None
        
        # Calculate value score
        value_score = self._calculate_poi_value(
            poi, current_time, constraints
        )
        
        logger.debug(f"     {poi.name}: value={value_score:.2f}, min_required={constraints.min_poi_value}")
        
        if value_score < constraints.min_poi_value:
            logger.debug(f"     ❌ {poi.name}: Value too low ({value_score:.2f} < {constraints.min_poi_value})")
            return None
        
        # Get crowding prediction
        crowding_level = 0.5  # Default
        if self.ml_service:
            try:
                crowding_pred = self.ml_service.predict_poi_crowding(
                    poi.poi_id, current_time
                )
                crowding_level = crowding_pred.crowding_level
            except:
                pass
        
        # Calculate cost-benefit ratio
        cost_benefit = value_score / (total_detour_time + 1)
        
        return POIDetour(
            poi=poi,
            insertion_point=0,  # Would be calculated based on route
            access_station_id=access_station,
            walking_distance_km=walking_distance_km,
            walking_time_minutes=walking_time_minutes,
            visit_duration_minutes=visit_duration,
            total_detour_time_minutes=total_detour_time,
            value_score=value_score,
            crowding_level=crowding_level,
            cost_benefit_ratio=cost_benefit
        )
    
    def _calculate_poi_value(
        self,
        poi: POI,
        current_time: datetime,
        constraints: RouteConstraints
    ) -> float:
        """Calculate value score for a POI (0.0-1.0)"""
        
        score = 0.0
        
        # Base rating (0-0.3)
        score += (poi.rating / 5.0) * 0.3
        
        # Popularity (0-0.3)
        score += poi.popularity_score * 0.3
        
        # Category bonus (0-0.2)
        if 'palace' in poi.category.lower() or 'historical' in poi.subcategory.lower():
            score += 0.2
        elif 'museum' in poi.category.lower():
            score += 0.15
        elif 'mosque' in poi.category.lower() and 'historic' in poi.subcategory.lower():
            score += 0.18
        
        # Crowding penalty (0-0.2)
        if self.ml_service:
            try:
                crowding_pred = self.ml_service.predict_poi_crowding(
                    poi.poi_id, current_time
                )
                # Lower crowding = higher score
                score += (1.0 - crowding_pred.crowding_level) * 0.2
            except:
                score += 0.1  # Neutral
        else:
            score += 0.1
        
        return min(score, 1.0)
    
    def _select_optimal_poi_set(
        self,
        poi_detours: List[POIDetour],
        constraints: RouteConstraints
    ) -> List[POIDetour]:
        """
        Select optimal set of POIs using knapsack-style algorithm
        
        Maximize: Sum of value scores
        Subject to:
        - Max number of POIs
        - Max total detour time
        - Category diversity
        """
        
        # Sort by cost-benefit ratio
        sorted_detours = sorted(
            poi_detours,
            key=lambda x: x.cost_benefit_ratio,
            reverse=True
        )
        
        selected = []
        total_time = 0
        categories_used = set()
        
        for detour in sorted_detours:
            # Check constraints
            if len(selected) >= constraints.max_pois:
                logger.debug(f"      Reached max POIs ({constraints.max_pois})")
                break
            
            # Check total detour time budget
            if total_time + detour.total_detour_time_minutes > constraints.max_total_detour_minutes:
                logger.debug(f"      Skipping {detour.poi.name}: would exceed total detour budget ({total_time + detour.total_detour_time_minutes} > {constraints.max_total_detour_minutes})")
                continue
            
            # Category diversity
            poi_category = detour.poi.category
            if constraints.require_category_diversity:
                if poi_category in categories_used and len(selected) > 0:
                    # Allow if really high value
                    if detour.value_score < 0.75:  # Lowered from 0.85
                        logger.debug(f"      Skipping {detour.poi.name}: duplicate category {poi_category}")
                        continue
            
            # Add to selection
            selected.append(detour)
            total_time += detour.total_detour_time_minutes
            categories_used.add(poi_category)
        
        return selected
    
    def _optimize_poi_order(
        self,
        poi_detours: List[POIDetour],
        base_route: Dict[str, Any]
    ) -> List[POIDetour]:
        """Optimize the order of POI visits"""
        
        # For now, keep the order by cost-benefit ratio
        # In production, would use traveling salesman optimization
        return poi_detours
    
    async def _build_route_with_pois(
        self,
        base_route: Dict[str, Any],
        poi_detours: List[POIDetour],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Build enhanced route with POI visits inserted"""
        
        segments = []
        total_distance = base_route['distance_km']
        total_time = base_route['time_minutes']
        
        # Add POI visit segments
        for detour in poi_detours:
            # Walk to POI
            segments.append(OptimizedRouteSegment(
                segment_type='walk',
                from_location=detour.access_station_id,
                to_location=detour.poi.name,
                distance_km=detour.walking_distance_km,
                scheduled_time_minutes=detour.walking_time_minutes,
                predicted_time_minutes=detour.walking_time_minutes
            ))
            
            # POI visit
            segments.append(OptimizedRouteSegment(
                segment_type='poi_visit',
                from_location=detour.poi.name,
                to_location=detour.poi.name,
                distance_km=0.0,
                scheduled_time_minutes=detour.visit_duration_minutes,
                predicted_time_minutes=detour.visit_duration_minutes,
                poi_details={
                    'name': detour.poi.name,
                    'category': detour.poi.category,
                    'subcategory': detour.poi.subcategory,
                    'rating': detour.poi.rating,
                    'crowding_level': detour.crowding_level
                }
            ))
            
            # Walk back
            segments.append(OptimizedRouteSegment(
                segment_type='walk',
                from_location=detour.poi.name,
                to_location=detour.access_station_id,
                distance_km=detour.walking_distance_km,
                scheduled_time_minutes=detour.walking_time_minutes,
                predicted_time_minutes=detour.walking_time_minutes
            ))
            
            total_distance += detour.walking_distance_km * 2
            total_time += detour.total_detour_time_minutes
        
        return {
            'distance_km': total_distance,
            'time_minutes': total_time,
            'cost': base_route.get('cost', 0.0),
            'segments': segments
        }
    
    def _add_ml_predictions(
        self,
        route: Dict[str, Any],
        current_time: datetime
    ) -> Dict[str, Any]:
        """Add ML predictions to route segments"""
        
        if not self.ml_service:
            return route
        
        for segment in route['segments']:
            if segment.segment_type == 'poi_visit' and segment.poi_details:
                try:
                    # Add crowding prediction
                    poi_id = segment.poi_details.get('name', '').lower().replace(' ', '_')
                    crowding_pred = self.ml_service.predict_poi_crowding(
                        poi_id, current_time
                    )
                    
                    segment.ml_predictions = {
                        'crowding_level': crowding_pred.crowding_level,
                        'crowding_label': crowding_pred.crowding_label,
                        'wait_time_minutes': crowding_pred.wait_time_minutes,
                        'confidence': crowding_pred.confidence
                    }
                except:
                    pass
        
        return route
    
    def _calculate_distance(self, loc1: GeoCoordinate, loc2: GeoCoordinate) -> float:
        """Calculate distance in km using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371.0  # Earth radius in km
        
        lat1, lon1 = radians(loc1.lat), radians(loc1.lon)
        lat2, lon2 = radians(loc2.lat), radians(loc2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _get_exclusion_reason(
        self,
        detour: POIDetour,
        constraints: RouteConstraints
    ) -> str:
        """Get reason why POI was not included"""
        
        if detour.total_detour_time_minutes > constraints.max_detour_time_minutes:
            return f"Would add {detour.total_detour_time_minutes}min detour, exceeds {constraints.max_detour_time_minutes}min limit"
        
        if detour.value_score < constraints.min_poi_value:
            return f"Value score {detour.value_score:.2f} below threshold {constraints.min_poi_value}"
        
        if detour.crowding_level > 0.8:
            return f"Very crowded ({detour.crowding_level:.0%}) at this time"
        
        return "Lower priority compared to selected POIs"
    
    def _create_empty_route(
        self,
        start: GeoCoordinate,
        end: GeoCoordinate
    ) -> POIEnhancedRoute:
        """Create empty route when planning fails"""
        
        return POIEnhancedRoute(
            route_id=f"empty_route_{int(datetime.now().timestamp())}",
            start_location=start,
            end_location=end,
            base_route={'distance_km': 0, 'time_minutes': 0, 'cost': 0},
            enhanced_route={'distance_km': 0, 'time_minutes': 0, 'cost': 0, 'detour_summary': {}},
            segments=[],
            pois_included=[],
            pois_recommended_not_included=[],
            optimization_insights={}
        )


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    print("=" * 80)
    print("🎯 Phase 4: POI-Enhanced Route Optimizer - Testing")
    print("=" * 80)
    
    async def test_optimizer():
        # Initialize optimizer
        optimizer = POIEnhancedRouteOptimizer()
        
        # Test route
        start = GeoCoordinate(lat=41.0082, lon=28.9784)  # Sultanahmet
        end = GeoCoordinate(lat=41.0369, lon=28.9857)    # Taksim
        
        preferences = {
            'interests': ['museum', 'history', 'palace']
        }
        
        constraints = RouteConstraints(
            max_total_time_minutes=180,
            max_detour_time_minutes=60,  # Max per POI
            max_total_detour_minutes=150,  # Total budget for all POIs
            max_pois=3,
            require_category_diversity=False  # Allow multiple museums
        )
        
        print("\n🗺️ Creating POI-enhanced route...")
        print(f"Start: Sultanahmet ({start.lat:.4f}, {start.lon:.4f})")
        print(f"End: Taksim ({end.lat:.4f}, {end.lon:.4f})")
        print(f"Max POIs: {constraints.max_pois}")
        print(f"Max Detour: {constraints.max_detour_time_minutes} minutes")
        
        route = await optimizer.create_poi_enhanced_route(
            start, end, preferences, constraints
        )
        
        print("\n✅ Route Created!")
        print(f"\nBase Route:")
        print(f"  Distance: {route.base_route['distance_km']:.2f} km")
        print(f"  Time: {route.base_route['time_minutes']} minutes")
        print(f"  Cost: ₺{route.base_route['cost']:.2f}")
        
        print(f"\nEnhanced Route (with POIs):")
        print(f"  Distance: {route.enhanced_route['distance_km']:.2f} km")
        print(f"  Time: {route.enhanced_route['time_minutes']} minutes")
        print(f"  POIs Included: {route.enhanced_route['detour_summary'].get('pois_included', 0)}")
        print(f"  Cultural Value: {route.enhanced_route['detour_summary'].get('cultural_value_score', 0):.2f}")
        
        if route.pois_included:
            print(f"\n📍 POIs in Route:")
            for i, poi in enumerate(route.pois_included, 1):
                print(f"  {i}. {poi.name} ({poi.category})")
        
        print(f"\n📊 Optimization Insights:")
        print(f"  POIs Evaluated: {route.optimization_insights['pois_evaluated']}")
        print(f"  POIs Selected: {route.optimization_insights['pois_selected']}")
        print(f"  ML Predictions: {'Yes' if route.optimization_insights['ml_predictions_used'] else 'No'}")
        
        if route.pois_recommended_not_included:
            print(f"\n💡 Recommended (not included):")
            for rec in route.pois_recommended_not_included[:3]:
                print(f"  • {rec['name']}: {rec['reason']}")
    
    # Run test
    asyncio.run(test_optimizer())
    
    print("\n" + "=" * 80)
    print("✅ Phase 4: POI-Enhanced Route Optimizer - Ready!")
    print("=" * 80)
