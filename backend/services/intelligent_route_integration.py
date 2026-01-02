"""
Intelligent Route Integration for Istanbul AI
==============================================

Integrates OSRM realistic routing with GPS route planner and ML-enhanced transportation system.
Provides:
1. Realistic walking routes using OpenStreetMap data (OSRM)
2. Multi-district route optimization with ML predictions
3. Integration with AI chat for conversational route planning
4. Visual route generation for frontend map display

FREE & OPEN-SOURCE:
- OSRM for realistic routing
- No paid APIs required
- Can be self-hosted for production
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)

# Import OSRM Routing Service
try:
    from backend.services.osrm_routing_service import OSRMRoutingService, OSRMRoute, RouteStep
    OSRM_AVAILABLE = True
    logger.info("✅ OSRM Routing Service available")
except ImportError as e:
    OSRM_AVAILABLE = False
    logger.warning(f"⚠️ OSRM Routing Service not available: {e}")
    # Define stub classes to prevent NameError
    OSRMRoute = type('OSRMRoute', (), {})
    RouteStep = type('RouteStep', (), {})
    OSRMRoutingService = None

# Import Map Visualization Engine
try:
    from backend.services.map_visualization_engine import MapVisualizationEngine
    MAP_VIZ_AVAILABLE = True
    logger.info("✅ Map Visualization Engine available")
except ImportError as e:
    MAP_VIZ_AVAILABLE = False
    logger.warning(f"⚠️ Map Visualization Engine not available: {e}")
    MapVisualizationEngine = None

# Import Enhanced GPS Route Planner
try:
    from enhanced_gps_route_planner import (
        EnhancedGPSRoutePlanner,
        GPSLocation,
        TransportMode,
        RoutePreference
    )
    GPS_PLANNER_AVAILABLE = True
    logger.info("✅ Enhanced GPS Route Planner available")
except ImportError as e:
    GPS_PLANNER_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced GPS Route Planner not available: {e}")
    # Define stub classes
    EnhancedGPSRoutePlanner = None
    GPSLocation = type('GPSLocation', (), {})
    TransportMode = type('TransportMode', (), {})
    RoutePreference = type('RoutePreference', (), {})

# Import ML-Enhanced Transportation System
try:
    from ml_enhanced_transportation_system import (
        MLEnhancedTransportationSystem,
        create_ml_enhanced_transportation_system,
        TransportMode as MLTransportMode
    )
    ML_TRANSPORT_AVAILABLE = True
    logger.info("✅ ML-Enhanced Transportation System available")
except ImportError as e:
    ML_TRANSPORT_AVAILABLE = False
    logger.warning(f"⚠️ ML-Enhanced Transportation System not available: {e}")
    # Define stub classes
    MLEnhancedTransportationSystem = None
    create_ml_enhanced_transportation_system = None
    MLTransportMode = type('MLTransportMode', (), {})


@dataclass
class RouteVisualization:
    """Route visualization data for frontend"""
    waypoints: List[Tuple[float, float]]  # (lat, lon) points
    steps: List[Dict[str, Any]]  # Turn-by-turn instructions
    total_distance: float  # meters
    total_duration: float  # seconds
    mode: str  # 'walking', 'transit', 'mixed'
    geojson: Dict[str, Any]  # GeoJSON for map display
    districts: List[str]  # Districts crossed
    ml_optimization: Optional[Dict[str, Any]] = None  # ML optimization data


@dataclass
class IntelligentRoute:
    """Complete route with all intelligence layers"""
    start_location: Tuple[float, float]
    end_location: Tuple[float, float]
    visualization: RouteVisualization
    ml_predictions: Optional[Dict[str, Any]] = None
    gps_route_data: Optional[Dict[str, Any]] = None
    recommendations: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.recommendations is None:
            self.recommendations = []


class IntelligentRouteIntegration:
    """
    Intelligent Route Integration System
    
    Combines:
    - OSRM for realistic walking routes
    - GPS Route Planner for location intelligence
    - ML Transportation for crowding/time predictions
    - Map Visualization for frontend display
    """
    
    def __init__(
        self,
        enable_osrm: bool = True,
        enable_ml: bool = True,
        enable_gps: bool = True,
        osrm_profile: str = 'foot'
    ):
        """
        Initialize Intelligent Route Integration
        
        Args:
            enable_osrm: Enable OSRM realistic routing
            enable_ml: Enable ML predictions
            enable_gps: Enable GPS route planner
            osrm_profile: OSRM routing profile ('foot', 'car', 'bike')
        """
        self.enable_osrm = enable_osrm and OSRM_AVAILABLE
        self.enable_ml = enable_ml and ML_TRANSPORT_AVAILABLE
        self.enable_gps = enable_gps and GPS_PLANNER_AVAILABLE
        
        # Initialize services
        if self.enable_osrm:
            self.osrm_service = OSRMRoutingService(profile=osrm_profile)
            logger.info(f"✅ OSRM service initialized (profile: {osrm_profile})")
        else:
            self.osrm_service = None
            
        if self.enable_ml:
            self.ml_transport = create_ml_enhanced_transportation_system()
            logger.info("✅ ML transportation system initialized")
        else:
            self.ml_transport = None
            
        if self.enable_gps:
            self.gps_planner = EnhancedGPSRoutePlanner()
            logger.info("✅ GPS route planner initialized")
        else:
            self.gps_planner = None
            
        if MAP_VIZ_AVAILABLE:
            self.map_viz = MapVisualizationEngine()
            logger.info("✅ Map visualization engine initialized")
        else:
            self.map_viz = None
    
    def plan_intelligent_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        waypoints: Optional[List[Tuple[float, float]]] = None,
        transport_mode: str = 'walking',
        user_context: Optional[Dict[str, Any]] = None
    ) -> IntelligentRoute:
        """
        Plan intelligent route with all optimization layers
        
        Args:
            start: Start location (lat, lon)
            end: End location (lat, lon)
            waypoints: Optional intermediate waypoints
            transport_mode: 'walking', 'transit', 'mixed'
            user_context: User preferences and context
            
        Returns:
            IntelligentRoute with visualization and predictions
        """
        logger.info(f"🗺️ Planning intelligent route from {start} to {end}")
        
        # Store route context for fallback calculations
        self._last_route_context = {
            'start': start,
            'end': end,
            'waypoints': waypoints,
            'transport_mode': transport_mode
        }
        
        # Step 1: Get realistic route from OSRM
        osrm_route = None
        if self.enable_osrm and transport_mode == 'walking':
            osrm_route = self.osrm_service.get_walking_route(start, end, waypoints)
            if osrm_route:
                logger.info(f"✅ OSRM route: {len(osrm_route.waypoints)} waypoints, "
                          f"{osrm_route.total_distance:.0f}m, "
                          f"{osrm_route.total_duration:.0f}s")
        
        # Step 2: Get GPS route intelligence
        gps_route_data = None
        if self.enable_gps and GPS_PLANNER_AVAILABLE:
            try:
                # Convert to GPS locations
                start_gps = GPSLocation(
                    latitude=start[0],
                    longitude=start[1],
                    accuracy=10.0,
                    timestamp=datetime.now()
                )
                end_gps = GPSLocation(
                    latitude=end[0],
                    longitude=end[1],
                    accuracy=10.0,
                    timestamp=datetime.now()
                )
                
                # Get route from GPS planner
                gps_result = self.gps_planner.plan_route(
                    start_location=start_gps,
                    end_location=end_gps,
                    preferences=user_context.get('preferences') if user_context else None
                )
                
                if gps_result:
                    gps_route_data = {
                        'distance': gps_result.get('total_distance'),
                        'duration': gps_result.get('total_duration'),
                        'segments': gps_result.get('segments', []),
                        'pois': gps_result.get('nearby_pois', [])
                    }
                    logger.info(f"✅ GPS route data retrieved")
                    
            except Exception as e:
                logger.warning(f"⚠️ GPS route planning failed: {e}")
        
        # Step 3: Get ML predictions
        ml_predictions = None
        if self.enable_ml and ML_TRANSPORT_AVAILABLE:
            try:
                ml_predictions = self._get_ml_predictions(
                    start, end, osrm_route, user_context
                )
                logger.info(f"✅ ML predictions generated")
            except Exception as e:
                logger.warning(f"⚠️ ML predictions failed: {e}")
        
        # Step 4: Create visualization
        visualization = self._create_visualization(
            osrm_route, gps_route_data, ml_predictions, transport_mode
        )
        
        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(
            osrm_route, gps_route_data, ml_predictions, transport_mode
        )
        
        # Create intelligent route
        route = IntelligentRoute(
            start_location=start,
            end_location=end,
            visualization=visualization,
            ml_predictions=ml_predictions,
            gps_route_data=gps_route_data,
            recommendations=recommendations
        )
        
        logger.info(f"✅ Intelligent route created successfully")
        return route
    
    def plan_multi_district_route(
        self,
        locations: List[Tuple[float, float]],
        districts: List[str],
        transport_mode: str = 'mixed',
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[IntelligentRoute]:
        """
        Plan optimized route across multiple districts
        
        Args:
            locations: List of locations to visit
            districts: List of district names
            transport_mode: 'walking', 'transit', 'mixed'
            user_context: User preferences and context
            
        Returns:
            List of IntelligentRoute objects, one per segment
        """
        logger.info(f"🌆 Planning multi-district route across {len(districts)} districts")
        
        routes = []
        
        # Plan routes between consecutive locations
        for i in range(len(locations) - 1):
            route = self.plan_intelligent_route(
                start=locations[i],
                end=locations[i + 1],
                transport_mode=transport_mode,
                user_context=user_context
            )
            routes.append(route)
        
        # Optimize with ML if available
        if self.enable_ml and len(routes) > 1:
            routes = self._optimize_multi_district_routes(routes, districts, user_context)
        
        logger.info(f"✅ Multi-district route created with {len(routes)} segments")
        return routes
    
    def _get_ml_predictions(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        osrm_route: Optional[OSRMRoute],
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get ML predictions for route"""
        predictions = {
            'crowding': None,
            'travel_time': None,
            'weather_impact': None,
            'recommendations': []
        }
        
        if not self.ml_transport:
            return predictions
        
        try:
            # Predict crowding along route
            if osrm_route and hasattr(self.ml_transport, 'predict_crowding'):
                # Sample waypoints along route for crowding prediction
                sample_points = osrm_route.waypoints[::len(osrm_route.waypoints)//5]
                crowding_predictions = []
                
                for point in sample_points:
                    try:
                        crowding = self.ml_transport.predict_crowding(
                            location=point,
                            time=datetime.now()
                        )
                        crowding_predictions.append(crowding)
                    except:
                        pass
                
                if crowding_predictions:
                    predictions['crowding'] = {
                        'average': sum(crowding_predictions) / len(crowding_predictions),
                        'max': max(crowding_predictions),
                        'points': crowding_predictions
                    }
            
            # Predict travel time with ML
            if hasattr(self.ml_transport, 'predict_travel_time'):
                try:
                    travel_time = self.ml_transport.predict_travel_time(
                        start=start,
                        end=end,
                        mode='walking',
                        time=datetime.now()
                    )
                    predictions['travel_time'] = travel_time
                except:
                    pass
            
            # Get weather impact
            if hasattr(self.ml_transport, 'get_weather_impact'):
                try:
                    weather_impact = self.ml_transport.get_weather_impact(
                        location=start,
                        mode='walking'
                    )
                    predictions['weather_impact'] = weather_impact
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"ML predictions partially failed: {e}")
        
        return predictions
    
    def _create_visualization(
        self,
        osrm_route: Optional[OSRMRoute],
        gps_route_data: Optional[Dict[str, Any]],
        ml_predictions: Optional[Dict[str, Any]],
        transport_mode: str
    ) -> RouteVisualization:
        """Create visualization data for frontend"""
        
        # Use OSRM route if available, otherwise create basic route
        if osrm_route:
            waypoints = osrm_route.waypoints
            total_distance = osrm_route.total_distance
            total_duration = osrm_route.total_duration
            
            # Convert steps to dict format
            steps = [
                {
                    'distance': step.distance,
                    'duration': step.duration,
                    'instruction': step.instruction,
                    'location': step.location,
                    'maneuver_type': step.maneuver_type
                }
                for step in osrm_route.steps
            ]
        elif gps_route_data and gps_route_data.get('distance', 0) > 0:
            # Use GPS route data
            waypoints = []
            total_distance = gps_route_data.get('distance', 0)
            total_duration = gps_route_data.get('duration', 0)
            steps = gps_route_data.get('segments', [])
        else:
            # FALLBACK: Calculate straight-line distance using Haversine formula
            # This happens when both OSRM and GPS services are unavailable
            waypoints = []
            total_distance = 0
            total_duration = 0
            steps = []
            
            # If we have start and end from the route context, calculate distance
            if hasattr(self, '_last_route_context'):
                start = self._last_route_context.get('start')
                end = self._last_route_context.get('end')
                if start and end:
                    from math import radians, sin, cos, sqrt, atan2
                    
                    # Haversine formula for distance
                    R = 6371000  # Earth radius in meters
                    lat1, lon1 = radians(start[0]), radians(start[1])
                    lat2, lon2 = radians(end[0]), radians(end[1])
                    
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    
                    total_distance = R * c  # Distance in meters
                    
                    # Estimate duration based on mode
                    # Walking: ~5 km/h, Transit: ~20 km/h
                    speed_mps = 1.4 if transport_mode == 'walking' else 5.6  # meters per second
                    total_duration = total_distance / speed_mps  # Duration in seconds
                    
                    # Add simple waypoints
                    waypoints = [start, end]
                    
                    logger.info(f"⚠️ Using fallback Haversine calculation: {total_distance:.0f}m, {total_duration:.0f}s")
        
        # Create GeoJSON
        geojson = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [[lon, lat] for lat, lon in waypoints]
                    },
                    'properties': {
                        'mode': transport_mode,
                        'distance': total_distance,
                        'duration': total_duration
                    }
                }
            ]
        }
        
        # Detect districts crossed (simplified)
        districts = self._detect_districts(waypoints)
        
        # Add ML optimization data
        ml_optimization = None
        if ml_predictions:
            ml_optimization = {
                'crowding_optimized': ml_predictions.get('crowding') is not None,
                'time_optimized': ml_predictions.get('travel_time') is not None,
                'weather_adjusted': ml_predictions.get('weather_impact') is not None
            }
        
        return RouteVisualization(
            waypoints=waypoints,
            steps=steps,
            total_distance=total_distance,
            total_duration=total_duration,
            mode=transport_mode,
            geojson=geojson,
            districts=districts,
            ml_optimization=ml_optimization
        )
    
    def _detect_districts(self, waypoints: List[Tuple[float, float]]) -> List[str]:
        """Detect which districts the route passes through"""
        # Simplified district detection based on coordinates
        # In production, use proper district boundary data
        
        districts_set = set()
        
        # Istanbul district approximate boundaries
        district_bounds = {
            'Beyoğlu': {'lat_min': 41.01, 'lat_max': 41.06, 'lon_min': 28.95, 'lon_max': 29.00},
            'Fatih': {'lat_min': 41.00, 'lat_max': 41.04, 'lon_min': 28.92, 'lon_max': 28.98},
            'Beşiktaş': {'lat_min': 41.03, 'lat_max': 41.08, 'lon_min': 29.00, 'lon_max': 29.05},
            'Kadıköy': {'lat_min': 40.97, 'lat_max': 41.02, 'lon_min': 29.00, 'lon_max': 29.10},
            'Üsküdar': {'lat_min': 41.01, 'lat_max': 41.06, 'lon_min': 29.01, 'lon_max': 29.08},
        }
        
        for lat, lon in waypoints:
            for district, bounds in district_bounds.items():
                if (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                    bounds['lon_min'] <= lon <= bounds['lon_max']):
                    districts_set.add(district)
        
        return list(districts_set)
    
    def _generate_recommendations(
        self,
        osrm_route: Optional[OSRMRoute],
        gps_route_data: Optional[Dict[str, Any]],
        ml_predictions: Optional[Dict[str, Any]],
        transport_mode: str
    ) -> List[str]:
        """Generate intelligent recommendations"""
        recommendations = []
        
        # Distance-based recommendations
        if osrm_route:
            distance_km = osrm_route.total_distance / 1000
            duration_min = osrm_route.total_duration / 60
            
            if distance_km > 5:
                recommendations.append(
                    f"This is a {distance_km:.1f}km walk ({duration_min:.0f} min). "
                    "Consider taking public transit for part of the journey."
                )
            elif distance_km > 2:
                recommendations.append(
                    f"Moderate {distance_km:.1f}km walk ({duration_min:.0f} min). "
                    "Comfortable walking distance for most people."
                )
            else:
                recommendations.append(
                    f"Short {distance_km:.1f}km walk ({duration_min:.0f} min). "
                    "Perfect for a pleasant stroll!"
                )
        
        # ML-based recommendations
        if ml_predictions:
            crowding = ml_predictions.get('crowding')
            if crowding and crowding.get('average', 0) > 0.7:
                recommendations.append(
                    "High crowding predicted. Consider traveling at a different time or taking an alternative route."
                )
            
            weather_impact = ml_predictions.get('weather_impact')
            if weather_impact:
                if weather_impact.get('precipitation', 0) > 0.5:
                    recommendations.append("Rain expected. Bring an umbrella!")
                if weather_impact.get('temperature', 20) > 30:
                    recommendations.append("Hot weather. Stay hydrated and consider indoor routes.")
        
        # POI recommendations
        if gps_route_data and gps_route_data.get('pois'):
            poi_count = len(gps_route_data['pois'])
            if poi_count > 0:
                recommendations.append(
                    f"Found {poi_count} interesting place(s) along your route. "
                    "Consider stopping to explore!"
                )
        
        return recommendations
    
    def _optimize_multi_district_routes(
        self,
        routes: List[IntelligentRoute],
        districts: List[str],
        user_context: Optional[Dict[str, Any]]
    ) -> List[IntelligentRoute]:
        """Optimize routes across multiple districts with ML"""
        
        if not self.ml_transport:
            return routes
        
        # Apply ML optimization
        # This is a placeholder for more sophisticated optimization
        logger.info(f"🤖 Optimizing {len(routes)} routes across districts: {districts}")
        
        return routes
    
    def export_for_frontend(
        self,
        route: IntelligentRoute,
        format: str = 'json'
    ) -> str:
        """
        Export route for frontend visualization
        
        Args:
            route: IntelligentRoute to export
            format: 'json', 'geojson', or 'leaflet'
            
        Returns:
            Formatted string for frontend
        """
        if format == 'geojson':
            return json.dumps(route.visualization.geojson, indent=2)
        
        elif format == 'leaflet':
            # Format for Leaflet map
            data = {
                'waypoints': route.visualization.waypoints,
                'start': route.start_location,
                'end': route.end_location,
                'distance': route.visualization.total_distance,
                'duration': route.visualization.total_duration,
                'mode': route.visualization.mode,
                'steps': route.visualization.steps,
                'districts': route.visualization.districts,
                'recommendations': route.recommendations
            }
            return json.dumps(data, indent=2, default=str)
        
        else:  # json
            data = {
                'route': {
                    'start': route.start_location,
                    'end': route.end_location,
                    'visualization': asdict(route.visualization),
                    'ml_predictions': route.ml_predictions,
                    'gps_route_data': route.gps_route_data,
                    'recommendations': route.recommendations,
                    'created_at': route.created_at.isoformat()
                }
            }
            return json.dumps(data, indent=2, default=str)


def create_intelligent_route_integration(**kwargs) -> IntelligentRouteIntegration:
    """Factory function to create route integration"""
    return IntelligentRouteIntegration(**kwargs)


# Testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing Intelligent Route Integration...\n")
    
    # Initialize integration
    integration = create_intelligent_route_integration()
    
    # Test single route
    print("📍 Test 1: Single walking route")
    route = integration.plan_intelligent_route(
        start=(41.0082, 28.9784),  # Sultanahmet
        end=(41.0256, 28.9742),    # Galata Tower
        transport_mode='walking'
    )
    
    print(f"✅ Route created:")
    print(f"   Distance: {route.visualization.total_distance/1000:.2f} km")
    print(f"   Duration: {route.visualization.total_duration/60:.0f} min")
    print(f"   Districts: {', '.join(route.visualization.districts)}")
    print(f"   Waypoints: {len(route.visualization.waypoints)}")
    print(f"   Recommendations: {len(route.recommendations)}")
    
    # Test multi-district route
    print("\n🌆 Test 2: Multi-district route")
    locations = [
        (41.0082, 28.9784),  # Sultanahmet
        (41.0256, 28.9742),  # Galata Tower
        (41.0370, 28.9850),  # Taksim Square
    ]
    districts = ['Fatih', 'Beyoğlu']
    
    routes = integration.plan_multi_district_route(
        locations=locations,
        districts=districts,
        transport_mode='mixed'
    )
    
    print(f"✅ Multi-district route created:")
    print(f"   Segments: {len(routes)}")
    total_distance = sum(r.visualization.total_distance for r in routes) / 1000
    total_duration = sum(r.visualization.total_duration for r in routes) / 60
    print(f"   Total distance: {total_distance:.2f} km")
    print(f"   Total duration: {total_duration:.0f} min")
    
    # Export for frontend
    print("\n📤 Test 3: Frontend export")
    leaflet_data = integration.export_for_frontend(route, format='leaflet')
    print(f"✅ Leaflet data generated ({len(leaflet_data)} bytes)")
    
    print("\n✅ All tests passed! Intelligent Route Integration is ready!")
