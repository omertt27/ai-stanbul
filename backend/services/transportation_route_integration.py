"""
Transportation Route Integration - Moovit-Style Multi-Route System
==================================================================

Integrates route_optimizer.py with transportation_rag_system.py and
transportation_directions_service.py to provide Moovit-style multi-route
alternatives with comfort scoring and smart ranking.

Features:
✅ Multi-route generation (fastest, best, least transfers, etc.)
✅ Comfort scoring with Istanbul-specific transfer quality
✅ Peak hour awareness and crowding predictions
✅ LLM-powered route summaries
✅ Visual route comparison data for frontend

Author: AI Istanbul Team
Date: January 2025
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Import route optimizer
try:
    from services.route_optimizer import (
        get_route_optimizer,
        RoutePreference,
        RouteOption
    )
    ROUTE_OPTIMIZER_AVAILABLE = True
    logger.info("✅ Route optimizer imported successfully")
except ImportError as e:
    ROUTE_OPTIMIZER_AVAILABLE = False
    logger.warning(f"⚠️ Route optimizer not available: {e}")

# Import transportation services
try:
    from services.transportation_rag_system import get_transportation_rag
    from services.transportation_directions_service import TransportationDirectionsService
    RAG_AVAILABLE = True
    logger.info("✅ Transportation services imported successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    logger.warning(f"⚠️ Transportation services not available: {e}")


class TransportationRouteIntegration:
    """
    Integrated transportation routing with multi-route optimization.
    
    Combines RAG knowledge base, directions service, and route optimizer
    to provide Moovit-style route alternatives.
    """
    
    def __init__(self):
        """Initialize the integration layer"""
        self.rag = None
        self.directions_service = None
        self.route_optimizer = None
        
        if RAG_AVAILABLE:
            try:
                self.rag = get_transportation_rag()
                self.directions_service = TransportationDirectionsService()
                logger.info("✅ Transportation services initialized")
            except Exception as e:
                logger.error(f"Failed to initialize transportation services: {e}")
        
        if ROUTE_OPTIMIZER_AVAILABLE:
            try:
                self.route_optimizer = get_route_optimizer()
                logger.info("✅ Route optimizer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize route optimizer: {e}")
    
    def get_route_alternatives(
        self,
        origin: str,
        destination: str,
        origin_gps: Optional[Dict[str, float]] = None,
        destination_gps: Optional[Dict[str, float]] = None,
        departure_time: datetime = None,
        num_alternatives: int = 3,
        generate_llm_summaries: bool = False,
        user_language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Get multiple route alternatives with Moovit-style optimization.
        
        Args:
            origin: Origin location name
            destination: Destination location name
            origin_gps: Optional origin GPS {"lat": float, "lon": float}
            destination_gps: Optional destination GPS {"lat": float, "lon": float}
            departure_time: Departure time for time-based routing
            num_alternatives: Number of route alternatives to generate
            generate_llm_summaries: Whether to generate LLM summaries
            user_language: User's preferred language
            
        Returns:
            Dict containing:
            - primary_route: Main recommended route (TransitRoute)
            - alternatives: List of alternative routes with scoring (RouteOption)
            - route_comparison: Comparison data for frontend visualization
            - map_data: Map visualization data
        """
        if departure_time is None:
            departure_time = datetime.now()
        
        logger.info(f"🎯 Getting route alternatives: {origin} → {destination}")
        
        # Step 1: Get primary route from RAG system
        if not self.rag:
            logger.error("RAG system not available")
            return self._error_response("Transportation system not available")
        
        primary_route = self.rag.find_route(
            origin=origin,
            destination=destination,
            origin_gps=origin_gps,
            destination_gps=destination_gps
        )
        
        if not primary_route:
            logger.warning(f"No route found: {origin} → {destination}")
            return self._error_response(f"No route found from {origin} to {destination}")
        
        # Step 2: Get detailed directions with steps
        routes = []
        
        if self.directions_service and origin_gps and destination_gps:
            try:
                # Get directions with GPS coordinates
                detailed_route = self.directions_service.get_directions(
                    start=(origin_gps['lat'], origin_gps['lon']),
                    end=(destination_gps['lat'], destination_gps['lon']),
                    start_name=origin,
                    end_name=destination
                )
                
                if detailed_route:
                    routes.append(detailed_route)
                    logger.info(f"✅ Got detailed route: {detailed_route.total_duration} min")
            except Exception as e:
                logger.warning(f"Failed to get detailed route: {e}")
        
        # If we don't have routes yet, convert RAG route to TransportRoute
        if not routes:
            routes.append(self._convert_rag_to_transport_route(primary_route))
        
        # Step 3: Optimize routes with different preferences
        alternatives = []
        
        if self.route_optimizer and ROUTE_OPTIMIZER_AVAILABLE:
            try:
                optimized = self.route_optimizer.optimize_routes(
                    routes=routes,
                    preferences=[
                        RoutePreference.BEST,
                        RoutePreference.FASTEST,
                        RoutePreference.LEAST_TRANSFERS,
                        RoutePreference.LEAST_WALKING,
                        RoutePreference.MOST_COMFORTABLE
                    ],
                    departure_time=departure_time,
                    generate_llm_summaries=generate_llm_summaries,
                    user_language=user_language
                )
                
                alternatives = optimized[:num_alternatives]
                logger.info(f"✅ Generated {len(alternatives)} optimized alternatives")
                
            except Exception as e:
                logger.error(f"Failed to optimize routes: {e}")
        
        # Step 4: Generate route comparison data
        route_comparison = self._generate_route_comparison(alternatives)
        
        # Step 5: Get map visualization data
        map_data = self._get_map_data(primary_route)
        
        return {
            'success': True,
            'primary_route': self._serialize_transit_route(primary_route),
            'alternatives': [self._serialize_route_option(opt) for opt in alternatives],
            'route_comparison': route_comparison,
            'map_data': map_data,
            'metadata': {
                'origin': origin,
                'destination': destination,
                'departure_time': departure_time.isoformat(),
                'num_alternatives': len(alternatives),
                'has_llm_summaries': generate_llm_summaries
            }
        }
    
    def _convert_rag_to_transport_route(self, rag_route):
        """Convert TransitRoute (RAG) to TransportRoute (directions service)"""
        from services.transportation_directions_service import TransportRoute, TransportStep
        
        steps = []
        for step_dict in rag_route.steps:
            step = TransportStep(
                mode=step_dict.get('type', 'walk'),
                instruction=step_dict.get('instruction', ''),
                distance=step_dict.get('distance', 0) * 1000,  # km to meters
                duration=step_dict.get('time', 0),
                start_location=(0, 0),  # Would need GPS lookup
                end_location=(0, 0),
                line_name=step_dict.get('line'),
                stops_count=step_dict.get('stops')
            )
            steps.append(step)
        
        return TransportRoute(
            steps=steps,
            total_distance=rag_route.total_distance * 1000,  # km to meters
            total_duration=rag_route.total_time,
            summary=f"{rag_route.origin} to {rag_route.destination}",
            modes_used=rag_route.lines_used,
            estimated_cost=0.0  # Would calculate from steps
        )
    
    def _generate_route_comparison(self, alternatives: List[RouteOption]) -> Dict[str, Any]:
        """Generate comparison data for frontend visualization"""
        if not alternatives:
            return {}
        
        comparison = {
            'metrics': {
                'duration': [],
                'transfers': [],
                'walking': [],
                'comfort': []
            },
            'preferences': []
        }
        
        for opt in alternatives:
            comparison['metrics']['duration'].append({
                'preference': opt.preference.value,
                'value': opt.duration_minutes,
                'label': f"{opt.duration_minutes} min"
            })
            
            comparison['metrics']['transfers'].append({
                'preference': opt.preference.value,
                'value': opt.num_transfers,
                'label': f"{opt.num_transfers} transfer{'s' if opt.num_transfers != 1 else ''}"
            })
            
            comparison['metrics']['walking'].append({
                'preference': opt.preference.value,
                'value': opt.walking_meters,
                'label': f"{int(opt.walking_meters)}m"
            })
            
            comparison['metrics']['comfort'].append({
                'preference': opt.preference.value,
                'value': opt.comfort_score.overall_comfort,
                'label': f"{int(opt.comfort_score.overall_comfort)}/100"
            })
            
            comparison['preferences'].append({
                'name': opt.preference.value,
                'score': opt.overall_score,
                'highlights': opt.highlights,
                'warnings': opt.warnings
            })
        
        return comparison
    
    def _get_map_data(self, route) -> Optional[Dict[str, Any]]:
        """Get map visualization data for a route"""
        if not self.rag:
            return None
        
        try:
            map_data = self.rag.get_map_data_for_last_route()
            return map_data
        except Exception as e:
            logger.warning(f"Failed to get map data: {e}")
            return None
    
    def _serialize_transit_route(self, route) -> Dict[str, Any]:
        """Serialize TransitRoute for JSON response"""
        return {
            'origin': route.origin,
            'destination': route.destination,
            'total_time': route.total_time,
            'total_distance': route.total_distance,
            'transfers': route.transfers,
            'lines_used': route.lines_used,
            'steps': route.steps,
            'time_confidence': route.time_confidence
        }
    
    def _serialize_route_option(self, option: RouteOption) -> Dict[str, Any]:
        """Serialize RouteOption for JSON response"""
        return {
            'preference': option.preference.value,
            'duration_minutes': option.duration_minutes,
            'walking_meters': option.walking_meters,
            'num_transfers': option.num_transfers,
            'cost_tl': option.cost_tl,
            'comfort_score': {
                'mode_comfort': option.comfort_score.mode_comfort,
                'transfer_quality': option.comfort_score.transfer_quality,
                'crowding_penalty': option.comfort_score.crowding_penalty,
                'walking_comfort': option.comfort_score.walking_comfort,
                'overall_comfort': option.comfort_score.overall_comfort,
                'highlights': option.comfort_score.highlights,
                'warnings': option.comfort_score.warnings
            },
            'overall_score': option.overall_score,
            'highlights': option.highlights,
            'warnings': option.warnings,
            'llm_summary': option.llm_summary,
            'route': self._serialize_transport_route(option.route)
        }
    
    def _serialize_transport_route(self, route) -> Dict[str, Any]:
        """Serialize TransportRoute for JSON response"""
        return {
            'total_duration': route.total_duration,
            'total_distance': route.total_distance,
            'summary': route.summary,
            'modes_used': route.modes_used if hasattr(route, 'modes_used') else [],
            'estimated_cost': route.estimated_cost if hasattr(route, 'estimated_cost') else 0.0,
            'steps': [self._serialize_transport_step(s) for s in route.steps]
        }
    
    def _serialize_transport_step(self, step) -> Dict[str, Any]:
        """Serialize TransportStep for JSON response"""
        return {
            'mode': step.mode,
            'instruction': step.instruction,
            'distance': step.distance,
            'duration': step.duration,
            'line_name': step.line_name,
            'stops_count': step.stops_count
        }
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'success': False,
            'error': message,
            'primary_route': None,
            'alternatives': [],
            'route_comparison': {},
            'map_data': None
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_route_integration_instance = None


def get_route_integration() -> TransportationRouteIntegration:
    """Get or create singleton TransportationRouteIntegration instance"""
    global _route_integration_instance
    if _route_integration_instance is None:
        _route_integration_instance = TransportationRouteIntegration()
        logger.info("✅ TransportationRouteIntegration initialized")
    return _route_integration_instance


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Test the route integration
    """
    import json
    
    print("🚀 Testing Transportation Route Integration\n")
    
    integration = get_route_integration()
    
    # Test case: Taksim to Kadıköy
    result = integration.get_route_alternatives(
        origin="Taksim",
        destination="Kadıköy",
        origin_gps={"lat": 41.0370, "lon": 28.9850},
        destination_gps={"lat": 40.9900, "lon": 29.0250},
        num_alternatives=3,
        generate_llm_summaries=False
    )
    
    if result['success']:
        print("✅ Route alternatives generated successfully!\n")
        print(f"📍 {result['metadata']['origin']} → {result['metadata']['destination']}")
        print(f"🕐 Departure: {result['metadata']['departure_time']}")
        print(f"🔢 Alternatives: {result['metadata']['num_alternatives']}\n")
        
        print("=" * 60)
        print("PRIMARY ROUTE")
        print("=" * 60)
        pr = result['primary_route']
        print(f"⏱️  Duration: {pr['total_time']} min")
        print(f"📏 Distance: {pr['total_distance']:.1f} km")
        print(f"🔄 Transfers: {pr['transfers']}")
        print(f"🚇 Lines: {', '.join(pr['lines_used'])}\n")
        
        print("=" * 60)
        print("ALTERNATIVE ROUTES")
        print("=" * 60)
        for i, alt in enumerate(result['alternatives'], 1):
            print(f"\n{i}. {alt['preference'].upper()}")
            print(f"   ⏱️  Duration: {alt['duration_minutes']} min")
            print(f"   🔄 Transfers: {alt['num_transfers']}")
            print(f"   👟 Walking: {int(alt['walking_meters'])}m")
            print(f"   ⭐ Comfort: {alt['comfort_score']['overall_comfort']:.0f}/100")
            print(f"   📊 Score: {alt['overall_score']:.1f}/100")
            
            if alt['highlights']:
                print(f"   ✨ Highlights:")
                for h in alt['highlights']:
                    print(f"      {h}")
            
            if alt['llm_summary']:
                print(f"   📝 Summary: {alt['llm_summary']}")
        
        print("\n" + "=" * 60)
        print("ROUTE COMPARISON")
        print("=" * 60)
        comp = result['route_comparison']
        if comp:
            print("\n📊 Metrics:")
            for metric_name, metric_data in comp['metrics'].items():
                print(f"\n   {metric_name.title()}:")
                for m in metric_data:
                    print(f"      {m['preference']}: {m['label']}")
    else:
        print(f"❌ Error: {result['error']}")
