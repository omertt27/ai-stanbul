"""
Integration service for Live Location Routing System
Connects algorithmic location-based routing with the existing comprehensive domain system

This service provides:
- API endpoints for live location features
- Integration with existing Istanbul AI backend
- Privacy-safe location handling
- Real-time route updates
- Offline mode support
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging
from dataclasses import asdict

from .live_location_routing_system import (
    LiveLocationRoutingSystem, 
    Coordinates, 
    POICategory, 
    FilterCriteria, 
    RoutingAlgorithm
)
from .comprehensive_domain_system import ComprehensiveDomainSystem

logger = logging.getLogger(__name__)

class LiveLocationIntegrationService:
    """Integration service for live location features with existing AI system"""
    
    def __init__(self):
        self.routing_system = LiveLocationRoutingSystem()
        self.domain_system = ComprehensiveDomainSystem()
        self.active_sessions: Dict[str, Dict] = {}
        logger.info("Live Location Integration Service initialized")
    
    def start_location_session(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new live location session
        
        Args:
            user_input: {
                "user_id": str,
                "latitude": float,
                "longitude": float,
                "accuracy": Optional[float],
                "preferences": Optional[Dict]
            }
        """
        try:
            # Extract location data
            latitude = user_input.get("latitude")
            longitude = user_input.get("longitude")
            accuracy = user_input.get("accuracy")
            user_id = user_input.get("user_id", "anonymous")
            preferences = user_input.get("preferences", {})
            
            if not latitude or not longitude:
                return {
                    "error": "Location coordinates required",
                    "required_fields": ["latitude", "longitude"]
                }
            
            # Create coordinates object
            location = Coordinates(
                latitude=float(latitude),
                longitude=float(longitude),
                accuracy=accuracy,
                timestamp=datetime.now()
            )
            
            # Create routing session
            session_id = self.routing_system.create_user_session(user_id, location)
            
            # Store integration session data
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "preferences": preferences,
                "created_at": datetime.now(),
                "last_activity": datetime.now()
            }
            
            # Get initial recommendations
            initial_recommendations = self.routing_system.get_smart_recommendations(
                session_id=session_id,
                limit=8
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "location_privacy_hash": location.privacy_hash,
                "message": "Live location session started successfully",
                "initial_recommendations": initial_recommendations,
                "features_available": [
                    "real_time_poi_recommendations",
                    "dynamic_route_planning",
                    "live_location_updates",
                    "offline_mode_support",
                    "smart_filtering",
                    "district_estimates"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error starting location session: {str(e)}")
            return {
                "error": "Failed to start location session",
                "details": str(e)
            }
    
    def update_location(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Update user's current location
        
        Args:
            user_input: {
                "session_id": str,
                "latitude": float,
                "longitude": float,
                "accuracy": Optional[float]
            }
        """
        try:
            session_id = user_input.get("session_id")
            latitude = user_input.get("latitude")
            longitude = user_input.get("longitude")
            accuracy = user_input.get("accuracy")
            
            if not session_id or not latitude or not longitude:
                return {
                    "error": "Session ID and location coordinates required",
                    "required_fields": ["session_id", "latitude", "longitude"]
                }
            
            # Create new coordinates
            new_location = Coordinates(
                latitude=float(latitude),
                longitude=float(longitude),
                accuracy=accuracy,
                timestamp=datetime.now()
            )
            
            # Update location in routing system
            location_updated = self.routing_system.update_user_location(session_id, new_location)
            
            if not location_updated:
                return {
                    "success": True,
                    "location_updated": False,
                    "message": "Location change too small, no update needed"
                }
            
            # Update session activity
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["last_activity"] = datetime.now()
            
            # Get updated recommendations
            updated_recommendations = self.routing_system.get_smart_recommendations(
                session_id=session_id,
                limit=6
            )
            
            return {
                "success": True,
                "location_updated": True,
                "session_id": session_id,
                "updated_recommendations": updated_recommendations,
                "message": "Location updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error updating location: {str(e)}")
            return {
                "error": "Failed to update location",
                "details": str(e)
            }
    
    def get_filtered_recommendations(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get POI recommendations with advanced filtering
        
        Args:
            user_input: {
                "session_id": str,
                "categories": Optional[List[str]],
                "cuisine_types": Optional[List[str]],
                "price_ranges": Optional[List[str]],
                "open_now": Optional[bool],
                "min_rating": Optional[float],
                "accessibility_required": Optional[List[str]],
                "dietary_requirements": Optional[List[str]],
                "max_distance_km": Optional[float],
                "limit": Optional[int]
            }
        """
        try:
            session_id = user_input.get("session_id")
            if not session_id:
                return {"error": "Session ID required"}
            
            # Parse categories
            categories = None
            if user_input.get("categories"):
                categories = []
                for cat_str in user_input["categories"]:
                    try:
                        categories.append(POICategory(cat_str.lower()))
                    except ValueError:
                        continue
            
            # Build filters
            filters = {}
            
            if user_input.get("cuisine_types"):
                filters[FilterCriteria.CUISINE_TYPE] = user_input["cuisine_types"]
            
            if user_input.get("price_ranges"):
                filters[FilterCriteria.PRICE_RANGE] = user_input["price_ranges"]
            
            if user_input.get("open_now") is True:
                filters[FilterCriteria.OPEN_HOURS] = True
            
            if user_input.get("min_rating"):
                filters[FilterCriteria.RATING] = float(user_input["min_rating"])
            
            if user_input.get("accessibility_required"):
                filters[FilterCriteria.ACCESSIBILITY] = user_input["accessibility_required"]
            
            if user_input.get("dietary_requirements"):
                filters[FilterCriteria.DIETARY] = user_input["dietary_requirements"]
            
            # Get filtered recommendations
            limit = user_input.get("limit", 10)
            recommendations = self.routing_system.get_smart_recommendations(
                session_id=session_id,
                categories=categories,
                filters=filters if filters else None,
                limit=limit
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "filters_applied": {
                    "categories": [cat.value for cat in categories] if categories else None,
                    "filters": {k.value: v for k, v in filters.items()},
                    "limit": limit
                },
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting filtered recommendations: {str(e)}")
            return {
                "error": "Failed to get filtered recommendations",
                "details": str(e)
            }
    
    def plan_live_route(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a dynamic route with real-time updates
        
        Args:
            user_input: {
                "session_id": str,
                "target_poi_ids": List[str],
                "algorithm": Optional[str],  # "nearest", "dijkstra", "a_star", "greedy"
                "transport_mode": Optional[str],  # "walking", "public_transport", "mixed"
                "optimize_for": Optional[str]  # "time", "distance", "experience"
            }
        """
        try:
            session_id = user_input.get("session_id")
            target_poi_ids = user_input.get("target_poi_ids", [])
            
            if not session_id or not target_poi_ids:
                return {
                    "error": "Session ID and target POI IDs required",
                    "required_fields": ["session_id", "target_poi_ids"]
                }
            
            # Parse algorithm
            algorithm_str = user_input.get("algorithm", "nearest").lower()
            algorithm_map = {
                "nearest": RoutingAlgorithm.TSP_NEAREST_NEIGHBOR,
                "dijkstra": RoutingAlgorithm.DIJKSTRA,
                "a_star": RoutingAlgorithm.A_STAR,
                "greedy": RoutingAlgorithm.TSP_GREEDY
            }
            algorithm = algorithm_map.get(algorithm_str, RoutingAlgorithm.TSP_NEAREST_NEIGHBOR)
            
            # Set transport mode
            transport_mode = user_input.get("transport_mode", "mixed")
            
            # Plan the route
            route_plan = self.routing_system.plan_dynamic_route(
                session_id=session_id,
                target_poi_ids=target_poi_ids,
                algorithm=algorithm,
                transport_mode=transport_mode
            )
            
            # Add optimization insights
            optimization_insights = self._generate_route_insights(route_plan, user_input.get("optimize_for"))
            
            return {
                "success": True,
                "session_id": session_id,
                "route_plan": route_plan,
                "optimization_insights": optimization_insights,
                "live_updates_enabled": True,
                "message": "Dynamic route planned successfully"
            }
            
        except Exception as e:
            logger.error(f"Error planning live route: {str(e)}")
            return {
                "error": "Failed to plan live route",
                "details": str(e)
            }
    
    def get_district_navigation(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get navigation info to different Istanbul districts
        
        Args:
            user_input: {
                "session_id": Optional[str],
                "latitude": Optional[float],
                "longitude": Optional[float],
                "target_districts": Optional[List[str]]
            }
        """
        try:
            # Get current location
            current_location = None
            
            if user_input.get("session_id"):
                session_id = user_input["session_id"]
                if session_id in self.routing_system.user_sessions:
                    current_location = self.routing_system.user_sessions[session_id]["current_location"]
            
            elif user_input.get("latitude") and user_input.get("longitude"):
                current_location = Coordinates(
                    latitude=float(user_input["latitude"]),
                    longitude=float(user_input["longitude"])
                )
            
            if not current_location:
                return {
                    "error": "Current location required",
                    "options": "Provide either session_id or latitude/longitude"
                }
            
            # Get district estimates
            district_estimates = self.routing_system.get_district_estimates(current_location)
            
            # Filter by target districts if specified
            target_districts = user_input.get("target_districts")
            if target_districts:
                filtered_estimates = {
                    district: estimates for district, estimates in district_estimates.items()
                    if district.lower() in [td.lower() for td in target_districts]
                }
                district_estimates = filtered_estimates
            
            # Sort by distance
            sorted_districts = sorted(
                district_estimates.items(),
                key=lambda x: x[1]["distance_km"]
            )
            
            return {
                "success": True,
                "current_location": {
                    "latitude": current_location.latitude,
                    "longitude": current_location.longitude
                },
                "district_navigation": dict(sorted_districts),
                "nearest_districts": [district for district, _ in sorted_districts[:5]],
                "total_districts": len(district_estimates)
            }
            
        except Exception as e:
            logger.error(f"Error getting district navigation: {str(e)}")
            return {
                "error": "Failed to get district navigation",
                "details": str(e)
            }
    
    def get_offline_mode_data(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for offline/minimal AI operation
        
        Args:
            user_input: {
                "latitude": float,
                "longitude": float,
                "radius_km": Optional[float],
                "include_categories": Optional[List[str]]
            }
        """
        try:
            latitude = user_input.get("latitude")
            longitude = user_input.get("longitude")
            
            if not latitude or not longitude:
                return {
                    "error": "Location coordinates required for offline mode",
                    "required_fields": ["latitude", "longitude"]
                }
            
            location = Coordinates(
                latitude=float(latitude),
                longitude=float(longitude)
            )
            
            radius_km = user_input.get("radius_km", 2.0)
            
            # Get offline recommendations
            offline_data = self.routing_system.get_offline_recommendations(
                location=location,
                radius_km=radius_km
            )
            
            # Add basic routing between nearby POIs
            nearby_pois = self.routing_system.find_nearby_pois(
                location=location,
                radius_km=radius_km,
                limit=15
            )
            
            # Simple route suggestions
            simple_routes = []
            if len(nearby_pois) >= 2:
                # Create basic walking routes
                for i in range(min(3, len(nearby_pois) - 1)):
                    poi1, dist1 = nearby_pois[i]
                    poi2, dist2 = nearby_pois[i + 1]
                    
                    route_distance = self.routing_system.location_data._haversine_distance(
                        poi1.coordinates, poi2.coordinates
                    )
                    
                    simple_routes.append({
                        "from": poi1.name,
                        "to": poi2.name,
                        "distance_km": round(route_distance, 2),
                        "walking_minutes": int(route_distance * 12),
                        "categories": [poi1.category.value, poi2.category.value]
                    })
            
            offline_data["simple_routes"] = simple_routes
            offline_data["data_generated_at"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "offline_mode": True,
                "offline_data": offline_data,
                "message": "Offline mode data generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error getting offline mode data: {str(e)}")
            return {
                "error": "Failed to get offline mode data",
                "details": str(e)
            }
    
    def _generate_route_insights(self, route_plan: Dict[str, Any], optimize_for: Optional[str]) -> Dict[str, Any]:
        """Generate optimization insights for a route plan"""
        
        if "route" not in route_plan:
            return {}
        
        route = route_plan["route"]
        insights = {
            "total_pois": len(route.get("waypoints", [])),
            "average_segment_time": 0,
            "longest_segment": {},
            "shortest_segment": {},
            "optimization_score": route.get("confidence_score", 0),
            "recommendations": []
        }
        
        segments = route.get("segments", [])
        if not segments:
            return insights
        
        # Calculate segment statistics
        segment_times = [seg["time_minutes"] for seg in segments]
        insights["average_segment_time"] = sum(segment_times) // len(segment_times)
        
        # Find longest and shortest segments
        longest_seg = max(segments, key=lambda x: x["time_minutes"])
        shortest_seg = min(segments, key=lambda x: x["time_minutes"])
        
        insights["longest_segment"] = {
            "from": longest_seg["from"],
            "to": longest_seg["to"],
            "time_minutes": longest_seg["time_minutes"]
        }
        
        insights["shortest_segment"] = {
            "from": shortest_seg["from"],
            "to": shortest_seg["to"],
            "time_minutes": shortest_seg["time_minutes"]
        }
        
        # Generate recommendations based on optimization preference
        if optimize_for == "time":
            if route["total_time_minutes"] > 240:  # 4 hours
                insights["recommendations"].append(
                    "Consider splitting this route across multiple days for a more relaxed experience"
                )
        elif optimize_for == "distance":
            if route["total_distance_km"] > 10:
                insights["recommendations"].append(
                    "This route covers significant distance - consider using public transport between distant POIs"
                )
        elif optimize_for == "experience":
            insights["recommendations"].append(
                "Route optimized for experience - includes varied POI types and considers visit durations"
            )
        
        return insights
    
    def cleanup_inactive_sessions(self, max_inactive_hours: int = 24):
        """Clean up inactive sessions to free memory"""
        
        current_time = datetime.now()
        inactive_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            last_activity = session_data.get("last_activity", session_data["created_at"])
            hours_inactive = (current_time - last_activity).total_seconds() / 3600
            
            if hours_inactive > max_inactive_hours:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            # Remove from both systems
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.routing_system.user_sessions:
                del self.routing_system.user_sessions[session_id]
        
        logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
        return len(inactive_sessions)

    # FastAPI-compatible method aliases
    def get_recommendations(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for get_filtered_recommendations for FastAPI compatibility"""
        return self.get_filtered_recommendations(user_input)
    
    def plan_route(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for plan_live_route for FastAPI compatibility"""
        return self.plan_live_route(user_input)
    
    def update_route(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing route with new conditions"""
        try:
            session_id = user_input.get("session_id")
            route_id = user_input.get("route_id")
            
            if not session_id or not route_id:
                return {
                    "error": "Session ID and route ID required for route updates",
                    "required_fields": ["session_id", "route_id"]
                }
            
            # Get current route from session
            if session_id not in self.routing_system.user_sessions:
                return {"error": "Session not found"}
            
            session = self.routing_system.user_sessions[session_id]
            
            # Handle POI modifications
            skip_poi_ids = user_input.get("skip_poi_ids", [])
            add_poi_ids = user_input.get("add_poi_ids", [])
            current_poi_id = user_input.get("current_poi_id")
            time_delay_minutes = user_input.get("time_delay_minutes", 0)
            
            # Get original POI list and modify it
            original_pois = session.get("target_poi_ids", [])
            updated_pois = [poi_id for poi_id in original_pois if poi_id not in skip_poi_ids]
            updated_pois.extend(add_poi_ids)
            
            # Re-plan route with updated POI list
            if updated_pois:
                replan_input = {
                    "session_id": session_id,
                    "target_poi_ids": updated_pois,
                    "algorithm": user_input.get("algorithm", "nearest"),
                    "transport_mode": user_input.get("transport_mode", "mixed")
                }
                
                result = self.plan_live_route(replan_input)
                
                if result.get("success"):
                    result["route_updated"] = True
                    result["modifications"] = {
                        "skipped_pois": skip_poi_ids,
                        "added_pois": add_poi_ids,
                        "time_delay_minutes": time_delay_minutes
                    }
                    
                return result
            
            return {"error": "No POIs remaining after modifications"}
            
        except Exception as e:
            logger.error(f"Error updating route: {str(e)}")
            return {
                "error": "Failed to update route",
                "details": str(e)
            }
    
    def search_nearby(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Search for POIs near specified location"""
        try:
            # Get search location
            session_id = user_input.get("session_id")
            latitude = user_input.get("latitude")
            longitude = user_input.get("longitude")
            
            search_location = None
            
            if session_id and session_id in self.routing_system.user_sessions:
                search_location = self.routing_system.user_sessions[session_id]["current_location"]
            elif latitude and longitude:
                search_location = Coordinates(
                    latitude=float(latitude),
                    longitude=float(longitude)
                )
            
            if not search_location:
                return {
                    "error": "Search location required",
                    "options": "Provide either session_id or latitude/longitude"
                }
            
            radius_km = user_input.get("radius_km", 1.0)
            limit = user_input.get("limit", 10)
            categories = user_input.get("categories")
            keywords = user_input.get("keywords", [])
            
            # Find nearby POIs
            nearby_pois = self.routing_system.find_nearby_pois(
                location=search_location,
                radius_km=radius_km,
                limit=limit * 2  # Get more to filter
            )
            
            # Filter by categories if specified
            if categories:
                cat_values = [cat.lower() for cat in categories]
                nearby_pois = [
                    (poi, dist) for poi, dist in nearby_pois
                    if poi.category.value.lower() in cat_values
                ]
            
            # Filter by keywords if specified
            if keywords:
                keyword_lower = [kw.lower() for kw in keywords]
                nearby_pois = [
                    (poi, dist) for poi, dist in nearby_pois
                    if any(kw in poi.name.lower() or kw in poi.description.lower() 
                           for kw in keyword_lower)
                ]
            
            # Limit results
            nearby_pois = nearby_pois[:limit]
            
            # Format results
            search_results = []
            for poi, distance in nearby_pois:
                search_results.append({
                    "id": poi.id,
                    "name": poi.name,
                    "category": poi.category.value,
                    "distance_km": round(distance, 2),
                    "walking_minutes": int(distance * 12),
                    "rating": getattr(poi, 'rating', None),
                    "price_range": getattr(poi, 'price_range', None),
                    "coordinates": {
                        "latitude": poi.coordinates.latitude,
                        "longitude": poi.coordinates.longitude
                    }
                })
            
            return {
                "success": True,
                "search_location": {
                    "latitude": search_location.latitude,
                    "longitude": search_location.longitude
                },
                "search_radius_km": radius_km,
                "nearby_pois": search_results,
                "total_found": len(search_results),
                "filters_applied": {
                    "categories": categories,
                    "keywords": keywords
                }
            }
            
        except Exception as e:
            logger.error(f"Error searching nearby POIs: {str(e)}")
            return {
                "error": "Failed to search nearby POIs",
                "details": str(e)
            }
    
    def get_offline_data(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for get_offline_mode_data for FastAPI compatibility"""
        return self.get_offline_mode_data(user_input)
    
    # Additional FastAPI-compatible methods
    def create_session(self, user_input: Dict[str, Any]) -> str:
        """Create a new session - alias for start_location_session"""
        result = self.start_location_session(user_input)
        return result.get('session_id', '')
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a specific session"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"Session {session_id} cleaned up")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
            return False
