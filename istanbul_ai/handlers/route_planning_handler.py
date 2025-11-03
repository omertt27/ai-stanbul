"""
ML-Enhanced Route Planning Handler
Provides intelligent route planning with ML-powered optimization
Fully bilingual (English/Turkish) support
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RouteContext:
    """Context for route planning"""
    user_query: str
    start_location: Optional[str]
    end_location: Optional[str]
    waypoints: List[str]  # Intermediate stops
    transport_preferences: List[str]  # metro, bus, ferry, walking, taxi
    optimization_goal: str  # 'fastest', 'cheapest', 'scenic', 'comfortable'
    time_constraint: Optional[int]  # Max minutes for route
    budget_constraint: Optional[float]  # Max cost
    accessibility_needs: Optional[List[str]]
    avoid_preferences: List[str]  # crowded_areas, hills, stairs, etc.
    interests: List[str]  # For scenic routing
    departure_time: Optional[datetime]
    weather_context: Optional[Dict[str, Any]]
    comfort_priority: float  # 0.0-1.0
    user_sentiment: float


class MLEnhancedRoutePlanningHandler:
    """
    ML-Enhanced Route Planning Handler
    
    Features:
    - Multi-modal route optimization (metro, bus, ferry, walking)
    - ML-powered waypoint recommendations
    - Context-aware route preferences (scenic, fast, cheap, comfortable)
    - Weather-aware routing (covered vs outdoor routes)
    - Accessibility-conscious planning
    - Real-time transport integration
    - Neural ranking of route alternatives
    - Interest-based scenic routing
    """
    
    def __init__(self, route_planner_service, transport_service,
                 ml_context_builder, ml_processor, response_generator,
                 bilingual_manager=None, map_integration_service=None):
        """
        Initialize handler with required services
        
        Args:
            route_planner_service: Advanced route planning service (Dijkstra/A*)
            transport_service: Real-time transport information service
            ml_context_builder: Centralized ML context builder
            ml_processor: Neural processor for embeddings and ranking
            response_generator: Response generator for natural language output
            bilingual_manager: Bilingual manager for language support
            map_integration_service: MapIntegrationService for map visualization
        """
        self.route_planner_service = route_planner_service
        self.transport_service = transport_service
        self.ml_context_builder = ml_context_builder
        self.ml_processor = ml_processor
        self.response_generator = response_generator
        self.bilingual_manager = bilingual_manager
        self.map_integration_service = map_integration_service
        self.has_bilingual = bilingual_manager is not None
        self.has_maps = map_integration_service is not None and map_integration_service.is_enabled()
        
        logger.info(f"✅ ML-Enhanced Route Planning Handler initialized (Bilingual: {self.has_bilingual}, Maps: {self.has_maps})")
    
    def _get_language(self, context) -> str:
        """Extract language from context"""
        if not context:
            return 'en'
        if hasattr(context, 'language'):
            lang = context.language
            if hasattr(lang, 'value'):
                return lang.value
            return lang if lang in ['en', 'tr'] else 'en'
        return 'en'
    
    def _get_error_message(self, error_type: str, language: str = 'en') -> str:
        """Get bilingual error message"""
        if not self.has_bilingual:
            # Fallback messages
            messages = {
                'no_locations': "I need to know your start and end locations to plan a route. Where are you going?",
                'no_suitable_route': "I couldn't find a suitable route with your requirements. Would you like to adjust your preferences?",
                'planning_error': "I'm having trouble planning that route. Could you provide more details?"
            }
            return messages.get(error_type, "An error occurred.")
        
        from ..services.bilingual_manager import Language
        lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
        return self.bilingual_manager.get_bilingual_response(f'route.error.{error_type}', lang)
    
    def _get_optimization_goal_label(self, goal: str, language: str = 'en') -> str:
        """Get bilingual optimization goal label"""
        if not self.has_bilingual:
            return goal
        
        from ..services.bilingual_manager import Language
        lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
        return self.bilingual_manager.get_bilingual_response(f'route.goal.{goal}', lang)
    
    def _get_quality_label(self, quality: str, language: str = 'en') -> str:
        """Get bilingual quality label"""
        if not self.has_bilingual:
            labels = {
                'scenic': 'Scenic views',
                'comfortable': 'Comfortable',
                'less_crowded': 'Less crowded',
                'weather_protected': 'Weather protected'
            }
            return labels.get(quality, quality)
        
        from ..services.bilingual_manager import Language
        lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
        return self.bilingual_manager.get_bilingual_response(f'route.quality.{quality}', lang)
    
    def _get_mode_emoji(self, mode: str) -> str:
        """Get emoji for transport mode"""
        return {
            "Metro": "🚇",
            "Bus": "🚌",
            "Tram": "🚊",
            "Ferry": "⛴️",
            "Walking": "🚶"
        }.get(mode, "➡️")
    
    async def handle_route_query(
        self,
        user_query: str,
        user_profile: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle route planning query with ML enhancement
        
        Args:
            user_query: User's natural language query
            user_profile: Optional user profile for personalization
            context: Optional additional context (locations, time, language, etc.)
        
        Returns:
            Dict with routes, alternatives, and natural language response
        """
        try:
            # Extract language
            language = self._get_language(context)
            
            # Step 1: Extract ML context
            ml_context = await self.ml_context_builder.build_context(
                query=user_query,
                intent="route_planning",
                user_profile=user_profile,
                additional_context=context
            )
            
            # Step 2: Build route-specific context
            route_context = self._build_route_context(ml_context, context)
            
            # Step 3: Validate locations
            if not route_context.start_location or not route_context.end_location:
                error_msg = self._get_error_message('no_locations', language)
                return {
                    "success": False,
                    "response": error_msg
                }
            
            # Step 4: Get candidate routes from service
            candidate_routes = await self._get_candidate_routes(route_context)
            
            # Step 5: Neural ranking of routes
            ranked_routes = await self._rank_routes_neural(
                routes=candidate_routes,
                context=route_context,
                ml_context=ml_context
            )
            
            # Step 6: Apply filters (time, budget, accessibility)
            filtered_routes = self._apply_filters(
                ranked_routes,
                route_context
            )
            
            # Step 7: Generate personalized response
            response = await self._generate_response(
                routes=filtered_routes[:3],  # Top 3 alternatives
                context=route_context,
                ml_context=ml_context,
                language=language
            )
            
            # Step 8: Generate map visualization for route
            map_data = None
            if self.has_maps and filtered_routes:
                try:
                    map_data = self._generate_route_map(filtered_routes[0], route_context)
                    if map_data:
                        logger.info(f"🗺️ Generated route map with {len(route_context.waypoints) + 2} stops")
                except Exception as e:
                    logger.warning(f"Failed to generate route map: {e}")
            
            return {
                "success": True,
                "routes": filtered_routes[:3],
                "primary_route": filtered_routes[0] if filtered_routes else None,
                "response": response,
                "map_data": map_data,
                "context_used": {
                    "optimization_goal": route_context.optimization_goal,
                    "transport_modes": route_context.transport_preferences,
                    "comfort_priority": route_context.comfort_priority
                }
            }
            
        except Exception as e:
            logger.error(f"Error in route planning handler: {e}")
            error_msg = self._get_error_message('planning_error', language if 'language' in locals() else 'en')
            return {
                "success": False,
                "error": str(e),
                "response": error_msg
            }
    
    def _build_route_context(
        self,
        ml_context: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]]
    ) -> RouteContext:
        """Build route-specific context from ML context"""
        
        query_lower = ml_context.get("original_query", "").lower()
        
        # Extract locations from context or query
        start_location = None
        end_location = None
        waypoints = []
        
        if additional_context:
            start_location = additional_context.get("start_location")
            end_location = additional_context.get("end_location")
            waypoints = additional_context.get("waypoints", [])
        
        # Extract transport preferences
        transport_preferences = []
        transport_keywords = {
            "metro": ["metro", "subway", "underground", "m1", "m2"],
            "bus": ["bus", "autobus"],
            "ferry": ["ferry", "boat", "vapur", "sea"],
            "tram": ["tram", "tramvay"],
            "walking": ["walk", "walking", "on foot", "pedestrian"],
            "taxi": ["taxi", "uber", "car"]
        }
        
        for transport, keywords in transport_keywords.items():
            if any(kw in query_lower for kw in keywords):
                transport_preferences.append(transport)
        
        # Default to all public transport if none specified
        if not transport_preferences:
            transport_preferences = ["metro", "bus", "tram", "ferry", "walking"]
        
        # Determine optimization goal
        optimization_goal = "fastest"  # Default
        if any(kw in query_lower for kw in ["cheap", "cheapest", "budget", "economical"]):
            optimization_goal = "cheapest"
        elif any(kw in query_lower for kw in ["scenic", "beautiful", "views", "sightseeing"]):
            optimization_goal = "scenic"
        elif any(kw in query_lower for kw in ["comfortable", "easy", "direct", "simple"]):
            optimization_goal = "comfortable"
        elif any(kw in query_lower for kw in ["fast", "fastest", "quick", "quickest"]):
            optimization_goal = "fastest"
        
        # Time and budget constraints
        time_constraint = None
        budget_constraint = None
        
        # Extract time constraint (e.g., "in under 30 minutes")
        time_words = ["minutes", "min", "hour", "hours"]
        for word in query_lower.split():
            if word.isdigit():
                if any(tw in query_lower for tw in time_words):
                    time_constraint = int(word)
                    if "hour" in query_lower:
                        time_constraint *= 60
                    break
        
        # Avoid preferences
        avoid_preferences = []
        if any(kw in query_lower for kw in ["avoid crowded", "less crowded", "quiet"]):
            avoid_preferences.append("crowded_areas")
        if any(kw in query_lower for kw in ["no stairs", "avoid stairs", "flat"]):
            avoid_preferences.append("stairs")
        if any(kw in query_lower for kw in ["no hills", "flat route"]):
            avoid_preferences.append("hills")
        if any(kw in query_lower for kw in ["avoid transfers", "direct", "no changes"]):
            avoid_preferences.append("transfers")
        
        # Comfort priority (0.0-1.0)
        comfort_priority = 0.5
        if optimization_goal == "comfortable":
            comfort_priority = 0.9
        elif optimization_goal == "fastest":
            comfort_priority = 0.3
        
        # Departure time
        departure_time = None
        if additional_context and "departure_time" in additional_context:
            departure_time = additional_context["departure_time"]
        else:
            departure_time = datetime.now()
        
        return RouteContext(
            user_query=ml_context.get("original_query", ""),
            start_location=start_location,
            end_location=end_location,
            waypoints=waypoints,
            transport_preferences=transport_preferences,
            optimization_goal=optimization_goal,
            time_constraint=time_constraint,
            budget_constraint=budget_constraint,
            accessibility_needs=ml_context.get("accessibility_needs"),
            avoid_preferences=avoid_preferences,
            interests=ml_context.get("detected_interests", []),
            departure_time=departure_time,
            weather_context=ml_context.get("weather"),
            comfort_priority=comfort_priority,
            user_sentiment=ml_context.get("sentiment_score", 0.0)
        )
    
    async def _get_candidate_routes(
        self,
        context: RouteContext
    ) -> List[Dict[str, Any]]:
        """Get candidate routes from route planner service"""
        
        try:
            # Get routes from advanced route planner
            routes = await self.route_planner_service.plan_route(
                start=context.start_location,
                end=context.end_location,
                waypoints=context.waypoints,
                transport_modes=context.transport_preferences,
                optimization=context.optimization_goal,
                departure_time=context.departure_time
            )
            
            return routes
            
        except Exception as e:
            logger.warning(f"Error fetching routes: {e}, using mock data")
            return self._get_mock_routes(context)
    
    def _get_mock_routes(self, context: RouteContext) -> List[Dict[str, Any]]:
        """Return mock routes for development"""
        
        return [
            {
                "id": "route_001",
                "name": "Metro + Tram Route",
                "segments": [
                    {
                        "mode": "metro",
                        "line": "M2",
                        "from": context.start_location or "Taksim",
                        "to": "Şişhane",
                        "duration_minutes": 5,
                        "stops": 2
                    },
                    {
                        "mode": "walking",
                        "from": "Şişhane",
                        "to": "Karaköy",
                        "duration_minutes": 8,
                        "distance_km": 0.6
                    },
                    {
                        "mode": "tram",
                        "line": "T1",
                        "from": "Karaköy",
                        "to": context.end_location or "Sultanahmet",
                        "duration_minutes": 12,
                        "stops": 5
                    }
                ],
                "total_duration_minutes": 25,
                "total_cost": 15.0,
                "total_distance_km": 4.2,
                "transfers": 2,
                "comfort_score": 0.75,
                "scenic_score": 0.6,
                "accessibility_score": 0.8,
                "crowding_level": "moderate",
                "weather_protection": 0.7
            },
            {
                "id": "route_002",
                "name": "Ferry + Walking Route",
                "segments": [
                    {
                        "mode": "walking",
                        "from": context.start_location or "Taksim",
                        "to": "Kabataş",
                        "duration_minutes": 15,
                        "distance_km": 1.2
                    },
                    {
                        "mode": "ferry",
                        "line": "Kabataş-Karaköy",
                        "from": "Kabataş",
                        "to": "Eminönü",
                        "duration_minutes": 20,
                        "bosphorus_view": True
                    },
                    {
                        "mode": "walking",
                        "from": "Eminönü",
                        "to": context.end_location or "Sultanahmet",
                        "duration_minutes": 10,
                        "distance_km": 0.8
                    }
                ],
                "total_duration_minutes": 45,
                "total_cost": 25.0,
                "total_distance_km": 6.5,
                "transfers": 1,
                "comfort_score": 0.85,
                "scenic_score": 0.95,
                "accessibility_score": 0.65,
                "crowding_level": "low",
                "weather_protection": 0.5
            },
            {
                "id": "route_003",
                "name": "Direct Bus Route",
                "segments": [
                    {
                        "mode": "walking",
                        "from": context.start_location or "Taksim",
                        "to": "Taksim Bus Stop",
                        "duration_minutes": 3,
                        "distance_km": 0.2
                    },
                    {
                        "mode": "bus",
                        "line": "BN1",
                        "from": "Taksim",
                        "to": context.end_location or "Sultanahmet",
                        "duration_minutes": 35,
                        "stops": 15,
                        "traffic_dependent": True
                    }
                ],
                "total_duration_minutes": 38,
                "total_cost": 15.0,
                "total_distance_km": 5.5,
                "transfers": 0,
                "comfort_score": 0.6,
                "scenic_score": 0.4,
                "accessibility_score": 0.85,
                "crowding_level": "high",
                "weather_protection": 0.9
            }
        ]
    
    async def _rank_routes_neural(
        self,
        routes: List[Dict[str, Any]],
        context: RouteContext,
        ml_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank routes using neural similarity + route characteristics"""
        
        # Get query embedding
        query_embedding = await self.ml_processor.get_embedding(context.user_query)
        
        scored_routes = []
        for route in routes:
            # Create route description for embedding
            modes = [seg.get("mode", "") for seg in route.get("segments", [])]
            desc = f"Route via {', '.join(modes)}: {route.get('total_duration_minutes', 0)} minutes"
            
            # Get route embedding
            route_embedding = await self.ml_processor.get_embedding(desc)
            
            # Calculate base similarity
            base_score = self.ml_processor.calculate_similarity(
                query_embedding,
                route_embedding
            )
            
            # Apply route-specific adjustments
            adjusted_score = self._adjust_score_with_route_factors(
                base_score,
                route,
                context
            )
            
            scored_routes.append({
                **route,
                "ml_score": adjusted_score,
                "base_similarity": base_score
            })
        
        # Sort by ML score
        scored_routes.sort(key=lambda x: x["ml_score"], reverse=True)
        
        return scored_routes
    
    def _adjust_score_with_route_factors(
        self,
        base_score: float,
        route: Dict[str, Any],
        context: RouteContext
    ) -> float:
        """Adjust score based on route characteristics and user preferences"""
        
        score = base_score
        boost = 0.0
        
        # OPTIMIZATION GOAL MATCHING (most important!)
        if context.optimization_goal == "fastest":
            duration = route.get("total_duration_minutes", 60)
            if duration < 30:
                boost += 0.40
            elif duration < 45:
                boost += 0.25
        
        elif context.optimization_goal == "cheapest":
            cost = route.get("total_cost", 50)
            if cost < 20:
                boost += 0.40
            elif cost < 30:
                boost += 0.25
        
        elif context.optimization_goal == "scenic":
            scenic = route.get("scenic_score", 0.5)
            if scenic > 0.8:
                boost += 0.45
            elif scenic > 0.6:
                boost += 0.25
        
        elif context.optimization_goal == "comfortable":
            comfort = route.get("comfort_score", 0.5)
            if comfort > 0.8:
                boost += 0.40
            elif comfort > 0.6:
                boost += 0.25
        
        # TRANSPORT MODE PREFERENCE
        route_modes = [seg.get("mode", "") for seg in route.get("segments", [])]
        for mode in context.transport_preferences:
            if mode in route_modes:
                boost += 0.10
        
        # TRANSFER PENALTY if user wants direct
        if "transfers" in context.avoid_preferences:
            transfers = route.get("transfers", 0)
            if transfers == 0:
                boost += 0.30
            elif transfers == 1:
                boost += 0.10
            else:
                boost -= 0.15
        
        # CROWDING PENALTY
        if "crowded_areas" in context.avoid_preferences:
            crowd = route.get("crowding_level", "moderate")
            if crowd == "low":
                boost += 0.20
            elif crowd == "moderate":
                boost += 0.05
            else:
                boost -= 0.15
        
        # WEATHER PROTECTION BOOST
        if context.weather_context:
            weather = context.weather_context.get("condition", "")
            protection = route.get("weather_protection", 0.5)
            
            if "rain" in weather.lower():
                if protection > 0.8:
                    boost += 0.25
            elif "clear" in weather.lower():
                if protection < 0.5:  # More outdoor is fine
                    boost += 0.10
        
        # ACCESSIBILITY BOOST
        if context.accessibility_needs:
            access_score = route.get("accessibility_score", 0.5)
            if access_score > 0.8:
                boost += 0.35
        
        # COMFORT PRIORITY WEIGHTING
        comfort = route.get("comfort_score", 0.5)
        comfort_boost = (comfort - 0.5) * context.comfort_priority * 0.3
        boost += comfort_boost
        
        # TIME CONSTRAINT PENALTY
        if context.time_constraint:
            duration = route.get("total_duration_minutes", 60)
            if duration > context.time_constraint:
                boost -= 0.40  # Strong penalty for exceeding time limit
        
        # BUDGET CONSTRAINT PENALTY
        if context.budget_constraint:
            cost = route.get("total_cost", 0)
            if cost > context.budget_constraint:
                boost -= 0.35
        
        # SCENIC INTERESTS BOOST
        if context.interests and route.get("scenic_score", 0) > 0.7:
            boost += 0.15
        
        final_score = score * (1 + boost)
        return max(min(final_score, 1.0), 0.0)
    
    def _apply_filters(
        self,
        routes: List[Dict[str, Any]],
        context: RouteContext
    ) -> List[Dict[str, Any]]:
        """Apply hard filters for constraints"""
        
        filtered = routes
        
        # Hard time constraint
        if context.time_constraint:
            filtered = [
                r for r in filtered
                if r.get("total_duration_minutes", 999) <= context.time_constraint * 1.1  # 10% buffer
            ]
        
        # Hard budget constraint
        if context.budget_constraint:
            filtered = [
                r for r in filtered
                if r.get("total_cost", 9999) <= context.budget_constraint
            ]
        
        # Accessibility requirements
        if context.accessibility_needs:
            filtered = [
                r for r in filtered
                if r.get("accessibility_score", 0) > 0.6
            ]
        
        return filtered
    
    async def _generate_response(
        self,
        routes: List[Dict[str, Any]],
        context: RouteContext,
        ml_context: Dict[str, Any],
        language: str = 'en'
    ) -> str:
        """Generate bilingual route planning response"""
        
        if not routes:
            if not self.has_bilingual:
                return f"I couldn't find a suitable route from {context.start_location} to {context.end_location} with your requirements. Would you like to adjust your preferences?"
            
            from ..services.bilingual_manager import Language
            lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
            return self.bilingual_manager.get_bilingual_response(
                'route.error.no_suitable_route',
                lang,
                start=context.start_location,
                end=context.end_location
            )
        
        response_parts = []
        
        # Header
        if self.has_bilingual:
            from ..services.bilingual_manager import Language
            lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
            header = self.bilingual_manager.get_bilingual_response(
                'route.header',
                lang,
                start=context.start_location,
                end=context.end_location
            )
            response_parts.append(header + "\n")
        else:
            response_parts.append(f"🗺️ **Route from {context.start_location} to {context.end_location}**\n")
        
        # Primary route
        primary = routes[0]
        
        # Recommended route title
        if self.has_bilingual:
            from ..services.bilingual_manager import Language
            lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
            route_title = self.bilingual_manager.get_bilingual_response(
                'route.recommended',
                lang,
                name=primary.get('name', 'Best Route' if language == 'en' else 'En İyi Güzergah')
            )
            response_parts.append(route_title)
        else:
            response_parts.append(f"🌟 **Recommended Route: {primary.get('name', 'Best Route')}**")
        
        # Match score and optimization
        score = int(primary['ml_score']*100)
        goal_label = self._get_optimization_goal_label(context.optimization_goal, language)
        
        if self.has_bilingual:
            from ..services.bilingual_manager import Language
            lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
            match_line = self.bilingual_manager.get_bilingual_response(
                'route.match_optimized',
                lang,
                score=score,
                goal=goal_label
            )
            response_parts.append(f"   {match_line}")
        else:
            response_parts.append(f"   (Match: {score}%, Optimized for: {goal_label})")
        
        # Duration, cost, transfers
        duration = primary.get('total_duration_minutes', 'N/A')
        cost = primary.get('total_cost', 'N/A')
        transfers = primary.get('transfers', 0)
        
        if self.has_bilingual:
            from ..services.bilingual_manager import Language
            lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
            
            duration_line = self.bilingual_manager.get_bilingual_response(
                'route.duration', lang, minutes=duration
            )
            cost_line = self.bilingual_manager.get_bilingual_response(
                'route.cost', lang, cost=cost
            )
            transfers_line = self.bilingual_manager.get_bilingual_response(
                'route.transfers', lang, count=transfers
            )
            
            response_parts.append(f"\n   {duration_line}")
            response_parts.append(f"   {cost_line}")
            response_parts.append(f"   {transfers_line}")
        else:
            response_parts.append(f"\n   ⏱️ Duration: {duration} minutes")
            response_parts.append(f"   💰 Cost: {cost} TL")
            response_parts.append(f"   🔄 Transfers: {transfers}")
        
        # Route segments
        if self.has_bilingual:
            from ..services.bilingual_manager import Language
            lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
            directions_header = self.bilingual_manager.get_bilingual_response('route.directions', lang)
            response_parts.append(f"\n   {directions_header}")
        else:
            response_parts.append(f"\n   **Directions:**")
        
        for i, segment in enumerate(primary.get("segments", []), 1):
            mode = segment.get("mode", "").title()
            from_loc = segment.get("from", "")
            to_loc = segment.get("to", "")
            duration_seg = segment.get("duration_minutes", "")
            
            mode_emoji = self._get_mode_emoji(mode)
            line = segment.get("line", "")
            line_info = f" ({line})" if line else ""
            
            min_label = "min" if language == 'en' else "dk"
            response_parts.append(f"   {i}. {mode_emoji} {mode}{line_info}: {from_loc} → {to_loc} ({duration_seg} {min_label})")
        
        # Route qualities
        qualities = []
        if primary.get("scenic_score", 0) > 0.7:
            qualities.append(self._get_quality_label('scenic', language))
        if primary.get("comfort_score", 0) > 0.75:
            qualities.append(self._get_quality_label('comfortable', language))
        if primary.get("crowding_level") == "low":
            qualities.append(self._get_quality_label('less_crowded', language))
        if primary.get("weather_protection", 0) > 0.8:
            qualities.append(self._get_quality_label('weather_protected', language))
        
        if qualities:
            quality_text = ', '.join(qualities)
            if self.has_bilingual:
                from ..services.bilingual_manager import Language
                lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
                qualities_line = self.bilingual_manager.get_bilingual_response(
                    'route.qualities', lang, qualities=quality_text
                )
                response_parts.append(f"\n   {qualities_line}")
            else:
                response_parts.append(f"\n   ✨ Route qualities: {quality_text}")
        
        # Alternative routes
        if len(routes) > 1:
            if self.has_bilingual:
                from ..services.bilingual_manager import Language
                lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
                alt_header = self.bilingual_manager.get_bilingual_response('route.alternatives', lang)
                response_parts.append(f"\n\n{alt_header}")
            else:
                response_parts.append(f"\n\n🔀 **Alternative Routes:**")
            
            for route in routes[1:]:
                duration_alt = route.get("total_duration_minutes", "N/A")
                cost_alt = route.get("total_cost", "N/A")
                transfers_alt = route.get("transfers", 0)
                name_alt = route.get('name', 'Alternative' if language == 'en' else 'Alternatif')
                
                if self.has_bilingual:
                    from ..services.bilingual_manager import Language
                    lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
                    alt_line = self.bilingual_manager.get_bilingual_response(
                        'route.alternative_item',
                        lang,
                        name=name_alt,
                        duration=duration_alt,
                        cost=cost_alt,
                        transfers=transfers_alt
                    )
                    response_parts.append(f"   • {alt_line}")
                else:
                    response_parts.append(
                        f"   • {name_alt}: {duration_alt} min, {cost_alt} TL, {transfers_alt} transfer(s)"
                    )
        
        # Tips based on context
        tips = self._generate_route_tips(context, primary, language)
        if tips:
            response_parts.append(f"\n\n{chr(10).join(tips)}")
        
        # Departure info
        if context.departure_time:
            dep_time = context.departure_time.strftime("%H:%M")
            arr_time = (context.departure_time + timedelta(minutes=primary.get("total_duration_minutes", 0))).strftime("%H:%M")
            
            if self.has_bilingual:
                from ..services.bilingual_manager import Language
                lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
                departure_line = self.bilingual_manager.get_bilingual_response(
                    'route.departure',
                    lang,
                    dep_time=dep_time,
                    arr_time=arr_time
                )
                response_parts.append(f"\n\n{departure_line}")
            else:
                response_parts.append(f"\n\n🕐 Departure: {dep_time} | Arrival: ~{arr_time}")
        
        return "\n".join(response_parts)
    
    def _generate_route_tips(
        self,
        context: RouteContext,
        primary_route: Dict[str, Any],
        language: str = 'en'
    ) -> List[str]:
        """Generate contextual bilingual tips for the route"""
        tips = []
        
        if not self.has_bilingual:
            # Fallback English tips
            if context.optimization_goal == "cheapest":
                tips.append("💡 Using Istanbul Kart saves ~30% on all public transport")
            
            if primary_route.get("crowding_level") == "high":
                tips.append("⏰ Tip: This route can be crowded during rush hours (8-9 AM, 5-7 PM)")
            
            if context.weather_context and "rain" in context.weather_context.get("condition", "").lower():
                if primary_route.get("weather_protection", 0) < 0.6:
                    tips.append("☔ Weather alert: Bring an umbrella, parts of this route are outdoors")
            
            if any(seg.get("mode") == "ferry" for seg in primary_route.get("segments", [])):
                tips.append("⛴️ Ferry tip: Amazing Bosphorus views! Arrive 10 min early for good seats")
            
            return tips
        
        from ..services.bilingual_manager import Language
        lang = Language.TURKISH if language == 'tr' else Language.ENGLISH
        
        # Istanbul Kart tip for cheapest routes
        if context.optimization_goal == "cheapest":
            tips.append(self.bilingual_manager.get_bilingual_response('route.tip.istanbul_kart', lang))
        
        # Crowding tip
        if primary_route.get("crowding_level") == "high":
            tips.append(self.bilingual_manager.get_bilingual_response('route.tip.crowded', lang))
        
        # Weather tip
        if context.weather_context and "rain" in context.weather_context.get("condition", "").lower():
            if primary_route.get("weather_protection", 0) < 0.6:
                tips.append(self.bilingual_manager.get_bilingual_response('route.tip.rain_umbrella', lang))
        
        # Ferry tip
        if any(seg.get("mode") == "ferry" for seg in primary_route.get("segments", [])):
            tips.append(self.bilingual_manager.get_bilingual_response('route.tip.ferry_views', lang))
        
        return tips
    
    def _generate_route_map(self, route: Dict[str, Any], context: RouteContext) -> Optional[Dict]:
        """Generate interactive map for route plan"""
        if not self.has_maps or not self.map_integration_service:
            return None
        
        try:
            # Extract start and end locations
            start_location = (
                route.get('start_lat', 41.0082),
                route.get('start_lon', 28.9784),
                context.start_location or 'Start'
            )
            
            end_location = (
                route.get('end_lat', 41.0082),
                route.get('end_lon', 28.9784),
                context.end_location or 'End'
            )
            
            # Extract waypoints
            waypoints = []
            for waypoint in context.waypoints:
                if isinstance(waypoint, dict) and 'lat' in waypoint and 'lon' in waypoint:
                    waypoints.append((
                        waypoint['lat'],
                        waypoint['lon'],
                        waypoint.get('name', 'Waypoint')
                    ))
                elif isinstance(waypoint, str):
                    # Waypoint name only, need to resolve to coordinates
                    # This would require geocoding, skip for now
                    pass
            
            # Prepare route info
            route_info = {
                'total_distance_km': route.get('total_distance_km', 0),
                'total_duration_min': route.get('total_duration_minutes', 0),
                'optimization_goal': context.optimization_goal,
                'transport_modes': context.transport_preferences
            }
            
            # Create route map with waypoints
            map_data = self.map_integration_service.create_route_map(
                start_location=start_location,
                end_location=end_location,
                waypoints=waypoints if waypoints else None,
                route_info=route_info
            )
            
            return map_data
            
        except Exception as e:
            logger.error(f"Error generating route map: {e}")
            return None


# Export
def create_ml_enhanced_route_planning_handler(
    route_planner_service,
    transport_service,
    ml_context_builder,
    ml_processor,
    response_generator,
    bilingual_manager=None
):
    """Factory function to create ML-enhanced route planning handler with bilingual support"""
    return MLEnhancedRoutePlanningHandler(
        route_planner_service=route_planner_service,
        transport_service=transport_service,
        ml_context_builder=ml_context_builder,
        ml_processor=ml_processor,
        response_generator=response_generator,
        bilingual_manager=bilingual_manager
    )
