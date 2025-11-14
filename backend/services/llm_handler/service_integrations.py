"""
Service Integrations Module
Manages external service integrations (weather, events, hidden gems, price filtering, maps)

Responsibilities:
- Weather service integration (weather_recommendations.py)
- Events service integration (events_service.py with IKSV)
- Hidden gems integration (hidden_gems_handler.py)
- Price filter integration (price_filter_service.py)
- Map visualization (map_visualization_engine.py)

Author: Istanbul AI Team
Date: Phase 2 - Service Integrations Extraction
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import asyncio

logger = logging.getLogger(__name__)


class ServiceIntegrations:
    """
    Manages all external service integrations for the LLM handler.
    
    Coordinates:
    - Weather recommendations service
    - Events service (IKSV integration)
    - Hidden gems handler
    - Price filter service
    - Map visualization engine
    
    All services are lazily loaded and cached for performance.
    Thread-safe and async-compatible.
    """
    
    def __init__(self, enable_services: bool = True):
        """
        Initialize service integrations.
        
        Args:
            enable_services: Whether to enable external services (default: True)
        """
        self.services_enabled = enable_services
        
        # Service instances (lazy-loaded)
        self._weather_service = None
        self._events_service = None
        self._hidden_gems_handler = None
        self._price_filter = None
        self._map_engine = None
        
        # Service availability flags
        self.weather_available = False
        self.events_available = False
        self.hidden_gems_available = False
        self.price_filter_available = False
        self.map_available = False
        
        # Service call statistics
        self.service_stats = {
            "weather_calls": 0,
            "events_calls": 0,
            "hidden_gems_calls": 0,
            "price_filter_calls": 0,
            "map_calls": 0,
            "total_calls": 0
        }
        
        # Initialize services if enabled
        if self.services_enabled:
            self._initialize_services()
        
        logger.info(f"🔌 ServiceIntegrations initialized (services_enabled={enable_services})")
    
    def _initialize_services(self):
        """Initialize all available services."""
        # Weather service
        try:
            from backend.services.weather_recommendations import get_weather_recommendations_service
            self._weather_service = get_weather_recommendations_service()
            self.weather_available = True
            logger.info("  ✅ Weather service initialized")
        except ImportError as e:
            logger.warning(f"  ⚠️ Weather service not available: {e}")
        
        # Events service
        try:
            from backend.services.events_service import get_events_service
            self._events_service = get_events_service()
            self.events_available = True
            logger.info("  ✅ Events service initialized")
        except ImportError as e:
            logger.warning(f"  ⚠️ Events service not available: {e}")
        
        # Hidden gems handler
        try:
            from backend.services.hidden_gems_handler import HiddenGemsHandler
            self._hidden_gems_handler = HiddenGemsHandler()
            self.hidden_gems_available = True
            logger.info("  ✅ Hidden gems handler initialized")
        except ImportError as e:
            logger.warning(f"  ⚠️ Hidden gems handler not available: {e}")
        
        # Price filter service
        try:
            from backend.services.price_filter_service import PriceFilterService
            self._price_filter = PriceFilterService()
            self.price_filter_available = True
            logger.info("  ✅ Price filter service initialized")
        except ImportError as e:
            logger.warning(f"  ⚠️ Price filter service not available: {e}")
        
        # Map visualization engine
        try:
            from backend.services.map_visualization_engine import MapVisualizationEngine
            self._map_engine = MapVisualizationEngine()
            self.map_available = True
            logger.info("  ✅ Map visualization engine initialized")
        except ImportError as e:
            logger.warning(f"  ⚠️ Map visualization engine not available: {e}")
    
    # ============================================================================
    # WEATHER SERVICE INTEGRATION
    # ============================================================================
    
    async def get_weather_context(
        self,
        query: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Get weather-based recommendations.
        
        Args:
            query: User query
            user_preferences: User preferences (budget, interests, etc.)
            
        Returns:
            Weather context string or None
        """
        if not self.weather_available or not self._weather_service:
            return None
        
        try:
            self.service_stats["weather_calls"] += 1
            self.service_stats["total_calls"] += 1
            
            # Get weather recommendations
            recommendations = self._weather_service.get_recommendations(
                query=query,
                preferences=user_preferences
            )
            
            if recommendations:
                return self._format_weather_context(recommendations)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting weather context: {e}")
            return None
    
    def _format_weather_context(self, recommendations: Dict[str, Any]) -> str:
        """Format weather recommendations for prompt context."""
        context_parts = ["🌤️ Weather-Based Recommendations:"]
        
        if "current_weather" in recommendations:
            weather = recommendations["current_weather"]
            context_parts.append(
                f"Current: {weather.get('temp', 'N/A')}°C, {weather.get('condition', 'N/A')}"
            )
        
        if "activities" in recommendations:
            context_parts.append("\nRecommended Activities:")
            for activity in recommendations["activities"][:5]:  # Top 5
                context_parts.append(f"- {activity.get('name')}: {activity.get('description')}")
        
        return "\n".join(context_parts)
    
    # ============================================================================
    # EVENTS SERVICE INTEGRATION
    # ============================================================================
    
    async def get_events_context(
        self,
        date_range: Optional[Tuple[str, str]] = None,
        event_types: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Get events and activities (including IKSV events).
        
        Args:
            date_range: Optional (start_date, end_date) tuple
            event_types: Optional list of event types to filter
            
        Returns:
            Events context string or None
        """
        if not self.events_available or not self._events_service:
            return None
        
        try:
            self.service_stats["events_calls"] += 1
            self.service_stats["total_calls"] += 1
            
            # Get events from service
            if date_range:
                events = self._events_service.get_events_by_date_range(*date_range)
            else:
                events = self._events_service.get_current_and_upcoming_events()
            
            # Filter by type if specified
            if event_types and events:
                events = [e for e in events if e.get('type') in event_types]
            
            if events:
                return self._format_events_context(events)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting events context: {e}")
            return None
    
    def _format_events_context(self, events: List[Dict[str, Any]]) -> str:
        """Format events for prompt context."""
        context_parts = ["🎭 Current & Upcoming Events:"]
        
        for event in events[:10]:  # Top 10 events
            event_line = f"- {event.get('name')}"
            if event.get('date'):
                event_line += f" ({event.get('date')})"
            if event.get('venue'):
                event_line += f" @ {event.get('venue')}"
            context_parts.append(event_line)
        
        return "\n".join(context_parts)
    
    # ============================================================================
    # HIDDEN GEMS INTEGRATION
    # ============================================================================
    
    async def get_hidden_gems_context(
        self,
        query: str,
        neighborhood: Optional[str] = None,
        gem_type: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get hidden gems recommendations.
        
        Args:
            query: User query
            neighborhood: Optional neighborhood filter
            gem_type: Optional type filter (cafe, restaurant, view, etc.)
            user_id: Optional user ID for personalization
            
        Returns:
            Hidden gems context string or None
        """
        if not self.hidden_gems_available or not self._hidden_gems_handler:
            return None
        
        try:
            self.service_stats["hidden_gems_calls"] += 1
            self.service_stats["total_calls"] += 1
            
            # Get hidden gems
            gems = self._hidden_gems_handler.get_hidden_gems(
                query=query,
                neighborhood=neighborhood,
                gem_type=gem_type,
                user_id=user_id
            )
            
            if gems:
                return self._format_hidden_gems_context(gems)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting hidden gems context: {e}")
            return None
    
    def _format_hidden_gems_context(self, gems: List[Dict[str, Any]]) -> str:
        """Format hidden gems for prompt context."""
        context_parts = ["💎 Hidden Gems & Secret Spots:"]
        
        for gem in gems[:8]:  # Top 8 gems
            gem_line = f"- {gem.get('name')}"
            if gem.get('neighborhood'):
                gem_line += f" ({gem.get('neighborhood')})"
            if gem.get('description'):
                gem_line += f": {gem.get('description')[:100]}"
            context_parts.append(gem_line)
        
        return "\n".join(context_parts)
    
    # ============================================================================
    # PRICE FILTER INTEGRATION
    # ============================================================================
    
    async def filter_by_budget(
        self,
        items: List[Dict[str, Any]],
        budget_level: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter items by budget level.
        
        Args:
            items: List of items to filter
            budget_level: Budget level (free, budget, moderate, upscale, luxury)
            user_preferences: Optional user preferences
            
        Returns:
            Filtered list of items
        """
        if not self.price_filter_available or not self._price_filter:
            return items
        
        try:
            self.service_stats["price_filter_calls"] += 1
            self.service_stats["total_calls"] += 1
            
            # Filter items by budget
            filtered = self._price_filter.filter_by_budget(
                items=items,
                budget_level=budget_level,
                preferences=user_preferences
            )
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering by budget: {e}")
            return items
    
    def get_budget_info(self, budget_level: str) -> Optional[Dict[str, Any]]:
        """
        Get budget information for a level.
        
        Args:
            budget_level: Budget level
            
        Returns:
            Budget info dict or None
        """
        if not self.price_filter_available or not self._price_filter:
            return None
        
        try:
            return self._price_filter.get_budget_info(budget_level)
        except Exception as e:
            logger.error(f"Error getting budget info: {e}")
            return None
    
    # ============================================================================
    # MAP VISUALIZATION INTEGRATION
    # ============================================================================
    
    async def generate_map_visualization(
        self,
        locations: List[Dict[str, Any]],
        user_location: Optional[Dict[str, float]] = None,
        route_optimization: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Generate map visualization for locations.
        
        Args:
            locations: List of location dicts with lat/lon
            user_location: Optional user location dict with lat/lon
            route_optimization: Whether to optimize route order
            
        Returns:
            Map visualization data or None
        """
        if not self.map_available or not self._map_engine:
            return None
        
        try:
            self.service_stats["map_calls"] += 1
            self.service_stats["total_calls"] += 1
            
            # Generate map
            map_data = self._map_engine.generate_map(
                locations=locations,
                user_location=user_location,
                optimize_route=route_optimization
            )
            
            return map_data
            
        except Exception as e:
            logger.error(f"Error generating map visualization: {e}")
            return None
    
    # ============================================================================
    # SERVICE MANAGEMENT
    # ============================================================================
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of all services.
        
        Returns:
            Service status dict
        """
        return {
            "services_enabled": self.services_enabled,
            "availability": {
                "weather": self.weather_available,
                "events": self.events_available,
                "hidden_gems": self.hidden_gems_available,
                "price_filter": self.price_filter_available,
                "map": self.map_available
            },
            "statistics": self.service_stats.copy()
        }
    
    def reset_statistics(self):
        """Reset service call statistics."""
        self.service_stats = {
            "weather_calls": 0,
            "events_calls": 0,
            "hidden_gems_calls": 0,
            "price_filter_calls": 0,
            "map_calls": 0,
            "total_calls": 0
        }
        logger.info("Service statistics reset")
    
    def disable_service(self, service_name: str):
        """
        Disable a specific service.
        
        Args:
            service_name: Name of service to disable (weather, events, hidden_gems, price_filter, map)
        """
        service_map = {
            "weather": "_weather_service",
            "events": "_events_service",
            "hidden_gems": "_hidden_gems_handler",
            "price_filter": "_price_filter",
            "map": "_map_engine"
        }
        
        if service_name in service_map:
            setattr(self, service_map[service_name], None)
            setattr(self, f"{service_name}_available", False)
            logger.info(f"Service '{service_name}' disabled")
        else:
            logger.warning(f"Unknown service: {service_name}")
    
    def enable_service(self, service_name: str):
        """
        Enable a specific service (re-initialize).
        
        Args:
            service_name: Name of service to enable
        """
        if not self.services_enabled:
            logger.warning("Services are globally disabled")
            return
        
        # Re-initialize the specific service
        if service_name == "weather" and not self.weather_available:
            try:
                from backend.services.weather_recommendations import get_weather_recommendations_service
                self._weather_service = get_weather_recommendations_service()
                self.weather_available = True
                logger.info("Weather service re-enabled")
            except ImportError as e:
                logger.error(f"Cannot enable weather service: {e}")
        
        elif service_name == "events" and not self.events_available:
            try:
                from backend.services.events_service import get_events_service
                self._events_service = get_events_service()
                self.events_available = True
                logger.info("Events service re-enabled")
            except ImportError as e:
                logger.error(f"Cannot enable events service: {e}")
        
        elif service_name == "hidden_gems" and not self.hidden_gems_available:
            try:
                from backend.services.hidden_gems_handler import HiddenGemsHandler
                self._hidden_gems_handler = HiddenGemsHandler()
                self.hidden_gems_available = True
                logger.info("Hidden gems handler re-enabled")
            except ImportError as e:
                logger.error(f"Cannot enable hidden gems handler: {e}")
        
        elif service_name == "price_filter" and not self.price_filter_available:
            try:
                from backend.services.price_filter_service import PriceFilterService
                self._price_filter = PriceFilterService()
                self.price_filter_available = True
                logger.info("Price filter service re-enabled")
            except ImportError as e:
                logger.error(f"Cannot enable price filter service: {e}")
        
        elif service_name == "map" and not self.map_available:
            try:
                from backend.services.map_visualization_engine import MapVisualizationEngine
                self._map_engine = MapVisualizationEngine()
                self.map_available = True
                logger.info("Map visualization engine re-enabled")
            except ImportError as e:
                logger.error(f"Cannot enable map visualization engine: {e}")
        
        else:
            logger.warning(f"Unknown service or already enabled: {service_name}")


# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================

def create_service_integrations(enable_services: bool = True) -> ServiceIntegrations:
    """
    Factory function to create a ServiceIntegrations instance.
    
    Args:
        enable_services: Whether to enable services
        
    Returns:
        ServiceIntegrations instance
    """
    return ServiceIntegrations(enable_services=enable_services)
