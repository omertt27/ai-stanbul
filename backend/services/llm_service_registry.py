"""
LLM Service Registry
Central registry of all services available to the LLM for real-time data access.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceCategory(Enum):
    """Categories of services"""
    RESTAURANT = "restaurant"
    TRANSPORTATION = "transportation"
    ATTRACTION = "attraction"
    WEATHER = "weather"
    NAVIGATION = "navigation"
    LOCAL_TIPS = "local_tips"
    EVENTS = "events"
    ACCOMMODATION = "accommodation"


@dataclass
class ServiceDefinition:
    """Definition of a service callable by LLM"""
    name: str
    category: ServiceCategory
    description: str
    handler: Callable
    parameters: Dict[str, Any]
    requires_location: bool = False
    requires_date: bool = False
    
    
class LLMServiceRegistry:
    """Registry of services available to LLM"""
    
    def __init__(self):
        self.services: Dict[str, ServiceDefinition] = {}
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all available services"""
        logger.info("🔧 Initializing LLM Service Registry...")
        
        # We'll populate this with actual service handlers
        # For now, define the service interfaces
        
        # Restaurant services
        self.register_service(
            name="get_restaurants",
            category=ServiceCategory.RESTAURANT,
            description="Get restaurant recommendations based on cuisine, location, budget, and dietary needs",
            handler=self._get_restaurants_handler,
            parameters={
                "cuisine": {"type": "string", "optional": True, "description": "Cuisine type (e.g., Turkish, Italian, seafood)"},
                "district": {"type": "string", "optional": True, "description": "Istanbul district (e.g., Sultanahmet, Beyoğlu)"},
                "budget": {"type": "string", "optional": True, "description": "Budget level: cheap, medium, expensive"},
                "dietary": {"type": "string", "optional": True, "description": "Dietary restrictions (vegetarian, vegan, halal)"},
                "user_location": {"type": "object", "optional": True, "description": "User GPS coordinates"}
            },
            requires_location=True
        )
        
        # Transportation services
        self.register_service(
            name="get_metro_route",
            category=ServiceCategory.TRANSPORTATION,
            description="Get metro/tram routes between locations",
            handler=self._get_metro_route_handler,
            parameters={
                "from_location": {"type": "string", "required": True, "description": "Starting location"},
                "to_location": {"type": "string", "required": True, "description": "Destination"},
                "user_location": {"type": "object", "optional": True}
            },
            requires_location=False
        )
        
        self.register_service(
            name="get_bus_routes",
            category=ServiceCategory.TRANSPORTATION,
            description="Get bus routes and schedules",
            handler=self._get_bus_routes_handler,
            parameters={
                "from_location": {"type": "string", "required": True},
                "to_location": {"type": "string", "required": True},
                "user_location": {"type": "object", "optional": True}
            }
        )
        
        self.register_service(
            name="get_ferry_schedule",
            category=ServiceCategory.TRANSPORTATION,
            description="Get ferry schedules and routes",
            handler=self._get_ferry_schedule_handler,
            parameters={
                "from_pier": {"type": "string", "optional": True},
                "to_pier": {"type": "string", "optional": True}
            }
        )
        
        # Navigation services
        self.register_service(
            name="get_walking_directions",
            category=ServiceCategory.NAVIGATION,
            description="Get walking directions using OSRM",
            handler=self._get_walking_directions_handler,
            parameters={
                "from_location": {"type": "string", "required": True},
                "to_location": {"type": "string", "required": True},
                "user_location": {"type": "object", "optional": True}
            },
            requires_location=True
        )
        
        self.register_service(
            name="get_nearby_pois",
            category=ServiceCategory.NAVIGATION,
            description="Get nearby points of interest",
            handler=self._get_nearby_pois_handler,
            parameters={
                "category": {"type": "string", "optional": True, "description": "POI category (restaurant, museum, mosque)"},
                "radius": {"type": "number", "optional": True, "description": "Search radius in meters"},
                "user_location": {"type": "object", "required": True}
            },
            requires_location=True
        )
        
        # Weather services
        self.register_service(
            name="get_weather",
            category=ServiceCategory.WEATHER,
            description="Get current weather and forecast for Istanbul",
            handler=self._get_weather_handler,
            parameters={
                "district": {"type": "string", "optional": True},
                "forecast_days": {"type": "number", "optional": True, "description": "Number of days for forecast"}
            },
            requires_date=False
        )
        
        # Attraction services
        self.register_service(
            name="get_attractions",
            category=ServiceCategory.ATTRACTION,
            description="Get information about attractions, museums, and historical sites",
            handler=self._get_attractions_handler,
            parameters={
                "category": {"type": "string", "optional": True, "description": "Attraction type (museum, mosque, palace)"},
                "district": {"type": "string", "optional": True},
                "user_location": {"type": "object", "optional": True}
            }
        )
        
        # Local tips services
        self.register_service(
            name="get_hidden_gems",
            category=ServiceCategory.LOCAL_TIPS,
            description="Get hidden gems and local recommendations",
            handler=self._get_hidden_gems_handler,
            parameters={
                "district": {"type": "string", "optional": True},
                "category": {"type": "string", "optional": True}
            }
        )
        
        self.register_service(
            name="get_neighborhood_guide",
            category=ServiceCategory.LOCAL_TIPS,
            description="Get comprehensive guide for a neighborhood",
            handler=self._get_neighborhood_guide_handler,
            parameters={
                "district": {"type": "string", "required": True}
            }
        )
        
        # Events services
        self.register_service(
            name="get_events",
            category=ServiceCategory.EVENTS,
            description="Get cultural events, exhibitions, and concerts",
            handler=self._get_events_handler,
            parameters={
                "date": {"type": "string", "optional": True, "description": "Date or date range"},
                "category": {"type": "string", "optional": True, "description": "Event category"}
            },
            requires_date=True
        )
        
        logger.info(f"✅ Registered {len(self.services)} services for LLM access")
    
    def register_service(
        self,
        name: str,
        category: ServiceCategory,
        description: str,
        handler: Callable,
        parameters: Dict[str, Any],
        requires_location: bool = False,
        requires_date: bool = False
    ):
        """Register a service"""
        service = ServiceDefinition(
            name=name,
            category=category,
            description=description,
            handler=handler,
            parameters=parameters,
            requires_location=requires_location,
            requires_date=requires_date
        )
        self.services[name] = service
        logger.debug(f"Registered service: {name}")
    
    def get_service(self, name: str) -> Optional[ServiceDefinition]:
        """Get service definition by name"""
        return self.services.get(name)
    
    def get_services_by_category(self, category: ServiceCategory) -> List[ServiceDefinition]:
        """Get all services in a category"""
        return [s for s in self.services.values() if s.category == category]
    
    def list_all_services(self) -> List[Dict[str, Any]]:
        """List all available services"""
        return [
            {
                "name": service.name,
                "category": service.category.value,
                "description": service.description,
                "parameters": service.parameters
            }
            for service in self.services.values()
        ]
    
    # Service handlers (these will be implemented to call actual services)
    
    async def _get_restaurants_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for restaurant queries"""
        # This will be implemented to call actual restaurant service
        from services.restaurant_service import get_restaurants  # noqa
        
        try:
            results = await get_restaurants(**kwargs)
            return {"status": "success", "data": results}
        except Exception as e:
            logger.error(f"Restaurant service error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_metro_route_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for metro route queries"""
        from services.transportation_service import get_metro_route  # noqa
        
        try:
            results = await get_metro_route(**kwargs)
            return {"status": "success", "data": results}
        except Exception as e:
            logger.error(f"Metro route service error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_bus_routes_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for bus route queries"""
        from services.enhanced_bus_route_service import get_bus_routes  # noqa
        
        try:
            results = await get_bus_routes(**kwargs)
            return {"status": "success", "data": results}
        except Exception as e:
            logger.error(f"Bus route service error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_ferry_schedule_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for ferry schedule queries"""
        # Implement ferry service call
        return {"status": "success", "data": {"message": "Ferry service integration pending"}}
    
    async def _get_walking_directions_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for walking directions"""
        from services.osrm_routing_service import get_walking_route  # noqa
        
        try:
            results = await get_walking_route(**kwargs)
            return {"status": "success", "data": results}
        except Exception as e:
            logger.error(f"Walking directions error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_nearby_pois_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for nearby POIs"""
        from services.poi_database_service import get_nearby_pois  # noqa
        
        try:
            results = await get_nearby_pois(**kwargs)
            return {"status": "success", "data": results}
        except Exception as e:
            logger.error(f"POI service error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_weather_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for weather queries"""
        # from services.weather_service import get_weather  # noqa
        
        try:
            # results = await get_weather(**kwargs)
            return {"status": "success", "data": {"message": "Weather service integration pending"}}
        except Exception as e:
            logger.error(f"Weather service error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_attractions_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for attraction queries"""
        # from services.attraction_service import get_attractions  # noqa
        
        try:
            # results = await get_attractions(**kwargs)
            return {"status": "success", "data": {"message": "Attraction service integration pending"}}
        except Exception as e:
            logger.error(f"Attraction service error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_hidden_gems_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for hidden gems"""
        # from services.hidden_gems_service import get_hidden_gems  # noqa
        
        try:
            # results = await get_hidden_gems(**kwargs)
            return {"status": "success", "data": {"message": "Hidden gems service integration pending"}}
        except Exception as e:
            logger.error(f"Hidden gems service error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_neighborhood_guide_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for neighborhood guides"""
        # from services.neighborhood_service import get_neighborhood_guide  # noqa
        
        try:
            # results = await get_neighborhood_guide(**kwargs)
            return {"status": "success", "data": {"message": "Neighborhood service integration pending"}}
        except Exception as e:
            logger.error(f"Neighborhood service error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_events_handler(self, **kwargs) -> Dict[str, Any]:
        """Handler for events queries"""
        # from services.events_service import get_events  # noqa
        
        try:
            # results = await get_events(**kwargs)
            return {"status": "success", "data": {"message": "Events service integration pending"}}
        except Exception as e:
            logger.error(f"Events service error: {e}")
            return {"status": "error", "message": str(e)}


# Global registry instance
_service_registry: Optional[LLMServiceRegistry] = None


def get_service_registry() -> LLMServiceRegistry:
    """Get or create global service registry"""
    global _service_registry
    if _service_registry is None:
        _service_registry = LLMServiceRegistry()
    return _service_registry
