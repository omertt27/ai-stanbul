"""
LLM Context Builder
Builds rich context from services to enhance LLM responses with real-time Istanbul data.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from services.llm_service_registry import get_service_registry, ServiceCategory
from services.location_based_context_enhancer import get_location_based_enhancer

logger = logging.getLogger(__name__)


class LLMContextBuilder:
    """Builds context from services for LLM queries"""
    
    def __init__(self):
        self.service_registry = get_service_registry()
        self.location_enhancer = get_location_based_enhancer()
        
        # Intent to service mapping
        self.intent_service_map = {
            # Restaurant intents
            "restaurant_recommendation": ["get_restaurants"],
            "find_restaurants": ["get_restaurants"],
            "restaurant_search": ["get_restaurants"],
            
            # Transportation intents
            "metro_route": ["get_metro_route"],
            "bus_route": ["get_bus_routes"],
            "ferry_schedule": ["get_ferry_schedule"],
            "transportation": ["get_metro_route", "get_bus_routes"],
            "route_planning": ["get_walking_directions", "get_metro_route"],
            
            # Navigation intents
            "directions": ["get_walking_directions"],
            "nearby_places": ["get_nearby_pois"],
            "find_location": ["get_nearby_pois"],
            
            # Weather intents
            "weather": ["get_weather"],
            "weather_forecast": ["get_weather"],
            
            # Attraction intents
            "attractions": ["get_attractions"],
            "museums": ["get_attractions"],
            "places_to_visit": ["get_attractions"],
            
            # Local tips intents
            "hidden_gems": ["get_hidden_gems"],
            "neighborhood_guide": ["get_neighborhood_guide"],
            "local_tips": ["get_hidden_gems"],
            
            # Events intents
            "events": ["get_events"],
            "whats_happening": ["get_events"],
        }
    
    async def build_context(
        self,
        query: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        user_location: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Build context for LLM by calling relevant services
        
        Args:
            query: User query
            intent: Detected intent type
            entities: Extracted entities (location, cuisine, etc.)
            user_location: User GPS coordinates {lat, lon}
            
        Returns:
            Context dict with service data
        """
        logger.info(f"🔍 Building context for intent: {intent}")
        
        context = {
            "query": query,
            "intent": intent,
            "entities": entities or {},
            "user_location": user_location,
            "service_data": {},
            "timestamp": datetime.now().isoformat()
        }
        
        if not intent:
            logger.warning("No intent provided, skipping service calls")
            return context
        
        # Get services to call based on intent
        service_names = self.intent_service_map.get(intent, [])
        
        if not service_names:
            logger.info(f"No services mapped for intent: {intent}")
            return context
        
        # Call services and collect data
        for service_name in service_names:
            service = self.service_registry.get_service(service_name)
            if not service:
                logger.warning(f"Service not found: {service_name}")
                continue
            
            try:
                # Build parameters for service call
                params = self._build_service_params(
                    service_name,
                    entities or {},
                    user_location
                )
                
                logger.info(f"📞 Calling service: {service_name} with params: {params}")
                
                # Call service handler
                result = await service.handler(**params)
                
                if result.get("status") == "success":
                    context["service_data"][service_name] = result.get("data")
                    logger.info(f"✅ {service_name} returned data")
                else:
                    logger.error(f"❌ {service_name} failed: {result.get('message')}")
                    
            except Exception as e:
                logger.error(f"Error calling {service_name}: {e}")
        
        # 🌟 ENHANCED: Add location-based enrichment with hidden gems
        try:
            logger.info("🗺️ Enriching context with location-based data...")
            context = await self.location_enhancer.enhance_context(
                query=query,
                base_context=context,
                intent=intent
            )
            logger.info("✅ Location-based enrichment complete")
        except Exception as e:
            logger.error(f"⚠️ Location enrichment failed: {e}")
        
        return context
    
    def _build_service_params(
        self,
        service_name: str,
        entities: Dict[str, Any],
        user_location: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Build parameters for service call from entities"""
        params = {}
        
        # Add user location if available
        if user_location:
            params["user_location"] = user_location
        
        # Map entities to service parameters based on service type
        if service_name == "get_restaurants":
            if "cuisine" in entities:
                params["cuisine"] = entities["cuisine"]
            if "district" in entities or "location" in entities:
                params["district"] = entities.get("district") or entities.get("location")
            if "budget" in entities:
                params["budget"] = entities["budget"]
            if "dietary" in entities:
                params["dietary"] = entities["dietary"]
        
        elif service_name == "get_metro_route":
            if "from_location" in entities:
                params["from_location"] = entities["from_location"]
            if "to_location" in entities:
                params["to_location"] = entities["to_location"]
        
        elif service_name == "get_bus_routes":
            if "from_location" in entities:
                params["from_location"] = entities["from_location"]
            if "to_location" in entities:
                params["to_location"] = entities["to_location"]
        
        elif service_name == "get_walking_directions":
            if "from_location" in entities:
                params["from_location"] = entities["from_location"]
            if "to_location" in entities:
                params["to_location"] = entities["to_location"]
        
        elif service_name == "get_nearby_pois":
            if "category" in entities:
                params["category"] = entities["category"]
            if "radius" in entities:
                params["radius"] = entities["radius"]
        
        elif service_name == "get_weather":
            if "district" in entities or "location" in entities:
                params["district"] = entities.get("district") or entities.get("location")
            if "forecast_days" in entities:
                params["forecast_days"] = entities["forecast_days"]
        
        elif service_name == "get_attractions":
            if "category" in entities:
                params["category"] = entities["category"]
            if "district" in entities or "location" in entities:
                params["district"] = entities.get("district") or entities.get("location")
        
        elif service_name == "get_hidden_gems":
            if "district" in entities or "location" in entities:
                params["district"] = entities.get("district") or entities.get("location")
            if "category" in entities:
                params["category"] = entities["category"]
        
        elif service_name == "get_neighborhood_guide":
            if "district" in entities or "location" in entities:
                params["district"] = entities.get("district") or entities.get("location")
        
        elif service_name == "get_events":
            if "date" in entities:
                params["date"] = entities["date"]
            if "category" in entities:
                params["category"] = entities["category"]
        
        return params
    
    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """
        Format context data into a readable string for LLM
        
        Args:
            context: Context dict from build_context()
            
        Returns:
            Formatted context string
        """
        formatted_parts = []
        
        # 🌟 ENHANCED: Add location-based enrichment first
        if context.get("location_enrichment"):
            location_context = self.location_enhancer.format_enriched_context_for_llm(context)
            if location_context:
                formatted_parts.append(location_context)
                logger.info("✅ Added location enrichment to LLM context")
        
        # Add service data
        if context.get("service_data"):
            for service_name, data in context["service_data"].items():
                if not data:
                    continue
                
                formatted_parts.append(self._format_service_data(service_name, data))
        
        if formatted_parts:
            return "\n\n".join(formatted_parts)
        
        return ""
    
    def _format_service_data(self, service_name: str, data: Any) -> str:
        """Format service data for LLM consumption"""
        
        if service_name == "get_restaurants":
            return self._format_restaurant_data(data)
        elif service_name in ["get_metro_route", "get_bus_routes"]:
            return self._format_transportation_data(data)
        elif service_name == "get_walking_directions":
            return self._format_directions_data(data)
        elif service_name == "get_nearby_pois":
            return self._format_poi_data(data)
        elif service_name == "get_weather":
            return self._format_weather_data(data)
        elif service_name == "get_attractions":
            return self._format_attraction_data(data)
        elif service_name in ["get_hidden_gems", "get_neighborhood_guide"]:
            return self._format_local_tips_data(data)
        elif service_name == "get_events":
            return self._format_events_data(data)
        else:
            return f"Data from {service_name}: {data}"
    
    def _format_restaurant_data(self, data: Any) -> str:
        """Format restaurant data"""
        if isinstance(data, list):
            formatted = "**Available Restaurants:**\n"
            for idx, restaurant in enumerate(data[:5], 1):  # Top 5
                name = restaurant.get("name", "Unknown")
                rating = restaurant.get("rating", "N/A")
                cuisine = restaurant.get("cuisine", "")
                price_range = restaurant.get("price_range", "")
                district = restaurant.get("district", "")
                
                formatted += f"{idx}. **{name}** - {rating}★\n"
                if cuisine:
                    formatted += f"   Cuisine: {cuisine}\n"
                if price_range:
                    formatted += f"   Price: {price_range}\n"
                if district:
                    formatted += f"   Location: {district}\n"
            return formatted
        return f"Restaurant data: {data}"
    
    def _format_transportation_data(self, data: Any) -> str:
        """Format transportation route data"""
        if isinstance(data, dict):
            formatted = "**Transportation Options:**\n"
            
            if "routes" in data:
                for idx, route in enumerate(data["routes"][:3], 1):
                    formatted += f"{idx}. {route.get('description', 'Route')}\n"
                    formatted += f"   Duration: {route.get('duration', 'N/A')} min\n"
                    formatted += f"   Transfers: {route.get('transfers', 0)}\n"
            
            return formatted
        return f"Transportation data: {data}"
    
    def _format_directions_data(self, data: Any) -> str:
        """Format walking directions data"""
        if isinstance(data, dict):
            formatted = "**Walking Directions:**\n"
            formatted += f"Distance: {data.get('distance', 'N/A')} km\n"
            formatted += f"Duration: {data.get('duration', 'N/A')} min\n"
            
            if "steps" in data:
                formatted += "\nSteps:\n"
                for idx, step in enumerate(data["steps"][:5], 1):
                    formatted += f"{idx}. {step.get('instruction', 'Continue')}\n"
            
            return formatted
        return f"Directions data: {data}"
    
    def _format_poi_data(self, data: Any) -> str:
        """Format POI data"""
        if isinstance(data, list):
            formatted = "**Nearby Places:**\n"
            for idx, poi in enumerate(data[:5], 1):
                name = poi.get("name", "Unknown")
                category = poi.get("category", "")
                distance = poi.get("distance", "")
                
                formatted += f"{idx}. **{name}**"
                if category:
                    formatted += f" ({category})"
                if distance:
                    formatted += f" - {distance}m away"
                formatted += "\n"
            
            return formatted
        return f"POI data: {data}"
    
    def _format_weather_data(self, data: Any) -> str:
        """Format weather data"""
        if isinstance(data, dict):
            formatted = "**Current Weather:**\n"
            formatted += f"Temperature: {data.get('temperature', 'N/A')}°C\n"
            formatted += f"Conditions: {data.get('conditions', 'N/A')}\n"
            formatted += f"Humidity: {data.get('humidity', 'N/A')}%\n"
            
            if "forecast" in data:
                formatted += "\n**Forecast:**\n"
                for day in data["forecast"][:3]:
                    formatted += f"- {day.get('date')}: {day.get('temp')}°C, {day.get('conditions')}\n"
            
            return formatted
        return f"Weather data: {data}"
    
    def _format_attraction_data(self, data: Any) -> str:
        """Format attraction data"""
        if isinstance(data, list):
            formatted = "**Attractions:**\n"
            for idx, attraction in enumerate(data[:5], 1):
                name = attraction.get("name", "Unknown")
                category = attraction.get("category", "")
                district = attraction.get("district", "")
                
                formatted += f"{idx}. **{name}**"
                if category:
                    formatted += f" ({category})"
                if district:
                    formatted += f" - {district}"
                formatted += "\n"
            
            return formatted
        return f"Attraction data: {data}"
    
    def _format_local_tips_data(self, data: Any) -> str:
        """Format local tips/hidden gems data"""
        if isinstance(data, dict) or isinstance(data, list):
            formatted = "**Local Insights:**\n"
            
            if isinstance(data, dict) and "tips" in data:
                for tip in data["tips"][:5]:
                    formatted += f"• {tip}\n"
            elif isinstance(data, list):
                for tip in data[:5]:
                    formatted += f"• {tip.get('description', tip)}\n"
            
            return formatted
        return f"Local tips data: {data}"
    
    def _format_events_data(self, data: Any) -> str:
        """Format events data"""
        if isinstance(data, list):
            formatted = "**Upcoming Events:**\n"
            for idx, event in enumerate(data[:5], 1):
                name = event.get("name", "Unknown")
                date = event.get("date", "")
                venue = event.get("venue", "")
                
                formatted += f"{idx}. **{name}**"
                if date:
                    formatted += f" - {date}"
                if venue:
                    formatted += f" at {venue}"
                formatted += "\n"
            
            return formatted
        return f"Events data: {data}"


# Global instance
_context_builder: Optional[LLMContextBuilder] = None


def get_context_builder() -> LLMContextBuilder:
    """Get or create global context builder"""
    global _context_builder
    if _context_builder is None:
        _context_builder = LLMContextBuilder()
    return _context_builder
