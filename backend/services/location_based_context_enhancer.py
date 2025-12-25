"""
Location-Based Context Enhancer for LLM
Automatically enriches LLM responses with hidden gems and local insights when users mention districts
"""

import logging
from typing import Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)


class LocationBasedContextEnhancer:
    """
    Enhances LLM context with location-based data from services
    Automatically adds hidden gems when districts are mentioned
    """
    
    def __init__(self):
        self.district_patterns = self._init_district_patterns()
        self.hidden_gems_service = None
        self.events_service = None
        self.restaurant_service = None
        self.attractions_service = None
        
        self._load_services()
    
    def _init_district_patterns(self) -> Dict[str, List[str]]:
        """Initialize district detection patterns"""
        return {
            'Sultanahmet': ['sultanahmet', 'old city', 'historic peninsula', 'blue mosque area'],
            'Beyoğlu': ['beyoglu', 'beyoğlu', 'istiklal', 'galata', 'taksim', 'cihangir', 'karakoy', 'karaköy'],
            'Kadıköy': ['kadikoy', 'kadıköy', 'asian side', 'moda', 'fenerbahce', 'fenerbahçe'],
            'Beşiktaş': ['besiktas', 'beşiktaş', 'ortakoy', 'ortaköy', 'bebek', 'arnavutkoy', 'arnavutköy'],
            'Üsküdar': ['uskudar', 'üsküdar', 'kuzguncuk', 'çengelköy', 'cengelkoy'],
            'Fatih': ['fatih', 'balat', 'fener', 'eminonu', 'eminönü', 'kumkapi', 'kumkapı'],
            'Sarıyer': ['sariyer', 'sarıyer', 'emirgan', 'istinye', 'tarabya', 'yeniköy', 'yenikoy'],
            'Şişli': ['sisli', 'şişli', 'nisantasi', 'nişantaşı', 'osmanbey'],
            'Bakırköy': ['bakirkoy', 'bakırköy', 'yesilkoy', 'yeşilköy', 'atakoy', 'ataköy'],
            'Eyüp': ['eyup', 'eyüp', 'pierre loti'],
            'Beylikdüzü': ['beylikduzu', 'beylikdüzü'],
            'Pendik': ['pendik'],
            'Maltepe': ['maltepe'],
            'Kartal': ['kartal'],
            'Bağcılar': ['bagcilar', 'bağcılar'],
            'Zeytinburnu': ['zeytinburnu'],
        }
    
    def _load_services(self):
        """Load service instances"""
        try:
            from services.hidden_gems_service import HiddenGemsService
            self.hidden_gems_service = HiddenGemsService()
            logger.info("✅ Hidden Gems Service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load Hidden Gems Service: {e}")
        
        try:
            from services.events_service import EventsService
            self.events_service = EventsService()
            logger.info("✅ Events Service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load Events Service: {e}")
        
        try:
            from services.restaurant_database_service import RestaurantDatabaseService
            self.restaurant_service = RestaurantDatabaseService()
            logger.info("✅ Restaurant Service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load Restaurant Service: {e}")
        
        try:
            from services.enhanced_attractions_service import EnhancedAttractionsService
            self.attractions_service = EnhancedAttractionsService()
            logger.info("✅ Attractions Service loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load Attractions Service: {e}")
    
    def detect_districts(self, query: str) -> List[str]:
        """
        Detect districts mentioned in the query
        
        Args:
            query: User query text
            
        Returns:
            List of detected district names
        """
        query_lower = query.lower()
        detected = []
        
        for district, patterns in self.district_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected.append(district)
        
        return detected
    
    def should_add_hidden_gems(self, query: str, intent: Optional[str] = None) -> bool:
        """
        Determine if hidden gems should be added to context
        
        Args:
            query: User query
            intent: Detected intent
            
        Returns:
            True if hidden gems should be included
        """
        query_lower = query.lower()
        
        # Always add for explicit hidden gems queries
        hidden_gem_keywords = [
            'hidden gem', 'secret', 'local favorite', 'off the beaten',
            'undiscovered', 'insider', 'gizli', 'saklı'
        ]
        if any(keyword in query_lower for keyword in hidden_gem_keywords):
            return True
        
        # Add when districts are mentioned
        if self.detect_districts(query):
            return True
        
        # Add for certain intents
        gem_relevant_intents = [
            'explore', 'visit', 'see', 'go to', 'discover',
            'what to do', 'things to do', 'recommendations'
        ]
        if intent and any(keyword in intent.lower() for keyword in gem_relevant_intents):
            return True
        
        # Add for general location queries
        location_keywords = ['where', 'what', 'neighborhood', 'district', 'area']
        if any(keyword in query_lower for keyword in location_keywords):
            return True
        
        return False
    
    async def enhance_context(
        self,
        query: str,
        base_context: Dict[str, Any],
        intent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance context with location-based data
        
        Args:
            query: User query
            base_context: Base context from other sources
            intent: Detected intent
            
        Returns:
            Enhanced context with service data
        """
        enhanced = base_context.copy()
        enhanced['location_enrichment'] = {}
        
        # Detect districts
        districts = self.detect_districts(query)
        if districts:
            enhanced['location_enrichment']['detected_districts'] = districts
            logger.info(f"📍 Detected districts: {districts}")
        
        # Add hidden gems if appropriate
        if self.should_add_hidden_gems(query, intent):
            hidden_gems_data = await self._get_hidden_gems_for_query(query, districts)
            if hidden_gems_data:
                enhanced['location_enrichment']['hidden_gems'] = hidden_gems_data
                logger.info(f"💎 Added {len(hidden_gems_data)} hidden gems to context")
        
        # Add district-specific events
        if districts and self.events_service:
            events_data = await self._get_events_for_districts(districts)
            if events_data:
                enhanced['location_enrichment']['events'] = events_data
                logger.info(f"🎭 Added {len(events_data)} events to context")
        
        # Add district-specific restaurants for food queries
        if self._is_food_query(query) and districts:
            restaurant_data = await self._get_restaurants_for_districts(districts)
            if restaurant_data:
                enhanced['location_enrichment']['restaurants'] = restaurant_data
                logger.info(f"🍽️ Added {len(restaurant_data)} restaurants to context")
        
        # Add attractions for sightseeing queries
        if self._is_sightseeing_query(query) and districts:
            attractions_data = await self._get_attractions_for_districts(districts)
            if attractions_data:
                enhanced['location_enrichment']['attractions'] = attractions_data
                logger.info(f"🏛️ Added {len(attractions_data)} attractions to context")
        
        return enhanced
    
    async def _get_hidden_gems_for_query(
        self,
        query: str,
        districts: List[str]
    ) -> List[Dict[str, Any]]:
        """Get hidden gems relevant to the query"""
        if not self.hidden_gems_service:
            return []
        
        try:
            # Parse query for hidden gems
            parsed_query = self.hidden_gems_service.parse_hidden_gems_query(query)
            
            # Set district if detected
            if districts:
                parsed_query.district = districts[0]  # Use first detected district
            
            # Use filter_gems directly with parsed query (not search which expects string)
            gems = self.hidden_gems_service.filter_gems(parsed_query)
            
            # Format for LLM context
            formatted_gems = []
            for gem in gems[:5]:  # Top 5 gems
                formatted_gems.append({
                    'name': gem.get('name'),
                    'district': gem.get('district'),
                    'category': gem.get('category'),
                    'description': gem.get('description'),
                    'why_hidden': gem.get('why_hidden'),
                    'best_time': gem.get('best_time'),
                    'insider_tip': gem.get('insider_tip'),
                    'cost': gem.get('cost'),
                    'difficulty': gem.get('difficulty')
                })
            
            return formatted_gems
        except Exception as e:
            logger.error(f"Error getting hidden gems: {e}")
            return []
    
    async def _get_events_for_districts(
        self,
        districts: List[str]
    ) -> List[Dict[str, Any]]:
        """Get events for specified districts"""
        if not self.events_service:
            return []
        
        try:
            # Get events (check if method exists)
            if not hasattr(self.events_service, 'get_upcoming_events'):
                logger.warning("Events service doesn't have get_upcoming_events method")
                return []
            
            all_events = self.events_service.get_upcoming_events()
            
            # Filter by district if possible
            relevant_events = []
            for event in all_events[:5]:  # Top 5 events
                relevant_events.append({
                    'title': event.get('title'),
                    'venue': event.get('venue'),
                    'date': event.get('date'),
                    'description': event.get('description'),
                    'category': event.get('category')
                })
            
            return relevant_events
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    async def _get_restaurants_for_districts(
        self,
        districts: List[str]
    ) -> List[Dict[str, Any]]:
        """Get restaurants for specified districts"""
        if not self.restaurant_service:
            return []
        
        try:
            # Get restaurants by district
            all_restaurants = []
            for district in districts[:2]:  # Max 2 districts
                restaurants = self.restaurant_service.get_by_district(district)
                all_restaurants.extend(restaurants[:3])  # Top 3 per district
            
            # Format for LLM
            formatted = []
            for restaurant in all_restaurants:
                formatted.append({
                    'name': restaurant.get('name'),
                    'district': restaurant.get('district'),
                    'cuisine': restaurant.get('cuisine'),
                    'price_range': restaurant.get('price_range'),
                    'rating': restaurant.get('rating'),
                    'description': restaurant.get('description')
                })
            
            return formatted
        except Exception as e:
            logger.error(f"Error getting restaurants: {e}")
            return []
    
    async def _get_attractions_for_districts(
        self,
        districts: List[str]
    ) -> List[Dict[str, Any]]:
        """Get attractions for specified districts"""
        if not self.attractions_service:
            return []
        
        try:
            # Get attractions by district
            all_attractions = []
            for district in districts[:2]:  # Max 2 districts
                attractions = self.attractions_service.get_by_district(district)
                all_attractions.extend(attractions[:3])  # Top 3 per district
            
            # Format for LLM
            formatted = []
            for attraction in all_attractions:
                formatted.append({
                    'name': attraction.get('name'),
                    'district': attraction.get('district'),
                    'category': attraction.get('category'),
                    'description': attraction.get('description'),
                    'opening_hours': attraction.get('opening_hours'),
                    'entry_fee': attraction.get('entry_fee')
                })
            
            return formatted
        except Exception as e:
            logger.error(f"Error getting attractions: {e}")
            return []
    
    def _is_food_query(self, query: str) -> bool:
        """Check if query is about food/restaurants"""
        food_keywords = [
            'restaurant', 'cafe', 'coffee', 'food', 'eat', 'dining',
            'breakfast', 'lunch', 'dinner', 'cuisine', 'dish',
            'restoran', 'yemek', 'kahvalti', 'kahvaltı'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in food_keywords)
    
    def _is_sightseeing_query(self, query: str) -> bool:
        """Check if query is about sightseeing/attractions"""
        sightseeing_keywords = [
            'visit', 'see', 'attraction', 'museum', 'mosque', 'palace',
            'historical', 'sight', 'landmark', 'monument',
            'gez', 'gör', 'cami', 'saray', 'müze'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in sightseeing_keywords)
    
    def format_enriched_context_for_llm(
        self,
        enriched_context: Dict[str, Any]
    ) -> str:
        """
        Format enriched context into a string for LLM prompt
        
        Args:
            enriched_context: Enhanced context with service data
            
        Returns:
            Formatted string for LLM prompt
        """
        enrichment = enriched_context.get('location_enrichment', {})
        if not enrichment:
            return ""
        
        context_parts = []
        
        # Add detected districts
        if 'detected_districts' in enrichment:
            districts = enrichment['detected_districts']
            context_parts.append(f"🗺️ **Districts mentioned**: {', '.join(districts)}")
        
        # Add hidden gems
        if 'hidden_gems' in enrichment:
            gems = enrichment['hidden_gems']
            context_parts.append(f"\n💎 **Hidden Gems** ({len(gems)} found):")
            for gem in gems:
                context_parts.append(
                    f"\n• **{gem['name']}** ({gem['district']}) - {gem['category']}"
                    f"\n  {gem['description'][:150]}..."
                    f"\n  💡 {gem.get('insider_tip', 'N/A')[:100]}"
                )
        
        # Add events
        if 'events' in enrichment:
            events = enrichment['events']
            context_parts.append(f"\n\n🎭 **Upcoming Events** ({len(events)} found):")
            for event in events:
                context_parts.append(
                    f"\n• **{event['title']}** at {event.get('venue', 'TBA')}"
                    f"\n  {event.get('description', 'N/A')[:100]}"
                )
        
        # Add restaurants
        if 'restaurants' in enrichment:
            restaurants = enrichment['restaurants']
            context_parts.append(f"\n\n🍽️ **Restaurants** ({len(restaurants)} found):")
            for restaurant in restaurants:
                context_parts.append(
                    f"\n• **{restaurant['name']}** - {restaurant.get('cuisine', 'N/A')} "
                    f"({restaurant.get('price_range', 'N/A')})"
                )
        
        # Add attractions
        if 'attractions' in enrichment:
            attractions = enrichment['attractions']
            context_parts.append(f"\n\n🏛️ **Attractions** ({len(attractions)} found):")
            for attraction in attractions:
                context_parts.append(
                    f"\n• **{attraction['name']}** - {attraction.get('category', 'N/A')}"
                )
        
        return "\n".join(context_parts)


# Singleton instance
_enhancer_instance = None


def get_location_based_enhancer() -> LocationBasedContextEnhancer:
    """Get singleton instance of location-based enhancer"""
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = LocationBasedContextEnhancer()
    return _enhancer_instance
