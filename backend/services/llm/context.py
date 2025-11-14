"""
context.py - Context Building System

Smart context building that fetches relevant information based on detected signals.

Context Sources:
- Database: Restaurants, attractions, neighborhoods
- RAG: Document embeddings and similarity search
- Weather Service: Current conditions and forecasts
- Events Service: Cultural events and activities
- Hidden Gems: Off-the-beaten-path locations
- Map Service: Visual maps and routing

Author: AI Istanbul Team
Date: November 2025
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Smart context building system.
    
    Builds context based on detected signals to minimize unnecessary
    database queries and service calls.
    """
    
    def __init__(
        self,
        db_connection,
        rag_service=None,
        weather_service=None,
        events_service=None,
        hidden_gems_service=None,
        map_service=None
    ):
        """
        Initialize context builder.
        
        Args:
            db_connection: Database connection
            rag_service: RAG service for embeddings
            weather_service: Weather service
            events_service: Events service
            hidden_gems_service: Hidden gems service
            map_service: Map generation service
        """
        self.db = db_connection
        self.rag_service = rag_service
        self.weather_service = weather_service
        self.events_service = events_service
        self.hidden_gems_service = hidden_gems_service
        self.map_service = map_service
        
        logger.info("✅ Context Builder initialized")
    
    async def build_context(
        self,
        query: str,
        signals: Dict[str, bool],
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Build smart context based on detected signals.
        
        Args:
            query: User query
            signals: Detected signals
            user_location: User GPS location
            language: Language code
            
        Returns:
            Dict with:
            - database: Database context (str)
            - rag: RAG context (str)
            - services: Service context dict
            - map_data: Map visualization data (if applicable)
        """
        context = {
            'database': '',
            'rag': '',
            'services': {},
            'map_data': None
        }
        
        # Build database context
        if self._needs_database_context(signals):
            context['database'] = await self._build_database_context(
                query=query,
                signals=signals,
                user_location=user_location,
                language=language
            )
        
        # Get RAG context
        if self.rag_service:
            try:
                context['rag'] = await self._get_rag_context(query, language)
            except Exception as e:
                logger.warning(f"RAG context failed: {e}")
        
        # Get weather context
        if signals.get('needs_weather') and self.weather_service:
            try:
                context['services']['weather'] = await self._get_weather_context(query)
            except Exception as e:
                logger.warning(f"Weather context failed: {e}")
        
        # Get events context
        if signals.get('needs_events') and self.events_service:
            try:
                context['services']['events'] = await self._get_events_context()
            except Exception as e:
                logger.warning(f"Events context failed: {e}")
        
        # Get hidden gems context
        if signals.get('needs_hidden_gems') and self.hidden_gems_service:
            try:
                context['services']['hidden_gems'] = await self._get_hidden_gems_context(query)
            except Exception as e:
                logger.warning(f"Hidden gems context failed: {e}")
        
        # Generate map visualization
        if (signals.get('needs_map') or signals.get('needs_gps_routing')) and self.map_service:
            try:
                context['map_data'] = await self._generate_map(
                    query=query,
                    signals=signals,
                    user_location=user_location,
                    language=language
                )
            except Exception as e:
                logger.warning(f"Map generation failed: {e}")
        
        return context
    
    def _needs_database_context(self, signals: Dict[str, bool]) -> bool:
        """Check if database context is needed."""
        db_signals = [
            'needs_restaurant',
            'needs_attraction',
            'needs_neighborhood',
            'needs_transportation'
        ]
        return any(signals.get(sig, False) for sig in db_signals)
    
    async def _build_database_context(
        self,
        query: str,
        signals: Dict[str, bool],
        user_location: Optional[Dict[str, float]],
        language: str
    ) -> str:
        """
        Build database context based on signals.
        
        Args:
            query: User query
            signals: Detected signals
            user_location: User GPS location
            language: Language code
            
        Returns:
            Formatted database context string
        """
        context_parts = []
        
        # Get restaurant context
        if signals.get('needs_restaurant'):
            restaurants = await self._get_restaurants(query, user_location, language)
            if restaurants:
                context_parts.append("=== RESTAURANTS ===")
                context_parts.append(restaurants)
        
        # Get attraction context
        if signals.get('needs_attraction'):
            attractions = await self._get_attractions(query, user_location, language)
            if attractions:
                context_parts.append("=== ATTRACTIONS ===")
                context_parts.append(attractions)
        
        # Get neighborhood context
        if signals.get('needs_neighborhood'):
            neighborhoods = await self._get_neighborhoods(query, language)
            if neighborhoods:
                context_parts.append("=== NEIGHBORHOODS ===")
                context_parts.append(neighborhoods)
        
        # Get transportation context
        if signals.get('needs_transportation'):
            transport = await self._get_transportation(query, language)
            if transport:
                context_parts.append("=== TRANSPORTATION ===")
                context_parts.append(transport)
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    async def _get_restaurants(
        self,
        query: str,
        user_location: Optional[Dict[str, float]],
        language: str
    ) -> str:
        """Get restaurant data from database."""
        try:
            # TODO: Implement actual database query
            # This is a placeholder
            cursor = await self.db.execute("""
                SELECT name, cuisine, district, price_range, rating
                FROM restaurants
                WHERE language = ?
                LIMIT 5
            """, (language,))
            
            rows = await cursor.fetchall()
            
            if not rows:
                return ""
            
            # Format results
            results = []
            for row in rows:
                results.append(
                    f"- {row[0]}: {row[1]} cuisine in {row[2]}, "
                    f"Price: {row[3]}, Rating: {row[4]}/5"
                )
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Failed to get restaurants: {e}")
            return ""
    
    async def _get_attractions(
        self,
        query: str,
        user_location: Optional[Dict[str, float]],
        language: str
    ) -> str:
        """Get attraction data from database."""
        try:
            # TODO: Implement actual database query
            cursor = await self.db.execute("""
                SELECT name, category, district, description
                FROM attractions
                WHERE language = ?
                LIMIT 5
            """, (language,))
            
            rows = await cursor.fetchall()
            
            if not rows:
                return ""
            
            # Format results
            results = []
            for row in rows:
                results.append(
                    f"- {row[0]} ({row[1]}): Located in {row[2]}. {row[3][:100]}..."
                )
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Failed to get attractions: {e}")
            return ""
    
    async def _get_neighborhoods(self, query: str, language: str) -> str:
        """Get neighborhood data from database."""
        try:
            # TODO: Implement actual database query
            return "Beyoglu: Historic district known for nightlife and culture..."
        except Exception as e:
            logger.error(f"Failed to get neighborhoods: {e}")
            return ""
    
    async def _get_transportation(self, query: str, language: str) -> str:
        """Get transportation data from database."""
        try:
            # TODO: Implement actual database query
            return "Metro M2 connects to Taksim and Sisli..."
        except Exception as e:
            logger.error(f"Failed to get transportation: {e}")
            return ""
    
    async def _get_rag_context(self, query: str, language: str) -> str:
        """Get RAG context from embeddings."""
        if not self.rag_service:
            return ""
        
        try:
            results = await self.rag_service.search(query, language=language, top_k=3)
            
            if not results:
                return ""
            
            # Format RAG results
            context_parts = []
            for result in results:
                context_parts.append(f"[Score: {result.get('score', 0):.2f}] {result.get('text', '')[:200]}...")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return ""
    
    async def _get_weather_context(self, query: str) -> str:
        """Get weather context."""
        if not self.weather_service:
            return ""
        
        try:
            weather = await self.weather_service.get_current_weather("Istanbul")
            
            return (
                f"Current weather in Istanbul: {weather.get('condition', 'Unknown')}, "
                f"{weather.get('temperature', '?')}°C. {weather.get('description', '')}"
            )
            
        except Exception as e:
            logger.error(f"Weather service failed: {e}")
            return ""
    
    async def _get_events_context(self) -> str:
        """Get events context."""
        if not self.events_service:
            return ""
        
        try:
            events = await self.events_service.get_upcoming_events(limit=5)
            
            if not events:
                return ""
            
            # Format events
            event_list = []
            for event in events:
                event_list.append(
                    f"- {event.get('name', 'Unknown')}: {event.get('date', 'TBA')} "
                    f"at {event.get('venue', 'Various locations')}"
                )
            
            return "\n".join(event_list)
            
        except Exception as e:
            logger.error(f"Events service failed: {e}")
            return ""
    
    async def _get_hidden_gems_context(self, query: str) -> str:
        """Get hidden gems context."""
        if not self.hidden_gems_service:
            return ""
        
        try:
            gems = await self.hidden_gems_service.get_recommendations(query, limit=3)
            
            if not gems:
                return ""
            
            # Format hidden gems
            gem_list = []
            for gem in gems:
                gem_list.append(
                    f"- {gem.get('name', 'Unknown')}: {gem.get('description', '')} "
                    f"(District: {gem.get('district', 'Unknown')})"
                )
            
            return "\n".join(gem_list)
            
        except Exception as e:
            logger.error(f"Hidden gems service failed: {e}")
            return ""
    
    async def _generate_map(
        self,
        query: str,
        signals: Dict[str, bool],
        user_location: Optional[Dict[str, float]],
        language: str
    ) -> Optional[Dict[str, Any]]:
        """Generate map visualization."""
        if not self.map_service:
            return None
        
        try:
            map_data = await self.map_service.generate_map(
                query=query,
                user_location=user_location,
                language=language,
                routing=signals.get('needs_gps_routing', False)
            )
            
            return map_data
            
        except Exception as e:
            logger.error(f"Map generation failed: {e}")
            return None
