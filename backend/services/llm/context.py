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

Features resilience patterns:
- Circuit breakers for external services
- Graceful degradation when services fail
- Timeout management

Author: AI Istanbul Team
Date: November 2025
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
from .resilience import (
    CircuitBreaker, 
    CircuitBreakerError,
    RetryStrategy,
    TimeoutManager,
    GracefulDegradation
)

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
        map_service=None,
        service_manager=None,
        circuit_breakers=None,
        timeout_manager=None,
        retry_strategy=None
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
            service_manager: Service Manager with all local services
            circuit_breakers: Dict of circuit breakers for services
            timeout_manager: Timeout manager instance
            retry_strategy: Retry strategy for transient failures
        """
        self.db = db_connection
        self.rag_service = rag_service
        self.weather_service = weather_service
        self.events_service = events_service
        self.hidden_gems_service = hidden_gems_service
        self.map_service = map_service
        self.service_manager = service_manager  # Service Manager for local services
        self.circuit_breakers = circuit_breakers or {}
        self.timeout_manager = timeout_manager
        self.retry_strategy = retry_strategy or RetryStrategy(max_retries=2, base_delay=0.5)
        
        logger.info("✅ Context Builder initialized")
        if service_manager:
            logger.info("   📦 Service Manager available for enhanced context building")
    
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
        
        # Get RAG context with retry and circuit breaker
        if self.rag_service:
            try:
                rag_cb = self.circuit_breakers.get('rag')
                if rag_cb:
                    context['rag'] = await rag_cb.call(
                        self._get_rag_context_with_retry, query, language
                    )
                else:
                    context['rag'] = await self._get_rag_context_with_retry(query, language)
            except CircuitBreakerError:
                logger.warning("RAG circuit breaker is open, using fallback")
                context['rag'] = GracefulDegradation.get_fallback_context('rag').get('message', '')
            except Exception as e:
                logger.warning(f"RAG context failed: {e}")
                context['rag'] = GracefulDegradation.get_fallback_context('rag').get('message', '')
        
        # Get weather context with retry and circuit breaker
        if signals.get('needs_weather') and self.weather_service:
            try:
                weather_cb = self.circuit_breakers.get('weather')
                if weather_cb:
                    context['services']['weather'] = await weather_cb.call(
                        self._get_weather_context_with_retry, query
                    )
                else:
                    context['services']['weather'] = await self._get_weather_context_with_retry(query)
            except CircuitBreakerError:
                logger.warning("Weather circuit breaker is open, using fallback")
                context['services']['weather'] = GracefulDegradation.get_fallback_context('weather')
            except Exception as e:
                logger.warning(f"Weather context failed: {e}")
                context['services']['weather'] = GracefulDegradation.get_fallback_context('weather')
        
        # Get events context with retry and circuit breaker
        if signals.get('needs_events') and self.events_service:
            try:
                events_cb = self.circuit_breakers.get('events')
                if events_cb:
                    context['services']['events'] = await events_cb.call(
                        self._get_events_context_with_retry
                    )
                else:
                    context['services']['events'] = await self._get_events_context_with_retry()
            except CircuitBreakerError:
                logger.warning("Events circuit breaker is open, using fallback")
                context['services']['events'] = GracefulDegradation.get_fallback_context('events')
            except Exception as e:
                logger.warning(f"Events context failed: {e}")
                context['services']['events'] = GracefulDegradation.get_fallback_context('events')
        
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
        """Get restaurant data from database with retry and timeout protection."""
        try:
            # Database query with circuit breaker and timeout
            async def _query_db():
                cursor = await self.db.execute("""
                    SELECT name, cuisine, district, price_range, rating
                    FROM restaurants
                    WHERE language = ?
                    LIMIT 5
                """, (language,))
                return await cursor.fetchall()
            
            # Apply timeout if timeout manager available
            if self.timeout_manager:
                rows = await self.timeout_manager.execute(
                    'database_query',
                    _query_db,
                    timeout=3.0
                )
            else:
                rows = await _query_db()
            
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
        """Get attraction data from database with retry and timeout protection."""
        try:
            # Database query with timeout
            async def _query_db():
                cursor = await self.db.execute("""
                    SELECT name, category, district, description
                    FROM attractions
                    WHERE language = ?
                    LIMIT 5
                """, (language,))
                return await cursor.fetchall()
            
            # Apply timeout if timeout manager available
            if self.timeout_manager:
                rows = await self.timeout_manager.execute(
                    'database_query',
                    _query_db,
                    timeout=3.0
                )
            else:
                rows = await _query_db()
            
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
        """Get RAG context from embeddings with circuit breaker protection."""
        if not self.rag_service:
            return ""
        
        try:
            # Use circuit breaker if available
            if 'rag' in self.circuit_breakers:
                async def _search():
                    return await self.rag_service.search(query, language=language, top_k=3)
                
                results = await self.circuit_breakers['rag'].call(_search)
            else:
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
            # Return graceful degradation message
            from .resilience import GracefulDegradation
            fallback = GracefulDegradation.get_fallback_context('rag')
            return fallback.get('message', '')
    
    async def _get_weather_context(self, query: str) -> str:
        """Get weather context with circuit breaker protection."""
        if not self.weather_service:
            return ""
        
        try:
            # Weather service is sync, so we call it directly
            weather = self.weather_service.get_current_weather("Istanbul")
            
            return (
                f"Current weather in Istanbul: {weather.get('condition', 'Unknown')}, "
                f"{weather.get('temperature', '?')}°C. {weather.get('description', '')}"
            )
        
        except Exception as e:
            logger.error(f"Weather service failed: {e}")
            # Return graceful degradation message
            from .resilience import GracefulDegradation
            fallback = GracefulDegradation.get_fallback_context('weather')
            return fallback.get('weather_info', '')
    
    async def _get_events_context(self) -> str:
        """Get events context with circuit breaker protection."""
        if not self.events_service:
            return ""
        
        try:
            # Use circuit breaker if available
            if 'events' in self.circuit_breakers:
                async def _get_events():
                    return await self.events_service.get_upcoming_events(limit=5)
                
                events = await self.circuit_breakers['events'].call(_get_events)
            else:
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
            # Return graceful degradation message
            from .resilience import GracefulDegradation
            fallback = GracefulDegradation.get_fallback_context('events')
            return fallback.get('message', '')
    
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
    
    async def _get_rag_context_with_retry(self, query: str, language: str) -> str:
        """Get RAG context with retry logic for transient failures."""
        async def _get_rag():
            if self.timeout_manager:
                return await self.timeout_manager.execute(
                    'rag_search',
                    self._get_rag_context,
                    query,
                    language,
                    timeout=4.0
                )
            return await self._get_rag_context(query, language)
        
        return await self.retry_strategy.execute(
            _get_rag,
            retryable_exceptions=[ConnectionError, TimeoutError, asyncio.TimeoutError]
        )
    
    async def _get_weather_context_with_retry(self, query: str) -> dict:
        """Get weather context with retry logic for transient failures."""
        async def _get_weather():
            if self.timeout_manager:
                return await self.timeout_manager.execute(
                    'weather_api',
                    self._get_weather_context,
                    query,
                    timeout=2.0
                )
            return await self._get_weather_context(query)
        
        return await self.retry_strategy.execute(
            _get_weather,
            retryable_exceptions=[ConnectionError, TimeoutError, asyncio.TimeoutError]
        )
    
    async def _get_events_context_with_retry(self) -> dict:
        """Get events context with retry logic for transient failures."""
        async def _get_events():
            if self.timeout_manager:
                return await self.timeout_manager.execute(
                    'events_api',
                    self._get_events_context,
                    timeout=2.0
                )
            return await self._get_events_context()
        
        return await self.retry_strategy.execute(
            _get_events,
            retryable_exceptions=[ConnectionError, TimeoutError, asyncio.TimeoutError]
        )
