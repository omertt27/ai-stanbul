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
- Location-Based Enrichment: Auto-adds hidden gems for districts

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
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Import location-based enhancer
try:
    from services.location_based_context_enhancer import get_location_based_enhancer
    LOCATION_ENHANCER_AVAILABLE = True
except ImportError:
    LOCATION_ENHANCER_AVAILABLE = False
    logger.warning("⚠️ Location-based context enhancer not available")

# Import industry-level transportation RAG system
try:
    from backend.services.transportation_rag_system import get_transportation_rag
    TRANSPORTATION_RAG_AVAILABLE = True
    logger.info("✅ Industry-level Transportation RAG system available")
except ImportError:
    try:
        from services.transportation_rag_system import get_transportation_rag
        TRANSPORTATION_RAG_AVAILABLE = True
        logger.info("✅ Industry-level Transportation RAG system available")
    except ImportError:
        TRANSPORTATION_RAG_AVAILABLE = False
        logger.warning("⚠️ Transportation RAG system not available")

# Import Istanbul Knowledge RAG system (neighborhoods, food, attractions, scams, etc.)
try:
    from backend.services.istanbul_knowledge_rag import get_knowledge_rag
    KNOWLEDGE_RAG_AVAILABLE = True
    logger.info("✅ Istanbul Knowledge RAG system available")
except ImportError:
    try:
        from services.istanbul_knowledge_rag import get_knowledge_rag
        KNOWLEDGE_RAG_AVAILABLE = True
        logger.info("✅ Istanbul Knowledge RAG system available")
    except ImportError:
        KNOWLEDGE_RAG_AVAILABLE = False
        logger.warning("⚠️ Istanbul Knowledge RAG system not available")


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
        self.weather_service = weather_service or (service_manager.weather_service if service_manager else None)
        self.events_service = events_service or (service_manager.events_service if service_manager else None)
        self.hidden_gems_service = hidden_gems_service or (service_manager.hidden_gems_service if service_manager else None)
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
        language: str = "en",
        signal_confidence: float = 1.0,
        original_query: str = None
    ) -> Dict[str, Any]:
        """
        Build smart context based on detected signals.
        
        PRIORITY 3 ENHANCEMENT: Adjust context breadth based on signal confidence.
        Low confidence → Provide MORE context to help LLM infer intent.
        High confidence → Provide focused context.
        
        Args:
            query: User query (may be rewritten)
            signals: Detected signals
            user_location: User GPS location
            language: Language code
            signal_confidence: Overall signal detection confidence (0.0-1.0)
            original_query: Original unmodified query (for location extraction)
            
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
        
        # PRIORITY 3: Adjust context provisioning based on signal confidence
        if signal_confidence < 0.5:
            logger.info(f"⚠️ Low signal confidence ({signal_confidence:.2f}), providing BROADER context")
            context_strategy = 'broad'  # Fetch more data
            rag_top_k = 10  # More RAG documents
        elif signal_confidence < 0.7:
            logger.info(f"ℹ️ Medium signal confidence ({signal_confidence:.2f}), providing STANDARD context")
            context_strategy = 'standard'
            rag_top_k = 5
        else:
            logger.info(f"✅ High signal confidence ({signal_confidence:.2f}), providing FOCUSED context")
            context_strategy = 'focused'  # Only relevant data
            rag_top_k = 3
        
        # Build database context (with strategy)
        if self._needs_database_context(signals):
            context['database'] = await self._build_database_context(
                query=query,
                signals=signals,
                user_location=user_location,
                language=language,
                context_strategy=context_strategy,  # Pass strategy
                original_query=original_query  # Pass original query for transportation
            )
            
            # CRITICAL: For transportation queries, also extract the route_data object
            # This enables the HYBRID ARCHITECTURE (template facts + LLM reasoning)
            if signals.get('needs_transportation') and TRANSPORTATION_RAG_AVAILABLE:
                try:
                    from services.transportation_rag_system import get_transportation_rag
                    transport_rag = get_transportation_rag()
                    if transport_rag.last_route:
                        # TransitRoute is an object - convert to dict for prompt builder
                        route_obj = transport_rag.last_route
                        context['route_data'] = {
                            'origin': getattr(route_obj, 'origin', None),
                            'destination': getattr(route_obj, 'destination', None),
                            'steps': getattr(route_obj, 'steps', []),
                            'total_time': getattr(route_obj, 'total_time', None),
                            'total_distance': getattr(route_obj, 'total_distance', None),
                            'transfers': getattr(route_obj, 'transfers', None),
                            'lines_used': getattr(route_obj, 'lines_used', [])
                        }
                        logger.info(f"✅ Route data extracted: {context['route_data']['origin']} → {context['route_data']['destination']}")
                except Exception as e:
                    logger.warning(f"Could not extract route_data: {e}")
        
        # Get RAG context with retry and circuit breaker (with confidence-based top_k)
        if self.rag_service:
            try:
                rag_cb = self.circuit_breakers.get('rag')
                if rag_cb:
                    context['rag'] = await rag_cb.call(
                        self._get_rag_context_with_retry, query, language, rag_top_k
                    )
                else:
                    context['rag'] = await self._get_rag_context_with_retry(query, language, rag_top_k)
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
                logger.info(f"✅ Weather context added: {context['services']['weather'][:200]}...")
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
        
        # Get airport transport context (NEW)
        if signals.get('needs_airport'):
            try:
                context['services']['airport'] = await self._get_airport_context(query, user_location, language)
            except Exception as e:
                logger.warning(f"Airport context failed: {e}")
        
        # Get daily life suggestions context (NEW - Phase 2)
        if signals.get('needs_daily_life'):
            try:
                context['services']['daily_life'] = await self._get_daily_life_context(query, language)
            except Exception as e:
                logger.warning(f"Daily life context failed: {e}")
        
        # Generate map visualization with multi-route support
        # Auto-generate maps for location-based queries (neighborhoods, attractions, restaurants)
        should_generate_map = (
            signals.get('needs_map') or 
            signals.get('needs_gps_routing') or
            signals.get('needs_neighborhood') or
            signals.get('needs_attraction') or
            signals.get('needs_restaurant') or
            signals.get('needs_hidden_gems') or
            signals.get('needs_transportation')
        )
        
        if should_generate_map:
            try:
                # Check if we have multi-route data from transportation context
                if hasattr(self, '_transport_alternatives') and self._transport_alternatives.get('alternatives'):
                    # Use enhanced map service for multi-route visualization
                    logger.info("🗺️ Generating multi-route map visualization")
                    
                    try:
                        from services.enhanced_map_visualization_service import get_enhanced_map_service
                        enhanced_map_service = get_enhanced_map_service()
                        
                        # Extract location info from alternatives
                        primary = self._transport_alternatives.get('primary_route', {})
                        origin = primary.get('origin', 'Unknown')
                        destination = primary.get('destination', 'Unknown')
                        
                        # Generate multi-route map
                        # Note: The enhanced map service will use the route integration
                        # which already has the routes we need
                        multi_map_data = await enhanced_map_service.generate_multi_route_map(
                            origin=origin,
                            destination=destination,
                            origin_gps=user_location,  # User's current location if available
                            destination_gps=None,  # Destination coords would need lookup
                            num_alternatives=len(self._transport_alternatives.get('alternatives', []))
                        )
                        
                        if multi_map_data and multi_map_data['type'] == 'multi_route':
                            context['map_data'] = multi_map_data
                            logger.info(f"✅ Multi-route map generated with {len(multi_map_data['routes'])} routes")
                        else:
                            # Fallback to stored map data if available
                            if self._transport_alternatives.get('map_data'):
                                context['map_data'] = self._transport_alternatives['map_data']
                                logger.info("✅ Using stored map data from transport alternatives")
                    except Exception as e:
                        logger.warning(f"Enhanced map generation failed, using fallback: {e}")
                        # Use stored map data if available
                        if self._transport_alternatives.get('map_data'):
                            context['map_data'] = self._transport_alternatives['map_data']
                
                elif self.map_service:
                    # Use standard map service
                    context['map_data'] = await self._generate_map(
                        query=query,
                        signals=signals,
                        user_location=user_location,
                        language=language
                    )
                    logger.info(f"✅ Standard map data generated for query with signals: {[k for k, v in signals.items() if v]}")
                    
            except Exception as e:
                logger.warning(f"Map generation failed: {e}")
        
        # NEW: Enhance context with location-based information
        # This automatically adds hidden gems when districts are mentioned
        if LOCATION_ENHANCER_AVAILABLE:
            try:
                enhancer = get_location_based_enhancer()
                
                # Get intent from signals for better context
                intent = None
                if signals.get('needs_restaurant'):
                    intent = 'restaurant'
                elif signals.get('needs_attraction'):
                    intent = 'attraction'
                elif signals.get('needs_hidden_gems'):
                    intent = 'hidden_gems'
                
                # Enhance context with location-based data (safely handle None user_location)
                enriched_context = await enhancer.enhance_context(
                    query=query,
                    base_context=context,
                    intent=intent
                )
                
                # Merge enriched context
                context = self._merge_location_enriched_context(context, enriched_context)
                logger.info("✅ Location-based context enhancement applied")
            except KeyError as ke:
                logger.error(f"Location-based context enhancement failed with KeyError: {ke} - likely missing user_location key")
            except Exception as e:
                logger.warning(f"Location-based context enhancement failed: {e}")
        
        # CRITICAL: Pass multi-route data to context return for frontend
        # If we have transport alternatives with multi-route data, include it in context
        if hasattr(self, '_transport_alternatives') and self._transport_alternatives:
            if self._transport_alternatives.get('alternatives') or self._transport_alternatives.get('primary_route'):
                context['transport_alternatives'] = self._transport_alternatives
                num_routes = len(self._transport_alternatives.get('alternatives', []))
                logger.info(f"✅ Added transport alternatives to context for frontend ({num_routes} routes)")
        
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
        language: str,
        context_strategy: str = 'standard',
        original_query: str = None
    ) -> str:
        """
        Build database context based on signals.
        
        Args:
            query: User query
            signals: Detected signals
            user_location: User GPS location
            language: Language code
            context_strategy: Strategy for context building ('broad', 'standard', 'focused')
            
        Returns:
            Formatted database context string
        """
        context_parts = []
        
        # Get restaurant context
        # Note: LLM will naturally extract relevant details from query
        # No need for separate entity extraction - LLM can handle it
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
        
        # Get transportation context with route alternatives
        if signals.get('needs_transportation'):
            # Check if we should use multi-route optimization
            # Enable for queries that seem to be asking for routes
            use_multi_route = any(keyword in query.lower() for keyword in [
                'how to get', 'how do i get', 'route to', 'way to',
                'directions to', 'how can i reach', 'best way to'
            ])
            
            if use_multi_route:
                # Use Moovit-style multi-route optimization
                transport_data = await self._get_transportation_with_alternatives(
                    query,
                    language,
                    original_query=original_query,
                    user_location=user_location,
                    num_alternatives=3,
                    generate_llm_summaries=False  # LLM will generate its own summary
                )
                
                if transport_data and transport_data.get('context'):
                    context_parts.append("=== TRANSPORTATION (MULTI-ROUTE) ===")
                    context_parts.append(transport_data['context'])
                    
                    # Store alternatives for later use in response
                    # This can be accessed by the prompt builder
                    if not hasattr(self, '_transport_alternatives'):
                        self._transport_alternatives = {}
                    self._transport_alternatives = {
                        'primary_route': transport_data.get('primary_route'),
                        'alternatives': transport_data.get('alternatives', []),
                        'route_comparison': transport_data.get('route_comparison', {}),
                        'map_data': transport_data.get('map_data')
                    }
            else:
                # Use standard transportation context
                transport = await self._get_transportation(
                    query,
                    language,
                    original_query=original_query,
                    user_location=user_location  # Pass GPS location
                )
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
                cursor = await self.db.execute(
                    text("""
                        SELECT name, cuisine, location, price_level, rating
                        FROM restaurants
                        LIMIT 5
                    """)
                )
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
                    f"Price Level: {row[3]}, Rating: {row[4]}/5"
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
        """Get attraction data with enhanced details if available."""
        try:
            # Try to use enhanced attractions service from ServiceManager first
            if self.service_manager and hasattr(self.service_manager, 'attractions_service'):
                if self.service_manager.attractions_service:
                    logger.debug("Using EnhancedAttractionsService from ServiceManager")
                    try:
                        # Use enhanced service for richer data
                        attractions = self.service_manager.search_attractions(
                            query='',
                            category=None,  # LLM will filter
                            district=None   # LLM will filter
                        )
                        
                        if attractions:
                            # Format enhanced results
                            results = []
                            for attr in attractions[:5]:  # Top 5
                                info = f"- {attr.get('name', 'Unknown')}"
                                if attr.get('category'):
                                    info += f" ({attr['category']})"
                                if attr.get('district'):
                                    info += f": Located in {attr['district']}"
                                if attr.get('description'):
                                    info += f". {attr['description'][:100]}..."
                                if attr.get('opening_hours'):
                                    info += f" | Hours: {attr['opening_hours']}"
                                if attr.get('entry_fee'):
                                    info += f" | Fee: {attr['entry_fee']}"
                                results.append(info)
                            
                            return "\n".join(results)
                    except Exception as e:
                        logger.warning(f"Enhanced attractions service failed: {e}, falling back to database")
            
            # Fallback: Basic database query (use 'places' table instead of 'attractions')
            async def _query_db():
                cursor = await self.db.execute(
                    text("""
                        SELECT name, category, district
                        FROM places
                        LIMIT 5
                    """)
                )
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
                    f"- {row[0]} ({row[1]}): Located in {row[2]}"
                )
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Failed to get attractions: {e}")
            return ""
    
    async def _get_neighborhoods(self, query: str, language: str) -> str:
        """Get neighborhood data with coordinates from Istanbul Knowledge RAG."""
        try:
            results = []
            
            # PART 1: Use new Knowledge RAG for comprehensive neighborhood info
            if KNOWLEDGE_RAG_AVAILABLE:
                try:
                    knowledge_rag = get_knowledge_rag()
                    neighborhood_info = knowledge_rag.get_neighborhood_info(query)
                    
                    if neighborhood_info:
                        name = neighborhood_info.get('name', 'Neighborhood')
                        results.append(f"=== {name.upper()} ===")
                        
                        if neighborhood_info.get('vibe'):
                            results.append(f"Vibe: {neighborhood_info['vibe']}")
                        if neighborhood_info.get('best_for'):
                            best_for = neighborhood_info['best_for']
                            if isinstance(best_for, list):
                                results.append(f"Best for: {', '.join(best_for)}")
                            else:
                                results.append(f"Best for: {best_for}")
                        if neighborhood_info.get('highlights'):
                            highlights = neighborhood_info['highlights']
                            if isinstance(highlights, list):
                                results.append(f"Highlights: {', '.join(highlights[:5])}")
                        if neighborhood_info.get('where_to_eat'):
                            results.append(f"Food: {neighborhood_info['where_to_eat'][:200]}...")
                        if neighborhood_info.get('getting_there'):
                            results.append(f"Access: {neighborhood_info['getting_there']}")
                        
                        logger.info(f"✅ Added neighborhood info from Knowledge RAG")
                except Exception as e:
                    logger.warning(f"Knowledge RAG neighborhood lookup failed: {e}")
            
            # PART 2: Fall back to Istanbul Knowledge for coordinates
            try:
                from .istanbul_knowledge import IstanbulKnowledge
                istanbul_kb = IstanbulKnowledge()
                
                # Extract neighborhood names from query
                query_lower = query.lower()
                mentioned_neighborhoods = []
                
                for name, neighborhood in istanbul_kb.neighborhoods.items():
                    if name.lower() in query_lower or any(alias.lower() in query_lower for alias in [name]):
                        mentioned_neighborhoods.append((name, neighborhood))
                
                # If no specific neighborhoods mentioned, return top popular ones
                if not mentioned_neighborhoods:
                    popular = ['Beyoğlu', 'Sultanahmet', 'Kadıköy', 'Beşiktaş']
                    for name in popular:
                        if name in istanbul_kb.neighborhoods:
                            mentioned_neighborhoods.append((name, istanbul_kb.neighborhoods[name]))
                
                # Format results with coordinates
                for name, neighborhood in mentioned_neighborhoods[:5]:  # Top 5
                    info = f"- {name}: {neighborhood.character}"
                    if neighborhood.center_location:
                        lat, lon = neighborhood.center_location
                        info += f" | Coordinates: ({lat}, {lon})"
                    if neighborhood.transport_hubs:
                        info += f" | Transport: {', '.join(neighborhood.transport_hubs[:3])}"
                    results.append(info)
            except Exception as e:
                logger.warning(f"Istanbul Knowledge neighborhood lookup failed: {e}")
            
            return "\n".join(results) if results else ""
            
        except Exception as e:
            logger.error(f"Failed to get neighborhoods: {e}")
            return ""
    
    async def _get_transportation(
        self,
        query: str,
        language: str,
        original_query: str = None,
        user_location: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Get INDUSTRY-LEVEL transportation data using Google Maps-quality RAG system.
        
        This provides:
        - Complete verified route information
        - Step-by-step directions
        - Real station names and connections
        - Transfer points and times
        - GPS-based routing when user location available
        
        Args:
            query: Current query (may be rewritten)
            language: Response language
            original_query: Original unmodified query (better for location extraction)
            user_location: User GPS location {"lat": float, "lon": float}
        """
        try:
            # Use industry-level Transportation RAG system
            if TRANSPORTATION_RAG_AVAILABLE:
                logger.info("🗺️ Using Industry-Level Transportation RAG System")
                
                transport_rag = get_transportation_rag()
                
                # IMPORTANT: Use original query for location extraction if available
                # Rewritten queries can confuse location extraction
                query_for_rag = original_query if original_query else query
                logger.info(f"🔍 Using query for RAG: '{query_for_rag}'")
                
                # === GPS PATTERN DETECTION ===
                # Check if query uses "from my location", "from here", or similar patterns
                query_lower = query_for_rag.lower()
                gps_origin_patterns = [
                    'from my location', 'from here', 'from current location',
                    'from where i am', 'from my position', 'starting from here',
                    'from my current location', 'from where i\'m at'
                ]
                
                gps_dest_patterns = [
                    'to my location', 'to here', 'to current location',
                    'to where i am', 'back here', 'to where i\'m at'
                ]
                
                uses_gps_origin = any(pattern in query_lower for pattern in gps_origin_patterns)
                uses_gps_dest = any(pattern in query_lower for pattern in gps_dest_patterns)
                
                # Also check if only ONE location is mentioned (implying GPS as origin)
                # Count location keywords
                location_count = sum([
                    1 for word in ['to ', 'from ', 'in '] 
                    if word in query_lower
                ])
                
                # If GPS detected, log it
                if uses_gps_origin or uses_gps_dest or (location_count == 1 and user_location):
                    if user_location:
                        logger.info(f"📍 GPS DETECTED in query! Location: {user_location}")
                        logger.info(f"🔍 GPS Origin: {uses_gps_origin}, GPS Dest: {uses_gps_dest}, Single location: {location_count == 1}")
                    else:
                        logger.warning(f"⚠️ GPS pattern detected but no GPS location available!")
                        logger.warning(f"   User should enable location permissions in browser")
                else:
                    if user_location:
                        logger.info(f"� GPS available but query doesn't use it: {user_location}")
                    else:
                        logger.info("❌ No GPS location provided")
                
                # Generate RAG context for this specific query, passing user_location
                # The Transportation RAG will handle GPS-based routing internally
                rag_context = transport_rag.get_rag_context_for_query(query_for_rag, user_location=user_location)
                
                logger.info(f"✅ Generated {len(rag_context)} chars of verified transportation context")
                logger.debug(f"📄 RAG context preview: {rag_context[:200]}...")
                return rag_context
            
            # Fallback: Try to use service_manager's transportation service
            transport_service = None
            
            if self.service_manager and hasattr(self.service_manager, 'transportation_service'):
                transport_service = self.service_manager.transportation_service
                logger.debug("Using transportation service from ServiceManager")
            
            # Fallback: import directly
            if not transport_service:
                from services.transportation_directions_service import get_transportation_service
                transport_service = get_transportation_service()
                logger.debug("Using standalone transportation service")
            
            # Provide comprehensive Istanbul transit information for LLM context
            transit_info = """Istanbul Public Transportation:

🚇 METRO LINES:
- M1 (Red): Yenikapı - Atatürk Airport/Kirazlı
- M2 (Green): Yenikapı - Hacıosman (serves Taksim, Şişhane, Osmanbey, Levent)
- M3 (Blue): Kirazlı - Başakşehir/Olimpiyat
- M4 (Pink): Kadıköy - Tavşantepe (Asian side) - **CONNECTS TO MARMARAY at Ayrılık Çeşmesi**
- M5 (Purple): Üsküdar - Çekmeköy (Asian side)
- M6, M7, M9, M11: Other metro lines

🚊 TRAM LINES:
- T1: Bağcılar - Kabataş (serves Sultanahmet, Eminönü, Karaköy)
- T4: Topkapı - Mescid-i Selam
- T5: Cibali - Alibeyköy

🚂 MARMARAY (Underground Rail):
- **VERIFIED: Connects Asian and European sides via underwater tunnel**
- **KEY: DOES serve Kadıköy via Ayrılık Çeşmesi station (M4 transfer point)**
- Route: Gebze ↔ Pendik ↔ Kartal ↔ Bostancı ↔ **Ayrılık Çeşmesi (Kadıköy)** ↔ Üsküdar ↔ Sirkeci ↔ Yenikapı ↔ Halkalı
- Major transfer hubs: 
  * Yenikapı (M1/M2 transfers)
  * Ayrılık Çeşmesi (M4 transfer - **KEY KADIKOY CONNECTION**)
  * Üsküdar (M5 transfer)
  * Sirkeci (T1 transfer)

🚡 FUNICULARS:
- F1: Kabataş ↔ Taksim (connects T1 tram to M2 metro)
- F2: Karaköy ↔ Tünel/Şişhane (connects T1 to M2)

⛴️ FERRIES:
- Kadıköy ↔ Karaköy (15-20 min)
- Kadıköy ↔ Eminönü (20 min)
- Üsküdar ↔ Eminönü (15 min)
- Beşiktaş ↔ Kadıköy (25 min)

**VERIFIED ROUTE: Kadıköy to Taksim:**
1. Take M4 metro to Ayrılık Çeşmesi station
2. Transfer to Marmaray (same station)
3. Take Marmaray to Yenikapı
4. Transfer to M2 metro
5. Take M2 to Taksim
Total time: ~35 minutes

**Alternative: Kadıköy to Taksim via Ferry:**
1. Take ferry from Kadıköy to Karaköy (~20 min)
2. Take F2 funicular to Tünel/Şişhane
3. Walk to Taksim or take M2 one stop
Total time: ~30 minutes (more scenic!)"""
            
            logger.debug("Transportation context built successfully")
            return transit_info
            
        except Exception as e:
            logger.error(f"Failed to get transportation info: {e}")
            # Fallback: basic transit info with KEY CORRECTION
            return """Istanbul has metro (M1-M11), tram (T1, T4, T5), Marmaray rail, funiculars (F1, F2), ferries, and metrobus services.

**IMPORTANT: Marmaray DOES serve Kadıköy via Ayrılık Çeşmesi station (M4 connection point).**"""
    
    async def _get_transportation_with_alternatives(
        self,
        query: str,
        language: str,
        original_query: str = None,
        user_location: Optional[Dict[str, float]] = None,
        num_alternatives: int = 3,
        generate_llm_summaries: bool = False
    ) -> Dict[str, Any]:
        """
        Get Moovit-style transportation data with multiple route alternatives.
        
        This provides:
        - Primary recommended route
        - 3-5 alternative routes with different optimizations
        - Comfort scoring for each route
        - Route comparison data
        - Map visualization data
        - Optional LLM-powered route summaries
        
        Args:
            query: Current query (may be rewritten)
            language: Response language
            original_query: Original unmodified query (better for location extraction)
            user_location: User GPS location {"lat": float, "lon": float}
            num_alternatives: Number of route alternatives to generate
            generate_llm_summaries: Whether to generate LLM summaries for routes
            
        Returns:
            Dict with primary_route, alternatives, route_comparison, map_data, and context string
        """
        try:
            # Check if route integration is available
            try:
                from services.transportation_route_integration import get_route_integration
                ROUTE_INTEGRATION_AVAILABLE = True
            except ImportError as e:
                ROUTE_INTEGRATION_AVAILABLE = False
                logger.warning(f"⚠️ Route integration not available: {e}")
            
            if not ROUTE_INTEGRATION_AVAILABLE:
                # Fallback to standard transportation
                context_str = await self._get_transportation(query, language, original_query, user_location)
                return {
                    'context': context_str,
                    'primary_route': None,
                    'alternatives': [],
                    'route_comparison': {},
                    'map_data': None
                }
            
            # Use industry-level Transportation RAG + Route Optimizer
            logger.info("🚀 Using Moovit-Style Route Integration with Multi-Route Optimization")
            
            # Extract origin and destination from query
            query_for_extraction = original_query if original_query else query
            locations = self._extract_transportation_locations(query_for_extraction, user_location)
            
            if not locations or not locations.get('origin') or not locations.get('destination'):
                # Can't extract locations - fallback to standard
                logger.warning("Could not extract origin/destination - using standard transportation")
                context_str = await self._get_transportation(query, language, original_query, user_location)
                return {
                    'context': context_str,
                    'primary_route': None,
                    'alternatives': [],
                    'route_comparison': {},
                    'map_data': None
                }
            
            # Get route integration
            route_integration = get_route_integration()
            
            # Get route alternatives
            result = route_integration.get_route_alternatives(
                origin=locations['origin'],
                destination=locations['destination'],
                origin_gps=locations.get('origin_gps'),
                destination_gps=locations.get('destination_gps'),
                num_alternatives=num_alternatives,
                generate_llm_summaries=generate_llm_summaries,
                user_language=language
            )
            
            if result['success']:
                logger.info(f"✅ Got {len(result.get('alternatives', []))} route alternatives")
                
                # Build context string from result
                context_str = self._build_multi_route_context(result, language)
                
                return {
                    'context': context_str,
                    'primary_route': result.get('primary_route'),
                    'alternatives': result.get('alternatives', []),
                    'route_comparison': result.get('route_comparison', {}),
                    'map_data': result.get('map_data')
                }
            else:
                # Error - fallback to standard
                logger.warning(f"Route integration failed: {result.get('error')}")
                context_str = await self._get_transportation(query, language, original_query, user_location)
                return {
                    'context': context_str,
                    'primary_route': None,
                    'alternatives': [],
                    'route_comparison': {},
                    'map_data': None
                }
                
        except Exception as e:
            logger.error(f"Failed to get transportation with alternatives: {e}", exc_info=True)
            # Fallback to standard transportation
            context_str = await self._get_transportation(query, language, original_query, user_location)
            return {
                'context': context_str,
                'primary_route': None,
                'alternatives': [],
                'route_comparison': {},
                'map_data': None
            }
    
    def _extract_transportation_locations(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract origin and destination from transportation query.
        
        Returns dict with:
        - origin: origin name
        - destination: destination name
        - origin_gps: optional GPS dict
        - destination_gps: optional GPS dict
        """
        import re
        
        query_lower = query.lower()
        
        # Check for GPS patterns
        uses_gps_origin = any(pattern in query_lower for pattern in [
            'from my location', 'from here', 'from current location',
            'from where i am', 'from my position', 'starting from here'
        ])
        
        uses_gps_dest = any(pattern in query_lower for pattern in [
            'to my location', 'to here', 'to current location',
            'to where i am', 'back here'
        ])
        
        # Extract locations using patterns
        # Pattern 1: "from X to Y" or "X to Y"
        match = re.search(r'(?:from\s+)?([a-zğüşöçıİ\s]+?)\s+to\s+([a-zğüşöçıİ\s]+)', query_lower, re.IGNORECASE)
        
        if match:
            origin = match.group(1).strip()
            destination = match.group(2).strip()
            
            # Replace GPS placeholders with actual location
            result = {}
            
            if 'my location' in origin or 'here' in origin or 'current location' in origin:
                if user_location:
                    result['origin'] = 'Current Location'
                    result['origin_gps'] = user_location
                else:
                    return None  # GPS origin but no location available
            else:
                result['origin'] = origin.title()
                
            if 'my location' in destination or 'here' in destination or 'current location' in destination:
                if user_location:
                    result['destination'] = 'Current Location'
                    result['destination_gps'] = user_location
                else:
                    return None  # GPS destination but no location available
            else:
                result['destination'] = destination.title()
            
            return result
        
        # Pattern 2: "how to get to X" (implies GPS origin if available)
        match = re.search(r'(?:how|way)\s+(?:do i |can i |to )?(?:get|go)\s+to\s+([a-zğüşöçıİ\s]+)', query_lower, re.IGNORECASE)
        
        if match and user_location:
            destination = match.group(1).strip()
            return {
                'origin': 'Current Location',
                'origin_gps': user_location,
                'destination': destination.title()
            }
        
        return None
    
    def _build_multi_route_context(self, result: Dict[str, Any], language: str) -> str:
        """Build context string from multi-route result for LLM"""
        lines = []
        
        # Primary route
        pr = result.get('primary_route')
        if pr:
            lines.append(f"PRIMARY ROUTE: {pr['origin']} → {pr['destination']}")
            lines.append(f"Duration: {pr['total_time']} minutes")
            lines.append(f"Distance: {pr['total_distance']:.1f} km")
            lines.append(f"Transfers: {pr['transfers']}")
            lines.append(f"Lines: {', '.join(pr.get('lines_used', []))}")
            lines.append("")
        
        # Alternatives
        alternatives = result.get('alternatives', [])
        if alternatives:
            lines.append(f"ALTERNATIVE ROUTES ({len(alternatives)} options):")
            for i, alt in enumerate(alternatives, 1):
                lines.append(f"\n{i}. {alt['preference'].upper()}")
                lines.append(f"   Duration: {alt['duration_minutes']} min")
                lines.append(f"   Transfers: {alt['num_transfers']}")
                lines.append(f"   Walking: {int(alt['walking_meters'])}m")
                lines.append(f"   Comfort: {alt['comfort_score']['overall_comfort']:.0f}/100")
                lines.append(f"   Score: {alt['overall_score']:.1f}/100")
                
                if alt.get('highlights'):
                    lines.append(f"   Highlights: {', '.join(alt['highlights'])}")
                
                if alt.get('llm_summary'):
                    lines.append(f"   Summary: {alt['llm_summary']}")
        
        return "\n".join(lines)
