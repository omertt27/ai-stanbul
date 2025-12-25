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
        
        # Generate map visualization
        # Auto-generate maps for location-based queries (neighborhoods, attractions, restaurants)
        should_generate_map = (
            signals.get('needs_map') or 
            signals.get('needs_gps_routing') or
            signals.get('needs_neighborhood') or
            signals.get('needs_attraction') or
            signals.get('needs_restaurant') or
            signals.get('needs_hidden_gems')
        )
        
        if should_generate_map and self.map_service:
            try:
                context['map_data'] = await self._generate_map(
                    query=query,
                    signals=signals,
                    user_location=user_location,
                    language=language
                )
                logger.info(f"✅ Map data generated for query with signals: {[k for k, v in signals.items() if v]}")
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
        
        # Get transportation context
        if signals.get('needs_transportation'):
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
                        SELECT name, cuisine, district, price_range, rating
                        FROM restaurants
                        WHERE language = :language
                        LIMIT 5
                    """),
                    {"language": language}
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
        """Get attraction data with enhanced details if available."""
        try:
            # Try to use enhanced attractions service from ServiceManager first
            if self.service_manager and hasattr(self.service_manager, 'attractions_service'):
                if self.service_manager.attractions_service:
                    logger.debug("Using EnhancedAttractionsService from ServiceManager")
                    try:
                        # Use enhanced service for richer data
                        attractions = self.service_manager.attractions_service.search_attractions(
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
            
            # Fallback: Basic database query
            async def _query_db():
                cursor = await self.db.execute(
                    text("""
                        SELECT name, category, district, description
                        FROM attractions
                        WHERE language = :language
                        LIMIT 5
                    """),
                    {"language": language}
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
                    f"- {row[0]} ({row[1]}): Located in {row[2]}. {row[3][:100]}..."
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
    
    async def _get_rag_context(self, query: str, language: str) -> str:
        """Get RAG context from embeddings with circuit breaker protection."""
        context_parts = []
        
        # PART 1: Database RAG (restaurants, museums, events, etc.)
        if self.rag_service:
            try:
                # Use circuit breaker if available
                if 'rag' in self.circuit_breakers:
                    async def _search():
                        return await self.rag_service.search(query, language=language, top_k=3)
                    
                    results = await self.circuit_breakers['rag'].call(_search)
                else:
                    results = await self.rag_service.search(query, language=language, top_k=3)
                
                if results:
                    # Format Database RAG results
                    for result in results:
                        context_parts.append(f"[Score: {result.get('score', 0):.2f}] {result.get('text', '')[:200]}...")
            
            except Exception as e:
                logger.error(f"Database RAG search failed: {e}")
        
        # PART 2: Istanbul Knowledge RAG (neighborhoods, food, attractions, scams, etc.)
        if KNOWLEDGE_RAG_AVAILABLE:
            try:
                knowledge_rag = get_knowledge_rag()
                knowledge_context = knowledge_rag.get_context_for_llm(query, max_length=1500)
                
                if knowledge_context:
                    context_parts.append("\n=== ISTANBUL KNOWLEDGE BASE ===")
                    context_parts.append(knowledge_context)
                    logger.info(f"✅ Added Knowledge RAG context: {len(knowledge_context)} chars")
            except Exception as e:
                logger.warning(f"Knowledge RAG failed: {e}")
        
        if not context_parts:
            # Return graceful degradation message
            from .resilience import GracefulDegradation
            fallback = GracefulDegradation.get_fallback_context('rag')
            return fallback.get('message', '')
        
        return "\n\n".join(context_parts)
    
    async def _get_weather_context(self, query: str) -> str:
        """Get weather context with smart recommendations based on conditions."""
        if not self.weather_service:
            return ""
        
        try:
            # Get current weather
            weather = self.weather_service.get_current_weather("Istanbul")
            
            # Extract condition from weather data (handle different formats)
            condition = weather.get('condition')
            if not condition and 'weather' in weather and isinstance(weather['weather'], list) and len(weather['weather']) > 0:
                condition = weather['weather'][0].get('main', 'Unknown')
            condition = condition or 'Unknown'
            
            temperature = weather.get('temperature', 20)
            description = weather.get('description', '')
            
            logger.info(f"🌤️ Weather data extracted: condition={condition}, temp={temperature}°C, desc={description}")
            
            # Try to get weather-based activity recommendations
            try:
                from services.weather_recommendations import WeatherRecommendationsService
                weather_rec = WeatherRecommendationsService()
                
                # Get formatted recommendations
                recommendations = weather_rec.format_weather_activities_response(
                    temperature=temperature,
                    weather_condition=condition.lower(),
                    limit=5
                )
                
                # Combine weather info with recommendations
                weather_info = (
                    f"Current weather in Istanbul: {condition}, "
                    f"{temperature}°C. {description}\n\n"
                    f"{recommendations}"
                )
                
                return weather_info
                
            except Exception as rec_error:
                logger.warning(f"Weather recommendations failed: {rec_error}")
                # Fallback to basic weather info
                return (
                    f"Current weather in Istanbul: {condition}, "
                    f"{temperature}°C. {description}"
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
            # Check if method exists
            if not hasattr(self.events_service, 'get_upcoming_events'):
                logger.warning("Events service doesn't have get_upcoming_events method")
                return ""
            
            # Use circuit breaker if available
            if 'events' in self.circuit_breakers:
                async def _get_events():
                    # Call synchronously if not async
                    events_result = self.events_service.get_upcoming_events(limit=5)
                    return events_result
                
                events = await self.circuit_breakers['events'].call(_get_events)
            else:
                # Call synchronously
                events = self.events_service.get_upcoming_events(limit=5)
            
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
    
    async def _get_airport_context(
        self,
        query: str,
        user_location: Optional[Dict[str, float]],
        language: str
    ) -> str:
        """Get airport transport information from AirportTransportService."""
        try:
            # Try to use service_manager's airport service first
            airport_service = None
            
            if self.service_manager and hasattr(self.service_manager, 'airport_service'):
                airport_service = self.service_manager.airport_service
                logger.debug("Using airport service from ServiceManager")
            
            # Fallback: import directly
            if not airport_service:
                from services.airport_transport_service import IstanbulAirportTransportService
                airport_service = IstanbulAirportTransportService()
                logger.debug("Using standalone airport service")
            
            # Determine which airport is being asked about
            query_lower = query.lower()
            airport_code = None
            
            if 'ist' in query_lower or 'istanbul airport' in query_lower or 'new airport' in query_lower:
                airport_code = 'IST'
            elif 'saw' in query_lower or 'sabiha' in query_lower or 'gokcen' in query_lower or 'gökçen' in query_lower:
                airport_code = 'SAW'
            
            # If specific airport mentioned, get detailed info for that airport
            if airport_code:
                airport_info = airport_service.get_route_recommendations(airport_code)
                return f"=== AIRPORT TRANSPORT ({airport_code}) ===\n{airport_info}"
            else:
                # General airport query - provide comparison
                comparison = airport_service.get_airport_comparison()
                return f"=== ISTANBUL AIRPORTS ===\n{comparison}"
                
        except Exception as e:
            logger.error(f"Failed to get airport info: {e}")
            # Fallback: basic airport info
            return """Istanbul has two main airports:
- Istanbul Airport (IST): European side, main international hub. Access via M11 metro or Havaist buses.
- Sabiha Gökçen Airport (SAW): Asian side. Access via buses, metro, or private shuttle."""
    
    async def _get_daily_life_context(self, query: str, language: str) -> str:
        """Get practical daily life suggestions from DailyLifeSuggestionsService."""
        try:
            # Try to use service_manager's daily life service first
            daily_service = None
            
            if self.service_manager and hasattr(self.service_manager, 'daily_life_service'):
                daily_service = self.service_manager.daily_life_service
                logger.debug("Using daily life service from ServiceManager")
            
            # Fallback: import directly
            if not daily_service:
                from services.daily_life_suggestions_service import DailyLifeSuggestionsService
                daily_service = DailyLifeSuggestionsService()
                logger.debug("Using standalone daily life service")
            
            # NEW: Get specific locations for the query
            location_data = daily_service.get_specific_locations(query, language)
            
            if location_data and location_data.get('type') != 'general':
                # Format specific location data
                result = f"=== {location_data.get('title', 'PRACTICAL INFORMATION')} ===\n\n"
                
                if 'locations' in location_data:
                    for loc in location_data['locations']:
                        result += f"📍 {loc['name']}\n"
                        result += f"   Areas: {', '.join(loc['areas'])}\n"
                        result += f"   {loc['description']}\n"
                        result += f"   💡 Tip: {loc['tip']}\n\n"
                
                if 'practical_tips' in location_data:
                    result += "PRACTICAL TIPS:\n"
                    for tip in location_data['practical_tips']:
                        result += f"• {tip}\n"
                
                return result
            else:
                # Fallback to general tips
                if 'tips' in location_data:
                    result = f"=== {location_data.get('title', 'PRACTICAL TIPS')} ===\n\n"
                    for tip in location_data['tips']:
                        result += f"• {tip}\n"
                    return result
                
                # Last resort fallback
                return """=== PRACTICAL LIVING TIPS ===
For groceries: Migros, Carrefour, or local markets
For pharmacy: Look for green cross sign "ECZANE"
For banks/ATM: Available throughout the city, many accept international cards
For SIM cards: Turkcell, Vodafone, Türk Telekom stores at airports and malls"""
                
        except Exception as e:
            logger.error(f"Failed to get daily life suggestions: {e}")
            # Fallback: basic tips
            return """=== PRACTICAL LIVING TIPS ===
For groceries: Migros, Carrefour, or local markets
For pharmacy: Look for green cross sign "ECZANE"
For banks/ATM: Available throughout the city, many accept international cards
For SIM cards: Turkcell, Vodafone, Türk Telekom stores at airports and malls"""
    
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
            # Enable routing for both GPS routing and general transportation queries
            routing = signals.get('needs_gps_routing', False) or signals.get('needs_transportation', False)
            
            map_data = await self.map_service.generate_map(
                query=query,
                user_location=user_location,
                language=language,
                routing=routing
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
    
    async def _get_daily_life_context(self, query: str, language: str) -> str:
        """Get daily life suggestions context."""
        try:
            # TODO: Implement actual daily life suggestions logic
            suggestions = [
                "Visit the historic Sultanahmet district.",
                "Take a Bosphorus cruise.",
                "Explore the Grand Bazaar.",
                "Visit the Hagia Sophia and Blue Mosque.",
                "Enjoy a Turkish bath experience."
            ]
            
            return "\n".join(f"- {s}" for s in suggestions)
        
        except Exception as e:
            logger.error(f"Failed to get daily life suggestions: {e}")
            return ""
    
    def _merge_location_enriched_context(
        self,
        context: Dict[str, Any],
        enriched_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge location-enriched context into main context.
        
        Extracts hidden gems, events, restaurants, and attractions from
        the location enrichment and adds them to the services context.
        """
        enrichment = enriched_context.get('location_enrichment', {})
        
        if not enrichment:
            return context
        
        # Add detected districts to services
        if 'detected_districts' in enrichment:
            context['services']['detected_districts'] = enrichment['detected_districts']
            logger.info(f"📍 Detected districts: {enrichment['detected_districts']}")
        
        # Add hidden gems
        if 'hidden_gems' in enrichment:
            gems = enrichment['hidden_gems']
            gems_text = self._format_hidden_gems_for_context(gems)
            
            # Append to existing hidden gems or create new
            if 'hidden_gems' in context['services']:
                context['services']['hidden_gems'] += "\n\n" + gems_text
            else:
                context['services']['hidden_gems'] = gems_text
            
            logger.info(f"💎 Added {len(gems)} hidden gems to context")
        
        # Add events
        if 'events' in enrichment:
            events = enrichment['events']
            events_text = self._format_events_for_context(events)
            
            if 'events' in context['services']:
                context['services']['events'] += "\n\n" + events_text
            else:
                context['services']['events'] = events_text
            
            logger.info(f"🎭 Added {len(events)} events to context")
        
        # Add restaurants
        if 'restaurants' in enrichment:
            restaurants = enrichment['restaurants']
            restaurant_text = self._format_restaurants_for_context(restaurants)
            
            # Append to database context
            if context['database']:
                context['database'] += "\n\n=== LOCAL RECOMMENDATIONS ===\n" + restaurant_text
            else:
                context['database'] = "=== LOCAL RECOMMENDATIONS ===\n" + restaurant_text
            
            logger.info(f"🍽️ Added {len(restaurants)} restaurants to context")
        
        # Add attractions
        if 'attractions' in enrichment:
            attractions = enrichment['attractions']
            attraction_text = self._format_attractions_for_context(attractions)
            
            if context['database']:
                context['database'] += "\n\n=== LOCAL ATTRACTIONS ===\n" + attraction_text
            else:
                context['database'] = "=== LOCAL ATTRACTIONS ===\n" + attraction_text
            
            logger.info(f"🏛️ Added {len(attractions)} attractions to context")
        
        return context
    
    def _format_hidden_gems_for_context(self, gems: List[Dict[str, Any]]) -> str:
        """Format hidden gems for LLM context"""
        formatted = []
        for gem in gems:
            text = f"💎 **{gem['name']}** ({gem['district']}) - {gem['category']}\n"
            text += f"   {gem['description']}\n"
            if gem.get('insider_tip'):
                text += f"   💡 Insider Tip: {gem['insider_tip']}\n"
            if gem.get('best_time'):
                text += f"   ⏰ Best Time: {gem['best_time']}\n"
            formatted.append(text)
        return "\n".join(formatted)
    
    def _format_events_for_context(self, events: List[Dict[str, Any]]) -> str:
        """Format events for LLM context"""
        formatted = []
        for event in events:
            text = f"🎭 **{event['title']}**\n"
            if event.get('venue'):
                text += f"   📍 Venue: {event['venue']}\n"
            if event.get('date'):
                text += f"   📅 Date: {event['date']}\n"
            if event.get('description'):
                text += f"   {event['description'][:150]}\n"
            formatted.append(text)
        return "\n".join(formatted)
    
    def _format_restaurants_for_context(self, restaurants: List[Dict[str, Any]]) -> str:
        """Format restaurants for LLM context"""
        formatted = []
        for restaurant in restaurants:
            text = f"🍽️ **{restaurant['name']}** - {restaurant.get('cuisine', 'N/A')}\n"
            text += f"   📍 {restaurant['district']} | {restaurant.get('price_range', 'N/A')}"
            if restaurant.get('rating'):
                text += f" | ⭐ {restaurant['rating']}/5"
            formatted.append(text)
        return "\n".join(formatted)
    
    def _format_attractions_for_context(self, attractions: List[Dict[str, Any]]) -> str:
        """Format attractions for LLM context"""
        formatted = []
        for attraction in attractions:
            text = f"🏛️ **{attraction['name']}** - {attraction.get('category', 'N/A')}\n"
            text += f"   📍 {attraction['district']}"
            if attraction.get('opening_hours'):
                text += f" | ⏰ {attraction['opening_hours']}"
            if attraction.get('entry_fee'):
                text += f" | 💰 {attraction['entry_fee']}"
            formatted.append(text)
        return "\n".join(formatted)
    
    def _merge_location_context(self, context: Dict[str, Any], location_context: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - kept for backward compatibility"""
        return self._merge_location_enriched_context(context, location_context)
