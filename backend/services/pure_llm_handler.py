"""
Pure LLM Query Handler
Routes ALL queries through RunPod LLM with database context injection
No rule-based fallback - LLM handles everything

ENHANCED: Now supports map visualization for transportation and route planning queries
by integrating with Istanbul Daily Talk AI system when needed.

SIGNAL-BASED DETECTION: Multilingual, semantic approach using embeddings for
flexible intent detection that supports multiple intents and languages.

Architecture:
- Single entry point for all queries
- Context injection from database
- RAG for similar queries
- Signal-based intent detection (multi-intent support)
- Redis caching for responses and signals
- MAP VISUALIZATION for routes and transportation (NEW)
- Semantic embeddings for language-independent detection

Author: Istanbul AI Team
Date: November 12, 2025
"""

import logging
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List, Set
from sqlalchemy.orm import Session
from datetime import datetime

# Semantic similarity for multilingual detection
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Signal detection will use keyword fallback only.")

logger = logging.getLogger(__name__)


class PureLLMHandler:
    """
    Pure LLM architecture - no rule-based processing
    All queries go through RunPod LLM with context injection
    
    ENHANCED: Now includes map visualization support by routing
    transportation/route queries to Istanbul Daily Talk AI when needed.
    """
    
    def __init__(
        self,
        runpod_client,
        db_session: Session,
        redis_client=None,
        context_builder=None,
        rag_service=None,
        istanbul_ai_system=None
    ):
        """
        Initialize Pure LLM Handler
        
        Args:
            runpod_client: RunPod LLM client instance
            db_session: SQLAlchemy database session
            redis_client: Redis client for caching (optional)
            context_builder: ML context builder (optional)
            rag_service: RAG vector service (optional)
            istanbul_ai_system: Istanbul Daily Talk AI for map visualization (optional)
        """
        self.llm = runpod_client
        self.db = db_session
        self.redis = redis_client
        self.context_builder = context_builder
        self.rag = rag_service
        self.istanbul_ai = istanbul_ai_system  # For map visualization
        
        # Initialize semantic embedding model for signal detection
        self.embedding_model = None
        self._signal_embeddings = {}  # Cache for signal pattern embeddings
        if EMBEDDINGS_AVAILABLE:
            try:
                # Use multilingual model for Turkish + English support
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self._init_signal_embeddings()
                logger.info("✅ Semantic embedding model loaded for signal detection")
            except Exception as e:
                logger.warning(f"⚠️ Could not load embedding model: {e}")
                self.embedding_model = None
        
        # Initialize additional services
        self._init_additional_services()
        
        # Load system prompts
        self._load_prompts()
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallback_calls": 0,
            "map_requests": 0,
            "weather_requests": 0,
            "hidden_gems_requests": 0,
            "signal_cache_hits": 0,
            "multi_signal_queries": 0
        }
        
        logger.info("✅ Pure LLM Handler initialized")
        logger.info(f"   RunPod LLM: {'✅ Enabled' if self.llm.enabled else '❌ Disabled'}")
        logger.info(f"   Redis Cache: {'✅ Enabled' if self.redis else '❌ Disabled'}")
        logger.info(f"   RAG Service: {'✅ Enabled' if self.rag else '❌ Disabled'}")
        logger.info(f"   Istanbul AI (Maps): {'✅ Enabled' if self.istanbul_ai else '❌ Disabled'}")
        logger.info(f"   Weather Service: {'✅ Enabled' if self.weather_service else '❌ Disabled'}")
        logger.info(f"   Events Service: {'✅ Enabled' if self.events_service else '❌ Disabled'}")
        logger.info(f"   Hidden Gems: {'✅ Enabled' if self.hidden_gems_handler else '❌ Disabled'}")
        logger.info(f"   Price Filter: {'✅ Enabled' if self.price_filter else '❌ Disabled'}")
        logger.info(f"   Semantic Embeddings: {'✅ Enabled' if self.embedding_model else '❌ Disabled (fallback to keywords)'}")
    
    def _init_additional_services(self):
        """Initialize additional services (weather, events, hidden gems, price filter)"""
        
        # Weather Recommendations Service
        try:
            from backend.services.weather_recommendations import get_weather_recommendations_service
            self.weather_service = get_weather_recommendations_service()
            logger.debug("Weather service loaded")
        except Exception as e:
            logger.warning(f"Weather service not available: {e}")
            self.weather_service = None
        
        # Events Service
        try:
            from backend.services.events_service import get_events_service
            self.events_service = get_events_service()
            logger.debug("Events service loaded")
        except Exception as e:
            logger.warning(f"Events service not available: {e}")
            self.events_service = None
        
        # Hidden Gems Handler
        try:
            from backend.services.hidden_gems_handler import HiddenGemsHandler
            self.hidden_gems_handler = HiddenGemsHandler()
            logger.debug("Hidden gems handler loaded")
        except Exception as e:
            logger.warning(f"Hidden gems handler not available: {e}")
            self.hidden_gems_handler = None
        
        # Price Filter Service
        try:
            from backend.services.price_filter_service import PriceFilterService
            self.price_filter = PriceFilterService()
            logger.debug("Price filter service loaded")
        except Exception as e:
            logger.warning(f"Price filter service not available: {e}")
            self.price_filter = None
    
    def _init_signal_embeddings(self):
        """
        Pre-compute embeddings for signal patterns.
        This enables semantic similarity matching for language-independent detection.
        """
        if not self.embedding_model:
            return
        
        try:
            # Define signal patterns in multiple languages
            signal_patterns = {
                'map_routing': [
                    "How do I get there?",
                    "Show me directions",
                    "What's the best route?",
                    "Navigate to this place",
                    "Oraya nasıl gidilir?",
                    "Yol tarifi",
                    "En iyi güzergah nedir?"
                ],
                'weather': [
                    "What's the weather like?",
                    "Will it rain today?",
                    "Temperature forecast",
                    "Is it sunny?",
                    "Hava durumu nasıl?",
                    "Yağmur yağacak mı?",
                    "Sıcaklık kaç derece?"
                ],
                'events': [
                    "What events are happening?",
                    "Any concerts tonight?",
                    "Show me festivals",
                    "Cultural activities",
                    "Hangi etkinlikler var?",
                    "Konser var mı?",
                    "Festival programı"
                ],
                'hidden_gems': [
                    "Local secrets",
                    "Off the beaten path",
                    "Where do locals go?",
                    "Authentic experiences",
                    "Gizli yerler",
                    "Yerel mekanlar",
                    "Turistik olmayan yerler"
                ],
                'budget': [
                    "Cheap options",
                    "Budget-friendly",
                    "Affordable places",
                    "Expensive restaurants",
                    "Ucuz yerler",
                    "Ekonomik",
                    "Pahalı mekanlar"
                ],
                'restaurant': [
                    "Where should I eat?",
                    "Good restaurants",
                    "Food recommendations",
                    "Nerede yemek yenir?",
                    "İyi restoranlar",
                    "Yemek önerisi"
                ],
                'attraction': [
                    "What should I visit?",
                    "Tourist attractions",
                    "Museums to see",
                    "Nereleri gezmeliyim?",
                    "Turistik yerler",
                    "Müzeler"
                ]
            }
            
            # Compute embeddings for each signal
            for signal_name, patterns in signal_patterns.items():
                embeddings = self.embedding_model.encode(patterns, convert_to_numpy=True)
                # Store mean embedding as the signal prototype
                self._signal_embeddings[signal_name] = np.mean(embeddings, axis=0)
            
            logger.debug(f"   Pre-computed {len(self._signal_embeddings)} signal embeddings")
            
        except Exception as e:
            logger.warning(f"Failed to initialize signal embeddings: {e}")
            self._signal_embeddings = {}
    
    def _load_prompts(self):
        """Load Istanbul-specific system prompts"""
        
        self.base_prompt = """You are AI Istanbul, an expert travel assistant for Istanbul, Turkey.

You have deep knowledge of:
🏛️ Attractions: Museums, mosques, palaces, historical sites
🍽️ Restaurants: Authentic Turkish cuisine, international options
🚇 Transportation: Metro, bus, ferry, tram routes
🏘️ Neighborhoods: Districts, areas, local culture
🎭 Events: Concerts, festivals, cultural activities
💎 Hidden Gems: Local favorites, off-the-beaten-path spots

Guidelines:
1. Provide specific names, locations, and details
2. Use provided database context
3. Include practical info (hours, prices, directions)
4. Be enthusiastic about Istanbul
5. Respond in the same language as the query
6. Never make up information - use context only

Format:
- Start with direct answer
- List 3-5 specific recommendations
- Include practical details
- Add a local tip or insight"""

        self.intent_prompts = {
            'restaurant': """
Focus on restaurants from the provided database context.
Include: name, location, cuisine, price range, rating.
Mention dietary options if relevant.""",

            'attraction': """
Focus on attractions and museums from the provided context.
Include: name, district, description, opening hours, ticket price.
Prioritize based on location and interests.""",

            'transportation': """
Provide clear transportation directions.
Include: metro lines, bus numbers, ferry routes.
Mention transfer points and approximate times.
If available, include a map visualization link.""",

            'neighborhood': """
Describe the neighborhood character and highlights.
Include: atmosphere, best areas, local tips.
Mention nearby attractions and dining.""",

            'events': """
Focus on current and upcoming events.
Include: event name, date, location, price.
Prioritize cultural and authentic experiences.""",

            'weather': """
Provide weather-aware recommendations.
Include current conditions and activity suggestions.
Recommend indoor options for bad weather, outdoor for good weather.""",

            'hidden_gems': """
Focus on local secrets and off-the-beaten-path spots.
Include authentic experiences away from tourist crowds.
Mention accessibility and best times to visit.""",

            'general': """
Provide helpful Istanbul travel information.
Draw from all available context.
Be comprehensive but concise."""
        }
    
    async def process_query(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en",
        intent: Optional[str] = None,
        max_tokens: int = 250
    ) -> Dict[str, Any]:
        """
        Process query using SIGNAL-BASED detection (multi-intent support)
        
        Pipeline:
        1. Check cache
        2. Detect service signals (multi-intent, semantic)
        3. Extract GPS location (if provided)
        4. Build smart database context based on signals
        5. Get RAG embeddings (if available)
        6. Conditionally call expensive services (maps, weather, events, etc.)
        7. Construct signal-aware prompt
        8. Call RunPod LLM
        9. Cache signals and response
        10. Return with metadata
        
        Args:
            query: User query string
            user_id: User identifier
            session_id: Session identifier for context tracking
            user_location: User GPS location {"lat": float, "lon": float}
            language: Response language (en/tr)
            intent: Pre-detected intent (optional, for backward compatibility)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with response, map_data (if applicable), signals, and metadata
        """
        
        self.stats["total_queries"] += 1
        start_time = datetime.now()
        
        logger.info(f"🔍 Processing query: {query[:50]}...")
        if user_location:
            logger.info(f"📍 User GPS: ({user_location.get('lat'):.4f}, {user_location.get('lon'):.4f})")
        
        # Step 1: Check cache
        cache_key = self._get_cache_key(query, language)
        cached_response = await self._get_cached_response(cache_key)
        
        if cached_response:
            self.stats["cache_hits"] += 1
            logger.info("✅ Cache hit!")
            return cached_response
        
        # Step 2: Detect service signals (NEW: multi-intent, semantic)
        signals = await self._detect_service_signals(query, user_location)
        active_signals = [k for k, v in signals.items() if v]
        logger.info(f"   Signals detected: {', '.join(active_signals) if active_signals else 'none'}")
        
        # Track multi-signal queries
        if len(active_signals) > 2:
            self.stats["multi_signal_queries"] += 1
            logger.info(f"   🎯 Multi-signal query detected ({len(active_signals)} signals)")
        
        # Step 3: Build smart context based on signals
        db_context = await self._build_smart_context(query, signals)
        logger.info(f"   DB Context: {len(db_context)} chars")
        
        # Step 4: Get RAG context (if available)
        rag_context = await self._get_rag_context(query)
        logger.info(f"   RAG Context: {len(rag_context)} chars")
        
        # Step 5: Conditionally call EXPENSIVE services based on signals
        map_data = None
        if signals['needs_map'] or signals['needs_gps_routing']:
            logger.info("🗺️ Map visualization needed - routing to Istanbul AI system")
            try:
                self.stats["map_requests"] += 1
                map_data = await self._get_map_visualization(
                    query, 'route_planning', user_id, language, user_location
                )
                logger.info(f"   ✅ Map data generated: {bool(map_data)}")
            except Exception as e:
                logger.warning(f"   ⚠️ Map generation failed: {e}")
                map_data = None
        
        weather_context = ""
        if signals['needs_weather'] and self.weather_service:
            self.stats["weather_requests"] += 1
            weather_context = await self._get_weather_context(query)
            logger.info(f"   ☀️ Weather context: {len(weather_context)} chars")
        
        events_context = ""
        if signals['needs_events'] and self.events_service:
            events_context = await self._get_events_context()
            logger.info(f"   🎭 Events context: {len(events_context)} chars")
        
        hidden_gems_context = ""
        if signals['needs_hidden_gems'] and self.hidden_gems_handler:
            self.stats["hidden_gems_requests"] += 1
            hidden_gems_context = await self._get_hidden_gems_context(query)
            logger.info(f"   💎 Hidden gems context: {len(hidden_gems_context)} chars")
        
        # Step 6: Build signal-aware system prompt
        system_prompt = self._build_system_prompt(signals)
        
        # Step 7: Build full prompt with all contexts
        full_prompt = self._build_prompt_with_signals(
            query=query,
            signals=signals,
            system_prompt=system_prompt,
            db_context=db_context,
            rag_context=rag_context,
            weather_context=weather_context,
            events_context=events_context,
            hidden_gems_context=hidden_gems_context,
            language=language
        )
        
        # Step 8: Call RunPod LLM
        try:
            self.stats["llm_calls"] += 1
            
            response_data = await self.llm.generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            if not response_data or "generated_text" not in response_data:
                raise Exception("Invalid LLM response")
            
            response_text = response_data["generated_text"]
            
            # Build result with signals and map_data
            result = {
                "status": "success",
                "response": response_text,
                "map_data": map_data,
                "signals": signals,  # Include detected signals
                "metadata": {
                    "signals_detected": active_signals,
                    "context_used": bool(db_context),
                    "rag_used": bool(rag_context),
                    "map_generated": bool(map_data),
                    "weather_used": bool(weather_context),
                    "events_used": bool(events_context),
                    "hidden_gems_used": bool(hidden_gems_context),
                    "source": "runpod_llm",
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "cached": False,
                    "multi_intent": len(active_signals) > 2
                }
            }
            
            # Step 9: Cache response
            await self._cache_response(cache_key, result)
            
            logger.info(f"✅ Query processed in {result['metadata']['processing_time']:.2f}s")
            if map_data:
                logger.info(f"   🗺️ Map visualization included")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            self.stats["fallback_calls"] += 1
            
            # Fallback to RAG-only or database context
            return await self._fallback_response(
                query=query,
                intent='general',
                db_context=db_context,
                rag_context=rag_context,
                map_data=map_data
            )
    
    def _detect_intent(self, query: str) -> str:
        """
        Simple keyword-based intent detection
        
        DEPRECATED: Use _detect_service_signals() instead for multi-intent support.
        Kept for backward compatibility.
        
        Categories:
        - restaurant: Food, dining, eat
        - attraction: Visit, see, museum
        - transportation: Metro, bus, get to (with map visualization)
        - route_planning: Directions, how to get (with map visualization)
        - neighborhood: District, area, stay
        - events: Concert, festival, show
        - weather: Weather, temperature, rain, forecast
        - general: Everything else
        """
        q = query.lower()
        
        # Route planning keywords (checked first, more specific)
        if any(w in q for w in [
            'how to get', 'directions', 'route', 'navigate', 'way to',
            'best way', 'nasıl gidilir', 'yol tarifi', 'güzergah'
        ]):
            return 'route_planning'
        
        # Weather keywords (before general checks)
        elif any(w in q for w in [
            'weather', 'temperature', 'rain', 'sunny', 'cold', 'hot', 
            'forecast', 'climate', 'will it rain', 'hava durumu', 'sıcaklık'
        ]):
            return 'weather'
        
        # Restaurant keywords
        elif any(w in q for w in [
            'restaurant', 'eat', 'food', 'dinner', 'lunch', 'breakfast',
            'cafe', 'coffee', 'cuisine', 'yemek', 'lokanta'
        ]):
            return 'restaurant'
        
        # Attraction keywords
        elif any(w in q for w in [
            'visit', 'see', 'attraction', 'place', 'museum', 'mosque',
            'palace', 'tower', 'church', 'ziyaret', 'gör'
        ]):
            return 'attraction'
        
        # Transportation keywords
        elif any(w in q for w in [
            'metro', 'bus', 'ferry', 'transport', 'how to reach',
            'tram', 'istanbulkart', 'otobüs', 'vapur'
        ]):
            return 'transportation'
        
        # Neighborhood keywords
        elif any(w in q for w in [
            'neighborhood', 'district', 'area', 'where to stay', 'region',
            'semt', 'mahalle', 'bölge'
        ]):
            return 'neighborhood'
        
        # Events keywords
        elif any(w in q for w in [
            'event', 'concert', 'show', 'festival', 'activity',
            'etkinlik', 'konser', 'festival'
        ]):
            return 'events'
        
        # Hidden gems keywords
        elif any(w in q for w in [
            'hidden', 'secret', 'local', 'authentic', 'off the beaten',
            'locals go', 'gizli', 'yerel', 'saklı'
        ]):
            return 'hidden_gems'
        
        else:
            return 'general'
    
    async def _build_database_context(
        self,
        query: str,
        intent: str
    ) -> str:
        """
        Build context from database based on intent
        
        DEPRECATED: Use _build_smart_context() for signal-based approach.
        Kept for backward compatibility.
        """
        
        context_parts = []
        
        try:
            if intent == 'restaurant':
                context_parts.append(await self._get_restaurant_context(query))
            
            elif intent == 'attraction':
                context_parts.append(await self._get_attraction_context(query))
            
            elif intent == 'transportation':
                context_parts.append(await self._get_transportation_context())
            
            elif intent == 'neighborhood':
                context_parts.append(await self._get_neighborhood_context(query))
            
            elif intent == 'events':
                context_parts.append(await self._get_events_context())
            
            elif intent == 'weather':
                context_parts.append(await self._get_weather_context(query))
            
            elif intent == 'hidden_gems':
                context_parts.append(await self._get_hidden_gems_context(query))
            
            else:
                # General: Include mix of everything
                context_parts.append(await self._get_restaurant_context(query, limit=3))
                context_parts.append(await self._get_attraction_context(query, limit=3))
        
        except Exception as e:
            logger.error(f"Error building database context: {e}")
        
        return "\n\n".join([c for c in context_parts if c])
    
    async def _build_smart_context(
        self, 
        query: str, 
        signals: Dict[str, bool]
    ) -> str:
        """
        Build database context smartly based on detected signals.
        Query only what's likely needed for better performance.
        
        NEW APPROACH: Signal-aware context building
        - Prioritizes relevant data based on signals
        - Supports multi-intent queries
        - Applies filters (budget, location) when needed
        
        Args:
            query: User query string
            signals: Detected service signals
            
        Returns:
            Formatted database context string
        """
        context_parts = []
        
        try:
            # Restaurant priority
            if signals['likely_restaurant']:
                context_parts.append(
                    await self._get_restaurant_context(query, limit=10)
                )
                # Also include nearby attractions for context
                if signals['likely_attraction'] or signals['mentions_location']:
                    context_parts.append(
                        await self._get_attraction_context(query, limit=3)
                    )
            
            # Attraction priority
            elif signals['likely_attraction']:
                context_parts.append(
                    await self._get_attraction_context(query, limit=10)
                )
                # Include nearby restaurants for recommendations
                context_parts.append(
                    await self._get_restaurant_context(query, limit=3)
                )
            
            # No clear domain - balanced mix
            else:
                context_parts.append(
                    await self._get_restaurant_context(query, limit=5)
                )
                context_parts.append(
                    await self._get_attraction_context(query, limit=5)
                )
            
            # Add transportation if needed
            if signals['needs_map'] or signals['needs_gps_routing']:
                context_parts.append(
                    await self._get_transportation_context()
                )
            
            # Apply budget filtering if needed
            if signals['has_budget_constraint'] and self.price_filter:
                # Price filter will be applied within context methods
                logger.debug("   💰 Budget filtering applied")
        
        except Exception as e:
            logger.error(f"Error building smart context: {e}")
        
        return "\n\n".join([c for c in context_parts if c])
    
    def _build_system_prompt(self, signals: Dict[str, bool]) -> str:
        """
        Build system prompt that adapts to detected signals.
        Guides LLM based on what services were called.
        
        Args:
            signals: Detected service signals
            
        Returns:
            Signal-aware system prompt string
        """
        base_prompt = """You are AI Istanbul, an expert travel assistant for Istanbul, Turkey.

You have deep knowledge of:
🏛️ Attractions: Museums, mosques, palaces, historical sites
🍽️ Restaurants: Authentic Turkish cuisine, international options
🚇 Transportation: Metro, bus, ferry, tram routes
🏘️ Neighborhoods: Districts, areas, local culture
🎭 Events: Concerts, festivals, cultural activities
💎 Hidden Gems: Local favorites, off-the-beaten-path spots

Guidelines:
1. Provide specific names, locations, and details
2. Use provided database context
3. Include practical info (hours, prices, directions)
4. Be enthusiastic about Istanbul
5. Respond in the same language as the query
6. Never make up information - use context only"""

        # Add signal-specific instructions
        adaptations = []
        
        if signals.get('needs_map'):
            adaptations.append("🗺️ IMPORTANT: A map has been generated. Reference it in your response and explain how to use it.")
        
        if signals.get('needs_weather'):
            adaptations.append("☀️ Consider current weather conditions in your recommendations. Suggest indoor options for bad weather, outdoor for good weather.")
        
        if signals.get('needs_events'):
            adaptations.append("🎭 Event information is provided. Highlight current and upcoming events relevant to the query.")
        
        if signals.get('needs_hidden_gems'):
            adaptations.append("💎 Focus on authentic, local experiences away from tourist crowds. Mention accessibility and best times.")
        
        if signals.get('has_budget_constraint'):
            adaptations.append("💰 Prioritize budget-appropriate options. Be clear about price ranges.")
        
        if signals.get('has_user_location'):
            adaptations.append("📍 User's current location is provided. Prioritize nearby options and mention distances.")
        
        if adaptations:
            base_prompt += "\n\n" + "\n".join(adaptations)
        
        return base_prompt
    
    def _build_prompt_with_signals(
        self,
        query: str,
        signals: Dict[str, bool],
        system_prompt: str,
        db_context: str,
        rag_context: str,
        weather_context: str,
        events_context: str,
        hidden_gems_context: str,
        language: str
    ) -> str:
        """
        Combine all context into final prompt with signal awareness.
        
        Args:
            query: User query
            signals: Detected signals
            system_prompt: Signal-aware system prompt
            db_context: Database context
            rag_context: RAG context
            weather_context: Weather-specific context
            events_context: Events context
            hidden_gems_context: Hidden gems context
            language: Response language
            
        Returns:
            Complete formatted prompt for LLM
        """
        parts = [system_prompt, ""]
        
        # Add contexts in priority order
        if db_context:
            parts.append("**DATABASE CONTEXT:**")
            parts.append(db_context)
            parts.append("")
        
        if weather_context:
            parts.append("**WEATHER INFORMATION:**")
            parts.append(weather_context)
            parts.append("")
        
        if events_context:
            parts.append("**EVENTS:**")
            parts.append(events_context)
            parts.append("")
        
        if hidden_gems_context:
            parts.append("**HIDDEN GEMS:**")
            parts.append(hidden_gems_context)
            parts.append("")
        
        if rag_context:
            parts.append("**KNOWLEDGE BASE:**")
            parts.append(rag_context)
            parts.append("")
        
        # Add query
        parts.append(f"**USER QUERY ({language.upper()}):**")
        parts.append(query)
        parts.append("")
        parts.append("**YOUR RESPONSE:**")
        
        return "\n".join(parts)
    
    async def _detect_service_signals(
        self, 
        query: str, 
        user_location: Optional[Dict[str, float]] = None
    ) -> Dict[str, bool]:
        """
        Detect which services are needed for this query using semantic similarity.
        
        NEW APPROACH: Multi-signal detection with:
        - Semantic embeddings for language independence (Turkish + English)
        - Keyword fallback for reliability
        - Caching for performance
        - Multi-intent support
        
        This is a lightweight approach that:
        - Only detects EXPENSIVE operations explicitly (maps, GPS routing)
        - Lets LLM handle nuanced understanding
        - Supports multi-service queries naturally
        
        Args:
            query: User query string
            user_location: User GPS coordinates
            
        Returns:
            Dict of signal names to boolean values
        """
        
        # Check cache first
        signal_cache_key = self._get_signal_cache_key(query)
        cached_signals = await self._get_cached_signals(signal_cache_key)
        if cached_signals:
            self.stats["signal_cache_hits"] += 1
            logger.debug("   📦 Signal cache hit")
            # Update user location signal if provided
            if user_location:
                cached_signals['has_user_location'] = True
            return cached_signals
        
        # Initialize signals
        signals = {
            # Expensive operations - detect explicitly
            'needs_map': False,
            'needs_gps_routing': False,
            
            # Service signals
            'needs_weather': False,
            'needs_events': False,
            'needs_hidden_gems': False,
            
            # Budget/price filtering
            'has_budget_constraint': False,
            
            # Location context
            'has_user_location': user_location is not None,
            'mentions_location': False,
            
            # Domain hints (lightweight, not rigid)
            'likely_restaurant': False,
            'likely_attraction': False,
        }
        
        # Use semantic similarity if available
        if self.embedding_model and self._signal_embeddings:
            try:
                semantic_signals = await self._detect_signals_semantic(query)
                # Merge semantic signals
                for key, value in semantic_signals.items():
                    if key in signals:
                        signals[key] = value
                logger.debug("   🧠 Semantic signal detection used")
            except Exception as e:
                logger.warning(f"Semantic signal detection failed: {e}, falling back to keywords")
        
        # Keyword fallback/enhancement (always run for reliability)
        keyword_signals = self._detect_signals_keywords(query)
        
        # Combine: Use OR logic (if either detects, signal is True)
        for key in signals.keys():
            if key in keyword_signals:
                signals[key] = signals[key] or keyword_signals[key]
        
        # Cache signals for future use
        await self._cache_signals(signal_cache_key, signals)
        
        return signals
    
    async def _detect_signals_semantic(self, query: str) -> Dict[str, bool]:
        """
        Detect signals using semantic similarity with pre-computed embeddings.
        Language-independent approach using sentence transformers.
        
        Args:
            query: User query string
            
        Returns:
            Dict of signal names to boolean values
        """
        if not self.embedding_model or not self._signal_embeddings:
            return {}
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            
            # Calculate similarity with each signal
            similarities = {}
            for signal_name, signal_embedding in self._signal_embeddings.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, signal_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(signal_embedding)
                )
                similarities[signal_name] = float(similarity)
            
            # Map signal names to our signal keys with threshold
            SIMILARITY_THRESHOLD = 0.4  # Adjust based on testing
            
            signals = {
                'needs_map': similarities.get('map_routing', 0) > SIMILARITY_THRESHOLD,
                'needs_gps_routing': similarities.get('map_routing', 0) > 0.5,  # Higher threshold
                'needs_weather': similarities.get('weather', 0) > SIMILARITY_THRESHOLD,
                'needs_events': similarities.get('events', 0) > SIMILARITY_THRESHOLD,
                'needs_hidden_gems': similarities.get('hidden_gems', 0) > SIMILARITY_THRESHOLD,
                'has_budget_constraint': similarities.get('budget', 0) > SIMILARITY_THRESHOLD,
                'likely_restaurant': similarities.get('restaurant', 0) > SIMILARITY_THRESHOLD,
                'likely_attraction': similarities.get('attraction', 0) > SIMILARITY_THRESHOLD,
            }
            
            # Log high-confidence detections
            high_confidence = {k: v for k, v in similarities.items() if v > 0.6}
            if high_confidence:
                logger.debug(f"   High confidence signals: {high_confidence}")
            
            return signals
            
        except Exception as e:
            logger.warning(f"Semantic signal detection error: {e}")
            return {}
    
    def _detect_signals_keywords(self, query: str) -> Dict[str, bool]:
        """
        Keyword-based signal detection as fallback/enhancement.
        Reliable for explicit mentions.
        
        Args:
            query: User query string
            
        Returns:
            Dict of signal names to boolean values
        """
        q = query.lower()
        
        return {
            # Map/routing keywords
            'needs_map': any(w in q for w in [
                'how to get', 'directions', 'route', 'navigate', 
                'take me', 'way to', 'path to', 'map',
                'nasıl giderim', 'yol tarifi', 'güzergah', 'harita'
            ]),
            
            'needs_gps_routing': any(w in q for w in [
                'fastest route', 'best route', 'driving directions',
                'walking directions', 'from here',
                'en hızlı yol', 'en iyi güzergah', 'buradan'
            ]),
            
            # Service keywords
            'needs_weather': any(w in q for w in [
                'weather', 'rain', 'temperature', 'sunny', 'cold', 'hot',
                'forecast', 'climate',
                'hava durumu', 'yağmur', 'sıcaklık', 'soğuk', 'sıcak'
            ]),
            
            'needs_events': any(w in q for w in [
                'event', 'concert', 'show', 'festival', 'activity',
                'happening', 'tonight', 'weekend',
                'etkinlik', 'konser', 'festival', 'gösteri'
            ]),
            
            'needs_hidden_gems': any(w in q for w in [
                'hidden', 'secret', 'local', 'authentic', 'off the beaten',
                'locals go', 'locals eat', 'locals love',
                'gizli', 'yerel', 'saklı', 'turistik olmayan'
            ]),
            
            # Budget keywords
            'has_budget_constraint': any(w in q for w in [
                'cheap', 'expensive', 'budget', 'affordable', 'costly',
                'price', 'cost', 'free',
                'ucuz', 'pahalı', 'ekonomik', 'fiyat', 'bedava'
            ]),
            
            # Location keywords
            'mentions_location': any(w in q for w in [
                'near', 'close to', 'around', 'nearby', 'next to',
                'walking distance', 'from here',
                'yakın', 'civarı', 'çevresinde', 'yanında'
            ]),
            
            # Domain keywords
            'likely_restaurant': any(w in q for w in [
                'restaurant', 'food', 'eat', 'dining', 'meal',
                'lunch', 'dinner', 'breakfast', 'cafe', 'coffee',
                'restoran', 'yemek', 'lokanta', 'kahve'
            ]),
            
            'likely_attraction': any(w in q for w in [
                'mosque', 'museum', 'palace', 'attraction', 'visit',
                'see', 'tower', 'church', 'monument', 'site',
                'cami', 'müze', 'saray', 'görmek', 'ziyaret'
            ])
        }
    
    def _get_cache_key(self, query: str, language: str) -> str:
        """Generate cache key from query"""
        key_string = f"{query.lower().strip()}_{language}"
        return f"llm_response:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_signal_cache_key(self, query: str) -> str:
        """Generate cache key for signal detection results"""
        key_string = f"signals:{query.lower().strip()}"
        return f"signals:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _get_cached_signals(self, cache_key: str) -> Optional[Dict[str, bool]]:
        """Get cached signal detection results from Redis"""
        if not self.redis:
            return None
        
        try:
            import json
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Signal cache get failed: {e}")
        
        return None
    
    async def _cache_signals(self, cache_key: str, signals: Dict[str, bool]):
        """Cache signal detection results in Redis"""
        if not self.redis:
            return
        
        try:
            import json
            # Cache signals for 1 hour (they're query-specific)
            self.redis.setex(
                cache_key,
                3600,
                json.dumps(signals)
            )
        except Exception as e:
            logger.debug(f"Signal cache set failed: {e}")
