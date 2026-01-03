"""
Unified Intent Router
=====================
Central router that detects intent from user queries (in any language)
and routes to the appropriate handler.

This replaces the scattered if/else blocks in chat.py with a clean,
extensible architecture.

Features Supported:
1. 🍽️ Restaurants
2. 🏛️ Places & Attractions  
3. 🏘️ Neighborhood Guides
4. 🚇 Transportation
5. 💬 Daily Talks
6. 💎 Hidden Gems/Local Tips
7. 🌦️ Weather Aware
8. 🎭 Events Advising
9. 🗺️ Route Planner

Author: AI Istanbul Team
Date: December 2025
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Import multilingual intent detection
from services.multilingual_intent_keywords import (
    detect_intent_multilingual,
    extract_neighborhood,
    HIDDEN_GEMS_KEYWORDS,
    RESTAURANT_KEYWORDS,
    TRANSPORTATION_KEYWORDS,
    ATTRACTIONS_KEYWORDS,
    WEATHER_KEYWORDS,
    EVENTS_KEYWORDS,
    NEIGHBORHOOD_GUIDE_KEYWORDS,
    ROUTE_PLANNING_KEYWORDS,
    DAILY_TALKS_KEYWORDS,
)


class IntentType(Enum):
    """All supported intent types"""
    RESTAURANT = "restaurant"
    ATTRACTIONS = "attractions"
    NEIGHBORHOOD = "neighborhood_guide"
    TRANSPORTATION = "transportation"
    DAILY_TALKS = "daily_talks"
    HIDDEN_GEMS = "hidden_gems"
    WEATHER = "weather"
    EVENTS = "events"
    ROUTE_PLANNING = "route_planning"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent detection"""
    intent: IntentType
    confidence: float
    language: str
    matched_keywords: List[str]
    neighborhood: Optional[str] = None
    entities: Dict[str, Any] = None
    
    def is_confident(self, threshold: float = 0.3) -> bool:
        """Check if detection confidence meets threshold"""
        return self.confidence >= threshold


@dataclass
class HandlerResult:
    """Result from a feature handler"""
    success: bool
    response: str
    intent: str
    data: Dict[str, Any] = None
    suggestions: List[str] = None
    map_data: Dict[str, Any] = None
    navigation_data: Dict[str, Any] = None
    error: str = None


class UnifiedIntentRouter:
    """
    Central router for all AI Istanbul intents.
    
    Detects user intent in any supported language and routes
    to the appropriate handler for fast, deterministic responses.
    """
    
    def __init__(self, db_session=None):
        """Initialize router with optional database session"""
        self.db = db_session
        self._handlers = {}
        self._initialize_handlers()
        logger.info("✅ Unified Intent Router initialized")
    
    def _initialize_handlers(self):
        """Lazy-load handlers to avoid circular imports"""
        pass  # Handlers loaded on demand
    
    def detect_intent(self, query: str, language: str = None) -> IntentResult:
        """
        Detect intent from user query.
        
        Language detection is NOT done here - let the LLM handle that naturally.
        We only detect INTENT using multilingual keywords (checks all languages).
        
        Args:
            query: User's message
            language: Optional language hint (passed from request, not detected here)
        
        Returns:
            IntentResult with detected intent and confidence
        """
        # Use multilingual intent detection - checks ALL languages for keywords
        # The function now returns 4 values including detected language from keywords
        result = detect_intent_multilingual(query, language)
        
        # Handle both old 3-tuple and new 4-tuple return formats
        if len(result) == 4:
            intent_name, confidence, matched_keywords, detected_lang = result
        else:
            intent_name, confidence, matched_keywords = result
            detected_lang = language or 'en'
        
        # Extract neighborhood if present
        neighborhood = extract_neighborhood(query)
        
        # Map to IntentType enum
        intent_type = IntentType.UNKNOWN
        if intent_name:
            try:
                intent_type = IntentType(intent_name)
            except ValueError:
                intent_type = IntentType.UNKNOWN
        
        return IntentResult(
            intent=intent_type,
            confidence=confidence,
            language=detected_lang,  # Language detected from keyword match
            matched_keywords=matched_keywords,
            neighborhood=neighborhood,
            entities={'neighborhood': neighborhood} if neighborhood else {}
        )
    
    async def route(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None,
        session_id: str = 'default',
        user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[HandlerResult]:
        """
        Route query to appropriate handler.
        
        Args:
            query: User's message
            user_location: Optional GPS coordinates {lat, lon}
            session_id: Session identifier
            user_context: Additional context (preferences, history, etc.)
        
        Returns:
            HandlerResult if handled, None if should fall through to LLM
        """
        # Detect intent
        intent_result = self.detect_intent(query)
        
        logger.info(
            f"🎯 Intent detected: {intent_result.intent.value} "
            f"(confidence: {intent_result.confidence:.2f}, "
            f"language: {intent_result.language}, "
            f"keywords: {intent_result.matched_keywords})"
        )
        
        # Check confidence threshold
        if not intent_result.is_confident(threshold=0.3):
            logger.info("❓ Low confidence - falling through to LLM")
            return None
        
        # Route to handler
        handler_result = await self._dispatch_to_handler(
            intent_result,
            query,
            user_location,
            session_id,
            user_context
        )
        
        return handler_result
    
    async def _dispatch_to_handler(
        self,
        intent_result: IntentResult,
        query: str,
        user_location: Optional[Dict[str, float]],
        session_id: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Optional[HandlerResult]:
        """Dispatch to the appropriate handler based on intent"""
        
        intent = intent_result.intent
        
        # ===== HIDDEN GEMS =====
        if intent == IntentType.HIDDEN_GEMS:
            return await self._handle_hidden_gems(
                query, user_location, session_id, intent_result
            )
        
        # ===== RESTAURANTS =====
        elif intent == IntentType.RESTAURANT:
            return await self._handle_restaurant(
                query, user_location, intent_result
            )
        
        # ===== TRANSPORTATION =====
        elif intent == IntentType.TRANSPORTATION:
            return await self._handle_transportation(
                query, user_location, user_context, intent_result
            )
        
        # ===== ATTRACTIONS =====
        elif intent == IntentType.ATTRACTIONS:
            return await self._handle_attractions(
                query, intent_result
            )
        
        # ===== WEATHER =====
        elif intent == IntentType.WEATHER:
            return await self._handle_weather(
                query, user_location, intent_result
            )
        
        # ===== EVENTS =====
        elif intent == IntentType.EVENTS:
            return await self._handle_events(
                query, intent_result
            )
        
        # ===== NEIGHBORHOOD GUIDE =====
        elif intent == IntentType.NEIGHBORHOOD:
            return await self._handle_neighborhood(
                query, intent_result
            )
        
        # ===== ROUTE PLANNING =====
        elif intent == IntentType.ROUTE_PLANNING:
            return await self._handle_route_planning(
                query, user_location, user_context, intent_result
            )
        
        # ===== DAILY TALKS =====
        elif intent == IntentType.DAILY_TALKS:
            return await self._handle_daily_talks(
                query, intent_result
            )
        
        return None
    
    # =========================================================================
    # HANDLER IMPLEMENTATIONS
    # =========================================================================
    
    async def _handle_hidden_gems(
        self,
        query: str,
        user_location: Optional[Dict[str, float]],
        session_id: str,
        intent: IntentResult
    ) -> Optional[HandlerResult]:
        """Handle hidden gems queries"""
        try:
            from services.hidden_gems_gps_integration import get_hidden_gems_gps_integration
            
            handler = get_hidden_gems_gps_integration(self.db)
            result = handler.handle_hidden_gem_chat_request(
                message=query,
                user_location=user_location,
                session_id=session_id
            )
            
            if result:
                return HandlerResult(
                    success=not result.get('error'),
                    response=result.get('message', ''),
                    intent='hidden_gems',
                    data={'gems': result.get('gems', [])},
                    suggestions=result.get('suggestions', [
                        "Show hidden restaurants",
                        "Navigate to first gem",
                        "Show more hidden gems"
                    ]),
                    map_data=result.get('map_data'),
                    navigation_data=result.get('navigation_data'),
                    error=result.get('error')
                )
        except Exception as e:
            logger.warning(f"Hidden gems handler error: {e}")
        
        return None
    
    async def _handle_restaurant(
        self,
        query: str,
        user_location: Optional[Dict[str, float]],
        intent: IntentResult
    ) -> Optional[HandlerResult]:
        """Handle restaurant queries"""
        try:
            from services.restaurant_query_handler import get_restaurant_handler
            
            handler = get_restaurant_handler(self.db)
            result = await handler.handle_query(
                query=query,
                user_location=user_location,
                neighborhood=intent.neighborhood
            )
            
            if result:
                return HandlerResult(
                    success=True,
                    response=result.get('response', ''),
                    intent='restaurant',
                    data={'restaurants': result.get('restaurants', [])},
                    suggestions=result.get('suggestions', [
                        "Show vegetarian options",
                        "Find cheaper restaurants",
                        "Show restaurants with sea view"
                    ]),
                    map_data=result.get('map_data')
                )
        except ImportError:
            logger.warning("Restaurant handler not available")
        except Exception as e:
            logger.warning(f"Restaurant handler error: {e}")
        
        return None
    
    async def _handle_transportation(
        self,
        query: str,
        user_location: Optional[Dict[str, float]],
        user_context: Optional[Dict[str, Any]],
        intent: IntentResult
    ) -> Optional[HandlerResult]:
        """Handle transportation queries"""
        try:
            from services.ai_chat_route_integration import get_chat_route_handler
            
            # Add user_location to user_context if available
            context = user_context or {}
            if user_location:
                context['gps'] = user_location
                context['location'] = user_location
            
            handler = get_chat_route_handler()
            # FIXED: Await the async function and pass user_location via context
            result = await handler.handle_route_request(
                message=query,
                user_context=context
            )
            
            if result:
                # Extract route_data for processing
                route_data = result.get('route_data', {})
                
                # Build proper map_data with route information at top level
                map_data = {
                    'type': result.get('type', 'route'),
                    'route_data': route_data,
                    # Extract origin/destination to top level for frontend compatibility
                    'origin': route_data.get('origin') or route_data.get('start'),
                    'destination': route_data.get('destination') or route_data.get('end'),
                    'total_time': route_data.get('total_time') or route_data.get('duration'),
                    'total_distance': route_data.get('total_distance') or route_data.get('distance'),
                }
                
                return HandlerResult(
                    success=result.get('type') != 'error',
                    response=result.get('message', ''),
                    intent='transportation',
                    data=route_data,
                    suggestions=result.get('suggestions', [
                        "Show alternative routes",
                        "How long by taxi?",
                        "Show walking directions"
                    ]),
                    map_data=map_data,  # Now includes origin/destination at top level
                    navigation_data=result.get('navigation_data')
                )
        except ImportError:
            logger.warning("Transportation handler not available")
        except Exception as e:
            logger.warning(f"Transportation handler error: {e}")
        
        return None
    
    async def _handle_attractions(
        self,
        query: str,
        intent: IntentResult
    ) -> Optional[HandlerResult]:
        """Handle attractions/places queries"""
        try:
            # Try loading attractions data
            from data.attractions_database import get_attractions
            
            attractions = get_attractions(
                neighborhood=intent.neighborhood,
                query=query
            )
            
            if attractions:
                response = self._format_attractions_response(attractions, intent.language)
                return HandlerResult(
                    success=True,
                    response=response,
                    intent='attractions',
                    data={'attractions': attractions},
                    suggestions=[
                        f"How do I get to {attractions[0]['name']}?" if attractions else "Show popular attractions",
                        "Show museums",
                        "What are the opening hours?"
                    ]
                )
        except ImportError:
            logger.warning("Attractions database not available")
        except Exception as e:
            logger.warning(f"Attractions handler error: {e}")
        
        return None
    
    async def _handle_weather(
        self,
        query: str,
        user_location: Optional[Dict[str, float]],
        intent: IntentResult
    ) -> Optional[HandlerResult]:
        """Handle weather queries"""
        try:
            from services.weather_service import get_weather_service
            
            service = get_weather_service()
            weather = await service.get_current_weather()
            
            if weather:
                response = self._format_weather_response(weather, intent.language)
                return HandlerResult(
                    success=True,
                    response=response,
                    intent='weather',
                    data=weather,
                    suggestions=[
                        "What should I wear today?",
                        "Best indoor activities?",
                        "Weather forecast for tomorrow"
                    ]
                )
        except ImportError:
            logger.warning("Weather service not available")
        except Exception as e:
            logger.warning(f"Weather handler error: {e}")
        
        return None
    
    async def _handle_events(
        self,
        query: str,
        intent: IntentResult
    ) -> Optional[HandlerResult]:
        """Handle events queries"""
        try:
            from services.events_service import get_events_service
            
            service = get_events_service()
            events = await service.get_upcoming_events(
                neighborhood=intent.neighborhood
            )
            
            if events:
                response = self._format_events_response(events, intent.language)
                return HandlerResult(
                    success=True,
                    response=response,
                    intent='events',
                    data={'events': events},
                    suggestions=[
                        "Show concerts this weekend",
                        "Family-friendly events?",
                        "Free events today"
                    ]
                )
        except ImportError:
            logger.warning("Events service not available")
        except Exception as e:
            logger.warning(f"Events handler error: {e}")
        
        return None
    
    async def _handle_neighborhood(
        self,
        query: str,
        intent: IntentResult
    ) -> Optional[HandlerResult]:
        """Handle neighborhood guide queries"""
        neighborhood = intent.neighborhood
        
        if not neighborhood:
            return HandlerResult(
                success=True,
                response="Which neighborhood would you like to explore? Popular areas include Sultanahmet, Beyoğlu, Kadıköy, Balat, and Beşiktaş.",
                intent='neighborhood_guide',
                suggestions=[
                    "Tell me about Balat",
                    "What's Kadıköy like?",
                    "Guide to Sultanahmet"
                ]
            )
        
        try:
            from istanbul_ai.services.neighborhood_guide_service import get_neighborhood_guide
            
            guide = get_neighborhood_guide()
            info = await guide.get_neighborhood_info(neighborhood)
            
            if info:
                return HandlerResult(
                    success=True,
                    response=info.get('description', ''),
                    intent='neighborhood_guide',
                    data=info,
                    suggestions=[
                        f"Hidden gems in {neighborhood}",
                        f"Best restaurants in {neighborhood}",
                        f"How to get to {neighborhood}"
                    ]
                )
        except ImportError:
            # Fallback to basic info
            return self._get_basic_neighborhood_info(neighborhood, intent.language)
        except Exception as e:
            logger.warning(f"Neighborhood handler error: {e}")
        
        return None
    
    async def _handle_route_planning(
        self,
        query: str,
        user_location: Optional[Dict[str, float]],
        user_context: Optional[Dict[str, Any]],
        intent: IntentResult
    ) -> Optional[HandlerResult]:
        """Handle route planning/itinerary queries"""
        try:
            from services.route_planner import get_route_planner
            
            planner = get_route_planner()
            itinerary = await planner.plan_day(
                query=query,
                start_location=user_location,
                preferences=user_context.get('preferences', {}) if user_context else {}
            )
            
            if itinerary:
                return HandlerResult(
                    success=True,
                    response=itinerary.get('summary', ''),
                    intent='route_planning',
                    data=itinerary,
                    suggestions=[
                        "Show on map",
                        "Add lunch break",
                        "Skip first stop"
                    ],
                    map_data=itinerary.get('map_data')
                )
        except ImportError:
            logger.warning("Route planner not available")
        except Exception as e:
            logger.warning(f"Route planning error: {e}")
        
        return None
    
    async def _handle_daily_talks(
        self,
        query: str,
        intent: IntentResult
    ) -> Optional[HandlerResult]:
        """Handle casual conversation"""
        query_lower = query.lower()
        lang = intent.language
        
        # Greeting responses
        greetings = {
            'en': {
                'response': "Hello! 👋 Welcome to Istanbul AI Guide. How can I help you explore this amazing city today?",
                'suggestions': ["Show hidden gems", "Best restaurants nearby", "What to do today"]
            },
            'tr': {
                'response': "Merhaba! 👋 İstanbul AI Rehberi'ne hoş geldiniz. Bugün bu harika şehri keşfetmenize nasıl yardımcı olabilirim?",
                'suggestions': ["Gizli mekanları göster", "Yakındaki en iyi restoranlar", "Bugün ne yapmalı"]
            },
            'ru': {
                'response': "Привет! 👋 Добро пожаловать в AI-гид по Стамбулу. Как я могу помочь вам исследовать этот удивительный город?",
                'suggestions': ["Покажи скрытые места", "Лучшие рестораны рядом", "Что делать сегодня"]
            },
            'de': {
                'response': "Hallo! 👋 Willkommen beim Istanbul AI Guide. Wie kann ich Ihnen helfen, diese wunderbare Stadt zu erkunden?",
                'suggestions': ["Zeig versteckte Orte", "Beste Restaurants in der Nähe", "Was tun heute"]
            },
            'ar': {
                'response': "مرحبا! 👋 أهلاً بك في دليل اسطنبول الذكي. كيف يمكنني مساعدتك في استكشاف هذه المدينة الرائعة؟",
                'suggestions': ["أظهر الأماكن المخفية", "أفضل المطاعم القريبة", "ماذا أفعل اليوم"]
            }
        }
        
        # Check for greetings
        greeting_words = ['hello', 'hi', 'hey', 'merhaba', 'selam', 'привет', 'hallo', 'مرحبا', 'أهلا']
        if any(g in query_lower for g in greeting_words):
            response_data = greetings.get(lang, greetings['en'])
            return HandlerResult(
                success=True,
                response=response_data['response'],
                intent='daily_talks',
                suggestions=response_data['suggestions']
            )
        
        # Thanks responses
        thanks_words = ['thanks', 'thank you', 'teşekkür', 'sağol', 'спасибо', 'danke', 'شكرا']
        if any(t in query_lower for t in thanks_words):
            responses = {
                'en': "You're welcome! 😊 Feel free to ask anything else about Istanbul.",
                'tr': "Rica ederim! 😊 İstanbul hakkında başka bir şey sormak isterseniz buradayım.",
                'ru': "Пожалуйста! 😊 Не стесняйтесь спрашивать что угодно о Стамбуле.",
                'de': "Gern geschehen! 😊 Fragen Sie gerne alles andere über Istanbul.",
                'ar': "عفواً! 😊 لا تتردد في السؤال عن أي شيء آخر عن اسطنبول."
            }
            return HandlerResult(
                success=True,
                response=responses.get(lang, responses['en']),
                intent='daily_talks',
                suggestions=greetings.get(lang, greetings['en'])['suggestions']
            )
        
        # Help requests
        help_words = ['help', 'yardım', 'помощь', 'hilfe', 'مساعدة']
        if any(h in query_lower for h in help_words):
            help_text = {
                'en': """I can help you with:
🍽️ Restaurant recommendations
🏛️ Places & attractions
🏘️ Neighborhood guides
🚇 Transportation & directions
💎 Hidden gems & local tips
🌦️ Weather-aware suggestions
🎭 Events & activities
🗺️ Day trip planning

Just ask me anything about Istanbul!""",
                'tr': """Size yardımcı olabileceğim konular:
🍽️ Restoran önerileri
🏛️ Gezilecek yerler
🏘️ Mahalle rehberleri
🚇 Ulaşım ve yol tarifi
💎 Gizli mekanlar ve yerel ipuçları
🌦️ Hava durumuna göre öneriler
🎭 Etkinlikler
🗺️ Günlük gezi planlaması

İstanbul hakkında her şeyi sorabilirsiniz!"""
            }
            return HandlerResult(
                success=True,
                response=help_text.get(lang, help_text['en']),
                intent='daily_talks',
                suggestions=greetings.get(lang, greetings['en'])['suggestions']
            )
        
        # Default - don't handle, let LLM handle
        return None
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _format_attractions_response(self, attractions: List[Dict], language: str) -> str:
        """Format attractions list into response text"""
        if not attractions:
            return "No attractions found for your query."
        
        lines = [f"🏛️ Found {len(attractions)} attractions:\n"]
        for i, attr in enumerate(attractions[:5], 1):
            lines.append(f"{i}. **{attr.get('name', 'Unknown')}** - {attr.get('description', '')[:100]}...")
        
        return "\n".join(lines)
    
    def _format_weather_response(self, weather: Dict, language: str) -> str:
        """Format weather data into response text"""
        temp = weather.get('temperature', 'N/A')
        condition = weather.get('condition', 'Unknown')
        
        templates = {
            'en': f"🌦️ Current weather in Istanbul: {temp}°C, {condition}",
            'tr': f"🌦️ İstanbul'da şu anki hava: {temp}°C, {condition}",
            'ru': f"🌦️ Текущая погода в Стамбуле: {temp}°C, {condition}",
            'de': f"🌦️ Aktuelles Wetter in Istanbul: {temp}°C, {condition}",
            'ar': f"🌦️ الطقس الحالي في اسطنبول: {temp}°C, {condition}"
        }
        
        return templates.get(language, templates['en'])
    
    def _format_events_response(self, events: List[Dict], language: str) -> str:
        """Format events list into response text"""
        if not events:
            return "No upcoming events found."
        
        lines = [f"🎭 Found {len(events)} upcoming events:\n"]
        for event in events[:5]:
            lines.append(f"• **{event.get('name', 'Unknown')}** - {event.get('date', 'TBA')}")
        
        return "\n".join(lines)
    
    def _get_basic_neighborhood_info(self, neighborhood: str, language: str) -> HandlerResult:
        """Fallback neighborhood info when service unavailable"""
        info = {
            'balat': {
                'en': "Balat is a historic neighborhood in Istanbul known for its colorful houses, antique shops, and authentic local cafes. It's perfect for photographers and those seeking off-the-beaten-path experiences.",
                'tr': "Balat, İstanbul'un tarihi bir mahallesi olup renkli evleri, antika dükkanları ve otantik yerel kafeleriyle bilinir."
            },
            'kadikoy': {
                'en': "Kadıköy is a vibrant district on the Asian side of Istanbul, famous for its markets, street food, and lively nightlife. The Moda area offers stunning Bosphorus views.",
                'tr': "Kadıköy, İstanbul'un Anadolu yakasında canlı bir ilçe olup pazarları, sokak yemekleri ve hareketli gece hayatıyla ünlüdür."
            },
            'sultanahmet': {
                'en': "Sultanahmet is Istanbul's historic heart, home to Hagia Sophia, Blue Mosque, and Topkapi Palace. A must-visit for history lovers.",
                'tr': "Sultanahmet, İstanbul'un tarihi kalbidir. Ayasofya, Sultanahmet Camii ve Topkapı Sarayı'na ev sahipliği yapar."
            }
        }
        
        neighborhood_lower = neighborhood.lower()
        if neighborhood_lower in info:
            content = info[neighborhood_lower]
            response = content.get(language, content.get('en', ''))
        else:
            response = f"I'd love to tell you about {neighborhood}! It's one of Istanbul's unique neighborhoods."
        
        return HandlerResult(
            success=True,
            response=response,
            intent='neighborhood_guide',
            suggestions=[
                f"Hidden gems in {neighborhood}",
                f"Best restaurants in {neighborhood}",
                f"How to get to {neighborhood}"
            ]
        )


# Singleton instance
_router_instance = None


def get_intent_router(db_session=None) -> UnifiedIntentRouter:
    """Get or create the unified intent router singleton"""
    global _router_instance
    
    if _router_instance is None:
        _router_instance = UnifiedIntentRouter(db_session=db_session)
    
    return _router_instance
