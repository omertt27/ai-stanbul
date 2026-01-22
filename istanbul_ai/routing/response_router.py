"""
Response Router - Route queries to appropriate handlers

This module handles intelligent routing of user queries to the most appropriate
handler (ML-enhanced or standard) based on intent, context, and availability.

Week 2 Refactoring: Extracted from main_system.py
Phase 2 Enhancement: Added neural response ranking for semantic result ordering
"""

import logging
from typing import Dict, List, Optional, Any, Union
from ..core.models import UserProfile, ConversationContext

logger = logging.getLogger(__name__)

# Import RestaurantQueryHandler for location extraction and map display
try:
    from backend.services.restaurant_query_handler import get_restaurant_query_handler, RestaurantQueryHandler
    RESTAURANT_QUERY_HANDLER_AVAILABLE = True
except ImportError:
    RESTAURANT_QUERY_HANDLER_AVAILABLE = False
    logger.warning("RestaurantQueryHandler not available - restaurant map centering disabled")

class ResponseRouter:
    """Route queries to appropriate handlers with intelligent fallback"""
    
    def __init__(self, neural_ranker=None, llm_service=None):
        """
        Initialize the response router
        
        Args:
            neural_ranker: Optional NeuralResponseRanker for semantic ranking
            llm_service: Optional UnifiedLLMService for LLM calls
        """
        self.ml_handler_priority = [
            'emergency_safety_handler',      # PRIORITY 1: Emergency & Safety
            'local_food_handler',             # PRIORITY 2: Local Food
            'ml_restaurant_handler',
            'ml_attraction_handler',
            'ml_event_handler',
            'ml_weather_handler',
            'ml_hidden_gems_handler',
            'ml_route_planning_handler',
            'ml_neighborhood_handler'
        ]
        
        # Neural ranker for semantic result ordering
        self.neural_ranker = neural_ranker
        self.use_neural_ranking = neural_ranker is not None and neural_ranker.available
        
        # UnifiedLLMService for all LLM calls (circuit breaker, caching, metadata)
        self.llm_service = llm_service
        if self.llm_service:
            logger.info("✅ Response Router initialized with UnifiedLLMService")
        
        if self.use_neural_ranking:
            logger.info("✅ Response Router initialized with neural ranking")
        else:
            logger.info("📝 Response Router initialized (keyword-only ranking)")
    
    def route_query(
        self,
        message: str,
        intent: str,
        entities: Dict[str, Any],
        user_profile: UserProfile,
        context: ConversationContext,
        handlers: Dict[str, Any],
        neural_insights: Optional[Dict] = None,
        return_structured: bool = False,
        intent_result: Optional[Any] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Route query to the most appropriate handler
        
        Args:
            message: User's input message
            intent: Classified intent
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context (should include language)
            handlers: Dictionary of available handlers
            neural_insights: Optional neural insights
            return_structured: Whether to return structured response
            intent_result: Optional IntentResult object with classification metadata
        
        Returns:
            Response string or structured response dict
        """
        logger.info(f"🎯 Routing query with intent: {intent}")
        
        # 🤖 LLM FALLBACK: If LLM was used to understand the intent, use LLM to generate response
        if intent_result and hasattr(intent_result, 'method') and intent_result.method == 'llm_fallback':
            logger.info(f"🤖 Intent was classified by LLM - using LLM to generate response for '{intent}'")
            return self._generate_llm_response(
                message=message,
                intent=intent,
                entities=entities,
                user_profile=user_profile,
                context=context,
                handlers=handlers,
                return_structured=return_structured
            )
        
        # 🌐 BILINGUAL: Ensure language is in context for all handlers
        language = self._ensure_language_context(context, user_profile)
        if language:
            logger.debug(f"🌐 Routing with language: {language}")
        
        # PRIORITY CHECK 1: Emergency & Safety queries (ALWAYS check first)
        emergency_handler = handlers.get('emergency_safety_handler')
        if emergency_handler and hasattr(emergency_handler, 'can_handle'):
            try:
                if emergency_handler.can_handle(message, entities):
                    logger.info("🚨 Routing to Emergency & Safety Handler (priority override)")
                    return self._route_emergency_safety_query(
                        message, entities, user_profile, context, handlers,
                        neural_insights, return_structured
                    )
            except Exception as e:
                logger.warning(f"Emergency handler can_handle check failed: {e}")
        
        # PRIORITY CHECK 2: Local Food queries (check before regular restaurants)
        local_food_handler = handlers.get('local_food_handler')
        if local_food_handler and hasattr(local_food_handler, 'can_handle'):
            try:
                if local_food_handler.can_handle(message, entities):
                    logger.info("🥙 Routing to Local Food Handler (priority override)")
                    return self._route_local_food_query(
                        message, entities, user_profile, context, handlers,
                        neural_insights, return_structured
                    )
            except Exception as e:
                logger.warning(f"Local food handler can_handle check failed: {e}")
        
        # Route based on intent
        if intent == 'restaurant':
            return self._route_restaurant_query(
                message, entities, user_profile, context, handlers, 
                neural_insights, return_structured
            )
        
        elif intent == 'attraction':
            return self._route_attraction_query(
                message, entities, user_profile, context, handlers,
                neural_insights, return_structured
            )
        
        elif intent == 'transportation':
            return self._route_transportation_query(
                message, entities, user_profile, context, handlers,
                neural_insights, return_structured
            )
        
        elif intent == 'neighborhood':
            return self._route_neighborhood_query(
                message, entities, user_profile, context, handlers,
                return_structured
            )
        
        elif intent == 'shopping':
            return self._route_shopping_query(
                entities, user_profile, context, handlers, neural_insights
            )
        
        elif intent == 'photography':
            return self._route_photography_query(
                message, entities, user_profile, context, handlers, neural_insights
            )
        
        elif intent == 'art_exhibitions':
            return self._route_art_exhibitions_query(
                message, entities, user_profile, context, handlers, neural_insights
            )
        
        elif intent == 'sports':
            return self._route_sports_query(
                message, entities, user_profile, context, handlers, neural_insights
            )
        
        elif intent == 'timing':
            return self._route_timing_query(
                message, entities, user_profile, context, handlers, neural_insights
            )
        
        elif intent == 'safety':
            return self._route_safety_query(
                message, entities, user_profile, context, handlers, neural_insights
            )
        
        elif intent == 'events':
            return self._route_events_query(
                message, entities, user_profile, context, handlers,
                neural_insights, return_structured
            )
        
        elif intent == 'weather':
            return self._route_weather_query(
                message, entities, user_profile, context, handlers,
                neural_insights, return_structured
            )
        
        elif intent == 'airport_transport':
            return self._route_airport_transport_query(
                entities, user_profile, context, handlers, return_structured
            )
        
        elif intent == 'hidden_gems':
            return self._route_hidden_gems_query(
                message, entities, user_profile, context, handlers,
                neural_insights, return_structured
            )
        
        elif intent == 'nearby_locations':
            return self._route_nearby_locations_query(
                message, entities, user_profile, context, handlers,
                neural_insights, return_structured
            )
        
        elif intent == 'gps_route_planning' or intent == 'gps_navigation':
            # Both gps_route_planning and gps_navigation should route to transportation
            if intent == 'gps_navigation':
                # gps_navigation is essentially a transportation query
                return self._route_transportation_query(
                    message, entities, user_profile, context, handlers,
                    neural_insights, return_structured
                )
            return self._route_planning_query(
                message, intent, entities, user_profile, context, handlers,
                neural_insights
            )
        
        elif intent == 'greeting':
            return self._route_greeting_query(
                message, user_profile, context, handlers
            )
        
        else:
            return self._route_general_query(
                message, entities, user_profile, context, handlers
            )
    
    def _route_restaurant_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route restaurant queries with language context and map location extraction"""
        logger.info(f"🍽️ _route_restaurant_query called | message: '{message}' | entities: {entities}")
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        logger.info(f"🌐 Language detected: {language}")
        
        # 🗺️ Extract location info for map display using RestaurantQueryHandler
        map_config = None
        if RESTAURANT_QUERY_HANDLER_AVAILABLE:
            try:
                handler = get_restaurant_query_handler()
                map_config = handler.get_map_display_config(message)
                if map_config.get('show_map'):
                    logger.info(f"🗺️ Restaurant query map config: center={map_config.get('center')}, "
                              f"locations={map_config.get('locations')}, cuisine={map_config.get('cuisine_filter')}")
            except Exception as e:
                logger.warning(f"Failed to extract restaurant map config: {e}")
        
        # Try ML handler first (if it exists)
        ml_handler = handlers.get('ml_restaurant_handler')
        if ml_handler:
            try:
                # Check if handler has handle_query or generate_response method
                if hasattr(ml_handler, 'handle_query'):
                    response = ml_handler.handle_query(
                        message=message,
                        entities=entities,
                        user_profile=user_profile,
                        context=context  # Context now includes language
                    )
                elif hasattr(ml_handler, 'generate_response'):
                    response = ml_handler.generate_response(
                        message=message,
                        neural_insights=neural_insights or {},
                        user_profile=user_profile,
                        context=context
                    )
                else:
                    logger.warning("ML Restaurant Handler has no handle_query or generate_response method")
                    response = None
                
                if response and (isinstance(response, dict) and response.get('response') or isinstance(response, str)):
                    logger.info(f"✅ ML Restaurant Handler processed query (lang: {language})")
                    if isinstance(response, dict):
                        # 🗺️ Enhance map_data with extracted location info if available
                        if map_config and map_config.get('show_map'):
                            if 'map_data' not in response or not response['map_data']:
                                # Create map_data from extracted location config
                                response['map_data'] = {
                                    'center': map_config.get('center'),
                                    'zoom': map_config.get('zoom', 14),
                                    'search_radius_km': map_config.get('search_radius_km', 2.0),
                                    'markers': map_config.get('markers', []),
                                    'cuisine_filter': map_config.get('cuisine_filter'),
                                    'locations': map_config.get('locations', []),
                                    'query_type': 'restaurant'
                                }
                                logger.info(f"🗺️ Added location-based map_data to response")
                            else:
                                # Enhance existing map_data with center if not set
                                existing_map = response['map_data']
                                if not existing_map.get('center') and map_config.get('center'):
                                    existing_map['center'] = map_config.get('center')
                                    existing_map['zoom'] = map_config.get('zoom', 14)
                                    logger.info(f"🗺️ Enhanced map_data center from location extraction")
                        return response if return_structured else response['response']
                    return response
            except Exception as e:
                logger.warning(f"ML Restaurant Handler failed: {e}")
        else:
            logger.warning("⚠️ ml_restaurant_handler not found in handlers, using response_generator fallback")
        
        # Try UnifiedLLMService for restaurant query
        if self.llm_service:
            logger.info(f"🤖 Using UnifiedLLMService for restaurant query (lang: {language})")
            
            # Build context from entities
            context_parts = []
            if entities.get('location'):
                context_parts.append(f"Location: {entities['location']}")
            if entities.get('cuisine'):
                context_parts.append(f"Cuisine: {entities['cuisine']}")
            if entities.get('price_range'):
                context_parts.append(f"Price range: {entities['price_range']}")
            
            context_str = "\n".join(context_parts) if context_parts else None
            
            # Build restaurant-specific prompt
            prompt = f"""You are an Istanbul restaurant expert assistant. A user is asking: "{message}"

{f'Context: {context_str}' if context_str else ''}

Provide a helpful, specific restaurant recommendation for Istanbul. Include:
- Restaurant names with brief descriptions
- Location/neighborhood
- Approximate prices
- What makes them special

Respond in {language} language. Be friendly and local."""

            try:
                # Run async method in sync context
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create new event loop in thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            llm_response = executor.submit(
                                asyncio.run,
                                self.llm_service.generate_with_prompt(
                                    prompt=prompt,
                                    temperature=0.7,
                                    max_tokens=500
                                )
                            ).result(timeout=35)
                    else:
                        llm_response = loop.run_until_complete(
                            self.llm_service.generate_with_prompt(
                                prompt=prompt,
                                temperature=0.7,
                                max_tokens=500
                            )
                        )
                except RuntimeError:
                    # No event loop
                    llm_response = asyncio.run(
                        self.llm_service.generate_with_prompt(
                            prompt=prompt,
                            temperature=0.7,
                            max_tokens=500
                        )
                    )
                
                if llm_response and len(llm_response.strip()) > 0:
                    logger.info(f"✅ UnifiedLLMService generated restaurant response ({len(llm_response)} chars)")
                    if return_structured:
                        result = {
                            'response': llm_response,
                            'intent': 'restaurant',
                            'language': language
                        }
                        if map_config and map_config.get('show_map'):
                            result['map_data'] = {
                                'center': map_config.get('center'),
                                'zoom': map_config.get('zoom', 14),
                                'markers': map_config.get('markers', []),
                                'locations': map_config.get('locations', []),
                                'query_type': 'restaurant'
                            }
                        return result
                    return llm_response
            except Exception as e:
                logger.warning(f"⚠️ UnifiedLLMService restaurant generation failed: {e}")
        else:
            logger.warning("⚠️ UnifiedLLMService not available, using fallback")
        
        # Fallback to response generator (ensures language context is passed)
        response_generator = handlers.get('response_generator')
        if response_generator:
            logger.info(f"📝 Using response_generator for restaurant query (lang: {language})")
            result = response_generator.generate_comprehensive_recommendation(
                'restaurant', entities, user_profile, context, 
                return_structured=return_structured
            )
            # 🗺️ Add map config to fallback response if available
            if return_structured and isinstance(result, dict) and map_config and map_config.get('show_map'):
                if 'map_data' not in result or not result['map_data']:
                    result['map_data'] = {
                        'center': map_config.get('center'),
                        'zoom': map_config.get('zoom', 14),
                        'search_radius_km': map_config.get('search_radius_km', 2.0),
                        'markers': map_config.get('markers', []),
                        'cuisine_filter': map_config.get('cuisine_filter'),
                        'locations': map_config.get('locations', []),
                        'query_type': 'restaurant'
                    }
            return result
        
        # Final fallback with bilingual support and map data
        fallback_text = ("İstanbul'daki harika restoranları bulmanıza yardımcı olabilirim! "
                        "Aradığınız şey hakkında daha fazla bilgi verebilir misiniz?") if language == 'tr' else \
                       "I can help you find great restaurants in Istanbul! Please tell me more about what you're looking for."
        
        if return_structured:
            result = {
                'response': fallback_text,
                'intent': 'restaurant',
                'language': language
            }
            # Add map config if available
            if map_config and map_config.get('show_map'):
                result['map_data'] = {
                    'center': map_config.get('center'),
                    'zoom': map_config.get('zoom', 14),
                    'markers': map_config.get('markers', []),
                    'locations': map_config.get('locations', []),
                    'query_type': 'restaurant'
                }
            return result
        return fallback_text
    
    def _route_attraction_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route attraction queries (museums, landmarks, etc.) with language context"""
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        message_lower = message.lower()
        
        # Check if this is specifically a museum query
        museum_keywords = ['museum', 'museums', 'gallery', 'exhibition', 'art', 'historical sites', 'cultural sites']
        is_museum_query = any(keyword in message_lower for keyword in museum_keywords)
        
        # Check if this is an attractions query
        attraction_keywords = ['attraction', 'attractions', 'place', 'places', 'landmark', 'sight', 'visit', 'see', 'tower', 'palace', 'mosque', 'bazaar']
        is_attraction_query = any(keyword in message_lower for keyword in attraction_keywords)
        
        # Route to appropriate specialized handler
        if is_museum_query and handlers.get('advanced_museum_system'):
            # Use advanced museum system
            museum_handler = handlers.get('museum_response_handler')
            if museum_handler:
                try:
                    response = museum_handler(message, entities, user_profile, context)
                    if isinstance(response, str) and handlers.get('weather_context_enhancer'):
                        response = handlers['weather_context_enhancer'](response)
                    return response
                except Exception as e:
                    logger.warning(f"Museum handler failed: {e}")
        
        if is_attraction_query and handlers.get('advanced_attractions_system'):
            # Use advanced attractions system
            attractions_handler = handlers.get('attractions_response_handler')
            if attractions_handler:
                try:
                    response = attractions_handler(message, entities, user_profile, context)
                    if isinstance(response, str) and handlers.get('weather_context_enhancer'):
                        response = handlers['weather_context_enhancer'](response)
                    return response
                except Exception as e:
                    logger.warning(f"Attractions handler failed: {e}")
        
        # Try ML attraction handler
        ml_handler = handlers.get('ml_attraction_handler')
        if ml_handler:
            try:
                # Check if handler has handle_query or generate_response method
                if hasattr(ml_handler, 'handle_query'):
                    response = ml_handler.handle_query(
                        message=message,
                        entities=entities,
                        user_profile=user_profile,
                        context=context
                    )
                elif hasattr(ml_handler, 'generate_response'):
                    response = ml_handler.generate_response(
                        message=message,
                        neural_insights=neural_insights or {},
                        user_profile=user_profile,
                        context=context
                    )
                else:
                    logger.warning("ML Attraction Handler has no handle_query or generate_response method")
                    response = None
                
                if response and (isinstance(response, dict) and response.get('response') or isinstance(response, str)):
                    logger.info(f"✅ ML Attraction Handler processed query (lang: {language})")
                    if isinstance(response, dict):
                        return response if return_structured else response['response']
                    return response
            except Exception as e:
                logger.warning(f"ML Attraction Handler failed: {e}")
        else:
            logger.warning("⚠️ ml_attraction_handler not found in handlers, using response_generator fallback")
        
        # Fallback to response generator
        response_generator = handlers.get('response_generator')
        if response_generator:
            response = response_generator.generate_comprehensive_recommendation(
                'attraction', entities, user_profile, context, 
                return_structured=return_structured
            )
            # Add weather context if available
            if isinstance(response, str) and handlers.get('weather_context_enhancer'):
                response = handlers['weather_context_enhancer'](response)
            return response
        
        return "I can help you discover amazing attractions in Istanbul! What interests you most?"
    
    def _route_transportation_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route transportation queries with language context"""
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        logger.info(f"🚇 Routing transportation query (lang: {language})")
        
        # Try new transportation handler first
        transportation_handler = handlers.get('transportation_handler')
        if transportation_handler and hasattr(transportation_handler, 'handle'):
            return transportation_handler.handle(
                message=message,
                entities=entities,
                user_profile=user_profile,
                context=context,  # Context now includes language
                neural_insights=neural_insights,
                return_structured=return_structured
            )
        
        # Fallback to legacy handler (if exists)
        legacy_handler = handlers.get('transportation_response_handler')
        if legacy_handler:
            return legacy_handler(
                message, entities, user_profile, context, 
                neural_insights, return_structured
            )
        
        return "I can help you navigate Istanbul's transportation system! What route are you planning?"
    
    def _route_neighborhood_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route neighborhood queries with language context"""
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        # Try ML handler first
        ml_handler = handlers.get('ml_neighborhood_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_neighborhood_query(
                    message, entities, user_profile, context  # Context now includes language
                )
                if return_structured:
                    return {
                        'response': response,
                        'intent': 'neighborhood',
                        'source': 'ml_neighborhood_handler',
                        'language': language
                    }
                return response
            except Exception as e:
                logger.warning(f"ML Neighborhood Handler failed: {e}")
        
        # Fallback to response generator
        response_generator = handlers.get('response_generator')
        if response_generator:
            return response_generator.generate_comprehensive_recommendation(
                'neighborhood', entities, user_profile, context,
                return_structured=return_structured
            )
        
        return "I can help you explore Istanbul's diverse neighborhoods! Which area interests you?"
    
    def _route_shopping_query(
        self, entities: Dict, user_profile: UserProfile, context: ConversationContext,
        handlers: Dict, neural_insights: Optional[Dict]
    ) -> str:
        """Route shopping queries"""
        shopping_handler = handlers.get('shopping_response_handler')
        if shopping_handler:
            return shopping_handler(entities, user_profile, context, neural_insights)
        
        return "I can guide you through Istanbul's amazing shopping scene! What are you looking for?"
    
    def _route_photography_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict]
    ) -> str:
        """Route photography-related queries"""
        photography_handler = handlers.get('photography_response_handler')
        if photography_handler:
            return photography_handler(message, entities, user_profile, context, neural_insights)
        
        # Fallback response
        response = "Istanbul offers amazing photography opportunities! Galata Tower, Blue Mosque, and Bosphorus provide stunning shots. Golden hour: sunrise at Pierre Loti Hill or sunset at Suleymaniye. What type of photography interests you?"
        context.add_interaction(message, response, 'photography')
        return response
    
    def _route_art_exhibitions_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict]
    ) -> str:
        """Route art exhibitions queries"""
        art_handler = handlers.get('art_exhibitions_response_handler')
        if art_handler:
            return art_handler(message, entities, user_profile, context, neural_insights)
        
        # Fallback response
        response = "Istanbul has a vibrant art scene! Major venues: Istanbul Modern (contemporary), Pera Museum (Orientalist art), SALT Beyoglu (cultural programs), and Arter (cutting-edge contemporary). Check their websites for current exhibitions!"
        context.add_interaction(message, response, 'art_exhibitions')
        return response
    
    def _route_sports_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict]
    ) -> str:
        """Route sports-related queries"""
        sports_handler = handlers.get('sports_response_handler')
        if sports_handler:
            return sports_handler(message, entities, user_profile, context, neural_insights)
        
        # Fallback response
        message_lower = message.lower()
        if 'football' in message_lower or 'soccer' in message_lower:
            response = "Istanbul's football scene is passionate! Major teams: Galatasaray, Fenerbahce, and Besiktas. Attending a derby match is an unforgettable experience. Tickets available online. Would you like stadium details?"
        else:
            response = "Istanbul offers diverse sports: watch football matches (Galatasaray, Fenerbahce, Besiktas), run along Bosphorus paths, cycle on Princes' Islands, or join the November Istanbul Marathon that crosses two continents!"
        
        context.add_interaction(message, response, 'sports')
        return response
    
    def _route_timing_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict]
    ) -> str:
        """Route timing-related queries"""
        timing_handler = handlers.get('timing_response_handler')
        if timing_handler:
            return timing_handler(message, entities, user_profile, context, neural_insights)
        
        # Fallback response
        message_lower = message.lower()
        if 'best time' in message_lower or 'when to visit' in message_lower:
            response = "Best times to visit Istanbul: Spring (April-May) for perfect weather and tulip festival, or Fall (September-October) for comfortable temperatures and fewer crowds. Summer is hot and crowded, winter offers lowest prices but cold weather."
        else:
            response = "Typical hours: Museums 9 AM-5 PM (closed Mondays), Grand Bazaar 9 AM-7 PM (closed Sundays), Mosques open daily except prayer times. What specific timing info do you need?"
        
        context.add_interaction(message, response, 'timing')
        return response
    
    def _route_safety_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict]
    ) -> str:
        """Route safety-related queries"""
        safety_handler = handlers.get('safety_response_handler')
        if safety_handler:
            return safety_handler(message, entities, user_profile, context, neural_insights)
        
        # Fallback response
        response = "Istanbul is generally safe for tourists! Tips: Keep valuables secure in crowded areas, use official yellow taxis or Uber, avoid unlicensed guides. Emergency: 112. Tourist police: 0212 527 4503. What specific safety concerns do you have?"
        context.add_interaction(message, response, 'safety')
        return response
    
    def _route_emergency_safety_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Route emergency and safety queries (hospitals, police, embassies, etc.)
        
        This is a PRIORITY handler - routes critical safety/emergency information.
        """
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        # Get emergency & safety handler
        emergency_handler = handlers.get('emergency_safety_handler')
        if emergency_handler:
            try:
                logger.info(f"🚨 Routing to Emergency & Safety Handler (lang: {language})")
                response = emergency_handler.handle(
                    message=message,
                    entities=entities,
                    user_profile=user_profile,
                    context=context,
                    return_structured=return_structured
                )
                
                if response:
                    if isinstance(response, dict):
                        return response if return_structured else response.get('response', '')
                    return response
                    
            except Exception as e:
                logger.error(f"Emergency & Safety Handler failed: {e}", exc_info=True)
        
        # Fallback to standard safety handler if available
        safety_handler = handlers.get('safety_response_handler')
        if safety_handler:
            logger.info("⚠️ Falling back to standard safety handler")
            try:
                return safety_handler(message, entities, user_profile, context, neural_insights)
            except Exception as e:
                logger.warning(f"Standard safety handler failed: {e}")
        
        # Final fallback with bilingual support
        if language == 'tr':
            fallback = (
                "🚨 ACİL DURUM BİLGİLERİ:\n"
                "• Acil: 112 (ambulans, itfaiye, polis)\n"
                "• Polis: 155\n"
                "• İtfaiye: 110\n"
                "• Turist Polisi: 0212 527 4503\n\n"
                "Daha fazla yardıma ihtiyacınız var mı?"
            )
        else:
            fallback = (
                "🚨 EMERGENCY INFORMATION:\n"
                "• Emergency: 112 (ambulance, fire, police)\n"
                "• Police: 155\n"
                "• Fire: 110\n"
                "• Tourist Police: 0212 527 4503\n\n"
                "Do you need more specific help?"
            )
        
        context.add_interaction(message, fallback, 'emergency_safety')
        return fallback
    
    def _route_weather_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Route weather-related queries to the ML weather handler
        """
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        # Get weather handler from handlers dict
        weather_handler = handlers.get('ml_weather_handler')
        
        if weather_handler:
            try:
                # Call handler with full context
                result = weather_handler.handle_query(
                    message,
                    entities=entities,
                    user_profile=user_profile,
                    context=context,
                    neural_insights=neural_insights
                )
                
                if return_structured and isinstance(result, dict):
                    return result
                elif isinstance(result, dict):
                    response = result.get('response', str(result))
                else:
                    response = str(result)
                    
                context.add_interaction(message, response, 'weather')
                return response
                
            except Exception as e:
                logger.warning(f"Weather handler failed: {e}")
        
        # Fallback response
        fallback = "I can help you with Istanbul weather! Please try asking about current weather, forecasts, or weather-based recommendations."
        context.add_interaction(message, fallback, 'weather')
        return fallback
    
    def _route_events_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Route events-related queries to the ML events handler
        """
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        # Get events handler from handlers dict
        events_handler = handlers.get('ml_event_handler')
        
        if events_handler:
            try:
                # Call handler with full context
                result = events_handler.handle_query(
                    message,
                    entities=entities,
                    user_profile=user_profile,
                    context=context,
                    neural_insights=neural_insights
                )
                
                if return_structured and isinstance(result, dict):
                    return result
                elif isinstance(result, dict):
                    response = result.get('response', str(result))
                else:
                    response = str(result)
                    
                context.add_interaction(message, response, 'events')
                return response
                
            except Exception as e:
                logger.warning(f"Events handler failed: {e}")
        
        # Fallback response
        fallback = "I can help you discover events in Istanbul! Please ask about concerts, festivals, exhibitions, or cultural events."
        context.add_interaction(message, fallback, 'events')
        return fallback
    
    def _route_local_food_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Route local food queries (Turkish street food, local specialties)
        
        This is a PRIORITY handler - routes queries about local/street food
        before generic restaurant queries.
        """
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        # Get local food handler
        local_food_handler = handlers.get('local_food_handler')
        if local_food_handler:
            try:
                logger.info(f"🥙 Routing to Local Food Handler (lang: {language})")
                response = local_food_handler.handle(
                    message=message,
                    entities=entities,
                    user_profile=user_profile,
                    context=context,
                    neural_insights=neural_insights,
                    return_structured=return_structured
                )
                
                if response:
                    if isinstance(response, dict):
                        return response if return_structured else response.get('response', '')
                    return response
                    
            except Exception as e:
                logger.error(f"Local Food Handler failed: {e}", exc_info=True)
        
        # Fallback to restaurant handler if available
        ml_restaurant_handler = handlers.get('ml_restaurant_handler')
        if ml_restaurant_handler:
            logger.info("🍽️ Falling back to restaurant handler for local food query")
            try:
                if hasattr(ml_restaurant_handler, 'handle_query'):
                    response = ml_restaurant_handler.handle_query(
                        message=message,
                        entities=entities,
                        user_profile=user_profile,
                        context=context
                    )
                    if response:
                        return response if return_structured else response.get('response', '')
            except Exception as e:
                logger.warning(f"Restaurant handler fallback failed: {e}")
        
        # Final fallback to response generator
        response_generator = handlers.get('response_generator')
        if response_generator:
            logger.info(f"📝 Using response_generator for local food query (lang: {language})")
            return response_generator.generate_comprehensive_recommendation(
                'restaurant', entities, user_profile, context, 
                return_structured=return_structured
            )
        
        # Ultimate fallback with bilingual support
        if language == 'tr':
            fallback = (
                "🥙 İstanbul'un harika sokak lezzetleri var! "
                "Balık ekmek, kumpir, midye dolma, simit gibi yerel lezzetler hakkında "
                "size yardımcı olabilirim. Ne tür bir yiyecek arıyorsunuz?"
            )
        else:
            fallback = (
                "🥙 Istanbul has amazing street food! "
                "I can help you find local specialties like balık ekmek (fish sandwich), "
                "kumpir (stuffed potato), midye dolma (stuffed mussels), simit, and more. "
                "What are you looking for?"
            )
        
        context.add_interaction(message, fallback, 'local_food')
        return fallback
    
    def _route_hidden_gems_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route hidden gems queries with language context
        
        Smart routing: If query mentions 'attractions', 'places to visit', 'best places',
        route to attraction handler instead of hidden gems handler.
        """
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        # Check if this is actually a general attractions query
        message_lower = message.lower()
        attraction_indicators = [
            'attraction', 'attractions', 'best place', 'best places', 'top place', 'top places',
            'places to visit', 'what to see', 'must see', 'must-see', 'sights', 'landmarks',
            'gezilecek yer', 'en iyi yer', 'görülecek yer', 'turistik yer'
        ]
        
        is_general_attractions = any(indicator in message_lower for indicator in attraction_indicators)
        
        # If it's about general attractions, route to attraction handler instead
        if is_general_attractions:
            logger.info(f"🏛️ Routing hidden_gems query to attraction handler (general attractions detected)")
            return self._route_attraction_query(
                message, entities, user_profile, context, handlers,
                neural_insights, return_structured
            )
        
        # Otherwise, proceed with hidden gems routing
        # Try ML handler first
        ml_handler = handlers.get('ml_hidden_gems_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_hidden_gems_query(
                    message, entities, user_profile, context  # Context now includes language
                )
                if return_structured:
                    query_params = (ml_handler.extract_query_parameters(message) 
                                   if hasattr(ml_handler, 'extract_query_parameters') else {})
                    return {
                        'response': response,
                        'intent': 'hidden_gems',
                        'source': 'ml_hidden_gems_handler',
                        'query_params': query_params,
                        'language': language
                    }
                return response
            except Exception as e:
                logger.warning(f"ML Hidden Gems Handler failed: {e}")
        
        # Fallback to hidden gems handler
        hidden_gems_handler = handlers.get('hidden_gems_handler')
        if hidden_gems_handler:
            try:
                query_params = hidden_gems_handler.extract_query_parameters(message)
                gems = hidden_gems_handler.get_hidden_gems(
                    location=query_params.get('location'),
                    gem_type=query_params.get('gem_type'),
                    budget=query_params.get('budget'),
                    limit=5
                )
                response = hidden_gems_handler.format_hidden_gem_response(
                    gems, query_params.get('location')
                )
                if return_structured:
                    return {
                        'response': response,
                        'gems': gems,
                        'query_params': query_params
                    }
                return response
            except Exception as e:
                logger.warning(f"Hidden Gems Handler failed: {e}")
        
        # Fallback to response generator
        response_generator = handlers.get('response_generator')
        if response_generator:
            return response_generator.generate_comprehensive_recommendation(
                'hidden_gems', entities, user_profile, context,
                return_structured=return_structured
            )
        
        return "I know some amazing hidden gems in Istanbul! What type of place are you looking for?"
    
    def _route_nearby_locations_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route nearby locations queries"""
        nearby_handler = handlers.get('nearby_locations_response_handler')
        if nearby_handler:
            return nearby_handler(
                message, entities, user_profile, context, 
                neural_insights, return_structured
            )
        
        # Fallback response
        return "I can help you find nearby locations in Istanbul! Please provide your current location or preferences."
    
    def _route_planning_query(
        self, message: str, intent: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict]
    ) -> str:
        """Route planning queries (route, itinerary, GPS, museum routes)"""
        # Try ML route planning handler first
        ml_handler = handlers.get('ml_route_planning_handler')
        if ml_handler:
            try:
                # Import asyncio to run async handler
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If event loop is running, we need to create a task
                    # For now, skip ML handler in async context
                    logger.debug("Skipping ML handler in running event loop")
                else:
                    response = loop.run_until_complete(
                        ml_handler.handle_route_query(message, user_profile, context)
                    )
                    return response
            except Exception as e:
                logger.warning(f"ML Route Planning Handler failed: {e}")
        
        # Route to specific planning handler based on intent
        if intent == 'gps_route_planning':
            # Use transportation handler for GPS/navigation queries
            transportation_handler = handlers.get('transportation_handler')
            if transportation_handler:
                try:
                    logger.info("📍 Routing GPS query to Transportation Handler")
                    response = transportation_handler.handle(
                        message, entities, user_profile, context, neural_insights
                    )
                    return response
                except Exception as e:
                    logger.warning(f"Transportation handler failed: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # Fallback to legacy GPS handler if available
            gps_handler = handlers.get('gps_route_response_handler')
            if gps_handler:
                return gps_handler(message, entities, user_profile, context, neural_insights)
        
        elif intent == 'museum_route_planning':
            museum_route_handler = handlers.get('museum_route_response_handler')
            if museum_route_handler:
                return museum_route_handler(message, entities, user_profile, context, neural_insights)
        
        # Fallback to general route planning
        route_handler = handlers.get('route_planning_response_handler')
        if route_handler:
            return route_handler(message, user_profile, context, neural_insights)
        
        return "I can help you plan your perfect Istanbul itinerary! How many days do you have?"
    
    def _route_greeting_query(
        self, message: str, user_profile: UserProfile, context: ConversationContext,
        handlers: Dict
    ) -> str:
        """Route greeting queries with language context"""
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        # Try conversation handler first
        conversation_handler = handlers.get('conversation_handler')
        if conversation_handler:
            try:
                response = conversation_handler.handle_conversation(message)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Conversation handler failed: {e}")
        
        # Fallback to greeting response handler
        greeting_handler = handlers.get('greeting_response_handler')
        if greeting_handler:
            return greeting_handler(user_profile, context)
        
        # Fallback with bilingual greeting
        if language == 'tr':
            return "🌟 Merhaba! İstanbul'a hoş geldiniz! Size bu muhteşem şehri keşfetmenizde yardımcı olmak için buradayım. Neyi keşfetmek istersiniz?"
        else:
            return "🌟 Merhaba! Welcome to Istanbul! I'm here to help you discover this amazing city. What would you like to explore?"
    
    def _route_general_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict
    ) -> str:
        """Route general/fallback queries with language context"""
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        logger.info(f"❓ General query routing - trying UnifiedLLMService first (lang: {language})")
        
        # 🤖 Try UnifiedLLMService for general queries
        if self.llm_service:
            logger.info(f"🤖 Using UnifiedLLMService for general query (lang: {language})")
            
            # Build context from entities
            context_parts = []
            if entities:
                for key, value in entities.items():
                    if value:
                        context_parts.append(f"{key}: {value}")
            
            context_str = "\n".join(context_parts) if context_parts else None
            
            # Build general query prompt
            prompt = f"""You are a helpful Istanbul tourism assistant. A user is asking: "{message}"

{f'Context: {context_str}' if context_str else ''}

Provide a helpful, informative answer about Istanbul. Be friendly and conversational.

Respond in {language} language."""

            try:
                # Run async method in sync context
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create new event loop in thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            llm_response = executor.submit(
                                asyncio.run,
                                self.llm_service.generate_with_prompt(
                                    prompt=prompt,
                                    temperature=0.8,
                                    max_tokens=400
                                )
                            ).result(timeout=35)
                    else:
                        llm_response = loop.run_until_complete(
                            self.llm_service.generate_with_prompt(
                                prompt=prompt,
                                temperature=0.8,
                                max_tokens=400
                            )
                        )
                except RuntimeError:
                    llm_response = asyncio.run(
                        self.llm_service.generate_with_prompt(
                            prompt=prompt,
                            temperature=0.8,
                            max_tokens=400
                        )
                    )
                
                if llm_response and len(llm_response.strip()) > 0:
                    logger.info(f"✅ UnifiedLLMService generated general response ({len(llm_response)} chars)")
                    context.add_interaction(message, llm_response, 'general_llm')
                    return llm_response
            except Exception as e:
                logger.warning(f"⚠️ UnifiedLLMService general query failed: {e}")
        else:
            logger.warning("⚠️ UnifiedLLMService not available")
        
        # Fallback to response generator
        response_generator = handlers.get('response_generator')
        if response_generator:
            logger.info("📝 Using response_generator for general query")
            return response_generator._generate_fallback_response(context, user_profile)
        
        # Final fallback with bilingual response
        if language == 'tr':
            return "İstanbul'u keşfetmenizde size yardımcı olmaktan mutluluk duyarım! Ne aradığınız hakkında daha fazla bilgi verebilir misiniz?"
        else:
            return "I'd be happy to help you explore Istanbul! Could you tell me more about what you're looking for?"
    
    def should_use_ml_handler(self, intent: str, handlers: Dict) -> bool:
        """
        Determine if ML handler should be used for this intent
        
        Args:
            intent: Classified intent
            handlers: Available handlers
        
        Returns:
            True if ML handler should be used
        """
        ml_handler_map = {
            'restaurant': 'ml_restaurant_handler',
            'attraction': 'ml_attraction_handler',
            'events': 'ml_event_handler',
            'weather': 'ml_weather_handler',
            'hidden_gems': 'ml_hidden_gems_handler',
            'route_planning': 'ml_route_planning_handler',
            'neighborhood': 'ml_neighborhood_handler'
        }
        
        handler_name = ml_handler_map.get(intent)
        return handler_name and handlers.get(handler_name) is not None
    
    def rank_results(
        self,
        query: str,
        results: List[Dict],
        user_context: Optional[Dict] = None,
        field: str = 'description'
    ) -> List[Dict]:
        """
        Rank results using neural semantic similarity or fallback
        
        Args:
            query: User's search query
            results: List of result dictionaries to rank
            user_context: Optional user context (preferences, history, etc.)
            field: Field to use for semantic matching
            
        Returns:
            Ranked list of results
        """
        if not results:
            return results
        
        # Use neural ranker if available
        if self.use_neural_ranking:
            try:
                ranking_result = self.neural_ranker.rank_results(
                    query=query,
                    results=results,
                    user_context=user_context,
                    field=field
                )
                
                if ranking_result.method == 'neural':
                    logger.info(
                        f"🧠 Neural ranking: {len(results)} results, "
                        f"avg similarity: {ranking_result.avg_semantic_score:.3f}"
                    )
                    return ranking_result.ranked_results
                    
            except Exception as e:
                logger.warning(f"Neural ranking failed, using fallback: {e}")
        
        # Fallback: rank by rating or default order
        return self._fallback_ranking(results)
    
    def _fallback_ranking(self, results: List[Dict]) -> List[Dict]:
        """
        Fallback ranking based on rating/popularity
        
        Args:
            results: List of results to rank
            
        Returns:
            Ranked results
        """
        # Sort by rating if available, otherwise maintain order
        try:
            return sorted(
                results,
                key=lambda x: x.get('rating', x.get('popularity', 0)),
                reverse=True
            )
        except:
            return results
    
    def rank_restaurants(
        self,
        query: str,
        restaurants: List[Dict],
        user_profile: Optional[UserProfile] = None
    ) -> List[Dict]:
        """
        Rank restaurant results with neural semantic similarity
        
        Args:
            query: User's restaurant query
            restaurants: List of restaurant dictionaries
            user_profile: Optional user profile for context
            
        Returns:
            Ranked restaurants
        """
        # Build user context from profile
        user_context = None
        if user_profile:
            user_context = {
                'preferences': {
                    'preferred_cuisines': user_profile.preferences.get('cuisines', []),
                    'price_range': user_profile.preferences.get('price_range', '$$'),
                    'dietary_restrictions': user_profile.preferences.get('dietary_restrictions', [])
                },
                'history': {
                    'liked_places': user_profile.history.get('liked_restaurants', [])
                }
            }
        
        return self.rank_results(
            query=query,
            results=restaurants,
            user_context=user_context,
            field='description'
        )
    
    def rank_attractions(
        self,
        query: str,
        attractions: List[Dict],
        user_profile: Optional[UserProfile] = None
    ) -> List[Dict]:
        """
        Rank attraction results with neural semantic similarity
        
        Args:
            query: User's attraction query
            attractions: List of attraction dictionaries
            user_profile: Optional user profile for context
            
        Returns:
            Ranked attractions
        """
        user_context = None
        if user_profile:
            user_context = {
                'preferences': {
                    'interests': user_profile.preferences.get('interests', []),
                    'activity_level': user_profile.preferences.get('activity_level', 'moderate')
                },
                'history': {
                    'liked_places': user_profile.history.get('liked_attractions', [])
                }
            }
        
        return self.rank_results(
            query=query,
            results=attractions,
            user_context=user_context,
            field='description'
        )
    
    def rank_events(
        self,
        query: str,
        events: List[Dict],
        user_profile: Optional[UserProfile] = None
    ) -> List[Dict]:
        """
        Rank event results with neural semantic similarity
        
        Args:
            query: User's event query
            events: List of event dictionaries
            user_profile: Optional user profile for context
            
        Returns:
            Ranked events
        """
        user_context = None
        if user_profile:
            user_context = {
                'preferences': {
                    'event_types': user_profile.preferences.get('event_types', []),
                    'interests': user_profile.preferences.get('interests', [])
                },
                'history': {
                    'liked_events': user_profile.history.get('liked_events', [])
                }
            }
        
        return self.rank_results(
            query=query,
            results=events,
            user_context=user_context,
            field='description'
        )
    
    def get_ranking_stats(self) -> Dict:
        """
        Get neural ranking statistics
        
        Returns:
            Statistics dictionary
        """
        if self.neural_ranker:
            return self.neural_ranker.get_stats()
        return {'error': 'Neural ranker not available'}
    
    def _ensure_language_context(
        self, 
        context: ConversationContext, 
        user_profile: UserProfile
    ) -> Optional[str]:
        """
        Ensure language is present in context for handlers
        
        Args:
            context: Conversation context
            user_profile: User profile
            
        Returns:
            Language code ('en' or 'tr') or None
        """
        # Check if language is already in context
        if hasattr(context, 'language'):
            # Handle both string and Language enum
            if hasattr(context.language, 'value'):
                return context.language.value  # Language enum
            return context.language  # String
        
        # Try to get from user profile
        if user_profile and hasattr(user_profile, 'language_preference'):
            lang = user_profile.language_preference
            # Store back in context for consistency
            context.language = lang
            return lang if isinstance(lang, str) else lang.value if hasattr(lang, 'value') else 'en'
        
        # Default to English
        context.language = 'en'
        return 'en'
    
    def _generate_llm_response(
        self,
        message: str,
        intent: str,
        entities: Dict[str, Any],
        user_profile: UserProfile,
        context: ConversationContext,
        handlers: Dict[str, Any],
        return_structured: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate response using UnifiedLLMService when intent was classified by LLM
        
        This method is called when the hybrid intent classifier had low confidence
        and used the LLM to understand the intent. In such cases, we also use the
        LLM to generate the response for better quality.
        
        Args:
            message: User's input message
            intent: Classified intent
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context
            handlers: Dictionary of available handlers
            return_structured: Whether to return structured response
            
        Returns:
            LLM-generated response (string or dict)
        """
        try:
            # Get language from context
            language = self._ensure_language_context(context, user_profile)
            
            # Use UnifiedLLMService if available
            if not self.llm_service:
                logger.warning("⚠️ UnifiedLLMService not available - falling back to standard handler")
                return self._fallback_to_standard_routing(
                    message, intent, entities, user_profile, context, handlers, 
                    None, return_structured
                )
            
            # Build context string from conversation history
            context_str = ""
            if hasattr(context, 'interactions') and context.interactions:
                recent_interactions = context.interactions[-3:]  # Last 3 interactions
                for interaction in recent_interactions:
                    if hasattr(interaction, 'user_message') and hasattr(interaction, 'ai_response'):
                        context_str += f"User: {interaction.user_message}\nAI: {interaction.ai_response}\n"
            
            # Build entity context
            entity_str = ""
            if entities:
                entity_parts = [f"{k}: {v}" for k, v in entities.items() if v]
                if entity_parts:
                    entity_str = "\nExtracted information: " + ", ".join(entity_parts)
            
            logger.info(f"🤖 Generating UnifiedLLM response for intent='{intent}', language='{language}'")
            
            # Build intent-specific prompt - avoid backslash in f-string expression
            context_section = f"Previous conversation:\n{context_str}" if context_str else ""
            prompt = f"""You are a helpful Istanbul tourism assistant. A user is asking: "{message}"

Intent: {intent}
{entity_str}

{context_section}

Provide a helpful, accurate answer about Istanbul. Be friendly and conversational.

Respond in {language} language."""

            # Run async method in sync context
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create new event loop in thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        llm_response = executor.submit(
                            asyncio.run,
                            self.llm_service.generate_with_prompt(
                                prompt=prompt,
                                temperature=0.7,
                                max_tokens=500
                            )
                        ).result(timeout=35)
                else:
                    llm_response = loop.run_until_complete(
                        self.llm_service.generate_with_prompt(
                            prompt=prompt,
                            temperature=0.7,
                            max_tokens=500
                        )
                    )
            except RuntimeError:
                llm_response = asyncio.run(
                    self.llm_service.generate_with_prompt(
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=500
                    )
                )
            
            if llm_response and len(llm_response) > 10:
                logger.info(f"✅ UnifiedLLMService generated response ({len(llm_response)} chars)")
                
                if return_structured:
                    return {
                        'response': llm_response,
                        'map_data': {},
                        'intent': intent,
                        'method': 'llm_fallback'
                    }
                return llm_response
            else:
                logger.warning("⚠️ UnifiedLLMService returned empty or very short response")
            
        except Exception as e:
            logger.error(f"❌ UnifiedLLMService response generation failed: {e}", exc_info=True)
        
        # Fallback to standard routing if LLM fails
        logger.info("⬇️ Falling back to standard handler routing")
        return self._fallback_to_standard_routing(
            message, intent, entities, user_profile, context, handlers, 
            None, return_structured
        )
    
    def _fallback_to_standard_routing(
        self,
        message: str,
        intent: str,
        entities: Dict[str, Any],
        user_profile: UserProfile,
        context: ConversationContext,
        handlers: Dict[str, Any],
        neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Fallback to standard intent-based routing when LLM is unavailable
        
        This method routes to the appropriate handler based on intent,
        without the LLM override.
        """
        # Route based on intent (same logic as route_query, but without LLM override)
        if intent == 'restaurant':
            return self._route_restaurant_query(
                message, entities, user_profile, context, handlers, 
                neural_insights, return_structured
            )
        elif intent == 'attraction':
            return self._route_attraction_query(
                message, entities, user_profile, context, handlers,
                neural_insights, return_structured
            )
        elif intent == 'transportation':
            return self._route_transportation_query(
                message, entities, user_profile, context, handlers,
                neural_insights, return_structured
            )
        elif intent == 'general':
            return self._route_general_query(
                message, entities, user_profile, context, handlers, neural_insights
            )
        else:
            # Default fallback
            response_generator = handlers.get('response_generator')
            if response_generator:
                return response_generator.generate_response(
                    intent, entities, user_profile, context
                )
            return "I'm here to help you explore Istanbul! How can I assist you?"
