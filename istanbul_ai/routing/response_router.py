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


class ResponseRouter:
    """Route queries to appropriate handlers with intelligent fallback"""
    
    def __init__(self, neural_ranker=None):
        """
        Initialize the response router
        
        Args:
            neural_ranker: Optional NeuralResponseRanker for semantic ranking
        """
        self.ml_handler_priority = [
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
        return_structured: bool = False
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
        
        Returns:
            Response string or structured response dict
        """
        logger.info(f"🎯 Routing query with intent: {intent}")
        
        # 🌐 BILINGUAL: Ensure language is in context for all handlers
        language = self._ensure_language_context(context, user_profile)
        if language:
            logger.debug(f"🌐 Routing with language: {language}")
        
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
        
        elif intent == 'gps_route_planning':
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
        """Route restaurant queries with language context"""
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        # Try ML handler first
        ml_handler = handlers.get('ml_restaurant_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_query(
                    message=message,
                    entities=entities,
                    user_profile=user_profile,
                    context=context  # Context now includes language
                )
                if response and response.get('response'):
                    logger.info(f"✅ ML Restaurant Handler processed query (lang: {language})")
                    return response if return_structured else response['response']
            except Exception as e:
                logger.warning(f"ML Restaurant Handler failed: {e}")
        
        # Fallback to response generator
        response_generator = handlers.get('response_generator')
        if response_generator:
            return response_generator.generate_comprehensive_recommendation(
                'restaurant', entities, user_profile, context, 
                return_structured=return_structured
            )
        
        return "I can help you find great restaurants in Istanbul! Please tell me more about what you're looking for."
    
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
                response = ml_handler.handle_query(
                    message=message,
                    entities=entities,
                    user_profile=user_profile,
                    context=context
                )
                if response and response.get('response'):
                    logger.info(f"✅ ML Attraction Handler processed query (lang: {language})")
                    return response if return_structured else response['response']
            except Exception as e:
                logger.warning(f"ML Attraction Handler failed: {e}")
        
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
    
    def _route_events_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route event-related queries with language context"""
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        # Try ML handler first
        ml_handler = handlers.get('ml_event_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_query(
                    message=message,
                    entities=entities,
                    user_profile=user_profile,
                    context=context  # Context now includes language
                )
                if response and response.get('response'):
                    logger.info(f"✅ ML Event Handler processed query (lang: {language})")
                    return response if return_structured else response['response']
            except Exception as e:
                logger.warning(f"ML Event Handler failed: {e}")
        
        # Fallback to response generator
        response_generator = handlers.get('response_generator')
        if response_generator:
            return response_generator.generate_comprehensive_recommendation(
                'event', entities, user_profile, context, 
                return_structured=return_structured
            )
        
        return "I can help you find interesting events in Istanbul! What type of events are you interested in?"
    
    def _route_weather_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route weather-related queries with language context"""
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
        weather_handler = handlers.get('weather_response_handler')
        if weather_handler:
            return weather_handler(
                message, entities, user_profile, context,  # Context now includes language
                neural_insights, return_structured
            )
        
        # Fallback response
        return "I can provide weather information for Istanbul! Please specify a date or time period."
    
    def _route_airport_transport_query(
        self, entities: Dict, user_profile: UserProfile, context: ConversationContext,
        handlers: Dict, return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route airport transportation queries"""
        # Try ML handler first
        ml_handler = handlers.get('ml_airport_transport_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_airport_transport_query(
                    entities, user_profile, context
                )
                if return_structured:
                    return {
                        'response': response,
                        'intent': 'airport_transport',
                        'source': 'ml_airport_transport_handler'
                    }
                return response
            except Exception as e:
                logger.warning(f"ML Airport Transport Handler failed: {e}")
        
        # Fallback to response generator
        response_generator = handlers.get('response_generator')
        if response_generator:
            return response_generator.generate_comprehensive_recommendation(
                'airport_transport', entities, user_profile, context,
                return_structured=return_structured
            )
        
        return "I can help you with airport transportation in Istanbul! Do you need information on shuttles, taxis, or public transport?"
    
    def _route_hidden_gems_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route hidden gems queries with language context"""
        # 🌐 BILINGUAL: Ensure language is in context
        language = self._ensure_language_context(context, user_profile)
        
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
                response = ml_handler.handle_route_query(
                    message, entities, user_profile, context
                )
                return response
            except Exception as e:
                logger.warning(f"ML Route Planning Handler failed: {e}")
        
        # Route to specific planning handler based on intent
        if intent == 'gps_route_planning':
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
        
        response_generator = handlers.get('response_generator')
        if response_generator:
            return response_generator._generate_fallback_response(context, user_profile)
        
        # Fallback with bilingual response
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
