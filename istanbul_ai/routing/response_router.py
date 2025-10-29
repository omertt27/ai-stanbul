"""
Response Router - Route queries to appropriate handlers

This module handles intelligent routing of user queries to the most appropriate
handler (ML-enhanced or standard) based on intent, context, and availability.

Week 2 Refactoring: Extracted from main_system.py
"""

import logging
from typing import Dict, List, Optional, Any, Union
from ..core.models import UserProfile, ConversationContext

logger = logging.getLogger(__name__)


class ResponseRouter:
    """Route queries to appropriate handlers with intelligent fallback"""
    
    def __init__(self):
        """Initialize the response router"""
        self.ml_handler_priority = [
            'ml_restaurant_handler',
            'ml_attraction_handler',
            'ml_event_handler',
            'ml_weather_handler',
            'ml_hidden_gems_handler',
            'ml_route_planning_handler',
            'ml_neighborhood_handler'
        ]
    
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
            context: Conversation context
            handlers: Dictionary of available handlers
            neural_insights: Optional neural insights
            return_structured: Whether to return structured response
        
        Returns:
            Response string or structured response dict
        """
        logger.info(f"🎯 Routing query with intent: {intent}")
        
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
        
        elif intent in ['route_planning', 'gps_route_planning', 'museum_route_planning']:
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
        """Route restaurant queries"""
        # Try ML handler first
        ml_handler = handlers.get('ml_restaurant_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_query(
                    message=message,
                    entities=entities,
                    user_profile=user_profile,
                    context=context
                )
                if response and response.get('response'):
                    logger.info("✅ ML Restaurant Handler processed query")
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
        """Route attraction queries (museums, landmarks, etc.)"""
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
                    logger.info("✅ ML Attraction Handler processed query")
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
        """Route transportation queries"""
        transportation_handler = handlers.get('transportation_response_handler')
        if transportation_handler:
            return transportation_handler(
                message, entities, user_profile, context, 
                neural_insights, return_structured
            )
        
        return "I can help you navigate Istanbul's transportation system! What route are you planning?"
    
    def _route_neighborhood_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route neighborhood queries"""
        # Try ML handler first
        ml_handler = handlers.get('ml_neighborhood_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_neighborhood_query(
                    message, entities, user_profile, context
                )
                if return_structured:
                    return {
                        'response': response,
                        'intent': 'neighborhood',
                        'source': 'ml_neighborhood_handler'
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
    
    def _route_events_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route events queries"""
        # Try ML handler first
        ml_handler = handlers.get('ml_event_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_event_query(
                    message, entities, user_profile, context
                )
                if return_structured:
                    return {
                        'response': response,
                        'intent': 'events',
                        'source': 'ml_event_handler'
                    }
                return response
            except Exception as e:
                logger.warning(f"ML Event Handler failed: {e}")
        
        # Fallback to events response handler
        events_handler = handlers.get('events_response_handler')
        if events_handler:
            from datetime import datetime
            return events_handler(entities, user_profile, context, datetime.now(), neural_insights)
        
        return "Istanbul has amazing events happening! What type of event interests you?"
    
    def _route_weather_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route weather queries"""
        # Try ML handler first
        ml_handler = handlers.get('ml_weather_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_weather_query(
                    message, entities, user_profile, context
                )
                if return_structured:
                    return {
                        'response': response,
                        'intent': 'weather',
                        'source': 'ml_weather_handler'
                    }
                return response
            except Exception as e:
                logger.warning(f"ML Weather Handler failed: {e}")
        
        # Fallback to response generator
        response_generator = handlers.get('response_generator')
        if response_generator:
            return response_generator.generate_comprehensive_recommendation(
                'weather', entities, user_profile, context,
                return_structured=return_structured
            )
        
        return "I can provide weather information for Istanbul! What would you like to know?"
    
    def _route_airport_transport_query(
        self, entities: Dict, user_profile: UserProfile, context: ConversationContext,
        handlers: Dict, return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route airport transport queries"""
        response_generator = handlers.get('response_generator')
        if response_generator:
            return response_generator.generate_comprehensive_recommendation(
                'airport_transport', entities, user_profile, context,
                return_structured=return_structured
            )
        
        return "I can help you get to/from Istanbul airports! Which airport are you using?"
    
    def _route_hidden_gems_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict, neural_insights: Optional[Dict],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """Route hidden gems queries"""
        # Try ML handler first
        ml_handler = handlers.get('ml_hidden_gems_handler')
        if ml_handler:
            try:
                response = ml_handler.handle_hidden_gems_query(
                    message, entities, user_profile, context
                )
                if return_structured:
                    query_params = (ml_handler.extract_query_parameters(message) 
                                   if hasattr(ml_handler, 'extract_query_parameters') else {})
                    return {
                        'response': response,
                        'intent': 'hidden_gems',
                        'source': 'ml_hidden_gems_handler',
                        'query_params': query_params
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
        """Route greeting queries"""
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
        
        return "🌟 Merhaba! Welcome to Istanbul! I'm here to help you discover this amazing city. What would you like to explore?"
    
    def _route_general_query(
        self, message: str, entities: Dict, user_profile: UserProfile,
        context: ConversationContext, handlers: Dict
    ) -> str:
        """Route general/fallback queries"""
        response_generator = handlers.get('response_generator')
        if response_generator:
            return response_generator._generate_fallback_response(context, user_profile)
        
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
