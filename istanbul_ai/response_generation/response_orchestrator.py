"""
Response Orchestrator - Coordinate response generation across all modules

This module orchestrates the entire response generation pipeline, coordinating
all response generation modules to create cohesive, intelligent responses.

Week 7-8 Refactoring: Extracted from main_system.py
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from .language_handler import LanguageHandler
from .context_builder import ContextBuilder
from .response_formatter import ResponseFormatter
from .bilingual_responder import BilingualResponder

logger = logging.getLogger(__name__)


class ResponseOrchestrator:
    """
    Orchestrates response generation across all modules
    
    Coordinates:
    - Language detection and handling
    - Context building
    - Response formatting
    - Multi-intent handling
    - Fallback responses
    - Response composition
    """
    
    def __init__(
        self,
        language_handler: Optional[LanguageHandler] = None,
        context_builder: Optional[ContextBuilder] = None,
        response_formatter: Optional[ResponseFormatter] = None,
        bilingual_responder: Optional[BilingualResponder] = None
    ):
        """
        Initialize response orchestrator
        
        Args:
            language_handler: Language handler instance (creates new if None)
            context_builder: Context builder instance (creates new if None)
            response_formatter: Response formatter instance (creates new if None)
            bilingual_responder: Bilingual responder instance (creates new if None)
        """
        self.language_handler = language_handler or LanguageHandler()
        self.context_builder = context_builder or ContextBuilder()
        self.response_formatter = response_formatter or ResponseFormatter()
        self.bilingual_responder = bilingual_responder or BilingualResponder()
        
        logger.info("✅ ResponseOrchestrator initialized")
    
    def generate_response(
        self,
        query: str,
        intent: str,
        entities: Optional[Dict[str, Any]] = None,
        results: Optional[List[Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete response for a query
        
        Args:
            query: User's query text
            intent: Detected intent
            entities: Extracted entities
            results: Results from handlers (restaurants, attractions, etc.)
            user_profile: User profile data
            session_context: Session context data
            conversation_history: Conversation history
        
        Returns:
            Response dictionary with text, language, metadata
        """
        language = 'en'  # Default language
        try:
            # 1. Detect language
            language = self.language_handler.detect_language(
                message=query,
                user_profile=user_profile
            )
            
            # 2. Check for special cases (greetings, thanks, goodbye)
            special_response = self._handle_special_cases(query, language, session_context)
            if special_response:
                return special_response
            
            # 3. Build context
            context = self.context_builder.build_response_context(
                user_profile=user_profile,
                conversation_context=session_context,
                message=query,
                entities=entities
            )
            
            # 4. Check if we have results
            if not results:
                return self._handle_no_results(intent, language, query, context)
            
            # 5. Format response based on intent and results
            response_text = self._format_intent_response(
                intent=intent,
                results=results,
                language=language,
                context=context
            )
            
            # 6. Add recommendations if appropriate
            response_text = self._add_recommendations(
                response_text=response_text,
                intent=intent,
                context=context,
                language=language
            )
            
            # 7. Apply user preferences (if any)
            response_text = self._apply_user_preferences(
                response_text=response_text,
                user_profile=user_profile,
                language=language
            )
            
            return {
                'text': response_text,
                'language': language,
                'intent': intent,
                'result_count': len(results),
                'has_recommendations': '💡' in response_text,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return self._handle_error(language, str(e))
    
    def handle_multi_intent(
        self,
        intents: List[Tuple[str, float]],
        query: str,
        results_by_intent: Dict[str, List[Any]],
        user_profile: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle multiple intents in a single query
        
        Args:
            intents: List of (intent, confidence) tuples
            query: User's query text
            results_by_intent: Results grouped by intent
            user_profile: User profile data
            session_context: Session context data
        
        Returns:
            Combined response dictionary
        """
        language = 'en'  # Default language
        try:
            # Detect language
            language = self.language_handler.detect_language(
                message=query,
                user_profile=user_profile
            )
            
            # Sort intents by confidence
            intents = sorted(intents, key=lambda x: x[1], reverse=True)
            
            # Take top 2 intents (don't overwhelm user)
            top_intents = intents[:2]
            
            # Build responses for each intent
            intent_responses = []
            total_results = 0
            
            for intent, confidence in top_intents:
                results = results_by_intent.get(intent, [])
                if results:
                    response_text = self._format_intent_response(
                        intent=intent,
                        results=results[:5],  # Limit to 5 per intent
                        language=language,
                        context={}
                    )
                    intent_responses.append(response_text)
                    total_results += len(results)
            
            # Compose final response
            if not intent_responses:
                return self._handle_no_results('general', language, query, {})
            
            # Create header
            header = self._get_multi_intent_header(len(top_intents), language)
            
            # Combine responses
            combined_text = header + "\n\n" + "\n\n---\n\n".join(intent_responses)
            
            return {
                'text': combined_text,
                'language': language,
                'intents': [i[0] for i in top_intents],
                'result_count': total_results,
                'is_multi_intent': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling multi-intent: {e}", exc_info=True)
            return self._handle_error(language, str(e))
    
    def compose_response(
        self,
        parts: List[str],
        language: str = 'en',
        add_separator: bool = True
    ) -> str:
        """
        Compose response from multiple parts
        
        Args:
            parts: List of response parts to combine
            language: Language code
            add_separator: Whether to add separators between parts
        
        Returns:
            Composed response string
        """
        if not parts:
            return self.bilingual_responder.get_fallback_response('general', language)
        
        # Filter out empty parts
        parts = [p for p in parts if p and p.strip()]
        
        if not parts:
            return self.bilingual_responder.get_fallback_response('general', language)
        
        # Combine parts
        if add_separator:
            return "\n\n---\n\n".join(parts)
        else:
            return "\n\n".join(parts)
    
    def _handle_special_cases(
        self,
        query: str,
        language: str,
        session_context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Handle special cases like greetings, thanks, goodbye"""
        
        # Normalize query for detection (lowercase)
        query_normalized = query.lower()
        
        # Check for greeting
        if self.language_handler.is_greeting(query_normalized):
            time_of_day = self._get_time_of_day()
            template_key = f'greeting_{time_of_day}'
            response_text = self.bilingual_responder.get_special_case_response(template_key, language)
            
            return {
                'text': response_text,
                'language': language,
                'intent': 'greeting',
                'result_count': 0,
                'is_special_case': True,
                'timestamp': datetime.now().isoformat()
            }
        
        # Check for thanks
        if self.language_handler.is_thanks(query_normalized):
            response_text = self.bilingual_responder.get_special_case_response('thanks', language)
            return {
                'text': response_text,
                'language': language,
                'intent': 'thanks',
                'result_count': 0,
                'is_special_case': True,
                'timestamp': datetime.now().isoformat()
            }
        
        # Check for goodbye
        if self.language_handler.is_goodbye(query_normalized):
            response_text = self.bilingual_responder.get_special_case_response('goodbye', language)
            return {
                'text': response_text,
                'language': language,
                'intent': 'goodbye',
                'result_count': 0,
                'is_special_case': True,
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _handle_no_results(
        self,
        intent: str,
        language: str,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle case when no results are found"""
        
        # Generate suggestions based on intent
        suggestions = self._generate_suggestions(intent, language, context)
        
        response_text = self.bilingual_responder.get_no_results_response(
            query_type=intent,
            language=language,
            suggestions=suggestions
        )
        
        return {
            'text': response_text,
            'language': language,
            'intent': intent,
            'result_count': 0,
            'has_suggestions': len(suggestions) > 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_error(
        self,
        language: str,
        error_details: str
    ) -> Dict[str, Any]:
        """Handle errors during response generation"""
        
        response_text = self.bilingual_responder.get_emergency_response(
            error_type='system_error',
            language=language,
            error_details=error_details
        )
        
        return {
            'text': response_text,
            'language': language,
            'intent': 'error',
            'result_count': 0,
            'is_error': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_intent_response(
        self,
        intent: str,
        results: List[Any],
        language: str,
        context: Dict[str, Any]
    ) -> str:
        """Format response based on intent type"""
        
        # Get appropriate header
        header = self._get_intent_header(intent, len(results), language)
        
        # Format based on intent
        if intent in ['restaurant', 'attraction', 'hotel', 'shopping']:
            return self._format_place_response(results, header, language)
        
        elif intent == 'transportation':
            return self._format_transportation_response(results, header, language)
        
        elif intent == 'weather':
            return self._format_weather_response(results, header, language)
        
        else:
            # Generic list formatting
            return self.response_formatter.format_list_response(
                items=results,
                header=header,
                language=language
            )
    
    def _format_place_response(
        self,
        results: List[Any],
        header: str,
        language: str
    ) -> str:
        """Format response for places (restaurants, attractions, hotels)"""
        
        response = f"{header}\n\n"
        
        for i, place in enumerate(results[:10], 1):
            # Extract place details
            name = getattr(place, 'name', str(place))
            rating = getattr(place, 'rating', None)
            distance = getattr(place, 'distance', None)
            price = getattr(place, 'price', None)
            
            # Format item
            response += f"{i}. **{name}**"
            
            # Add rating if available
            if rating:
                stars = self.response_formatter.format_rating(rating)
                response += f" {stars}"
            
            # Add distance if available
            if distance:
                dist_str = self.response_formatter.format_distance(distance)
                response += f" • {dist_str}"
            
            # Add price if available
            if price:
                price_str = self.response_formatter.format_price(price, language)
                response += f" • {price_str}"
            
            response += "\n"
        
        # Add truncation notice if needed
        if len(results) > 10:
            total = len(results)
            truncation_msg = (
                f"\n💡 Toplam {total} sonuçtan ilk 10'u gösteriliyor."
                if language == 'tr' else
                f"\n💡 Showing first 10 of {total} results."
            )
            response += truncation_msg
        
        return response
    
    def _format_transportation_response(
        self,
        results: List[Any],
        header: str,
        language: str
    ) -> str:
        """Format response for transportation routes"""
        
        response = f"{header}\n\n"
        
        for i, route in enumerate(results[:3], 1):  # Show top 3 routes
            # Extract route details
            duration = getattr(route, 'duration', 'N/A')
            distance = getattr(route, 'distance', None)
            mode = getattr(route, 'mode', 'Unknown')
            
            response += f"{i}. "
            
            # Add route mode emoji
            mode_emoji = self._get_transport_emoji(mode)
            response += f"{mode_emoji} **{mode}**"
            
            # Add duration
            response += f" • ⏱️ {duration}"
            
            # Add distance if available
            if distance:
                dist_str = self.response_formatter.format_distance(distance)
                response += f" • 📍 {dist_str}"
            
            response += "\n"
        
        return response
    
    def _format_weather_response(
        self,
        results: List[Any],
        header: str,
        language: str
    ) -> str:
        """Format response for weather information"""
        
        if not results:
            return self.bilingual_responder.get_no_results_response('weather', language)
        
        weather = results[0]  # Get first weather result
        
        # Extract weather details
        temp = getattr(weather, 'temperature', 'N/A')
        condition = getattr(weather, 'condition', 'N/A')
        humidity = getattr(weather, 'humidity', None)
        
        response = f"{header}\n\n"
        response += f"🌡️ **{temp}°C** • {condition}"
        
        if humidity:
            response += f" • 💧 {humidity}%"
        
        return response
    
    def _add_recommendations(
        self,
        response_text: str,
        intent: str,
        context: Dict[str, Any],
        language: str
    ) -> str:
        """Add helpful recommendations based on context"""
        
        # Get recommendations based on time of day
        time_of_day = context.get('temporal_context', {}).get('time_of_day', 'afternoon')
        
        recommendations = self._get_time_based_recommendations(
            intent=intent,
            time_of_day=time_of_day,
            language=language
        )
        
        if recommendations:
            tip_header = "\n\n💡 **İpucu:**" if language == 'tr' else "\n\n💡 **Tip:**"
            response_text += f"{tip_header} {recommendations}"
        
        return response_text
    
    def _apply_user_preferences(
        self,
        response_text: str,
        user_profile: Optional[Dict[str, Any]],
        language: str
    ) -> str:
        """Apply user-specific preferences to response"""
        
        if not user_profile:
            return response_text
        
        # Check for dietary restrictions
        dietary = user_profile.get('dietary_restrictions', [])
        if dietary and 'restaurant' in response_text.lower():
            restriction_text = ", ".join(dietary)
            note = (
                f"\n\n📋 Not: {restriction_text} tercihleriniz dikkate alınmıştır."
                if language == 'tr' else
                f"\n\n📋 Note: Your {restriction_text} preferences have been considered."
            )
            response_text += note
        
        return response_text
    
    def _get_intent_header(self, intent: str, count: int, language: str) -> str:
        """Get header for intent response"""
        
        headers = {
            'restaurant': {
                'tr': f"🍽️ **{count} Restoran Bulundu**",
                'en': f"🍽️ **Found {count} Restaurants**"
            },
            'attraction': {
                'tr': f"🏛️ **{count} Yer Bulundu**",
                'en': f"🏛️ **Found {count} Attractions**"
            },
            'transportation': {
                'tr': f"🚇 **{count} Rota Bulundu**",
                'en': f"🚇 **Found {count} Routes**"
            },
            'hotel': {
                'tr': f"🏨 **{count} Konaklama Bulundu**",
                'en': f"🏨 **Found {count} Accommodations**"
            },
            'weather': {
                'tr': "🌤️ **Hava Durumu**",
                'en': "🌤️ **Weather Forecast**"
            }
        }
        
        lang_key = 'tr' if language == 'tr' else 'en'
        return headers.get(intent, {}).get(lang_key, f"**{count} Results**")
    
    def _get_multi_intent_header(self, intent_count: int, language: str) -> str:
        """Get header for multi-intent response"""
        
        if language == 'tr':
            return f"🎯 **Sizin için {intent_count} farklı kategoride sonuçlar buldum:**"
        else:
            return f"🎯 **I found results in {intent_count} different categories for you:**"
    
    def _get_time_of_day(self) -> str:
        """Get current time of day category"""
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'evening'  # Default to evening for late night
    
    def _get_transport_emoji(self, mode: str) -> str:
        """Get emoji for transport mode"""
        emojis = {
            'metro': '🚇',
            'bus': '🚌',
            'tram': '🚊',
            'ferry': '⛴️',
            'taxi': '🚕',
            'walk': '🚶',
            'car': '🚗'
        }
        return emojis.get(mode.lower(), '🚇')
    
    def _get_time_based_recommendations(
        self,
        intent: str,
        time_of_day: str,
        language: str
    ) -> str:
        """Get recommendations based on time of day"""
        
        recommendations = {
            'restaurant': {
                'morning': {
                    'tr': "Kahvaltı için Türk kahvaltısı yapan yerlere göz atın!",
                    'en': "Try places serving traditional Turkish breakfast!"
                },
                'afternoon': {
                    'tr': "Öğle yemeği için vakit! Popüler yerler dolu olabilir.",
                    'en': "Lunch time! Popular places might be busy."
                },
                'evening': {
                    'tr': "Akşam yemeği için rezervasyon yaptırmayı unutmayın!",
                    'en': "Don't forget to make reservations for dinner!"
                }
            },
            'attraction': {
                'morning': {
                    'tr': "Sabah saatleri müzeler için idealdir, daha az kalabalık!",
                    'en': "Morning is ideal for museums, less crowded!"
                },
                'afternoon': {
                    'tr': "Gün ışığında fotoğraf çekmek için harika bir zaman!",
                    'en': "Perfect time for photos in daylight!"
                },
                'evening': {
                    'tr': "Akşam aydınlatmaları muhteşem! Boğaz manzarası tavsiye edilir.",
                    'en': "Evening lights are stunning! Bosphorus views recommended."
                }
            }
        }
        
        lang_key = 'tr' if language == 'tr' else 'en'
        return recommendations.get(intent, {}).get(time_of_day, {}).get(lang_key, '')
    
    def _generate_suggestions(
        self,
        intent: str,
        language: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate alternative suggestions for no results"""
        
        suggestions = {
            'restaurant': {
                'tr': [
                    "Farklı bir mutfak türü deneyin",
                    "Bütçe aralığınızı genişletin",
                    "Başka bir bölgeye bakın"
                ],
                'en': [
                    "Try a different cuisine type",
                    "Expand your budget range",
                    "Look in a different area"
                ]
            },
            'attraction': {
                'tr': [
                    "Daha geniş bir kategori seçin",
                    "Popüler turistik yerlere bakın",
                    "Gizli cennetleri keşfedin"
                ],
                'en': [
                    "Choose a broader category",
                    "Check popular tourist spots",
                    "Discover hidden gems"
                ]
            }
        }
        
        lang_key = 'tr' if language == 'tr' else 'en'
        return suggestions.get(intent, {}).get(lang_key, [])
