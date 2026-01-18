"""
Local Food Handler for Istanbul AI
Handles: Turkish street food, local specialties, traditional snacks, food culture

🍴 SPECIALTIES:
- Balık Ekmek (fish sandwich)
- Kumpir (stuffed baked potato)
- Midye Dolma (stuffed mussels)
- Simit (sesame bread ring)
- Börek (savory pastry)
- Döner, Kebab, Lahmacun
- Turkish breakfast spots
- Street food locations

🌐 MULTILINGUAL: LLM automatically detects and responds in user's language
🗺️ GPS-AWARE: Recommends food spots near user's location
💎 HIDDEN GEMS: Includes local favorites, not just tourist spots

Created: November 5, 2025
"""

from typing import Dict, Optional, List, Any, Union
import logging
import sys
import os

# Initialize logger early
logger = logging.getLogger(__name__)

# Add backend/services to path for mixin import
backend_services_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'services')
if backend_services_path not in sys.path:
    sys.path.insert(0, backend_services_path)

# Import handler LLM mixin for UnifiedLLMService integration
try:
    from backend.services.handler_llm_mixin import HandlerLLMMixin
    HANDLER_MIXIN_AVAILABLE = True
    logger.info("✅ HandlerLLMMixin loaded successfully")
except ImportError:
    try:
        # Fallback: try direct import after path modification
        from handler_llm_mixin import HandlerLLMMixin
        HANDLER_MIXIN_AVAILABLE = True
        logger.info("✅ HandlerLLMMixin loaded successfully (fallback path)")
    except ImportError as e:
        HANDLER_MIXIN_AVAILABLE = False
        logger.warning(f"⚠️ HandlerLLMMixin not available: {e}")
        # Create a dummy mixin if not available
        class HandlerLLMMixin:
            def _init_handler_llm(self): pass
            def _llm_generate(self, prompt, component, **kwargs):
                if hasattr(self, 'llm_service') and self.llm_service:
                    return self.llm_service.generate(prompt=prompt, **kwargs)
                raise RuntimeError("No LLM service available")


class LocalFoodHandler(HandlerLLMMixin):
    """
    Local Food & Street Food Handler
    
    Provides recommendations for:
    - Turkish street food (balık ekmek, kumpir, midye, simit)
    - Traditional snacks and quick bites
    - Best locations for specific foods
    - Food culture explanations
    - Dietary considerations (vegetarian, halal, etc.)
    
    Integrated with UnifiedLLMService via HandlerLLMMixin for consistent
    LLM calls, metrics tracking, and caching.
    """
    
    def __init__(
        self,
        llm_service=None,
        gps_location_service=None,
        hidden_gems_context_service=None,
        rag_service=None,
        neural_processor=None  # Add neural_processor parameter (optional, for compatibility)
    ):
        """
        Initialize local food handler.
        
        Args:
            llm_service: LLM service for natural language responses
            gps_location_service: GPS service for location-based recommendations
            hidden_gems_context_service: Hidden gems service for local spots
            rag_service: RAG service for food knowledge retrieval
            neural_processor: Neural processor (optional, not used but accepted for compatibility)
        """
        self.llm_service = llm_service
        self.gps_location_service = gps_location_service
        self.hidden_gems_context_service = hidden_gems_context_service
        self.rag_service = rag_service
        self.neural_processor = neural_processor  # Store but don't require
        
        # Initialize UnifiedLLMService integration via mixin
        self._init_handler_llm()
        
        # Service availability flags
        self.has_llm = llm_service is not None or hasattr(self, 'unified_llm')
        self.has_gps = gps_location_service is not None
        self.has_hidden_gems = hidden_gems_context_service is not None
        self.has_rag = rag_service is not None and getattr(rag_service, 'available', False)
        
        logger.info(
            f"🍴 Local Food Handler initialized - "
            f"LLM: {self.has_llm}, "
            f"GPS: {self.has_gps}, "
            f"HiddenGems: {self.has_hidden_gems}, "
            f"RAG: {self.has_rag}"
        )
    
    def can_handle(self, message: str, entities: Dict[str, Any]) -> bool:
        """
        Determine if this handler should process the query.
        
        Args:
            message: User's query
            entities: Extracted entities
            
        Returns:
            True if this is a local food query
        """
        message_lower = message.lower()
        
        # Turkish street food keywords
        food_keywords = [
            # Turkish street foods
            'balık ekmek', 'balik ekmek', 'fish sandwich',
            'kumpir', 'stuffed potato', 'baked potato',
            'midye', 'midye dolma', 'stuffed mussels', 'mussel',
            'simit', 'sesame bread', 'bagel',
            'börek', 'borek', 'pastry', 'savory pastry',
            'lahmacun', 'turkish pizza',
            'döner', 'doner', 'kebab', 'kebap',
            'kokoreç', 'kokorec',
            'islak burger', 'wet burger',
            'çay', 'turkish tea', 'apple tea',
            'ayran', 'yogurt drink',
            'künefe', 'kunefe', 'dessert',
            'baklava',
            
            # General food queries
            'street food', 'local food', 'traditional food',
            'turkish food', 'turkish cuisine',
            'what to eat', 'where to eat', 'food recommendation',
            'cheap food', 'quick bite', 'snack',
            'breakfast', 'kahvaltı', 'kahvalti',
            
            # Turkish food queries
            'ne yenir', 'ne yemeli', 'nerede yenir',
            'sokak yemeği', 'yerel yemek', 'türk mutfağı'
        ]
        
        return any(keyword in message_lower for keyword in food_keywords)
    
    def handle(
        self,
        message: str,
        entities: Dict[str, Any],
        user_profile=None,
        context=None,
        neural_insights: Optional[Dict] = None,
        return_structured: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle local food queries.
        
        Args:
            message: User's query (in any language)
            entities: Extracted entities
            user_profile: User profile (may contain GPS location)
            context: Conversation context
            neural_insights: ML insights
            return_structured: Whether to return structured data
            
        Returns:
            Food recommendation response
        """
        try:
            logger.info(f"🍴 Local food query received")
            
            # Build GPS context
            gps_context = self._build_gps_context(user_profile)
            
            # Classify query type
            query_type = self._classify_food_query(message)
            
            # Route to appropriate handler
            if query_type == 'specific_food':
                return self._handle_specific_food_query(
                    message, gps_context, entities, return_structured
                )
            elif query_type == 'location_based':
                return self._handle_location_based_query(
                    message, gps_context, entities, return_structured
                )
            elif query_type == 'dietary':
                return self._handle_dietary_query(
                    message, gps_context, entities, return_structured
                )
            else:
                return self._handle_general_food_query(
                    message, gps_context, entities, return_structured
                )
                
        except Exception as e:
            logger.error(f"Local food handler error: {e}", exc_info=True)
            return self._get_fallback_response(return_structured)
    
    def _classify_food_query(self, message: str) -> str:
        """
        Classify the type of food query.
        
        Args:
            message: User's query
            
        Returns:
            Query type: 'specific_food', 'location_based', 'dietary', or 'general'
        """
        message_lower = message.lower()
        
        # Specific food mentioned
        specific_foods = [
            'balık ekmek', 'kumpir', 'midye', 'simit', 'börek',
            'döner', 'kebab', 'lahmacun', 'kokoreç', 'künefe', 'baklava'
        ]
        if any(food in message_lower for food in specific_foods):
            return 'specific_food'
        
        # Location-based query
        location_indicators = ['near me', 'nearby', 'around here', 'in', 'at']
        if any(indicator in message_lower for indicator in location_indicators):
            return 'location_based'
        
        # Dietary considerations
        dietary_keywords = [
            'vegetarian', 'vegan', 'halal', 'kosher',
            'gluten free', 'dairy free', 'allergy', 'allergic'
        ]
        if any(keyword in message_lower for keyword in dietary_keywords):
            return 'dietary'
        
        return 'general'
    
    def _handle_specific_food_query(
        self,
        message: str,
        gps_context: Dict[str, Any],
        entities: Dict[str, Any],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle queries about specific Turkish foods.
        
        Args:
            message: User's query
            gps_context: GPS location context
            entities: Extracted entities
            return_structured: Whether to return structured data
            
        Returns:
            Food-specific recommendation
        """
        if not self.has_llm:
            return self._get_fallback_response(return_structured)
        
        try:
            # Get RAG context for the specific food
            rag_context = ""
            if self.has_rag:
                try:
                    rag_context = self.rag_service.get_context_for_llm(
                        query=f"turkish street food {message}",
                        top_k=3,
                        max_length=600
                    )
                    if rag_context:
                        logger.info(f"✅ RAG context retrieved for food query")
                except Exception as e:
                    logger.warning(f"⚠️ RAG retrieval failed: {e}")
            
            # Get location info
            location_text = ""
            if gps_context.get('has_gps'):
                district = gps_context.get('district', 'your area')
                location_text = f"\nUser's Location: {district}"
            
            # Build LLM prompt
            prompt = f"""You are KAM, Istanbul's friendly food guide. Answer this query about Turkish street food.

User Question: {message}
{location_text}

{rag_context if rag_context else ''}

Provide a helpful response (3-4 sentences) that includes:
1. Brief explanation of the food (what it is, how it's made)
2. Why it's special or popular in Istanbul
3. Best places to try it{' near ' + gps_context.get('district') if gps_context.get('has_gps') else ''}

Use food emojis (🐟🥔🦪🥯🥖) and keep it conversational. Respond in the same language as the user's question.

Response:"""
            
            # Generate response using UnifiedLLMService via mixin
            logger.info("🤖 Generating food recommendation with UnifiedLLM...")
            response = self._llm_generate(
                prompt=prompt,
                component="food_handler.street_food",
                max_tokens=200,
                temperature=0.7
            )
            
            if return_structured:
                return {
                    'response': response,
                    'handler': 'local_food_handler',
                    'query_type': 'specific_food',
                    'gps_context': gps_context,
                    'success': True
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error generating food recommendation: {e}")
            return self._get_fallback_response(return_structured)
    
    def _handle_location_based_query(
        self,
        message: str,
        gps_context: Dict[str, Any],
        entities: Dict[str, Any],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle location-based food queries (e.g., "street food near me").
        
        Args:
            message: User's query
            gps_context: GPS location context
            entities: Extracted entities
            return_structured: Whether to return structured data
            
        Returns:
            Location-based food recommendations
        """
        if not self.has_llm:
            return self._get_fallback_response(return_structured)
        
        try:
            # Get location
            if gps_context.get('has_gps'):
                location = gps_context.get('district', 'your area')
            else:
                # Try to extract location from message or entities
                location = self._extract_location_from_query(message, entities)
                if not location:
                    return self._prompt_for_location(return_structured)
            
            # Get hidden gems context for the area
            hidden_gems_text = ""
            if self.has_hidden_gems and gps_context.get('district'):
                try:
                    gems_data = self.hidden_gems_context_service.get_gems_for_route(
                        destination_district=gps_context['district'],
                        max_gems_per_district=3
                    )
                    if gems_data.get('destination_text'):
                        hidden_gems_text = gems_data['destination_text']
                        logger.info(f"💎 Found hidden gems for {location}")
                except Exception as e:
                    logger.warning(f"Could not fetch hidden gems: {e}")
            
            # Get RAG context
            rag_context = ""
            if self.has_rag:
                try:
                    rag_context = self.rag_service.get_context_for_llm(
                        query=f"street food restaurants {location} Istanbul",
                        top_k=3,
                        max_length=600
                    )
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
            
            # Build prompt
            prompt = f"""You are KAM, Istanbul's friendly food guide. Help find street food near the user.

User Question: {message}
Location: {location}

{rag_context if rag_context else ''}
{hidden_gems_text if hidden_gems_text else ''}

Provide 2-3 specific recommendations in {location} (3-4 sentences total):
1. Mention specific street food spots or areas
2. What foods they're famous for
3. Approximate prices or how to get there

Use food emojis (🐟🥔🦪🥯) and keep it practical. Respond in the same language as the user's question.

Response:"""
            
            # Generate response using UnifiedLLMService via mixin
            logger.info(f"🤖 Generating location-based food recommendations for {location}...")
            response = self._llm_generate(
                prompt=prompt,
                component="food_handler.restaurant_question",
                max_tokens=200,
                temperature=0.7
            )
            
            if return_structured:
                return {
                    'response': response,
                    'handler': 'local_food_handler',
                    'query_type': 'location_based',
                    'location': location,
                    'gps_context': gps_context,
                    'success': True
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error generating location-based recommendations: {e}")
            return self._get_fallback_response(return_structured)
    
    def _handle_dietary_query(
        self,
        message: str,
        gps_context: Dict[str, Any],
        entities: Dict[str, Any],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle dietary requirement queries (vegetarian, vegan, halal, etc.).
        
        Args:
            message: User's query
            gps_context: GPS location context
            entities: Extracted entities
            return_structured: Whether to return structured data
            
        Returns:
            Dietary-appropriate recommendations
        """
        if not self.has_llm:
            return self._get_fallback_response(return_structured)
        
        try:
            # Get RAG context
            rag_context = ""
            if self.has_rag:
                try:
                    rag_context = self.rag_service.get_context_for_llm(
                        query=f"turkish street food {message}",
                        top_k=3,
                        max_length=600
                    )
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
            
            # Build prompt
            location_text = ""
            if gps_context.get('has_gps'):
                location_text = f"near {gps_context.get('district')}"
            
            prompt = f"""You are KAM, Istanbul's friendly food guide. Help with dietary requirements.

User Question: {message}
{f"Location: {location_text}" if location_text else ""}

{rag_context if rag_context else ''}

Provide helpful advice (3-4 sentences):
1. Which Turkish street foods match their dietary needs
2. What to look for or ask for
3. Specific recommendations {location_text if location_text else ''}

Be specific about dietary considerations (halal, vegetarian, etc.) and use food emojis. Respond in the same language as the user's question.

Response:"""
            
            # Generate response using UnifiedLLMService via mixin
            logger.info("🤖 Generating dietary-specific recommendations...")
            response = self._llm_generate(
                prompt=prompt,
                component="food_handler.cuisine_question",
                max_tokens=200,
                temperature=0.7
            )
            
            if return_structured:
                return {
                    'response': response,
                    'handler': 'local_food_handler',
                    'query_type': 'dietary',
                    'gps_context': gps_context,
                    'success': True
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error generating dietary recommendations: {e}")
            return self._get_fallback_response(return_structured)
    
    def _handle_general_food_query(
        self,
        message: str,
        gps_context: Dict[str, Any],
        entities: Dict[str, Any],
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle general food culture queries.
        
        Args:
            message: User's query
            gps_context: GPS location context
            entities: Extracted entities
            return_structured: Whether to return structured data
            
        Returns:
            General food recommendations
        """
        if not self.has_llm:
            return self._get_fallback_response(return_structured)
        
        try:
            # Get RAG context
            rag_context = ""
            if self.has_rag:
                try:
                    rag_context = self.rag_service.get_context_for_llm(
                        query=f"istanbul street food culture {message}",
                        top_k=4,
                        max_length=700
                    )
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
            
            # Build prompt
            prompt = f"""You are KAM, Istanbul's friendly food guide. Answer this general question about Turkish street food.

User Question: {message}

{rag_context if rag_context else ''}

Provide a helpful overview (3-4 sentences):
1. Introduction to Turkish street food culture
2. Must-try items for first-time visitors
3. General tips (prices, where to find, when to eat)

Use food emojis (🐟🥔🦪🥯🥖🍽️) and be welcoming. Respond in the same language as the user's question.

Response:"""
            
            # Generate response using UnifiedLLMService via mixin
            logger.info("🤖 Generating general food recommendations...")
            response = self._llm_generate(
                prompt=prompt,
                component="food_handler.price_budget_question",
                max_tokens=200,
                temperature=0.7
            )
            
            if return_structured:
                return {
                    'response': response,
                    'handler': 'local_food_handler',
                    'query_type': 'general',
                    'gps_context': gps_context,
                    'success': True
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error generating general recommendations: {e}")
            return self._get_fallback_response(return_structured)
    
    # ===== HELPER METHODS =====
    
    def _build_gps_context(self, user_profile) -> Dict[str, Any]:
        """
        Build GPS location context from user profile.
        
        Args:
            user_profile: User profile with GPS location data
            
        Returns:
            GPS context dict
        """
        gps_context = {
            'has_gps': False,
            'location': None,
            'district': None,
            'coordinates': None
        }
        
        if not user_profile or not hasattr(user_profile, 'current_location'):
            return gps_context
        
        gps_location = user_profile.current_location
        if not gps_location or not isinstance(gps_location, tuple) or len(gps_location) != 2:
            return gps_context
        
        gps_context['has_gps'] = True
        gps_context['coordinates'] = {
            'lat': gps_location[0],
            'lng': gps_location[1]
        }
        
        # Try to get district name
        if self.gps_location_service:
            try:
                district = self.gps_location_service.get_district_from_coords(
                    gps_location[0],
                    gps_location[1]
                )
                gps_context['district'] = district
            except Exception as e:
                logger.warning(f"Could not get district from GPS: {e}")
        
        return gps_context
    
    def _extract_location_from_query(
        self,
        message: str,
        entities: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract location from query or entities.
        
        Args:
            message: User's query
            entities: Extracted entities
            
        Returns:
            Location string or None
        """
        # Try entities first
        if 'location' in entities and entities['location']:
            locations = entities['location']
            if locations:
                return locations[0]
        
        # Try to extract from message
        message_lower = message.lower()
        common_locations = [
            'taksim', 'beyoglu', 'beyoğlu', 'karakoy', 'karaköy',
            'kadikoy', 'kadıköy', 'besiktas', 'beşiktaş',
            'sultanahmet', 'eminonu', 'eminönü', 'ortakoy', 'ortaköy',
            'galata', 'istiklal', 'uskudar', 'üsküdar'
        ]
        
        for location in common_locations:
            if location in message_lower:
                return location.title()
        
        return None
    
    def _prompt_for_location(self, return_structured: bool) -> Union[str, Dict[str, Any]]:
        """
        Prompt user to provide location or enable GPS.
        
        Args:
            return_structured: Whether to return structured data
            
        Returns:
            Location prompt response
        """
        response = (
            "📍 To give you the best street food recommendations, please tell me which area of Istanbul "
            "you're in (e.g., 'Taksim', 'Kadıköy', 'Sultanahmet'), or enable GPS by clicking the 📍 button."
        )
        
        if return_structured:
            return {
                'response': response,
                'handler': 'local_food_handler',
                'needs_location': True,
                'success': False
            }
        else:
            return response
    
    def _get_fallback_response(self, return_structured: bool) -> Union[str, Dict[str, Any]]:
        """
        Generate fallback response when services unavailable.
        
        Args:
            return_structured: Whether to return structured data
            
        Returns:
            Fallback response
        """
        response = """🍴 Istanbul Street Food Favorites:

🐟 **Balık Ekmek** (Fish Sandwich) - Try at Eminönü fish boats
🥔 **Kumpir** - Best in Ortaköy, loaded baked potato
🦪 **Midye Dolma** (Stuffed Mussels) - Found at street carts
🥯 **Simit** - Sesame bread rings, available everywhere
🥖 **Döner & Kebab** - Authentic spots in Beyoğlu

For specific recommendations, tell me which area you're interested in!"""
        
        if return_structured:
            return {
                'response': response,
                'handler': 'local_food_handler',
                'method': 'fallback',
                'success': True
            }
        else:
            return response


# ===== HELPER FUNCTION =====

def get_local_food_handler(
    llm_service=None,
    gps_location_service=None,
    hidden_gems_context_service=None,
    rag_service=None
) -> LocalFoodHandler:
    """
    Factory function to create LocalFoodHandler instance.
    
    Args:
        llm_service: LLM service
        gps_location_service: GPS location service
        hidden_gems_context_service: Hidden gems service
        rag_service: RAG service
        
    Returns:
        LocalFoodHandler instance
    """
    return LocalFoodHandler(
        llm_service=llm_service,
        gps_location_service=gps_location_service,
        hidden_gems_context_service=hidden_gems_context_service,
        rag_service=rag_service
    )


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example usage
    handler = LocalFoodHandler()
    
    # Test queries
    test_queries = [
        "Where can I find balık ekmek?",
        "What is kumpir?",
        "Best street food near Taksim",
        "Vegetarian Turkish street food options"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        if handler.can_handle(query, {}):
            print("✅ Handler can process this query")
        else:
            print("❌ Handler cannot process this query")
