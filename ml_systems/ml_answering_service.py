"""
ML-Powered Answering Service for Istanbul AI Guide
Integrates intent classification, semantic search, and LLM generation
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Intent categories recognized by the system"""
    RESTAURANT_RECOMMENDATION = "restaurant_recommendation"
    ATTRACTION_QUERY = "attraction_query"
    NEIGHBORHOOD_INFO = "neighborhood_info"
    TRANSPORTATION_HELP = "transportation_help"
    DAILY_TALK = "daily_talk"
    LOCAL_TIPS = "local_tips"
    WEATHER_INFO = "weather_info"
    EVENTS_QUERY = "events_query"
    ROUTE_PLANNING = "route_planning"


@dataclass
class QueryContext:
    """Structured context for ML answering"""
    query: str
    intent: Intent
    user_location: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None
    search_results: Optional[List[Dict]] = None
    structured_data: Optional[Dict] = None
    metadata: Optional[Dict] = None


class MLAnsweringService:
    """
    Unified ML service for answering user queries
    
    Pipeline:
    1. Intent Classification (DistilBERT)
    2. Context Retrieval (Semantic Search + Structured Queries)
    3. Response Generation (Local LLM)
    """
    
    def __init__(self, 
                 intent_classifier=None,
                 semantic_search=None,
                 llm_generator=None,
                 data_sources: Optional[Dict] = None):
        """
        Initialize ML answering service
        
        Args:
            intent_classifier: Intent classification model
            semantic_search: Semantic search engine
            llm_generator: LLM response generator
            data_sources: Dictionary of data sources (restaurants, attractions, etc.)
        """
        self.intent_classifier = intent_classifier
        self.semantic_search = semantic_search
        self.llm_generator = llm_generator
        self.data_sources = data_sources or {}
        
        logger.info("🤖 ML Answering Service initialized")
        if intent_classifier:
            logger.info("  ✅ Intent Classifier loaded")
        if semantic_search:
            logger.info("  ✅ Semantic Search loaded")
        if llm_generator:
            logger.info("  ✅ LLM Generator loaded")
    
    async def answer_query(self, 
                          query: str, 
                          user_location: Optional[Dict[str, float]] = None,
                          conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Main entry point: Answer user query with ML pipeline
        
        Args:
            query: User's question
            user_location: Optional GPS coordinates {"lat": float, "lng": float}
            conversation_history: Previous messages for context
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Classify intent
            intent, confidence = await self._classify_intent(query)
            logger.info(f"🎯 Intent: {intent.value} (confidence: {confidence:.2f})")
            
            # Step 2: Build context based on intent
            context = await self._build_context(query, intent, user_location)
            
            # Step 3: Generate response with LLM
            response = await self._generate_response(context)
            
            # Step 4: Generate map data for transportation queries
            map_data = await self._generate_map_data(context, intent)
            
            elapsed_time = time.time() - start_time
            
            return {
                "answer": response,
                "intent": intent.value,
                "confidence": confidence,
                "processing_time": elapsed_time,
                "sources": context.search_results if context.search_results else [],
                "map_data": map_data,
                "has_map_visualization": map_data is not None,
                "metadata": {
                    "user_location": user_location,
                    "context_used": bool(context.search_results or context.structured_data)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in ML answering pipeline: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing it.",
                "intent": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _classify_intent(self, query: str) -> Tuple[Intent, float]:
        """Classify user intent with smart overrides"""
        
        # FIRST: Check for keyword overrides (catches neural misclassifications)
        keyword_override = self._check_keyword_override(query)
        if keyword_override:
            intent, confidence = keyword_override
            logger.info(f"🔑 Keyword override: {intent.value} ({confidence:.2f})")
            return intent, confidence
        
        # SECOND: Try neural classification
        if not self.intent_classifier:
            return self._keyword_classify(query)
        
        try:
            # Neural classifier returns tuple (intent_str, confidence)
            intent_str, confidence = self.intent_classifier.predict(query)
            
            # Map neural intents to our Intent enum
            intent = self._map_neural_intent(intent_str)
            logger.info(f"🧠 Neural: {intent.value} ({confidence:.2f})")
            return intent, confidence
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}, using fallback")
            return self._keyword_classify(query)
    
    def _check_keyword_override(self, query: str) -> Optional[Tuple[Intent, float]]:
        """
        Check for patterns that should override neural classification
        Returns None if no override needed
        """
        query_lower = query.lower()
        
        # Famous Istanbul attractions - override neural when mentioned
        famous_attractions = {
            "hagia sophia": "Byzantine masterpiece",
            "blue mosque": "Iconic mosque",
            "sultanahmet mosque": "Historic mosque",
            "topkapi": "Ottoman palace",
            "topkapı": "Ottoman palace",
            "galata tower": "Medieval tower",
            "basilica cistern": "Underground cistern",
            "grand bazaar": "Historic market",
            "spice bazaar": "Colorful market",
            "dolmabahce": "Ottoman palace",
            "dolmabahçe": "Ottoman palace",
            "maiden's tower": "Historic tower",
            "chora church": "Byzantine church"
        }
        
        # Check if query is asking about a famous attraction
        for attraction_name in famous_attractions.keys():
            if attraction_name in query_lower:
                # Likely asking about the attraction
                return Intent.ATTRACTION_QUERY, 0.95
        
        # Strong transportation patterns
        transportation_patterns = [
            "how do i get", "how can i get", "how to get", "how to go",
            "directions to", "navigate to", "way to get",
            "take me to", "show me the way", "from my location", "from here to"
        ]
        if any(pattern in query_lower for pattern in transportation_patterns):
            return Intent.TRANSPORTATION_HELP, 0.95
        
        # No override needed
        return None
    
    def _map_neural_intent(self, neural_intent: str) -> Intent:
        """Map neural classifier intents to Intent enum"""
        # Neural classifier uses: restaurant, attraction, neighborhood, transportation,
        # daily_talks, hidden_gems, weather, events, route_planning, general_info
        
        mapping = {
            "restaurant": Intent.RESTAURANT_RECOMMENDATION,
            "attraction": Intent.ATTRACTION_QUERY,
            "neighborhood": Intent.NEIGHBORHOOD_INFO,
            "transportation": Intent.TRANSPORTATION_HELP,
            "daily_talks": Intent.DAILY_TALK,
            "hidden_gems": Intent.LOCAL_TIPS,
            "weather": Intent.WEATHER_INFO,
            "events": Intent.EVENTS_QUERY,
            "route_planning": Intent.ROUTE_PLANNING,
            "general_info": Intent.DAILY_TALK
        }
        
        return mapping.get(neural_intent, Intent.DAILY_TALK)
    
    def _keyword_classify(self, query: str) -> Tuple[Intent, float]:
        """
        Enhanced keyword-based classification with better pattern matching
        
        This is more sophisticated than simple keyword matching - it looks for
        patterns and context to avoid misclassification.
        """
        query_lower = query.lower()
        
        # Famous attractions (highest priority - very specific)
        famous_attractions = [
            "hagia sophia", "blue mosque", "sultanahmet", "topkapi", "topkapı",
            "galata tower", "basilica cistern", "grand bazaar", "spice bazaar",
            "dolmabahce", "dolmabahçe", "palace", "maiden's tower", "bosphorus cruise"
        ]
        if any(attr in query_lower for attr in famous_attractions):
            return Intent.ATTRACTION_QUERY, 0.90
        
        # Neighborhood names (high priority - specific location queries)
        neighborhoods = [
            "kadıköy", "kadiköy", "kadikoy",
            "beyoğlu", "beyoglu",
            "beşiktaş", "besiktas",
            "sultanahmet",
            "taksim",
            "ortaköy", "ortakoy",
            "balat",
            "karaköy", "karakoy",
            "üsküdar", "uskudar",
            "fatih",
            "şişli", "sisli"
        ]
        # Check if query is ABOUT a neighborhood (not just mentions it for location)
        neighborhood_query_patterns = ["tell me about", "what is", "describe", "neighborhood", "area", "district"]
        if any(n in query_lower for n in neighborhoods):
            if any(pattern in query_lower for pattern in neighborhood_query_patterns):
                return Intent.NEIGHBORHOOD_INFO, 0.90
        
        # Transportation (high priority - clear action verbs)
        transportation_patterns = [
            "how do i get", "how can i get", "how to get", "how to go",
            "directions to", "navigate to", "route to", "way to get",
            "take me to", "show me the way", "from my location", "from here",
            "metro to", "bus to", "tram to", "ferry to",
            "how far is", "distance to", "travel to",
            "transportation", "public transport"
        ]
        if any(pattern in query_lower for pattern in transportation_patterns):
            return Intent.TRANSPORTATION_HELP, 0.95
        
        # Weather (distinct keywords, unlikely to confuse)
        weather_keywords = [
            "weather", "temperature", "rain", "sunny", "forecast",
            "hot", "cold", "humid", "climate", "umbrella"
        ]
        if any(word in query_lower for word in weather_keywords):
            return Intent.WEATHER_INFO, 0.90
        
        # Events (distinct keywords)
        event_keywords = [
            "event", "concert", "festival", "exhibition", "show",
            "performance", "happening tonight", "happening today",
            "what's on", "whats on"
        ]
        if any(word in query_lower for word in event_keywords):
            return Intent.EVENTS_QUERY, 0.85
        
        # Route planning (looking for itinerary/planning words)
        route_keywords = [
            "itinerary", "plan my day", "day trip", "route",
            "visiting multiple", "tour plan", "schedule"
        ]
        if any(word in query_lower for word in route_keywords):
            return Intent.ROUTE_PLANNING, 0.85
        
        # Local tips/hidden gems
        tips_keywords = [
            "hidden gem", "local tip", "secret", "off the beaten",
            "locals go", "insider", "authentic", "non-touristy"
        ]
        if any(word in query_lower for word in tips_keywords):
            return Intent.LOCAL_TIPS, 0.85
        
        # Restaurant (food-related, but check it's not just mentioning food in attraction context)
        restaurant_keywords = ["restaurant", "food", "eat", "dining", "cuisine", "meal", "lunch", "dinner", "breakfast"]
        if any(word in query_lower for word in restaurant_keywords):
            # Make sure it's actually about finding food, not about an attraction that serves food
            if not any(word in query_lower for word in ["museum", "palace", "tower", "mosque"]):
                return Intent.RESTAURANT_RECOMMENDATION, 0.80
        
        # Attraction (broader check, lower confidence as it's more generic)
        attraction_keywords = ["museum", "mosque", "church", "palace", "visit", "see", "tourist", "sightseeing"]
        if any(word in query_lower for word in attraction_keywords):
            return Intent.ATTRACTION_QUERY, 0.75
        
        # Default to daily talk with low confidence
        return Intent.DAILY_TALK, 0.50
    
    async def _build_context(self, 
                            query: str, 
                            intent: Intent, 
                            user_location: Optional[Dict[str, float]]) -> QueryContext:
        """Build context based on intent"""
        context = QueryContext(
            query=query,
            intent=intent,
            user_location=user_location,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Route to appropriate context builder
        if intent == Intent.RESTAURANT_RECOMMENDATION:
            await self._build_restaurant_context(context)
        elif intent == Intent.ATTRACTION_QUERY:
            await self._build_attraction_context(context)
        elif intent == Intent.NEIGHBORHOOD_INFO:
            await self._build_neighborhood_context(context)
        elif intent == Intent.TRANSPORTATION_HELP:
            await self._build_transportation_context(context)
        elif intent == Intent.WEATHER_INFO:
            await self._build_weather_context(context)
        elif intent == Intent.EVENTS_QUERY:
            await self._build_events_context(context)
        elif intent == Intent.ROUTE_PLANNING:
            await self._build_route_context(context)
        elif intent == Intent.LOCAL_TIPS:
            await self._build_tips_context(context)
        # DAILY_TALK doesn't need additional context
        
        return context
    
    async def _build_restaurant_context(self, context: QueryContext):
        """Build context for restaurant recommendations"""
        if not self.semantic_search:
            logger.warning("Semantic search not available")
            return
        
        try:
            # Perform semantic search on restaurants
            results = self.semantic_search.search(
                query=context.query,
                top_k=5,
                collection="restaurants"
            )
            
            context.search_results = results
            logger.info(f"  📍 Found {len(results)} restaurants")
            
        except Exception as e:
            logger.error(f"Error building restaurant context: {e}")
    
    async def _build_attraction_context(self, context: QueryContext):
        """Build context for attraction queries"""
        if not self.semantic_search:
            return
        
        try:
            results = self.semantic_search.search(
                query=context.query,
                top_k=5,
                collection="attractions"
            )
            
            context.search_results = results
            logger.info(f"  🏛️ Found {len(results)} attractions")
            
        except Exception as e:
            logger.error(f"Error building attraction context: {e}")
    
    async def _build_neighborhood_context(self, context: QueryContext):
        """Build context for neighborhood information"""
        # Extract neighborhood name from query
        neighborhoods = [
            "Sultanahmet", "Beyoğlu", "Kadıköy", "Beşiktaş",
            "Ortaköy", "Balat", "Karaköy", "Üsküdar"
        ]
        
        detected = None
        for neighborhood in neighborhoods:
            if neighborhood.lower() in context.query.lower():
                detected = neighborhood
                break
        
        if detected:
            context.structured_data = {
                "neighborhood": detected,
                "type": "neighborhood_guide"
            }
            logger.info(f"  🏘️ Detected neighborhood: {detected}")
    
    async def _build_transportation_context(self, context: QueryContext):
        """Build context for transportation queries"""
        # Use semantic search to get relevant transportation info
        if self.semantic_search:
            try:
                results = self.semantic_search.search(
                    query=context.query,
                    top_k=3,
                    collection="transportation"
                )
                context.search_results = results
                logger.info(f"  🚇 Found {len(results)} transportation results")
            except Exception as e:
                logger.warning(f"  ⚠️ Transportation search failed: {e}")
        
        context.structured_data = {
            "type": "transportation",
            "user_location": context.user_location,
            "query": context.query
        }
        logger.info("  🚇 Transportation context built")
    
    async def _build_weather_context(self, context: QueryContext):
        """Build context for weather queries"""
        # In production, this would call weather API
        context.structured_data = {
            "type": "weather",
            "city": "Istanbul"
        }
        logger.info("  ☀️ Weather context built")
    
    async def _build_events_context(self, context: QueryContext):
        """Build context for events queries"""
        # Use semantic search to get relevant events
        if self.semantic_search:
            try:
                results = self.semantic_search.search(
                    query=context.query,
                    top_k=5,
                    collection="events"
                )
                context.search_results = results
                logger.info(f"  🎭 Found {len(results)} event results")
            except Exception as e:
                logger.warning(f"  ⚠️ Events search failed: {e}")
        
        context.structured_data = {
            "type": "events",
            "city": "Istanbul"
        }
        logger.info("  🎭 Events context built")
    
    async def _build_route_context(self, context: QueryContext):
        """Build context for route planning"""
        context.structured_data = {
            "type": "route_planning",
            "user_location": context.user_location
        }
        logger.info("  🗺️ Route planning context built")
    
    async def _build_tips_context(self, context: QueryContext):
        """Build context for local tips"""
        if self.semantic_search:
            try:
                results = self.semantic_search.search(
                    query=context.query,
                    top_k=3,
                    collection="tips"
                )
                context.search_results = results
                logger.info(f"  💡 Found {len(results)} local tips")
            except:
                pass
    
    async def _generate_response(self, context: QueryContext) -> str:
        """Generate natural language response using LLM"""
        if not self.llm_generator:
            # Fallback to template-based response
            return self._template_response(context)
        
        try:
            # Generate response using LLM
            # LocalLLMGenerator.generate() expects query and context_data
            response = self.llm_generator.generate(
                query=context.query,
                context_data=context.search_results or context.structured_data,
                max_tokens=512
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._template_response(context)
    
    def _build_llm_prompt(self, context: QueryContext) -> str:
        """Build prompt for LLM based on intent and context"""
        prompt_parts = [
            "You are Istanbul AI, a helpful and knowledgeable guide for Istanbul, Turkey.",
            f"\nUser Question: {context.query}",
        ]
        
        # Add context based on what's available
        if context.search_results:
            prompt_parts.append("\nRelevant Information:")
            for i, result in enumerate(context.search_results[:3], 1):
                prompt_parts.append(f"{i}. {result.get('name', 'N/A')}")
                if 'description' in result:
                    prompt_parts.append(f"   {result['description'][:200]}")
        
        if context.structured_data:
            prompt_parts.append(f"\nAdditional Context: {context.structured_data}")
        
        prompt_parts.append("\nProvide a helpful, accurate, and friendly response:")
        
        return "\n".join(prompt_parts)
    
    def _template_response(self, context: QueryContext) -> str:
        """Fallback template-based response"""
        if context.intent == Intent.RESTAURANT_RECOMMENDATION:
            if context.search_results:
                names = [r.get('name', 'Unknown') for r in context.search_results[:3]]
                return f"Based on your query, I recommend these restaurants: {', '.join(names)}."
            return "I can help you find great restaurants in Istanbul! Could you tell me more about what you're looking for?"
        
        elif context.intent == Intent.ATTRACTION_QUERY:
            if context.search_results:
                names = [r.get('name', 'Unknown') for r in context.search_results[:3]]
                return f"Here are some great attractions: {', '.join(names)}."
            return "Istanbul has amazing attractions! What type of place are you interested in?"
        
        else:
            return "I'm here to help you explore Istanbul! Could you tell me more about what you'd like to know?"


# Factory function to create fully initialized service
async def create_ml_service(enable_llm: bool = False, enable_neural: bool = True) -> MLAnsweringService:
    """
    Factory function to create and initialize ML service
    
    Args:
        enable_llm: Whether to load the LLM (slower, ~18s on CPU)
        enable_neural: Whether to load neural intent classifier (recommended)
        
    Returns:
        Initialized MLAnsweringService
    """
    from ml_systems.semantic_search_engine import SemanticSearchEngine
    
    logger.info("🚀 Initializing ML Answering Service...")
    
    # Load semantic search
    try:
        semantic_search = SemanticSearchEngine()
        await semantic_search.initialize()
        logger.info("  ✅ Semantic search engine loaded")
    except Exception as e:
        logger.error(f"  ❌ Failed to load semantic search: {e}")
        semantic_search = None
    
    # Load neural intent classifier
    intent_classifier = None
    if enable_neural:
        try:
            from neural_query_classifier import NeuralQueryClassifier
            intent_classifier = NeuralQueryClassifier(
                model_path="models/istanbul_intent_classifier_10_final",
                confidence_threshold=0.60
            )
            logger.info("  ✅ Neural intent classifier loaded")
        except Exception as e:
            logger.warning(f"  ⚠️ Neural classifier not loaded: {e}")
            logger.info("  → Will use enhanced keyword-based classification")
    
    # Load LLM generator (optional)
    llm_generator = None
    if enable_llm:
        try:
            from ml_systems.local_llm_generator import LocalLLMGenerator
            llm_generator = LocalLLMGenerator()
            logger.info("  ✅ LLM generator loaded")
        except Exception as e:
            logger.error(f"  ⚠️ LLM not loaded: {e}")
    
    service = MLAnsweringService(
        intent_classifier=intent_classifier,
        semantic_search=semantic_search,
        llm_generator=llm_generator
    )
    
    logger.info("✨ ML Answering Service ready!")
    return service
