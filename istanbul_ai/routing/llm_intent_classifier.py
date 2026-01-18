"""
LLM-Based Intent Classifier

This module uses LLM (Language Learning Model) for advanced intent classification.
It leverages the LLMServiceWrapper to classify user intents using natural language understanding,
replacing the traditional keyword-based or neural network classifier.

Features:
- Natural language understanding for intent classification
- Multilingual support (English, Turkish, French, German, Russian, Arabic, and more)
- Contextual awareness
- Confidence scoring
- Multi-intent detection
- Fallback to neural/keyword classifier for reliability
- LLM response caching for performance optimization
- Performance metrics tracking

Author: Istanbul AI Team
Date: December 2024
"""

import logging
import json
import hashlib
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Intent classification result"""
    primary_intent: str
    confidence: float = 0.0
    intents: List[str] = field(default_factory=list)
    is_multi_intent: bool = False
    multi_intent_response: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    method: str = "llm"  # 'llm', 'keyword', or 'hybrid'


class LLMIntentClassifier:
    """
    LLM-based intent classifier using LLMServiceWrapper
    
    This classifier uses the LLM to understand user intent through natural language,
    providing more accurate and contextual intent classification than keyword matching.
    
    Features:
    - Natural language understanding
    - Context-aware classification
    - Multi-intent detection
    - Multilingual support (EN, TR, FR, DE, RU, AR, and more)
    - Confidence scoring
    - Graceful fallback to neural/keyword classifier
    """
    
    # Define all supported intents
    SUPPORTED_INTENTS = [
        'restaurant',           # Food and dining queries
        'attraction',           # Tourist attractions, museums, landmarks
        'transportation',       # Public transport, metro, bus, taxi
        'weather',             # Weather information and forecasts
        'events',              # Events, concerts, festivals
        'neighborhood',        # Neighborhood information
        'shopping',            # Shopping locations and recommendations
        'hidden_gems',         # Local secrets and hidden gems
        'airport_transport',   # Airport transportation
        'route_planning',      # Route and itinerary planning
        'museum_route_planning',  # Museum-specific route planning
        'gps_route_planning',  # GPS-based route planning
        'nearby_locations',    # Nearby POI search
        'greeting',            # Greetings and casual conversation
        'general'              # General queries
    ]
    
    def __init__(self, llm_service=None, keyword_classifier=None, neural_classifier=None):
        """
        Initialize LLM intent classifier
        
        Args:
            llm_service: LLMServiceWrapper instance (optional, auto-initialized)
                        Will use UnifiedLLMService if USE_UNIFIED_LLM is enabled
            keyword_classifier: Fallback keyword-based IntentClassifier (optional)
            neural_classifier: NeuralQueryClassifier for better fallback (optional, preferred over keyword)
        """
        self.llm_service = llm_service
        self.keyword_classifier = keyword_classifier
        self.neural_classifier = neural_classifier
        self.use_llm = llm_service is not None
        self.has_neural_fallback = neural_classifier is not None
        self.has_keyword_fallback = keyword_classifier is not None
        self.unified_llm_service = None
        
        # Intent cache for LLM responses
        self.intent_cache = {}
        self.cache_ttl = int(os.getenv('LLM_INTENT_CACHE_TTL', '3600'))  # 1 hour default
        self.enable_cache = os.getenv('LLM_INTENT_CACHE_ENABLED', 'true').lower() == 'true'
        
        # Enhanced statistics with performance metrics
        self.stats = {
            'llm_used': 0,
            'neural_fallback': 0,
            'keyword_fallback': 0,
            'llm_failures': 0,
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_response_times': [],  # Track LLM latency in ms
            'avg_llm_latency_ms': 0.0
        }
        
        # Try to initialize UnifiedLLMService if enabled
        if os.getenv('USE_UNIFIED_LLM', 'false').lower() == 'true':
            try:
                from unified_system.services.unified_llm_service import UnifiedLLMService
                self.unified_llm_service = UnifiedLLMService()
                logger.info("✅ LLM Intent Classifier initialized with UnifiedLLMService")
                self.use_llm = True
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize UnifiedLLMService: {e}")
        
        # Initialize LLM if not provided and UnifiedLLM not available
        if not self.use_llm and not self.unified_llm_service:
            try:
                from ml_systems.llm_service_wrapper import LLMServiceWrapper
                self.llm_service = LLMServiceWrapper()
                self.use_llm = True
                logger.info("✅ LLM Intent Classifier initialized with auto-loaded LLM service")
            except Exception as e:
                logger.warning(f"⚠️ Failed to auto-load LLM service: {e}")
                self.use_llm = False
        
        if self.use_llm or self.unified_llm_service:
            if self.unified_llm_service:
                logger.info("✅ LLM Intent Classifier initialized (Using UnifiedLLMService)")
            else:
                logger.info(f"✅ LLM Intent Classifier initialized (Model: {self.llm_service.model_name})")
            if self.has_neural_fallback:
                logger.info("   → Primary fallback: Neural classifier (DistilBERT)")
            if self.has_keyword_fallback:
                logger.info("   → Secondary fallback: Keyword classifier")
        else:
            if self.has_neural_fallback:
                logger.warning("⚠️ LLM Intent Classifier initialized (LLM unavailable - will use neural fallback)")
            elif self.has_keyword_fallback:
                logger.warning("⚠️ LLM Intent Classifier initialized (LLM unavailable - will use keyword fallback)")
            else:
                logger.warning("⚠️ LLM Intent Classifier initialized (No LLM or fallback available)")
    
    def classify_intent(
        self,
        message: str,
        entities: Dict,
        context: Optional[Any] = None,
        neural_insights: Optional[Dict] = None,
        preprocessed_query: Optional[Any] = None,
        **kwargs
    ) -> IntentResult:
        """
        Classify intent using LLM with fallback to keyword classifier
        
        Args:
            message: User's input message
            entities: Extracted entities
            context: Conversation context (optional)
            neural_insights: Neural processing insights (optional)
            preprocessed_query: Preprocessed query data (optional)
            **kwargs: Additional arguments
        
        Returns:
            IntentResult with classification details
        """
        self.stats['total_requests'] += 1
        
        # Try LLM classification first
        if self.use_llm:
            try:
                # Generate cache key for the message
                cache_key = self._get_cache_key(message)
                
                # Check cache first
                cached_result = self._get_cached_intent(cache_key)
                if cached_result:
                    logger.info(f"💾 Cache hit for message: '{message}' (key: {cache_key[:8]}...)")
                    return cached_result
                
                result = self._classify_with_llm(message, entities, context)
                self.stats['llm_used'] += 1
                
                # Cache the result
                self._cache_intent(cache_key, result)
                
                return result
            except Exception as e:
                logger.warning(f"⚠️ LLM classification failed: {e}")
                self.stats['llm_failures'] += 1
        
        # Fallback to neural classifier (preferred - more accurate than keyword)
        if self.has_neural_fallback:
            logger.debug("Using neural fallback for intent classification")
            self.stats['neural_fallback'] += 1
            try:
                # Neural classifier returns (intent, confidence)
                intent, confidence = self.neural_classifier.predict(message)
                
                # Map neural classifier intents to our intent names
                intent = self._map_neural_intent(intent)
                
                return IntentResult(
                    primary_intent=intent,
                    confidence=confidence,
                    intents=[intent],
                    entities=entities,
                    method='neural_fallback'
                )
            except Exception as e:
                logger.warning(f"⚠️ Neural classifier fallback failed: {e}")
        
        # Fallback to keyword classifier (last resort before 'general')
        if self.has_keyword_fallback:
            logger.debug("Using keyword fallback for intent classification")
            self.stats['keyword_fallback'] += 1
            result = self.keyword_classifier.classify_intent(
                message, entities, context, neural_insights, preprocessed_query
            )
            # Update method to indicate fallback was used
            result.method = "keyword_fallback"
            return result
        
        # Last resort: return general intent
        logger.warning("No classifier available, returning general intent")
        return IntentResult(
            primary_intent='general',
            confidence=0.5,
            intents=['general'],
            entities=entities,
            method='default'
        )
    
    async def _classify_with_llm_async(
        self,
        message: str,
        entities: Dict,
        context: Optional[Any] = None
    ) -> IntentResult:
        """
        Async version of LLM classification (for UnifiedLLMService)
        
        Args:
            message: User's input message
            entities: Extracted entities
            context: Conversation context
            
        Returns:
            IntentResult with classification
        """
        # Extract language from context
        language = self._get_language(context)
        
        # Build classification prompt
        prompt = self._build_classification_prompt(message, entities, language, context)
        
        # Track LLM response time
        start_time = time.time()
        
        # Use UnifiedLLMService
        llm_response = await self.unified_llm_service.complete_text(
            prompt=prompt,
            max_tokens=100,
            temperature=0.2,
            component="llm_intent_classifier"
        )
        
        # Track response time
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        self.stats['llm_response_times'].append(latency_ms)
        
        # Update average latency
        self.stats['avg_llm_latency_ms'] = sum(self.stats['llm_response_times']) / len(self.stats['llm_response_times'])
        
        logger.debug(f"⏱️  LLM classification latency: {latency_ms:.2f}ms")
        
        # Parse LLM response
        result = self._parse_llm_response(llm_response, message, entities)
        
        return result
    
    def _classify_with_llm(
        self,
        message: str,
        entities: Dict,
        context: Optional[Any] = None
    ) -> IntentResult:
        """
        Classify intent using LLM with caching
        
        Args:
            message: User's input message
            entities: Extracted entities
            context: Conversation context
            
        Returns:
            IntentResult with classification
        """
        # Check cache first
        cache_key = self._get_cache_key(message)
        cached_result = self._get_cached_intent(cache_key)
        
        if cached_result:
            return cached_result
        
        # Use async version for UnifiedLLMService
        if self.unified_llm_service:
            import asyncio
            try:
                # Check if we can use async directly
                try:
                    loop = asyncio.get_running_loop()
                    # We're already in an async event loop
                    # We can't use asyncio.run() here, so we need to schedule it properly
                    # For now, we'll fall back to synchronous mode
                    logger.warning("Cannot use UnifiedLLMService from sync context in async loop - falling back")
                    # Fall through to legacy service
                except RuntimeError:
                    # No running loop - we can create one
                    result = asyncio.run(self._classify_with_llm_async(message, entities, context))
                    self._cache_intent(cache_key, result)
                    return result
            except Exception as e:
                logger.error(f"Unified LLM classification failed: {e}")
                # Fall through to legacy service
        
        # Legacy LLM service (synchronous)
        if not self.llm_service:
            # No LLM service available at all
            logger.warning("No LLM service available for classification")
            raise ValueError("No LLM service available")
        
        # Extract language from context
        language = self._get_language(context)
        
        # Build classification prompt
        prompt = self._build_classification_prompt(message, entities, language, context)
        
        # Track LLM response time
        start_time = time.time()
        
        # Check if using Llama 3.x for optimized parameters
        is_llama_3 = self.llm_service and hasattr(self.llm_service, 'model_name') and 'llama-3' in self.llm_service.model_name.lower()
        
        # Get LLM response with model-specific parameters
        if is_llama_3:
            # Llama 3.x: Better at following instructions, can use slightly higher temperature
            llm_response = self.llm_service.generate(
                prompt=prompt,
                max_tokens=100,  # Llama 3 can handle structured output better
                temperature=0.2,  # Low but not too restrictive
                stop_sequences=['<|eot_id|>', '<|end_of_text|>']  # Llama 3 stop tokens
            )
        else:
            # TinyLlama: Needs very low temperature and minimal tokens
            llm_response = self.llm_service.generate(
                prompt=prompt,
                max_tokens=80,
                temperature=0.1  # Very low for deterministic classification
            )
        
        # Track response time
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        self.stats['llm_response_times'].append(latency_ms)
        
        # Update average latency
        self.stats['avg_llm_latency_ms'] = sum(self.stats['llm_response_times']) / len(self.stats['llm_response_times'])
        
        logger.debug(f"⏱️  LLM classification latency: {latency_ms:.2f}ms")
        
        # Parse LLM response
        result = self._parse_llm_response(llm_response, message, entities)
        
        # Cache the result
        self._cache_intent(cache_key, result)
        
        return result
    
    def _build_classification_prompt(
        self,
        message: str,
        entities: Dict,
        language: str,
        context: Optional[Any] = None
    ) -> str:
        """
        Build classification prompt for LLM
        
        Args:
            message: User's input message
            entities: Extracted entities
            language: User's language ('en' or 'tr')
            context: Conversation context
            
        Returns:
            Formatted prompt string
        """
        # Build entity context
        entity_context = ""
        if entities:
            entity_items = [f"- {k}: {v}" for k, v in entities.items() if v]
            if entity_items:
                entity_context = "Entities found:\n" + "\n".join(entity_items) + "\n\n"
        
        # Build conversation context
        conversation_context = ""
        if context and hasattr(context, 'recent_intents') and context.recent_intents:
            recent = context.recent_intents[-3:]  # Last 3 intents
            conversation_context = f"Recent conversation topics: {', '.join(recent)}\n\n"
        
        # Check if using Llama 3.x model (better instruction following)
        is_llama_3 = self.llm_service and hasattr(self.llm_service, 'model_name') and 'llama-3' in self.llm_service.model_name.lower()
        
        if is_llama_3:
            # Llama 3.x Instruct format - optimized for instruction following and JSON output
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an intent classification system for Istanbul travel queries. Analyze user messages and return ONLY valid JSON.

Output format: {{"primary_intent":"intent_name","confidence":0.95,"all_intents":["intent_name"]}}

Supported intents: greeting, restaurant, attraction, transportation, weather, events, neighborhood, shopping, hidden_gems, airport_transport, route_planning, museum_route_planning, gps_route_planning, nearby_locations, general

Classification rules:
- greeting: Hello, hi, thanks, goodbye, how are you, good morning/evening, greetings in any language
- restaurant: Food, dining, eat, cuisine queries
- attraction: Sightseeing, landmarks, places to visit
- transportation: How to get, metro, bus, taxi, ferry
- weather: Weather, temperature, forecast
- general: Everything else

Examples:
"Hello!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Merhaba!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"How are you?" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Good morning!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Günaydın!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Thanks!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Goodbye!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Hoşça kal!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Hey there!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Selam!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"What's up?" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Bonjour!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Find restaurant" -> {{"primary_intent":"restaurant","confidence":0.90,"all_intents":["restaurant"]}}
"Weather today?" -> {{"primary_intent":"weather","confidence":0.90,"all_intents":["weather"]}}
"Hagia Sophia" -> {{"primary_intent":"attraction","confidence":0.90,"all_intents":["attraction"]}}
"How to Taksim?" -> {{"primary_intent":"transportation","confidence":0.90,"all_intents":["transportation"]}}
<|eot_id|><|start_header_id|>user<|end_header_id|>

Classify: "{message}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            # TinyLlama fallback - ultra-compact format
            prompt = f"""Classify intent. Output only JSON: {{"primary_intent":"X","confidence":0.9,"all_intents":["X"]}}

"Hello!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Merhaba!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"How are you?" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Good morning!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Günaydın!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Thanks!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Goodbye!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Hoşça kal!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Hey!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Selam!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"What's up?" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Bonjour!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Weather?" = {{"primary_intent":"weather","confidence":0.90,"all_intents":["weather"]}}
"Hava nasıl?" = {{"primary_intent":"weather","confidence":0.90,"all_intents":["weather"]}}
"Find restaurant" = {{"primary_intent":"restaurant","confidence":0.90,"all_intents":["restaurant"]}}
"Taksim?" = {{"primary_intent":"transportation","confidence":0.90,"all_intents":["transportation"]}}
"Hagia Sophia" = {{"primary_intent":"attraction","confidence":0.90,"all_intents":["attraction"]}}

"{message}" = """
        
        return prompt
    
    def _parse_llm_response(
        self,
        llm_response: str,
        message: str,
        entities: Dict
    ) -> IntentResult:
        """
        Parse LLM response and extract intent classification
        
        Args:
            llm_response: Raw LLM response
            message: Original user message
            entities: Extracted entities
            
        Returns:
            IntentResult with classification
        """
        try:
            # Try to extract JSON from response
            # LLM might return text before/after JSON, so we need to find it
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Extract fields - handle both snake_case and camelCase
                primary_intent = (
                    parsed.get('primary_intent') or 
                    parsed.get('primaryIntent') or 
                    parsed.get('intent') or 
                    'general'
                )
                
                confidence = float(
                    parsed.get('confidence') or 
                    parsed.get('score') or 
                    0.8
                )
                
                # Handle all_intents field - can be array of strings or array of objects
                all_intents_raw = (
                    parsed.get('all_intents') or 
                    parsed.get('allIntents') or 
                    parsed.get('intents') or 
                    [primary_intent]
                )
                
                # Normalize all_intents to array of strings
                all_intents = []
                if isinstance(all_intents_raw, list):
                    for item in all_intents_raw:
                        if isinstance(item, str):
                            all_intents.append(item)
                        elif isinstance(item, dict):
                            # Handle {"intentName": "weather"} format
                            intent_name = (
                                item.get('intentName') or 
                                item.get('intent_name') or 
                                item.get('name') or 
                                item.get('intent')
                            )
                            if intent_name:
                                all_intents.append(intent_name)
                
                # Ensure primary_intent is in all_intents
                if not all_intents or primary_intent not in all_intents:
                    all_intents = [primary_intent]
                
                # Validate intent
                if primary_intent not in self.SUPPORTED_INTENTS:
                    logger.warning(f"LLM returned unsupported intent: {primary_intent}, defaulting to 'general'")
                    primary_intent = 'general'
                    confidence = 0.6
                
                # Validate all_intents
                all_intents = [
                    intent for intent in all_intents 
                    if intent in self.SUPPORTED_INTENTS
                ]
                if not all_intents:
                    all_intents = [primary_intent]
                
                # Determine if multi-intent
                is_multi_intent = len(all_intents) > 1
                
                logger.debug(f"LLM classified '{message}' as '{primary_intent}' (confidence: {confidence:.2f})")
                
                return IntentResult(
                    primary_intent=primary_intent,
                    confidence=min(max(confidence, 0.0), 1.0),  # Clamp to [0, 1]
                    intents=all_intents,
                    is_multi_intent=is_multi_intent,
                    entities=entities,
                    method='llm'
                )
            else:
                raise ValueError("No JSON found in LLM response")
                
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"LLM response was: {llm_response[:200]}")  # Log first 200 chars
            
            # Fallback: Try to extract intent from the original message using keywords
            # This is more reliable than trying to parse the LLM's text output
            primary_intent = self._extract_intent_from_message(message)
            
            logger.debug(f"Using keyword fallback: '{message}' -> '{primary_intent}'")
            
            return IntentResult(
                primary_intent=primary_intent,
                confidence=0.7,
                intents=[primary_intent],
                entities=entities,
                method='llm_fallback'
            )
    
    def _extract_intent_from_message(self, message: str) -> str:
        """
        Extract intent directly from user message using keyword matching
        
        Args:
            message: User's message
            
        Returns:
            Intent name
        """
        message_lower = message.lower()
        
        # Define keyword patterns for each intent (more specific patterns first)
        # Include both English and Turkish keywords
        intent_patterns = [
            # Weather (very specific)
            ('weather', [
                # English
                'weather', 'temperature', 'rain', 'sunny', 'forecast', 'climate', 'hot', 'cold',
                # Turkish
                'hava', 'sıcaklık', 'yağmur', 'güneşli', 'tahmin', 'iklim', 'sıcak', 'soğuk',
                'hava durumu', 'bugün hava', 'yarın hava',
                # French
                'météo', 'temps', 'température', 'pluie', 'prévis',
                # German  
                'wetter', 'temperatur', 'regen', 'sonnig',
                # Russian (transliterated)
                'pogoda', 'temperatura', 'dozhd',
                # Arabic (transliterated)
                'altaqs', 'matara', 'jaww'
            ]),
            
            # Transportation
            ('transportation', [
                # English
                'metro', 'bus', 'ferry', 'tram', 'train', 'public transport', 'how to get', 'how do i get',
                # Turkish
                'otobüs', 'vapur', 'tramvay', 'tren', 'toplu taşıma', 'nasıl giderim', 'nasıl gidilir',
                'ulaşım', 'nasıl ulaşırım', 'yol tarifi',
                # French
                'comment aller', 'transport', 'métro', 'tramway', 'autobus',
                # German
                'wie komme ich', 'u-bahn', 'straßenbahn', 'verkehr',
                # Russian (transliterated)
                'kak dobratsya', 'transport', 'avtobus',
                # Arabic (transliterated)
                'kayfa', 'naql', 'metro'
            ]),
            ('airport_transport', [
                # English
                'airport', 'flight', 'terminal', 'istanbul airport', 'sabiha gokcen',
                # Turkish
                'havalimanı', 'havaalanı', 'uçuş', 'istanbul havalimanı', 'sabiha gökçen',
                # French
                'aéroport', 'vol',
                # German
                'flughafen', 'flug',
                # Russian (transliterated)
                'aeroport', 'samolet',
                # Arabic (transliterated)
                'matar', 'tayara'
            ]),
            
            # Route planning
            ('museum_route_planning', [
                # English
                'museum route', 'museum tour', 'museum itinerary', 'visit museums',
                # Turkish
                'müze rotası', 'müze turu', 'müze ziyareti', 'müzeleri gez'
            ]),
            ('gps_route_planning', [
                # English
                'gps', 'directions', 'navigate', 'navigation', 'turn by turn', 'walking directions',
                # Turkish
                'gps', 'yön tarifi', 'navigasyon', 'yaya yolu', 'adım adım'
            ]),
            ('route_planning', [
                # English
                'route', 'itinerary', 'plan my day', 'day trip', 'visit', 'tour plan',
                # Turkish
                'rota', 'plan', 'günümü planla', 'gezi planı', 'ziyaret', 'tur planı',
                'ne yapmalıyım', 'nereye gideyim',
                # French
                'itinéraire', 'route', 'planifier', 'visite',
                # German
                'reiseplan', 'route', 'besuch',
                # Russian (transliterated)
                'marshrut', 'plan', 'poseshchat',
                # Arabic (transliterated)
                'tareeq', 'khitta', 'ziyara'
            ]),
            
            # Restaurants and food
            ('restaurant', [
                # English
                'restaurant', 'food', 'eat', 'dining', 'kebab', 'meal', 'cuisine', 'where to eat', 'hungry',
                # Turkish
                'restoran', 'lokanta', 'yemek', 'ye', 'yiyelim', 'kebap', 'kebab', 'mutfak', 
                'nerede yenir', 'acıktım', 'yemek nerede', 'ne yesem',
                # French
                'restaurant', 'manger', 'nourriture', 'où manger', 'repas',
                # German
                'restaurant', 'essen', 'wo essen', 'mahlzeit',
                # Russian (transliterated)
                'restoran', 'eda', 'kushat', 'gde poest',
                # Arabic (transliterated)
                'mataam', 'akl', 'taaam'
            ]),
            
            # Attractions
            ('attraction', [
                # English
                'hagia sophia', 'blue mosque', 'topkapi', 'galata tower', 'landmark', 'monument', 
                'tourist attraction', 'sights', 'see in istanbul',
                # Turkish
                'ayasofya', 'sultanahmet camii', 'topkapı', 'galata kulesi', 'anıt', 
                'turistik yer', 'gezilecek yer', 'görülecek yer', 'müze'
            ]),
            
            # Events
            ('events', [
                # English
                'event', 'concert', 'festival', 'exhibition', 'show', 'performance', 'happening', 'whats on',
                # Turkish
                'etkinlik', 'konser', 'festival', 'sergi', 'gösteri', 'performans', 'ne var', 'ne yapılıyor'
            ]),
            
            # Hidden gems
            ('hidden_gems', [
                # English
                'hidden gem', 'secret', 'local spot', 'authentic', 'off the beaten', 'locals go',
                # Turkish
                'gizli', 'saklı', 'yerel', 'otantik', 'yerli', 'turistik olmayan',
                'yerel mekan', 'gizli cennet'
            ]),
            
            # Neighborhood
            ('neighborhood', [
                # English
                'neighborhood', 'district', 'area', 'where to stay', 'besiktas', 'kadikoy', 'taksim', 'sultanahmet',
                # Turkish
                'mahalle', 'semt', 'bölge', 'nerede kalmalı', 'beşiktaş', 'kadıköy', 'taksim', 'sultanahmet',
                'hangi semtte', 'hangi bölge'
            ]),
            
            # Shopping
            ('shopping', [
                # English
                'shopping', 'shop', 'market', 'bazaar', 'grand bazaar', 'buy', 'souvenir',
                # Turkish
                'alışveriş', 'market', 'pazar', 'çarşı', 'kapalıçarşı', 'al', 'satın al', 'hediyelik'
            ]),
            
            # Nearby
            ('nearby_locations', [
                # English
                'nearby', 'near me', 'close to', 'around here', 'in the area',
                # Turkish
                'yakın', 'yakında', 'yakınımda', 'burada', 'civarda', 'çevrede'
            ]),
            
            # Greetings
            ('greeting', [
                # English
                'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye', 'good morning', 'good evening',
                # Turkish
                'merhaba', 'selam', 'teşekkür', 'teşekkürler', 'sağol', 'güle güle', 
                'günaydın', 'iyi akşamlar', 'hoşçakal'
            ]),
        ]
        
        # Check patterns in order (most specific first)
        for intent, keywords in intent_patterns:
            for keyword in keywords:
                if keyword in message_lower:
                    return intent
        
        # Default to general
        return 'general'
    
    def _extract_intent_from_text(self, text: str) -> str:
        """
        Extract intent from text using keyword matching (fallback)
        
        Args:
            text: Text to extract intent from
            
        Returns:
            Intent name
        """
        text_lower = text.lower()
        
        # Define keyword patterns for each intent (more specific first)
        intent_keywords = {
            'weather': ['weather', 'temperature', 'rain', 'sunny', 'forecast', 'climate'],
            'restaurant': ['restaurant', 'food', 'eat', 'dining', 'kebab', 'meal', 'cuisine'],
            'transportation': ['transport', 'metro', 'bus', 'ferry', 'taxi', 'train', 'tram'],
            'airport_transport': ['airport', 'flight', 'terminal', 'istanbul airport'],
            'museum_route_planning': ['museum route', 'museum tour', 'museum itinerary'],
            'gps_route_planning': ['gps', 'directions', 'navigate', 'navigation', 'turn by turn'],
            'route_planning': ['route', 'itinerary', 'plan', 'schedule', 'day trip'],
            'attraction': ['attraction', 'landmark', 'monument', 'hagia sophia', 'blue mosque', 'tourist'],
            'events': ['event', 'concert', 'festival', 'exhibition', 'show', 'performance'],
            'hidden_gems': ['hidden gem', 'secret', 'local spot', 'authentic', 'off the beaten'],
            'neighborhood': ['neighborhood', 'district', 'area', 'where to stay', 'besiktas', 'kadikoy'],
            'shopping': ['shopping', 'shop', 'market', 'bazaar', 'grand bazaar', 'buy'],
            'nearby_locations': ['nearby', 'near me', 'close to', 'around here'],
            'greeting': ['hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye'],
        }
        
        # Check for keyword matches (prioritize more specific intents)
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    logger.debug(f"Fallback matched '{keyword}' -> {intent}")
                    return intent
        
        # Default to general
        return 'general'
    
    def _get_language(self, context) -> str:
        """
        Extract language from context
        
        Args:
            context: Conversation context
            
        Returns:
            Language code ('en' or 'tr')
        """
        if not context:
            return 'en'
        
        # Check if language is in context
        if hasattr(context, 'language'):
            lang = context.language
            if hasattr(lang, 'value'):
                return lang.value  # Language enum
            return lang if lang in ['en', 'tr'] else 'en'
        
        # Default to English
        return 'en'
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier usage statistics
        
        Returns:
            Dictionary with statistics
        """
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'llm_success_rate': f"{(self.stats['llm_used'] / total * 100):.1f}%",
            'neural_fallback_rate': f"{(self.stats['neural_fallback'] / total * 100):.1f}%",
            'keyword_fallback_rate': f"{(self.stats['keyword_fallback'] / total * 100):.1f}%",
            'failure_rate': f"{(self.stats['llm_failures'] / total * 100):.1f}%"
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Dictionary with performance metrics
        """
        total_requests = self.stats['total_requests']
        
        if total_requests == 0:
            return {'error': 'No data available'}
        
        llm_success = self.stats['llm_used']
        llm_total = llm_success + self.stats['llm_failures']
        
        report = {
            'total_requests': total_requests,
            'llm_used': llm_success,
            'llm_usage_rate_pct': round(llm_success / total_requests * 100, 2) if total_requests > 0 else 0,
            'llm_success_rate_pct': round(llm_success / llm_total * 100, 2) if llm_total > 0 else 0,
            'llm_failures': self.stats['llm_failures'],
            'neural_fallback_count': self.stats['neural_fallback'],
            'keyword_fallback_count': self.stats['keyword_fallback'],
            'avg_llm_latency_ms': round(self.stats['avg_llm_latency_ms'], 2),
            'cache_stats': self.get_cache_stats()
        }
        
        return report
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'llm_used': 0,
            'neural_fallback': 0,
            'keyword_fallback': 0,
            'llm_failures': 0,
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_response_times': [],
            'avg_llm_latency_ms': 0.0
        }
        logger.info("📊 LLM classifier statistics reset")
    
    def _map_neural_intent(self, neural_intent: str) -> str:
        """
        Map neural classifier intent names to our intent names
        
        The neural classifier uses slightly different intent names:
        - "daily_talks" -> "greeting"
        - "general_info" -> "general"
        - others map directly
        
        Args:
            neural_intent: Intent from neural classifier
            
        Returns:
            Mapped intent name
        """
        intent_mapping = {
            'daily_talks': 'greeting',
            'general_info': 'general',
            # These map directly:
            'restaurant': 'restaurant',
            'attraction': 'attraction',
            'neighborhood': 'neighborhood',
            'transportation': 'transportation',
            'hidden_gems': 'hidden_gems',
            'weather': 'weather',
            'events': 'events',
            'route_planning': 'route_planning'
        }
        
        mapped = intent_mapping.get(neural_intent, neural_intent)
        
        # Ensure the mapped intent is supported
        if mapped not in self.SUPPORTED_INTENTS:
            logger.warning(f"Neural classifier returned unmapped intent: {neural_intent}, using 'general'")
            return 'general'
        
        return mapped
    
    def _get_cache_key(self, message: str) -> str:
        """
        Generate cache key for message
        
        Args:
            message: User's input message
            
        Returns:
            Cache key (MD5 hash of normalized message)
        """
        # Normalize message for better cache hits
        normalized = message.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_cached_intent(self, cache_key: str) -> Optional[IntentResult]:
        """
        Get cached intent if valid and not expired
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached IntentResult or None if not found/expired
        """
        if not self.enable_cache:
            return None
            
        if cache_key in self.intent_cache:
            cached_data = self.intent_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_ttl):
                self.stats['cache_hits'] += 1
                logger.info(f"💾 LLM intent cache HIT (key: {cache_key[:8]}...)")
                return cached_data['result']
            else:
                # Expired - remove from cache
                del self.intent_cache[cache_key]
                logger.debug(f"🗑️  Cache entry expired and removed (key: {cache_key[:8]}...)")
        
        self.stats['cache_misses'] += 1
        return None
    
    def _cache_intent(self, cache_key: str, result: IntentResult):
        """
        Cache intent classification result
        
        Args:
            cache_key: Cache key
            result: IntentResult to cache
        """
        if not self.enable_cache:
            return
            
        self.intent_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        logger.debug(f"💾 Cached LLM intent (key: {cache_key[:8]}..., intent: {result.primary_intent})")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Dictionary with cache metrics
        """
        total_cache_attempts = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (
            round(self.stats['cache_hits'] / total_cache_attempts * 100, 2)
            if total_cache_attempts > 0 else 0.0
        )
        
        return {
            'cache_enabled': self.enable_cache,
            'cache_ttl_seconds': self.cache_ttl,
            'cache_size': len(self.intent_cache),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate_pct': cache_hit_rate
        }
    
    def clear_cache(self):
        """Clear the intent cache"""
        cache_size = len(self.intent_cache)
        self.intent_cache.clear()
        logger.info(f"🗑️  Cleared LLM intent cache ({cache_size} entries removed)")
    

def create_llm_intent_classifier(llm_service=None, keyword_classifier=None, neural_classifier=None) -> LLMIntentClassifier:
    """
    Factory function to create LLM intent classifier
    
    Args:
        llm_service: Optional LLM service instance
        keyword_classifier: Optional keyword classifier for fallback
        neural_classifier: Optional neural classifier for better fallback (preferred)
        
    Returns:
        LLMIntentClassifier instance
    """
    return LLMIntentClassifier(
        llm_service=llm_service,
        keyword_classifier=keyword_classifier,
        neural_classifier=neural_classifier
    )


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create classifier
    classifier = LLMIntentClassifier()
    
    # Test queries
    test_queries = [
        "Where can I find good kebab restaurants?",
        "How do I get to Taksim from here?",
        "What's the weather like today?",
        "Show me some hidden gems in Kadıköy",
        "Plan a route to visit 3 museums tomorrow",
        "What events are happening tonight?",
        "Hello! Can you help me?",
    ]
    
    print("\n" + "="*80)
    print("LLM Intent Classifier - Test Results")
    print("="*80 + "\n")
    
    for query in test_queries:
        result = classifier.classify_intent(query, {})
        print(f"Query: {query}")
        print(f"Intent: {result.primary_intent} (confidence: {result.confidence:.2f})")
        if result.is_multi_intent:
            print(f"Multiple intents detected: {', '.join(result.intents)}")
        print(f"Method: {result.method}")
        print("-" * 80)
    
    # Print statistics
    print("\nClassifier Statistics:")
    print(json.dumps(classifier.get_statistics(), indent=2))
    print("\nCache Statistics:")
    print(json.dumps(classifier.get_cache_stats(), indent=2))
    print("\nPerformance Report:")
    print(json.dumps(classifier.get_performance_report(), indent=2))
