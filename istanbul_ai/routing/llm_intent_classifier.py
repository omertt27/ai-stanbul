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

Author: Istanbul AI Team
Date: December 2024
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

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
            keyword_classifier: Fallback keyword-based IntentClassifier (optional)
            neural_classifier: NeuralQueryClassifier for better fallback (optional, preferred over keyword)
        """
        self.llm_service = llm_service
        self.keyword_classifier = keyword_classifier
        self.neural_classifier = neural_classifier
        self.use_llm = llm_service is not None
        self.has_neural_fallback = neural_classifier is not None
        self.has_keyword_fallback = keyword_classifier is not None
        
        # Statistics
        self.stats = {
            'llm_used': 0,
            'neural_fallback': 0,
            'keyword_fallback': 0,
            'llm_failures': 0,
            'total_requests': 0
        }
        
        # Initialize LLM if not provided
        if not self.use_llm:
            try:
                from ml_systems.llm_service_wrapper import LLMServiceWrapper
                self.llm_service = LLMServiceWrapper()
                self.use_llm = True
                logger.info("✅ LLM Intent Classifier initialized with auto-loaded LLM service")
            except Exception as e:
                logger.warning(f"⚠️ Failed to auto-load LLM service: {e}")
                self.use_llm = False
        
        if self.use_llm:
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
                result = self._classify_with_llm(message, entities, context)
                self.stats['llm_used'] += 1
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
    
    def _classify_with_llm(
        self,
        message: str,
        entities: Dict,
        context: Optional[Any] = None
    ) -> IntentResult:
        """
        Classify intent using LLM
        
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
        
        # Get LLM response
        llm_response = self.llm_service.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.3  # Lower temperature for more consistent classification
        )
        
        # Parse LLM response
        result = self._parse_llm_response(llm_response, message, entities)
        
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
        
        # Build the prompt - use few-shot examples to guide the model
        # Small LLMs like TinyLlama work better with concrete examples than abstract instructions
        # Include multilingual examples (EN, TR, FR, DE, RU, AR) for better multilingual support
        prompt = f"""Classify user intent for Istanbul travel queries. Supports all languages (English, Turkish, French, German, Russian, Arabic, etc).

Examples:

Q: "What's the weather today?"
A: {{"primary_intent": "weather", "confidence": 0.95, "all_intents": ["weather"]}}

Q: "Bugün hava nasıl?" (Turkish)
A: {{"primary_intent": "weather", "confidence": 0.95, "all_intents": ["weather"]}}

Q: "Where can I eat kebab?"
A: {{"primary_intent": "restaurant", "confidence": 0.95, "all_intents": ["restaurant"]}}

Q: "Où puis-je manger des kebabs?" (French)
A: {{"primary_intent": "restaurant", "confidence": 0.95, "all_intents": ["restaurant"]}}

Q: "How do I get to Taksim?"
A: {{"primary_intent": "transportation", "confidence": 0.95, "all_intents": ["transportation"]}}

Q: "Wie komme ich nach Taksim?" (German)
A: {{"primary_intent": "transportation", "confidence": 0.95, "all_intents": ["transportation"]}}

Q: "Show me Hagia Sophia"
A: {{"primary_intent": "attraction", "confidence": 0.95, "all_intents": ["attraction"]}}

Q: "أرني آيا صوفيا" (Arabic)
A: {{"primary_intent": "attraction", "confidence": 0.95, "all_intents": ["attraction"]}}

Now classify (any language):
Q: "{message}"
A:"""
        
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
