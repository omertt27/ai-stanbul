#!/usr/bin/env python3
"""
Advanced Restaurant System Integration
Integrates enhanced context memory, neural intent classification, multi-intent handling,
and advanced input processing for robust restaurant recommendations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import advanced modules
try:
    from enhanced_context_memory import EnhancedContextMemory, ContextType
    CONTEXT_MEMORY_AVAILABLE = True
    logger.info("✅ Enhanced Context Memory loaded")
except ImportError as e:
    CONTEXT_MEMORY_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced Context Memory not available: {e}")

try:
    from enhanced_neural_intent_classifier import EnhancedNeuralIntentClassifier, IntentPrediction
    NEURAL_INTENT_AVAILABLE = True
    logger.info("✅ Enhanced Neural Intent Classifier loaded")
except ImportError as e:
    NEURAL_INTENT_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced Neural Intent Classifier not available: {e}")

try:
    from multi_intent_query_handler import MultiIntentQueryHandler, IntentType
    MULTI_INTENT_AVAILABLE = True
    logger.info("✅ Multi-Intent Query Handler loaded")
except ImportError as e:
    MULTI_INTENT_AVAILABLE = False
    logger.warning(f"⚠️ Multi-Intent Query Handler not available: {e}")

try:
    from backend.enhanced_input_processor import EnhancedInputProcessor
    INPUT_PROCESSOR_AVAILABLE = True
    logger.info("✅ Enhanced Input Processor loaded")
except ImportError as e:
    INPUT_PROCESSOR_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced Input Processor not available: {e}")

try:
    from thefuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("⚠️ thefuzz not available for fuzzy matching")


class AdvancedRestaurantSystemIntegration:
    """
    Advanced Restaurant System with:
    - Context-aware memory (short-term, working, long-term)
    - Neural intent classification
    - Multi-intent query handling
    - Advanced typo correction and input processing
    - Conflict detection and ambiguity handling
    """
    
    def __init__(self):
        """Initialize all advanced components"""
        self.logger = logger
        
        # Initialize context memory
        if CONTEXT_MEMORY_AVAILABLE:
            self.context_memory = EnhancedContextMemory()
            self.session_id = self.context_memory.start_new_session()
            self.logger.info(f"✅ Context memory initialized (session: {self.session_id})")
        else:
            self.context_memory = None
            self.session_id = None
        
        # Initialize neural intent classifier
        if NEURAL_INTENT_AVAILABLE:
            try:
                self.intent_classifier = EnhancedNeuralIntentClassifier()
                self.logger.info("✅ Neural intent classifier initialized")
            except Exception as e:
                self.intent_classifier = None
                self.logger.warning(f"⚠️ Could not initialize intent classifier: {e}")
        else:
            self.intent_classifier = None
        
        # Initialize multi-intent handler
        if MULTI_INTENT_AVAILABLE:
            try:
                self.multi_intent_handler = MultiIntentQueryHandler()
                self.logger.info("✅ Multi-intent handler initialized")
            except Exception as e:
                self.multi_intent_handler = None
                self.logger.warning(f"⚠️ Could not initialize multi-intent handler: {e}")
        else:
            self.multi_intent_handler = None
        
        # Initialize input processor
        if INPUT_PROCESSOR_AVAILABLE:
            self.input_processor = EnhancedInputProcessor()
            self.logger.info("✅ Enhanced input processor initialized")
        else:
            self.input_processor = None
        
        # Basic typo dictionary (fallback)
        self.typo_dict = {
            'resturant': 'restaurant',
            'restarant': 'restaurant',
            'restaurent': 'restaurant',
            'restuarant': 'restaurant',
            'resteraunt': 'restaurant',
            'reastaurant': 'restaurant',
            'restrant': 'restaurant',
            'sultanamet': 'sultanahmet',
            'sultanhmet': 'sultanahmet',
            'kadiköy': 'kadikoy',
            'beyoğlu': 'beyoglu',
            'taksm': 'taksim',
            'takism': 'taksim',
            'chep': 'cheap',
            'cheep': 'cheap',
            'expensve': 'expensive',
            'luxry': 'luxury',
            'vegaterian': 'vegetarian',
            'vegitarian': 'vegetarian',
            'tradtional': 'traditional',
            'tradisional': 'traditional',
            'seefood': 'seafood',
            'see food': 'seafood',
            'turksh': 'turkish',
            'turkisch': 'turkish'
        }
        
        # Cuisine keywords with variations
        self.cuisine_keywords = {
            'turkish': ['turkish', 'turkish cuisine', 'ottoman', 'anatolian', 'turksh', 'turkisch'],
            'seafood': ['seafood', 'fish', 'sea food', 'seefood'],
            'kebab': ['kebab', 'kebap', 'kabab', 'kebob'],
            'vegetarian': ['vegetarian', 'vegan', 'veggie', 'vegaterian', 'vegitarian'],
            'italian': ['italian', 'pizza', 'pasta', 'itallian'],
            'asian': ['asian', 'chinese', 'japanese', 'sushi', 'thai', 'asian fusion'],
            'mediterranean': ['mediterranean', 'greek', 'lebanese', 'mediteranean'],
            'international': ['international', 'fusion', 'world cuisine'],
            'cafe': ['cafe', 'coffee', 'brunch', 'breakfast']
        }
        
        # Price level keywords
        self.price_keywords = {
            'budget': ['cheap', 'budget', 'affordable', 'inexpensive', 'economical', 'chep', 'cheep'],
            'moderate': ['moderate', 'mid-range', 'reasonable', 'average'],
            'expensive': ['expensive', 'upscale', 'fine dining', 'luxury', 'expensve', 'luxry', 'high-end']
        }
        
        # Atmosphere keywords
        self.atmosphere_keywords = {
            'romantic': ['romantic', 'date', 'couples', 'intimate'],
            'family': ['family', 'kids', 'children', 'family-friendly'],
            'casual': ['casual', 'relaxed', 'informal', 'laid-back'],
            'formal': ['formal', 'elegant', 'sophisticated', 'upscale'],
            'rooftop': ['rooftop', 'terrace', 'outdoor', 'view'],
            'traditional': ['traditional', 'authentic', 'local', 'tradtional', 'tradisional']
        }
        
        self.logger.info("🎉 Advanced Restaurant System Integration ready")
    
    def process_restaurant_query(self, query: str, location_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process restaurant query with all advanced features:
        1. Input enhancement and typo correction
        2. Context-aware intent classification
        3. Multi-intent detection
        4. Conflict and ambiguity detection
        5. Context memory integration
        
        Args:
            query: User's restaurant query
            location_context: Optional GPS and location information
            
        Returns:
            Dict with processed query, detected intents, conflicts, and recommendations
        """
        result = {
            'original_query': query,
            'processed_query': query,
            'typo_corrections': [],
            'detected_intents': [],
            'conflicts': [],
            'ambiguities': [],
            'recommendations': [],
            'context_used': [],
            'needs_clarification': False,
            'clarification_prompt': None
        }
        
        # Step 1: Enhanced input processing and typo correction
        if self.input_processor:
            try:
                enhanced_context = self.input_processor.enhance_query_context(query)
                result['processed_query'] = enhanced_context.get('corrected_query', query)
                result['detected_locations'] = enhanced_context.get('detected_locations', [])
                result['detected_landmarks'] = enhanced_context.get('detected_landmarks', [])
                result['query_type'] = enhanced_context.get('query_type', 'restaurant')
                
                if enhanced_context.get('enhancement_applied'):
                    result['typo_corrections'].append({
                        'original': query,
                        'corrected': enhanced_context.get('corrected_query')
                    })
                    
                self.logger.info(f"✅ Input processing: {query} -> {result['processed_query']}")
            except Exception as e:
                self.logger.error(f"Input processing error: {e}")
                # Fallback to basic typo correction
                result['processed_query'] = self._apply_basic_typo_correction(query)
        else:
            # Fallback to basic typo correction
            result['processed_query'] = self._apply_basic_typo_correction(query)
        
        # Step 2: Get relevant context from memory
        if self.context_memory:
            try:
                relevant_context = self.context_memory.get_relevant_context(
                    result['processed_query'],
                    intent='restaurant',
                    max_items=5
                )
                result['context_used'] = [
                    {
                        'type': ctx.type.value,
                        'content': ctx.content,
                        'confidence': ctx.confidence
                    }
                    for ctx in relevant_context
                ]
                self.logger.info(f"✅ Retrieved {len(relevant_context)} context items")
            except Exception as e:
                self.logger.error(f"Context retrieval error: {e}")
        
        # Step 3: Neural intent classification
        if self.intent_classifier:
            try:
                intent_prediction = self.intent_classifier.classify_intent(
                    result['processed_query'],
                    context=result.get('context_used', [])
                )
                if intent_prediction:
                    result['detected_intents'].append({
                        'intent': intent_prediction.intent,
                        'confidence': intent_prediction.confidence,
                        'sub_intents': intent_prediction.sub_intents
                    })
                    self.logger.info(f"✅ Intent classified: {intent_prediction.intent} ({intent_prediction.confidence:.2f})")
            except Exception as e:
                self.logger.error(f"Intent classification error: {e}")
        
        # Step 4: Multi-intent detection
        if self.multi_intent_handler:
            try:
                multi_intent_result = self.multi_intent_handler.process_query(result['processed_query'])
                if multi_intent_result and hasattr(multi_intent_result, 'detected_intents'):
                    for intent in multi_intent_result.detected_intents:
                        result['detected_intents'].append({
                            'intent': intent.intent_type.value if hasattr(intent.intent_type, 'value') else str(intent.intent_type),
                            'confidence': intent.confidence,
                            'parameters': intent.parameters
                        })
                    self.logger.info(f"✅ Multi-intent detection: {len(multi_intent_result.detected_intents)} intents found")
            except Exception as e:
                self.logger.error(f"Multi-intent detection error: {e}")
        
        # Step 5: Extract restaurant requirements
        requirements = self._extract_restaurant_requirements(result['processed_query'])
        result['requirements'] = requirements
        
        # Step 6: Detect conflicts
        conflicts = self._detect_query_conflicts(requirements)
        if conflicts:
            result['conflicts'] = conflicts
            result['needs_clarification'] = True
            result['clarification_prompt'] = self._generate_conflict_clarification(conflicts)
            self.logger.warning(f"⚠️ Conflicts detected: {conflicts}")
        
        # Step 7: Check for ambiguity
        if self._is_ambiguous_restaurant_query(requirements):
            result['ambiguities'] = self._identify_ambiguities(requirements)
            result['needs_clarification'] = True
            if not result['clarification_prompt']:  # Don't override conflict clarification
                result['clarification_prompt'] = self._generate_ambiguity_clarification(requirements)
            self.logger.warning(f"⚠️ Ambiguous query detected")
        
        # Step 8: Add to context memory
        if self.context_memory and not result['needs_clarification']:
            try:
                # Add restaurant search context
                self.context_memory.add_context_item(
                    ContextType.INTENT,
                    {
                        'intent': 'restaurant_search',
                        'query': result['processed_query'],
                        'requirements': requirements
                    },
                    confidence=0.9
                )
                
                # Add location context if available
                if location_context:
                    self.context_memory.update_location_context(location_context, confidence=0.9)
                
                self.logger.info("✅ Context added to memory")
            except Exception as e:
                self.logger.error(f"Context storage error: {e}")
        
        return result
    
    def _apply_basic_typo_correction(self, query: str) -> str:
        """Apply basic dictionary-based typo correction"""
        words = query.lower().split()
        corrected_words = []
        
        for word in words:
            # Check dictionary
            if word in self.typo_dict:
                corrected_words.append(self.typo_dict[word])
            # Try fuzzy matching if available
            elif FUZZY_AVAILABLE and len(word) > 3:
                best_match = None
                best_score = 0
                for correct_word in self.typo_dict.values():
                    score = fuzz.ratio(word, correct_word)
                    if score > best_score and score > 85:
                        best_score = score
                        best_match = correct_word
                corrected_words.append(best_match if best_match else word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _extract_restaurant_requirements(self, query: str) -> Dict[str, Any]:
        """Extract detailed restaurant requirements from query"""
        requirements = {
            'cuisine': [],
            'price_level': None,
            'atmosphere': [],
            'location': [],
            'dietary': [],
            'meal_type': None,
            'view': None
        }
        
        query_lower = query.lower()
        
        # Extract cuisine types
        for cuisine, keywords in self.cuisine_keywords.items():
            if any(kw in query_lower for kw in keywords):
                requirements['cuisine'].append(cuisine)
        
        # Extract price level
        for level, keywords in self.price_keywords.items():
            if any(kw in query_lower for kw in keywords):
                requirements['price_level'] = level
                break
        
        # Extract atmosphere
        for atmos, keywords in self.atmosphere_keywords.items():
            if any(kw in query_lower for kw in keywords):
                requirements['atmosphere'].append(atmos)
        
        # Extract meal type
        if any(word in query_lower for word in ['breakfast', 'brunch']):
            requirements['meal_type'] = 'breakfast'
        elif any(word in query_lower for word in ['lunch']):
            requirements['meal_type'] = 'lunch'
        elif any(word in query_lower for word in ['dinner']):
            requirements['meal_type'] = 'dinner'
        
        # Extract view preference
        if any(word in query_lower for word in ['bosphorus', 'sea view', 'water view', 'view']):
            requirements['view'] = 'bosphorus'
        
        return requirements
    
    def _detect_query_conflicts(self, requirements: Dict[str, Any]) -> List[Dict[str, str]]:
        """Detect conflicting requirements in the query"""
        conflicts = []
        
        # Price conflicts
        if requirements.get('price_level') == 'budget':
            if 'luxury' in requirements.get('atmosphere', []) or 'formal' in requirements.get('atmosphere', []):
                conflicts.append({
                    'type': 'price_atmosphere',
                    'conflict': 'cheap/budget vs luxury/formal atmosphere',
                    'message': 'You requested both budget-friendly prices and luxury atmosphere. Which is more important?'
                })
        
        if requirements.get('price_level') == 'expensive':
            if 'casual' in requirements.get('atmosphere', []):
                conflicts.append({
                    'type': 'price_atmosphere',
                    'conflict': 'expensive vs casual atmosphere',
                    'message': 'You requested expensive restaurants but casual atmosphere. Would you like upscale-casual or formal dining?'
                })
        
        # Atmosphere conflicts
        atmospheres = requirements.get('atmosphere', [])
        if 'romantic' in atmospheres and 'family' in atmospheres:
            conflicts.append({
                'type': 'atmosphere',
                'conflict': 'romantic vs family-friendly',
                'message': 'You requested both romantic and family-friendly atmosphere. Which is your priority?'
            })
        
        if 'formal' in atmospheres and 'casual' in atmospheres:
            conflicts.append({
                'type': 'atmosphere',
                'conflict': 'formal vs casual',
                'message': 'You requested both formal and casual atmosphere. Please clarify your preference.'
            })
        
        return conflicts
    
    def _is_ambiguous_restaurant_query(self, requirements: Dict[str, Any]) -> bool:
        """Check if query is too ambiguous to provide good recommendations"""
        # Check if any meaningful criteria were extracted
        has_cuisine = len(requirements.get('cuisine', [])) > 0
        has_price = requirements.get('price_level') is not None
        has_atmosphere = len(requirements.get('atmosphere', [])) > 0
        has_location = len(requirements.get('location', [])) > 0
        has_meal_type = requirements.get('meal_type') is not None
        
        # Query is ambiguous if it has fewer than 2 criteria
        criteria_count = sum([has_cuisine, has_price, has_atmosphere, has_location, has_meal_type])
        
        return criteria_count < 2
    
    def _identify_ambiguities(self, requirements: Dict[str, Any]) -> List[str]:
        """Identify specific ambiguities in the query"""
        ambiguities = []
        
        if not requirements.get('cuisine'):
            ambiguities.append('No cuisine type specified')
        
        if not requirements.get('price_level'):
            ambiguities.append('No price range specified')
        
        if not requirements.get('atmosphere'):
            ambiguities.append('No atmosphere preference specified')
        
        if not requirements.get('location'):
            ambiguities.append('No specific location/district specified')
        
        return ambiguities
    
    def _generate_conflict_clarification(self, conflicts: List[Dict[str, str]]) -> str:
        """Generate clarification prompt for conflicts"""
        if not conflicts:
            return None
        
        messages = [conflict['message'] for conflict in conflicts]
        
        return (
            "I noticed some conflicting requirements in your request:\n\n" +
            "\n".join(f"• {msg}" for msg in messages) +
            "\n\nPlease clarify your preferences so I can provide better recommendations."
        )
    
    def _generate_ambiguity_clarification(self, requirements: Dict[str, Any]) -> str:
        """Generate clarification prompt for ambiguous queries"""
        missing_info = []
        
        if not requirements.get('cuisine'):
            missing_info.append("type of cuisine (Turkish, Italian, seafood, etc.)")
        
        if not requirements.get('price_level'):
            missing_info.append("budget preference (budget-friendly, moderate, or upscale)")
        
        if not requirements.get('atmosphere'):
            missing_info.append("atmosphere (casual, romantic, family-friendly, etc.)")
        
        if len(missing_info) == 0:
            return None
        
        return (
            "To provide better restaurant recommendations, could you please specify:\n\n" +
            "\n".join(f"• {info}" for info in missing_info[:3])  # Limit to top 3
        )
    
    def add_conversation_turn(self, user_query: str, ai_response: str, 
                             extracted_entities: Dict[str, List[str]],
                             intent: str, confidence: float = 0.9):
        """Add conversation turn to context memory"""
        if self.context_memory:
            try:
                self.context_memory.add_conversation_turn(
                    user_query=user_query,
                    ai_response=ai_response,
                    extracted_entities=extracted_entities,
                    intent=intent,
                    confidence=confidence
                )
                self.logger.info("✅ Conversation turn added to memory")
            except Exception as e:
                self.logger.error(f"Failed to add conversation turn: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        if self.context_memory:
            try:
                return self.context_memory.get_session_summary()
            except Exception as e:
                self.logger.error(f"Failed to get session summary: {e}")
                return {}
        return {}


# Global instance
_advanced_restaurant_system = None

def get_advanced_restaurant_system() -> AdvancedRestaurantSystemIntegration:
    """Get or create the global advanced restaurant system instance"""
    global _advanced_restaurant_system
    if _advanced_restaurant_system is None:
        _advanced_restaurant_system = AdvancedRestaurantSystemIntegration()
    return _advanced_restaurant_system


if __name__ == "__main__":
    # Test the advanced restaurant system
    print("🧪 Testing Advanced Restaurant System Integration")
    print("=" * 70)
    
    system = AdvancedRestaurantSystemIntegration()
    
    test_queries = [
        "cheap luxury resturant near sultanamet",  # Conflict + typo
        "restaurant",  # Ambiguous
        "romantic restaurant with bosphorus view",  # Good query
        "family-friendly romantic italian restaurant",  # Conflict
        "expensive casual turkish food in beyoglu",  # Slight conflict
        "vegitarian seefood restaurant"  # Typo + potential conflict
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print(f"{'='*70}")
        
        result = system.process_restaurant_query(query)
        
        print(f"\n📝 Processed Query: {result['processed_query']}")
        
        if result['typo_corrections']:
            print(f"\n✏️ Typo Corrections:")
            for correction in result['typo_corrections']:
                print(f"   {correction['original']} → {correction['corrected']}")
        
        if result['detected_intents']:
            print(f"\n🎯 Detected Intents:")
            for intent in result['detected_intents']:
                print(f"   - {intent['intent']} (confidence: {intent['confidence']:.2f})")
        
        if result['requirements']:
            print(f"\n📋 Requirements:")
            for key, value in result['requirements'].items():
                if value:
                    print(f"   - {key}: {value}")
        
        if result['conflicts']:
            print(f"\n⚠️ Conflicts Detected:")
            for conflict in result['conflicts']:
                print(f"   - {conflict['conflict']}")
        
        if result['ambiguities']:
            print(f"\n❓ Ambiguities:")
            for ambiguity in result['ambiguities']:
                print(f"   - {ambiguity}")
        
        if result['needs_clarification']:
            print(f"\n💬 Clarification Needed:")
            print(f"   {result['clarification_prompt']}")
        
        if result['context_used']:
            print(f"\n🧠 Context Used: {len(result['context_used'])} items")
