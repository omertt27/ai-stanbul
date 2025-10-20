#!/usr/bin/env python3
"""
Advanced Systems Integration Module
====================================

Integrates all advanced AI systems across ALL Istanbul AI areas:
- Enhanced Context Memory (multi-layered memory management)
- Neural Intent Classification (transformer-based)
- Fuzzy Matching & Typo Correction
- Context-aware filtering
- Advanced multi-intent handling

Areas covered:
1. Restaurant Advising
2. Daily Talks
3. Museum Advising
4. Route Planning
5. Transportation Advice
6. Local Tips
7. Districts Advice
8. Events Advising
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class AdvancedSystemsIntegrator:
    """
    Master integrator for all advanced AI systems across ALL Istanbul AI areas
    """
    
    def __init__(self):
        """Initialize all advanced systems"""
        self.logger = logger
        
        # Initialize advanced systems
        self._init_context_memory()
        self._init_neural_intent_classifier()
        self._init_fuzzy_matching()
        self._init_context_aware_filtering()
        
        logger.info("🚀 Advanced Systems Integration initialized for ALL areas")
    
    def _init_context_memory(self):
        """Initialize Enhanced Context Memory System"""
        try:
            from enhanced_context_memory import EnhancedContextMemory, ContextType
            self.context_memory = EnhancedContextMemory()
            self.context_available = True
            logger.info("✅ Enhanced Context Memory System loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Enhanced Context Memory not available: {e}")
            self.context_memory = None
            self.context_available = False
    
    def _init_neural_intent_classifier(self):
        """Initialize Neural Intent Classifier"""
        try:
            from enhanced_neural_intent_classifier import EnhancedNeuralIntentClassifier
            self.neural_intent_classifier = EnhancedNeuralIntentClassifier()
            self.neural_intent_available = True
            logger.info("✅ Neural Intent Classifier loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Neural Intent Classifier not available: {e}")
            self.neural_intent_classifier = None
            self.neural_intent_available = False
    
    def _init_fuzzy_matching(self):
        """Initialize Fuzzy Matching System"""
        try:
            from thefuzz import fuzz, process
            self.fuzz = fuzz
            self.process = process
            self.fuzzy_available = True
            logger.info("✅ Fuzzy Matching System (thefuzz) loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Fuzzy Matching not available: {e}")
            self.fuzzy_available = False
    
    def _init_context_aware_filtering(self):
        """Initialize Context-Aware Filtering"""
        try:
            import sys
            import os
            parent_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.join(parent_dir, 'backend')
            if backend_dir not in sys.path:
                sys.path.append(backend_dir)
            
            from context_aware_filtering import ContextAwareFilter
            self.context_filter = ContextAwareFilter()
            self.context_filter_available = True
            logger.info("✅ Context-Aware Filtering loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Context-Aware Filtering not available: {e}")
            self.context_filter = None
            self.context_filter_available = False
    
    # ==================== RESTAURANT ADVISING ====================
    
    def enhance_restaurant_query(self, query: str, user_context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance restaurant query with typo correction, context, and intent analysis
        
        Returns:
            Tuple of (corrected_query, enhancement_metadata)
        """
        metadata = {
            'original_query': query,
            'corrections_made': [],
            'context_used': [],
            'intent_confidence': 0.0
        }
        
        # Step 1: Typo correction with fuzzy matching
        corrected_query, typos_fixed = self._correct_typos(
            query, 
            domain='restaurant',
            terms=['restaurant', 'vegetarian', 'seafood', 'halal', 'cuisine', 'traditional', 'modern']
        )
        if typos_fixed:
            metadata['corrections_made'].extend(typos_fixed)
        
        # Step 2: Context-aware enhancement
        if self.context_available and user_context:
            context_enhancements = self._apply_context_memory(corrected_query, user_context, 'restaurant')
            metadata['context_used'] = context_enhancements
        
        # Step 3: Intent analysis
        if self.neural_intent_available:
            intent_result = self._classify_intent(corrected_query, 'restaurant')
            metadata['intent_confidence'] = intent_result.get('confidence', 0.0)
            metadata['detected_intents'] = intent_result.get('intents', [])
        
        return corrected_query, metadata
    
    # ==================== DAILY TALKS ====================
    
    def enhance_daily_talk(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Enhance daily talk with context-aware responses and personality
        
        Returns:
            Enhancement metadata for personalizing responses
        """
        metadata = {
            'sentiment': 'neutral',
            'formality_level': 'casual',
            'conversation_context': [],
            'suggested_tone': 'friendly',
            'cultural_context': []
        }
        
        # Analyze conversation context
        if self.context_available and conversation_history:
            # Get conversation patterns
            recent_topics = self._extract_conversation_topics(conversation_history)
            metadata['conversation_context'] = recent_topics
            
            # Determine appropriate formality
            metadata['formality_level'] = self._detect_formality(message, conversation_history)
        
        # Cultural context detection
        cultural_keywords = ['culture', 'tradition', 'custom', 'etiquette', 'turkish']
        if any(word in message.lower() for word in cultural_keywords):
            metadata['cultural_context'] = self._get_cultural_context(message)
        
        # Sentiment analysis (simple rule-based)
        metadata['sentiment'] = self._detect_sentiment(message)
        
        return metadata
    
    # ==================== MUSEUM ADVISING ====================
    
    def enhance_museum_query(self, query: str, location_context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance museum query with corrections and context
        
        Returns:
            Tuple of (corrected_query, enhancement_metadata)
        """
        metadata = {
            'original_query': query,
            'corrections_made': [],
            'museum_type_detected': [],
            'location_refined': None
        }
        
        # Typo correction for museum names and types
        museum_terms = [
            'museum', 'palace', 'mosque', 'gallery', 'exhibition', 'contemporary',
            'hagia sophia', 'topkapi', 'blue mosque', 'archaeology', 'basilica'
        ]
        corrected_query, typos = self._correct_typos(query, 'museum', museum_terms)
        if typos:
            metadata['corrections_made'] = typos
        
        # Detect museum type preferences
        if 'contemporary' in corrected_query.lower() or 'modern art' in corrected_query.lower():
            metadata['museum_type_detected'].append('contemporary_art')
        if 'palace' in corrected_query.lower() or 'ottoman' in corrected_query.lower():
            metadata['museum_type_detected'].append('palace')
        if 'archaeology' in corrected_query.lower() or 'ancient' in corrected_query.lower():
            metadata['museum_type_detected'].append('archaeological')
        
        # Location refinement
        if location_context:
            metadata['location_refined'] = self._refine_location(
                corrected_query, location_context, 'museum'
            )
        
        return corrected_query, metadata
    
    # ==================== ROUTE PLANNING ====================
    
    def enhance_route_query(self, query: str, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance route planning query with context and preferences
        
        Returns:
            Route enhancement metadata
        """
        metadata = {
            'transport_mode': 'walking',
            'pace_preference': 'moderate',
            'accessibility_needs': [],
            'time_constraints': None,
            'interest_categories': []
        }
        
        # Detect transport mode preference
        if any(word in query.lower() for word in ['walk', 'walking', 'foot']):
            metadata['transport_mode'] = 'walking'
        elif any(word in query.lower() for word in ['metro', 'subway', 'train']):
            metadata['transport_mode'] = 'metro'
        elif any(word in query.lower() for word in ['bus', 'dolmuş']):
            metadata['transport_mode'] = 'bus'
        elif any(word in query.lower() for word in ['ferry', 'boat', 'vapur']):
            metadata['transport_mode'] = 'ferry'
        
        # Detect pace preference
        if any(word in query.lower() for word in ['quick', 'fast', 'rushed', 'hurry']):
            metadata['pace_preference'] = 'fast'
        elif any(word in query.lower() for word in ['relaxed', 'slow', 'leisurely', 'take time']):
            metadata['pace_preference'] = 'relaxed'
        
        # Accessibility needs
        if any(word in query.lower() for word in ['wheelchair', 'accessible', 'disability']):
            metadata['accessibility_needs'].append('wheelchair_accessible')
        if 'elevator' in query.lower() or 'lift' in query.lower():
            metadata['accessibility_needs'].append('elevator_required')
        
        # Time constraints
        time_match = re.search(r'(\d+)\s*(hour|hr|h)', query.lower())
        if time_match:
            metadata['time_constraints'] = int(time_match.group(1))
        
        # Interest categories
        if any(word in query.lower() for word in ['museum', 'art', 'culture', 'history']):
            metadata['interest_categories'].append('cultural')
        if any(word in query.lower() for word in ['food', 'restaurant', 'cafe', 'dining']):
            metadata['interest_categories'].append('culinary')
        if any(word in query.lower() for word in ['shopping', 'bazaar', 'market']):
            metadata['interest_categories'].append('shopping')
        if any(word in query.lower() for word in ['scenic', 'view', 'photo', 'beautiful']):
            metadata['interest_categories'].append('scenic')
        
        return metadata
    
    # ==================== TRANSPORTATION ADVICE ====================
    
    def enhance_transportation_query(self, query: str, location_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance transportation query with context and corrections
        
        Returns:
            Transportation enhancement metadata
        """
        metadata = {
            'origin': None,
            'destination': None,
            'time_preference': 'any',
            'cost_sensitivity': 'moderate',
            'comfort_preference': 'standard'
        }
        
        # Typo correction for location names
        location_terms = [
            'Beyoğlu', 'Kadıköy', 'Taksim', 'Sultanahmet', 'Beşiktaş',
            'Ortaköy', 'Eminönü', 'Galata', 'Üsküdar', 'Şişli'
        ]
        corrected_query, _ = self._correct_typos(query, 'location', location_terms)
        
        # Extract origin and destination
        if ' to ' in corrected_query.lower() or ' from ' in corrected_query.lower():
            # Try to extract "from X to Y" pattern
            from_to_match = re.search(r'from\s+([a-zA-ZğüşıöçĞÜŞİÖÇ\s]+)\s+to\s+([a-zA-ZğüşıöçĞÜŞİÖÇ\s]+)', corrected_query, re.IGNORECASE)
            if from_to_match:
                metadata['origin'] = from_to_match.group(1).strip()
                metadata['destination'] = from_to_match.group(2).strip()
        
        # Time preference
        if any(word in query.lower() for word in ['fastest', 'quickest', 'fast', 'quick']):
            metadata['time_preference'] = 'fastest'
        elif any(word in query.lower() for word in ['scenic', 'beautiful', 'sightseeing']):
            metadata['time_preference'] = 'scenic'
        
        # Cost sensitivity
        if any(word in query.lower() for word in ['cheap', 'cheapest', 'budget', 'affordable']):
            metadata['cost_sensitivity'] = 'budget'
        elif any(word in query.lower() for word in ['taxi', 'uber', 'comfortable', 'private']):
            metadata['cost_sensitivity'] = 'premium'
        
        # Comfort preference
        if any(word in query.lower() for word in ['comfortable', 'air condition', 'seated']):
            metadata['comfort_preference'] = 'high'
        elif any(word in query.lower() for word in ['crowded', 'packed', 'busy']):
            metadata['comfort_preference'] = 'any'
        
        return metadata
    
    # ==================== LOCAL TIPS ====================
    
    def enhance_local_tips_query(self, query: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance local tips query with user context
        
        Returns:
            Local tips enhancement metadata
        """
        metadata = {
            'tip_category': [],
            'experience_level': 'tourist',
            'budget_category': 'moderate',
            'safety_concerns': False,
            'cultural_interest': False
        }
        
        # Detect tip categories
        if any(word in query.lower() for word in ['hidden', 'secret', 'local', 'authentic']):
            metadata['tip_category'].append('hidden_gems')
        if any(word in query.lower() for word in ['avoid', 'scam', 'tourist trap', 'overpriced']):
            metadata['tip_category'].append('warnings')
        if any(word in query.lower() for word in ['save money', 'discount', 'free', 'cheap']):
            metadata['tip_category'].append('budget_tips')
        if any(word in query.lower() for word in ['etiquette', 'custom', 'culture', 'polite']):
            metadata['tip_category'].append('cultural_tips')
        if any(word in query.lower() for word in ['safe', 'safety', 'dangerous', 'secure']):
            metadata['tip_category'].append('safety')
            metadata['safety_concerns'] = True
        
        # Experience level
        if any(word in query.lower() for word in ['first time', 'never been', 'new to']):
            metadata['experience_level'] = 'first_timer'
        elif any(word in query.lower() for word in ['local', 'live here', 'resident']):
            metadata['experience_level'] = 'local'
        elif any(word in query.lower() for word in ['been before', 'visited', 'return']):
            metadata['experience_level'] = 'returning'
        
        # Budget category
        if any(word in query.lower() for word in ['luxury', 'expensive', 'premium', 'high-end']):
            metadata['budget_category'] = 'luxury'
        elif any(word in query.lower() for word in ['budget', 'cheap', 'backpacker', 'affordable']):
            metadata['budget_category'] = 'budget'
        
        # Cultural interest
        if any(word in query.lower() for word in ['culture', 'tradition', 'history', 'authentic']):
            metadata['cultural_interest'] = True
        
        return metadata
    
    # ==================== DISTRICTS ADVICE ====================
    
    def enhance_district_query(self, query: str, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance district advice query
        
        Returns:
            District advice enhancement metadata
        """
        metadata = {
            'districts_mentioned': [],
            'comparison_request': False,
            'lifestyle_preference': [],
            'accommodation_search': False,
            'nightlife_interest': False
        }
        
        # Detect districts mentioned
        districts = {
            'beyoğlu': 'Beyoğlu', 'beyoglu': 'Beyoğlu',
            'kadıköy': 'Kadıköy', 'kadikoy': 'Kadıköy',
            'sultanahmet': 'Sultanahmet',
            'beşiktaş': 'Beşiktaş', 'besiktas': 'Beşiktaş',
            'ortaköy': 'Ortaköy', 'ortakoy': 'Ortaköy',
            'şişli': 'Şişli', 'sisli': 'Şişli',
            'nişantaşı': 'Nişantaşı', 'nisantasi': 'Nişantaşı',
            'üsküdar': 'Üsküdar', 'uskudar': 'Üsküdar',
            'fatih': 'Fatih',
            'eminönü': 'Eminönü', 'eminonu': 'Eminönü'
        }
        
        query_lower = query.lower()
        for key, value in districts.items():
            if key in query_lower and value not in metadata['districts_mentioned']:
                metadata['districts_mentioned'].append(value)
        
        # Comparison request
        if any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'better']):
            metadata['comparison_request'] = True
        
        # Lifestyle preferences
        if any(word in query_lower for word in ['quiet', 'peaceful', 'residential', 'calm']):
            metadata['lifestyle_preference'].append('quiet')
        if any(word in query_lower for word in ['lively', 'busy', 'vibrant', 'active']):
            metadata['lifestyle_preference'].append('lively')
        if any(word in query_lower for word in ['hipster', 'trendy', 'modern', 'cool']):
            metadata['lifestyle_preference'].append('trendy')
        if any(word in query_lower for word in ['traditional', 'historic', 'old', 'authentic']):
            metadata['lifestyle_preference'].append('traditional')
        if any(word in query_lower for word in ['expat', 'international', 'foreigner']):
            metadata['lifestyle_preference'].append('expat_friendly')
        
        # Accommodation search
        if any(word in query_lower for word in ['stay', 'hotel', 'accommodation', 'airbnb', 'where to live']):
            metadata['accommodation_search'] = True
        
        # Nightlife interest
        if any(word in query_lower for word in ['nightlife', 'bar', 'club', 'party', 'night out']):
            metadata['nightlife_interest'] = True
        
        return metadata
    
    # ==================== EVENTS ADVISING ====================
    
    def enhance_events_query(self, query: str, date_context: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Enhance events query
        
        Returns:
            Events enhancement metadata
        """
        metadata = {
            'event_types': [],
            'date_preference': None,
            'venue_preference': [],
            'price_range': 'any',
            'cultural_level': 'all'
        }
        
        query_lower = query.lower()
        
        # Event types
        if any(word in query_lower for word in ['concert', 'music', 'live music', 'performance']):
            metadata['event_types'].append('concert')
        if any(word in query_lower for word in ['theater', 'theatre', 'play', 'drama']):
            metadata['event_types'].append('theater')
        if any(word in query_lower for word in ['exhibition', 'art show', 'gallery']):
            metadata['event_types'].append('exhibition')
        if any(word in query_lower for word in ['festival', 'celebration', 'street']):
            metadata['event_types'].append('festival')
        if any(word in query_lower for word in ['workshop', 'class', 'seminar', 'talk']):
            metadata['event_types'].append('workshop')
        if any(word in query_lower for word in ['film', 'movie', 'cinema', 'screening']):
            metadata['event_types'].append('film')
        
        # Date preference
        if 'tonight' in query_lower or 'today' in query_lower:
            metadata['date_preference'] = 'today'
        elif 'tomorrow' in query_lower:
            metadata['date_preference'] = 'tomorrow'
        elif 'weekend' in query_lower or 'this weekend' in query_lower:
            metadata['date_preference'] = 'weekend'
        elif 'this week' in query_lower:
            metadata['date_preference'] = 'this_week'
        elif 'this month' in query_lower:
            metadata['date_preference'] = 'this_month'
        
        # Venue preference
        if any(word in query_lower for word in ['outdoor', 'outside', 'open air']):
            metadata['venue_preference'].append('outdoor')
        if any(word in query_lower for word in ['indoor', 'inside']):
            metadata['venue_preference'].append('indoor')
        
        # Price range
        if any(word in query_lower for word in ['free', 'no cost', 'complimentary']):
            metadata['price_range'] = 'free'
        elif any(word in query_lower for word in ['cheap', 'affordable', 'budget']):
            metadata['price_range'] = 'budget'
        elif any(word in query_lower for word in ['expensive', 'premium', 'vip']):
            metadata['price_range'] = 'premium'
        
        # Cultural level
        if any(word in query_lower for word in ['family', 'kids', 'children', 'child-friendly']):
            metadata['cultural_level'] = 'family_friendly'
        elif any(word in query_lower for word in ['classical', 'opera', 'symphony', 'ballet']):
            metadata['cultural_level'] = 'classical'
        elif any(word in query_lower for word in ['contemporary', 'modern', 'experimental']):
            metadata['cultural_level'] = 'contemporary'
        
        return metadata
    
    # ==================== HELPER METHODS ====================
    
    def _correct_typos(self, query: str, domain: str, terms: List[str]) -> Tuple[str, List[str]]:
        """
        Correct typos using fuzzy matching
        
        Args:
            query: Original query
            domain: Domain type (restaurant, museum, location, etc.)
            terms: List of correct terms for the domain
            
        Returns:
            Tuple of (corrected_query, list_of_corrections_made)
        """
        if not self.fuzzy_available:
            return query, []
        
        corrections_made = []
        words = query.split()
        corrected_words = []
        
        for word in words:
            word_clean = word.lower().strip('.,!?')
            
            # Skip short words
            if len(word_clean) < 4:
                corrected_words.append(word)
                continue
            
            # Try fuzzy matching
            best_match = self.process.extractOne(word_clean, terms, scorer=self.fuzz.ratio)
            if best_match and best_match[1] >= 80 and best_match[1] < 100:
                corrections_made.append(f"{word} → {best_match[0]}")
                corrected_words.append(best_match[0])
                self.logger.debug(f"Fuzzy corrected '{word}' to '{best_match[0]}' (score: {best_match[1]})")
            else:
                corrected_words.append(word)
        
        corrected_query = ' '.join(corrected_words)
        return corrected_query, corrections_made
    
    def _apply_context_memory(self, query: str, user_context: Dict[str, Any], domain: str) -> List[str]:
        """Apply context memory enhancements"""
        if not self.context_available:
            return []
        
        context_used = []
        
        # Add relevant context from user history
        if 'previous_locations' in user_context:
            context_used.append(f"Previous locations: {', '.join(user_context['previous_locations'][:3])}")
        
        if 'preferences' in user_context:
            context_used.append(f"User preferences considered")
        
        return context_used
    
    def _classify_intent(self, query: str, domain: str) -> Dict[str, Any]:
        """Classify intent using neural classifier"""
        if not self.neural_intent_available:
            return {'confidence': 0.0, 'intents': []}
        
        # Use neural classifier
        # For now, return mock result (would use actual classifier)
        return {
            'confidence': 0.85,
            'intents': [domain]
        }
    
    def _extract_conversation_topics(self, conversation_history: List[Dict]) -> List[str]:
        """Extract topics from conversation history"""
        topics = set()
        
        for turn in conversation_history[-5:]:  # Last 5 turns
            message = turn.get('message', '').lower()
            
            if any(word in message for word in ['restaurant', 'food', 'eat']):
                topics.add('dining')
            if any(word in message for word in ['museum', 'art', 'culture']):
                topics.add('culture')
            if any(word in message for word in ['transport', 'metro', 'bus']):
                topics.add('transportation')
            if any(word in message for word in ['hotel', 'stay', 'accommodation']):
                topics.add('accommodation')
        
        return list(topics)
    
    def _detect_formality(self, message: str, history: List[Dict]) -> str:
        """Detect appropriate formality level"""
        # Simple heuristic: if user uses formal language, respond formally
        formal_indicators = ['please', 'could you', 'would you', 'thank you']
        casual_indicators = ['hey', 'yeah', 'cool', 'thanks', 'hi']
        
        message_lower = message.lower()
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in message_lower)
        casual_count = sum(1 for indicator in casual_indicators if indicator in message_lower)
        
        if formal_count > casual_count:
            return 'formal'
        elif casual_count > formal_count:
            return 'casual'
        else:
            return 'moderate'
    
    def _get_cultural_context(self, message: str) -> List[str]:
        """Get cultural context relevant to the query"""
        cultural_topics = []
        
        message_lower = message.lower()
        
        if 'mosque' in message_lower or 'prayer' in message_lower:
            cultural_topics.append('religious_etiquette')
        if 'tea' in message_lower or 'çay' in message_lower:
            cultural_topics.append('tea_culture')
        if 'bargain' in message_lower or 'haggle' in message_lower:
            cultural_topics.append('bargaining_culture')
        if 'family' in message_lower:
            cultural_topics.append('family_values')
        
        return cultural_topics
    
    def _detect_sentiment(self, message: str) -> str:
        """Simple sentiment detection"""
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'love', 'good', 'nice', 'beautiful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'poor', 'disappointing', 'worst']
        
        message_lower = message.lower()
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _refine_location(self, query: str, location_context: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Refine location context"""
        refined = {
            'primary_location': location_context.get('detected_location'),
            'nearby_districts': [],
            'coordinates': location_context.get('coordinates')
        }
        
        # Add nearby districts based on domain
        if domain == 'museum' and refined['primary_location']:
            # Museums concentrated in certain areas
            museum_districts = {
                'Sultanahmet': ['Fatih', 'Eminönü'],
                'Beyoğlu': ['Taksim', 'Galata', 'Karaköy'],
                'Beşiktaş': ['Ortaköy', 'Kabataş']
            }
            refined['nearby_districts'] = museum_districts.get(refined['primary_location'], [])
        
        return refined
    
    def get_system_status(self) -> Dict[str, bool]:
        """Get status of all advanced systems"""
        return {
            'enhanced_context_memory': self.context_available,
            'neural_intent_classifier': self.neural_intent_available,
            'fuzzy_matching': self.fuzzy_available,
            'context_aware_filtering': self.context_filter_available
        }


# Global instance for easy access
_advanced_integrator = None

def get_advanced_integrator() -> AdvancedSystemsIntegrator:
    """Get or create the global advanced integrator instance"""
    global _advanced_integrator
    if _advanced_integrator is None:
        _advanced_integrator = AdvancedSystemsIntegrator()
    return _advanced_integrator
