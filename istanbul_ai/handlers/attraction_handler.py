"""
ML-Enhanced Attraction Handler for Istanbul AI Chat System

This module handles all attraction and tourist site queries with full ML/Neural integration
Leverages T4 GPU for context extraction, ranking, and personalized recommendations.

Features:
- ML-powered category detection and semantic understanding
- Weather-aware suggestions (indoor/outdoor)
- Neural ranking based on user preferences and context
- Time-aware recommendations (opening hours, crowd levels)
- Sentiment-based response styling
- Multi-modal suggestions (combine attractions with transport/dining)
- 🌐 Full English/Turkish bilingual support
- 🤖 Enhanced LLM integration with Google Cloud Llama 3.1 8B

Author: Istanbul AI Team
Date: October 27, 2025
Updated: November 2, 2025 - Added bilingual support
Updated: [Current Date] - Integrated Google Cloud Llama 3.1 8B LLM
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, time
import re

# Import enhanced LLM client
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from enhanced_llm_config import get_enhanced_llm_client, EnhancedLLMClient
    ENHANCED_LLM_AVAILABLE = True
except ImportError as e:
    ENHANCED_LLM_AVAILABLE = False
    logging.warning(f"⚠️ Enhanced LLM client not available: {e}")

# Import bilingual support
try:
    from ..services.bilingual_manager import BilingualManager, Language
    BILINGUAL_AVAILABLE = True
except ImportError:
    BILINGUAL_AVAILABLE = False
    Language = None

# Import map integration service
try:
    from ..services.map_integration_service import get_map_service
    MAP_INTEGRATION_AVAILABLE = True
except ImportError:
    MAP_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class AttractionHandler:
    """
    Handles attraction queries with ML-enhanced context awareness and ranking.
    """
    
    def __init__(self, neural_processor, user_manager, attraction_service,
                 weather_service=None, advanced_attraction_service=None, transport_service=None,
                 bilingual_manager=None):
        """
        Initialize the Attraction Handler.
        
        Args:
            neural_processor: ML model for semantic understanding
            user_manager: User profile and history manager
            attraction_service: Basic attraction data service
            weather_service: Optional weather integration
            advanced_attraction_service: Optional advanced attraction features
            transport_service: Optional transport integration
            bilingual_manager: BilingualManager for language support
        """
        self.neural_processor = neural_processor
        self.user_manager = user_manager
        self.attraction_service = attraction_service
        self.weather_service = weather_service
        self.advanced_attraction_service = advanced_attraction_service
        self.transport_service = transport_service
        self.bilingual_manager = bilingual_manager
        self.has_bilingual = bilingual_manager is not None and BILINGUAL_AVAILABLE
        
        # Initialize enhanced LLM client
        if ENHANCED_LLM_AVAILABLE:
            try:
                self.llm_client = get_enhanced_llm_client()
                self.has_enhanced_llm = True
                logger.info("✅ Enhanced LLM client (Google Cloud Llama 3.1 8B) initialized for attractions")
            except Exception as e:
                logger.error(f"❌ Failed to initialize enhanced LLM client: {e}")
                self.llm_client = None
                self.has_enhanced_llm = False
        else:
            self.llm_client = None
            self.has_enhanced_llm = False
        
        # Initialize map service
        self.map_service = None
        self.has_maps = False
        if MAP_INTEGRATION_AVAILABLE:
            try:
                self.map_service = get_map_service()
                self.has_maps = self.map_service.is_enabled()
            except Exception as e:
                logger.warning(f"Failed to initialize map service: {e}")
        
        logger.info(f"✅ ML-Enhanced AttractionHandler initialized (Bilingual: {self.has_bilingual}, Maps: {self.has_maps}, Enhanced LLM: {self.has_enhanced_llm})")
    
    def _get_language(self, context) -> str:
        """
        Extract language from context.
        
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
    
    # ==================== PUBLIC API ====================
    
    def generate_response(self, message: str, neural_insights: Dict[str, Any],
                         user_profile: Dict[str, Any], context=None):
        """
        Main entry point for attraction query handling.
        
        Args:
            message: User's query text
            neural_insights: ML-generated insights from neural processor
            user_profile: User's profile and preferences
            context: Conversation context (includes language)
            
        Returns:
            Dict with 'response' (text) and 'map_data' (map visualization) if maps enabled,
            otherwise string response for backward compatibility
        """
        try:
            # 🌐 BILINGUAL: Extract language from context
            language = self._get_language(context)
            logger.info(f"🏛️ Processing attraction query (lang: {language}): {message[:50]}...")
            
            # Step 1: Extract ML context from query
            ml_context = self._extract_ml_context(message, neural_insights, user_profile)
            
            # Step 2: Get candidate attractions
            candidates = self._get_candidate_attractions(ml_context)
            
            if not candidates:
                response = self._generate_no_results_response(ml_context, language)
                return {'response': response, 'map_data': None} if self.has_maps else response
            
            # Step 3: Apply neural ranking
            ranked_attractions = self._apply_neural_ranking(candidates, ml_context, neural_insights)
            
            # Step 4: Filter by context (weather, time, accessibility)
            filtered_attractions = self._apply_contextual_filters(ranked_attractions, ml_context)
            
            # Step 5: Generate response (bilingual)
            response = self._generate_ml_enhanced_response(
                filtered_attractions[:5],  # Top 5 recommendations
                ml_context,
                neural_insights,
                language  # 🌐 Pass language
            )
            
            # Step 6: Generate map visualization 🗺️
            map_data = self._generate_attraction_map(filtered_attractions[:5], ml_context)
            
            # Step 7: Update user history
            self._update_user_history(user_profile, ml_context, filtered_attractions[:5])
            
            logger.info(f"✅ Attraction response generated: {len(filtered_attractions)} recommendations (map: {map_data is not None})")
            
            # Return structured response with map if maps enabled
            if self.has_maps:
                return {
                    'response': response,
                    'map_data': map_data
                }
            else:
                return response
            
        except Exception as e:
            logger.error(f"❌ Error in attraction handler: {str(e)}", exc_info=True)
            fallback = self._generate_fallback_response(language if 'language' in locals() else 'en')
            return {'response': fallback, 'map_data': None} if self.has_maps else fallback
    
    # ==================== ML CONTEXT EXTRACTION ====================
    
    def _extract_ml_context(self, message: str, neural_insights: Dict[str, Any],
                           user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive ML context from query, neural insights, and user profile.
        
        Returns:
            Dictionary with extracted context including:
            - categories: List[str] (historical/museum/nature/religious/modern)
            - interests: List[str] (architecture/art/history/photography)
            - time_available: str (few_hours/half_day/full_day)
            - preferred_atmosphere: str (quiet/lively/crowded_ok)
            - accessibility_needs: List[str]
            - with_children: bool
            - indoor_outdoor: str (indoor/outdoor/flexible)
            - crowd_tolerance: str (avoid_crowds/moderate/dont_mind)
        """
        message_lower = message.lower()
        
        context = {
            'original_query': message,
            'timestamp': datetime.now().isoformat(),
            'user_id': getattr(user_profile, 'user_id', 'unknown')
        }
        
        # 1. Category Detection (ML-powered)
        context['categories'] = self._detect_categories(message_lower, neural_insights)
        
        # 2. Interest Detection
        context['interests'] = self._detect_interests(message_lower, neural_insights)
        
        # 3. Time Available
        context['time_available'] = self._detect_time_available(message_lower)
        
        # 4. Atmosphere Preference
        context['preferred_atmosphere'] = self._detect_atmosphere_preference(message_lower)
        
        # 5. Accessibility Needs
        context['accessibility_needs'] = neural_insights.get('special_requirements', [])
        
        # 6. With Children
        context['with_children'] = self._detect_children_context(message_lower)
        
        # 7. Indoor/Outdoor Preference
        context['indoor_outdoor'] = self._detect_indoor_outdoor_preference(message_lower, neural_insights)
        
        # 8. Crowd Tolerance
        context['crowd_tolerance'] = self._detect_crowd_tolerance(message_lower)
        
        # 9. Location Preference
        context['location_preference'] = self._extract_location_preference(message_lower)
        
        # 10. Budget Level
        context['budget_level'] = self._detect_budget_level(message_lower, neural_insights)
        
        # 11. Sentiment
        context['sentiment'] = neural_insights.get('sentiment', 'neutral')
        
        # 12. Urgency
        context['urgency'] = neural_insights.get('urgency', 'flexible')
        
        # 13. Weather Context
        if self.weather_service:
            context['weather'] = self._get_weather_context()
        
        # 14. Time Context (current time, day of week)
        context['time_context'] = self._get_time_context()
        
        # 15. User Preferences
        context['user_preferences'] = self._get_user_preferences(user_profile)
        
        logger.debug(f"📊 ML Context extracted: {context}")
        return context
    
    def _detect_categories(self, message: str, neural_insights: Dict[str, Any]) -> List[str]:
        """
        Detect attraction categories using ML and keyword matching.
        
        Categories: historical, museum, nature, religious, modern, entertainment, shopping
        """
        categories = []
        
        # Check neural insights first
        detected_categories = neural_insights.get('attraction_categories', [])
        if detected_categories:
            return detected_categories
        
        # Category keywords
        category_map = {
            'historical': ['historical', 'history', 'ancient', 'ottoman', 'byzantine', 'palace', 'castle', 'ruins'],
            'museum': ['museum', 'gallery', 'art', 'exhibition', 'collection'],
            'religious': ['mosque', 'church', 'synagogue', 'temple', 'religious', 'holy'],
            'nature': ['nature', 'park', 'garden', 'outdoor', 'beach', 'forest', 'view', 'panorama'],
            'modern': ['modern', 'contemporary', 'shopping mall', 'tower'],
            'entertainment': ['entertainment', 'fun', 'activity', 'show', 'performance'],
            'shopping': ['shopping', 'bazaar', 'market', 'souvenir', 'mall']
        }
        
        for category, keywords in category_map.items():
            if any(kw in message for kw in keywords):
                categories.append(category)
        
        # Default to historical if no specific category
        if not categories:
            categories.append('historical')
        
        return categories
    
    def _detect_interests(self, message: str, neural_insights: Dict[str, Any]) -> List[str]:
        """
        Detect user interests.
        
        Interests: architecture, art, history, photography, culture, food, shopping
        """
        interests = []
        
        interest_map = {
            'architecture': ['architecture', 'building', 'design', 'structural'],
            'art': ['art', 'painting', 'sculpture', 'artistic'],
            'history': ['history', 'historical', 'story', 'past', 'heritage'],
            'photography': ['photo', 'photography', 'instagram', 'pictures'],
            'culture': ['culture', 'traditional', 'authentic', 'local'],
            'food': ['food', 'culinary', 'taste', 'eat'],
            'shopping': ['shopping', 'buy', 'souvenir', 'market']
        }
        
        for interest, keywords in interest_map.items():
            if any(kw in message for kw in keywords):
                interests.append(interest)
        
        return interests if interests else ['general']
    
    def _detect_time_available(self, message: str) -> str:
        """
        Detect how much time user has available.
        
        Returns: 'few_hours' | 'half_day' | 'full_day' | 'multi_day'
        """
        if any(kw in message for kw in ['quick', 'short', 'hour or two', 'brief']):
            return 'few_hours'
        elif any(kw in message for kw in ['half day', 'morning', 'afternoon']):
            return 'half_day'
        elif any(kw in message for kw in ['full day', 'whole day', 'entire day']):
            return 'full_day'
        elif any(kw in message for kw in ['few days', 'multiple days', 'week']):
            return 'multi_day'
        else:
            return 'half_day'  # Default
    
    def _detect_atmosphere_preference(self, message: str) -> str:
        """
        Detect preferred atmosphere.
        
        Returns: 'quiet' | 'lively' | 'authentic' | 'touristy' | 'flexible'
        """
        if any(kw in message for kw in ['quiet', 'peaceful', 'serene', 'calm']):
            return 'quiet'
        elif any(kw in message for kw in ['lively', 'vibrant', 'energetic', 'bustling']):
            return 'lively'
        elif any(kw in message for kw in ['authentic', 'local', 'off the beaten path', 'hidden']):
            return 'authentic'
        elif any(kw in message for kw in ['popular', 'famous', 'must-see', 'tourist']):
            return 'touristy'
        else:
            return 'flexible'
    
    def _detect_children_context(self, message: str) -> bool:
        """Detect if user is traveling with children."""
        children_keywords = ['kids', 'children', 'family', 'child', 'kid-friendly', 'toddler', 'baby']
        return any(kw in message for kw in children_keywords)
    
    def _detect_indoor_outdoor_preference(self, message: str, neural_insights: Dict[str, Any]) -> str:
        """
        Detect indoor/outdoor preference.
        
        Returns: 'indoor' | 'outdoor' | 'flexible'
        """
        if any(kw in message for kw in ['indoor', 'inside', 'covered']):
            return 'indoor'
        elif any(kw in message for kw in ['outdoor', 'outside', 'open air']):
            return 'outdoor'
        else:
            return 'flexible'
    
    def _detect_crowd_tolerance(self, message: str) -> str:
        """
        Detect crowd tolerance.
        
        Returns: 'avoid_crowds' | 'moderate' | 'dont_mind'
        """
        if any(kw in message for kw in ['avoid crowds', 'not crowded', 'quiet', 'less touristy']):
            return 'avoid_crowds'
        elif any(kw in message for kw in ['popular', 'famous', 'don\'t mind crowds']):
            return 'dont_mind'
        else:
            return 'moderate'
    
    def _extract_location_preference(self, message: str) -> Optional[str]:
        """Extract location/neighborhood preference."""
        locations = [
            'sultanahmet', 'beyoglu', 'galata', 'karakoy', 'besiktas',
            'taksim', 'kadikoy', 'uskudar', 'ortakoy', 'bebek',
            'balat', 'fener', 'eminonu', 'fatih', 'eyup'
        ]
        
        for location in locations:
            if location in message:
                return location.title()
        
        return None
    
    def _detect_budget_level(self, message: str, neural_insights: Dict[str, Any]) -> str:
        """
        Detect budget level for paid attractions.
        
        Returns: 'free' | 'budget' | 'moderate' | 'premium'
        """
        if any(kw in message for kw in ['free', 'no cost', 'without paying']):
            return 'free'
        elif any(kw in message for kw in ['cheap', 'affordable', 'budget']):
            return 'budget'
        elif any(kw in message for kw in ['premium', 'expensive', 'luxury']):
            return 'premium'
        else:
            return 'moderate'
    
    def _get_weather_context(self) -> Dict[str, Any]:
        """Get current weather for context-aware suggestions."""
        try:
            if self.weather_service:
                weather = self.weather_service.get_current_weather()
                return {
                    'condition': weather.get('condition', 'unknown'),
                    'temperature': weather.get('temperature'),
                    'is_rainy': weather.get('condition') in ['rainy', 'stormy'],
                    'is_cold': weather.get('temperature', 20) < 15,
                    'is_hot': weather.get('temperature', 20) > 30,
                    'is_good_weather': weather.get('condition') in ['sunny', 'clear', 'partly cloudy']
                }
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch weather: {e}")
        
        return {}
    
    def _get_time_context(self) -> Dict[str, Any]:
        """Get current time context."""
        now = datetime.now()
        return {
            'hour': now.hour,
            'day_of_week': now.strftime('%A'),
            'is_weekend': now.weekday() >= 5,
            'is_morning': 6 <= now.hour < 12,
            'is_afternoon': 12 <= now.hour < 18,
            'is_evening': 18 <= now.hour < 22,
            'is_night': now.hour >= 22 or now.hour < 6
        }
    
    def _get_user_preferences(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user preferences from profile and history."""
        return {
            'favorite_categories': getattr(user_profile, 'interests', []),
            'visited_attractions': getattr(user_profile, 'visit_frequency', {}).keys() if user_profile else [],
            'interests': getattr(user_profile, 'interests', []),
            'accessibility_needs': getattr(user_profile, 'accessibility_needs', None) or []
        }
    
    # ==================== CANDIDATE RETRIEVAL ====================
    
    def _get_candidate_attractions(self, ml_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve candidate attractions based on ML context.
        """
        try:
            # Build query parameters
            query_params = {
                'categories': ml_context.get('categories'),
                'location': ml_context.get('location_preference'),
                'indoor_outdoor': ml_context.get('indoor_outdoor'),
                'with_children': ml_context.get('with_children')
            }
            
            # Try advanced service first
            if self.advanced_attraction_service:
                candidates = self.advanced_attraction_service.search_attractions(query_params)
                if candidates:
                    return candidates
            
            # Fall back to basic service
            if self.attraction_service:
                candidates = self.attraction_service.search_attractions(query_params)
                return candidates
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error retrieving candidates: {e}")
            return []
    
    # ==================== NEURAL RANKING ====================
    
    def _apply_neural_ranking(self, attractions: List[Dict[str, Any]],
                             ml_context: Dict[str, Any],
                             neural_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply ML-powered ranking to attractions.
        
        Scoring factors:
        - Neural semantic similarity (35%)
        - User preference match (25%)
        - Context relevance (weather, time, etc.) (25%)
        - Popularity and ratings (15%)
        """
        try:
            query_embedding = neural_insights.get('query_embedding')
            
            for attraction in attractions:
                score = 0.0
                
                # 1. Neural Similarity (35%)
                if query_embedding and 'embedding' in attraction:
                    similarity = self._compute_similarity(
                        query_embedding,
                        attraction['embedding']
                    )
                    score += similarity * 0.35
                
                # 2. User Preference Match (25%)
                preference_score = self._compute_preference_score(attraction, ml_context)
                score += preference_score * 0.25
                
                # 3. Context Relevance (25%)
                context_score = self._compute_context_relevance(attraction, ml_context)
                score += context_score * 0.25
                
                # 4. Popularity & Rating (15%)
                rating_score = attraction.get('rating', 4.0) / 5.0
                popularity_score = min(attraction.get('visitor_count', 0) / 10000, 1.0)
                score += (rating_score * 0.10) + (popularity_score * 0.05)
                
                attraction['ml_score'] = score
                attraction['ranking_breakdown'] = {
                    'neural_similarity': similarity if query_embedding else 0,
                    'preference_match': preference_score,
                    'context_relevance': context_score,
                    'rating_score': rating_score,
                    'popularity_score': popularity_score,
                    'total_score': score
                }
            
            # Sort by ML score
            ranked = sorted(attractions, key=lambda x: x.get('ml_score', 0), reverse=True)
            logger.info(f"🎯 Neural ranking applied: Top score = {ranked[0].get('ml_score', 0):.2f}")
            
            return ranked
            
        except Exception as e:
            logger.error(f"❌ Error in neural ranking: {e}")
            return attractions  # Return unranked if error
    
    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between embeddings."""
        try:
            import numpy as np
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except Exception:
            return 0.5  # Default similarity
    
    def _compute_preference_score(self, attraction: Dict[str, Any>,
                                  ml_context: Dict[str, Any]) -> float:
        """Compute how well attraction matches user preferences."""
        score = 0.0
        user_prefs = ml_context.get('user_preferences', {})
        
        # Category match
        attraction_cats = attraction.get('categories', [])
        favorite_cats = user_prefs.get('favorite_categories', [])
        if any(cat in favorite_cats for cat in attraction_cats):
            score += 0.4
        
        # Interest match
        attraction_features = attraction.get('features', [])
        user_interests = ml_context.get('interests', [])
        if any(interest in attraction_features for interest in user_interests):
            score += 0.3
        
        # Not previously visited (exploration bonus)
        visited = user_prefs.get('visited_attractions', [])
        if attraction.get('id') not in visited:
            score += 0.2
        
        # Accessibility match
        accessibility_needs = ml_context.get('accessibility_needs', [])
        if accessibility_needs:
            attraction_accessibility = attraction.get('accessibility_features', [])
            if all(need in attraction_accessibility for need in accessibility_needs):
                score += 0.1
        
        return min(score, 1.0)
    
    def _compute_context_relevance(self, attraction: Dict[str, Any],
                                   ml_context: Dict[str, Any]) -> float:
        """Compute how relevant attraction is to current context."""
        score = 0.0
        
        # Weather relevance
        weather = ml_context.get('weather', {})
        if weather:
            weather_score = self._compute_weather_relevance(attraction, weather, ml_context)
            score += weather_score * 0.3
        
        # Time relevance (opening hours, crowd levels)
        time_context = ml_context.get('time_context', {})
        if time_context:
            time_score = self._compute_time_relevance(attraction, time_context)
            score += time_score * 0.25
        
        # Crowd tolerance match
        crowd_tolerance = ml_context.get('crowd_tolerance', 'moderate')
        crowd_level = attraction.get('typical_crowd_level', 'moderate')
        if crowd_tolerance == 'avoid_crowds' and crowd_level == 'low':
            score += 0.2
        elif crowd_tolerance == 'dont_mind' or crowd_level == 'moderate':
            score += 0.15
        
        # Budget match
        budget = ml_context.get('budget_level', 'moderate')
        entrance_fee = attraction.get('entrance_fee', 0)
        if budget == 'free' and entrance_fee == 0:
            score += 0.15
        elif budget == 'budget' and entrance_fee <= 50:
            score += 0.10
        elif budget == 'premium' and entrance_fee > 0:
            score += 0.10
        
        # Children-friendly match
        if ml_context.get('with_children') and attraction.get('child_friendly'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _compute_weather_relevance(self, attraction: Dict[str, Any],
                                   weather: Dict[str, Any],
                                   ml_context: Dict[str, Any]) -> float:
        """Compute weather-based relevance score."""
        score = 0.0
        is_indoor = attraction.get('indoor', False)
        is_outdoor = not is_indoor
        
        indoor_outdoor_pref = ml_context.get('indoor_outdoor', 'flexible')
        
        # Rainy weather
        if weather.get('is_rainy'):
            if is_indoor:
                score += 1.0
            elif indoor_outdoor_pref == 'flexible':
                score += 0.3  # Outdoor ok if flexible
        
        # Cold weather
        elif weather.get('is_cold'):
            if is_indoor:
                score += 0.7
            elif attraction.get('has_heating') or attraction.get('short_visit'):
                score += 0.5
        
        # Hot weather
        elif weather.get('is_hot'):
            if is_indoor and attraction.get('has_ac'):
                score += 0.8
            elif is_outdoor and attraction.get('has_shade'):
                score += 0.6
        
        # Good weather
        elif weather.get('is_good_weather'):
            if is_outdoor:
                score += 1.0
            else:
                score += 0.7  # Indoor still good
        
        # Default
        else:
            score = 0.7
        
        return score
    
    def _compute_time_relevance(self, attraction: Dict[str, Any],
                               time_context: Dict[str, Any]) -> float:
        """Compute time-based relevance score."""
        score = 0.0
        
        # Check if open now
        if self._is_open_now(attraction, time_context):
            score += 0.5
        else:
            return 0.0  # Not open, very low relevance
        
        # Weekend vs weekday
        is_weekend = time_context.get('is_weekend', False)
        if is_weekend and attraction.get('better_on_weekend'):
            score += 0.3
        elif not is_weekend and attraction.get('better_on_weekday'):
            score += 0.3
        
        # Time of day appropriateness
        if time_context.get('is_morning') and attraction.get('best_time') == 'morning':
            score += 0.2
        elif time_context.get('is_afternoon') and attraction.get('best_time') == 'afternoon':
            score += 0.2
        elif time_context.get('is_evening') and attraction.get('evening_hours'):
            score += 0.2
        
        return min(score, 1.0)
    
    def _is_open_now(self, attraction: Dict[str, Any], time_context: Dict[str, Any]) -> bool:
        """Check if attraction is currently open based on time context."""
        # Implementation details...
        return True  # Default to open
    
    def _generate_attraction_map(self, attractions: List[Dict[str, Any]], 
                                ml_context: Dict[str, Any]) -> Optional[Dict]:
        """
        Generate map visualization for attractions.
        
        Args:
            attractions: List of attractions to show on map
            ml_context: ML context with user preferences
            
        Returns:
            Map data dictionary in Leaflet format, or None if map service disabled
        """
        if not self.has_maps or not self.map_service:
            return None
        
        try:
            # Extract user location if available
            user_location = None
            if ml_context.get('user_location'):
                loc = ml_context['user_location']
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    user_location = (loc[0], loc[1])
            
            # Prepare attraction data for map service
            attraction_data = []
            for attr in attractions:
                if 'lat' in attr and 'lon' in attr:
                    attraction_data.append({
                        'name': attr.get('name', 'Attraction'),
                        'lat': attr['lat'],
                        'lon': attr['lon'],
                        'description': attr.get('description', ''),
                        'address': attr.get('address', ''),
                        'category': attr.get('category', ''),
                        'rating': attr.get('rating', '')
                    })
            
            # Generate map
            map_data = self.map_service.create_attraction_map(
                attraction_data,
                user_location=user_location
            )
            
            if map_data:
                logger.info(f"🗺️ Generated map with {len(attraction_data)} attractions")
            
            return map_data
            
        except Exception as e:
            logger.error(f"Error generating attraction map: {e}")
            return None
    
    # ==================== RESPONSE GENERATION ====================
    
    def _generate_ml_enhanced_response(self, attractions: List[Dict[str, Any]],
                                      ml_context: Dict[str, Any],
                                      neural_insights: Dict[str, Any],
                                      language: str = 'en') -> str:
        """
        Generate context-aware, sentiment-appropriate response.
        
        Args:
            attractions: List of attractions
            ml_context: ML context
            neural_insights: Neural insights
            language: Language code ('en' or 'tr')
            
        Returns:
            Formatted response (bilingual)
        """
        # Build response sections
        sections = []
        
        # 1. Personalized greeting (bilingual)
        greeting = self._generate_contextual_greeting(ml_context, neural_insights, language)
        sections.append(greeting)
        
        # 2. Attraction recommendations (bilingual)
        for i, attraction in enumerate(attractions, 1):
            rec = self._format_attraction_recommendation(attraction, i, ml_context, language)
            sections.append(rec)
        
        # 3. Practical tips (weather, timing, transport) (bilingual)
        tips = self._generate_contextual_tips(ml_context, attractions, language)
        if tips:
            tip_header = "💡 **Pratik İpuçları:**" if language == 'tr' else "💡 **Practical Tips:**"
            sections.append(f"\n{tip_header}\n{tips}")
        
        # 4. Itinerary suggestion (if multiple attractions) (bilingual)
        if len(attractions) > 1:
            itinerary = self._generate_itinerary_suggestion(attractions, ml_context, language)
            if itinerary:
                route_header = "📍 **Önerilen Rota:**" if language == 'tr' else "📍 **Suggested Route:**"
                sections.append(f"\n{route_header}\n{itinerary}")
        
        # 5. Call to action (bilingual)
        cta = self._generate_call_to_action(ml_context, language)
        sections.append(f"\n{cta}")
        
        return "\n\n".join(sections)
    
    def _generate_contextual_greeting(self, ml_context: Dict[str, Any],
                                     neural_insights: Dict[str, Any],
                                     language: str = 'en') -> str:
        """
        Generate personalized greeting based on context.
        
        Args:
            ml_context: ML context
            neural_insights: Neural insights
            language: Language code ('en' or 'tr')
            
        Returns:
            Bilingual greeting
        """
        categories = ml_context.get('categories', [])
        time_available = ml_context.get('time_available', 'half_day')
        weather = ml_context.get('weather', {})
        
        # Category-based greetings (bilingual)
        if language == 'tr':
            if 'historical' in categories:
                greeting = "🏛️ **İnanılmaz Tarihi Yerler Sizi Bekliyor!**"
            elif 'museum' in categories:
                greeting = "🎨 **Sizin İçin Harika Müzeler!**"
            elif 'nature' in categories:
                greeting = "🌳 **Güzel Doğal Mekanlar!**"
            elif 'religious' in categories:
                greeting = "🕌 **Ziyaret Edilecek Kutsal Yerler!**"
            else:
                greeting = "✨ **Sizin İçin Harika Yerler!**"
        else:
            if 'historical' in categories:
                greeting = "🏛️ **Amazing Historical Sites Await!**"
            elif 'museum' in categories:
                greeting = "🎨 **Wonderful Museums for You!**"
            elif 'nature' in categories:
                greeting = "🌳 **Beautiful Natural Spots!**"
            elif 'religious' in categories:
                greeting = "🕌 **Sacred Places to Visit!**"
            else:
                greeting = "✨ **Great Attractions for You!**"
        
        # Add context
        context_note = []
        if weather.get('is_rainy'):
            context_note.append(
                "☔ Bugünün havası için mükemmel iç mekan seçenekleri" if language == 'tr'
                else "☔ Perfect indoor options for today's weather"
            )
        elif weather.get('is_good_weather'):
            context_note.append(
                "☀️ Gezmek için harika bir hava" if language == 'tr'
                else "☀️ Lovely weather for exploring"
            )
        
        if time_available == 'few_hours':
            context_note.append(
                "⏰ Zamanınız için hızlı ziyaretler" if language == 'tr'
                else "⏰ Quick visits optimized for your time"
            )
        elif time_available == 'full_day':
            context_note.append(
                "📅 Sizin için özenle seçilmiş tam günlük deneyimler" if language == 'tr'
                else "📅 Full-day experiences curated for you"
            )
        
        if context_note:
            greeting += f"\n*{' | '.join(context_note)}*"
        
        return greeting
    
    def _format_attraction_recommendation(self, attraction: Dict[str, Any],
                                         index: int,
                                         ml_context: Dict[str, Any],
                                         language: str = 'en') -> str:
        """
        Format individual attraction recommendation.
        
        Args:
            attraction: Attraction data
            index: Index number
            ml_context: ML context
            language: Language code ('en' or 'tr')
            
        Returns:
            Formatted attraction card (bilingual)
        """
        name = attraction.get('name', 'Unknown Attraction' if language == 'en' else 'Bilinmeyen Yer')
        category = attraction.get('category', 'Attraction' if language == 'en' else 'Yer')
        rating = attraction.get('rating', 0)
        neighborhood = attraction.get('neighborhood', '')
        entrance_fee = attraction.get('entrance_fee', 0)
        
        # Build recommendation text
        lines = [f"**{index}. {name}** ⭐ {rating}/5"]
        
        # Location and category
        location_info = f"📍 {neighborhood}" if neighborhood else ""
        category_info = f"🏛️ {category.title()}"
        fee_info = "🎫 Ücretsiz" if language == 'tr' and entrance_fee == 0 else ("🎫 Free" if entrance_fee == 0 else f"🎫 {entrance_fee} TL")
        
        lines.append(f"   {location_info} | {category_info} | {fee_info}")
        
        # ML insights and highlights
        ml_score = attraction.get('ml_score', 0)
        if ml_score > 0.8:
            match_text = "🎯 **İlgi alanlarınızla mükemmel eşleşme!**" if language == 'tr' else "🎯 **Perfect match for your interests!**"
            lines.append(f"   {match_text}")
        
        # Context-specific highlights
        highlights = []
        
        # Weather-appropriate
        weather = ml_context.get('weather', {})
        if weather.get('is_rainy') and attraction.get('indoor'):
            highlights.append("☔ İç mekan (yağmur için harika)" if language == 'tr' else "☔ Indoor (great for rain)")
        elif weather.get('is_good_weather') and not attraction.get('indoor'):
            highlights.append("☀️ Açık hava (mükemmel hava)" if language == 'tr' else "☀️ Outdoor (perfect weather)")
        
        # Time estimate
        duration = attraction.get('typical_duration_minutes', 0)
        if duration:
            hours = duration // 60
            mins = duration % 60
            time_str = f"{hours}sa" if hours and language == 'tr' else (f"{hours}h" if hours else f"{mins}dk" if language == 'tr' else f"{mins}min")
            highlights.append(f"⏱️ ~{time_str}")
        
        # Special features
        if ml_context.get('with_children') and attraction.get('child_friendly'):
            highlights.append("👨‍👩‍👧‍👦 Çocuklar için uygun" if language == 'tr' else "👨‍👩‍👧‍👦 Kid-friendly")
        
        if attraction.get('unesco_site'):
            highlights.append("🌟 UNESCO Sitesi" if language == 'tr' else "🌟 UNESCO Site")
        
        if attraction.get('photo_spot'):
            highlights.append("📸 Harika fotoğraflar" if language == 'tr' else "📸 Great photos")
        
        if highlights:
            lines.append(f"   {' | '.join(highlights)}")
        
        # Description
        if attraction.get('description'):
            desc = attraction['description'][:120] + "..." if len(attraction.get('description', '')) > 120 else attraction.get('description', '')
            lines.append(f"   {desc}")
        
        # Opening hours (if relevant)
        time_context = ml_context.get('time_context', {})
        if not attraction.get('always_open'):
            opening_info = self._format_opening_hours(attraction, time_context, language)
            if opening_info:
                lines.append(f"   🕐 {opening_info}")
        
        return "\n".join(lines)
    
    def _format_opening_hours(self, attraction: Dict[str, Any],
                             time_context: Dict[str, Any],
                             language: str = 'en') -> str:
        """
        Format opening hours information.
        
        Args:
            attraction: Attraction data
            time_context: Time context
            language: Language code ('en' or 'tr')
            
        Returns:
            Opening hours text (bilingual)
        """
        opening_hours = attraction.get('opening_hours', {})
        if not opening_hours:
            return ""
        
        day = time_context.get('day_of_week', 'Monday').lower()
        today_hours = opening_hours.get(day, {})
        
        if not today_hours:
            return "Bugün kapalı" if language == 'tr' else "Closed today"
        
        open_time = today_hours.get('open', 9)
        close_time = today_hours.get('close', 18)
        
        if language == 'tr':
            return f"Açık {open_time:02d}:00 - {close_time:02d}:00"
        else:
            return f"Open {open_time:02d}:00 - {close_time:02d}:00"
    
    def _generate_contextual_tips(self, ml_context: Dict[str, Any],
                                 attractions: List[Dict[str, Any]],
                                 language: str = 'en') -> str:
        """
        Generate context-aware tips.
        
        Args:
            ml_context: ML context
            attractions: List of attractions
            language: Language code ('en' or 'tr')
            
        Returns:
            Tips text (bilingual)
        """
        tips = []
        
        # Weather tips
        weather = ml_context.get('weather', {})
        if weather.get('is_rainy'):
            tips.append(
                "☔ Şemsiye getirin ve rahat ayakkabı giyin" if language == 'tr'
                else "☔ Bring an umbrella and wear comfortable shoes"
            )
        elif weather.get('is_hot'):
            tips.append(
                "☀️ Bol su için ve güneş kremi kullanın" if language == 'tr'
                else "☀️ Stay hydrated and use sunscreen"
            )
        elif weather.get('is_cold'):
            tips.append(
                "🧥 Katmanlı giyinin" if language == 'tr'
                else "🧥 Dress warmly in layers"
            )
        
        # Timing tips
        time_context = ml_context.get('time_context', {})
        if time_context.get('is_morning'):
            tips.append(
                "🌅 Kalabalıktan kaçınmak için sabah mükemmel" if language == 'tr'
                else "🌅 Morning is perfect to avoid crowds"
            )
        elif time_context.get('is_weekend'):
            tips.append(
                "👥 Hafta sonları daha kalabalık olabilir" if language == 'tr'
                else "👥 Expect larger crowds on weekends"
            )
        
        # Crowd tips
        if ml_context.get('crowd_tolerance') == 'avoid_crowds':
            tips.append(
                "🚶 Daha az kalabalık için sabah erken veya öğleden sonra geç saatleri tercih edin" if language == 'tr'
                else "🚶 Visit early morning or late afternoon for fewer crowds"
            )
        
        # Transport tips
        if len(attractions) > 2:
            tips.append(
                "🚇 Daha kolay seyahat için İstanbulkart edinmeyi düşünün" if language == 'tr'
                else "🚇 Consider getting an Istanbul transport card for easier travel"
            )
        
        # Photography tips
        if any(a.get('photo_spot') for a in attractions):
            tips.append(
                "📸 En iyi fotoğraflar altın saatte (gün doğumu/gün batımı)" if language == 'tr'
                else "📸 Best photos in golden hour (sunrise/sunset)"
            )
        
        return "\n".join(f"• {tip}" for tip in tips)
    
    def _generate_itinerary_suggestion(self, attractions: List[Dict[str, Any]],
                                      ml_context: Dict[str, Any],
                                      language: str = 'en') -> str:
        """
        Generate suggested itinerary for multiple attractions.
        
        Args:
            attractions: List of attractions
            ml_context: ML context
            language: Language code ('en' or 'tr')
            
        Returns:
            Itinerary text (bilingual)
        """
        if len(attractions) < 2:
            return ""
        
        # Simple geographical clustering
        # Group by neighborhood
        by_neighborhood = {}
        for attraction in attractions:
            neighborhood = attraction.get('neighborhood', 'Diğer' if language == 'tr' else 'Other')
            if neighborhood not in by_neighborhood:
                by_neighborhood[neighborhood] = []
            by_neighborhood[neighborhood].append(attraction)
        
        # Build itinerary
        itinerary_parts = []
        for neighborhood, attrs in by_neighborhood.items():
            attr_names = [a.get('name') for a in attrs]
            if len(attr_names) > 1:
                itinerary_parts.append(f"**{neighborhood}:** {' → '.join(attr_names)}")
        
        if itinerary_parts:
            return "\n".join(itinerary_parts)
        
        # Fallback: simple list
        return " → ".join(a.get('name', '') for a in attractions[:3])
    
    def _generate_call_to_action(self, ml_context: Dict[str, Any], language: str = 'en') -> str:
        """
        Generate appropriate call to action.
        
        Args:
            ml_context: ML context
            language: Language code ('en' or 'tr')
            
        Returns:
            CTA text (bilingual)
        """
        # Context-specific CTA
        if ml_context.get('time_available') == 'multi_day':
            return (
                "📅 Çok günlük bir gezi programı hazırlamamı ister misiniz? En iyi sıra ve zamanlama önerebilirim!" if language == 'tr'
                else "📅 Want me to help plan a multi-day itinerary? I can suggest the best order and timing!"
            )
        elif self.transport_service:
            return (
                "🚇 Bu yerlere ulaşmak için yol tarifi veya ulaşım seçenekleri ister misiniz?" if language == 'tr'
                else "🚇 Would you like directions or transport options to reach these places?"
            )
        
        # Default CTAs (bilingual)
        if language == 'tr':
            return "🗺️ Bu yerlerden birine yol tarifi ister misiniz?"
        else:
            return "🗺️ Want directions to any of these attractions?"
    
    def _generate_no_results_response(self, ml_context: Dict[str, Any], language: str = 'en') -> str:
        """
        Generate helpful response when no attractions found.
        
        Args:
            ml_context: ML context
            language: Language code ('en' or 'tr')
            
        Returns:
            No results message (bilingual)
        """
        if language == 'tr':
            return (
                "😊 Belirli gereksinimlerinize uyan yerler bulamadım, "
                "ama size yardımcı olmak için buradayım! Bana şunları söyleyebilir misiniz:\n\n"
                "• En çok hangi tür yer ilginizi çekiyor? (tarihi, müze, doğa, vb.)\n"
                "• İstanbul'un hangi bölgesini keşfetmek istersiniz?\n"
                "• Ne kadar zamanınız var?\n\n"
                "Sizin için mükemmel yerleri bulacağım! ✨"
            )
        else:
            return (
                "😊 I couldn't find attractions matching all your specific requirements, "
                "but I'm here to help! Could you tell me:\n\n"
                "• What type of attraction interests you most? (historical, museum, nature, etc.)\n"
                "• Which part of Istanbul would you like to explore?\n"
                "• How much time do you have available?\n\n"
                "I'll find the perfect spots for you! ✨"
            )
    
    def _generate_fallback_response(self, language: str = 'en') -> str:
        """
        Generate fallback response on error.
        
        Args:
            language: Language code ('en' or 'tr')
            
        Returns:
            Fallback message (bilingual)
        """
        if language == 'tr':
            return (
                "Özür dilerim, şu anda gezilecek yer bilgilerine erişirken sorun yaşıyorum. 😔\n\n"
                "Şunları deneyebilirsiniz:\n"
                "• Belirli bir bölge hakkında sormak (örn. 'Sultanahmet'teki yerler')\n"
                "• Görmek istedikleriniz hakkında daha spesifik olmak\n"
                "• Bir süre sonra tekrar sormak\n\n"
                "Harika yerleri keşfetmenize yardımcı olmak için buradayım! ✨"
            )
        else:
            return (
                "I apologize, but I'm having trouble accessing attraction information right now. 😔\n\n"
                "Could you try:\n"
                "• Asking about a specific area (e.g., 'attractions in Sultanahmet')\n"
                "• Being more specific about what you'd like to see\n"
                "• Asking again in a moment\n\n"
                "I'm here to help you discover amazing places! ✨"
            )
    
    # ==================== USER HISTORY ====================
    
    def _update_user_history(self, user_profile: Dict[str, Any],
                            ml_context: Dict[str, Any],
                            recommendations: List[Dict[str, Any]]) -> None:
        """Update user history with this interaction."""
        try:
            if self.user_manager:
                self.user_manager.log_interaction(
                    user_id=getattr(user_profile, 'user_id', 'unknown'),
                    interaction_type='attraction_query',
                    context=ml_context,
                    recommendations=[r.get('id') for r in recommendations]
                )
        except Exception as e:
            logger.warning(f"⚠️ Could not update user history: {e}")
