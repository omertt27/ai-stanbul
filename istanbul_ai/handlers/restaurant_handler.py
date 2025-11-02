"""
ML-Enhanced Restaurant Handler for Istanbul AI Chat System

This module handles all restaurant-related queries with full neural ML integration.
Leverages T4 GPU for context extraction, ranking, and response generation.

Features:
- ML-powered budget detection and filtering
- Neural sentiment analysis for occasion detection
- Temporal context awareness (meal times, urgency)
- Dietary restriction extraction
- Cuisine preference matching
- ML-based restaurant ranking and scoring
- Context-aware response generation
- 🌐 Full English/Turkish bilingual support

Author: Istanbul AI Team
Date: October 27, 2025
Updated: November 2, 2025 - Added bilingual support
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import bilingual support
try:
    from ..services.bilingual_manager import BilingualManager, Language
    BILINGUAL_AVAILABLE = True
except ImportError:
    BILINGUAL_AVAILABLE = False
    Language = None

logger = logging.getLogger(__name__)


class RestaurantHandler:
    """
    Handles restaurant queries with full ML/neural integration.
    
    This handler uses neural insights to:
    1. Extract rich context (budget, dietary, occasion, urgency)
    2. Apply ML-powered filtering and ranking
    3. Generate intelligent, personalized responses
    """
    
    def __init__(self, neural_processor, restaurant_service, response_generator, bilingual_manager=None):
        """
        Initialize the Restaurant Handler.
        
        Args:
            neural_processor: Neural network processor for ML insights
            restaurant_service: Service for fetching restaurant data
            response_generator: Service for generating responses
            bilingual_manager: BilingualManager for language support
        """
        self.neural_processor = neural_processor
        self.restaurant_service = restaurant_service
        self.response_generator = response_generator
        self.bilingual_manager = bilingual_manager
        self.has_bilingual = bilingual_manager is not None and BILINGUAL_AVAILABLE
        
        # Budget keywords for ML context
        self.budget_keywords = {
            'cheap': ['cheap', 'affordable', 'budget', 'inexpensive', 'ucuz', 'ekonomik'],
            'moderate': ['moderate', 'mid-range', 'reasonable', 'orta', 'makul'],
            'expensive': ['expensive', 'luxury', 'fine dining', 'upscale', 'pahalı', 'lüks'],
            'very_expensive': ['very expensive', 'michelin', 'exclusive', 'çok pahalı']
        }
        
        # Dietary restriction patterns
        self.dietary_patterns = {
            'vegetarian': r'\b(vegetarian|vegan|veggie|vejetaryen)\b',
            'halal': r'\b(halal|helal|islamic)\b',
            'gluten_free': r'\b(gluten[- ]?free|glutensiz)\b',
            'kosher': r'\b(kosher|kaşer)\b',
            'lactose_free': r'\b(lactose[- ]?free|laktozsuz|dairy[- ]?free)\b'
        }
        
        # Occasion indicators
        self.occasion_patterns = {
            'romantic': r'\b(romantic|date|anniversary|couple|romantik|çift)\b',
            'family': r'\b(family|kids|children|aile|çocuk)\b',
            'business': r'\b(business|meeting|corporate|iş|toplantı)\b',
            'celebration': r'\b(celebration|birthday|party|kutlama|doğum günü)\b',
            'casual': r'\b(casual|quick|fast|günlük|hızlı)\b'
        }
        
        # Meal time patterns
        self.meal_patterns = {
            'breakfast': r'\b(breakfast|morning|kahvaltı|sabah)\b',
            'brunch': r'\b(brunch)\b',
            'lunch': r'\b(lunch|öğle|öğlen)\b',
            'dinner': r'\b(dinner|evening|akşam|gece)\b',
            'late_night': r'\b(late night|midnight|gece yarısı)\b'
        }
        
        logger.info(f"✅ RestaurantHandler initialized with ML integration (Bilingual: {self.has_bilingual})")
    
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
    
    def generate_response(
        self,
        message: str,
        neural_insights: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None,
        user_location: Optional[Tuple[float, float]] = None,
        context=None
    ) -> Dict[str, Any]:
        """
        Generate ML-enhanced restaurant response.
        
        Args:
            message: User's message
            neural_insights: Neural processor insights
            user_profile: User profile data
            user_location: User's current location (lat, lon)
            context: Conversation context (includes language)
            
        Returns:
            Response dictionary with recommendations
        """
        # 🌐 BILINGUAL: Extract language from context
        language = self._get_language(context)
        logger.info(f"🍽️ Processing restaurant query with ML enhancement (lang: {language})")
        
        try:
            # Step 1: Extract ML context
            ml_context = self._extract_ml_context(message, neural_insights, user_profile)
            logger.info(f"📊 ML Context: {ml_context}")
            
            # Step 2: Get cuisine preference
            cuisine = self._detect_cuisine_preference(message, neural_insights)
            
            # Step 3: Fetch restaurants based on context
            restaurants = self._fetch_restaurants(
                cuisine=cuisine,
                budget=ml_context.get('budget'),
                location=user_location,
                dietary=ml_context.get('dietary_restrictions', [])
            )
            
            if not restaurants:
                return self._generate_no_results_response(message, ml_context, language)
            
            # Step 4: Apply ML-powered ranking
            ranked_restaurants = self._apply_neural_ranking(
                restaurants=restaurants,
                ml_context=ml_context,
                neural_insights=neural_insights,
                user_profile=user_profile
            )
            
            # Step 5: Generate intelligent response (bilingual)
            response = self._generate_ml_response(
                restaurants=ranked_restaurants[:5],  # Top 5
                ml_context=ml_context,
                neural_insights=neural_insights,
                message=message,
                language=language  # 🌐 Pass language
            )
            
            logger.info(f"✅ Generated response with {len(ranked_restaurants)} restaurants")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in restaurant handler: {e}", exc_info=True)
            return self._generate_fallback_response(message, language)
    
    def _extract_ml_context(
        self,
        message: str,
        neural_insights: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract rich ML context from message and neural insights.
        
        Uses neural processor insights for:
        - Budget detection (from sentiment + keywords)
        - Occasion detection (from temporal + sentiment)
        - Urgency level (from temporal context)
        - Dietary restrictions (from keywords + profile)
        - Meal time context (from temporal data)
        
        Args:
            message: User's message
            neural_insights: Neural processor insights
            user_profile: User profile data
            
        Returns:
            ML context dictionary
        """
        message_lower = message.lower()
        
        context = {
            'budget': self._ml_detect_budget(message_lower, neural_insights),
            'dietary_restrictions': self._extract_dietary_restrictions(message_lower, user_profile),
            'occasion': self._ml_detect_occasion(message_lower, neural_insights),
            'meal_time': self._detect_meal_time(message_lower, neural_insights),
            'urgency': self._ml_detect_urgency(neural_insights),
            'sentiment': neural_insights.get('sentiment', {}).get('overall', 'neutral'),
            'keywords': neural_insights.get('keywords', []),
            'group_size': self._estimate_group_size(message_lower, neural_insights),
            'preferences': self._extract_preferences(message_lower, neural_insights, user_profile)
        }
        
        return context
    
    def _ml_detect_budget(
        self,
        message: str,
        neural_insights: Dict[str, Any]
    ) -> str:
        """
        ML-powered budget detection using neural insights.
        
        Combines:
        - Explicit budget keywords
        - Sentiment analysis (positive = willing to spend more)
        - Keywords from neural processor
        - User query complexity
        
        Args:
            message: User's message (lowercase)
            neural_insights: Neural insights
            
        Returns:
            Budget level: 'cheap', 'moderate', 'expensive', 'very_expensive', or 'any'
        """
        # Check explicit budget keywords
        for budget_level, keywords in self.budget_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    logger.info(f"💰 Explicit budget detected: {budget_level}")
                    return budget_level
        
        # Use neural sentiment to infer budget flexibility
        sentiment = neural_insights.get('sentiment', {})
        overall_sentiment = sentiment.get('overall', 'neutral')
        confidence = sentiment.get('confidence', 0.5)
        
        # Check neural keywords for budget hints
        keywords = neural_insights.get('keywords', [])
        keyword_str = ' '.join(keywords).lower()
        
        for budget_level, budget_keywords in self.budget_keywords.items():
            for keyword in budget_keywords:
                if keyword in keyword_str:
                    logger.info(f"💰 Neural keyword budget detected: {budget_level}")
                    return budget_level
        
        # Infer from sentiment and context
        if overall_sentiment == 'positive' and confidence > 0.7:
            # Happy user might be willing to spend more
            if any(word in message for word in ['best', 'great', 'amazing', 'perfect', 'special']):
                logger.info("💰 ML-inferred budget: expensive (positive sentiment + quality keywords)")
                return 'expensive'
            return 'moderate'
        elif overall_sentiment == 'negative' or any(word in message for word in ['need', 'want', 'looking for']):
            # Neutral or need-based queries → moderate
            return 'moderate'
        
        # Default to moderate for balanced results
        logger.info("💰 Default budget: moderate")
        return 'moderate'
    
    def _extract_dietary_restrictions(
        self,
        message: str,
        user_profile: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract dietary restrictions from message and user profile.
        
        Args:
            message: User's message (lowercase)
            user_profile: User profile data
            
        Returns:
            List of dietary restrictions
        """
        restrictions = []
        
        # Check message for dietary patterns
        for diet_type, pattern in self.dietary_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                restrictions.append(diet_type)
        
        # Add from user profile if available
        if user_profile and 'dietary_restrictions' in user_profile:
            profile_restrictions = user_profile.get('dietary_restrictions', [])
            restrictions.extend([r for r in profile_restrictions if r not in restrictions])
        
        if restrictions:
            logger.info(f"🥗 Dietary restrictions: {restrictions}")
        
        return restrictions
    
    def _ml_detect_occasion(
        self,
        message: str,
        neural_insights: Dict[str, Any]
    ) -> Optional[str]:
        """
        ML-powered occasion detection using neural insights.
        
        Args:
            message: User's message (lowercase)
            neural_insights: Neural insights
            
        Returns:
            Occasion type or None
        """
        # Check explicit occasion patterns
        for occasion, pattern in self.occasion_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                logger.info(f"🎉 Occasion detected: {occasion}")
                return occasion
        
        # Infer from sentiment and keywords
        sentiment = neural_insights.get('sentiment', {})
        keywords = neural_insights.get('keywords', [])
        
        # Romantic inference
        if sentiment.get('overall') == 'positive' and any(
            word in keywords for word in ['special', 'nice', 'romantic', 'date']
        ):
            logger.info("🎉 ML-inferred occasion: romantic")
            return 'romantic'
        
        # Business inference
        if any(word in keywords for word in ['meeting', 'business', 'professional', 'work']):
            logger.info("🎉 ML-inferred occasion: business")
            return 'business'
        
        return None
    
    def _detect_meal_time(
        self,
        message: str,
        neural_insights: Dict[str, Any]
    ) -> Optional[str]:
        """
        Detect meal time from message and temporal context.
        
        Args:
            message: User's message (lowercase)
            neural_insights: Neural insights
            
        Returns:
            Meal time or None
        """
        # Check explicit meal time patterns
        for meal, pattern in self.meal_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                logger.info(f"🕐 Meal time detected: {meal}")
                return meal
        
        # Use temporal context from neural insights
        temporal = neural_insights.get('temporal', {})
        time_sensitivity = temporal.get('time_sensitivity', 'none')
        
        if time_sensitivity in ['urgent', 'immediate']:
            # Infer from current time
            current_hour = datetime.now().hour
            if 6 <= current_hour < 11:
                return 'breakfast'
            elif 11 <= current_hour < 15:
                return 'lunch'
            elif 15 <= current_hour < 18:
                return 'brunch'
            elif 18 <= current_hour < 23:
                return 'dinner'
            else:
                return 'late_night'
        
        return None
    
    def _ml_detect_urgency(self, neural_insights: Dict[str, Any]) -> str:
        """
        Detect urgency level from neural temporal context.
        
        Args:
            neural_insights: Neural insights
            
        Returns:
            Urgency level: 'immediate', 'soon', 'flexible', 'planning'
        """
        temporal = neural_insights.get('temporal', {})
        time_sensitivity = temporal.get('time_sensitivity', 'none')
        
        urgency_map = {
            'urgent': 'immediate',
            'immediate': 'immediate',
            'soon': 'soon',
            'flexible': 'flexible',
            'none': 'planning'
        }
        
        urgency = urgency_map.get(time_sensitivity, 'flexible')
        logger.info(f"⏰ Urgency detected: {urgency}")
        return urgency
    
    def _estimate_group_size(
        self,
        message: str,
        neural_insights: Dict[str, Any]
    ) -> int:
        """
        Estimate group size from message.
        
        Args:
            message: User's message (lowercase)
            neural_insights: Neural insights
            
        Returns:
            Estimated group size
        """
        # Look for explicit numbers
        numbers = re.findall(r'\b(\d+)\s*(people|person|pax|kişi)\b', message)
        if numbers:
            return int(numbers[0][0])
        
        # Infer from keywords
        if any(word in message for word in ['family', 'group', 'friends', 'aile', 'grup']):
            return 4
        elif any(word in message for word in ['couple', 'two', 'date', 'çift', 'iki']):
            return 2
        
        # Default to 2
        return 2
    
    def _extract_preferences(
        self,
        message: str,
        neural_insights: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract user preferences from message, neural insights, and profile.
        
        Args:
            message: User's message (lowercase)
            neural_insights: Neural insights
            user_profile: User profile data
            
        Returns:
            Preferences dictionary
        """
        preferences = {
            'view_preference': None,
            'ambiance': None,
            'seating': None
        }
        
        # View preferences
        if any(word in message for word in ['view', 'bosphorus', 'sea', 'boğaz', 'deniz', 'manzara']):
            preferences['view_preference'] = 'waterfront'
        elif any(word in message for word in ['garden', 'outdoor', 'terrace', 'bahçe', 'açık']):
            preferences['view_preference'] = 'outdoor'
        
        # Ambiance
        keywords = neural_insights.get('keywords', [])
        if any(word in keywords for word in ['quiet', 'peaceful', 'calm', 'sakin']):
            preferences['ambiance'] = 'quiet'
        elif any(word in keywords for word in ['lively', 'vibrant', 'busy', 'canlı']):
            preferences['ambiance'] = 'lively'
        
        # Seating
        if any(word in message for word in ['private', 'room', 'özel', 'oda']):
            preferences['seating'] = 'private'
        elif any(word in message for word in ['bar', 'counter']):
            preferences['seating'] = 'bar'
        
        return preferences
    
    def _detect_cuisine_preference(
        self,
        message: str,
        neural_insights: Dict[str, Any]
    ) -> Optional[str]:
        """
        Detect cuisine preference from message and neural keywords.
        
        Args:
            message: User's message
            neural_insights: Neural insights
            
        Returns:
            Cuisine type or None
        """
        message_lower = message.lower()
        keywords = ' '.join(neural_insights.get('keywords', [])).lower()
        search_text = message_lower + ' ' + keywords
        
        cuisine_patterns = {
            'Turkish': r'\b(turkish|türk|ottoman|osmanlı|kebab|kebap)\b',
            'Seafood': r'\b(seafood|fish|balık|deniz ürünleri)\b',
            'Italian': r'\b(italian|pizza|pasta|italyan)\b',
            'Asian': r'\b(asian|chinese|japanese|sushi|thai|asya|çin|japon)\b',
            'Mediterranean': r'\b(mediterranean|akdeniz)\b',
            'International': r'\b(international|fusion|uluslararası)\b'
        }
        
        for cuisine, pattern in cuisine_patterns.items():
            if re.search(pattern, search_text, re.IGNORECASE):
                logger.info(f"🍴 Cuisine detected: {cuisine}")
                return cuisine
        
        return None
    
    def _fetch_restaurants(
        self,
        cuisine: Optional[str],
        budget: str,
        location: Optional[Tuple[float, float]],
        dietary: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch restaurants from service based on criteria.
        
        Args:
            cuisine: Cuisine type
            budget: Budget level
            location: User location
            dietary: Dietary restrictions
            
        Returns:
            List of restaurant data
        """
        try:
            restaurants = self.restaurant_service.search_restaurants(
                cuisine=cuisine,
                budget=budget,
                location=location,
                dietary_restrictions=dietary
            )
            logger.info(f"📍 Fetched {len(restaurants)} restaurants")
            return restaurants
        except Exception as e:
            logger.error(f"❌ Error fetching restaurants: {e}")
            return []
    
    def _apply_neural_ranking(
        self,
        restaurants: List[Dict[str, Any]],
        ml_context: Dict[str, Any],
        neural_insights: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply ML-powered ranking to restaurants.
        
        Ranking factors:
        1. Budget match score (40%)
        2. Occasion appropriateness (20%)
        3. Dietary compatibility (15%)
        4. Sentiment alignment (10%)
        5. User profile match (10%)
        6. Location proximity (5%)
        
        Args:
            restaurants: List of restaurants
            ml_context: ML context
            neural_insights: Neural insights
            user_profile: User profile
            
        Returns:
            Ranked list of restaurants
        """
        logger.info("🧠 Applying neural ranking to restaurants")
        
        for restaurant in restaurants:
            score = 0.0
            
            # 1. Budget match (40%)
            budget_score = self._score_budget_match(restaurant, ml_context.get('budget'))
            score += budget_score * 0.4
            
            # 2. Occasion appropriateness (20%)
            occasion_score = self._score_occasion_match(restaurant, ml_context.get('occasion'))
            score += occasion_score * 0.2
            
            # 3. Dietary compatibility (15%)
            dietary_score = self._score_dietary_match(restaurant, ml_context.get('dietary_restrictions', []))
            score += dietary_score * 0.15
            
            # 4. Sentiment alignment (10%)
            sentiment_score = self._score_sentiment_match(restaurant, neural_insights)
            score += sentiment_score * 0.1
            
            # 5. User profile match (10%)
            profile_score = self._score_profile_match(restaurant, user_profile)
            score += profile_score * 0.1
            
            # 6. Base rating (5%)
            rating_score = restaurant.get('rating', 3.5) / 5.0
            score += rating_score * 0.05
            
            restaurant['ml_score'] = round(score, 3)
        
        # Sort by ML score
        ranked = sorted(restaurants, key=lambda r: r.get('ml_score', 0), reverse=True)
        
        logger.info(f"✅ Ranked {len(ranked)} restaurants (top score: {ranked[0].get('ml_score', 0):.3f})")
        return ranked
    
    def _score_budget_match(self, restaurant: Dict[str, Any], budget: str) -> float:
        """Score how well restaurant matches budget."""
        restaurant_budget = restaurant.get('price_level', 'moderate')
        
        budget_levels = ['cheap', 'moderate', 'expensive', 'very_expensive']
        
        if budget == 'any':
            return 1.0
        
        try:
            user_idx = budget_levels.index(budget)
            rest_idx = budget_levels.index(restaurant_budget)
            diff = abs(user_idx - rest_idx)
            
            # Perfect match = 1.0, 1 level off = 0.7, 2 levels = 0.4, 3 levels = 0.1
            score_map = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}
            return score_map.get(diff, 0.1)
        except ValueError:
            return 0.5
    
    def _score_occasion_match(self, restaurant: Dict[str, Any], occasion: Optional[str]) -> float:
        """Score how well restaurant matches occasion."""
        if not occasion:
            return 0.8  # Neutral score
        
        restaurant_tags = restaurant.get('tags', [])
        tag_str = ' '.join(restaurant_tags).lower()
        
        occasion_keywords = {
            'romantic': ['romantic', 'intimate', 'cozy', 'date'],
            'family': ['family', 'kids', 'casual', 'spacious'],
            'business': ['quiet', 'professional', 'upscale', 'private'],
            'celebration': ['lively', 'festive', 'spacious', 'upscale'],
            'casual': ['casual', 'quick', 'relaxed', 'friendly']
        }
        
        keywords = occasion_keywords.get(occasion, [])
        matches = sum(1 for kw in keywords if kw in tag_str)
        
        return min(1.0, 0.5 + (matches * 0.2))
    
    def _score_dietary_match(self, restaurant: Dict[str, Any], restrictions: List[str]) -> float:
        """Score dietary compatibility."""
        if not restrictions:
            return 1.0
        
        restaurant_dietary = restaurant.get('dietary_options', [])
        
        if not restaurant_dietary:
            return 0.5  # Unknown = neutral
        
        matches = sum(1 for r in restrictions if r in restaurant_dietary)
        return matches / len(restrictions) if restrictions else 1.0
    
    def _score_sentiment_match(self, restaurant: Dict[str, Any], neural_insights: Dict[str, Any]) -> float:
        """Score based on sentiment alignment."""
        sentiment = neural_insights.get('sentiment', {}).get('overall', 'neutral')
        rating = restaurant.get('rating', 3.5)
        
        if sentiment == 'positive':
            # Positive users prefer higher-rated places
            return min(1.0, rating / 4.5)
        elif sentiment == 'negative':
            # Concerned users appreciate reliable choices
            return 1.0 if rating >= 4.0 else 0.7
        else:
            return 0.8
    
    def _score_profile_match(self, restaurant: Dict[str, Any], user_profile: Optional[Dict[str, Any]]) -> float:
        """Score based on user profile preferences."""
        if not user_profile:
            return 0.7
        
        score = 0.0
        
        # Check cuisine preferences
        preferred_cuisines = user_profile.get('preferred_cuisines', [])
        if preferred_cuisines:
            restaurant_cuisine = restaurant.get('cuisine', '')
            if restaurant_cuisine in preferred_cuisines:
                score += 0.5
        
        # Check past visits
        visited = user_profile.get('visited_restaurants', [])
        if restaurant.get('id') in visited:
            score += 0.3  # Bonus for familiar places
        
        return min(1.0, score + 0.5)
    
    def _generate_ml_response(
        self,
        restaurants: List[Dict[str, Any]],
        ml_context: Dict[str, Any],
        neural_insights: Dict[str, Any],
        message: str,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Generate ML-enhanced response with restaurant recommendations.
        
        Args:
            restaurants: Ranked restaurants
            ml_context: ML context
            neural_insights: Neural insights
            message: Original message
            language: Language code ('en' or 'tr')
            
        Returns:
            Response dictionary
        """
        sentiment = neural_insights.get('sentiment', {}).get('overall', 'neutral')
        urgency = ml_context.get('urgency', 'flexible')
        
        # Generate intro based on context (bilingual)
        intro = self._generate_contextual_intro(ml_context, sentiment, urgency, language)
        
        # Format restaurants (bilingual)
        restaurant_list = self._format_restaurant_list(restaurants, ml_context, language)
        
        # Generate tips based on ML context (bilingual)
        tips = self._generate_ml_tips(ml_context, restaurants, language)
        
        # Build response
        tip_prefix = "💡" if language == 'en' else "💡"
        response_text = f"{intro}\n\n{restaurant_list}"
        if tips:
            response_text += f"\n\n{tip_prefix} {tips}"
        
        return {
            'response': response_text,
            'intent': 'restaurant',
            'restaurants': restaurants,
            'ml_context': ml_context,
            'confidence': 0.9,
            'language': language
        }
    
    def _generate_contextual_intro(
        self,
        ml_context: Dict[str, Any],
        sentiment: str,
        urgency: str,
        language: str = 'en'
    ) -> str:
        """
        Generate contextual introduction based on ML context.
        
        Args:
            ml_context: ML context
            sentiment: User sentiment
            urgency: Urgency level
            language: Language code ('en' or 'tr')
            
        Returns:
            Bilingual introduction text
        """
        occasion = ml_context.get('occasion')
        budget = ml_context.get('budget')
        meal_time = ml_context.get('meal_time')
        
        # Base intro based on urgency
        if language == 'tr':
            if urgency == 'immediate':
                intro = "🍽️ Şu anda müsait olan harika restoranlar buldum"
            elif urgency == 'soon':
                intro = "🍽️ Yakında için mükemmel restoranlar buldum"
            else:
                intro = "🍽️ Sizin için harika restoranlar buldum"
        else:
            if urgency == 'immediate':
                intro = "🍽️ I found some great restaurants available right now"
            elif urgency == 'soon':
                intro = "🍽️ Here are some excellent restaurants for you soon"
            else:
                intro = "🍽️ I've found some wonderful restaurants for you"
        
        # Add occasion context
        if occasion == 'romantic':
            intro += " romantik bir akşam için mükemmel" if language == 'tr' else " perfect for a romantic evening"
        elif occasion == 'family':
            intro += " aileler için harika" if language == 'tr' else " great for families"
        elif occasion == 'business':
            intro += " iş toplantıları için ideal" if language == 'tr' else " ideal for business meetings"
        elif occasion == 'celebration':
            intro += " kutlamanız için mükemmel" if language == 'tr' else " perfect for your celebration"
        
        # Add meal time
        if meal_time:
            meal_translations = {
                'breakfast': ('kahvaltı', 'breakfast'),
                'brunch': ('brunch', 'brunch'),
                'lunch': ('öğle yemeği', 'lunch'),
                'dinner': ('akşam yemeği', 'dinner'),
                'late_night': ('gece geç', 'late night')
            }
            meal_tr, meal_en = meal_translations.get(meal_time, (meal_time, meal_time))
            meal_text = meal_tr if language == 'tr' else meal_en
            intro += f" {meal_text} için" if language == 'tr' else f" for {meal_text}"
        
        # Add budget descriptor
        if budget and budget != 'moderate':
            if language == 'tr':
                budget_desc = {
                    'cheap': 'ekonomik',
                    'expensive': 'üst düzey',
                    'very_expensive': 'lüks'
                }.get(budget, '')
            else:
                budget_desc = {
                    'cheap': 'budget-friendly',
                    'expensive': 'upscale',
                    'very_expensive': 'fine dining'
                }.get(budget, '')
            
            if budget_desc:
                intro += f" ({budget_desc})"
        
        intro += ":"
        return intro
    
    def _format_restaurant_list(
        self,
        restaurants: List[Dict[str, Any]],
        ml_context: Dict[str, Any],
        language: str = 'en'
    ) -> str:
        """
        Format restaurant list for response.
        
        Args:
            restaurants: List of restaurants
            ml_context: ML context
            language: Language code ('en' or 'tr')
            
        Returns:
            Formatted restaurant list (bilingual)
        """
        lines = []
        
        for i, restaurant in enumerate(restaurants, 1):
            name = restaurant.get('name', 'Unknown' if language == 'en' else 'Bilinmiyor')
            cuisine = restaurant.get('cuisine', 'Restaurant' if language == 'en' else 'Restoran')
            rating = restaurant.get('rating', 0)
            price_level = restaurant.get('price_level', 'moderate')
            ml_score = restaurant.get('ml_score', 0)
            
            # Price symbols
            price_symbols = {
                'cheap': '💵',
                'moderate': '💵💵',
                'expensive': '💵💵💵',
                'very_expensive': '💵💵💵💵'
            }
            price = price_symbols.get(price_level, '💵💵')
            
            line = f"\n{i}. **{name}** ({cuisine})\n"
            line += f"   ⭐ {rating}/5 | {price}"
            
            # Add ML match indicator (bilingual)
            if ml_score >= 0.8:
                match_text = "🎯 Mükemmel Eşleşme" if language == 'tr' else "🎯 Perfect Match"
                line += f" | {match_text}"
            elif ml_score >= 0.7:
                match_text = "✨ Harika Eşleşme" if language == 'tr' else "✨ Great Match"
                line += f" | {match_text}"
            
            # Add relevant highlights (bilingual)
            highlights = self._get_restaurant_highlights(restaurant, ml_context, language)
            if highlights:
                line += f"\n   {highlights}"
            
            lines.append(line)
        
        return ''.join(lines)
    
    def _get_restaurant_highlights(
        self,
        restaurant: Dict[str, Any],
        ml_context: Dict[str, Any],
        language: str = 'en'
    ) -> str:
        """
        Get relevant highlights for restaurant based on ML context.
        
        Args:
            restaurant: Restaurant data
            ml_context: ML context
            language: Language code ('en' or 'tr')
            
        Returns:
            Formatted highlights (bilingual)
        """
        highlights = []
        
        tags = restaurant.get('tags', [])
        dietary = ml_context.get('dietary_restrictions', [])
        occasion = ml_context.get('occasion')
        
        # Dietary matches (bilingual)
        for diet in dietary:
            if diet in restaurant.get('dietary_options', []):
                diet_label = diet.replace('_', ' ').title()
                if language == 'tr':
                    diet_translations = {
                        'Vegetarian': 'Vejetaryen',
                        'Vegan': 'Vegan',
                        'Halal': 'Helal',
                        'Gluten Free': 'Glutensiz',
                        'Kosher': 'Kaşer',
                        'Lactose Free': 'Laktozsuz'
                    }
                    diet_label = diet_translations.get(diet_label, diet_label)
                    highlights.append(f"{diet_label} mevcut")
                else:
                    highlights.append(f"{diet_label} available")
        
        # Occasion-relevant tags (bilingual)
        if occasion == 'romantic' and 'romantic' in tags:
            highlights.append("🌹 Romantik ortam" if language == 'tr' else "🌹 Romantic ambiance")
        elif occasion == 'family' and 'family-friendly' in tags:
            highlights.append("👨‍👩‍👧‍👦 Aile dostu" if language == 'tr' else "👨‍👩‍👧‍👦 Family-friendly")
        elif occasion == 'business' and 'quiet' in tags:
            highlights.append("💼 İş için harika" if language == 'tr' else "💼 Great for business")
        
        # Special features (bilingual)
        if 'bosphorus view' in tags:
            highlights.append("🌊 Boğaz manzarası" if language == 'tr' else "🌊 Bosphorus view")
        if 'terrace' in tags:
            highlights.append("🌳 Teras" if language == 'tr' else "🌳 Terrace seating")
        
        return ' • '.join(highlights[:3])  # Max 3 highlights
    
    def _generate_ml_tips(
        self,
        ml_context: Dict[str, Any],
        restaurants: List[Dict[str, Any]],
        language: str = 'en'
    ) -> str:
        """
        Generate ML-powered tips based on context.
        
        Args:
            ml_context: ML context
            restaurants: List of restaurants
            language: Language code ('en' or 'tr')
            
        Returns:
            Tips text (bilingual)
        """
        tips = []
        
        urgency = ml_context.get('urgency')
        occasion = ml_context.get('occasion')
        meal_time = ml_context.get('meal_time')
        
        # Urgency tips
        if urgency == 'immediate':
            tips.append(
                "Müsaitlik için önceden aramayı öneririm" if language == 'tr' 
                else "I recommend calling ahead for availability"
            )
        
        # Occasion tips
        if occasion == 'romantic':
            tips.append(
                "En iyi atmosfer için pencere kenarı masa isteyin" if language == 'tr'
                else "Consider requesting a table by the window for the best atmosphere"
            )
        elif occasion == 'business':
            tips.append(
                "Bu lokasyonlarda özel odalar mevcut" if language == 'tr'
                else "Private rooms available at most of these locations"
            )
        
        # Meal time tips
        if meal_time == 'dinner' and any(r.get('ml_score', 0) > 0.8 for r in restaurants):
            tips.append(
                "Akşam yemeği için rezervasyon önerilir" if language == 'tr'
                else "Reservations recommended for dinner time"
            )
        
        return ' • '.join(tips[:2])  # Max 2 tips
    
    def _generate_no_results_response(
        self,
        message: str,
        ml_context: Dict[str, Any],
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Generate response when no restaurants found.
        
        Args:
            message: User's message
            ml_context: ML context
            language: Language code ('en' or 'tr')
            
        Returns:
            Response dictionary (bilingual)
        """
        budget = ml_context.get('budget')
        dietary = ml_context.get('dietary_restrictions', [])
        
        if language == 'tr':
            response = "🍽️ Tüm kriterlerinize uyan restoran bulamadım."
            
            suggestions = []
            if budget in ['very_expensive']:
                suggestions.append("Pahalı restoranlara genişletmeyi deneyin")
            if dietary:
                suggestions.append("Esnek diyet seçenekleri olan restoranları değerlendirin")
            
            if suggestions:
                response += "\n\n💡 Öneriler:\n" + '\n'.join(f"• {s}" for s in suggestions)
            
            response += "\n\nKriterleri gevşeterek size restoranlar göstermemi ister misiniz?"
        else:
            response = "🍽️ I couldn't find restaurants matching all your criteria."
            
            suggestions = []
            if budget in ['very_expensive']:
                suggestions.append("Try expanding to expensive restaurants")
            if dietary:
                suggestions.append("Consider restaurants with flexible dietary options")
            
            if suggestions:
                response += "\n\n💡 Suggestions:\n" + '\n'.join(f"• {s}" for s in suggestions)
            
            response += "\n\nWould you like me to show you restaurants with relaxed criteria?"
        
        return {
            'response': response,
            'intent': 'restaurant',
            'restaurants': [],
            'ml_context': ml_context,
            'confidence': 0.7,
            'language': language
        }
    
    def _generate_fallback_response(self, message: str, language: str = 'en') -> Dict[str, Any]:
        """
        Generate fallback response on error.
        
        Args:
            message: User's message
            language: Language code ('en' or 'tr')
            
        Returns:
            Fallback response dictionary (bilingual)
        """
        if language == 'tr':
            response = (
                "🍽️ Şu anda restoran önerilerini işlerken sorun yaşıyorum. "
                "Lütfen isteğinizi yeniden ifade edebilir misiniz? Örneğin: "
                "'Romantik bir Türk restoranı arıyorum' veya 'Taksim yakınında ucuz vejetaryen yerler'"
            )
        else:
            response = (
                "🍽️ I'm having trouble processing restaurant recommendations right now. "
                "Could you please try rephrasing your request? For example: "
                "'I'm looking for a romantic Turkish restaurant' or 'cheap vegetarian places near Taksim'"
            )
        
        return {
            'response': response,
            'intent': 'restaurant',
            'restaurants': [],
            'confidence': 0.3,
            'language': language
        }


# Export
__all__ = ['RestaurantHandler']
