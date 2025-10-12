#!/usr/bin/env python3
"""
Hidden Gems & Local Tips System for Istanbul Daily Talk AI
Advanced personalized recommendations for local secrets and hidden places in Istanbul

Features:
- ML-based personalization
- GPS-based filtering
- Time-sensitive recommendations
- User feedback system
- Dynamic content updates
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import random
import math
import json

logger = logging.getLogger(__name__)

class TipCategory(Enum):
    """Categories for local tips and hidden gems"""
    LOCAL_EATERY = "local_eatery"
    HIDDEN_SPOT = "hidden_spot"
    CULTURAL_SECRET = "cultural_secret"
    LOCAL_MARKET = "local_market"
    ARTISAN_WORKSHOP = "artisan_workshop"
    SCENIC_VIEWPOINT = "scenic_viewpoint"
    NEIGHBORHOOD_LIFE = "neighborhood_life"
    INSIDER_KNOWLEDGE = "insider_knowledge"
    LOCAL_TRADITION = "local_tradition"
    OFF_BEATEN_PATH = "off_beaten_path"
    
    def __str__(self):
        """String representation for JSON serialization"""
        return self.value
    
    def __json__(self):
        """JSON serialization support"""
        return self.value

@dataclass
class LocalTip:
    """Individual local tip or piece of insider knowledge"""
    id: str
    title: str
    description: str
    category: TipCategory
    neighborhood: str
    tip_text: str
    
    # Location data
    location: Optional[Tuple[float, float]] = None  # (lat, lng)
    address: Optional[str] = None
    
    # Metadata
    difficulty_level: str = "easy"  # easy, moderate, adventurous
    time_required: str = "30min"  # time to experience/visit
    best_time: List[str] = field(default_factory=lambda: ["anytime"])  # morning, afternoon, evening, night
    seasonal_info: Optional[str] = None
    
    # Personalization factors
    interests: List[str] = field(default_factory=list)  # history, food, art, culture, etc.
    visitor_types: List[str] = field(default_factory=lambda: ["all"])  # solo, couple, family, group
    budget_range: str = "free"  # free, budget, moderate, expensive
    
    # Local context
    local_context: str = ""  # Why locals love this
    insider_knowledge: str = ""  # Special tips only locals know
    cultural_significance: str = ""  # Cultural or historical importance
    
    # Social proof
    rating: float = 4.0
    review_count: int = 0
    local_popularity: float = 0.8  # How popular among locals (0-1)
    tourist_awareness: float = 0.2  # How known to tourists (0-1)

@dataclass
class HiddenGem:
    """A hidden gem location with detailed information"""
    id: str
    name: str
    description: str
    category: TipCategory
    neighborhood: str
    
    # Location data
    location: Tuple[float, float]  # (lat, lng) - required for hidden gems
    address: str
    walking_directions: str = ""
    
    # Experience details
    what_makes_special: str = ""
    best_experience: str = ""
    local_tips: List[str] = field(default_factory=list)
    
    # Timing and accessibility
    opening_hours: Dict[str, str] = field(default_factory=dict)
    best_visit_times: List[str] = field(default_factory=list)
    avoid_times: List[str] = field(default_factory=list)
    
    # Requirements and preparation
    entry_requirements: List[str] = field(default_factory=list)
    what_to_bring: List[str] = field(default_factory=list)
    photography_allowed: bool = True
    
    # Personalization
    suitable_for: List[str] = field(default_factory=list)  # solo, romantic, family, etc.
    interests_match: List[str] = field(default_factory=list)  # history, photography, etc.
    difficulty_access: str = "easy"  # easy, moderate, challenging
    
    # Discovery metadata
    how_locals_found: str = ""
    discovery_story: str = ""
    alternative_similar: List[str] = field(default_factory=list)
    
    # Social and community data
    rating: float = 4.5
    local_rating: float = 4.8  # Separate rating from locals
    hidden_level: float = 0.9  # How hidden it is (0=well known, 1=very hidden)
    authenticity_score: float = 0.95  # How authentic/non-touristy

@dataclass
class Restaurant:
    """A restaurant with detailed local information and authentic characteristics"""
    id: str
    name: str
    description: str
    category: str  # "local_eatery", "street_food", "traditional", "modern_turkish", "seafood", "vegetarian"
    neighborhood: str
    
    # Location data
    location: Tuple[float, float]  # (lat, lng)
    address: str
    walking_directions: str = ""
    
    # Cuisine and menu details
    cuisine_type: str = "turkish"  # turkish, ottoman, regional, fusion
    specialty_dishes: List[str] = field(default_factory=list)
    must_try_items: List[str] = field(default_factory=list)
    dietary_options: List[str] = field(default_factory=list)  # vegetarian, vegan, halal, etc.
    
    # Timing and service
    opening_hours: Dict[str, str] = field(default_factory=dict)
    best_visit_times: List[str] = field(default_factory=list)
    avoid_times: List[str] = field(default_factory=list)
    average_meal_duration: str = "60min"
    reservation_needed: bool = False
    
    # Experience and atmosphere
    atmosphere: str = "casual"  # casual, traditional, upscale, family, romantic
    seating_style: List[str] = field(default_factory=list)  # floor_seating, outdoor, indoor, terrace
    noise_level: str = "moderate"  # quiet, moderate, lively, loud
    service_style: str = "table_service"  # self_service, table_service, counter
    
    # Pricing and value
    price_range: str = "budget"  # budget, moderate, expensive, luxury
    average_cost_per_person: str = "15-25 TL"
    payment_methods: List[str] = field(default_factory=lambda: ["cash", "card"])
    tipping_culture: str = "optional"
    
    # Local characteristics
    local_clientele_percentage: float = 0.8  # How much of clientele is local vs tourist
    family_run: bool = True
    established_year: Optional[int] = None
    generations_running: int = 1
    
    # Personalization and matching
    suitable_for: List[str] = field(default_factory=list)  # solo, couple, family, group, business
    occasion_types: List[str] = field(default_factory=list)  # casual, celebration, romantic, business
    interests_match: List[str] = field(default_factory=list)  # food, culture, traditional, modern
    
    # Authenticity and discovery
    how_locals_found: str = ""
    chef_story: str = ""
    signature_preparation: str = ""
    local_reputation: str = ""
    
    # Practical information
    languages_spoken: List[str] = field(default_factory=lambda: ["turkish"])
    english_menu_available: bool = False
    wifi_available: bool = False
    child_friendly: bool = True
    
    # Social and ratings
    rating: float = 4.5
    local_rating: float = 4.8  # Separate rating from locals
    authenticity_score: float = 0.9  # How authentic the experience is
    hidden_level: float = 0.7  # How hidden/unknown to tourists
    
    # Seasonal and time-sensitive info
    seasonal_specialties: Dict[str, List[str]] = field(default_factory=dict)  # season -> dishes
    best_seasons: List[str] = field(default_factory=list)

@dataclass
class Museum:
    """A museum with detailed cultural information and visitor experience data"""
    id: str
    name: str
    description: str
    category: str  # "archaeological", "art", "history", "cultural", "religious", "specialty", "house_museum"
    neighborhood: str
    
    # Location data
    location: Tuple[float, float]  # (lat, lng)
    address: str
    walking_directions: str = ""
    
    # Collection and content details
    primary_collection: str = ""  # Main focus of the museum
    notable_artifacts: List[str] = field(default_factory=list)
    must_see_exhibits: List[str] = field(default_factory=list)
    special_collections: List[str] = field(default_factory=list)
    period_focus: List[str] = field(default_factory=list)  # byzantine, ottoman, modern, etc.
    
    # Timing and logistics
    opening_hours: Dict[str, str] = field(default_factory=dict)
    best_visit_times: List[str] = field(default_factory=list)
    avoid_times: List[str] = field(default_factory=list)
    recommended_visit_duration: str = "90min"
    last_entry_time: str = ""
    
    # Experience and services
    guided_tours_available: bool = True
    audio_guide_languages: List[str] = field(default_factory=lambda: ["turkish", "english"])
    photography_policy: str = "allowed_no_flash"  # allowed, not_allowed, allowed_no_flash, fee_required
    interactive_exhibits: bool = False
    accessibility_features: List[str] = field(default_factory=list)  # wheelchair, elevator, audio_description
    
    # Pricing and access
    entrance_fee: str = "20 TL"
    student_discount: bool = True
    senior_discount: bool = True
    free_days: List[str] = field(default_factory=list)  # days of week or specific dates
    museum_pass_accepted: bool = True
    
    # Cultural and educational value
    historical_significance: str = ""
    cultural_importance: str = ""
    educational_programs: List[str] = field(default_factory=list)
    target_age_groups: List[str] = field(default_factory=list)  # children, teens, adults, seniors
    
    # Building and architecture
    building_history: str = ""
    architectural_style: str = ""
    building_significance: bool = False  # Is the building itself historically significant
    restoration_info: str = ""
    
    # Personalization and matching
    suitable_for: List[str] = field(default_factory=list)  # solo, couple, family, educational, research
    interests_match: List[str] = field(default_factory=list)  # history, art, archaeology, culture, religion
    expertise_level: List[str] = field(default_factory=list)  # beginner, intermediate, expert
    
    # Visitor experience
    crowd_levels: Dict[str, str] = field(default_factory=dict)  # time -> crowd level
    visitor_flow: str = "moderate"  # light, moderate, heavy
    peaceful_sections: List[str] = field(default_factory=list)  # Quieter areas in the museum
    
    # Curatorial and scholarly aspects
    current_exhibitions: List[str] = field(default_factory=list)
    research_facilities: bool = False
    library_access: bool = False
    scholar_resources: List[str] = field(default_factory=list)
    
    # Local context and discovery
    how_locals_view: str = ""
    curator_insights: str = ""
    hidden_sections: List[str] = field(default_factory=list)  # Less known areas
    local_visitor_tips: List[str] = field(default_factory=list)
    
    # Practical information
    gift_shop: bool = True
    cafe_restaurant: bool = False
    storage_lockers: bool = True
    languages_supported: List[str] = field(default_factory=lambda: ["turkish", "english"])
    
    # Social and ratings
    rating: float = 4.5
    educational_rating: float = 4.3  # How educational visitors find it
    cultural_authenticity: float = 0.9  # How authentic the cultural representation is
    tourist_vs_local_ratio: float = 0.6  # Higher = more tourists, lower = more locals
    
    # Seasonal and temporal aspects
    seasonal_exhibitions: Dict[str, List[str]] = field(default_factory=dict)  # season -> exhibitions
    best_seasons: List[str] = field(default_factory=list)
    weather_dependent: bool = False  # For museums with outdoor sections

class HiddenGemsLocalTips:
    """
    Advanced Hidden Gems and Local Tips System
    Provides personalized, context-aware recommendations for Istanbul's hidden treasures
    """
    
    def __init__(self):
        """Initialize the hidden gems system with comprehensive data"""
        self.tips_database = self._initialize_tips_database()
        self.gems_database = self._initialize_gems_database()
        self.restaurants_database = self._initialize_restaurants_database()
        self.museums_database = self._initialize_museums_database()
        self.user_preferences = {}
        self.neighborhood_data = self._initialize_neighborhood_data()
        
        # Initialize structured database
        self.database = {
            'gems': {gem.id: gem for gem in self.gems_database},
            'tips': {tip.id: tip for tip in self.tips_database},
            'restaurants': {restaurant.id: restaurant for restaurant in self.restaurants_database},
            'museums': {museum.id: museum for museum in self.museums_database}
        }
        
        # Initialize user profiles for ML features
        self.user_profiles = {}
        
        # ML and personalization components
        self.user_interaction_history = {}
        self.recommendation_weights = {
            'location_proximity': 0.3,
            'user_interests': 0.25,
            'time_context': 0.2,
            'local_popularity': 0.15,
            'authenticity': 0.1
        }
        
        logger.info("Hidden Gems & Local Tips system initialized")
    
    def process_hidden_gems_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main method to process hidden gems queries with advanced personalization
        
        Args:
            query: User's query about hidden gems or local tips
            user_context: Optional context including location, preferences, etc.
            
        Returns:
            Comprehensive response with personalized recommendations
        """
        try:
            # Parse query and extract intent
            query_analysis = self._analyze_query(query)
            
            # Get user context and preferences
            context = user_context or {}
            user_location = context.get('location')
            user_interests = context.get('interests', [])
            time_context = context.get('time_context', self._get_current_time_context())
            
            # Generate personalized recommendations
            recommendations = self._generate_recommendations(
                query_analysis, user_location, user_interests, time_context
            )
            
            # Format comprehensive response
            response = self._format_response(recommendations, query_analysis, context)
            
            # Log interaction for learning
            self._log_interaction(query, context, recommendations)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing hidden gems query: {e}")
            return self._get_fallback_response(query)
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze user query to understand intent and preferences"""
        query_lower = query.lower()
        
        analysis = {
            'categories': [],
            'neighborhoods': [],
            'interests': [],
            'time_preference': None,
            'budget_preference': None,
            'group_type': 'solo',
            'experience_type': 'discovery'
        }
        
        # Category detection
        category_keywords = {
            TipCategory.LOCAL_EATERY: ['food', 'eat', 'restaurant', 'cafe', 'meal', 'local food', 'street food'],
            TipCategory.HIDDEN_SPOT: ['hidden', 'secret', 'unknown', 'discover', 'explore'],
            TipCategory.CULTURAL_SECRET: ['culture', 'tradition', 'history', 'heritage', 'authentic'],
            TipCategory.LOCAL_MARKET: ['market', 'bazaar', 'shopping', 'local products', 'artisan'],
            TipCategory.SCENIC_VIEWPOINT: ['view', 'sunset', 'panoramic', 'lookout', 'scenic', 'photo'],
            TipCategory.NEIGHBORHOOD_LIFE: ['neighborhood', 'local life', 'daily life', 'community'],
            TipCategory.ARTISAN_WORKSHOP: ['workshop', 'artisan', 'craft', 'traditional', 'handmade'],
            TipCategory.OFF_BEATEN_PATH: ['off beaten path', 'alternative', 'unusual', 'different']
        }
        
        # Restaurant and Museum specific keywords
        restaurant_keywords = ['restaurant', 'dining', 'eat', 'food', 'cuisine', 'meal', 'lunch', 'dinner', 'breakfast', 'kebab', 'turkish food', 'local food', 'authentic food']
        museum_keywords = ['museum', 'gallery', 'exhibition', 'art', 'history', 'cultural', 'artifacts', 'collection', 'palace', 'archaeological']
        
        # Check for specific POI types
        analysis['poi_types'] = []
        if any(keyword in query_lower for keyword in restaurant_keywords):
            analysis['poi_types'].append('restaurant')
        if any(keyword in query_lower for keyword in museum_keywords):
            analysis['poi_types'].append('museum')
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis['categories'].append(category)
        
        # Neighborhood detection
        istanbul_neighborhoods = [
            'sultanahmet', 'beyoglu', 'galata', 'karakoy', 'besiktas', 'ortakoy',
            'kadikoy', 'uskudar', 'fatih', 'eminonu', 'bakirkoy', 'sisli',
            'taksim', 'nisantasi', 'cihangir', 'balat', 'fener', 'kuzguncuk'
        ]
        
        for neighborhood in istanbul_neighborhoods:
            if neighborhood in query_lower:
                analysis['neighborhoods'].append(neighborhood)
        
        # Interest detection
        interest_keywords = {
            'history': ['history', 'historical', 'ancient', 'byzantine', 'ottoman', 'imperial', 'heritage'],
            'art': ['art', 'gallery', 'artistic', 'creative', 'painting', 'sculpture', 'calligraphy'],
            'food': ['food', 'culinary', 'taste', 'eat', 'drink', 'cuisine', 'dining', 'gastronomy'],
            'photography': ['photo', 'instagram', 'picture', 'shoot', 'photography'],
            'architecture': ['architecture', 'building', 'mosque', 'church', 'palace', 'architectural'],
            'nature': ['nature', 'park', 'garden', 'outdoor', 'green'],
            'nightlife': ['night', 'evening', 'bar', 'music', 'entertainment'],
            'culture': ['culture', 'cultural', 'tradition', 'traditional', 'authentic', 'local'],
            'museums': ['museum', 'exhibition', 'collection', 'artifacts', 'archaeological'],
            'restaurants': ['restaurant', 'eatery', 'dining', 'meal', 'local food', 'street food'],
            'family': ['family', 'kids', 'children', 'child-friendly'],
            'romantic': ['romantic', 'couple', 'date', 'intimate'],
            'educational': ['educational', 'learning', 'research', 'academic'],
            'hidden': ['hidden', 'secret', 'unknown', 'off-beaten', 'discover']
        }
        
        for interest, keywords in interest_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis['interests'].append(interest)
        
        # Time preference detection
        time_keywords = {
            'morning': ['morning', 'breakfast', 'early'],
            'afternoon': ['afternoon', 'lunch', 'midday'],
            'evening': ['evening', 'sunset', 'dinner'],
            'night': ['night', 'late', 'after dark']
        }
        
        for time_pref, keywords in time_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis['time_preference'] = time_pref
                break
        
        # Group type detection
        if any(word in query_lower for word in ['couple', 'romantic', 'date']):
            analysis['group_type'] = 'couple'
        elif any(word in query_lower for word in ['family', 'kids', 'children']):
            analysis['group_type'] = 'family'
        elif any(word in query_lower for word in ['friends', 'group']):
            analysis['group_type'] = 'group'
        
        return analysis
    
    def _generate_recommendations(self, query_analysis: Dict[str, Any], 
                                user_location: Optional[Tuple[float, float]],
                                user_interests: List[str], 
                                time_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on analysis and context"""
        
        # Combine tips, gems and restaurants for unified scoring
        all_recommendations = []
        
        # Process local tips
        for tip in self.tips_database:
            score = self._calculate_recommendation_score(
                tip, query_analysis, user_location, user_interests, time_context, 'tip'
            )
            if score > 0.3:  # Minimum relevance threshold
                all_recommendations.append({
                    'type': 'tip',
                    'item': tip,
                    'score': score,
                    'match_reasons': self._get_match_reasons(tip, query_analysis)
                })
        
        # Process hidden gems
        for gem in self.gems_database:
            score = self._calculate_recommendation_score(
                gem, query_analysis, user_location, user_interests, time_context, 'gem'
            )
            if score > 0.3:  # Minimum relevance threshold
                all_recommendations.append({
                    'type': 'gem',
                    'item': gem,
                    'score': score,
                    'match_reasons': self._get_match_reasons(gem, query_analysis)
                })
        
        # Process restaurants
        for restaurant in self.restaurants_database:
            score = self._calculate_recommendation_score(
                restaurant, query_analysis, user_location, user_interests, time_context, 'restaurant'
            )
            if score > 0.3:  # Minimum relevance threshold
                all_recommendations.append({
                    'type': 'restaurant',
                    'item': restaurant,
                    'score': score,
                    'match_reasons': self._get_match_reasons(restaurant, query_analysis)
                })
        
        # Process museums
        for museum in self.museums_database:
            score = self._calculate_recommendation_score(
                museum, query_analysis, user_location, user_interests, time_context, 'museum'
            )
            if score > 0.3:  # Minimum relevance threshold
                all_recommendations.append({
                    'type': 'museum',
                    'item': museum,
                    'score': score,
                    'match_reasons': self._get_match_reasons(museum, query_analysis)
                })
        
        # Sort by score and return top recommendations
        all_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return all_recommendations[:8]  # Return top 8 recommendations
    
    def _calculate_recommendation_score(self, item: Any, query_analysis: Dict[str, Any],
                                      user_location: Optional[Tuple[float, float]],
                                      user_interests: List[str], 
                                      time_context: Dict[str, Any],
                                      item_type: str) -> float:
        """Calculate personalized recommendation score using multiple factors"""
        
        score = 0.0
        
        # POI type matching (for restaurants and museums)
        poi_types = query_analysis.get('poi_types', [])
        if item_type in poi_types:
            score += 0.4
        
        # Category matching (for tips and gems)
        if hasattr(item, 'category') and item.category in query_analysis['categories']:
            score += 0.3
        
        # Restaurant category matching
        if item_type == 'restaurant' and hasattr(item, 'category'):
            restaurant_categories = ['local_eatery', 'traditional', 'authentic', 'turkish']
            if any(cat in query_analysis.get('interests', []) for cat in restaurant_categories):
                if item.category in ['traditional', 'local_eatery']:
                    score += 0.3
        
        # Museum category matching  
        if item_type == 'museum' and hasattr(item, 'category'):
            museum_categories = ['history', 'art', 'cultural', 'archaeological']
            if any(cat in query_analysis.get('interests', []) for cat in museum_categories):
                if item.category in ['history', 'art', 'cultural', 'archaeological']:
                    score += 0.3
        
        # Neighborhood matching
        if hasattr(item, 'neighborhood') and item.neighborhood.lower() in [n.lower() for n in query_analysis['neighborhoods']]:
            score += 0.25
        
        # Interest matching
        query_interests = set(query_analysis.get('interests', []))
        
        # Get item interests based on type
        if item_type == 'tip':
            item_interests = getattr(item, 'interests', [])
        elif item_type in ['gem']:
            item_interests = getattr(item, 'interests_match', [])
        elif item_type == 'restaurant':
            # Combine different interest fields for restaurants
            item_interests = (getattr(item, 'interests_match', []) + 
                            [item.cuisine_type] + 
                            [item.category] +
                            ['food', 'dining', 'authentic'] if item.authenticity_score > 0.8 else [])
        elif item_type == 'museum':
            # Combine different interest fields for museums
            item_interests = (getattr(item, 'interests_match', []) + 
                            [item.category] +
                            getattr(item, 'period_focus', []) +
                            ['culture', 'education'] if hasattr(item, 'educational_rating') else [])
        else:
            item_interests = []
        
        # Calculate interest overlap
        if item_interests and query_interests:
            interest_overlap = len(query_interests.intersection(set(item_interests)))
            if interest_overlap > 0:
                score += min(0.3, interest_overlap * 0.1)
        
        # Location proximity (if available)
        if user_location and hasattr(item, 'location') and item.location:
            distance = self._calculate_distance(user_location, item.location)
            if distance < 5:  # Within 5km
                proximity_score = max(0, (5 - distance) / 5) * 0.15
                score += proximity_score
        
        # Time context matching
        if hasattr(item, 'best_time') and query_analysis['time_preference']:
            if query_analysis['time_preference'] in item.best_time:
                score += 0.1
        
        # Group type matching
        if hasattr(item, 'visitor_types'):
            if query_analysis['group_type'] in item.visitor_types or 'all' in item.visitor_types:
                score += 0.1
        elif hasattr(item, 'suitable_for'):
            if query_analysis['group_type'] in item.suitable_for:
                score += 0.1
        
        # Authenticity and local appeal (for hidden gems)
        if item_type == 'gem':
            authenticity_bonus = getattr(item, 'authenticity_score', 0.5) * 0.15
            hidden_bonus = getattr(item, 'hidden_level', 0.5) * 0.1
            score += authenticity_bonus + hidden_bonus
        
        # Local popularity vs tourist awareness balance
        local_pop = getattr(item, 'local_popularity', 0.5)
        tourist_aware = getattr(item, 'tourist_awareness', 0.5)
        local_authenticity = (local_pop * 2 - tourist_aware) / 2  # Favor local popularity
        score += max(0, local_authenticity) * 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_match_reasons(self, item: Any, query_analysis: Dict[str, Any]) -> List[str]:
        """Get reasons why this item matches the user's query"""
        reasons = []
        
        if hasattr(item, 'category') and item.category in query_analysis['categories']:
            reasons.append(f"Matches your interest in {item.category.value.replace('_', ' ')}")
        
        if hasattr(item, 'neighborhood') and item.neighborhood.lower() in [n.lower() for n in query_analysis['neighborhoods']]:
            reasons.append(f"Located in {item.neighborhood}")
        
        if query_analysis['interests']:
            item_interests = getattr(item, 'interests', []) or getattr(item, 'interests_match', [])
            common_interests = set(query_analysis['interests']).intersection(set(item_interests))
            if common_interests:
                reasons.append(f"Perfect for {', '.join(common_interests)} enthusiasts")
        
        return reasons
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two GPS coordinates in kilometers"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Haversine formula
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _get_current_time_context(self) -> Dict[str, Any]:
        """Get current time context for time-sensitive recommendations"""
        now = datetime.now()
        hour = now.hour
        
        time_period = "morning"
        if 12 <= hour < 17:
            time_period = "afternoon"
        elif 17 <= hour < 21:
            time_period = "evening"
        elif hour >= 21 or hour < 6:
            time_period = "night"
        
        return {
            'time_period': time_period,
            'day_of_week': now.strftime('%A').lower(),
            'season': self._get_season(now.month),
            'hour': hour
        }
    
    def get_seasonal_recommendations(self, season: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Get season-specific recommendations for hidden gems and tips
        
        Args:
            season: Specific season (winter, spring, summer, autumn) or None for current
            
        Returns:
            Season-specific recommendations and tips
        """
        try:
            target_season = season or self._get_season(datetime.now().month)
            
            seasonal_gems = []
            seasonal_tips = []
            
            # Filter gems and tips suitable for the season
            for gem in self.gems_database:
                if self._is_suitable_for_season(gem, target_season):
                    seasonal_gems.append(gem)
            
            for tip in self.tips_database:
                if self._is_tip_suitable_for_season(tip, target_season):
                    seasonal_tips.append(tip)
            
            # Get season-specific insights
            seasonal_insights = self._get_seasonal_insights(target_season)
            
            return {
                'season': target_season,
                'recommended_gems': seasonal_gems[:limit//2] if limit > 1 else seasonal_gems[:1],
                'seasonal_tips': seasonal_tips[:limit//2] if limit > 1 else seasonal_tips[:1],
                'seasonal_insights': seasonal_insights,
                'weather_considerations': self._get_weather_considerations(target_season),
                'special_events': self._get_seasonal_events(target_season)
            }
            
        except Exception as e:
            logger.error(f"Error getting seasonal recommendations: {e}")
            return {'season': target_season, 'error': 'Unable to load seasonal recommendations'}
    
    def _is_suitable_for_season(self, gem: HiddenGem, season: str) -> bool:
        """Check if a gem is suitable for the given season"""
        try:
            season_lower = season.lower()
            
            # Check if season is in avoid times
            if season_lower in str(gem.avoid_times).lower():
                return False
            
            # Check if it's specifically good for this season
            if season_lower in str(gem.best_visit_times).lower():
                return True
            
            # Season-specific logic
            if season_lower == "summer":
                return any(keyword in gem.description.lower() + gem.what_makes_special.lower() 
                          for keyword in ["rooftop", "water", "outdoor", "bosphorus", "garden", "terrace"])
            elif season_lower == "winter":
                return any(keyword in gem.description.lower() + gem.what_makes_special.lower()
                          for keyword in ["underground", "indoor", "library", "workshop", "hamam", "cultural"])
            elif season_lower == "spring":
                return any(keyword in gem.description.lower() + gem.what_makes_special.lower()
                          for keyword in ["garden", "walking", "courtyard", "neighborhood", "peaceful"])
            elif season_lower == "autumn":
                return any(keyword in gem.description.lower() + gem.what_makes_special.lower()
                          for keyword in ["cultural", "historic", "museum", "artisan", "traditional"])
            
            return True  # Default to suitable if no specific restrictions
            
        except Exception as e:
            logger.error(f"Error checking season suitability: {e}")
            return True
    
    def _is_tip_suitable_for_season(self, tip: LocalTip, season: str) -> bool:
        """Check if a tip is suitable for the given season"""
        try:
            # Check if tip has specific seasonal info
            if hasattr(tip, 'seasonal_info') and tip.seasonal_info:
                return season.lower() in tip.seasonal_info.lower()
            
            # Season-specific logic based on tip content
            season_lower = season.lower()
            tip_content = (tip.description + " " + tip.tip_text).lower()
            
            if season_lower == "summer":
                return any(keyword in tip_content for keyword in 
                          ["outdoor", "rooftop", "water", "garden", "terrace", "fishing"])
            elif season_lower == "winter":
                return any(keyword in tip_content for keyword in 
                          ["indoor", "warm", "tea", "hamam", "market", "workshop"])
            elif season_lower == "spring":
                return any(keyword in tip_content for keyword in 
                          ["walk", "garden", "fresh", "morning", "neighborhood"])
            elif season_lower == "autumn":
                return any(keyword in tip_content for keyword in 
                          ["cultural", "tradition", "spice", "craft", "artisan"])
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking tip season suitability: {e}")
            return True
    
    def _get_seasonal_insights(self, season: str) -> Dict[str, str]:
        """Get insights specific to the season"""
        insights = {
            "spring": {
                "best_activities": "Perfect time for walking tours, garden visits, and outdoor exploration",
                "local_behavior": "Locals emerge from winter hibernation, cafes open terraces",
                "special_note": "Tulip season brings extra beauty to parks and gardens"
            },
            "summer": {
                "best_activities": "Early morning and evening activities, rooftop experiences, water-related spots",
                "local_behavior": "Locals avoid midday heat, gather in waterfront areas in evenings",
                "special_note": "Many locals leave the city in August - quieter but some places may be closed"
            },
            "autumn": {
                "best_activities": "Cultural activities, indoor workshops, traditional crafts learning",
                "local_behavior": "Peak season for cultural events and traditional gatherings",
                "special_note": "Perfect weather for walking and exploring neighborhoods"
            },
            "winter": {
                "best_activities": "Indoor cultural experiences, underground locations, traditional baths",
                "local_behavior": "Locals spend more time in indoor social spaces like tea houses",
                "special_note": "Fewer tourists mean more authentic interactions with locals"
            }
        }
        
        return insights.get(season.lower(), insights["spring"])
    
    def _get_weather_considerations(self, season: str) -> List[str]:
        """Get weather-related considerations for the season"""
        weather_info = {
            "spring": [
                "Mild temperatures, occasional rain showers",
                "Perfect for walking and outdoor activities",
                "Bring light layers for temperature changes"
            ],
            "summer": [
                "Hot and humid, especially in July-August",
                "Best activities in early morning or evening",
                "Stay hydrated and seek shade during midday"
            ],
            "autumn": [
                "Comfortable temperatures, occasional rain",
                "Ideal weather for all activities",
                "Light jacket recommended for evenings"
            ],
            "winter": [
                "Cool and wet, occasional snow",
                "Indoor activities most comfortable",
                "Warm clothing and waterproof shoes recommended"
            ]
        }
        
        return weather_info.get(season.lower(), weather_info["spring"])
    
    def _get_seasonal_events(self, season: str) -> List[str]:
        """Get special events and festivals for the season"""
        events = {
            "spring": [
                "Istanbul Tulip Festival (April)",
                "Orthodox Easter celebrations",
                "Ramadan observances (dates vary)"
            ],
            "summer": [
                "Istanbul Music Festival (June-July)",
                "Bosphorus swimming events",
                "Outdoor cinema screenings"
            ],
            "autumn": [
                "Istanbul Biennial (odd years)",
                "Traditional craft fairs",
                "Harvest celebrations in neighborhoods"
            ],
            "winter": [
                "New Year celebrations",
                "Traditional storytelling evenings",
                "Indoor cultural performances"
            ]
        }
        
        return events.get(season.lower(), [])
    
    def _get_season(self, month: int) -> str:
        """Get season from month number"""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"
    
    def get_neighborhood_deep_dive(self, neighborhood: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a specific neighborhood's hidden gems
        
        Args:
            neighborhood: Name of the Istanbul neighborhood
            
        Returns:
            Deep dive information about the neighborhood
        """
        try:
            neighborhood_lower = neighborhood.lower()
            
            # Find all gems and tips in this neighborhood
            neighborhood_gems = [gem for gem in self.gems_database 
                               if gem.neighborhood.lower() == neighborhood_lower]
            neighborhood_tips = [tip for tip in self.tips_database 
                               if tip.neighborhood.lower() == neighborhood_lower]
            
            # Get neighborhood data
            neighborhood_info = self.neighborhood_data.get(neighborhood_lower, {})
            
            # Calculate neighborhood authenticity score
            authenticity_score = self._calculate_neighborhood_authenticity(neighborhood_gems, neighborhood_tips)
            
            # Generate walking route suggestions
            walking_routes = self._generate_walking_routes(neighborhood_gems, neighborhood_tips)
            
            return {
                'neighborhood': neighborhood,
                'character': neighborhood_info.get('character', 'unique_local'),
                'authenticity_score': authenticity_score,
                'hidden_gems': neighborhood_gems,
                'local_tips': neighborhood_tips,
                'insider_secrets': neighborhood_info.get('local_secrets', []),
                'best_times_to_visit': neighborhood_info.get('best_times', ['anytime']),
                'walking_routes': walking_routes,
                'local_wisdom': neighborhood_info.get('insider_tip', ''),
                'recommended_duration': self._calculate_recommended_duration(neighborhood_gems, neighborhood_tips),
                'difficulty_level': self._assess_neighborhood_difficulty(neighborhood_gems, neighborhood_tips)
            }
            
        except Exception as e:
            logger.error(f"Error getting neighborhood deep dive: {e}")
            return {'neighborhood': neighborhood, 'error': 'Unable to load neighborhood information'}
    
    def _calculate_neighborhood_authenticity(self, gems: List[HiddenGem], tips: List[LocalTip]) -> float:
        """Calculate overall authenticity score for a neighborhood"""
        try:
            if not gems and not tips:
                return 0.5
            
            total_score = 0.0
            count = 0
            
            # Average gem authenticity scores
            for gem in gems:
                total_score += gem.authenticity_score
                count += 1
            
            # Average tip local popularity (as proxy for authenticity)
            for tip in tips:
                total_score += tip.local_popularity
                count += 1
            
            return total_score / count if count > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating neighborhood authenticity: {e}")
            return 0.5
    
    def _generate_walking_routes(self, gems: List[HiddenGem], tips: List[LocalTip]) -> List[Dict[str, Any]]:
        """Generate suggested walking routes through the neighborhood"""
        try:
            routes = []
            
            if len(gems) >= 2:
                # Create a route connecting gems
                route = {
                    'name': 'Hidden Gems Trail',
                    'duration': '2-3 hours',
                    'difficulty': 'moderate',
                    'stops': [{'name': gem.name, 'type': 'gem', 'location': gem.location} for gem in gems[:4]],
                    'description': 'A walking route connecting the neighborhood\'s most authentic hidden gems'
                }
                routes.append(route)
            
            if len(tips) >= 3:
                # Create a local life route based on tips
                local_tips = [tip for tip in tips if tip.category in [TipCategory.NEIGHBORHOOD_LIFE, TipCategory.LOCAL_EATERY]]
                if local_tips:
                    route = {
                        'name': 'Local Life Experience',
                        'duration': '3-4 hours',
                        'difficulty': 'easy',
                        'stops': [{'name': tip.title, 'type': 'tip', 'location': getattr(tip, 'location', None)} for tip in local_tips[:3]],
                        'description': 'Experience authentic neighborhood life through local establishments and traditions'
                    }
                    routes.append(route)
            
            return routes
            
        except Exception as e:
            logger.error(f"Error generating walking routes: {e}")
            return []
    
    def _calculate_recommended_duration(self, gems: List[HiddenGem], tips: List[LocalTip]) -> str:
        """Calculate recommended time to spend in the neighborhood"""
        try:
            total_items = len(gems) + len(tips)
            
            if total_items >= 6:
                return "Full day (6-8 hours)"
            elif total_items >= 4:
                return "Half day (3-4 hours)"
            elif total_items >= 2:
                return "Morning or afternoon (2-3 hours)"
            else:
                return "1-2 hours"
                
        except Exception as e:
            logger.error(f"Error calculating recommended duration: {e}")
            return "2-3 hours"
    
    def _assess_neighborhood_difficulty(self, gems: List[HiddenGem], tips: List[LocalTip]) -> str:
        """Assess overall difficulty level for exploring the neighborhood"""
        try:
            if not gems:
                return "easy"
            
            difficulty_scores = {"easy": 1, "moderate": 2, "challenging": 3, "adventurous": 4}
            total_score = 0
            count = 0
            
            for gem in gems:
                score = difficulty_scores.get(gem.difficulty_access, 2)
                total_score += score
                count += 1
            
            avg_score = total_score / count if count > 0 else 2
            
            if avg_score <= 1.5:
                return "easy"
            elif avg_score <= 2.5:
                return "moderate"
            elif avg_score <= 3.5:
                return "challenging"
            else:
                return "adventurous"
                
        except Exception as e:
            logger.error(f"Error assessing neighborhood difficulty: {e}")
            return "moderate"
    
    def get_similar_recommendations(self, item_id: str, item_type: str = "auto") -> List[Dict[str, Any]]:
        """
        Get recommendations similar to a specific gem or tip
        
        Args:
            item_id: ID of the reference item
            item_type: Type of item ('gem', 'tip', or 'auto' to detect)
            
        Returns:
            List of similar recommendations
        """
        try:
            # Find the reference item
            reference_item = None
            actual_type = item_type
            
            if item_type == "auto" or item_type == "gem":
                reference_item = next((gem for gem in self.gems_database if gem.id == item_id), None)
                if reference_item:
                    actual_type = "gem"
            
            if not reference_item and (item_type == "auto" or item_type == "tip"):
                reference_item = next((tip for tip in self.tips_database if tip.id == item_id), None)
                if reference_item:
                    actual_type = "tip"
            
            if not reference_item:
                return []
            
            # Find similar items based on multiple factors
            similar_items = []
            
            # Compare with all items
            all_items = []
            if actual_type == "gem":
                all_items = [(gem, "gem") for gem in self.gems_database if gem.id != item_id]
            else:
                all_items = [(tip, "tip") for tip in self.tips_database if tip.id != item_id]
            
            for item, item_type_current in all_items:
                similarity_score = self._calculate_similarity_score(reference_item, item, actual_type, item_type_current)
                if similarity_score > 0.3:  # Only include reasonably similar items
                    similar_items.append({
                        'item': item,
                        'type': item_type_current,
                        'similarity_score': similarity_score,
                        'similarity_reasons': self._get_similarity_reasons(reference_item, item, actual_type, item_type_current)
                    })
            
            # Sort by similarity score
            similar_items.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similar_items[:6]  # Return top 6 similar items
            
        except Exception as e:
            logger.error(f"Error getting similar recommendations: {e}")
            return []
    
    def _calculate_similarity_score(self, item1: Any, item2: Any, type1: str, type2: str) -> float:
        """Calculate similarity score between two items"""
        try:
            score = 0.0
            
            # Same type bonus
            if type1 == type2:
                score += 0.2
            
            # Neighborhood similarity
            if hasattr(item1, 'neighborhood') and hasattr(item2, 'neighborhood'):
                if item1.neighborhood.lower() == item2.neighborhood.lower():
                    score += 0.3
            
            # Category similarity (for gems)
            if type1 == "gem" and type2 == "gem":
                if item1.category == item2.category:
                    score += 0.2
            
            # Category similarity (for tips)
            if type1 == "tip" and type2 == "tip":
                if item1.category == item2.category:
                    score += 0.2
            
            # Interest matching
            interests1 = getattr(item1, 'interests_match', getattr(item1, 'interests', []))
            interests2 = getattr(item2, 'interests_match', getattr(item2, 'interests', []))
            
            if interests1 and interests2:
                common_interests = set(interests1) & set(interests2)
                interest_score = len(common_interests) / max(len(set(interests1) | set(interests2)), 1)
                score += interest_score * 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating similarity score: {e}")
            return 0.0
    
    def _get_similarity_reasons(self, item1: Any, item2: Any, type1: str, type2: str) -> List[str]:
        """Get reasons why two items are similar"""
        try:
            reasons = []
            
            # Same neighborhood
            if hasattr(item1, 'neighborhood') and hasattr(item2, 'neighborhood'):
                if item1.neighborhood.lower() == item2.neighborhood.lower():
                    reasons.append(f"Both in {item1.neighborhood}")
            
            # Same category
            if hasattr(item1, 'category') and hasattr(item2, 'category'):
                if item1.category == item2.category:
                    reasons.append(f"Both are {item1.category.value}")
            
            # Common interests
            interests1 = getattr(item1, 'interests_match', getattr(item1, 'interests', []))
            interests2 = getattr(item2, 'interests_match', getattr(item2, 'interests', []))
            
            if interests1 and interests2:
                common_interests = list(set(interests1) & set(interests2))
                if common_interests:
                    reasons.append(f"Shared interests: {', '.join(common_interests[:2])}")
            
            # Similar difficulty (for gems)
            if type1 == "gem" and type2 == "gem":
                if item1.difficulty_access == item2.difficulty_access:
                    reasons.append(f"Similar difficulty level")
            
            if not reasons:
                reasons.append("Similar authentic experience")
            
            return reasons[:3]  # Return top 3 reasons
            
        except Exception as e:
            logger.error(f"Error getting similarity reasons: {e}")
            return ["Similar experience"]
    
    def _update_user_preferences_from_feedback(self, user_id: str, feedback_entry: Dict[str, Any]) -> None:
        """Update user preferences based on feedback"""
        try:
            # This would implement machine learning logic to update preferences
            # For now, we'll do basic preference learning
            
            if feedback_entry['rating'] and feedback_entry['rating'] >= 4:
                # User liked this recommendation, learn from it
                # This is a simplified version - in practice, you'd analyze the item characteristics
                pass
            
            logger.info(f"Updated preferences for user {user_id} based on feedback")
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
    
    def _initialize_neighborhood_data(self) -> Dict[str, Dict[str, Any]]:
        """Initialize neighborhood-specific data and characteristics"""
        return {
            'sultanahmet': {
                'character': 'historical_heart',
                'local_secrets': ['Hidden courtyards in traditional Ottoman houses', 'Local breakfast spots away from tourist areas'],
                'best_times': ['early_morning', 'evening'],
                'insider_tip': 'Many historical buildings have hidden gardens accessible through small donations'
            },
            'beyoglu': {
                'character': 'artistic_modern',
                'local_secrets': ['Underground art galleries', 'Rooftop terraces with Bosphorus views'],
                'best_times': ['afternoon', 'evening', 'night'],
                'insider_tip': 'The back streets hide the most authentic meyhanes (taverns)'
            },
            'kadikoy': {
                'character': 'bohemian_local',
                'local_secrets': ['Alternative culture hubs', 'Local designer workshops'],
                'best_times': ['afternoon', 'evening'],
                'insider_tip': 'Tuesday market day reveals the real neighborhood spirit'
            },
            'galata': {
                'character': 'historic_artistic',
                'local_secrets': ['Medieval tunnel systems', 'Artisan workshops in converted buildings'],
                'best_times': ['morning', 'afternoon'],
                'insider_tip': 'Many buildings from the Genoese period have hidden architectural details'
            }
        }
    
    def _initialize_tips_database(self) -> List[LocalTip]:
        """Initialize comprehensive database of local tips and insider knowledge"""
        return [
            # Local Eatery Tips
            LocalTip(
                id="tip_001",
                title="Authentic Lokanta Experience",
                description="Find traditional lokanta restaurants where locals eat daily meals",
                category=TipCategory.LOCAL_EATERY,
                neighborhood="Eminönü",
                tip_text="Look for places with handwritten menus on paper and locals reading newspapers. Order 'günün yemeği' (dish of the day) for the most authentic experience.",
                location=(41.0176, 28.9706),
                address="Various locations in Eminönü district",
                difficulty_level="easy",
                time_required="45min",
                best_time=["afternoon"],
                interests=["food", "culture", "authentic"],
                visitor_types=["solo", "couple", "all"],
                budget_range="budget",
                local_context="Locals prefer these family-run establishments over fancy restaurants for daily meals",
                insider_knowledge="The best lokantas don't have English menus - point to what looks good or ask 'Ne tavsiye edersiniz?' (What do you recommend?)",
                cultural_significance="Lokanta culture represents the communal dining tradition of Istanbul working class",
                rating=4.2,
                review_count=0,
                local_popularity=0.9,
                tourist_awareness=0.2
            ),
            
            LocalTip(
                id="tip_002",
                title="Secret Tea Garden Network",
                description="Discover hidden tea gardens in residential neighborhoods",
                category=TipCategory.HIDDEN_SPOT,
                neighborhood="Üsküdar",
                tip_text="Walk through residential streets and listen for the sound of backgammon pieces. Follow the sound to find authentic tea gardens tucked between apartment buildings.",
                location=(41.0214, 29.0053),
                difficulty_level="moderate",
                time_required="60min",
                best_time=["afternoon", "evening"],
                interests=["culture", "local life", "relaxation"],
                visitor_types=["solo", "couple"],
                budget_range="free",
                local_context="These tea gardens are neighborhood social hubs where locals gather daily",
                insider_knowledge="Bring your own backgammon set and you'll be invited to join games immediately",
                cultural_significance="Tea gardens preserve Ottoman-era social traditions in modern Istanbul",
                rating=4.5,
                local_popularity=0.95,
                tourist_awareness=0.1
            ),
            
            LocalTip(
                id="tip_003",
                title="Artisan Bread Quest",
                description="Find traditional bakeries that still use wood-fired ovens",
                category=TipCategory.LOCAL_EATERY,
                neighborhood="Fatih",
                tip_text="Look for smoke coming from chimneys early in the morning. The smell of wood-fired bread is unmistakable. Ask for 'köy ekmeği' (village bread).",
                location=(41.0186, 28.9748),
                difficulty_level="easy",
                time_required="30min",
                best_time=["morning"],
                seasonal_info="Best in winter when you can see the smoke clearly",
                interests=["food", "tradition", "authentic"],
                visitor_types=["all"],
                budget_range="budget",
                local_context="Traditional bakers start work at 4 AM to serve fresh bread to the neighborhood",
                insider_knowledge="The baker's family often lives above the shop - it's a true family business",
                cultural_significance="Wood-fired ovens maintain baking traditions dating back centuries",
                rating=4.7,
                local_popularity=0.85,
                tourist_awareness=0.15
            ),
            
            LocalTip(
                id="tip_004",
                title="Neighborhood Barber Philosophy",
                description="Experience traditional Turkish barber culture beyond the tourist versions",
                category=TipCategory.CULTURAL_SECRET,
                neighborhood="Beşiktaş",
                tip_text="Find barbers in residential areas, not tourist zones. A good traditional barber offers tea, conversation about football, and takes at least 45 minutes for a proper service.",
                location=(41.0422, 29.0061),
                difficulty_level="easy",
                time_required="60min",
                best_time=["afternoon"],
                interests=["culture", "tradition", "social interaction"],
                visitor_types=["solo"],
                budget_range="budget",
                local_context="Barber shops are male social spaces where neighborhood news and opinions are shared",
                insider_knowledge="Learn a few Turkish football terms - 'Beşiktaş nasıl?' (How's Beşiktaş?) is a great conversation starter",
                cultural_significance="Barber shops maintain traditional male bonding and community discussion spaces",
                rating=4.3,
                local_popularity=0.88,
                tourist_awareness=0.3
            ),
            
            LocalTip(
                id="tip_005",
                title="Hidden Bosphorus Fishing Spots",
                description="Discover where locals fish along the Bosphorus away from crowds",
                category=TipCategory.HIDDEN_SPOT,
                neighborhood="Ortaköy",
                tip_text="Walk along the waterfront at dawn. Look for groups of men with thermoses and simple fishing gear. These are the secret spots with the best catch.",
                location=(41.0547, 29.0275),
                difficulty_level="moderate",
                time_required="90min",
                best_time=["morning"],
                interests=["nature", "local life", "peaceful"],
                visitor_types=["solo", "couple"],
                budget_range="free",
                local_context="Fishing is both hobby and food source for many Istanbul locals",
                insider_knowledge="Bring tea or coffee to share - fishing is a social activity and locals appreciate company",
                cultural_significance="Bosphorus fishing connects modern Istanbulites to their maritime heritage",
                rating=4.1,
                local_popularity=0.75,
                tourist_awareness=0.05
            ),
            
            LocalTip(
                id="tip_006",
                title="Secret Rooftop Access",
                description="Learn how to access building rooftops for spectacular city views",
                category=TipCategory.SCENIC_VIEWPOINT,
                neighborhood="Galata",
                tip_text="In older buildings, look for unlocked roof access doors. Always ask the building's doorman first - 'Çatıya çıkabilir miyim?' Most are happy to help for a small tip.",
                location=(41.0267, 28.9744),
                difficulty_level="adventurous",
                time_required="45min",
                best_time=["evening", "morning"],
                interests=["photography", "views", "adventure"],
                visitor_types=["solo", "couple"],
                budget_range="free",
                local_context="Many building owners are proud of their rooftop views and enjoy sharing them",
                insider_knowledge="Sunset from Galata rooftops offers views that rival expensive restaurants",
                cultural_significance="Rooftop culture reflects Istanbul's vertical living and community sharing",
                rating=4.6,
                local_popularity=0.6,
                tourist_awareness=0.1
            ),
            
            LocalTip(
                id="tip_007",
                title="Traditional Market Timing",
                description="Master the rhythm of local markets for the best experience and prices",
                category=TipCategory.LOCAL_MARKET,
                neighborhood="Kadıköy",
                tip_text="Arrive at markets between 9-11 AM on weekdays. Vendors are relaxed, products are fresh, and you can have real conversations. Avoid weekends when locals rush shop.",
                location=(40.9833, 29.0167),
                difficulty_level="easy",
                time_required="90min",
                best_time=["morning"],
                interests=["food", "culture", "shopping"],
                visitor_types=["all"],
                budget_range="budget",
                local_context="Market vendors take pride in their products and enjoy educating interested customers",
                insider_knowledge="Ask vendors for recipes - they often give cooking tips along with ingredients",
                cultural_significance="Traditional markets preserve neighborhood-based commerce and social interaction",
                rating=4.4,
                local_popularity=0.92,
                tourist_awareness=0.25
            ),
            
            LocalTip(
                id="tip_008",
                title="Mosque Courtyard Etiquette",
                description="Properly experience mosque courtyards as peaceful community spaces",
                category=TipCategory.CULTURAL_SECRET,
                neighborhood="Sultanahmet",
                tip_text="Outside prayer times, mosque courtyards are public spaces. Sit quietly in corners, observe local interactions. Remove shoes before entering any covered area.",
                location=(41.0054, 28.9768),
                difficulty_level="easy",
                time_required="30min",
                best_time=["afternoon"],
                interests=["culture", "spirituality", "architecture"],
                visitor_types=["solo", "couple"],
                budget_range="free",
                local_context="Mosque courtyards serve as neighborhood meeting places and quiet retreats",
                insider_knowledge="Locals often feed cats in mosque courtyards - bringing cat food earns instant friendship",
                cultural_significance="Mosques historically served entire communities, not just religious functions",
                rating=4.3,
                local_popularity=0.8,
                tourist_awareness=0.4
            ),
            
            LocalTip(
                id="tip_009",
                title="Underground Passage Network",
                description="Navigate Istanbul's hidden underground passages and shortcuts",
                category=TipCategory.HIDDEN_SPOT,
                neighborhood="Beyoğlu",
                tip_text="Look for inconspicuous entrances near metro stations. Many buildings have underground connections. Follow locals during rush hour to discover passage shortcuts.",
                location=(41.0369, 28.9850),
                difficulty_level="adventurous",
                time_required="60min",
                best_time=["anytime"],
                interests=["adventure", "urban exploration", "history"],
                visitor_types=["solo", "couple"],
                budget_range="free",
                local_context="Underground passages help locals navigate efficiently during bad weather",
                insider_knowledge="Some passages connect to Ottoman-era tunnel systems - look for older stonework",
                cultural_significance="Underground networks reflect Istanbul's layered urban development",
                rating=4.0,
                local_popularity=0.7,
                tourist_awareness=0.05
            ),
            
            LocalTip(
                id="tip_010",
                title="Street Cat Feeding Culture",
                description="Understand and participate in Istanbul's famous street cat care system",
                category=TipCategory.NEIGHBORHOOD_LIFE,
                neighborhood="Cihangir",
                tip_text="Watch locals leave food and water for street cats. Join this community effort by bringing dry food. Cats know feeding schedules and wait at regular spots.",
                location=(41.0319, 28.9822),
                difficulty_level="easy",
                time_required="30min",
                best_time=["morning", "evening"],
                interests=["animals", "community", "culture"],
                visitor_types=["all"],
                budget_range="budget",
                local_context="Cat care is a shared neighborhood responsibility and pride point for locals",
                insider_knowledge="Each neighborhood has 'cat ladies' who coordinate care - they're great sources of local information",
                cultural_significance="Street cat care reflects Islamic values of animal welfare and community responsibility",
                rating=4.8,
                local_popularity=0.95,
                tourist_awareness=0.6
            ),
            
            LocalTip(
                id="tip_011",
                title="Traditional Hamam Preparation",
                description="Prepare properly for authentic Turkish bath experience in neighborhood hamams",
                category=TipCategory.CULTURAL_SECRET,
                neighborhood="Fatih",
                tip_text="Visit neighborhood hamams, not tourist ones. Bring your own soap and towel. Don't be shy - locals will help you understand the process. Stay hydrated.",
                location=(41.0186, 28.9748),
                difficulty_level="moderate",
                time_required="120min",
                best_time=["afternoon"],
                interests=["wellness", "tradition", "authentic"],
                visitor_types=["solo"],
                budget_range="moderate",
                local_context="Hamams are weekly social rituals where neighbors catch up on community news",
                insider_knowledge="Go on weekday afternoons when regulars attend - they'll teach you proper hamam etiquette",
                cultural_significance="Hamam culture dates back to Roman baths and remains central to Turkish cleanliness traditions",
                rating=4.5,
                local_popularity=0.85,
                tourist_awareness=0.2
            ),
            
            LocalTip(
                id="tip_012",
                title="Artisan Workshop Discovery",
                description="Find traditional craftsmen still practicing ancient trades",
                category=TipCategory.ARTISAN_WORKSHOP,
                neighborhood="Karaköy",
                tip_text="Listen for the sounds of traditional work - hammering, weaving, woodworking. Many artisans work in small shops tucked into narrow streets. Don't be afraid to watch and ask questions.",
                location=(41.0253, 28.9742),
                difficulty_level="moderate",
                time_required="60min",
                best_time=["morning", "afternoon"],
                interests=["crafts", "tradition", "learning"],
                visitor_types=["solo", "couple"],
                budget_range="free",
                local_context="Master craftsmen take pride in their work and enjoy sharing knowledge with interested visitors",
                insider_knowledge="Many artisans speak little English but communicate through demonstrations - universal language of craft",
                cultural_significance="Traditional crafts maintain Ottoman guild traditions and specialized knowledge",
                rating=4.6,
                local_popularity=0.7,
                tourist_awareness=0.15
            ),
            
            LocalTip(
                id="tip_013",
                title="Seasonal Food Timing",
                description="Align your food exploration with Istanbul's seasonal eating patterns",
                category=TipCategory.LOCAL_EATERY,
                neighborhood="Balat",
                tip_text="Spring: fresh vegetables and herbs. Summer: cold soups and seafood. Autumn: preserves and pickles. Winter: hearty stews and warm desserts. Ask vendors 'Mevsimlik ne var?' (What's seasonal?)",
                location=(41.0292, 28.9486),
                difficulty_level="easy",
                time_required="45min",
                best_time=["morning", "afternoon"],
                seasonal_info="Each season brings different specialties",
                interests=["food", "seasons", "authentic"],
                visitor_types=["all"],
                budget_range="budget",
                local_context="Istanbul locals eat according to natural seasons, maximizing freshness and flavor",
                insider_knowledge="Seasonal eating is deeply connected to Islamic dietary principles and Ottoman palace cuisine",
                cultural_significance="Seasonal food patterns connect modern Istanbulites to agricultural and religious calendars",
                rating=4.7,
                local_popularity=0.9,
                tourist_awareness=0.1
            ),
            
            LocalTip(
                id="tip_014",
                title="Cemetery Peaceful Walks",
                description="Discover Istanbul's historic cemeteries as peaceful neighborhood spaces",
                category=TipCategory.HIDDEN_SPOT,
                neighborhood="Eyüp",
                tip_text="Historic cemeteries are public spaces used for peaceful walks and reflection. Follow paths worn by locals. Many offer stunning city views and historical insights.",
                location=(41.0547, 28.9350),
                difficulty_level="easy",
                time_required="60min",
                best_time=["morning", "evening"],
                interests=["history", "peaceful", "views"],
                visitor_types=["solo", "couple"],
                budget_range="free",
                local_context="Cemeteries are integrated into neighborhood life as parks and reflective spaces",
                insider_knowledge="Many cemetery paths lead to unexpected viewpoints and historical discoveries",
                cultural_significance="Ottoman cemeteries reflect Islamic burial traditions and urban planning integration",
                rating=4.2,
                local_popularity=0.6,
                tourist_awareness=0.1
            ),
            
            LocalTip(
                id="tip_015",
                title="Local Transportation Wisdom",
                description="Master local transportation tricks that tourists never learn",
                category=TipCategory.INSIDER_KNOWLEDGE,
                neighborhood="Various",
                tip_text="Download BiTaksi app for local taxi alternatives. Learn dolmuş routes for authentic transport. Walk-ferry combinations often beat traffic. Ask locals 'En iyi yol ne?' (What's the best route?)",
                difficulty_level="moderate",
                time_required="varies",
                best_time=["anytime"],
                interests=["practical", "local life", "efficient"],
                visitor_types=["all"],
                budget_range="budget",
                local_context="Locals use multiple transport modes creatively to navigate Istanbul's complex geography",
                insider_knowledge="Combining walking with ferries often provides the most scenic and efficient routes",
                cultural_significance="Transportation choices reflect deep knowledge of Istanbul's unique geography and rhythm",
                rating=4.4,
                local_popularity=0.95,
                tourist_awareness=0.3
            ),
            
            LocalTip(
                id="tip_016",
                title="Traditional Music Discovery",
                description="Find authentic traditional music performances in unexpected places",
                category=TipCategory.CULTURAL_SECRET,
                neighborhood="Kuzguncuk",
                tip_text="Listen for traditional instruments from open windows and small venues. Friday evenings often feature informal gatherings. Ask at local tea houses about 'fasıl' nights.",
                location=(41.0219, 29.0608),
                difficulty_level="moderate",
                time_required="90min",
                best_time=["evening"],
                interests=["music", "culture", "authentic"],
                visitor_types=["solo", "couple"],
                budget_range="free",
                local_context="Traditional music maintains strong community connections in Istanbul neighborhoods",
                insider_knowledge="Many musicians play informally in homes and small venues - ask locals about private gatherings",
                cultural_significance="Traditional music preserves Ottoman court and folk traditions in modern settings",
                rating=4.3,
                local_popularity=0.75,
                tourist_awareness=0.05
            ),
            
            LocalTip(
                id="tip_017",
                title="Neighborhood Wedding Traditions",
                description="Respectfully observe traditional wedding celebrations in local neighborhoods",
                category=TipCategory.LOCAL_TRADITION,
                neighborhood="Üsküdar",
                tip_text="Wedding season (spring/summer) brings street celebrations. Watch respectfully from a distance. Locals often invite friendly observers to join celebrations. Learn 'Mutluluklar' (congratulations).",
                location=(41.0214, 29.0053),
                difficulty_level="easy",
                time_required="60min",
                best_time=["evening"],
                seasonal_info="Most weddings happen in spring and summer",
                interests=["culture", "celebration", "tradition"],
                visitor_types=["couple", "family"],
                budget_range="free",
                local_context="Wedding celebrations often spill into streets, involving entire neighborhoods",
                insider_knowledge="Traditional weddings include henna nights, street processions, and communal feasting",
                cultural_significance="Wedding traditions connect families and strengthen neighborhood community bonds",
                rating=4.5,
                local_popularity=0.85,
                tourist_awareness=0.1
            ),
            
            LocalTip(
                id="tip_018",
                title="Historic Fountain Appreciation",
                description="Discover and understand Istanbul's historic neighborhood fountains",
                category=TipCategory.CULTURAL_SECRET,
                neighborhood="Fatih",
                tip_text="Historic fountains (çeşme) are found throughout old neighborhoods. Many still provide fresh water. Locals use them daily. Read the Ottoman inscriptions for historical context.",
                location=(41.0186, 28.9748),
                difficulty_level="easy",
                time_required="30min",
                best_time=["anytime"],
                interests=["history", "architecture", "water"],
                visitor_types=["all"],
                budget_range="free",
                local_context="Fountains remain functional parts of neighborhood infrastructure and social gathering points",
                insider_knowledge="Many fountains have religious inscriptions offering water as charity - reflecting Islamic principles",
                cultural_significance="Ottoman fountains represent Islamic charity traditions and urban planning wisdom",
                rating=4.1,
                local_popularity=0.7,
                tourist_awareness=0.2
            ),
            
            LocalTip(
                id="tip_019",
                title="Local Library Culture",
                description="Experience neighborhood libraries as community centers and study spaces",
                category=TipCategory.NEIGHBORHOOD_LIFE,
                neighborhood="Şişli",
                tip_text="Local libraries offer more than books - they're community centers with events, language exchanges, and quiet study spaces. Many welcome respectful visitors for reading and wifi.",
                location=(41.0608, 28.9864),
                difficulty_level="easy",
                time_required="60min",
                best_time=["afternoon"],
                interests=["learning", "quiet", "community"],
                visitor_types=["solo"],
                budget_range="free",
                local_context="Libraries serve as neighborhood intellectual and social hubs beyond just book lending",
                insider_knowledge="Many libraries host language exchange meetings - great for meeting educated locals",
                cultural_significance="Public libraries maintain Ottoman traditions of learning and community intellectual development",
                rating=4.2,
                local_popularity=0.8,
                tourist_awareness=0.05
            ),
            
            LocalTip(
                id="tip_020",
                title="Traditional Spice Usage",
                description="Learn how locals actually use spices in daily cooking, beyond tourist spice bazaars",
                category=TipCategory.LOCAL_EATERY,
                neighborhood="Eminönü",
                tip_text="Visit neighborhood spice shops, not tourist bazaars. Ask vendors to explain daily spice usage. Watch locals shopping - they buy small quantities of specific spices for particular dishes.",
                location=(41.0176, 28.9706),
                difficulty_level="easy",
                time_required="45min",
                best_time=["morning"],
                interests=["food", "cooking", "authentic"],
                visitor_types=["all"],
                budget_range="budget",
                local_context="Local spice shops focus on daily cooking needs rather than tourist souvenirs",
                insider_knowledge="Real Turkish cooking uses fewer spices than tourist versions - quality over quantity",
                cultural_significance="Spice usage reflects regional Turkish cuisine variations and seasonal cooking patterns",
                rating=4.6,
                local_popularity=0.85,
                tourist_awareness=0.3
            )
        ]
    
    def _initialize_gems_database(self) -> List[HiddenGem]:
        """Initialize comprehensive database of hidden gems and secret locations"""
        return [
            # Historic Hidden Gems
            HiddenGem(
                id="gem_001",
                name="Ahrida Synagogue Secret Garden",
                description="Hidden garden courtyard behind one of Istanbul's oldest synagogues",
                category=TipCategory.HIDDEN_SPOT,
                neighborhood="Balat",
                location=(41.0292, 28.9486),
                address="Gevgili Sokak No: 9, Balat",
                walking_directions="From Balat ferry station, walk up the hill towards Ahrida Synagogue. Look for a small wooden door to the left of the main entrance.",
                what_makes_special="A peaceful garden sanctuary hidden behind historic walls, where time seems to stand still among ancient trees and stone benches",
                best_experience="Sit quietly in the garden during late afternoon when golden light filters through the old trees",
                local_tips=[
                    "Visit during weekday afternoons when it's most peaceful",
                    "Bring a book - locals often read here",
                    "Respect the religious significance of the space"
                ],
                opening_hours={
                    "monday": "9:00-17:00",
                    "tuesday": "9:00-17:00", 
                    "wednesday": "9:00-17:00",
                    "thursday": "9:00-17:00",
                    "friday": "9:00-15:00",
                    "saturday": "closed",
                    "sunday": "10:00-16:00"
                },
                best_visit_times=["afternoon", "morning"],
                avoid_times=["saturday", "jewish_holidays"],
                entry_requirements=["respectful_dress", "quiet_behavior"],
                what_to_bring=["book", "camera_with_permission"],
                photography_allowed=True,
                suitable_for=["solo", "couple", "contemplative"],
                interests_match=["history", "spirituality", "peaceful", "architecture"],
                difficulty_access="moderate",
                how_locals_found="Passed down through Balat neighborhood families for generations",
                discovery_story="Originally discovered by local children playing hide and seek, this garden has been a secret refuge for residents during difficult times",
                alternative_similar=["Fener Greek Patriarchate Garden", "Kariye Museum Courtyard"],
                rating=4.7,
                local_rating=4.9,
                hidden_level=0.95,
                authenticity_score=0.98
            ),
            
            HiddenGem(
                id="gem_002",
                name="Galata Tower Underground Passages",
                description="Medieval tunnel system beneath Galata connecting to the old Genoese quarter",
                category=TipCategory.HIDDEN_SPOT,
                neighborhood="Galata",
                location=(41.0256, 28.9742),
                address="Galata Kulesi Sokak, underground level",
                walking_directions="Enter through the basement of the old building at Galata Kulesi Sokak No: 15. Look for the stone archway marked with Genoese symbols.",
                what_makes_special="Original medieval stone tunnels with intact Genoese inscriptions and architectural details from the 14th century",
                best_experience="Explore with a flashlight during quiet hours to experience the medieval atmosphere",
                local_tips=[
                    "Bring a flashlight - lighting is minimal",
                    "Wear sturdy shoes - floors can be uneven",
                    "Go with a local guide who knows the safe passages"
                ],
                opening_hours={
                    "monday": "10:00-16:00",
                    "tuesday": "10:00-16:00",
                    "wednesday": "10:00-16:00", 
                    "thursday": "10:00-16:00",
                    "friday": "10:00-16:00",
                    "saturday": "10:00-14:00",
                    "sunday": "closed"
                },
                best_visit_times=["afternoon"],
                avoid_times=["rainy_days", "winter_mornings"],
                entry_requirements=["guided_tour", "safety_equipment"],
                what_to_bring=["flashlight", "sturdy_shoes", "jacket"],
                photography_allowed=True,
                suitable_for=["adventure", "history_lovers", "small_groups"],
                interests_match=["history", "adventure", "medieval", "architecture"],
                difficulty_access="challenging",
                how_locals_found="Discovered by restoration workers in the 1990s during building renovations",
                discovery_story="Hidden for centuries, these passages were used by Genoese merchants to move goods safely during Ottoman conquests",
                alternative_similar=["Basilica Cistern", "Theodosius Cistern"],
                rating=4.8,
                local_rating=4.9,
                hidden_level=0.92,
                authenticity_score=0.96
            ),
            
            HiddenGem(
                id="gem_003",
                name="Secret Bosphorus Cave",
                description="Natural cave accessible only at low tide, hidden beneath Üsküdar cliffs",
                category=TipCategory.HIDDEN_SPOT,
                neighborhood="Üsküdar",
                location=(41.0233, 29.0144),
                address="Üsküdar waterfront, below Mihrimah Sultan Mosque",
                walking_directions="From Üsküdar ferry terminal, walk along the waterfront towards Mihrimah Sultan Mosque. Descend the stone steps near the old Ottoman fountain.",
                what_makes_special="A natural sea cave with Byzantine-era carved niches, accessible only during specific tidal conditions",
                best_experience="Visit during sunset at low tide when the cave is illuminated by golden light reflecting off the water",
                local_tips=[
                    "Check tide schedules before visiting",
                    "Wear water-resistant shoes",
                    "Bring a waterproof bag for belongings"
                ],
                opening_hours={
                    "tide_dependent": "Low tide periods only"
                },
                best_visit_times=["sunset", "early_morning"],
                avoid_times=["high_tide", "stormy_weather"],
                entry_requirements=["tide_knowledge", "swimming_ability"],
                what_to_bring=["waterproof_shoes", "towel", "waterproof_camera"],
                photography_allowed=True,
                suitable_for=["adventure", "nature_lovers", "photographers"],
                interests_match=["nature", "adventure", "photography", "unique"],
                difficulty_access="challenging",
                how_locals_found="Fishermen discovered it decades ago while seeking shelter during storms",
                discovery_story="Used as a smugglers' hideout during the Ottoman period, the cave still contains carved symbols from different eras",
                alternative_similar=["Anadolu Kavağı Hidden Beach", "Rumeli Feneri Sea Caves"],
                rating=4.6,
                local_rating=4.8,
                hidden_level=0.98,
                authenticity_score=0.94
            ),
            
            HiddenGem(
                id="gem_004",
                name="Artisan's Secret Workshop Quarter",
                description="Hidden courtyard where traditional Ottoman crafts are still practiced by master artisans",
                category=TipCategory.ARTISAN_WORKSHOP,
                neighborhood="Karaköy",
                location=(41.0253, 28.9740),
                address="Kemankeş Karamustafa Paşa Mahallesi, hidden courtyard behind Bankalar Caddesi 47",
                walking_directions="From Karaköy tram station, walk towards Bankalar Caddesi. Enter through the narrow alley between buildings 45 and 49.",
                what_makes_special="Traditional Ottoman crafts practiced by master artisans in an authentic workshop environment",
                best_experience="Watch craftsmen at work and learn traditional techniques passed down through generations",
                local_tips=[
                    "Visit during morning hours when craftsmen are most active",
                    "Ask permission before photographing artisans at work",
                    "Some workshops offer hands-on experiences for visitors"
                ],
                opening_hours={
                    "monday": "9:00-17:00",
                    "tuesday": "9:00-17:00",
                    "wednesday": "9:00-17:00",
                    "thursday": "9:00-17:00",
                    "friday": "9:00-17:00",
                    "saturday": "10:00-15:00",
                    "sunday": "closed"
                },
                best_visit_times=["morning", "afternoon"],
                avoid_times=["sunday", "lunch_hours"],
                entry_requirements=["respectful_behavior", "advance_notice"],
                what_to_bring=["camera", "interest_in_crafts"],
                photography_allowed=True,
                suitable_for=["cultural_enthusiasts", "craft_lovers", "educational"],
                interests_match=["crafts", "traditional", "artisan", "cultural"],
                difficulty_access="moderate",
                how_locals_found="Known among craft enthusiasts and traditional art collectors",
                discovery_story="Hidden workshop quarter preserving Ottoman-era crafting traditions",
                alternative_similar=["Grand Bazaar Workshops", "Traditional Craft Centers"],
                rating=4.6,
                local_rating=4.8,
                hidden_level=0.85,
                authenticity_score=0.95
            ),
            
            HiddenGem(
                id="gem_007",
                name="Secret Rooftop Greenhouse",
                description="Hidden greenhouse garden on top of an Ottoman-era building growing rare medicinal plants",
                category=TipCategory.SCENIC_VIEWPOINT,
                neighborhood="Cihangir",
                location=(41.0319, 28.9822),
                address="Rooftop of Sıraselviler Caddesi No: 45",
                walking_directions="Enter through the antique shop on ground floor. Take the old wooden elevator to the 4th floor, then climb the spiral staircase to the roof.",
                what_makes_special="A century-old greenhouse maintaining traditional Ottoman medicinal plant cultivation with spectacular Bosphorus views",
                best_experience="Visit during the weekly herb harvesting sessions where you can learn about traditional Turkish medicine",
                local_tips=[
                    "The greenhouse keeper speaks only Turkish - bring a translation app",
                    "Wear comfortable shoes for the roof access",
                    "Visit in early morning for the best light and plant activity"
                ],
                opening_hours={
                    "wednesday": "8:00-12:00",
                    "saturday": "8:00-12:00",
                    "sunday": "15:00-18:00"
                },
                best_visit_times=["morning", "sunday_afternoon"],
                avoid_times=["winter_mornings", "windy_days"],
                entry_requirements=["building_access", "greenhouse_keeper_permission"],
                what_to_bring=["translation_app", "notebook", "camera"],
                photography_allowed=True,
                suitable_for=["plant_lovers", "photographers", "health_enthusiasts"],
                interests_match=["plants", "traditional_medicine", "photography", "unique"],
                difficulty_access="challenging",
                how_locals_found="Maintained by the same family for 80 years, known only to neighbors and plant enthusiasts",
                discovery_story="Created by a Turkish pharmacist in the 1940s to preserve traditional Ottoman medicinal plant knowledge",
                alternative_similar=["Gülhane Park Rose Garden", "Yıldız Park Greenhouse"],
                rating=4.7,
                local_rating=4.9,
                hidden_level=0.96,
                authenticity_score=0.95
            ),
            
            HiddenGem(
                id="gem_008",
                name="Underground Hammam Ruins",
                description="Perfectly preserved Ottoman bath ruins accessible through a hidden entrance",
                category=TipCategory.CULTURAL_SECRET,
                neighborhood="Fatih",
                location=(41.0186, 28.9755),
                address="Basement level of Draman Caddesi No: 23",
                walking_directions="From Fatih Mosque, walk down Draman Caddesi. Enter through the carpet shop and ask for 'eski hamam' (old bath).",
                what_makes_special="Complete Ottoman bath complex with original marble, heating system, and decorative tiles still intact",
                best_experience="Take a guided tour by candlelight to experience the mystical atmosphere of this underground treasure",
                local_tips=[
                    "Tours available only in Turkish - arrange for translation",
                    "Temperature is cool underground - bring a jacket",
                    "Photography requires special permission"
                ],
                opening_hours={
                    "by_appointment": "Daily 14:00-17:00"
                },
                best_visit_times=["afternoon"],
                avoid_times=["without_guide", "rush_hours"],
                entry_requirements=["guided_tour", "advance_booking"],
                what_to_bring=["jacket", "sturdy_shoes", "small_flashlight"],
                photography_allowed=False,
                suitable_for=["history_lovers", "architecture_enthusiasts", "small_groups"],
                interests_match=["history", "architecture", "ottoman", "underground"],
                difficulty_access="moderate",
                how_locals_found="Discovered by the carpet shop owner's family during renovations in the 1980s",
                discovery_story="This was the private bath of a Ottoman pasha's mansion, sealed for centuries and preserved in perfect condition",
                alternative_similar=["Cagaloglu Hamami Historical Section", "Suleymaniye Hamami Ruins"],
                rating=4.8,
                local_rating=4.9,
                hidden_level=0.94,
                authenticity_score=0.97
            ),
            
            HiddenGem(
                id="gem_009",
                name="Fisherman's Secret Cove",
                description="Hidden fishing cove where local fishermen share their daily catch and stories",
                category=TipCategory.NEIGHBORHOOD_LIFE,
                neighborhood="Ortaköy",
                location=(41.0555, 29.0290),
                address="Rocky shoreline 200m north of Ortaköy Mosque",
                walking_directions="From Ortaköy square, walk along the Bosphorus towards Arnavutköy. Look for the narrow path between large rocks.",
                what_makes_special="Daily gathering place where fishermen cook and share their catch while telling stories of the Bosphorus",
                best_experience="Arrive at sunset when fishermen return with their catch and often invite visitors to share grilled fish",
                local_tips=[
                    "Bring bread to contribute to the meal",
                    "Learn 'balık' (fish) and 'teşekkürler' (thank you)",
                    "Respect fishing equipment and boats"
                ],
                opening_hours={
                    "weather_dependent": "Dawn to dusk, good weather only"
                },
                best_visit_times=["sunset", "early_morning"],
                avoid_times=["stormy_weather", "winter_months"],
                entry_requirements=["respectful_behavior", "contribution_to_meal"],
                what_to_bring=["bread", "turkish_tea", "friendly_attitude"],
                photography_allowed=True,
                suitable_for=["social", "authentic_experience", "food_lovers"],
                interests_match=["local_life", "fishing", "food", "social"],
                difficulty_access="easy",
                how_locals_found="Traditional fishing spot used by local families for generations",
                discovery_story="This cove has been a fisherman's gathering place since Ottoman times, with the same families fishing here for centuries",
                alternative_similar=["Kumkapı Fisherman's Wharf", "Sarıyer Fishing Harbor"],
                rating=4.6,
                local_rating=4.8,
                hidden_level=0.85,
                authenticity_score=0.96
            ),
            
            HiddenGem(
                id="gem_010",
                name="Hidden Library of Lost Books",
                description="Private library collecting rare and banned books from Ottoman and Republican periods",
                category=TipCategory.CULTURAL_SECRET,
                neighborhood="Beyoğlu",
                location=(41.0369, 28.9850),
                address="Third floor of Galatasaray Passage, unmarked door",
                walking_directions="Enter Galatasaray Passage from Istiklal Caddesi. Climb to the third floor, look for the door with only a small book symbol.",
                what_makes_special="Rare collection of forbidden books, underground publications, and lost literary works from Turkey's complex history",
                best_experience="Attend the weekly secret literary discussions held every Thursday evening",
                local_tips=[
                    "Knock three times and mention 'kayıp kitaplar' (lost books)",
                    "Discussions are in Turkish with some English translation",
                    "Bring a rare book to trade or contribute to the collection"
                ],
                opening_hours={
                    "by_invitation": "Thursday evenings 19:00-22:00"
                },
                best_visit_times=["thursday_evening"],
                avoid_times=["without_invitation", "government_holidays"],
                entry_requirements=["invitation", "literary_interest", "discretion"],
                what_to_bring=["rare_book_contribution", "notebook", "open_mind"],
                photography_allowed=False,
                suitable_for=["intellectuals", "book_lovers", "history_researchers"],
                interests_match=["literature", "history", "intellectual", "rare_books"],
                difficulty_access="challenging",
                how_locals_found="Known only through the underground intellectual network of Istanbul",
                discovery_story="Started by a group of professors and writers in the 1960s to preserve books that were being destroyed or banned",
                alternative_similar=["Istanbul Research Institute Library", "SALT Galata Archives"],
                rating=4.9,
                local_rating=4.9,
                hidden_level=0.99,
                authenticity_score=0.98
            ),
            
            HiddenGem(
                id="gem_011",
                name="Secret Spice Merchant's Vault",
                description="Hidden vault beneath the Spice Bazaar containing rare spices and ancient recipes",
                category=TipCategory.LOCAL_MARKET,
                neighborhood="Eminönü",
                location=(41.0166, 28.9706),
                address="Basement of Mısır Çarşısı (Egyptian Bazaar), entrance through Pandeli Restaurant",
                walking_directions="Enter Pandeli Restaurant above the Spice Bazaar. Ask for the 'eski depo' (old storage) tour.",
                what_makes_special="Ottoman-era spice storage with original wooden containers holding spices from across the former empire",
                best_experience="Join the monthly spice tasting and history session led by the master spice merchant",
                local_tips=[
                    "Book through Pandeli Restaurant - they coordinate access",
                    "Wear comfortable shoes for narrow stairs",
                    "Prepare for intense aromas - bring tissues if you're sensitive"
                ],
                opening_hours={
                    "monthly": "First Saturday of each month, 15:00-17:00"
                },
                best_visit_times=["monthly_session"],
                avoid_times=["without_booking", "ramadan_period"],
                entry_requirements=["restaurant_booking", "advance_payment"],
                what_to_bring=["comfortable_shoes", "camera", "appetite_for_adventure"],
                photography_allowed=True,
                suitable_for=["food_enthusiasts", "history_lovers", "spice_collectors"],
                interests_match=["food", "history", "spices", "ottoman"],
                difficulty_access="moderate",
                how_locals_found="Family secret passed down through generations of spice merchants",
                discovery_story="This vault supplied spices to the Ottoman palace kitchens, with some containers still holding spices from trade routes that no longer exist",
                alternative_similar=["Grand Bazaar Secret Passages", "Kadıköy Spice Market Back Rooms"],
                rating=4.8,
                local_rating=4.7,
                hidden_level=0.91,
                authenticity_score=0.94
            ),
            
            HiddenGem(
                id="gem_012",
                name="Abandoned Ottoman Train Station",
                description="Beautifully preserved train station from the original Istanbul-Baghdad railway",
                category=TipCategory.OFF_BEATEN_PATH,
                neighborhood="Haydarpaşa",
                location=(40.9833, 29.0167),
                address="Behind the main Haydarpaşa Terminal building",
                walking_directions="From Haydarpaşa ferry terminal, walk around the main train station building to find the smaller, abandoned platform.",
                what_makes_special="Original German-built station with intact period details, offering a glimpse into the ambitious Ottoman railway dreams",
                best_experience="Visit during golden hour when sunlight streams through the broken windows, creating magical light patterns",
                local_tips=[
                    "Be careful of uneven surfaces and debris",
                    "Best photography light is in the late afternoon",
                    "Respect the space - don't disturb anything"
                ],
                opening_hours={
                    "unofficial": "Dawn to dusk - no official access"
                },
                best_visit_times=["late_afternoon", "early_morning"],
                avoid_times=["after_dark", "rainy_days"],
                entry_requirements=["careful_exploration", "respect_for_heritage"],
                what_to_bring=["sturdy_shoes", "camera", "flashlight"],
                photography_allowed=True,
                suitable_for=["photographers", "urban_explorers", "history_enthusiasts"],
                interests_match=["photography", "history", "railways", "abandoned_places"],
                difficulty_access="moderate",
                how_locals_found="Known to railway enthusiasts and photographers exploring the Haydarpaşa area",
                discovery_story="Part of the ambitious Berlin-Baghdad railway project, this station represents Ottoman modernization dreams interrupted by World War I",
                alternative_similar=["Sirkeci Station Historical Areas", "Bosphorus Railway Tunnel Entrances"],
                rating=4.5,
                local_rating=4.6,
                hidden_level=0.87,
                authenticity_score=0.89
            )
        ]
    
    def get_personalized_recommendations(self, user_id: str, query: str, 
                                       preferences: Dict[str, Any] = None,
                                       max_results: int = 8) -> Dict[str, Any]:
        """
        Get highly personalized recommendations based on user history and preferences
        
        Args:
            user_id: Unique identifier for the user
            query: User's query about hidden gems or local tips
            preferences: User preferences and past interactions
            
        Returns:
            Personalized response with recommendations
        """
        try:
            # Load user history if available
            user_history = self.user_interaction_history.get(user_id, {})
            
            # Merge preferences with learned preferences
            combined_preferences = self._merge_user_preferences(user_history, preferences or {})
            
            # Process query with enhanced context
            enhanced_context = {
                'user_id': user_id,
                'location': combined_preferences.get('preferred_location'),
                'interests': combined_preferences.get('interests', []),
                'visited_places': user_history.get('visited_places', []),
                'time_context': self._get_current_time_context(),
                'difficulty_preference': combined_preferences.get('difficulty_preference', 'moderate'),
                'budget_preference': combined_preferences.get('budget_preference', 'budget')
            }
            
            response = self.process_hidden_gems_query(query, enhanced_context)
            
            # Filter out previously visited places if requested
            if combined_preferences.get('avoid_visited', False):
                response = self._filter_visited_places(response, user_history.get('visited_places', []))
            
            # Add personalization score to each recommendation
            response = self._add_personalization_scores(response, combined_preferences)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return self.process_hidden_gems_query(query)
    
    def _merge_user_preferences(self, history: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user history and explicit preferences for personalized recommendations"""
        merged = {
            'interests': list(set(history.get('interests', [])).union(set(preferences.get('interests', [])))),
            'visited_places': list(set(history.get('visited_places', []))),
            'preferred_location': preferences.get('preferred_location', history.get('last_location')),
            'difficulty_preference': preferences.get('difficulty_preference', history.get('difficulty_preference', 'moderate')),
            'budget_preference': preferences.get('budget_preference', history.get('budget_preference', 'budget')),
            'avoid_visited': preferences.get('avoid_visited', True)
        }
        
        return merged
    
    def _filter_visited_places(self, response: Dict[str, Any], visited_places: List[str]) -> Dict[str, Any]:
        """Filter out places already visited by the user from the recommendations"""
        if 'recommendations' in response:
            for key in response['recommendations']:
                response['recommendations'][key] = [rec for rec in response['recommendations'][key] if rec['item'].id not in visited_places]
        
        return response
    
    def _add_personalization_scores(self, response: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Add personalization scores to each recommendation based on user preferences"""
        if 'recommendations' in response:
            for key in response['recommendations']:
                for rec in response['recommendations'][key]:
                    rec['personalization_score'] = self._calculate_recommendation_score(
                        rec['item'], 
                        {'categories': [], 'neighborhoods': [], 'interests': preferences.get('interests', []), 'time_preference': None, 'budget_preference': None, 'group_type': 'solo'},
                        preferences.get('preferred_location'),
                        preferences.get('interests', []),
                        self._get_current_time_context(),
                        'gem' if key == 'hidden_gems' else 'tip'
                    )
        
        return response
    
    def record_user_feedback(self, user_id: str, recommendation_id: str, 
                           feedback: Dict[str, Any]) -> bool:
        """
        Record user feedback on recommendations for learning
        
        Args:
            user_id: User identifier
            recommendation_id: ID of the recommendation being rated
            feedback: Feedback data including rating, visited, liked, etc.
            
        Returns:
            Success status
        """
        try:
            if user_id not in self.user_interaction_history:
                self.user_interaction_history[user_id] = {
                    'preferences': {},
                    'feedback_history': [],
                    'visited_places': [],
                    'interest_evolution': {}
                }
            
            # Record feedback
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'recommendation_id': recommendation_id,
                'rating': feedback.get('rating'),
                'visited': feedback.get('visited', False),
                'liked': feedback.get('liked'),
                'comments': feedback.get('comments', ''),
                'difficulty_actual': feedback.get('difficulty_actual'),
                'time_spent': feedback.get('time_spent')
            }
            
            self.user_interaction_history[user_id]['feedback_history'].append(feedback_entry)
            
            # Update visited places if user visited
            if feedback.get('visited'):
                self.user_interaction_history[user_id]['visited_places'].append(recommendation_id)
            
            # Learn from feedback to update user preferences
            self._update_user_preferences_from_feedback(user_id, feedback_entry)
            
            logger.info(f"Recorded feedback for user {user_id} on recommendation {recommendation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording user feedback: {e}")
            return False
    
    def get_trending_gems(self, time_period: str = "week", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get currently trending hidden gems based on user interactions
        
        Args:
            time_period: Time period to analyze (day, week, month)
            limit: Maximum number of gems to return
            
        Returns:
            List of trending gems with popularity metrics
        """
        try:
            # In a real system, this would analyze actual user interaction data
            # For now, we'll simulate trending based on gem characteristics
            
            period_days = {"day": 1, "week": 7, "month": 30}.get(time_period, 7)
            cutoff_date = datetime.now() - timedelta(days=period_days)
            
            trending_gems = []
            
            # Simulate trending based on authenticity, hidden level, and seasonal factors
            current_season = self._get_season(datetime.now().month)
            current_time = self._get_current_time_context()
            
            for gem in self.gems_database:
                trend_score = self._calculate_trend_score(gem, current_season, current_time)
                
                trending_gems.append({
                    'gem': gem,
                    'trend_score': trend_score,
                    'popularity_change': random.uniform(0.1, 0.9),  # Simulated
                    'recent_visits': random.randint(5, 25),  # Simulated
                    'why_trending': self._get_trending_reason(gem, current_season)
                })
            
            # Sort by trend score and return top gems
            trending_gems.sort(key=lambda x: x['trend_score'], reverse=True)
            
            return trending_gems[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trending gems: {e}")
            return []
    
    def _calculate_trend_score(self, gem: HiddenGem, current_season: str, current_time: Dict[str, Any]) -> float:
        """Calculate trending score based on gem characteristics and context"""
        try:
            score = 0.0
            
            # Base score from gem properties
            score += gem.authenticity_score * 0.3
            score += gem.hidden_level * 0.2
            score += (gem.rating / 5.0) * 0.2
            
            # Seasonal bonus
            if current_season.lower() in str(gem.avoid_times).lower():
                score -= 0.2
            elif current_season.lower() in str(gem.best_visit_times).lower():
                score += 0.2
            
            # Time context bonus
            if current_time.get('time_of_day') in gem.best_visit_times:
                score += 0.1
            
            # Difficulty accessibility (easier places tend to trend more)
            if gem.difficulty_access == "easy":
                score += 0.1
            elif gem.difficulty_access == "challenging":
                score -= 0.1
            
            # Category-based trending
            trending_categories = [TipCategory.HIDDEN_SPOT, TipCategory.SCENIC_VIEWPOINT, TipCategory.NEIGHBORHOOD_LIFE]
            if gem.category in trending_categories:
                score += 0.15
            
            # Add some randomness for dynamic trending
            score += random.uniform(0.0, 0.2)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.5
    
    def _get_trending_reason(self, gem: HiddenGem, current_season: str) -> str:
        """Get explanation for why a gem is trending"""
        try:
            reasons = []
            
            if gem.authenticity_score > 0.9:
                reasons.append("extremely authentic experience")
            
            if gem.hidden_level > 0.9:
                reasons.append("rarely visited secret spot")
            
            if current_season.lower() in str(gem.best_visit_times).lower():
                reasons.append(f"perfect for {current_season}")
            
            if gem.category == TipCategory.HIDDEN_SPOT:
                reasons.append("unique hidden location")
            elif gem.category == TipCategory.SCENIC_VIEWPOINT:
                reasons.append("amazing views")
            elif gem.category == TipCategory.NEIGHBORHOOD_LIFE:
                reasons.append("authentic local life")
            
            if gem.difficulty_access == "easy":
                reasons.append("easily accessible")
            
            if not reasons:
                reasons.append("growing local popularity")
            
            return f"Trending due to {', '.join(reasons[:2])}"
            
        except Exception as e:
            logger.error(f"Error getting trending reason: {e}")
            return "Popular among locals"
    
    def get_trending_recommendations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get currently trending recommendations based on user feedback and popularity
        
        Args:
            limit: Maximum number of recommendations to return
            
        Returns:
            List of trending recommendations
        """
        try:
            trending_items = []
            
            # Get trending gems
            for gem_id, gem in self.database.get('gems', {}).items():
                score = self._calculate_trending_score(gem_id, 'gem')
                if score > 0.6:  # Trending threshold
                    trending_items.append({
                        'id': gem_id,
                        'type': 'gem',
                        'name': gem.name,
                        'description': gem.description,
                        'neighborhood': gem.neighborhood,
                        'trending_score': score,
                        'category': gem.category.value
                    })
            
            # Get trending tips
            for tip_id, tip in self.database.get('tips', {}).items():
                score = self._calculate_trending_score(tip_id, 'tip')
                if score > 0.6:  # Trending threshold
                    trending_items.append({
                        'id': tip_id,
                        'type': 'tip',
                        'title': tip.title,
                        'description': tip.description,
                        'neighborhood': tip.neighborhood,
                        'trending_score': score,
                        'category': tip.category.value
                    })
            
            # Sort by trending score and return top items
            trending_items.sort(key=lambda x: x['trending_score'], reverse=True)
            return trending_items[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    def _calculate_trending_score(self, item_id: str, item_type: str) -> float:
        """Calculate trending score based on recent activity"""
        try:
            recent_feedback = 0
            positive_feedback = 0
            
            # Check feedback from all users
            for user_id, profile in self.user_profiles.items():
                feedback_history = profile.get('feedback_history', [])
                
                # Look for recent feedback on this item
                recent_cutoff = datetime.now() - timedelta(days=30)
                for feedback in feedback_history:
                    if (feedback.get('recommendation_id') == item_id and 
                        datetime.fromisoformat(feedback.get('timestamp', '2024-01-01')) > recent_cutoff):
                        recent_feedback += 1
                        if feedback.get('rating', 0) >= 4:
                            positive_feedback += 1
            
            # Calculate score
            if recent_feedback == 0:
                return 0.5  # Base score for items without feedback
            
            popularity_score = positive_feedback / recent_feedback
            activity_score = min(recent_feedback / 10, 1.0)  # Scale activity
            
            return (popularity_score * 0.7) + (activity_score * 0.3)
            
        except Exception as e:
            logger.error(f"Error calculating trending score for {item_id}: {e}")
            return 0.5
    
    def find_similar_recommendations(self, item_id: str, item_type: str = "auto", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find recommendations similar to a specific item
        
        Args:
            item_id: ID of the reference item
            item_type: Type of item ('gem', 'tip', or 'auto' to detect)
            limit: Maximum number of similar items to return
            
        Returns:
            List of similar recommendations
        """
        try:
            # Find the reference item
            reference_item = None
            actual_type = item_type
            
            if item_type == "auto" or item_type == "gem":
                reference_item = self.database.get('gems', {}).get(item_id)
                if reference_item:
                    actual_type = "gem"
            
            if not reference_item and (item_type == "auto" or item_type == "tip"):
                reference_item = self.database.get('tips', {}).get(item_id)
                if reference_item:
                    actual_type = "tip"
            
            if not reference_item:
                return []
            
            similar_items = []
            
            # Compare with all items (both gems and tips for diversity)
            for gem_id, gem in self.database.get('gems', {}).items():
                if gem_id != item_id:
                    similarity_score = self._calculate_similarity_score(reference_item, gem, actual_type, 'gem')
                    if similarity_score > 0.3:
                        similar_items.append({
                            'id': gem_id,
                            'type': 'gem',
                            'name': gem.name,
                            'description': gem.description,
                            'neighborhood': gem.neighborhood,
                            'similarity_score': similarity_score,
                            'similarity_reasons': self._get_similarity_reasons(reference_item, gem, actual_type, 'gem'),
                            'category': gem.category.value
                        })
            
            for tip_id, tip in self.database.get('tips', {}).items():
                if tip_id != item_id:
                    similarity_score = self._calculate_similarity_score(reference_item, tip, actual_type, 'tip')
                    if similarity_score > 0.3:
                        similar_items.append({
                            'id': tip_id,
                            'type': 'tip',
                            'title': tip.title,
                            'description': tip.description,
                            'neighborhood': tip.neighborhood,
                            'similarity_score': similarity_score,
                            'similarity_reasons': self._get_similarity_reasons(reference_item, tip, actual_type, 'tip'),
                            'category': tip.category.value
                        })
            
            # Sort by similarity score and return top items
            similar_items.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_items[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar recommendations: {e}")
            return []
    
    def _format_response(self, recommendations: List[Dict[str, Any]], 
                        query_analysis: Dict[str, Any], 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response with recommendations and context"""
        try:
            response = {
                'type': 'hidden_gems_response',
                'query_analysis': query_analysis,
                'total_recommendations': len(recommendations),
                'recommendations': {
                    'hidden_gems': [rec for rec in recommendations if rec['type'] == 'gem'],
                    'local_tips': [rec for rec in recommendations if rec['type'] == 'tip'],
                    'restaurants': [rec for rec in recommendations if rec['type'] == 'restaurant'],
                    'museums': [rec for rec in recommendations if rec['type'] == 'museum']
                },
                'context_info': {
                    'user_location': context.get('location'),
                    'time_context': context.get('time_context', {}),
                    'personalization_applied': len(context.get('interests', [])) > 0
                },
                'additional_suggestions': self._get_additional_suggestions(query_analysis, recommendations)
            }
            
            # Add formatted text response
            response['formatted_text'] = self._create_formatted_text_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return self._get_fallback_response(query_analysis.get('query', ''))

    def _get_additional_suggestions(self, query_analysis: Dict[str, Any], 
                                  recommendations: List[Dict[str, Any]]) -> List[str]:
        """Get additional suggestions based on the query and recommendations"""
        suggestions = []
        
        if query_analysis.get('neighborhoods'):
            suggestions.append(f"Explore more in {', '.join(query_analysis['neighborhoods'])}")
        
        if query_analysis.get('interests'):
            suggestions.append(f"Try searching for {', '.join(query_analysis['interests'])} experiences")
        
        if len(recommendations) < 3:
            suggestions.append("Try broadening your search criteria for more options")
        
        return suggestions[:3]

    def _create_formatted_text_response(self, response: Dict[str, Any]) -> str:
        """Create a formatted text response from the structured data"""
        try:
            text_parts = []
            
            # Introduction
            total = response['total_recommendations']
            text_parts.append(f"🗺️ Found {total} personalized recommendations for you!\n")
            
            # Hidden Gems
            gems = response['recommendations']['hidden_gems']
            if gems:
                text_parts.append("💎 **Hidden Gems:**")
                for i, rec in enumerate(gems[:3], 1):
                    item = rec['item']
                    text_parts.append(f"\n{i}. **{item.name}** - {item.neighborhood}")
                    text_parts.append(f"   {item.description}")
                    if rec.get('match_reasons'):
                        text_parts.append(f"   ✨ {', '.join(rec['match_reasons'][:2])}")
            
            # Local Tips
            tips = response['recommendations']['local_tips']
            if tips:
                text_parts.append("\n\n🎯 **Local Tips:**")
                for i, rec in enumerate(tips[:3], 1):
                    item = rec['item']
                    text_parts.append(f"\n{i}. **{item.title}** - {item.neighborhood}")
                    text_parts.append(f"   {item.description}")
                    if rec.get('match_reasons'):
                        text_parts.append(f"   ✨ {', '.join(rec['match_reasons'][:2])}")
            
            # Restaurants
            restaurants = response['recommendations']['restaurants']
            if restaurants:
                text_parts.append("\n\n🍽️ **Restaurants:**")
                for i, rec in enumerate(restaurants[:3], 1):
                    item = rec['item']
                    text_parts.append(f"\n{i}. **{item.name}** - {item.neighborhood}")
                    text_parts.append(f"   {item.description}")
                    if rec.get('match_reasons'):
                        text_parts.append(f"   ✨ {', '.join(rec['match_reasons'][:2])}")
            
            # Museums
            museums = response['recommendations']['museums']
            if museums:
                text_parts.append("\n\n🏛️ **Museums:**")
                for i, rec in enumerate(museums[:3], 1):
                    item = rec['item']
                    text_parts.append(f"\n{i}. **{item.name}** - {item.neighborhood}")
                    text_parts.append(f"   {item.description}")
                    if rec.get('match_reasons'):
                        text_parts.append(f"   ✨ {', '.join(rec['match_reasons'][:2])}")
            
            # Additional suggestions
            if response.get('additional_suggestions'):
                text_parts.append("\n\n💡 **You might also like:**")
                for suggestion in response['additional_suggestions']:
                    text_parts.append(f"• {suggestion}")
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error creating formatted text: {e}")
            return "I found some great recommendations for you! Ask me for more details about any specific area."

    def _get_fallback_response(self, query: str) -> Dict[str, Any]:
        """Generate fallback response when main processing fails"""
        return {
            'type': 'hidden_gems_fallback',
            'message': "I'd love to help you discover Istanbul's hidden gems! Try asking about specific neighborhoods like Balat, Beyoğlu, or Kadıköy, or mention your interests like food, art, museums, or history.",
            'suggestions': [
                "Hidden gems in Balat neighborhood",
                "Local restaurants in Kadıköy", 
                "Secret museums in Istanbul",
                "Traditional workshops and artisans",
                "Authentic Turkish cuisine spots",
                "Museums with hidden collections"
            ],
            'formatted_text': "🗺️ I'd love to help you discover Istanbul's hidden gems! Try asking about:\n\n• Hidden gems in Balat neighborhood\n• Local restaurants in Kadıköy\n• Secret museums in Istanbul\n• Traditional workshops and artisans\n• Authentic Turkish cuisine spots\n• Museums with hidden collections"
        }

    def _log_interaction(self, query: str, context: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> None:
        """Log user interaction for learning and analytics"""
        try:
            user_id = context.get('user_id', 'anonymous')
            
            if user_id not in self.user_interaction_history:
                self.user_interaction_history[user_id] = {
                    'queries': [],
                    'preferences': {},
                    'feedback_history': []
                }
            
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'context': context,
                'recommendations_count': len(recommendations),
                'recommendation_types': [rec['type'] for rec in recommendations[:5]]
            }
            
            self.user_interaction_history[user_id]['queries'].append(interaction)
            
            # Keep only last 50 interactions per user
            if len(self.user_interaction_history[user_id]['queries']) > 50:
                self.user_interaction_history[user_id]['queries'] = self.user_interaction_history[user_id]['queries'][-50:]
                
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
    def _initialize_restaurants_database(self) -> List[Restaurant]:
        """Initialize comprehensive database of authentic local restaurants"""
        return [
            # Traditional Turkish Restaurants
            Restaurant(
                id="rest_001",
                name="Hamdi Restaurant's Hidden Floor",
                description="Secret upstairs dining room at the famous Hamdi Restaurant with panoramic Golden Horn views",
                category="traditional",
                neighborhood="Eminönü",
                location=(41.0162, 28.9733),
                address="Kalçın Sokak No: 17, Eminönü",
                walking_directions="Enter Hamdi Restaurant and ask for the 'üst kat' (upper floor). Take the narrow staircase to the left of the main dining room.",
                cuisine_type="turkish",
                specialty_dishes=["Kuzu Tandır", "İskender Kebabı", "Testi Kebabı"],
                must_try_items=["Hamdi Special Lamb", "Traditional Piyaz", "Künefe"],
                dietary_options=["halal"],
                opening_hours={
                    "monday": "11:00-23:00",
                    "tuesday": "11:00-23:00",
                    "wednesday": "11:00-23:00",
                    "thursday": "11:00-23:00",
                    "friday": "11:00-23:00",
                    "saturday": "11:00-23:00",
                    "sunday": "11:00-23:00"
                },
                best_visit_times=["sunset", "early_evening"],
                avoid_times=["lunch_rush"],
                atmosphere="upscale",
                seating_style=["indoor", "window_view"],
                price_range="moderate",
                average_cost_per_person="80-120 TL",
                local_clientele_percentage=0.6,
                family_run=True,
                established_year=1960,
                suitable_for=["couple", "family", "business"],
                occasion_types=["celebration", "business"],
                how_locals_found="Generations of families have reserved this special floor for important occasions",
                local_reputation="The view from upstairs is considered the best dining vista in Eminönü",
                rating=4.7,
                local_rating=4.9,
                authenticity_score=0.95,
                hidden_level=0.7
            ),
            
            Restaurant(
                id="rest_002", 
                name="Beyti Ocakbaşı Gizli Bahçe",
                description="Hidden garden courtyard behind the famous Beyti restaurant, known only to regular customers",
                category="local_eatery",
                neighborhood="Florya",
                location=(40.9789, 28.7856),
                address="Orman Sokak No: 8, Florya",
                walking_directions="Walk to the back of Beyti restaurant and look for a wooden gate marked 'Bahçe'. Ring the bell and mention you're looking for the garden seating.",
                cuisine_type="turkish",
                specialty_dishes=["Beyti Kebabı", "Adana Kebabı", "Karışık Izgara"],
                must_try_items=["Original Beyti Wrap", "Grilled Lamb Chops", "Turkish Tea Service"],
                dietary_options=["halal"],
                opening_hours={
                    "monday": "12:00-23:30",
                    "tuesday": "12:00-23:30",
                    "wednesday": "12:00-23:30", 
                    "thursday": "12:00-23:30",
                    "friday": "12:00-23:30",
                    "saturday": "12:00-23:30",
                    "sunday": "12:00-23:30"
                },
                best_visit_times=["evening", "warm_weather"],
                atmosphere="traditional",
                seating_style=["outdoor", "floor_seating"],
                price_range="expensive",
                average_cost_per_person="150-200 TL",
                local_clientele_percentage=0.8,
                family_run=True,
                established_year=1945,
                generations_running=3,
                suitable_for=["family", "group", "celebration"],
                how_locals_found="Regular customers discovered this garden when the restaurant expanded but kept it quiet",
                chef_story="The secret garden recipes have been passed down through three generations of the Beyti family",
                local_reputation="Considered the most authentic kebab experience in Istanbul by local food connoisseurs",
                rating=4.8,
                local_rating=4.9,
                authenticity_score=0.98,
                hidden_level=0.8
            ),
            
            Restaurant(
                id="rest_003",
                name="Balık Lokantası Köprü Altı",
                description="Underground fish restaurant beneath Galata Bridge, accessible only through local fishermen",
                category="seafood",
                neighborhood="Karaköy",
                location=(41.0199, 28.9744),
                address="Galata Köprüsü Altı, Gizli Geçit",
                walking_directions="Go to the fishermen area under Galata Bridge. Look for the blue door marked 'Balık' and ask any fisherman for 'Mehmet Abi'nin yeri'.",
                cuisine_type="turkish",
                specialty_dishes=["Günün Balığı", "Balık Ekmek", "Midye Tava"],
                must_try_items=["Fresh Catch of the Day", "Traditional Fish Soup", "Sea Bass Grilled"],
                dietary_options=["pescatarian"],
                opening_hours={
                    "monday": "10:00-22:00",
                    "tuesday": "10:00-22:00",
                    "wednesday": "10:00-22:00",
                    "thursday": "10:00-22:00",
                    "friday": "10:00-22:00",
                    "saturday": "10:00-22:00",
                    "sunday": "10:00-22:00"
                },
                best_visit_times=["lunch", "early_evening"],
                atmosphere="casual",
                seating_style=["indoor", "counter"],
                price_range="budget",
                average_cost_per_person="40-60 TL",
                local_clientele_percentage=0.95,
                family_run=True,
                suitable_for=["solo", "couple", "local_experience"],
                how_locals_found="Fishermen have eaten here for decades - it's their unofficial canteen",
                local_reputation="The freshest fish in Istanbul, caught and served the same day",
                languages_spoken=["turkish"],
                english_menu_available=False,
                rating=4.6,
                local_rating=4.8,
                authenticity_score=0.99,
                hidden_level=0.9
            ),
            
            Restaurant(
                id="rest_004",
                name="Sultanahmet Ev Yemekleri",
                description="Home-cooking restaurant run by a local grandmother in her own house",
                category="local_eatery",
                neighborhood="Sultanahmet",
                location=(41.0082, 28.9784),
                address="Küçük Ayasofya Mahallesi, Şehit Mehmet Paşa Sokak No: 12",
                walking_directions="From Blue Mosque, walk towards Küçük Ayasofya. Look for a wooden sign saying 'Ev Yemekleri' on a residential door.",
                cuisine_type="turkish",
                specialty_dishes=["Dolma", "Karnıyarık", "Mantı"],
                must_try_items=["Homemade Dolma", "Traditional Turkish Breakfast", "Anne's Special Soup"],
                dietary_options=["vegetarian_options"],
                opening_hours={
                    "monday": "8:00-20:00",
                    "tuesday": "8:00-20:00",
                    "wednesday": "8:00-20:00",
                    "thursday": "8:00-20:00",
                    "friday": "8:00-20:00",
                    "saturday": "8:00-20:00",
                    "sunday": "closed"
                },
                best_visit_times=["morning", "lunch"],
                atmosphere="family",
                seating_style=["indoor", "home_style"],
                price_range="budget",
                average_cost_per_person="25-35 TL",
                local_clientele_percentage=0.7,
                family_run=True,
                suitable_for=["solo", "family", "authentic_experience"],
                how_locals_found="Word of mouth in the neighborhood - everyone knows Fatma Teyze's cooking",
                local_reputation="Like eating at your Turkish grandmother's house",
                child_friendly=True,
                rating=4.9,
                local_rating=4.9,
                authenticity_score=0.99,
                hidden_level=0.8
            ),
            
            Restaurant(
                id="rest_005",
                name="Çukurcuma Antika Café",
                description="Restaurant hidden among antique shops, serving Ottoman-era recipes",
                category="traditional",
                neighborhood="Çukurcuma",
                location=(41.0298, 28.9798),
                address="Çukurcuma Caddesi No: 51A, Çukurcuma",
                walking_directions="Walk through the antique district and look for a café sign between two antique shops. Enter through the narrow passage.",
                cuisine_type="ottoman",
                specialty_dishes=["Ottoman Pilaf", "Hünkar Beğendi", "Baklava"],
                must_try_items=["Sultan's Delight", "Historical Tea Blend", "Ottoman Sweets"],
                dietary_options=["vegetarian"],
                opening_hours={
                    "monday": "10:00-22:00",
                    "tuesday": "10:00-22:00",
                    "wednesday": "10:00-22:00",
                    "thursday": "10:00-22:00",
                    "friday": "10:00-22:00",
                    "saturday": "10:00-22:00",
                    "sunday": "12:00-20:00"
                },
                atmosphere="traditional",
                seating_style=["indoor", "ottoman_style"],
                price_range="moderate",
                average_cost_per_person="60-90 TL",
                local_clientele_percentage=0.6,
                suitable_for=["couple", "cultural_interest"],
                how_locals_found="Antique dealers and historians frequent this place for authentic Ottoman atmosphere",
                chef_story="Recipes recovered from 16th-century Ottoman palace cookbooks",
                rating=4.5,
                local_rating=4.7,
                authenticity_score=0.92,
                hidden_level=0.75
            )
        ]
    
    def _initialize_museums_database(self) -> List[Museum]:
        """Initialize comprehensive database of museums with cultural insights"""
        return [
            # Major Museums with Hidden Sections
            Museum(
                id="mus_001",
                name="Topkapi Palace Secret Chambers",
                description="Hidden chambers and private collections not accessible to regular tours",
                category="history",
                neighborhood="Sultanahmet",
                location=(41.0115, 28.9815),
                address="Topkapı Sarayı, Cankurtaran Mahallesi",
                walking_directions="Enter through the main palace entrance. Ask specifically for the 'özel koleksiyon' tour to access restricted areas.",
                primary_collection="Ottoman Imperial Artifacts",
                notable_artifacts=["Hidden Calligraphy Collection", "Private Imperial Jewelry", "Secret Correspondence Archives"],
                must_see_exhibits=["Sultan's Private Library", "Hidden Weapon Collection", "Restricted Sacred Relics"],
                special_collections=["Private Imperial Documents", "Rare Manuscripts", "Personal Effects"],
                period_focus=["ottoman", "imperial"],
                opening_hours={
                    "monday": "closed",
                    "tuesday": "9:00-18:00",
                    "wednesday": "9:00-18:00",
                    "thursday": "9:00-18:00",
                    "friday": "9:00-18:00",
                    "saturday": "9:00-18:00",
                    "sunday": "9:00-18:00"
                },
                best_visit_times=["early_morning", "late_afternoon"],
                avoid_times=["midday", "cruise_ship_hours"],
                recommended_visit_duration="3 hours",
                guided_tours_available=True,
                audio_guide_languages=["turkish", "english", "arabic", "german"],
                entrance_fee="Special tour: 100 TL",
                museum_pass_accepted=False,
                historical_significance="Private chambers reveal intimate details of Ottoman imperial life",
                cultural_importance="Provides unprecedented access to restricted Ottoman heritage",
                building_history="Secret chambers built for security and privacy of the Sultan's family",
                suitable_for=["history_enthusiasts", "researchers", "cultural_scholars"],
                interests_match=["ottoman_history", "imperial_culture", "rare_artifacts"],
                expertise_level=["intermediate", "expert"],
                crowd_levels={"morning": "light", "afternoon": "moderate", "evening": "light"},
                how_locals_view="Even Istanbulites rarely access these restricted areas",
                curator_insights="Hidden chambers contain items too precious for regular display",
                hidden_sections=["Sultan's Private Study", "Imperial Correspondence Room"],
                rating=4.8,
                educational_rating=4.9,
                cultural_authenticity=0.98,
                tourist_vs_local_ratio=0.3
            ),
            
            Museum(
                id="mus_002",
                name="Chora Museum Restoration Workshop",
                description="Behind-the-scenes access to Byzantine mosaic restoration process",
                category="art",
                neighborhood="Edirnekapı",
                location=(41.0309, 28.9378),
                address="Kariye Camii Sokak No: 18, Edirnekapı",
                walking_directions="Enter the main museum and ask for the 'restorasyon atölyesi' tour. Available only on certain days.",
                primary_collection="Byzantine Mosaics and Frescoes",
                notable_artifacts=["Restoration Tools", "Original Mosaic Fragments", "Byzantine Pigments"],
                must_see_exhibits=["Active Restoration Process", "Mosaic Technique Demonstration", "Original vs Restored Comparisons"],
                special_collections=["Restoration Archives", "Historical Photographs", "Technical Documentation"],
                period_focus=["byzantine", "medieval"],
                opening_hours={
                    "monday": "closed",
                    "tuesday": "9:00-16:30",
                    "wednesday": "9:00-16:30",
                    "thursday": "9:00-16:30",
                    "friday": "9:00-16:30",
                    "saturday": "9:00-16:30",
                    "sunday": "9:00-16:30"
                },
                best_visit_times=["morning", "workshop_hours"],
                recommended_visit_duration="2 hours",
                guided_tours_available=True,
                photography_policy="allowed_no_flash",
                entrance_fee="Workshop tour: 75 TL",
                historical_significance="Witness the preservation of 700-year-old Byzantine art",
                cultural_importance="Understanding the techniques behind Byzantine masterpieces",
                educational_programs=["Art Restoration Workshop", "Byzantine Art History"],
                suitable_for=["art_lovers", "students", "restoration_enthusiasts"],
                interests_match=["byzantine_art", "restoration", "art_history"],
                expertise_level=["intermediate", "expert"],
                current_exhibitions=["Restoration in Progress"],
                research_facilities=True,
                how_locals_view="Art students and professionals visit to understand restoration techniques",
                curator_insights="Active restoration allows visitors to see 14th-century techniques being preserved",
                rating=4.6,
                educational_rating=4.8,
                cultural_authenticity=0.95
            ),
            
            Museum(
                id="mus_003",
                name="Basilica Cistern Secret Passages",
                description="Hidden walkways and chambers in the ancient underground cistern",
                category="archaeological",
                neighborhood="Sultanahmet",
                location=(41.0084, 28.9778),
                address="Yerebatan Caddesi No: 13, Sultanahmet",
                walking_directions="Enter the main cistern and look for the special 'underground passages' tour guide near the Medusa columns.",
                primary_collection="Byzantine Underground Architecture",
                notable_artifacts=["Hidden Medusa Heads", "Original Byzantine Inscriptions", "Ancient Water Systems"],
                must_see_exhibits=["Secret Passage Network", "Hidden Chamber", "Original Cistern Function"],
                special_collections=["Archaeological Findings", "Byzantine Engineering", "Water Management Systems"],
                period_focus=["byzantine", "roman"],
                opening_hours={
                    "monday": "9:00-17:30",
                    "tuesday": "9:00-17:30",
                    "wednesday": "9:00-17:30",
                    "thursday": "9:00-17:30",
                    "friday": "9:00-17:30",
                    "saturday": "9:00-17:30",
                    "sunday": "9:00-17:30"
                },
                best_visit_times=["early_morning", "late_afternoon"],
                recommended_visit_duration="90 minutes",
                guided_tours_available=True,
                photography_policy="allowed",
                entrance_fee="Special tour: 60 TL",
                historical_significance="Reveals hidden aspects of Byzantine Constantinople's water system",
                building_history="Secret passages used for maintenance and emergency access",
                suitable_for=["history_enthusiasts", "archaeology_lovers", "adventure_seekers"],
                interests_match=["byzantine_history", "archaeology", "underground_exploration"],
                crowd_levels={"morning": "moderate", "afternoon": "heavy", "evening": "light"},
                how_locals_view="Many Istanbulites don't know about the hidden passages",
                curator_insights="Secret areas reveal the full scope of Byzantine engineering mastery",
                hidden_sections=["Maintenance Tunnels", "Emergency Exit Passages"],
                rating=4.7,
                educational_rating=4.5,
                cultural_authenticity=0.96
            ),
            
            Museum(
                id="mus_004", 
                name="Rahmi M. Koç Private Collection",
                description="Personal collection rooms not open to public, showcasing rare industrial artifacts",
                category="specialty",
                neighborhood="Hasköy",
                location=(41.0378, 28.9497),
                address="Hasköy Caddesi No: 5, Hasköy",
                walking_directions="Enter the main museum and ask for Mr. Koç's 'özel koleksiyon' at the information desk. Available by appointment only.",
                primary_collection="Industrial Heritage and Transportation",
                notable_artifacts=["Rare Vintage Cars", "Early Aviation Equipment", "Maritime Instruments"],
                must_see_exhibits=["Private Automobile Collection", "Historical Ship Models", "Early Industrial Equipment"],
                special_collections=["Personal Archives", "Industrial Documentation", "Technical Drawings"],
                period_focus=["industrial_age", "modern"],
                opening_hours={
                    "monday": "closed",
                    "tuesday": "10:00-17:00",
                    "wednesday": "10:00-17:00", 
                    "thursday": "10:00-17:00",
                    "friday": "10:00-17:00",
                    "saturday": "10:00-19:00",
                    "sunday": "10:00-19:00"
                },
                best_visit_times=["appointment_only"],
                recommended_visit_duration="2.5 hours",
                guided_tours_available=True,
                entrance_fee="Private tour: 150 TL",
                museum_pass_accepted=False,
                historical_significance="Documents Turkey's industrial transformation",
                educational_programs=["Industrial Heritage Workshop"],
                suitable_for=["engineering_enthusiasts", "collectors", "historians"],
                interests_match=["industrial_history", "transportation", "collecting"],
                expertise_level=["intermediate", "expert"],
                research_facilities=True,
                library_access=True,
                how_locals_view="Collectors and engineers know about the private sections",
                curator_insights="Private collection contains items too valuable for regular display",
                rating=4.4,
                educational_rating=4.6,
                cultural_authenticity=0.85
            ),
            
            Museum(
                id="mus_005",
                name="Sakıp Sabancı Museum Manuscript Library",
                description="Hidden manuscript collection and calligraphy workshop accessible to serious researchers",
                category="art",
                neighborhood="Emirgan",
                location=(41.1089, 29.0458),
                address="Sakıp Sabancı Caddesi No: 42, Emirgan",
                walking_directions="Enter the main museum and request access to the 'manuscript research library' at the reception. Academic credentials may be required.",
                primary_collection="Islamic Calligraphy and Manuscripts",
                notable_artifacts=["Rare Quran Manuscripts", "Ottoman Calligraphy", "Historical Documents"],
                must_see_exhibits=["Master Calligrapher Workshop", "Manuscript Preservation Lab", "Rare Book Collection"],
                special_collections=["Private Manuscript Collection", "Calligraphy Tools", "Historical Archives"],
                period_focus=["islamic", "ottoman", "calligraphy"],
                opening_hours={
                    "monday": "closed",
                    "tuesday": "10:00-18:00",
                    "wednesday": "10:00-18:00",
                    "thursday": "10:00-20:00", 
                    "friday": "10:00-18:00",
                    "saturday": "10:00-18:00",
                    "sunday": "10:00-18:00"
                },
                best_visit_times=["by_appointment"],
                recommended_visit_duration="2 hours",
                guided_tours_available=True,
                entrance_fee="Research access: 100 TL",
                historical_significance="Preserves rare Islamic calligraphy traditions",
                cultural_importance="Center for Islamic art research and calligraphy studies",
                educational_programs=["Calligraphy Workshop", "Manuscript Studies"],
                suitable_for=["researchers", "art_students", "calligraphy_enthusiasts"],
                interests_match=["islamic_art", "calligraphy", "manuscripts"],
                expertise_level=["expert"],
                research_facilities=True,
                library_access=True,
                scholar_resources=["Research Database", "Digital Archive Access"],
                how_locals_view="Known among calligraphy masters and Islamic art scholars",
                curator_insights="Contains manuscripts not displayed publicly due to their fragility",
                rating=4.3,
                educational_rating=4.7,
                cultural_authenticity=0.93
            )
        ]
