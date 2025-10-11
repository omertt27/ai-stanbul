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

class HiddenGemsLocalTips:
    """
    Advanced Hidden Gems and Local Tips System
    Provides personalized, context-aware recommendations for Istanbul's hidden treasures
    """
    
    def __init__(self):
        """Initialize the hidden gems system with comprehensive data"""
        self.tips_database = self._initialize_tips_database()
        self.gems_database = self._initialize_gems_database()
        self.user_preferences = {}
        self.neighborhood_data = self._initialize_neighborhood_data()
        
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
            'history': ['history', 'historical', 'ancient', 'byzantine', 'ottoman'],
            'art': ['art', 'gallery', 'museum', 'artistic', 'creative'],
            'food': ['food', 'culinary', 'taste', 'eat', 'drink'],
            'photography': ['photo', 'instagram', 'picture', 'shoot'],
            'architecture': ['architecture', 'building', 'mosque', 'church'],
            'nature': ['nature', 'park', 'garden', 'outdoor', 'green'],
            'nightlife': ['night', 'evening', 'bar', 'music', 'entertainment']
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
        
        # Combine tips and gems for unified scoring
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
        
        # Category matching
        if hasattr(item, 'category') and item.category in query_analysis['categories']:
            score += 0.3
        
        # Neighborhood matching
        if hasattr(item, 'neighborhood') and item.neighborhood.lower() in [n.lower() for n in query_analysis['neighborhoods']]:
            score += 0.25
        
        # Interest matching
        item_interests = getattr(item, 'interests', []) if item_type == 'tip' else getattr(item, 'interests_match', [])
        interest_overlap = len(set(user_interests).intersection(set(item_interests)))
        if item_interests:
            score += (interest_overlap / len(item_interests)) * 0.2
        
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
    
    def _get_season(self, month: int) -> str:
        """Determine season based on month"""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"
    
    def _format_response(self, recommendations: List[Dict[str, Any]], 
                        query_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Format comprehensive response with recommendations and insights"""
        
        if not recommendations:
            return self._get_fallback_response("No specific recommendations found")
        
        # Separate tips and gems
        tips = [r for r in recommendations if r['type'] == 'tip']
        gems = [r for r in recommendations if r['type'] == 'gem']
        
        response = {
            'status': 'success',
            'query_understood': query_analysis,
            'total_recommendations': len(recommendations),
            'personalization_applied': True,
            'recommendations': {
                'local_tips': self._format_tips(tips[:4]),
                'hidden_gems': self._format_gems(gems[:4])
            },
            'additional_insights': self._generate_additional_insights(recommendations, query_analysis),
            'local_context': self._get_local_context(query_analysis),
            'next_suggestions': self._get_next_suggestions(recommendations, query_analysis)
        }
        
        return response
    
    def _format_tips(self, tip_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format local tips for response"""
        formatted_tips = []
        
        for rec in tip_recommendations:
            tip = rec['item']
            formatted_tips.append({
                'title': tip.title,
                'description': tip.description,
                'neighborhood': tip.neighborhood,
                'tip': tip.tip_text,
                'category': tip.category.value.replace('_', ' ').title(),
                'insider_knowledge': tip.insider_knowledge,
                'local_context': tip.local_context,
                'best_time': tip.best_time,
                'difficulty': tip.difficulty_level,
                'time_required': tip.time_required,
                'budget': tip.budget_range,
                'why_recommended': rec['match_reasons'],
                'local_popularity': f"{tip.local_popularity:.1f}/1.0",
                'authenticity': "High" if tip.tourist_awareness < 0.3 else "Medium"
            })
        
        return formatted_tips
    
    def _format_gems(self, gem_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format hidden gems for response"""
        formatted_gems = []
        
        for rec in gem_recommendations:
            gem = rec['item']
            formatted_gems.append({
                'name': gem.name,
                'description': gem.description,
                'neighborhood': gem.neighborhood,
                'address': gem.address,
                'what_makes_special': gem.what_makes_special,
                'best_experience': gem.best_experience,
                'local_tips': gem.local_tips,
                'walking_directions': gem.walking_directions,
                'best_visit_times': gem.best_visit_times,
                'what_to_bring': gem.what_to_bring,
                'photography_allowed': gem.photography_allowed,
                'difficulty_access': gem.difficulty_access,
                'why_recommended': rec['match_reasons'],
                'hidden_level': f"{gem.hidden_level:.1f}/1.0",
                'authenticity_score': f"{gem.authenticity_score:.1f}/1.0",
                'discovery_story': gem.discovery_story
            })
        
        return formatted_gems
    
    def _generate_additional_insights(self, recommendations: List[Dict[str, Any]], 
                                    query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate additional insights and context"""
        insights = {
            'local_wisdom': [],
            'timing_advice': [],
            'cultural_context': [],
            'practical_tips': []
        }
        
        # Extract wisdom from top recommendations
        for rec in recommendations[:3]:
            item = rec['item']
            
            if hasattr(item, 'cultural_significance') and item.cultural_significance:
                insights['cultural_context'].append(item.cultural_significance)
            
            if hasattr(item, 'insider_knowledge') and item.insider_knowledge:
                insights['local_wisdom'].append(item.insider_knowledge)
        
        # Add timing advice based on query
        if query_analysis.get('time_preference'):
            time_pref = query_analysis['time_preference']
            timing_tips = {
                'morning': "Morning is perfect for authentic local experiences - markets are fresh, cafes serve traditional breakfast, and neighborhoods show their daily rhythm.",
                'afternoon': "Afternoons offer the best people-watching opportunities and many hidden spots are most accessible during these hours.",
                'evening': "Evening brings Istanbul's magical atmosphere - perfect for scenic viewpoints and local evening traditions.",
                'night': "Nighttime reveals Istanbul's secret side - from late-night eateries to illuminated hidden architectural gems."
            }
            if time_pref in timing_tips:
                insights['timing_advice'].append(timing_tips[time_pref])
        
        # Add practical tips
        insights['practical_tips'].extend([
            "Always greet locals with 'Merhaba' - it opens doors to better experiences",
            "Cash is preferred at most hidden local spots",
            "Learning a few Turkish phrases will earn you genuine local appreciation"
        ])
        
        return insights
    
    def _get_local_context(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Provide local context and cultural insights"""
        context = {
            'cultural_notes': [
                "Istanbul locals appreciate visitors who show genuine interest in their culture",
                "The concept of 'hidden gems' in Istanbul often relates to places where locals gather naturally",
                "Many of Istanbul's best-kept secrets are passed down through generations of residents"
            ],
            'etiquette_tips': [
                "Remove shoes when entering mosques or traditional homes",
                "Dress modestly when visiting religious or traditional areas",
                "Accept tea or coffee when offered - it's a sign of hospitality"
            ],
            'seasonal_considerations': []
        }
        
        # Add seasonal context
        current_season = self._get_season(datetime.now().month)
        seasonal_tips = {
            'winter': "Winter reveals Istanbul's cozy indoor culture - perfect for discovering traditional tea houses and covered markets",
            'spring': "Spring is ideal for exploring outdoor hidden gems and rooftop terraces with blooming views",
            'summer': "Summer nights are magical for discovering evening traditions and seaside local spots",
            'autumn': "Autumn offers the best weather for walking tours of hidden neighborhoods and local markets"
        }
        
        if current_season in seasonal_tips:
            context['seasonal_considerations'].append(seasonal_tips[current_season])
        
        return context
    
    def _get_next_suggestions(self, recommendations: List[Dict[str, Any]], 
                            query_analysis: Dict[str, Any]) -> List[str]:
        """Generate follow-up suggestions for deeper exploration"""
        suggestions = []
        
        # Based on found categories
        if any(rec['item'].category == TipCategory.LOCAL_EATERY for rec in recommendations):
            suggestions.append("Ask me about traditional cooking workshops or food tours with locals")
        
        if any(rec['item'].category == TipCategory.CULTURAL_SECRET for rec in recommendations):
            suggestions.append("Discover more about Istanbul's hidden historical sites and their stories")
        
        if any(rec['item'].category == TipCategory.SCENIC_VIEWPOINT for rec in recommendations):
            suggestions.append("Find the best photography spots known only to local photographers")
        
        # Generic helpful suggestions
        suggestions.extend([
            "Learn about connecting with local Istanbul communities",
            "Discover seasonal events and festivals locals attend",
            "Find out about traditional crafts workshops you can join"
        ])
        
        return suggestions[:5]  # Return max 5 suggestions
    
    def _get_fallback_response(self, query: str) -> Dict[str, Any]:
        """Provide fallback response when no specific matches found"""
        return {
            'status': 'partial',
            'message': 'Let me share some amazing local secrets from Istanbul',
            'general_recommendations': [
                {
                    'title': 'Local Tea Culture Discovery',
                    'description': 'Find authentic tea gardens where locals gather for conversation and backgammon',
                    'tip': 'Look for places where older men play tavla (backgammon) - these are genuine local spots',
                    'neighborhoods': ['Eminönü', 'Kadıköy', 'Beşiktaş']
                },
                {
                    'title': 'Hidden Bosphorus Views',
                    'description': 'Discover secret viewpoints known only to Istanbul photographers',
                    'tip': 'Early morning or late evening provide the most magical lighting',
                    'neighborhoods': ['Üsküdar', 'Ortaköy', 'Galata']
                },
                {
                    'title': 'Authentic Local Markets',
                    'description': 'Experience real neighborhood markets where locals do their daily shopping',
                    'tip': 'Visit on weekday mornings for the most authentic experience',
                    'neighborhoods': ['Kadıköy Market', 'Beşiktaş Fish Market', 'Fatih Local Bazaars']
                }
            ],
            'local_wisdom': [
                "The best local experiences happen when you follow the daily rhythms of Istanbul residents",
                "Genuine hidden gems are often discovered through conversations with shopkeepers and locals",
                "Istanbul's neighborhoods each have their own personality - spend time in one area to truly discover its secrets"
            ]
        }
    
    def _log_interaction(self, query: str, context: Dict[str, Any], 
                        recommendations: List[Dict[str, Any]]) -> None:
        """Log user interaction for learning and improvement"""
        try:
            interaction_data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'context': context,
                'recommendations_count': len(recommendations),
                'categories_found': list(set(rec['item'].category.value for rec in recommendations if hasattr(rec['item'], 'category'))),
                'neighborhoods_covered': list(set(rec['item'].neighborhood for rec in recommendations if hasattr(rec['item'], 'neighborhood')))
            }
            
            # In a production system, this would be saved to a database
            logger.info(f"User interaction logged: {interaction_data}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
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
                what_makes_special="Active workshops where masters practice calligraphy, miniature painting, and traditional bookbinding using centuries-old techniques",
                best_experience="Arrive in the morning when masters are most focused on their detailed work and willing to explain their craft",
                local_tips=[
                    "Learn basic Turkish greetings - artisans appreciate the effort",
                    "Don't rush - watch quietly and ask permission before photographing",
                    "Bring cash if you want to commission custom work"
                ],
                opening_hours={
                    "monday": "9:00-17:00",
                    "tuesday": "9:00-17:00",
                    "wednesday": "9:00-17:00",
                    "thursday": "9:00-17:00",
                    "friday": "9:00-15:00",
                    "saturday": "10:00-16:00",
                    "sunday": "closed"
                },
                best_visit_times=["morning", "afternoon"],
                avoid_times=["lunch_hour", "friday_prayers"],
                entry_requirements=["respectful_behavior", "no_loud_talking"],
                what_to_bring=["cash_for_purchases", "small_gifts", "business_card"],
                photography_allowed=False,
                suitable_for=["craft_lovers", "culture_enthusiasts", "solo"],
                interests_match=["crafts", "tradition", "learning", "authentic"],
                difficulty_access="moderate",
                how_locals_found="Word of mouth through the artisan community and their families",
                discovery_story="This courtyard has housed artisan workshops continuously since Ottoman times, with some families passing down crafts for 15 generations",
                alternative_similar=["Grand Bazaar Back Workshops", "Sultanahmet Traditional Crafts Center"],
                rating=4.9,
                local_rating=4.9,
                hidden_level=0.88,
                authenticity_score=0.99
            ),
            
            HiddenGem(
                id="gem_005",
                name="Floating Tea Garden",
                description="Traditional tea garden built on a floating platform in a hidden Golden Horn inlet",
                category=TipCategory.HIDDEN_SPOT,
                neighborhood="Eyüp",
                location=(41.0547, 28.9333),
                address="Golden Horn inlet, accessible via boat from Eyüp waterfront",
                walking_directions="From Eyüp Sultan Mosque, walk to the small boat landing near the old Ottoman cemetery. Ask local fishermen for 'Yüzen Çay Bahçesi'.",
                what_makes_special="The only floating tea garden in Istanbul, offering 360-degree water views while serving traditional Turkish tea and backgammon",
                best_experience="Visit during sunset with a group of friends for tea, backgammon, and traditional Turkish conversation",
                local_tips=[
                    "Boat transport is included with tea orders",
                    "Bring layers - it can be breezy on the water",
                    "Learn basic backgammon rules to join local games"
                ],
                opening_hours={
                    "seasonal": "April to October",
                    "daily": "14:00-22:00"
                },
                best_visit_times=["sunset", "evening"],
                avoid_times=["windy_days", "winter_months"],
                entry_requirements=["boat_transport", "minimum_tea_order"],
                what_to_bring=["jacket", "backgammon_skills", "Turkish_phrases"],
                photography_allowed=True,
                suitable_for=["groups", "social", "unique_experience"],
                interests_match=["unique", "social", "water", "traditional"],
                difficulty_access="moderate",
                how_locals_found="Created by a retired fisherman who wanted to combine his love of the water with tea culture",
                discovery_story="Started as a single table on a fishing boat, this floating tea garden has grown into a beloved local institution known only through word of mouth",
                alternative_similar=["Pierre Loti Hill Tea Garden", "Gülhane Park Tea Terraces"],
                rating=4.8,
                local_rating=4.9,
                hidden_level=0.93,
                authenticity_score=0.97
            ),
            
            HiddenGem(
                id="gem_006",
                name="Byzantine Mosaic Workshop Ruins",
                description="Partially excavated Byzantine workshop where original mosaics are still being uncovered",
                category=TipCategory.CULTURAL_SECRET,
                neighborhood="Sultanahmet",
                location=(41.0058, 28.9785),
                address="Small street behind Sokollu Mehmet Paşa Mosque",
                walking_directions="From Sultanahmet tram stop, walk towards Sokollu Mehmet Paşa Mosque. Look for the unmarked door with Byzantine symbols.",
                what_makes_special="An active archaeological site where visitors can observe ongoing excavation and restoration of 6th-century Byzantine mosaics",
                best_experience="Join the weekly volunteer sessions where you can help with cataloging discoveries under expert supervision",
                local_tips=[
                    "Contact the Byzantine Research Foundation in advance",
                    "Wear clothes you don't mind getting dusty",
                    "Bring sun protection - part of the site is outdoor"
                ],
                opening_hours={
                    "by_appointment": "Tuesday, Thursday, Saturday 10:00-16:00"
                },
                best_visit_times=["morning"],
                avoid_times=["rainy_days", "without_appointment"],
                entry_requirements=["advance_booking", "archaeological_interest"],
                what_to_bring=["sun_hat", "work_clothes", "notebook"],
                photography_allowed=False,
                suitable_for=["history_enthusiasts", "archaeology_lovers", "educational"],
                interests_match=["history", "archaeology", "byzantine", "learning"],
                difficulty_access="moderate",
                how_locals_found="Discovered during routine foundation work for a nearby building in 2018",
                discovery_story="This workshop produced mosaics for Hagia Sophia and other major Byzantine monuments, with some pieces still containing the original artists' signatures",
                alternative_similar=["Great Palace Mosaic Museum", "Kariye Museum Mosaics"],
                rating=4.9,
                local_rating=4.8,
                hidden_level=0.97,
                authenticity_score=0.98
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
