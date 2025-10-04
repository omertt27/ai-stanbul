"""
Recommendation Engine for AI Istanbul System

This service handles personalized recommendations, itinerary planning,
and suggestion generation without using GPT. Uses rule-based algorithms,
user preferences, and structured data for intelligent recommendations.
Enhanced with advanced collaborative filtering and personalization.
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import advanced recommendation components
try:
    from .advanced_recommendation_engine import (
        AdvancedRecommendationEngine, AdvancedRecommendation, 
        UserBehavior, ContentFeatures, LocationContext
    )
    from .rule_based_personalization import (
        RuleBasedPersonalizationEngine, UserContext, PersonalizationResult
    )
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced recommendation components not available: {e}")
    ADVANCED_COMPONENTS_AVAILABLE = False

class RecommendationType(Enum):
    """Types of recommendations"""
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    ITINERARY = "itinerary"
    ACTIVITY = "activity"
    ROUTE = "route"

@dataclass
class Recommendation:
    """A single recommendation item"""
    item_id: str
    name: str
    type: RecommendationType
    score: float
    reasons: List[str]
    metadata: Dict[str, Any]

@dataclass
class UserProfile:
    """User preference profile"""
    interests: List[str]
    visited_places: List[str]
    preferred_duration: str
    budget_range: str
    preferred_districts: List[str]
    travel_style: str  # cultural, adventure, relaxed, foodie, etc.

class RecommendationEngine:
    """
    Advanced recommendation system that generates personalized suggestions
    without GPT using collaborative filtering, content-based filtering,
    and rule-based algorithms.
    """
    
    def __init__(self):
        self.attractions_db = self._load_attractions_database()
        self.restaurants_db = self._load_restaurants_database()
        self.itinerary_templates = self._load_itinerary_templates()
        self.recommendation_rules = self._load_recommendation_rules()
        self.scoring_weights = self._load_scoring_weights()
        
        # Initialize advanced recommendation components
        if ADVANCED_COMPONENTS_AVAILABLE:
            try:
                self.advanced_engine = AdvancedRecommendationEngine()
                self.personalization_engine = RuleBasedPersonalizationEngine()
                print("✅ Advanced recommendation and personalization engines initialized")
            except Exception as e:
                print(f"⚠️ Advanced engines initialization failed: {e}")
                self.advanced_engine = None
                self.personalization_engine = None
        else:
            self.advanced_engine = None
            self.personalization_engine = None
        
    def _load_attractions_database(self) -> Dict[str, Any]:
        """Load attractions database with metadata for recommendations"""
        return {
            "ayasofya": {
                "name": "Ayasofya",
                "english_name": "Hagia Sophia",
                "type": "museum",
                "district": "sultanahmet",
                "interests": ["history", "culture", "architecture", "religion"],
                "visit_duration": 90,  # minutes
                "best_time": ["morning", "afternoon"],
                "crowd_level": "high",
                "nearby": ["sultanahmet_mosque", "topkapi_palace", "basilica_cistern"],
                "score_base": 9.5,
                "popularity": 0.95,
                "cultural_significance": 0.98,
                "accessibility": 0.8,
                "photo_opportunity": 0.9
            },
            
            "sultanahmet_mosque": {
                "name": "Sultanahmet Camii",
                "english_name": "Blue Mosque",
                "type": "mosque",
                "district": "sultanahmet",
                "interests": ["history", "culture", "architecture", "religion"],
                "visit_duration": 60,
                "best_time": ["morning", "afternoon"],
                "crowd_level": "high",
                "nearby": ["ayasofya", "topkapi_palace", "hippodrome"],
                "score_base": 9.2,
                "popularity": 0.92,
                "cultural_significance": 0.95,
                "accessibility": 0.85,
                "photo_opportunity": 0.95
            },
            
            "topkapi_palace": {
                "name": "Topkapı Sarayı",
                "english_name": "Topkapi Palace",
                "type": "palace",
                "district": "sultanahmet",
                "interests": ["history", "culture", "architecture", "ottoman"],
                "visit_duration": 120,
                "best_time": ["morning", "afternoon"],
                "crowd_level": "medium",
                "nearby": ["ayasofya", "sultanahmet_mosque", "gulhane_park"],
                "score_base": 9.0,
                "popularity": 0.88,
                "cultural_significance": 0.95,
                "accessibility": 0.7,
                "photo_opportunity": 0.85
            },
            
            "galata_tower": {
                "name": "Galata Kulesi",
                "english_name": "Galata Tower",
                "type": "tower",
                "district": "galata",
                "interests": ["history", "view", "photography"],
                "visit_duration": 45,
                "best_time": ["sunset", "evening"],
                "crowd_level": "medium",
                "nearby": ["galata_bridge", "karakoy", "istiklal_street"],
                "score_base": 8.5,
                "popularity": 0.85,
                "cultural_significance": 0.8,
                "accessibility": 0.6,
                "photo_opportunity": 0.95
            },
            
            "grand_bazaar": {
                "name": "Kapalıçarşı",
                "english_name": "Grand Bazaar",
                "type": "bazaar",
                "district": "beyazit",
                "interests": ["shopping", "culture", "history"],
                "visit_duration": 90,
                "best_time": ["morning", "afternoon"],
                "crowd_level": "very_high",
                "nearby": ["spice_bazaar", "sultanahmet", "beyazit_mosque"],
                "score_base": 8.8,
                "popularity": 0.9,
                "cultural_significance": 0.85,
                "accessibility": 0.7,
                "photo_opportunity": 0.8
            },
            
            "basilica_cistern": {
                "name": "Yerebatan Sarnıcı",
                "english_name": "Basilica Cistern",
                "type": "cistern",
                "district": "sultanahmet",
                "interests": ["history", "architecture", "underground"],
                "visit_duration": 45,
                "best_time": ["any"],
                "crowd_level": "medium",
                "nearby": ["ayasofya", "sultanahmet_mosque", "topkapi_palace"],
                "score_base": 8.3,
                "popularity": 0.8,
                "cultural_significance": 0.85,
                "accessibility": 0.6,
                "photo_opportunity": 0.9
            }
        }
    
    def _load_restaurants_database(self) -> Dict[str, Any]:
        """Load restaurants database with metadata"""
        return {
            "pandeli": {
                "name": "Pandeli",
                "type": "restaurant",
                "cuisine": "ottoman",
                "district": "eminonu",
                "price_range": "expensive",
                "specialties": ["ottoman cuisine", "historical"],
                "best_for": ["special_occasion", "cultural_experience"],
                "atmosphere": "elegant",
                "score_base": 9.0,
                "food_quality": 0.95,
                "service": 0.9,
                "ambiance": 0.95,
                "value": 0.7
            },
            
            "hamdi_restaurant": {
                "name": "Hamdi Restaurant",
                "type": "restaurant",
                "cuisine": "turkish",
                "district": "eminonu",
                "price_range": "moderate",
                "specialties": ["kebab", "meat dishes"],
                "best_for": ["authentic_taste", "group_dining"],
                "atmosphere": "traditional",
                "score_base": 8.7,
                "food_quality": 0.9,
                "service": 0.85,
                "ambiance": 0.8,
                "value": 0.9
            },
            
            "maiden_tower_restaurant": {
                "name": "Kız Kulesi Restaurant",
                "english_name": "Maiden's Tower Restaurant",
                "type": "restaurant",
                "cuisine": "international",
                "district": "uskudar",
                "price_range": "expensive",
                "specialties": ["seafood", "view", "romantic"],
                "best_for": ["romantic_dinner", "special_occasion"],
                "atmosphere": "romantic",
                "score_base": 8.5,
                "food_quality": 0.8,
                "service": 0.9,
                "ambiance": 0.95,
                "value": 0.6
            }
        }
    
    def _load_itinerary_templates(self) -> Dict[str, Any]:
        """Load pre-built itinerary templates"""
        return {
            "classic_1_day": {
                "name": "Classic Istanbul - 1 Day",
                "duration": "1 day",
                "theme": "essential",
                "activities": [
                    {
                        "time": "09:00",
                        "duration": 90,
                        "activity": "ayasofya",
                        "type": "attraction",
                        "notes": "Start early to avoid crowds"
                    },
                    {
                        "time": "10:45",
                        "duration": 60,
                        "activity": "sultanahmet_mosque",
                        "type": "attraction",
                        "notes": "Respect prayer times"
                    },
                    {
                        "time": "12:00",
                        "duration": 60,
                        "activity": "lunch_sultanahmet",
                        "type": "dining",
                        "notes": "Traditional Turkish lunch"
                    },
                    {
                        "time": "13:15",
                        "duration": 120,
                        "activity": "topkapi_palace",
                        "type": "attraction",
                        "notes": "Allow time for treasury and harem"
                    },
                    {
                        "time": "15:30",
                        "duration": 45,
                        "activity": "basilica_cistern",
                        "type": "attraction",
                        "notes": "Cool underground break"
                    },
                    {
                        "time": "16:30",
                        "duration": 90,
                        "activity": "grand_bazaar",
                        "type": "shopping",
                        "notes": "Shopping and souvenirs"
                    },
                    {
                        "time": "18:30",
                        "duration": 45,
                        "activity": "galata_tower",
                        "type": "attraction",
                        "notes": "Sunset views"
                    }
                ]
            },
            
            "foodie_1_day": {
                "name": "Istanbul Food Tour - 1 Day",
                "duration": "1 day",
                "theme": "food",
                "activities": [
                    {
                        "time": "09:00",
                        "duration": 60,
                        "activity": "breakfast_karakoy",
                        "type": "dining",
                        "notes": "Traditional Turkish breakfast"
                    },
                    {
                        "time": "10:30",
                        "duration": 90,
                        "activity": "spice_bazaar",
                        "type": "market",
                        "notes": "Spice shopping and tasting"
                    },
                    {
                        "time": "12:30",
                        "duration": 60,
                        "activity": "fish_sandwich_galata",
                        "type": "street_food",
                        "notes": "Famous fish sandwich"
                    },
                    {
                        "time": "14:00",
                        "duration": 90,
                        "activity": "cooking_class",
                        "type": "activity",
                        "notes": "Learn to cook Turkish dishes"
                    },
                    {
                        "time": "16:00",
                        "duration": 60,
                        "activity": "turkish_coffee_tasting",
                        "type": "cultural",
                        "notes": "UNESCO heritage coffee culture"
                    },
                    {
                        "time": "18:00",
                        "duration": 120,
                        "activity": "dinner_ottoman_cuisine",
                        "type": "dining",
                        "notes": "Traditional Ottoman dinner"
                    }
                ]
            }
        }
    
    def _load_recommendation_rules(self) -> Dict[str, Any]:
        """Load recommendation rules and logic"""
        return {
            "interest_matching": {
                "history": ["ayasofya", "sultanahmet_mosque", "topkapi_palace", "basilica_cistern"],
                "culture": ["ayasofya", "grand_bazaar", "spice_bazaar", "turkish_bath"],
                "architecture": ["ayasofya", "sultanahmet_mosque", "galata_tower"],
                "shopping": ["grand_bazaar", "spice_bazaar", "istiklal_street"],
                "food": ["spice_bazaar", "karakoy", "kadikoy_market"],
                "nightlife": ["taksim", "beyoglu", "ortakoy"],
                "nature": ["emirgan_park", "gulhane_park", "bosphorus_cruise"],
                "photography": ["galata_tower", "pierre_loti", "maiden_tower"]
            },
            
            "district_combinations": {
                "sultanahmet": ["ayasofya", "sultanahmet_mosque", "topkapi_palace", "basilica_cistern"],
                "galata": ["galata_tower", "karakoy", "galata_bridge"],
                "beyoglu": ["istiklal_street", "taksim", "pera_museum"],
                "uskudar": ["maiden_tower", "camlica_hill", "uskudar_waterfront"]
            },
            
            "time_based_rules": {
                "morning": ["ayasofya", "topkapi_palace", "grand_bazaar"],
                "afternoon": ["basilica_cistern", "spice_bazaar", "galata_bridge"],
                "evening": ["galata_tower", "bosphorus_cruise", "istiklal_street"],
                "sunset": ["galata_tower", "maiden_tower", "pierre_loti"]
            },
            
            "weather_based_rules": {
                "sunny": ["bosphorus_cruise", "parks", "outdoor_attractions"],
                "rainy": ["museums", "covered_bazaars", "indoor_activities"],
                "cold": ["turkish_bath", "covered_areas", "warm_restaurants"],
                "hot": ["underground_cisterns", "air_conditioned_museums"]
            }
        }
    
    def _load_scoring_weights(self) -> Dict[str, float]:
        """Load scoring weights for different factors"""
        return {
            "interest_match": 0.3,
            "popularity": 0.2,
            "cultural_significance": 0.15,
            "proximity": 0.1,
            "time_appropriateness": 0.1,
            "crowd_avoidance": 0.05,
            "accessibility": 0.05,
            "novelty": 0.05
        }
    
    def generate_recommendations(self, user_profile: UserProfile, 
                               recommendation_type: RecommendationType,
                               context: Dict[str, Any] = None) -> List[Recommendation]:
        """
        Generate personalized recommendations based on user profile
        
        Args:
            user_profile: User's preferences and history
            recommendation_type: Type of recommendation to generate
            context: Additional context (time, location, etc.)
            
        Returns:
            List of ranked recommendations
        """
        context = context or {}
        
        if recommendation_type == RecommendationType.ATTRACTION:
            return self._recommend_attractions(user_profile, context)
        elif recommendation_type == RecommendationType.RESTAURANT:
            return self._recommend_restaurants(user_profile, context)
        elif recommendation_type == RecommendationType.ITINERARY:
            return self._recommend_itinerary(user_profile, context)
        else:
            return []
    
    def _recommend_attractions(self, user_profile: UserProfile, 
                             context: Dict[str, Any]) -> List[Recommendation]:
        """Generate attraction recommendations"""
        recommendations = []
        
        for attraction_id, attraction_data in self.attractions_db.items():
            if attraction_id in user_profile.visited_places:
                continue  # Skip already visited
                
            score = self._calculate_attraction_score(
                attraction_data, user_profile, context
            )
            
            reasons = self._generate_recommendation_reasons(
                attraction_data, user_profile, "attraction"
            )
            
            recommendation = Recommendation(
                item_id=attraction_id,
                name=attraction_data["name"],
                type=RecommendationType.ATTRACTION,
                score=score,
                reasons=reasons,
                metadata=attraction_data
            )
            
            recommendations.append(recommendation)
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:10]
    
    def _recommend_restaurants(self, user_profile: UserProfile,
                             context: Dict[str, Any]) -> List[Recommendation]:
        """Generate restaurant recommendations"""
        recommendations = []
        
        for restaurant_id, restaurant_data in self.restaurants_db.items():
            score = self._calculate_restaurant_score(
                restaurant_data, user_profile, context
            )
            
            reasons = self._generate_recommendation_reasons(
                restaurant_data, user_profile, "restaurant"
            )
            
            recommendation = Recommendation(
                item_id=restaurant_id,
                name=restaurant_data["name"],
                type=RecommendationType.RESTAURANT,
                score=score,
                reasons=reasons,
                metadata=restaurant_data
            )
            
            recommendations.append(recommendation)
        
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:5]
    
    def _recommend_itinerary(self, user_profile: UserProfile,
                           context: Dict[str, Any]) -> List[Recommendation]:
        """Generate itinerary recommendations"""
        duration = context.get("duration", user_profile.preferred_duration)
        
        # Find matching templates
        matching_templates = []
        for template_id, template_data in self.itinerary_templates.items():
            if duration in template_data["duration"]:
                score = self._calculate_itinerary_score(
                    template_data, user_profile, context
                )
                
                recommendation = Recommendation(
                    item_id=template_id,
                    name=template_data["name"],
                    type=RecommendationType.ITINERARY,
                    score=score,
                    reasons=[f"Matches your {duration} requirement"],
                    metadata=template_data
                )
                
                matching_templates.append(recommendation)
        
        # If no templates match, generate custom itinerary
        if not matching_templates:
            custom_itinerary = self._generate_custom_itinerary(user_profile, context)
            return [custom_itinerary] if custom_itinerary else []
        
        matching_templates.sort(key=lambda x: x.score, reverse=True)
        return matching_templates[:3]
    
    def _calculate_attraction_score(self, attraction: Dict[str, Any],
                                  user_profile: UserProfile,
                                  context: Dict[str, Any]) -> float:
        """Enhanced attraction scoring with sophisticated algorithms"""
        base_score = attraction["score_base"]
        weights = self.scoring_weights
        
        # 1. Enhanced Interest Matching
        user_interests = set(user_profile.interests)
        attraction_interests = set(attraction["interests"])
        
        # Direct interest overlap
        direct_match = len(user_interests & attraction_interests)
        interest_score = weights["interest_match"] * direct_match * 1.5
        
        # Semantic interest matching
        related_interests = {
            "history": ["culture", "architecture", "museum"],
            "culture": ["history", "art", "religion"],
            "architecture": ["history", "culture", "art"],
            "art": ["culture", "museum", "architecture"],
            "photography": ["viewpoint", "scenic", "architecture"]
        }
        
        for user_int in user_interests:
            if user_int in related_interests:
                related_match = len(set(related_interests[user_int]) & attraction_interests)
                interest_score += weights["interest_match"] * related_match * 0.3
        
        # 2. Quality Score
        quality_score = (
            weights["popularity"] * attraction.get("popularity", 0.5) * 2 +
            weights["cultural_significance"] * attraction.get("cultural_significance", 0.5) * 2 +
            0.2 * attraction.get("accessibility", 0.5) * 2
        )
        
        # 3. Contextual Factors
        contextual_score = 0
        
        # Time optimization
        current_time = context.get("time_of_day", "afternoon")
        if current_time in attraction.get("best_time", []):
            contextual_score += weights["time_appropriateness"] * 1.5
        
        # District preference
        if attraction["district"] in user_profile.preferred_districts:
            contextual_score += weights["proximity"] * 2
        elif len(user_profile.preferred_districts) == 0:
            contextual_score += weights["proximity"] * 0.5
        
        # 4. Penalties
        crowd_penalties = {
            "very_high": -1.2, "high": -0.6, "medium": 0, "low": 0.3
        }
        crowd_score = weights["crowd_avoidance"] * crowd_penalties.get(
            attraction.get("crowd_level", "medium"), 0
        )
        
        # Final score
        final_score = base_score + interest_score + quality_score + contextual_score + crowd_score
        return max(0, min(10, final_score))
    
    def _calculate_restaurant_score(self, restaurant: Dict[str, Any],
                                  user_profile: UserProfile,
                                  context: Dict[str, Any]) -> float:
        """Calculate restaurant recommendation score"""
        score = restaurant["score_base"]
        
        # Budget matching
        if restaurant["price_range"] == user_profile.budget_range:
            score += 1.5
        elif (restaurant["price_range"] == "moderate" and 
              user_profile.budget_range in ["budget", "expensive"]):
            score += 0.5
        
        # Cuisine interest
        if "food" in user_profile.interests:
            score += 1.0
        
        # District preference
        if restaurant["district"] in user_profile.preferred_districts:
            score += 1.0
        
        return max(0, min(10, score))
    
    def _calculate_itinerary_score(self, itinerary: Dict[str, Any],
                                 user_profile: UserProfile,
                                 context: Dict[str, Any]) -> float:
        """Calculate itinerary recommendation score"""
        score = 5.0  # Base score
        
        # Theme matching
        theme = itinerary.get("theme", "")
        if theme in user_profile.interests or theme == user_profile.travel_style:
            score += 2.0
        
        # Activity interest matching
        activities = itinerary.get("activities", [])
        matched_interests = 0
        for activity in activities:
            activity_type = activity.get("type", "")
            if activity_type in user_profile.interests:
                matched_interests += 1
        
        score += (matched_interests / len(activities)) * 2.0 if activities else 0
        
        return max(0, min(10, score))
    
    def _generate_recommendation_reasons(self, item: Dict[str, Any],
                                       user_profile: UserProfile,
                                       item_type: str) -> List[str]:
        """Generate reasons why this item is recommended"""
        reasons = []
        
        if item_type == "attraction":
            # Interest-based reasons
            matching_interests = set(item.get("interests", [])) & set(user_profile.interests)
            if matching_interests:
                interests_str = ", ".join(matching_interests)
                reasons.append(f"Matches your interest in {interests_str}")
            
            # Popularity reasons
            if item.get("popularity", 0) > 0.9:
                reasons.append("Highly popular among visitors")
            
            # Cultural significance
            if item.get("cultural_significance", 0) > 0.9:
                reasons.append("Significant cultural and historical importance")
            
            # Photo opportunity
            if item.get("photo_opportunity", 0) > 0.9:
                reasons.append("Great photo opportunities")
        
        elif item_type == "restaurant":
            # Budget matching
            if item.get("price_range") == user_profile.budget_range:
                reasons.append(f"Fits your {user_profile.budget_range} budget")
            
            # Cuisine matching
            if "food" in user_profile.interests:
                reasons.append("Perfect for food enthusiasts")
            
            # Special atmosphere
            atmosphere = item.get("atmosphere", "")
            if atmosphere in ["romantic", "elegant", "traditional"]:
                reasons.append(f"Offers a {atmosphere} atmosphere")
        
        # Default reason if no specific reasons
        if not reasons:
            reasons.append("Recommended based on your preferences")
        
        return reasons
    
    def _generate_custom_itinerary(self, user_profile: UserProfile,
                                 context: Dict[str, Any]) -> Optional[Recommendation]:
        """Generate a custom itinerary based on user preferences"""
        duration = context.get("duration", "1 day")
        
        # Get recommended attractions
        attraction_recs = self._recommend_attractions(user_profile, context)
        if not attraction_recs:
            return None
        
        # Create custom itinerary
        activities = []
        current_time = datetime.strptime("09:00", "%H:%M")
        
        for i, rec in enumerate(attraction_recs[:5]):  # Limit to top 5
            activity = {
                "time": current_time.strftime("%H:%M"),
                "duration": rec.metadata.get("visit_duration", 60),
                "activity": rec.item_id,
                "type": "attraction",
                "notes": rec.reasons[0] if rec.reasons else ""
            }
            activities.append(activity)
            
            # Add time for next activity
            current_time += timedelta(minutes=activity["duration"] + 30)  # 30 min buffer
        
        custom_itinerary_data = {
            "name": f"Custom {duration} Itinerary",
            "duration": duration,
            "theme": "personalized",
            "activities": activities
        }
        
        return Recommendation(
            item_id="custom_itinerary",
            name=custom_itinerary_data["name"],
            type=RecommendationType.ITINERARY,
            score=8.0,
            reasons=["Personalized based on your preferences"],
            metadata=custom_itinerary_data
        )
    
    def create_user_profile(self, preferences: Dict[str, Any]) -> UserProfile:
        """Create user profile from preferences"""
        return UserProfile(
            interests=preferences.get("interests", []),
            visited_places=preferences.get("visited_places", []),
            preferred_duration=preferences.get("duration", "1 day"),
            budget_range=preferences.get("budget", "moderate"),
            preferred_districts=preferences.get("districts", []),
            travel_style=preferences.get("travel_style", "cultural")
        )
    
    def get_advanced_recommendations(self, user_profile: UserProfile, 
                                   context: Optional[Dict[str, Any]] = None,
                                   recommendation_type: str = "attraction",
                                   n_recommendations: int = 5) -> List[Recommendation]:
        """
        Get advanced recommendations using collaborative filtering and personalization
        """
        if not self.advanced_engine or not self.personalization_engine:
            # Fallback to basic recommendations
            return self.get_recommendations(user_profile, context, recommendation_type, n_recommendations)
        
        try:
            # Create user context for personalization
            user_context = self._create_user_context(context or {})
            
            # Use collaborative filtering for base recommendations
            user_id = context.get('user_id', 'anonymous_user') if context else 'anonymous_user'
            
            # Get candidates from different sources
            candidates = []
            
            if recommendation_type == "attraction":
                # Get collaborative filtering recommendations
                collab_recs = self.advanced_engine.collaborative_filtering_recommend(
                    user_id, n_recommendations=15
                )
                candidates.extend(self._convert_advanced_to_candidates(collab_recs))
                
                # Get content-based recommendations
                content_recs = self.advanced_engine.content_based_recommend(
                    user_id, n_recommendations=15
                )
                candidates.extend(self._convert_advanced_to_candidates(content_recs))
                
                # Get location-based recommendations if location available
                if context and 'location' in context:
                    location = context['location']
                    location_recs = self.advanced_engine.location_based_recommend(
                        user_id, current_location=location, n_recommendations=10
                    )
                    candidates.extend(self._convert_advanced_to_candidates(location_recs))
            
            # If no advanced candidates, use basic database
            if not candidates:
                candidates = self._get_basic_candidates(recommendation_type, user_profile)
            
            # Apply personalization rules
            personalization_result = self.personalization_engine.personalize_recommendations(
                user_id, candidates, user_context
            )
            
            # Convert to standard Recommendation format
            recommendations = []
            for rec_data in personalization_result.recommendations[:n_recommendations]:
                recommendation = Recommendation(
                    item_id=rec_data.get('id', rec_data.get('item_id', 'unknown')),
                    name=rec_data.get('name', 'Unknown'),
                    type=RecommendationType.ATTRACTION if recommendation_type == "attraction" else RecommendationType.RESTAURANT,
                    score=rec_data.get('personalization_score', rec_data.get('overall_score', 0.5)),
                    reasons=rec_data.get('preference_reasons', rec_data.get('reasons', [])),
                    metadata={
                        'advanced_recommendation': True,
                        'collaborative_score': rec_data.get('collaborative_score', 0),
                        'content_score': rec_data.get('content_score', 0),
                        'location_score': rec_data.get('location_score', 0),
                        'confidence': personalization_result.confidence_score,
                        'applied_rules': personalization_result.applied_rules,
                        'explanations': personalization_result.explanation
                    }
                )
                recommendations.append(recommendation)
            
            print(f"✅ Generated {len(recommendations)} advanced recommendations with {personalization_result.confidence_score:.2f} confidence")
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Advanced recommendations failed, falling back to basic: {e}")
            return self.get_recommendations(user_profile, context, recommendation_type, n_recommendations)
    
    def _create_user_context(self, context_data: Dict[str, Any]) -> 'UserContext':
        """Create UserContext object from context data"""
        current_time = datetime.now()
        
        # Determine time of day
        hour = current_time.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Determine season
        month = current_time.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "autumn"
        
        return UserContext(
            time_of_day=context_data.get('time_of_day', time_of_day),
            day_of_week=current_time.strftime('%A').lower(),
            season=context_data.get('season', season),
            weather=context_data.get('weather', 'sunny'),
            temperature_c=context_data.get('temperature', 20),
            location=context_data.get('location'),
            group_size=context_data.get('group_size', 2),
            budget_range=context_data.get('budget_range', 'moderate'),
            available_time_hours=context_data.get('available_time_hours', 4.0),
            mobility_level=context_data.get('mobility_level', 'moderate'),
            special_occasions=context_data.get('special_occasions', []),
            energy_level=context_data.get('energy_level', 'moderate')
        )
    
    def _convert_advanced_to_candidates(self, advanced_recs: List['AdvancedRecommendation']) -> List[Dict[str, Any]]:
        """Convert AdvancedRecommendation objects to candidate dictionaries"""
        candidates = []
        
        for rec in advanced_recs:
            candidate = {
                'id': rec.item_id,
                'item_id': rec.item_id,
                'name': rec.name,
                'type': rec.type,
                'base_score': rec.overall_score,
                'overall_score': rec.overall_score,
                'collaborative_score': rec.collaborative_score,
                'content_score': rec.content_score,
                'location_score': rec.location_score,
                'confidence': rec.confidence,
                'reasons': rec.reasons,
                'features': rec.metadata.get('features', []) if hasattr(rec.metadata, 'get') else [],
                'duration_minutes': rec.metadata.get('duration_minutes', 60) if hasattr(rec.metadata, 'get') else 60,
                'cost': rec.metadata.get('cost', 15) if hasattr(rec.metadata, 'get') else 15
            }
            candidates.append(candidate)
        
        return candidates
    
    def _get_basic_candidates(self, recommendation_type: str, user_profile: UserProfile) -> List[Dict[str, Any]]:
        """Get basic candidates from database when advanced recommendations are not available"""
        candidates = []
        
        if recommendation_type == "attraction":
            for item_id, item_data in self.attractions_db.items():
                candidate = {
                    'id': item_id,
                    'item_id': item_id,
                    'name': item_data.get('name', item_id),
                    'type': item_data.get('type', 'attraction'),
                    'base_score': item_data.get('score_base', 0.5),
                    'features': item_data.get('features', []),
                    'duration_minutes': item_data.get('duration_minutes', 60),
                    'cost': item_data.get('cost', 15),
                    'district': item_data.get('district', 'unknown')
                }
                candidates.append(candidate)
        
        elif recommendation_type == "restaurant":
            for item_id, item_data in self.restaurants_db.items():
                candidate = {
                    'id': item_id,
                    'item_id': item_id,
                    'name': item_data.get('name', item_id),
                    'type': 'restaurant',
                    'cuisine': item_data.get('cuisine_type', 'turkish'),
                    'base_score': item_data.get('score_base', 0.5),
                    'features': item_data.get('features', []),
                    'duration_minutes': 90,  # Default restaurant visit time
                    'cost': item_data.get('price_range_numeric', 25),
                    'district': item_data.get('district', 'unknown')
                }
                candidates.append(candidate)
        
        return candidates
    
    def add_user_feedback(self, user_id: str, item_id: str, feedback: Dict[str, Any]):
        """Add user feedback to improve future recommendations"""
        if self.personalization_engine:
            try:
                # Convert feedback to behavior data
                behavior_data = {
                    'place_id': item_id,
                    'place_type': feedback.get('type', 'attraction'),
                    'rating': feedback.get('rating', 3.0),
                    'duration_minutes': feedback.get('duration_minutes', 60),
                    'time_of_day': feedback.get('time_of_day', 'afternoon'),
                    'liked': feedback.get('liked', feedback.get('rating', 3.0) >= 4.0),
                    'context': feedback.get('context', {})
                }
                
                self.personalization_engine.add_user_behavior(user_id, behavior_data)
                print(f"✅ Added user feedback for {item_id} from user {user_id}")
                
            except Exception as e:
                print(f"⚠️ Failed to add user feedback: {e}")
    
    def get_user_profile_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user preferences and behavior"""
        if self.personalization_engine:
            try:
                return self.personalization_engine.get_user_profile_summary(user_id)
            except Exception as e:
                print(f"⚠️ Failed to get user profile insights: {e}")
        
        return {
            'user_id': user_id,
            'message': 'Advanced profiling not available',
            'basic_profile': True
        }
