#!/usr/bin/env python3
"""
Advanced Restaurant Recommendation Service for Istanbul AI
Provides sophisticated restaurant matching with multiple criteria
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from ..core.user_profile import UserProfile
from ..core.conversation_context import ConversationContext

logger = logging.getLogger(__name__)

class CuisineType(Enum):
    """Restaurant cuisine types"""
    TURKISH_TRADITIONAL = "turkish_traditional"
    TURKISH_MODERN = "turkish_modern"
    OTTOMAN = "ottoman"
    MEDITERRANEAN = "mediterranean"
    SEAFOOD = "seafood"
    ITALIAN = "italian"
    FRENCH = "french"
    ASIAN = "asian"
    INTERNATIONAL = "international"
    STREET_FOOD = "street_food"
    FINE_DINING = "fine_dining"
    MEYHANE = "meyhane"
    KEBAB = "kebab"
    MEZE = "meze"

class DietaryRequirement(Enum):
    """Dietary requirements and restrictions"""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    GLUTEN_FREE = "gluten_free"
    LACTOSE_FREE = "lactose_free"
    DIABETIC_FRIENDLY = "diabetic_friendly"
    PREGNANCY_SAFE = "pregnancy_safe"
    LOW_SODIUM = "low_sodium"
    KETO = "keto"
    PALEO = "paleo"

class PriceRange(Enum):
    """Price ranges for restaurants"""
    BUDGET = "$"           # 0-50 TL per person
    MODERATE = "$$"        # 50-150 TL per person
    UPSCALE = "$$$"        # 150-300 TL per person
    LUXURY = "$$$$"        # 300+ TL per person

class AmbientType(Enum):
    """Restaurant ambience types"""
    CASUAL = "casual"
    ROMANTIC = "romantic"
    FAMILY_FRIENDLY = "family_friendly"
    BUSINESS = "business"
    TRENDY = "trendy"
    HISTORIC = "historic"
    WATERFRONT = "waterfront"
    ROOFTOP = "rooftop"
    COZY = "cozy"
    LIVELY = "lively"

@dataclass
class RestaurantFeatures:
    """Restaurant features and amenities"""
    outdoor_seating: bool = False
    wheelchair_accessible: bool = False
    live_music: bool = False
    wifi: bool = True
    parking: bool = False
    delivery: bool = False
    reservations_required: bool = False
    alcohol_served: bool = True
    smoking_area: bool = False
    credit_cards: bool = True
    english_menu: bool = False
    view: Optional[str] = None  # "bosphorus", "city", "garden", etc.

@dataclass
class Restaurant:
    """Complete restaurant information"""
    name: str
    district: str
    cuisine_types: List[CuisineType]
    price_range: PriceRange
    ambient_type: AmbientType
    features: RestaurantFeatures
    dietary_options: List[DietaryRequirement]
    signature_dishes: List[str]
    address: str
    phone: Optional[str] = None
    website: Optional[str] = None
    rating: float = 0.0
    review_count: int = 0
    opening_hours: Dict[str, str] = None
    description: str = ""
    special_notes: List[str] = None

@dataclass
class RestaurantRecommendation:
    """Restaurant recommendation with matching score"""
    restaurant: Restaurant
    matching_score: float  # 0.0 to 1.0
    match_reasons: List[str]
    warnings: List[str]
    alternative_suggestions: List[str]

class AdvancedRestaurantService:
    """
    Advanced restaurant recommendation service with sophisticated matching
    """
    
    def __init__(self):
        self.logger = logger
        self.restaurants = self._load_restaurant_database()
        
        # Cuisine keywords mapping
        self.cuisine_keywords = {
            CuisineType.TURKISH_TRADITIONAL: ['traditional', 'authentic', 'turkish', 'ottoman', 'lokanta'],
            CuisineType.TURKISH_MODERN: ['modern turkish', 'contemporary', 'fusion turkish'],
            CuisineType.SEAFOOD: ['seafood', 'fish', 'balik', 'sea', 'maritime'],
            CuisineType.MEYHANE: ['meyhane', 'meze', 'raki', 'traditional music'],
            CuisineType.KEBAB: ['kebab', 'grill', 'ocakbasi', 'meat'],
            CuisineType.STREET_FOOD: ['street food', 'fast', 'quick', 'doner', 'sandwich'],
            CuisineType.FINE_DINING: ['fine dining', 'upscale', 'elegant', 'luxury', 'michelin'],
            CuisineType.INTERNATIONAL: ['international', 'world cuisine', 'fusion']
        }
        
        # Dietary keywords mapping
        self.dietary_keywords = {
            DietaryRequirement.VEGETARIAN: ['vegetarian', 'veggie', 'no meat'],
            DietaryRequirement.VEGAN: ['vegan', 'plant based', 'no dairy'],
            DietaryRequirement.GLUTEN_FREE: ['gluten free', 'celiac', 'no gluten'],
            DietaryRequirement.LACTOSE_FREE: ['lactose free', 'dairy free', 'no dairy'],
            DietaryRequirement.DIABETIC_FRIENDLY: ['diabetic', 'sugar free', 'low sugar'],
            DietaryRequirement.PREGNANCY_SAFE: ['pregnant', 'pregnancy', 'expecting']
        }
        
        # Ambient keywords mapping
        self.ambient_keywords = {
            AmbientType.ROMANTIC: ['romantic', 'date', 'intimate', 'cozy'],
            AmbientType.FAMILY_FRIENDLY: ['family', 'kids', 'children', 'child-friendly'],
            AmbientType.BUSINESS: ['business', 'meeting', 'professional', 'quiet'],
            AmbientType.TRENDY: ['trendy', 'hip', 'modern', 'stylish'],
            AmbientType.WATERFRONT: ['waterfront', 'sea view', 'bosphorus', 'water'],
            AmbientType.ROOFTOP: ['rooftop', 'terrace', 'view', 'panoramic']
        }

    def get_recommendations(
        self,
        user_input: str,
        user_profile: UserProfile,
        location: Optional[str] = None,
        max_recommendations: int = 5
    ) -> List[RestaurantRecommendation]:
        """
        Get sophisticated restaurant recommendations based on user query and profile
        """
        
        # Parse user requirements
        requirements = self._parse_user_requirements(user_input, user_profile)
        
        # Filter restaurants by location if specified
        candidate_restaurants = self._filter_by_location(location) if location else self.restaurants
        
        # Score and rank restaurants
        scored_restaurants = []
        
        for restaurant in candidate_restaurants:
            score, reasons, warnings = self._calculate_matching_score(restaurant, requirements, user_profile)
            
            if score > 0.2:  # Minimum relevance threshold
                scored_restaurants.append(
                    RestaurantRecommendation(
                        restaurant=restaurant,
                        matching_score=score,
                        match_reasons=reasons,
                        warnings=warnings,
                        alternative_suggestions=[]
                    )
                )
        
        # Sort by score and return top recommendations
        scored_restaurants.sort(key=lambda x: x.matching_score, reverse=True)
        top_recommendations = scored_restaurants[:max_recommendations]
        
        # Add alternative suggestions
        for rec in top_recommendations:
            rec.alternative_suggestions = self._get_alternatives(rec.restaurant, candidate_restaurants[:10])
        
        return top_recommendations

    def _parse_user_requirements(self, user_input: str, user_profile: UserProfile) -> Dict[str, Any]:
        """Parse user requirements from input and profile"""
        user_input_lower = user_input.lower()
        
        requirements = {
            'cuisines': [],
            'dietary': [],
            'price_range': None,
            'ambient': [],
            'features': [],
            'time_of_day': None,
            'group_size': None,
            'special_occasion': None
        }
        
        # Detect cuisine preferences
        for cuisine, keywords in self.cuisine_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                requirements['cuisines'].append(cuisine)
        
        # Detect dietary requirements
        for dietary, keywords in self.dietary_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                requirements['dietary'].append(dietary)
        
        # Detect ambient preferences
        for ambient, keywords in self.ambient_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                requirements['ambient'].append(ambient)
        
        # Detect price preferences
        if any(word in user_input_lower for word in ['cheap', 'budget', 'affordable']):
            requirements['price_range'] = PriceRange.BUDGET
        elif any(word in user_input_lower for word in ['expensive', 'luxury', 'upscale', 'fine']):
            requirements['price_range'] = PriceRange.LUXURY
        elif any(word in user_input_lower for word in ['mid-range', 'moderate']):
            requirements['price_range'] = PriceRange.MODERATE
        
        # Detect time context
        if any(word in user_input_lower for word in ['breakfast', 'morning']):
            requirements['time_of_day'] = 'breakfast'
        elif any(word in user_input_lower for word in ['lunch', 'afternoon']):
            requirements['time_of_day'] = 'lunch'
        elif any(word in user_input_lower for word in ['dinner', 'evening', 'night']):
            requirements['time_of_day'] = 'dinner'
        
        # Detect group context
        if any(word in user_input_lower for word in ['group', 'friends', 'party']):
            requirements['group_size'] = 'large'
        elif any(word in user_input_lower for word in ['couple', 'two', 'date']):
            requirements['group_size'] = 'couple'
        elif any(word in user_input_lower for word in ['solo', 'alone', 'myself']):
            requirements['group_size'] = 'solo'
        
        # Detect special occasions
        if any(word in user_input_lower for word in ['birthday', 'anniversary', 'celebration']):
            requirements['special_occasion'] = 'celebration'
        elif any(word in user_input_lower for word in ['business', 'meeting', 'work']):
            requirements['special_occasion'] = 'business'
        
        # Detect features
        if any(word in user_input_lower for word in ['outdoor', 'terrace', 'garden']):
            requirements['features'].append('outdoor_seating')
        if any(word in user_input_lower for word in ['accessible', 'wheelchair']):
            requirements['features'].append('wheelchair_accessible')
        if any(word in user_input_lower for word in ['music', 'live']):
            requirements['features'].append('live_music')
        if any(word in user_input_lower for word in ['view', 'scenic']):
            requirements['features'].append('view')
        
        return requirements

    def _calculate_matching_score(
        self, 
        restaurant: Restaurant, 
        requirements: Dict[str, Any], 
        user_profile: UserProfile
    ) -> Tuple[float, List[str], List[str]]:
        """Calculate how well a restaurant matches user requirements"""
        
        score = 0.0
        reasons = []
        warnings = []
        max_score = 0.0
        
        # Cuisine matching (weight: 0.3)
        cuisine_weight = 0.3
        max_score += cuisine_weight
        if requirements['cuisines']:
            cuisine_matches = len(set(requirements['cuisines']) & set(restaurant.cuisine_types))
            if cuisine_matches > 0:
                cuisine_score = min(1.0, cuisine_matches / len(requirements['cuisines']))
                score += cuisine_score * cuisine_weight
                reasons.append(f"Matches {cuisine_matches} of your cuisine preferences")
            else:
                warnings.append("Doesn't match your preferred cuisine types")
        else:
            # If no specific cuisine requested, give partial credit
            score += cuisine_weight * 0.5
        
        # Dietary requirements (weight: 0.25)
        dietary_weight = 0.25
        max_score += dietary_weight
        if requirements['dietary']:
            dietary_matches = len(set(requirements['dietary']) & set(restaurant.dietary_options))
            if dietary_matches == len(requirements['dietary']):
                score += dietary_weight
                reasons.append("Meets all your dietary requirements")
            elif dietary_matches > 0:
                score += (dietary_matches / len(requirements['dietary'])) * dietary_weight
                reasons.append(f"Meets {dietary_matches} of {len(requirements['dietary'])} dietary requirements")
                warnings.append("May not meet all dietary requirements - please verify")
            else:
                warnings.append("Doesn't explicitly cater to your dietary requirements")
        else:
            score += dietary_weight * 0.7  # Default score if no dietary requirements
        
        # Price range matching (weight: 0.15)
        price_weight = 0.15
        max_score += price_weight
        if requirements['price_range']:
            if restaurant.price_range == requirements['price_range']:
                score += price_weight
                reasons.append(f"Matches your {requirements['price_range'].value} budget")
            else:
                # Partial credit for adjacent price ranges
                price_levels = [PriceRange.BUDGET, PriceRange.MODERATE, PriceRange.UPSCALE, PriceRange.LUXURY]
                req_idx = price_levels.index(requirements['price_range'])
                rest_idx = price_levels.index(restaurant.price_range)
                if abs(req_idx - rest_idx) == 1:
                    score += price_weight * 0.6
                    reasons.append(f"Close to your budget range ({restaurant.price_range.value})")
                else:
                    warnings.append(f"Price range ({restaurant.price_range.value}) may not match your budget")
        else:
            score += price_weight * 0.8  # Default if no price preference
        
        # Ambient matching (weight: 0.15)
        ambient_weight = 0.15
        max_score += ambient_weight
        if requirements['ambient']:
            if restaurant.ambient_type in requirements['ambient']:
                score += ambient_weight
                reasons.append(f"Perfect {restaurant.ambient_type.value} atmosphere")
            else:
                warnings.append(f"Atmosphere is {restaurant.ambient_type.value}, not exactly what you requested")
        else:
            score += ambient_weight * 0.7
        
        # Features matching (weight: 0.1)
        features_weight = 0.1
        max_score += features_weight
        if requirements['features']:
            feature_matches = 0
            for feature in requirements['features']:
                if hasattr(restaurant.features, feature) and getattr(restaurant.features, feature):
                    feature_matches += 1
            
            if feature_matches > 0:
                feature_score = feature_matches / len(requirements['features'])
                score += feature_score * features_weight
                reasons.append(f"Has {feature_matches} of your requested features")
            else:
                warnings.append("May not have all the features you're looking for")
        else:
            score += features_weight * 0.8
        
        # User profile preferences (weight: 0.05)
        profile_weight = 0.05
        max_score += profile_weight
        
        # Add user type considerations
        if hasattr(user_profile, 'user_type') and user_profile.user_type:
            from ..utils.constants import UserType
            
            if user_profile.user_type == UserType.FIRST_TIME_VISITOR:
                # Tourist-friendly bonus
                if restaurant.features.english_menu or 'tourist' in restaurant.description.lower():
                    score += profile_weight
                    reasons.append("Tourist-friendly restaurant")
            elif user_profile.user_type == UserType.LOCAL_RESIDENT:
                # Authentic/local bonus
                if 'local' in restaurant.description.lower() or 'authentic' in restaurant.description.lower():
                    score += profile_weight
                    reasons.append("Popular with locals")
        
        # Normalize score
        final_score = score / max_score if max_score > 0 else 0
        
        return final_score, reasons, warnings

    def _filter_by_location(self, location: str) -> List[Restaurant]:
        """Filter restaurants by location"""
        return [r for r in self.restaurants if r.district.lower() == location.lower()]

    def _get_alternatives(self, restaurant: Restaurant, candidates: List[Restaurant]) -> List[str]:
        """Get alternative restaurant suggestions"""
        alternatives = []
        
        # Find restaurants with similar cuisine
        similar_cuisine = [
            r for r in candidates 
            if r.name != restaurant.name and 
            any(cuisine in restaurant.cuisine_types for cuisine in r.cuisine_types)
        ][:2]
        
        alternatives.extend([r.name for r in similar_cuisine])
        
        # Find restaurants in same price range
        same_price = [
            r for r in candidates
            if r.name != restaurant.name and 
            r.price_range == restaurant.price_range and
            r not in similar_cuisine
        ][:1]
        
        alternatives.extend([r.name for r in same_price])
        
        return alternatives[:3]

    def _load_restaurant_database(self) -> List[Restaurant]:
        """Load restaurant database - this would normally come from a database"""
        # For demo purposes, returning a curated list of Istanbul restaurants
        restaurants = [
            # Sultanahmet
            Restaurant(
                name="Pandeli",
                district="Sultanahmet",
                cuisine_types=[CuisineType.OTTOMAN, CuisineType.TURKISH_TRADITIONAL],
                price_range=PriceRange.UPSCALE,
                ambient_type=AmbientType.HISTORIC,
                features=RestaurantFeatures(
                    wheelchair_accessible=False,
                    wifi=True,
                    reservations_required=True,
                    english_menu=True,
                    view="spice_bazaar"
                ),
                dietary_options=[DietaryRequirement.HALAL, DietaryRequirement.VEGETARIAN],
                signature_dishes=["Ottoman Lamb Stew", "Stuffed Grape Leaves", "Turkish Delight"],
                address="Mısır Çarşısı No: 1, Eminönü",
                description="Historic Ottoman restaurant above the Spice Bazaar with traditional recipes dating back to 1901",
                special_notes=["Reservations recommended", "Historic venue"]
            ),
            
            Restaurant(
                name="Seven Hills Restaurant",
                district="Sultanahmet",
                cuisine_types=[CuisineType.TURKISH_MODERN, CuisineType.INTERNATIONAL],
                price_range=PriceRange.LUXURY,
                ambient_type=AmbientType.ROMANTIC,
                features=RestaurantFeatures(
                    outdoor_seating=True,
                    wheelchair_accessible=True,
                    wifi=True,
                    reservations_required=True,
                    english_menu=True,
                    view="blue_mosque"
                ),
                dietary_options=[DietaryRequirement.VEGETARIAN, DietaryRequirement.GLUTEN_FREE],
                signature_dishes=["Sea Bass with herbs", "Turkish mezze platter", "Baklava"],
                address="Tevkifhane Sk. No:8/A, Sultanahmet",
                description="Rooftop restaurant with stunning Blue Mosque views and modern Turkish cuisine"
            ),
            
            # Beyoğlu
            Restaurant(
                name="Mikla",
                district="Beyoğlu",
                cuisine_types=[CuisineType.TURKISH_MODERN, CuisineType.FINE_DINING],
                price_range=PriceRange.LUXURY,
                ambient_type=AmbientType.TRENDY,
                features=RestaurantFeatures(
                    outdoor_seating=True,
                    wheelchair_accessible=True,
                    wifi=True,
                    reservations_required=True,
                    english_menu=True,
                    view="bosphorus"
                ),
                dietary_options=[DietaryRequirement.VEGETARIAN, DietaryRequirement.GLUTEN_FREE, DietaryRequirement.LACTOSE_FREE],
                signature_dishes=["New Anatolian cuisine", "Seasonal tasting menu", "Local ingredient focus"],
                address="Mezzanine Floor, The Marmara Pera, Meşrutiyet Cd. No:15",
                description="Award-winning restaurant featuring New Anatolian cuisine with panoramic city views",
                special_notes=["Michelin recommended", "Advance reservations essential"]
            ),
            
            Restaurant(
                name="Nevizade Sokak Meyhanes",
                district="Beyoğlu",
                cuisine_types=[CuisineType.MEYHANE, CuisineType.MEZE],
                price_range=PriceRange.MODERATE,
                ambient_type=AmbientType.LIVELY,
                features=RestaurantFeatures(
                    outdoor_seating=True,
                    live_music=True,
                    wifi=False,
                    alcohol_served=True,
                    smoking_area=True
                ),
                dietary_options=[DietaryRequirement.HALAL, DietaryRequirement.VEGETARIAN],
                signature_dishes=["Mixed meze platter", "Grilled fish", "Rakı pairing"],
                address="Nevizade Sk., Beyoğlu",
                description="Traditional meyhane street with authentic Turkish tavern atmosphere and live music"
            ),
            
            # Kadıköy
            Restaurant(
                name="Çiya Sofrası",
                district="Kadıköy",
                cuisine_types=[CuisineType.TURKISH_TRADITIONAL],
                price_range=PriceRange.MODERATE,
                ambient_type=AmbientType.CASUAL,
                features=RestaurantFeatures(
                    wheelchair_accessible=True,
                    wifi=True,
                    credit_cards=True,
                    english_menu=False
                ),
                dietary_options=[DietaryRequirement.HALAL, DietaryRequirement.VEGETARIAN, DietaryRequirement.VEGAN],
                signature_dishes=["Regional Turkish dishes", "Antakya cuisine", "Traditional desserts"],
                address="Güneşlibahçe Sk. No:43, Kadıköy",
                description="Authentic Turkish regional cuisine restaurant popular with locals",
                special_notes=["Cash only", "No reservations", "Local favorite"]
            ),
            
            # Beşiktaş
            Restaurant(
                name="Sur Balık",
                district="Beşiktaş",
                cuisine_types=[CuisineType.SEAFOOD],
                price_range=PriceRange.UPSCALE,
                ambient_type=AmbientType.WATERFRONT,
                features=RestaurantFeatures(
                    outdoor_seating=True,
                    wheelchair_accessible=True,
                    wifi=True,
                    reservations_required=True,
                    view="bosphorus"
                ),
                dietary_options=[DietaryRequirement.GLUTEN_FREE],
                signature_dishes=["Fresh daily catch", "Sea bass in salt", "Seafood meze"],
                address="Muallim Naci Cd. No:54, Beşiktaş",
                description="Premium seafood restaurant with Bosphorus views"
            )
        ]
        
        return restaurants

    def format_recommendation(self, recommendation: RestaurantRecommendation, include_details: bool = True) -> str:
        """Format a restaurant recommendation for user display"""
        restaurant = recommendation.restaurant
        
        # Header with name and basic info
        output = f"**{restaurant.name}** {restaurant.price_range.value}\n"
        output += f"   {restaurant.description}\n"
        output += f"   🍽️ Cuisine: {', '.join([c.value.replace('_', ' ').title() for c in restaurant.cuisine_types])}\n"
        output += f"   📍 Location: {restaurant.district}\n"
        
        if include_details:
            # Matching reasons
            if recommendation.match_reasons:
                output += f"   ✅ Match: {', '.join(recommendation.match_reasons[:2])}\n"
            
            # Signature dishes
            if restaurant.signature_dishes:
                output += f"   🌟 Specialties: {', '.join(restaurant.signature_dishes[:3])}\n"
            
            # Special features
            features = []
            if restaurant.features.outdoor_seating:
                features.append("Outdoor seating")
            if restaurant.features.view:
                features.append(f"{restaurant.features.view.replace('_', ' ').title()} view")
            if restaurant.features.live_music:
                features.append("Live music")
            if restaurant.features.wheelchair_accessible:
                features.append("Wheelchair accessible")
            
            if features:
                output += f"   🎯 Features: {', '.join(features[:3])}\n"
            
            # Warnings
            if recommendation.warnings:
                output += f"   ⚠️ Note: {recommendation.warnings[0]}\n"
        
        return output

    def get_cuisine_suggestions(self, location: Optional[str] = None) -> Dict[str, List[str]]:
        """Get cuisine type suggestions for a location"""
        restaurants = self._filter_by_location(location) if location else self.restaurants
        
        cuisine_map = {}
        for restaurant in restaurants:
            for cuisine in restaurant.cuisine_types:
                cuisine_name = cuisine.value.replace('_', ' ').title()
                if cuisine_name not in cuisine_map:
                    cuisine_map[cuisine_name] = []
                cuisine_map[cuisine_name].append(restaurant.name)
        
        return {k: v[:3] for k, v in cuisine_map.items()}  # Top 3 per cuisine
