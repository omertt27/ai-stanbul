#!/usr/bin/env python3
"""
Context-Aware Ranking & Filtering System
========================================

Uses session context to dynamically filter and rank results based on:
- Location context (show results nearby)
- User preferences (budget, ratings, cuisine)
- Conversation history (previously mentioned places)
- Intent continuity (restaurant search → filter restaurants)

This creates the illusion of "conversation memory" without requiring GPT.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LocationContext:
    """User's current location context from conversation"""
    primary_location: Optional[str] = None  # Main area they're asking about
    nearby_landmarks: List[str] = None  # Mentioned landmarks
    district: Optional[str] = None  # Current district of interest
    coordinates: Optional[Tuple[float, float]] = None  # Lat/lon if available
    radius_km: float = 2.0  # Default search radius
    
    def __post_init__(self):
        if self.nearby_landmarks is None:
            self.nearby_landmarks = []

@dataclass
class UserPreferences:
    """User preferences extracted from conversation"""
    budget_level: str = "medium"  # low, medium, high
    preferred_cuisines: List[str] = None
    min_rating: float = 3.0
    max_distance_km: float = 3.0
    avoid_crowded: bool = False
    prefer_authentic: bool = False
    dietary_restrictions: List[str] = None
    preferred_districts: List[str] = None
    
    def __post_init__(self):
        if self.preferred_cuisines is None:
            self.preferred_cuisines = []
        if self.dietary_restrictions is None:
            self.dietary_restrictions = []
        if self.preferred_districts is None:
            self.preferred_districts = []

@dataclass
class FilteringCriteria:
    """Combined filtering criteria from context"""
    location_context: LocationContext
    user_preferences: UserPreferences
    intent_focus: str  # restaurant, museum, attraction, etc.
    conversation_entities: Dict[str, List[str]]
    exclude_mentioned: bool = False  # Exclude already mentioned places
    boost_related: bool = True  # Boost similar places to mentioned ones

class ContextAwareFilteringSystem:
    """Main system for context-aware filtering and ranking"""
    
    def __init__(self):
        self.istanbul_districts = {
            'sultanahmet': {'lat': 41.0086, 'lon': 28.9798, 'keywords': ['hagia sophia', 'blue mosque', 'topkapi']},
            'beyoglu': {'lat': 41.0369, 'lon': 28.9784, 'keywords': ['galata tower', 'istiklal', 'taksim']},
            'galata': {'lat': 41.0258, 'lon': 28.9743, 'keywords': ['galata tower', 'karakoy', 'galata bridge']},
            'kadikoy': {'lat': 40.9893, 'lon': 29.0297, 'keywords': ['moda', 'bahariye', 'ferry']},
            'besiktas': {'lat': 41.0422, 'lon': 29.0033, 'keywords': ['dolmabahce', 'ortakoy', 'bosphorus']},
            'fatih': {'lat': 41.0186, 'lon': 28.9647, 'keywords': ['grand bazaar', 'eminonu', 'suleymaniye']},
            'uskudar': {'lat': 41.0214, 'lon': 29.0128, 'keywords': ['maiden tower', 'camlica', 'asian side']},
            'sisli': {'lat': 41.0602, 'lon': 28.9897, 'keywords': ['nisantasi', 'mecidiyekoy', 'shopping']},
        }
        
        self.cuisine_preferences = {
            'turkish': ['kebab', 'meze', 'pide', 'baklava', 'turkish coffee'],
            'ottoman': ['ottoman', 'imperial', 'palace', 'traditional'],
            'seafood': ['fish', 'seafood', 'meze', 'rakı', 'bosphorus'],
            'street_food': ['simit', 'döner', 'kokoreç', 'balık ekmek', 'midye'],
            'international': ['italian', 'french', 'asian', 'mediterranean', 'fusion'],
            'vegetarian': ['vegetarian', 'vegan', 'plant-based', 'meze', 'salad'],
        }
        
        self.budget_indicators = {
            'low': ['cheap', 'budget', 'affordable', 'inexpensive', 'student'],
            'medium': ['moderate', 'reasonable', 'mid-range', 'decent'],
            'high': ['expensive', 'upscale', 'fine dining', 'luxury', 'premium'],
        }
    
    def extract_location_context(self, conversation_stack, current_query: str) -> LocationContext:
        """Extract location context from conversation and current query"""
        
        # Initialize location context
        location_context = LocationContext()
        
        # Check current query for location mentions
        query_lower = current_query.lower()
        
        # Extract district mentions
        for district, info in self.istanbul_districts.items():
            if district in query_lower:
                location_context.district = district
                location_context.primary_location = district
                location_context.coordinates = (info['lat'], info['lon'])
                break
        
        # Extract landmark mentions
        landmarks = ['hagia sophia', 'blue mosque', 'galata tower', 'topkapi palace', 
                    'grand bazaar', 'bosphorus', 'taksim square', 'istiklal street']
        
        for landmark in landmarks:
            if landmark in query_lower:
                location_context.nearby_landmarks.append(landmark)
                
                # Map landmarks to districts
                if landmark in ['hagia sophia', 'blue mosque', 'topkapi palace']:
                    location_context.district = 'sultanahmet'
                elif landmark in ['galata tower']:
                    location_context.district = 'galata'
                elif landmark in ['taksim square', 'istiklal street']:
                    location_context.district = 'beyoglu'
                elif landmark in ['grand bazaar']:
                    location_context.district = 'fatih'
        
        # Extract from conversation history
        if conversation_stack and conversation_stack.turns:
            for turn in conversation_stack.turns[-3:]:  # Last 3 turns
                for district in self.istanbul_districts.keys():
                    if district in turn.user_query.lower() or district in turn.ai_response.lower():
                        if not location_context.district:
                            location_context.district = district
                            location_context.primary_location = district
        
        # Extract distance preferences
        distance_patterns = [
            r'within\s+(\d+)\s*(?:km|kilometer)',
            r'(\d+)\s*(?:km|kilometer)\s+radius',
            r'close\s+to|nearby|near',
            r'walking\s+distance'
        ]
        
        for pattern in distance_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if match.groups():
                    try:
                        location_context.radius_km = float(match.group(1))
                    except (ValueError, IndexError):
                        pass
                elif 'walking' in pattern:
                    location_context.radius_km = 1.0  # Walking distance
                elif 'close' in pattern or 'nearby' in pattern:
                    location_context.radius_km = 1.5  # Nearby
        
        return location_context
    
    def extract_user_preferences(self, conversation_stack, current_query: str) -> UserPreferences:
        """Extract user preferences from conversation"""
        
        preferences = UserPreferences()
        query_lower = current_query.lower()
        
        # Extract budget preferences
        for budget_level, indicators in self.budget_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                preferences.budget_level = budget_level
                break
        
        # Extract cuisine preferences
        for cuisine, keywords in self.cuisine_preferences.items():
            if any(keyword in query_lower for keyword in keywords):
                preferences.preferred_cuisines.append(cuisine)
        
        # Extract rating preferences
        rating_patterns = [
            r'(?:rating|rated)\s+(?:above|over|more than)\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s+star',
            r'highly?\s+rated',
            r'best\s+rated'
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if match.groups():
                    try:
                        preferences.min_rating = float(match.group(1))
                    except (ValueError, IndexError):
                        pass
                elif 'highly' in match.group(0) or 'best' in match.group(0):
                    preferences.min_rating = 4.0
        
        # Extract other preferences
        if any(word in query_lower for word in ['authentic', 'traditional', 'local']):
            preferences.prefer_authentic = True
        
        if any(word in query_lower for word in ['quiet', 'peaceful', 'not crowded', 'avoid crowds']):
            preferences.avoid_crowded = True
        
        # Extract dietary restrictions
        dietary_keywords = ['vegetarian', 'vegan', 'halal', 'kosher', 'gluten-free', 'dairy-free']
        for dietary in dietary_keywords:
            if dietary in query_lower:
                preferences.dietary_restrictions.append(dietary)
        
        # Extract from conversation history
        if conversation_stack and conversation_stack.turns:
            for turn in conversation_stack.turns:
                turn_query = turn.user_query.lower()
                
                # Accumulate cuisine preferences
                for cuisine, keywords in self.cuisine_preferences.items():
                    if any(keyword in turn_query for keyword in keywords):
                        if cuisine not in preferences.preferred_cuisines:
                            preferences.preferred_cuisines.append(cuisine)
                
                # Update budget if mentioned
                for budget_level, indicators in self.budget_indicators.items():
                    if any(indicator in turn_query for indicator in indicators):
                        preferences.budget_level = budget_level
        
        return preferences
    
    def create_filtering_criteria(self, conversation_stack, current_query: str, intent: str) -> FilteringCriteria:
        """Create comprehensive filtering criteria"""
        
        location_context = self.extract_location_context(conversation_stack, current_query)
        user_preferences = self.extract_user_preferences(conversation_stack, current_query)
        
        # Extract conversation entities
        entities = {}
        if conversation_stack:
            for turn in conversation_stack.turns:
                for key, values in turn.entities.items():
                    if key not in entities:
                        entities[key] = []
                    entities[key].extend(values)
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
        
        return FilteringCriteria(
            location_context=location_context,
            user_preferences=user_preferences,
            intent_focus=intent,
            conversation_entities=entities,
            exclude_mentioned=False,  # Can be configured
            boost_related=True
        )
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km (simplified)"""
        # Simplified distance calculation for Istanbul area
        # For production, use proper geospatial libraries
        lat_diff = abs(lat1 - lat2)
        lon_diff = abs(lat1 - lon2)
        
        # Rough conversion for Istanbul area (latitude ~41°)
        km_per_lat = 111.0
        km_per_lon = 111.0 * 0.75  # Adjust for latitude
        
        distance = ((lat_diff * km_per_lat) ** 2 + (lon_diff * km_per_lon) ** 2) ** 0.5
        return distance
    
    def filter_restaurants(self, restaurants: List[Dict[str, Any]], criteria: FilteringCriteria) -> List[Dict[str, Any]]:
        """Filter restaurants based on context-aware criteria"""
        
        filtered = []
        
        for restaurant in restaurants:
            # Location filtering
            if criteria.location_context.district:
                restaurant_district = restaurant.get('district', '').lower()
                if criteria.location_context.district.lower() not in restaurant_district:
                    # Check if it's within radius if coordinates available
                    if criteria.location_context.coordinates:
                        rest_coords = self._get_restaurant_coordinates(restaurant)
                        if rest_coords:
                            distance = self.calculate_distance(
                                criteria.location_context.coordinates[0],
                                criteria.location_context.coordinates[1],
                                rest_coords[0], rest_coords[1]
                            )
                            if distance > criteria.location_context.radius_km:
                                continue  # Skip this restaurant
                        else:
                            continue  # Skip if no location match and no coordinates
                    else:
                        continue  # Skip if not in preferred district
            
            # Budget filtering
            if criteria.user_preferences.budget_level != "medium":
                restaurant_budget = self._estimate_restaurant_budget(restaurant)
                if criteria.user_preferences.budget_level == "low" and restaurant_budget == "high":
                    continue
                elif criteria.user_preferences.budget_level == "high" and restaurant_budget == "low":
                    continue
            
            # Cuisine filtering
            if criteria.user_preferences.preferred_cuisines:
                restaurant_cuisine = restaurant.get('cuisine', '').lower()
                restaurant_description = restaurant.get('description', '').lower()
                
                cuisine_match = False
                for preferred_cuisine in criteria.user_preferences.preferred_cuisines:
                    cuisine_keywords = self.cuisine_preferences.get(preferred_cuisine, [preferred_cuisine])
                    if any(keyword in restaurant_cuisine or keyword in restaurant_description 
                           for keyword in cuisine_keywords):
                        cuisine_match = True
                        break
                
                if not cuisine_match:
                    continue
            
            # Rating filtering
            restaurant_rating = restaurant.get('rating', 3.0)
            if isinstance(restaurant_rating, str):
                try:
                    restaurant_rating = float(restaurant_rating)
                except ValueError:
                    restaurant_rating = 3.0
            
            if restaurant_rating < criteria.user_preferences.min_rating:
                continue
            
            # Dietary restrictions
            if criteria.user_preferences.dietary_restrictions:
                restaurant_description = restaurant.get('description', '').lower()
                dietary_match = False
                for dietary in criteria.user_preferences.dietary_restrictions:
                    if dietary in restaurant_description:
                        dietary_match = True
                        break
                
                # If user has dietary restrictions but restaurant doesn't mention them, be cautious
                if not dietary_match and len(criteria.user_preferences.dietary_restrictions) > 0:
                    # Only skip if it's a strict dietary requirement
                    if any(strict in criteria.user_preferences.dietary_restrictions 
                           for strict in ['vegan', 'kosher', 'gluten-free']):
                        continue
            
            filtered.append(restaurant)
        
        return filtered
    
    def rank_results(self, results: List[Dict[str, Any]], criteria: FilteringCriteria) -> List[Dict[str, Any]]:
        """Rank results based on context-aware scoring"""
        
        scored_results = []
        
        for result in results:
            score = 0.0
            
            # Base score from rating
            rating = result.get('rating', 3.0)
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except ValueError:
                    rating = 3.0
            score += rating * 20  # Max 100 points from rating
            
            # Location proximity bonus
            if criteria.location_context.district:
                result_district = result.get('district', '').lower()
                if criteria.location_context.district.lower() in result_district:
                    score += 30  # District match bonus
                
                # Landmark proximity bonus
                result_description = result.get('description', '').lower()
                for landmark in criteria.location_context.nearby_landmarks:
                    if landmark.lower() in result_description:
                        score += 15  # Landmark proximity bonus
            
            # Cuisine preference bonus
            if criteria.user_preferences.preferred_cuisines:
                result_cuisine = result.get('cuisine', '').lower()
                result_description = result.get('description', '').lower()
                
                for preferred_cuisine in criteria.user_preferences.preferred_cuisines:
                    cuisine_keywords = self.cuisine_preferences.get(preferred_cuisine, [preferred_cuisine])
                    if any(keyword in result_cuisine or keyword in result_description 
                           for keyword in cuisine_keywords):
                        score += 25  # Cuisine preference bonus
                        break
            
            # Budget alignment bonus
            result_budget = self._estimate_restaurant_budget(result)
            if result_budget == criteria.user_preferences.budget_level:
                score += 20  # Budget alignment bonus
            
            # Authenticity bonus
            if criteria.user_preferences.prefer_authentic:
                result_description = result.get('description', '').lower()
                authentic_keywords = ['authentic', 'traditional', 'local', 'family-run', 'generations']
                if any(keyword in result_description for keyword in authentic_keywords):
                    score += 15  # Authenticity bonus
            
            # Conversation relevance bonus
            if criteria.conversation_entities:
                result_name = result.get('name', '').lower()
                result_description = result.get('description', '').lower()
                
                for entity_type, entity_values in criteria.conversation_entities.items():
                    for entity in entity_values:
                        if entity.lower() in result_name or entity.lower() in result_description:
                            score += 10  # Conversation relevance bonus
            
            # Avoid crowded bonus
            if criteria.user_preferences.avoid_crowded:
                result_description = result.get('description', '').lower()
                quiet_keywords = ['quiet', 'peaceful', 'hidden', 'small', 'intimate']
                crowded_keywords = ['busy', 'crowded', 'popular', 'touristy']
                
                if any(keyword in result_description for keyword in quiet_keywords):
                    score += 10
                elif any(keyword in result_description for keyword in crowded_keywords):
                    score -= 10
            
            scored_results.append((result, score))
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result for result, score in scored_results]
    
    def _get_restaurant_coordinates(self, restaurant: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Get restaurant coordinates (placeholder for actual implementation)"""
        # In a real implementation, this would look up coordinates from the database
        # or use the restaurant's district to estimate location
        
        district = restaurant.get('district', '').lower()
        for dist_name, dist_info in self.istanbul_districts.items():
            if dist_name in district:
                return (dist_info['lat'], dist_info['lon'])
        
        return None
    
    def _estimate_restaurant_budget(self, restaurant: Dict[str, Any]) -> str:
        """Estimate restaurant budget level from description"""
        
        description = restaurant.get('description', '').lower()
        name = restaurant.get('name', '').lower()
        
        # High-end indicators
        high_end_keywords = ['fine dining', 'upscale', 'luxury', 'michelin', 'premium', 'expensive', 'exclusive']
        if any(keyword in description or keyword in name for keyword in high_end_keywords):
            return "high"
        
        # Budget indicators
        budget_keywords = ['cheap', 'budget', 'affordable', 'inexpensive', 'student', 'local', 'street']
        if any(keyword in description or keyword in name for keyword in budget_keywords):
            return "low"
        
        # Default to medium
        return "medium"
    
    def apply_context_filtering(self, results: List[Dict[str, Any]], conversation_stack, 
                              current_query: str, intent: str) -> List[Dict[str, Any]]:
        """Main method to apply context-aware filtering and ranking"""
        
        if not results:
            return results
        
        # Create filtering criteria
        criteria = self.create_filtering_criteria(conversation_stack, current_query, intent)
        
        # Apply filtering
        filtered_results = results
        if intent in ['restaurant_search', 'food_recommendation']:
            filtered_results = self.filter_restaurants(results, criteria)
        
        # Apply ranking
        ranked_results = self.rank_results(filtered_results, criteria)
        
        # Log the filtering process
        logger.info(f"Context filtering: {len(results)} → {len(filtered_results)} → ranked")
        logger.info(f"Criteria: district={criteria.location_context.district}, "
                   f"budget={criteria.user_preferences.budget_level}, "
                   f"cuisines={criteria.user_preferences.preferred_cuisines}")
        
        return ranked_results

# Export main class
__all__ = ['ContextAwareFilteringSystem', 'LocationContext', 'UserPreferences', 'FilteringCriteria']
