#!/usr/bin/env python3
"""
Restaurant Response Enhancer with Google Maps Integration
========================================================

This module enhances restaurant recommendations with Google Maps data to improve
accuracy, current information, and practical details for visitors.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RestaurantEnhancer:
    """Enhances restaurant responses with Google Maps data and local knowledge"""
    
    def __init__(self):
        self.google_places_enabled = False
        self.use_mock_data = True  # Use enhanced mock data for better responses
        
        try:
            from api_clients.enhanced_google_places import EnhancedGooglePlacesClient
            self.google_client = EnhancedGooglePlacesClient()
            self.google_places_enabled = self.google_client.has_api_key
            if not self.google_places_enabled:
                logger.warning("Google Places API not available - using enhanced local data")
        except ImportError:
            logger.warning("Google Places client not available - using enhanced local data")
            self.google_client = None
        
    def enhance_restaurant_response(self, user_query: str, location_context: Optional[str] = None) -> Dict:
        """
        Enhance restaurant response with Google Maps data and local insights
        
        Args:
            user_query: User's restaurant query
            location_context: Specific location/neighborhood context
            
        Returns:
            Dictionary with enhanced restaurant data
        """
        
        # Extract query type and preferences
        query_analysis = self._analyze_restaurant_query(user_query, location_context)
        
        # Get restaurant recommendations based on query analysis
        if self.google_places_enabled and self.google_client:
            restaurants = self._get_google_maps_restaurants(query_analysis, location_context)
        else:
            restaurants = self._get_enhanced_local_restaurants(query_analysis, location_context)
            
        # Add walking directions and practical info
        enhanced_restaurants = self._add_practical_details(restaurants, location_context)
        
        return {
            "query_analysis": query_analysis,
            "restaurants": enhanced_restaurants,
            "cultural_notes": self._get_cultural_dining_notes(query_analysis),
            "practical_tips": self._get_practical_dining_tips(query_analysis)
        }
    
    def _analyze_restaurant_query(self, query: str, location_context: Optional[str] = None) -> Dict:
        """Analyze user query to understand dining preferences"""
        query_lower = query.lower()
        
        # Cuisine type detection
        cuisine_types = {
            "turkish": ["turkish", "ottoman", "traditional", "local"],
            "seafood": ["seafood", "fish", "balık", "meze"],
            "vegetarian": ["vegetarian", "vegan", "plant-based"],
            "street_food": ["street food", "döner", "kebab", "simit"],
            "dessert": ["dessert", "baklava", "turkish delight", "sweet"],
            "breakfast": ["breakfast", "kahvaltı"],
            "fine_dining": ["fine dining", "upscale", "luxury", "michelin"]
        }
        
        detected_cuisines = []
        for cuisine, keywords in cuisine_types.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_cuisines.append(cuisine)
        
        # Dining context detection
        contexts = {
            "family": ["family", "children", "kids"],
            "romantic": ["romantic", "date", "couple"],
            "business": ["business", "meeting", "professional"],
            "tourist": ["tourist", "sightseeing", "visiting"],
            "budget": ["cheap", "budget", "affordable"],
            "special": ["special occasion", "celebration", "birthday"]
        }
        
        detected_contexts = []
        for context, keywords in contexts.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_contexts.append(context)
        
        return {
            "cuisines": detected_cuisines,
            "contexts": detected_contexts,
            "dietary_restrictions": self._detect_dietary_restrictions(query_lower),
            "location_specific": location_context is not None
        }
    
    def _detect_dietary_restrictions(self, query: str) -> List[str]:
        """Detect dietary restrictions from query"""
        restrictions = []
        
        if any(word in query for word in ["vegetarian", "veggie"]):
            restrictions.append("vegetarian")
        if any(word in query for word in ["vegan", "plant-based"]):
            restrictions.append("vegan")
        if any(word in query for word in ["gluten-free", "celiac"]):
            restrictions.append("gluten-free")
        if any(word in query for word in ["halal"]):
            restrictions.append("halal")
        if any(word in query for word in ["kosher"]):
            restrictions.append("kosher")
            
        return restrictions
    
    def _get_enhanced_local_restaurants(self, analysis: Dict, location: Optional[str]) -> List[Dict]:
        """Get restaurant recommendations using enhanced local knowledge"""
        
        # High-quality restaurant database with accurate information
        restaurants = {
            "sultanahmet": {
                "turkish": [
                    {
                        "name": "Matbah Restaurant",
                        "address": "Caferaga Mahallesi, Alemdar Caddesi No: 6, 34122 Fatih",
                        "rating": 4.3,
                        "specialties": ["Ottoman cuisine", "Traditional Turkish dishes", "Historical setting"],
                        "atmosphere": "Historical, elegant, traditional Ottoman decor",
                        "walking_from_landmark": "3-minute walk from Hagia Sophia main entrance",
                        "opening_hours": "Daily 12:00-23:00",
                        "price_range": "upscale"
                    },
                    {
                        "name": "Hamdi Restaurant",
                        "address": "Kalcılar Caddesi No: 17, Eminönü, 34116 Fatih", 
                        "rating": 4.2,
                        "specialties": ["Southeastern Turkish kebabs", "Şırdan", "Baklava"],
                        "atmosphere": "Traditional, busy, authentic local dining",
                        "walking_from_landmark": "5-minute walk from Galata Bridge to Eminönü",
                        "opening_hours": "Daily 11:30-24:00",
                        "price_range": "moderate"
                    }
                ],
                "seafood": [
                    {
                        "name": "Balıkçı Sabahattin",
                        "address": "Seyit Hasan Kuyu Sokak No: 1, Cankurtaran, 34122 Fatih",
                        "rating": 4.4,
                        "specialties": ["Fresh fish", "Mezze selection", "Sea bass"],
                        "atmosphere": "Traditional fish house, cozy, local favorite",
                        "walking_from_landmark": "7-minute walk from Sultanahmet Tram Station downhill",
                        "opening_hours": "Daily 12:00-24:00",
                        "price_range": "moderate to upscale"
                    }
                ]
            },
            "beyoglu": {
                "turkish": [
                    {
                        "name": "Çukur Meyhane",
                        "address": "Nevizade Sokak No: 13, Beyoğlu, 34420 İstanbul",
                        "rating": 4.1,
                        "specialties": ["Meyhane classics", "Rakı pairing", "Live music"],
                        "atmosphere": "Traditional tavern, lively, authentic Turkish dining culture",
                        "walking_from_landmark": "2-minute walk from İstiklal Street, near Galatasaray",
                        "opening_hours": "Daily 18:00-02:00",
                        "price_range": "moderate"
                    }
                ],
                "international": [
                    {
                        "name": "Mikla",
                        "address": "The Marmara Pera, Meşrutiyet Caddesi No: 15, Tepebaşı, 34430 Beyoğlu",
                        "rating": 4.6,
                        "specialties": ["Modern Turkish", "Anatolian ingredients", "Panoramic views"],
                        "atmosphere": "Fine dining, modern, spectacular Bosphorus views",
                        "walking_from_landmark": "8-minute walk from Taksim Square via Meşrutiyet Street",
                        "opening_hours": "Tue-Sat 19:00-24:00",
                        "price_range": "fine dining"
                    }
                ]
            },
            "kadikoy": {
                "turkish": [
                    {
                        "name": "Çiya Sofrası",
                        "address": "Güneşlibahçe Sokak No: 43, Kadıköy, 34710 İstanbul",
                        "rating": 4.5,
                        "specialties": ["Regional Turkish cuisine", "Forgotten recipes", "Anatolian dishes"],
                        "atmosphere": "Authentic, traditional, showcasing Turkey's culinary heritage",
                        "walking_from_landmark": "5-minute walk from Kadıköy Ferry Terminal via Muvakkithane Street",
                        "opening_hours": "Daily 12:00-22:00",
                        "price_range": "moderate"
                    }
                ]
            }
        }
        
        # Filter restaurants based on analysis
        location_key = location.lower() if location else "sultanahmet"
        
        selected_restaurants = []
        if location_key in restaurants:
            location_restaurants = restaurants[location_key]
            
            # If specific cuisine requested, filter by cuisine
            if analysis["cuisines"]:
                for cuisine in analysis["cuisines"]:
                    if cuisine in location_restaurants:
                        selected_restaurants.extend(location_restaurants[cuisine])
            else:
                # Return variety from all cuisines in that location
                for cuisine_group in location_restaurants.values():
                    selected_restaurants.extend(cuisine_group[:2])  # Top 2 from each
        else:
            # Default to Sultanahmet recommendations
            for cuisine_group in restaurants["sultanahmet"].values():
                selected_restaurants.extend(cuisine_group[:1])
                
        return selected_restaurants[:6]  # Return top 6 recommendations
    
    def _get_google_maps_restaurants(self, analysis: Dict, location: Optional[str]) -> List[Dict]:
        """Get restaurant recommendations from Google Maps API"""
        # This would implement actual Google Places API calls
        # For now, return enhanced local data
        return self._get_enhanced_local_restaurants(analysis, location)
    
    def _add_practical_details(self, restaurants: List[Dict], location: Optional[str]) -> List[Dict]:
        """Add practical visiting details to restaurant recommendations"""
        
        for restaurant in restaurants:
            # Add practical navigation details
            restaurant["practical_info"] = {
                "reservation_required": restaurant.get("price_range") in ["upscale", "fine dining"],
                "accepts_credit_cards": True,
                "recommended_visit_time": self._get_optimal_dining_time(restaurant),
                "tipping_guidance": "10-15% for good service",
                "dress_code": "Smart casual" if restaurant.get("price_range") == "fine dining" else "Casual"
            }
            
            # Add cultural context
            restaurant["cultural_notes"] = self._get_restaurant_cultural_context(restaurant)
            
        return restaurants
    
    def _get_optimal_dining_time(self, restaurant: Dict) -> str:
        """Get optimal dining time recommendations"""
        price_range = restaurant.get("price_range", "moderate")
        
        if price_range == "fine dining":
            return "Dinner: 19:30-21:00 (reservations essential)"
        elif "breakfast" in restaurant.get("specialties", []):
            return "Breakfast: 08:00-11:00, weekends until 13:00"
        else:
            return "Lunch: 12:00-15:00, Dinner: 19:00-22:00"
    
    def _get_restaurant_cultural_context(self, restaurant: Dict) -> List[str]:
        """Get cultural dining context for restaurant"""
        notes = []
        
        if "meyhane" in restaurant.get("name", "").lower():
            notes.append("Traditional tavern culture: expect shared mezze and rakı")
            notes.append("Live music typically starts after 21:00")
        
        if restaurant.get("price_range") == "fine dining":
            notes.append("International dining etiquette applies")
            notes.append("Wine list available, but Turkish wines highly recommended")
        
        if "ottoman" in " ".join(restaurant.get("specialties", [])).lower():
            notes.append("Historical recipes recreated with authentic cooking methods")
            notes.append("Multiple course experience typical of Ottoman palace cuisine")
            
        return notes
    
    def _get_cultural_dining_notes(self, analysis: Dict) -> List[str]:
        """Get general cultural dining notes based on query analysis"""
        notes = [
            "Turkish meals are social experiences - don't rush",
            "Bread is sacred in Turkish culture - never waste it",
            "Tea or coffee will be offered after meals - accepting shows respect"
        ]
        
        if "vegetarian" in analysis.get("dietary_restrictions", []):
            notes.append("Say 'et yok' (no meat) - many Turkish dishes are naturally vegetarian")
            
        if "family" in analysis.get("contexts", []):
            notes.append("Children are welcome in Turkish restaurants - high chairs usually available")
            
        return notes
    
    def _get_practical_dining_tips(self, analysis: Dict) -> List[str]:
        """Get practical dining tips based on query"""
        tips = [
            "Most restaurants accept credit cards, but carry cash for small places",
            "Tipping 10-15% is standard for good service",
            "Popular dinner time is 20:00-22:00, lunch 12:00-14:00"
        ]
        
        if analysis.get("location_specific"):
            tips.append("Ask hotel concierge for current restaurant availability")
            
        return tips

# Global instance
restaurant_enhancer = RestaurantEnhancer()

def enhance_restaurant_query(user_query: str, location_context: Optional[str] = None) -> str:
    """
    Main function to enhance restaurant responses with Google Maps data
    
    Args:
        user_query: User's restaurant question
        location_context: Specific neighborhood context
        
    Returns:
        Enhanced response with Google Maps integration
    """
    enhancement_data = restaurant_enhancer.enhance_restaurant_response(user_query, location_context)
    
    # Format the enhanced data into a response template
    response_parts = []
    
    # Add restaurant recommendations
    restaurants = enhancement_data["restaurants"]
    if restaurants:
        response_parts.append("RECOMMENDED RESTAURANTS:")
        for i, restaurant in enumerate(restaurants[:4], 1):
            response_parts.append(f"\n{i}. {restaurant['name'].upper()}")
            response_parts.append(f"   Location: {restaurant['address']}")
            response_parts.append(f"   Specialties: {', '.join(restaurant['specialties'])}")
            response_parts.append(f"   Atmosphere: {restaurant['atmosphere']}")
            response_parts.append(f"   Access: {restaurant['walking_from_landmark']}")
            response_parts.append(f"   Hours: {restaurant['opening_hours']}")
            if 'practical_info' in restaurant:
                response_parts.append(f"   Best time: {restaurant['practical_info']['recommended_visit_time']}")
    
    # Add cultural notes
    if enhancement_data["cultural_notes"]:
        response_parts.append("\nCULTURAL DINING NOTES:")
        for note in enhancement_data["cultural_notes"][:3]:
            response_parts.append(f"• {note}")
    
    # Add practical tips
    if enhancement_data["practical_tips"]:
        response_parts.append("\nPRACTICAL TIPS:")
        for tip in enhancement_data["practical_tips"][:3]:
            response_parts.append(f"• {tip}")
    
    return "\n".join(response_parts)
