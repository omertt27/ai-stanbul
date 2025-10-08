#!/usr/bin/env python3
"""
Restaurant Integration Service for AI Chat System
Integrates the 143-restaurant mock database with AI responses
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

@dataclass
class RestaurantRecommendation:
    """Individual restaurant recommendation"""
    name: str
    district: str
    cuisine: str
    rating: float
    price_level: int
    budget: str
    description: str
    address: str
    place_id: str
    lat: float = None
    lng: float = None

class RestaurantIntegrationService:
    """Service to integrate restaurant database with AI chat responses"""
    
    def __init__(self):
        self.restaurants_data = None
        self.load_restaurant_data()
        
    def load_restaurant_data(self):
        """Load restaurant data from Google Places mock data"""
        try:
            from api_clients.google_places import GooglePlacesClient
            client = GooglePlacesClient("fake_key")  # Force mock data
            result = client._get_mock_restaurant_data()
            self.restaurants_data = result.get("results", [])
            print(f"✅ Loaded {len(self.restaurants_data)} restaurants for AI chat integration")
        except Exception as e:
            print(f"❌ Error loading restaurant data: {e}")
            self.restaurants_data = []
    
    def search_restaurants(self, 
                          district: Optional[str] = None,
                          cuisine: Optional[str] = None,
                          budget: Optional[str] = None,
                          keyword: Optional[str] = None,
                          limit: int = 5) -> List[RestaurantRecommendation]:
        """Search restaurants based on criteria"""
        if not self.restaurants_data:
            return []
        
        filtered_restaurants = self.restaurants_data.copy()
        
        # Filter by district
        if district:
            district_normalized = district.lower().strip()
            # Handle common district variations
            district_mapping = {
                'beyoglu': 'beyoğlu',
                'taksim': 'beyoğlu',
                'galata': 'beyoğlu',
                'karakoy': 'beyoğlu',
                'sultanahmet': 'fatih',
                'eminonu': 'fatih',
                'eminönü': 'fatih',
                'kadikoy': 'kadıköy',
                'besiktas': 'beşiktaş'
            }
            
            search_district = district_mapping.get(district_normalized, district_normalized)
            
            filtered_restaurants = [
                r for r in filtered_restaurants
                if search_district in r.get("vicinity", "").lower()
            ]
        
        # Filter by cuisine
        if cuisine:
            cuisine_normalized = cuisine.lower().strip()
            # Handle cuisine variations
            cuisine_mapping = {
                'kebab': 'turkish',
                'kebaps': 'turkish',
                'meze': 'turkish',
                'ottoman': 'turkish',
                'anatolian': 'turkish',
                'sushi': 'japanese',
                'pizza': 'italian',
                'pasta': 'italian'
            }
            
            search_cuisine = cuisine_mapping.get(cuisine_normalized, cuisine_normalized)
            
            filtered_restaurants = [
                r for r in filtered_restaurants
                if search_cuisine in r.get("cuisine", "").lower()
            ]
        
        # Filter by budget
        if budget:
            budget_normalized = budget.lower().strip()
            budget_mapping = {
                'cheap': 'budget',
                'affordable': 'budget',
                'expensive': 'luxury',
                'high-end': 'luxury',
                'moderate': 'mid-range',
                'premium': 'premium'
            }
            
            search_budget = budget_mapping.get(budget_normalized, budget_normalized)
            
            filtered_restaurants = [
                r for r in filtered_restaurants
                if r.get("budget", "").lower() == search_budget
            ]
        
        # Keyword search in name or description
        if keyword:
            keyword_lower = keyword.lower()
            filtered_restaurants = [
                r for r in filtered_restaurants
                if (keyword_lower in r.get("name", "").lower() or
                    keyword_lower in r.get("description", "").lower())
            ]
        
        # Sort by rating (highest first)
        filtered_restaurants.sort(key=lambda x: x.get("rating", 0), reverse=True)
        
        # Convert to RestaurantRecommendation objects
        recommendations = []
        for restaurant in filtered_restaurants[:limit]:
            try:
                location = restaurant.get("geometry", {}).get("location", {})
                
                recommendation = RestaurantRecommendation(
                    name=restaurant.get("name", "Unknown Restaurant"),
                    district=self._extract_district(restaurant.get("vicinity", "")),
                    cuisine=restaurant.get("cuisine", "International"),
                    rating=restaurant.get("rating", 4.0),
                    price_level=restaurant.get("price_level", 2),
                    budget=restaurant.get("budget", "mid-range"),
                    description=restaurant.get("description", "Great restaurant in Istanbul"),
                    address=restaurant.get("vicinity", "Istanbul"),
                    place_id=restaurant.get("place_id", ""),
                    lat=location.get("lat"),
                    lng=location.get("lng")
                )
                recommendations.append(recommendation)
            except Exception as e:
                print(f"⚠️ Error processing restaurant: {e}")
                continue
        
        return recommendations
    
    def _extract_district(self, vicinity: str) -> str:
        """Extract district name from vicinity string"""
        if not vicinity:
            return "Istanbul"
        
        # Extract first part before comma as district
        parts = vicinity.split(",")
        if parts:
            return parts[0].strip()
        return "Istanbul"
    
    def format_restaurant_response(self, 
                                 restaurants: List[RestaurantRecommendation],
                                 query_context: Dict[str, Any] = None) -> str:
        """Format restaurant recommendations into a conversational response"""
        if not restaurants:
            return self._get_no_results_response()
        
        # Determine response style based on number of results
        if len(restaurants) == 1:
            return self._format_single_restaurant(restaurants[0])
        else:
            return self._format_multiple_restaurants(restaurants, query_context)
    
    def _format_single_restaurant(self, restaurant: RestaurantRecommendation) -> str:
        """Format single restaurant recommendation"""
        budget_emoji = {
            'budget': '💰',
            'mid-range': '💰💰', 
            'premium': '💰💰💰',
            'luxury': '💰💰💰💰'
        }
        
        budget_text = {
            'budget': 'Budget-friendly (your wallet will thank you!)',
            'mid-range': 'Mid-range (great value)',
            'premium': 'Premium dining (worth the splurge)',
            'luxury': 'Luxury dining (special occasion territory)'
        }
        
        excitement_phrases = [
            "I'm so excited to recommend",
            "You're going to love",
            "I found the perfect spot for you:",
            "Here's an amazing place I think you'll adore:"
        ]
        
        import random
        excitement = random.choice(excitement_phrases)
        
        return f"""{excitement} **{restaurant.name}** in {restaurant.district}! 

{restaurant.description}

📍 **Where to find it:** {restaurant.address}
⭐ **Rating:** {restaurant.rating}/5 (that's really solid!)
{budget_emoji.get(restaurant.budget, '💰')} **Price level:** {budget_text.get(restaurant.budget, 'Mid-range (great value)')}
🍴 **Cuisine:** {restaurant.cuisine}

This place really delivers on authentic flavors and the service is genuinely welcoming. I think you're going to have a fantastic time here!"""
    
    def _format_multiple_restaurants(self, 
                                   restaurants: List[RestaurantRecommendation],
                                   query_context: Dict[str, Any] = None) -> str:
        """Format multiple restaurant recommendations"""
        
        # Group by district for better organization
        by_district = {}
        for restaurant in restaurants:
            district = restaurant.district
            if district not in by_district:
                by_district[district] = []
            by_district[district].append(restaurant)
        
        intro_phrases = [
            f"Perfect! I've found {len(restaurants)} fantastic restaurants that match what you're looking for:",
            f"Great news! Here are {len(restaurants)} amazing spots I'd recommend:",
            f"You're in luck! I've got {len(restaurants)} excellent restaurant suggestions for you:",
            f"Here are {len(restaurants)} wonderful restaurants that I think you'll really enjoy:"
        ]
        
        import random
        response = random.choice(intro_phrases) + "\n\n"
        
        for i, restaurant in enumerate(restaurants, 1):
            budget_emoji = {
                'budget': '💰',
                'mid-range': '💰💰', 
                'premium': '💰💰💰',
                'luxury': '💰💰💰💰'
            }
            
            response += f"**{i}. {restaurant.name}** in {restaurant.district}\n"
            response += f"⭐ {restaurant.rating}/5 • {budget_emoji.get(restaurant.budget, '💰')} {restaurant.budget.title()} • 🍴 {restaurant.cuisine}\n"
            response += f"{restaurant.description}\n\n"
        
        response += "Each of these places has its own character and charm. Would you like more details about any of them?"
        
        return response.strip()
    
    def _get_no_results_response(self) -> str:
        """Get response when no restaurants match criteria"""
        fallback_restaurants = self.search_restaurants(limit=3)  # Get top 3 popular restaurants
        
        if fallback_restaurants:
            response = "Hmm, I couldn't find restaurants that exactly match your specific criteria, but don't worry! Here are some absolutely fantastic options that I think you'll love:\n\n"
            response += self._format_multiple_restaurants(fallback_restaurants)
            return response
        
        return """You know what? Even though I couldn't find specific matches for your request right now, Istanbul is an absolute food paradise! Let me tell you what makes this city's dining scene so incredible:

🥘 **Traditional Turkish cuisine** - The kebabs, meze, and Turkish breakfasts are mind-blowing
🐟 **Fresh seafood** along the Bosphorus - caught daily and prepared with centuries-old techniques  
🌍 **International cuisines** in Beyoğlu and Nişantaşı - world-class dining from every corner of the globe
🥙 **Street food** in Kadıköy and Eminönü - some of the best meals you'll have will cost less than $5!

Tell me a bit more about what you're craving or which neighborhood you're in, and I'll find you something amazing. I'm really good at matching people with their perfect Istanbul food experience!"""
    
    def get_restaurant_stats(self) -> Dict[str, Any]:
        """Get statistics about the restaurant database"""
        if not self.restaurants_data:
            return {"total": 0}
        
        stats = {
            "total": len(self.restaurants_data),
            "by_district": {},
            "by_cuisine": {},
            "by_budget": {},
            "average_rating": 0
        }
        
        total_rating = 0
        for restaurant in self.restaurants_data:
            # District stats
            district = self._extract_district(restaurant.get("vicinity", ""))
            stats["by_district"][district] = stats["by_district"].get(district, 0) + 1
            
            # Cuisine stats
            cuisine = restaurant.get("cuisine", "Unknown")
            stats["by_cuisine"][cuisine] = stats["by_cuisine"].get(cuisine, 0) + 1
            
            # Budget stats
            budget = restaurant.get("budget", "Unknown")
            stats["by_budget"][budget] = stats["by_budget"].get(budget, 0) + 1
            
            # Rating
            total_rating += restaurant.get("rating", 0)
        
        stats["average_rating"] = round(total_rating / len(self.restaurants_data), 2)
        
        return stats

# Global instance for easy access
restaurant_service = RestaurantIntegrationService()
