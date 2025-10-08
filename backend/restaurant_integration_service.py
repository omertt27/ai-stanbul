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
            'budget': 'Budget-friendly',
            'mid-range': 'Mid-range',
            'premium': 'Premium dining',
            'luxury': 'Luxury dining'
        }
        
        return f"""I found an excellent restaurant for you: **{restaurant.name}** in {restaurant.district}.

{restaurant.description}

📍 **Location:** {restaurant.address}
⭐ **Rating:** {restaurant.rating}/5
{budget_emoji.get(restaurant.budget, '💰')} **Price:** {budget_text.get(restaurant.budget, 'Mid-range')}
🍴 **Cuisine:** {restaurant.cuisine}

This restaurant offers a great dining experience with authentic flavors and quality service."""
    
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
        
        response = f"Here are {len(restaurants)} restaurants in the area:\n\n"
        
        for i, restaurant in enumerate(restaurants, 1):
            budget_emoji = {
                'budget': '💰',
                'mid-range': '💰💰', 
                'premium': '💰💰💰',
                'luxury': '💰💰💰💰'
            }
            
            response += f"**{i}. {restaurant.name}** ({restaurant.district})\n"
            response += f"⭐ {restaurant.rating}/5 • {budget_emoji.get(restaurant.budget, '💰')} {restaurant.budget.title()} • 🍴 {restaurant.cuisine}\n"
            response += f"{restaurant.description}\n\n"
        
        return response.strip()
    
    def _get_no_results_response(self) -> str:
        """Get response when no restaurants match criteria"""
        fallback_restaurants = self.search_restaurants(limit=3)  # Get top 3 popular restaurants
        
        if fallback_restaurants:
            response = "I couldn't find restaurants matching your specific criteria, but here are some popular options in Istanbul:\n\n"
            response += self._format_multiple_restaurants(fallback_restaurants)
            return response
        
        return """I'm sorry, I couldn't find specific restaurant recommendations at the moment. However, Istanbul has amazing dining options including:

• Traditional Turkish cuisine (kebabs, meze, Turkish breakfast)
• Fresh seafood along the Bosphorus
• International cuisines in Beyoğlu and Nişantaşı
• Street food in Kadıköy and Eminönü

Would you like me to help you with a specific area or cuisine type?"""
    
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
