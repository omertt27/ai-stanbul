#!/usr/bin/env python3
"""
Restaurant Database Service
Handles restaurant queries and provides structured responses
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
import math
from datetime import datetime

@dataclass
class RestaurantQuery:
    """Restaurant query parameters"""
    cuisine_type: Optional[str] = None
    district: Optional[str] = None
    budget: Optional[str] = None  # budget, moderate, upscale, luxury
    rating_min: Optional[float] = None
    location: Optional[Tuple[float, float]] = None  # (lat, lng)
    radius_km: Optional[float] = None
    keywords: List[str] = None
    dietary_restrictions: List[str] = None  # vegetarian, vegan, halal, kosher, gluten_free

class RestaurantDatabaseService:
    """Service for restaurant database operations"""
    
    def __init__(self, database_path: str = None):
        if database_path is None:
            database_path = "/Users/omer/Desktop/ai-stanbul/backend/data/restaurants_database.json"
        
        self.database_path = database_path
        self.restaurants = []
        self.load_database()
        
        # Define response templates
        self.templates = {
            'restaurant_info': self._load_restaurant_templates(),
            'no_results': [
                "I couldn't find restaurants matching your criteria. Let me suggest some popular options in Istanbul instead.",
                "No restaurants found with those filters. Would you like me to broaden the search?",
                "I didn't find matches for that specific request. Here are some highly-rated alternatives:"
            ],
            'multiple_results': [
                "I found several great restaurants for you:",
                "Here are some excellent restaurant options:",
                "Based on your preferences, I recommend these restaurants:"
            ]
        }

    def load_database(self):
        """Load restaurant database from JSON file"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.restaurants = data.get('restaurants', [])
                print(f"✅ Loaded {len(self.restaurants)} restaurants from database")
            else:
                print(f"⚠️  Database not found at {self.database_path}")
                self.restaurants = []
        except Exception as e:
            print(f"❌ Error loading restaurant database: {e}")
            self.restaurants = []

    def _load_restaurant_templates(self) -> Dict[str, str]:
        """Load response templates for different restaurant types"""
        return {
            'single_restaurant': """🍽️ **{name}**
📍 {address}
⭐ Rating: {rating}/5 ({reviews_count} reviews)
💰 Price Level: {budget_category}
🍴 Cuisine: {cuisine_types}
📞 {phone}
🌐 {website}
⏰ {opening_status}

{description}""",

            'restaurant_list': """🍽️ **{name}** ({district})
⭐ {rating}/5 • 💰 {budget_category} • 🍴 {cuisine_types}
📍 {address}
{dietary_info}
{special_note}""",

            'cuisine_intro': {
                'turkish': "Turkish cuisine offers incredible diversity with rich flavors and fresh ingredients.",
                'kebab': "Istanbul's kebab scene ranges from street-side döner to upscale ocakbaşı restaurants.",
                'seafood': "Being surrounded by water, Istanbul offers exceptional fresh seafood.",
                'italian': "Istanbul has a growing Italian restaurant scene with authentic flavors.",
                'cafe': "Istanbul's café culture blends traditional Turkish coffee with modern coffeehouse vibes."
            }
        }

    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def parse_restaurant_query(self, query: str) -> RestaurantQuery:
        """Parse natural language query into structured parameters"""
        query_lower = query.lower()
        
        # Extract cuisine type
        cuisine_type = None
        cuisine_keywords = {
            'turkish': ['turkish', 'türk', 'ottoman', 'traditional'],
            'kebab': ['kebab', 'döner', 'şiş', 'adana', 'urfa'],
            'seafood': ['fish', 'seafood', 'balık', 'deniz'],
            'italian': ['italian', 'pizza', 'pasta', 'italiana'],
            'chinese': ['chinese', 'çin', 'asian'],
            'japanese': ['japanese', 'sushi', 'japon'],
            'cafe': ['cafe', 'coffee', 'kahve', 'kafé'],
            'fast_food': ['fast food', 'burger', 'fast'],
            'bakery': ['bakery', 'pastane', 'bread', 'ekmek']
        }
        
        for cuisine, keywords in cuisine_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                cuisine_type = cuisine
                break

        # Extract district
        district = None
        districts = [
            'sultanahmet', 'beyoğlu', 'galata', 'karaköy', 'beşiktaş', 
            'ortaköy', 'kadıköy', 'üsküdar', 'eminönü', 'fatih',
            'taksim', 'şişli', 'nişantaşı', 'etiler', 'levent'
        ]
        
        for d in districts:
            if d in query_lower:
                district = d.title()
                break

        # Extract budget preference
        budget = None
        budget_keywords = {
            'budget': ['cheap', 'budget', 'ucuz', 'affordable', 'ekonomik'],
            'moderate': ['moderate', 'mid-range', 'orta', 'reasonable'],
            'upscale': ['upscale', 'expensive', 'pahalı', 'high-end', 'fine dining'],
            'luxury': ['luxury', 'luxurious', 'lüks', 'premium', 'exclusive']
        }
        
        for budget_cat, keywords in budget_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                budget = budget_cat
                break

        # Extract rating requirement
        rating_min = None
        rating_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:star|yıldız)', query_lower)
        if rating_matches:
            rating_min = float(rating_matches[0])

        # Extract dietary restrictions
        dietary_restrictions = []
        dietary_patterns = {
            'vegetarian': ['vegetarian', 'vejetaryen', 'veggie', 'no meat', 'etsiz'],
            'vegan': ['vegan', 'plant-based', 'bitki bazlı', 'hayvansal ürün'],
            'halal': ['halal', 'helal', 'islamic', 'muslim'],
            'kosher': ['kosher', 'kasher', 'jewish', 'yahudi'],
            'gluten_free': ['gluten free', 'gluten-free', 'glutensiz', 'celiac', 'çölyak'],
            'dairy_free': ['dairy free', 'dairy-free', 'sütsüz', 'lactose', 'laktoz'],
            'nut_free': ['nut free', 'nut-free', 'fındıksız', 'allergy', 'alerji']
        }
        
        for restriction, patterns in dietary_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                dietary_restrictions.append(restriction)

        # Extract other keywords for additional filtering
        keywords = []
        keyword_patterns = [
            'rooftop', 'view', 'manzara', 'terrace', 'garden', 'bahçe',
            'romantic', 'family', 'business', 'casual', 'formal'
        ]
        
        for keyword in keyword_patterns:
            if keyword in query_lower:
                keywords.append(keyword)

        return RestaurantQuery(
            cuisine_type=cuisine_type,
            district=district,
            budget=budget,
            rating_min=rating_min,
            keywords=keywords,
            dietary_restrictions=dietary_restrictions
        )

    def filter_restaurants(self, query: RestaurantQuery, limit: int = 10) -> List[Dict]:
        """Filter restaurants based on query parameters"""
        filtered = []
        
        for restaurant in self.restaurants:
            # Filter by cuisine type
            if query.cuisine_type:
                if query.cuisine_type not in restaurant.get('cuisine_types', []):
                    continue
            
            # Filter by district
            if query.district:
                if restaurant.get('district', '').lower() != query.district.lower():
                    continue
            
            # Filter by budget
            if query.budget:
                if restaurant.get('budget_category') != query.budget:
                    continue
            
            # Filter by rating
            if query.rating_min:
                rating = restaurant.get('rating')
                if not rating or rating < query.rating_min:
                    continue
            
            # Filter by location/radius
            if query.location and query.radius_km:
                lat, lng = query.location
                rest_lat = restaurant.get('latitude', 0)
                rest_lng = restaurant.get('longitude', 0)
                
                distance = self.calculate_distance(lat, lng, rest_lat, rest_lng)
                if distance > query.radius_km:
                    continue
            
            # Filter by keywords (basic implementation)
            if query.keywords:
                restaurant_text = f"{restaurant.get('name', '')} {' '.join(restaurant.get('categories', []))}"
                if not any(keyword in restaurant_text.lower() for keyword in query.keywords):
                    continue
            
            # Filter by dietary restrictions
            if query.dietary_restrictions:
                if not self._check_dietary_compatibility(restaurant, query.dietary_restrictions):
                    continue
            
            filtered.append(restaurant)
        
        # Sort by rating and review count (handle None values)
        filtered.sort(key=lambda x: (x.get('rating') or 0, x.get('reviews_count') or 0), reverse=True)
        
        return filtered[:limit]

    def get_restaurant_by_name(self, name: str) -> Optional[Dict]:
        """Find restaurant by name (fuzzy matching)"""
        name_lower = name.lower()
        
        # Exact match first
        for restaurant in self.restaurants:
            if restaurant.get('name', '').lower() == name_lower:
                return restaurant
        
        # Partial match
        for restaurant in self.restaurants:
            if name_lower in restaurant.get('name', '').lower():
                return restaurant
        
        return None

    def _check_dietary_compatibility(self, restaurant: Dict, dietary_restrictions: List[str]) -> bool:
        """Check if restaurant meets dietary restriction requirements"""
        # Get restaurant's dietary options (if available in data)
        restaurant_dietary = restaurant.get('dietary_options', [])
        restaurant_name = restaurant.get('name', '').lower()
        restaurant_description = restaurant.get('description', '').lower()
        restaurant_categories = [cat.lower() for cat in restaurant.get('categories', [])]
        
        # Combine all restaurant text for keyword matching
        restaurant_text = f"{restaurant_name} {restaurant_description} {' '.join(restaurant_categories)}"
        
        for restriction in dietary_restrictions:
            # Check if restaurant explicitly supports this dietary restriction
            if restaurant_dietary and restriction in restaurant_dietary:
                continue  # This restriction is supported
            
            # Fallback: keyword-based matching for common restrictions
            restriction_keywords = {
                'vegetarian': ['vegetarian', 'vejetaryen', 'veggie', 'plant', 'veg'],
                'vegan': ['vegan', 'plant-based', 'bitki'],
                'halal': ['halal', 'helal', 'islamic', 'muslim', 'türk', 'turkish'],
                'kosher': ['kosher', 'kasher', 'jewish'],
                'gluten_free': ['gluten free', 'glutensiz', 'celiac'],
                'dairy_free': ['dairy free', 'sütsüz', 'lactose free'],
                'nut_free': ['nut free', 'fındıksız']
            }
            
            keywords = restriction_keywords.get(restriction, [])
            
            # Special logic for common restrictions
            if restriction == 'vegetarian':
                # Turkish and Mediterranean restaurants often have good vegetarian options
                if any(keyword in restaurant_text for keyword in ['türk', 'turkish', 'mediterranean', 'meze', 'sebze']):
                    continue
                # Also check if explicitly mentioned
                if any(keyword in restaurant_text for keyword in keywords):
                    continue 
                # Otherwise, this restaurant might not be suitable
                return False
                
            elif restriction == 'halal':
                # Most Turkish restaurants are halal by default
                if any(keyword in restaurant_text for keyword in ['türk', 'turkish', 'ottoman', 'kebab', 'döner']):
                    continue
                # Check explicit halal mention
                if any(keyword in restaurant_text for keyword in keywords):
                    continue
                # International restaurants need explicit halal certification
                if any(keyword in restaurant_text for keyword in ['italian', 'chinese', 'japanese', 'american']):
                    return False
                    
            elif restriction == 'vegan':
                # Stricter check for vegan options
                if any(keyword in restaurant_text for keyword in keywords):
                    continue
                # Some cuisines are more likely to have vegan options
                if any(keyword in restaurant_text for keyword in ['mediterranean', 'turkish', 'sebze']):
                    continue
                return False
                
            else:
                # For other restrictions, check keyword matching
                if not any(keyword in restaurant_text for keyword in keywords):
                    return False
        
        return True  # All restrictions can be accommodated

    def format_single_restaurant_response(self, restaurant: Dict) -> str:
        """Format detailed response for a single restaurant"""
        # Format phone
        phone = restaurant.get('phone', 'Not available')
        if phone and phone != 'Not available':
            phone = f"📞 {phone}"
        else:
            phone = ""
        
        # Format website
        website = restaurant.get('website', '')
        if website:
            website = f"🌐 [Website]({website})"
        else:
            website = ""
        
        # Format opening status
        opening_hours = restaurant.get('opening_hours', {})
        if opening_hours and 'open_now' in opening_hours:
            opening_status = "🟢 Open now" if opening_hours['open_now'] else "🔴 Closed now"
        else:
            opening_status = "⏰ Hours not available"
        
        # Format cuisine types
        cuisine_types = ', '.join(restaurant.get('cuisine_types', ['Restaurant']))
        
        # Generate description based on cuisine and features
        description = self.generate_restaurant_description(restaurant)
        
        return self.templates['restaurant_info']['single_restaurant'].format(
            name=restaurant.get('name', 'Unknown Restaurant'),
            address=restaurant.get('address', 'Address not available'),
            rating=restaurant.get('rating', 'N/A'),
            reviews_count=restaurant.get('reviews_count', 0),
            budget_category=restaurant.get('budget_category', 'Moderate').title(),
            cuisine_types=cuisine_types,
            phone=phone,
            website=website,
            opening_status=opening_status,
            description=description
        )

    def format_restaurant_list_response(self, restaurants: List[Dict], query: RestaurantQuery) -> str:
        """Format response for multiple restaurants"""
        if not restaurants:
            return self.templates['no_results'][0]
        
        # Add intro based on cuisine type
        intro = ""
        if query.cuisine_type and query.cuisine_type in self.templates['restaurant_info']['cuisine_intro']:
            intro = self.templates['restaurant_info']['cuisine_intro'][query.cuisine_type] + "\n\n"
        
        response = intro + self.templates['multiple_results'][0] + "\n\n"
        
        for i, restaurant in enumerate(restaurants, 1):
            cuisine_types = ', '.join(restaurant.get('cuisine_types', ['Restaurant']))
            
            # Add dietary information
            dietary_info = ""
            dietary_options = restaurant.get('dietary_options', [])
            if dietary_options:
                dietary_display = [opt.replace('_', '-') for opt in dietary_options]
                dietary_info = f"🥗 {', '.join(dietary_display)} options"
            
            # Add special notes
            special_note = ""
            rating = restaurant.get('rating', 0)
            if rating >= 4.5:
                special_note = "⭐ Highly rated!"
            elif restaurant.get('budget_category') == 'luxury':
                special_note = "✨ Premium experience"
            elif restaurant.get('budget_category') == 'budget':
                special_note = "💰 Great value"
            
            restaurant_text = self.templates['restaurant_info']['restaurant_list'].format(
                name=restaurant.get('name', 'Unknown Restaurant'),
                district=restaurant.get('district', 'Istanbul'),
                rating=restaurant.get('rating', 'N/A'),
                budget_category=restaurant.get('budget_category', 'Moderate').title(),
                cuisine_types=cuisine_types,
                address=restaurant.get('address', 'Address not available'),
                dietary_info=dietary_info,
                special_note=special_note
            )
            
            response += f"{i}. {restaurant_text}\n\n"
        
        return response.strip()

    def generate_restaurant_description(self, restaurant: Dict) -> str:
        """Generate contextual description for restaurant"""
        descriptions = []
        
        # Rating-based description
        rating = restaurant.get('rating', 0)
        if rating >= 4.5:
            descriptions.append("This highly-rated restaurant is a local favorite.")
        elif rating >= 4.0:
            descriptions.append("A well-regarded restaurant with consistently good reviews.")
        
        # Location-based description
        district = restaurant.get('district', '')
        district_descriptions = {
            'Sultanahmet': "Located in Istanbul's historic heart, perfect for sightseeing breaks.",
            'Beyoğlu': "In the vibrant Beyoğlu district, ideal for evening dining.",
            'Galata': "Near the iconic Galata Tower with a cosmopolitan atmosphere.",
            'Karaköy': "In trendy Karaköy, known for its artistic and culinary scene.",
            'Ortaköy': "By the Bosphorus with beautiful water views."
        }
        
        if district in district_descriptions:
            descriptions.append(district_descriptions[district])
        
        # Budget-based description
        budget = restaurant.get('budget_category', '')
        if budget == 'luxury':
            descriptions.append("An upscale dining experience with premium service.")
        elif budget == 'budget':
            descriptions.append("Great value for money with authentic flavors.")
        
        # Dietary options description
        dietary_options = restaurant.get('dietary_options', [])
        if dietary_options:
            dietary_text = ", ".join(dietary_options).replace('_', '-')
            descriptions.append(f"Offers {dietary_text} options.")
        else:
            # Infer dietary options from cuisine type and name
            restaurant_text = f"{restaurant.get('name', '')} {' '.join(restaurant.get('categories', []))}".lower()
            inferred_options = []
            
            if any(keyword in restaurant_text for keyword in ['türk', 'turkish', 'kebab', 'döner']):
                inferred_options.append("halal")
            if any(keyword in restaurant_text for keyword in ['vegetarian', 'vegan', 'sebze']):
                inferred_options.append("vegetarian")
            if any(keyword in restaurant_text for keyword in ['meze', 'mediterranean']):
                inferred_options.append("vegetarian-friendly")
                
            if inferred_options:
                options_text = ", ".join(inferred_options).replace('_', '-')
                descriptions.append(f"Likely offers {options_text} options.")
        
        return " ".join(descriptions) if descriptions else "A popular dining spot in Istanbul."

    def search_restaurants(self, query: str) -> str:
        """Main method to search restaurants and return formatted response"""
        try:
            # Parse the query
            parsed_query = self.parse_restaurant_query(query)
            
            # Check if it's a specific restaurant name query
            if not any([parsed_query.cuisine_type, parsed_query.district, parsed_query.budget]):
                # Might be looking for a specific restaurant
                restaurant = self.get_restaurant_by_name(query)
                if restaurant:
                    return self.format_single_restaurant_response(restaurant)
            
            # Filter restaurants
            restaurants = self.filter_restaurants(parsed_query)
            
            # Return formatted response
            if len(restaurants) == 1:
                return self.format_single_restaurant_response(restaurants[0])
            else:
                return self.format_restaurant_list_response(restaurants, parsed_query)
                
        except Exception as e:
            return f"I apologize, but I encountered an error while searching for restaurants: {str(e)}. Please try rephrasing your request."

    def get_popular_restaurants(self, district: str = None, limit: int = 5) -> str:
        """Get popular restaurants, optionally filtered by district"""
        query = RestaurantQuery(district=district, rating_min=4.0)
        restaurants = self.filter_restaurants(query, limit=limit)
        
        if restaurants:
            return self.format_restaurant_list_response(restaurants, query)
        else:
            return "I couldn't find highly-rated restaurants for that area. Let me suggest some popular options instead."

# Example usage and testing
if __name__ == "__main__":
    # Initialize service
    service = RestaurantDatabaseService()
    
    # Test queries
    test_queries = [
        "best Turkish restaurants in Sultanahmet",
        "cheap kebab places",
        "seafood restaurant with good view",
        "Italian restaurant in Beyoğlu",
        "romantic dinner place upscale"
    ]
    
    print("🍽️  Restaurant Database Service Test\n" + "="*50)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        response = service.search_restaurants(query)
        print(response)
        print()
