#!/usr/bin/env python3
"""
Google Places API fetcher for Istanbul restaurants
Fetches restaurant data and structures it for our database
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Restaurant:
    """Restaurant data structure"""
    place_id: str
    name: str
    address: str
    phone: Optional[str]
    website: Optional[str]
    rating: Optional[float]
    price_level: Optional[int]  # 1-4 scale
    cuisine_types: List[str]
    district: str
    latitude: float
    longitude: float
    opening_hours: Optional[Dict]
    photos: List[str]
    reviews_count: Optional[int]
    google_maps_url: str
    categories: List[str]
    budget_category: str  # "budget", "moderate", "upscale", "luxury"

class GooglePlacesFetcher:
    """Fetch restaurant data from Google Places API"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_PLACES_API_KEY or GOOGLE_MAPS_API_KEY environment variable not set")
        
        self.base_url = "https://maps.googleapis.com/maps/api/place"
        self.restaurants = []
        
        # Istanbul districts for targeted search
        self.istanbul_districts = [
            {"name": "Sultanahmet", "lat": 41.0058, "lng": 28.9784},
            {"name": "Beyoğlu", "lat": 41.0362, "lng": 28.9744},
            {"name": "Galata", "lat": 41.0255, "lng": 28.9732},
            {"name": "Karaköy", "lat": 41.0256, "lng": 28.9739},
            {"name": "Beşiktaş", "lat": 41.0422, "lng": 29.0094},
            {"name": "Ortaköy", "lat": 41.0555, "lng": 29.0267},
            {"name": "Kadıköy", "lat": 40.9923, "lng": 29.0243},
            {"name": "Üsküdar", "lat": 41.0214, "lng": 29.0164},
            {"name": "Eminönü", "lat": 41.0167, "lng": 28.9709},
            {"name": "Fatih", "lat": 41.0214, "lng": 28.9684},
            {"name": "Taksim", "lat": 41.0367, "lng": 28.9850},
            {"name": "Şişli", "lat": 41.0602, "lng": 28.9876},
            {"name": "Nişantaşı", "lat": 41.0468, "lng": 28.9903},
            {"name": "Etiler", "lat": 41.0778, "lng": 29.0264},
            {"name": "Levent", "lat": 41.0823, "lng": 29.0121}
        ]

    def search_restaurants_in_district(self, district: Dict, radius: int = 2000) -> List[Dict]:
        """Search for restaurants in a specific district"""
        url = f"{self.base_url}/nearbysearch/json"
        
        params = {
            'location': f"{district['lat']},{district['lng']}",
            'radius': radius,
            'type': 'restaurant',
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK':
                logger.info(f"Found {len(data['results'])} restaurants in {district['name']}")
                return data['results']
            else:
                logger.error(f"API error in {district['name']}: {data['status']}")
                return []
                
        except requests.RequestException as e:
            logger.error(f"Request failed for {district['name']}: {e}")
            return []

    def get_place_details(self, place_id: str) -> Optional[Dict]:
        """Get detailed information for a specific place"""
        url = f"{self.base_url}/details/json"
        
        params = {
            'place_id': place_id,
            'fields': 'name,formatted_address,formatted_phone_number,website,rating,price_level,opening_hours,photos,reviews,geometry,types,url',
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK':
                return data['result']
            else:
                logger.error(f"Details API error for {place_id}: {data['status']}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Details request failed for {place_id}: {e}")
            return None

    def classify_cuisine_type(self, types: List[str], name: str) -> List[str]:
        """Classify restaurant cuisine based on Google types and name"""
        cuisine_mapping = {
            'turkish': ['meal_takeaway', 'restaurant'],
            'kebab': ['meal_takeaway', 'restaurant'],
            'seafood': ['meal_takeaway', 'restaurant'],
            'mediterranean': ['meal_takeaway', 'restaurant'],
            'italian': ['meal_takeaway', 'restaurant'],
            'chinese': ['meal_takeaway', 'restaurant'],
            'japanese': ['meal_takeaway', 'restaurant'],
            'indian': ['meal_takeaway', 'restaurant'],
            'french': ['meal_takeaway', 'restaurant'],
            'american': ['meal_takeaway', 'restaurant'],
            'fast_food': ['meal_takeaway'],
            'cafe': ['cafe'],
            'bakery': ['bakery'],
            'bar': ['bar', 'night_club']
        }
        
        cuisines = []
        name_lower = name.lower()
        
        # Check name for cuisine indicators
        if any(word in name_lower for word in ['kebab', 'döner', 'şiş']):
            cuisines.append('kebab')
        elif any(word in name_lower for word in ['balık', 'fish', 'seafood']):
            cuisines.append('seafood')
        elif any(word in name_lower for word in ['pizza', 'italian']):
            cuisines.append('italian')
        elif any(word in name_lower for word in ['chinese', 'çin']):
            cuisines.append('chinese')
        elif any(word in name_lower for word in ['sushi', 'japanese']):
            cuisines.append('japanese')
        elif any(word in name_lower for word in ['cafe', 'kahve']):
            cuisines.append('cafe')
        elif any(word in name_lower for word in ['pastane', 'bakery']):
            cuisines.append('bakery')
        else:
            cuisines.append('turkish')  # Default for Istanbul
            
        return cuisines

    def determine_budget_category(self, price_level: Optional[int], rating: Optional[float]) -> str:
        """Determine budget category based on price level and rating"""
        if price_level is None:
            return "moderate"
        
        if price_level == 1:
            return "budget"
        elif price_level == 2:
            return "moderate"
        elif price_level == 3:
            return "upscale"
        else:  # price_level == 4
            return "luxury"

    def process_restaurant_data(self, place_data: Dict, district_name: str) -> Optional[Restaurant]:
        """Process raw Google Places data into Restaurant object"""
        try:
            # Get additional details
            details = self.get_place_details(place_data['place_id'])
            if not details:
                details = place_data  # Fallback to basic data
            
            # Extract basic info
            name = details.get('name', 'Unknown Restaurant')
            address = details.get('formatted_address', details.get('vicinity', 'Address not available'))
            
            # Extract location
            geometry = details.get('geometry', place_data.get('geometry', {}))
            location = geometry.get('location', {})
            
            # Extract cuisine types
            types = details.get('types', place_data.get('types', []))
            cuisine_types = self.classify_cuisine_type(types, name)
            
            # Extract photos
            photos = []
            if 'photos' in details and details['photos']:
                for photo in details['photos'][:3]:  # Limit to 3 photos
                    photo_ref = photo.get('photo_reference')
                    if photo_ref:
                        photo_url = f"{self.base_url}/photo?maxwidth=400&photo_reference={photo_ref}&key={self.api_key}"
                        photos.append(photo_url)
            
            # Extract opening hours
            opening_hours = None
            if 'opening_hours' in details:
                opening_hours = {
                    'open_now': details['opening_hours'].get('open_now'),
                    'weekday_text': details['opening_hours'].get('weekday_text', [])
                }
            
            restaurant = Restaurant(
                place_id=place_data['place_id'],
                name=name,
                address=address,
                phone=details.get('formatted_phone_number'),
                website=details.get('website'),
                rating=details.get('rating', place_data.get('rating')),
                price_level=details.get('price_level', place_data.get('price_level')),
                cuisine_types=cuisine_types,
                district=district_name,
                latitude=location.get('lat', 0),
                longitude=location.get('lng', 0),
                opening_hours=opening_hours,
                photos=photos,
                reviews_count=details.get('user_ratings_total', place_data.get('user_ratings_total')),
                google_maps_url=details.get('url', f"https://maps.google.com/?cid={place_data['place_id']}"),
                categories=types,
                budget_category=self.determine_budget_category(
                    details.get('price_level', place_data.get('price_level')),
                    details.get('rating', place_data.get('rating'))
                )
            )
            
            return restaurant
            
        except Exception as e:
            logger.error(f"Error processing restaurant data: {e}")
            return None

    def fetch_all_restaurants(self, max_per_district: int = 60) -> List[Restaurant]:
        """Fetch restaurants from all Istanbul districts"""
        all_restaurants = []
        
        for district in self.istanbul_districts:
            logger.info(f"Fetching restaurants in {district['name']}...")
            
            # Search restaurants in district
            restaurants_data = self.search_restaurants_in_district(district)
            
            # Process each restaurant (limit to avoid API quota issues)
            processed_count = 0
            for restaurant_data in restaurants_data[:max_per_district]:
                if processed_count >= max_per_district:
                    break
                    
                restaurant = self.process_restaurant_data(restaurant_data, district['name'])
                if restaurant:
                    all_restaurants.append(restaurant)
                    processed_count += 1
                
                # Rate limiting - avoid hitting API limits
                time.sleep(0.1)  # 100ms delay between requests
            
            logger.info(f"Processed {processed_count} restaurants in {district['name']}")
            
            # Longer delay between districts
            time.sleep(1)
        
        logger.info(f"Total restaurants fetched: {len(all_restaurants)}")
        return all_restaurants

    def save_to_database(self, restaurants: List[Restaurant], filename: str = "restaurants_database.json"):
        """Save restaurants to JSON database file"""
        # Convert to dict format
        restaurants_dict = {
            "metadata": {
                "total_restaurants": len(restaurants),
                "districts_covered": len(self.istanbul_districts),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Google Places API"
            },
            "restaurants": [asdict(restaurant) for restaurant in restaurants]
        }
        
        # Create data directory if it doesn't exist
        data_dir = "/Users/omer/Desktop/ai-stanbul/backend/data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to file
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(restaurants_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(restaurants)} restaurants to {filepath}")
        return filepath

def main():
    """Main function to fetch and save restaurant data"""
    try:
        fetcher = GooglePlacesFetcher()
        
        # Fetch restaurants
        logger.info("Starting restaurant data fetch...")
        restaurants = fetcher.fetch_all_restaurants(max_per_district=60)
        
        if restaurants:
            # Save to database
            filepath = fetcher.save_to_database(restaurants)
            
            # Print summary
            print(f"\n✅ Successfully fetched and saved {len(restaurants)} restaurants!")
            print(f"📁 Database saved to: {filepath}")
            
            # Print some statistics
            districts = {}
            cuisines = {}
            budget_categories = {}
            
            for restaurant in restaurants:
                # Count by district
                districts[restaurant.district] = districts.get(restaurant.district, 0) + 1
                
                # Count by cuisine
                for cuisine in restaurant.cuisine_types:
                    cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
                
                # Count by budget
                budget_categories[restaurant.budget_category] = budget_categories.get(restaurant.budget_category, 0) + 1
            
            print(f"\n📊 Statistics:")
            print(f"Districts: {len(districts)} covered")
            print(f"Top cuisines: {sorted(cuisines.items(), key=lambda x: x[1], reverse=True)[:5]}")
            print(f"Budget distribution: {budget_categories}")
            
        else:
            logger.error("No restaurants were fetched. Check API key and connection.")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
