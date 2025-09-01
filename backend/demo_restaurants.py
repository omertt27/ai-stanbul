#!/usr/bin/env python3
"""
Demo script to show restaurant descriptions functionality using coordinates.
This version uses direct coordinates to avoid geocoding API calls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from api_clients.google_places import GooglePlacesClient
import json

def demo_with_coordinates():
    """Demo using direct coordinates (Istanbul center)."""
    print("🍽️ Restaurant Descriptions Demo")
    print("=" * 60)
    print("📍 Using Istanbul coordinates: 41.0082, 28.9784")
    
    try:
        client = GooglePlacesClient()
        
        # Use Istanbul center coordinates
        restaurants = client.get_restaurants_with_descriptions(
            lat_lng="41.0082,28.9784",
            limit=3,
            radius=2000
        )
        
        if restaurants:
            print(f"\n✅ Found {len(restaurants)} restaurants!")
            
            for i, restaurant in enumerate(restaurants, 1):
                print(f"\n{'='*50}")
                print(f"🏪 Restaurant #{i}: {restaurant['name']}")
                print(f"{'='*50}")
                print(f"📍 Address: {restaurant.get('address', 'N/A')}")
                print(f"⭐ Rating: {restaurant.get('rating', 'N/A')}/5")
                print(f"👥 Reviews: {restaurant.get('user_ratings_total', 'N/A')}")
                print(f"🍴 Cuisine: {restaurant.get('cuisine_types', 'N/A')}")
                print(f"📝 Description: {restaurant.get('description', 'No description available')[:200]}...")
                
                if restaurant.get('phone'):
                    print(f"📞 Phone: {restaurant['phone']}")
                if restaurant.get('website'):
                    print(f"🌐 Website: {restaurant['website']}")
                    
        else:
            print("❌ No restaurants found.")
            print("💡 This might be due to:")
            print("   - Invalid or missing Google Places API key")
            print("   - API quota exceeded")
            print("   - Network connectivity issues")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 To fix this:")
        print("1. Get a Google Places API key from Google Cloud Console")
        print("2. Enable Places API, Geocoding API, and Place Photos API")
        print("3. Set the API key in api_clients/google_places.py")
        print("4. Make sure billing is enabled for the project")

def demo_api_structure():
    """Show the expected API response structure."""
    print("\n" + "="*60)
    print("📋 Expected Restaurant Data Structure")
    print("="*60)
    
    sample_restaurant = {
        "place_id": "ChIJ...",
        "name": "Amazing Istanbul Restaurant",
        "address": "Istiklal Caddesi, Beyoğlu/İstanbul",
        "phone": "+90 212 xxx xxxx",
        "website": "https://restaurant-website.com",
        "rating": 4.5,
        "user_ratings_total": 1250,
        "price_level": 2,
        "description": "This popular restaurant serves traditional Turkish cuisine with a modern twist. Known for their excellent kebabs and mezze platters. Recent review mentions great atmosphere and friendly service.",
        "cuisine_types": "Restaurant, Turkish Cuisine",
        "opening_hours": {
            "open_now": True,
            "weekday_text": [
                "Monday: 11:00 AM – 11:00 PM",
                "Tuesday: 11:00 AM – 11:00 PM",
                "Wednesday: 11:00 AM – 11:00 PM"
            ]
        },
        "location": {
            "lat": 41.0082,
            "lng": 28.9784
        },
        "photos": [
            "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference=...",
            "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference=..."
        ],
        "reviews_summary": {
            "total_reviews_shown": 5,
            "average_rating": 4.3,
            "recent_review_snippet": "Great food and amazing atmosphere! The staff was very friendly..."
        }
    }
    
    print(json.dumps(sample_restaurant, indent=2, ensure_ascii=False))

def demo_api_endpoints():
    """Show available API endpoints."""
    print("\n" + "="*60)
    print("🌐 Available API Endpoints")
    print("="*60)
    
    endpoints = [
        {
            "method": "GET",
            "endpoint": "/restaurants/search",
            "description": "Search restaurants with descriptions",
            "example": "http://localhost:8000/restaurants/search?district=Beyoğlu&limit=10"
        },
        {
            "method": "GET", 
            "endpoint": "/restaurants/istanbul/{district}",
            "description": "Get restaurants from specific Istanbul district",
            "example": "http://localhost:8000/restaurants/istanbul/Sultanahmet"
        },
        {
            "method": "GET",
            "endpoint": "/restaurants/popular",
            "description": "Get highly-rated restaurants",
            "example": "http://localhost:8000/restaurants/popular?min_rating=4.0&limit=15"
        },
        {
            "method": "GET",
            "endpoint": "/restaurants/details/{place_id}",
            "description": "Get detailed info for a specific restaurant",
            "example": "http://localhost:8000/restaurants/details/ChIJ..."
        },
        {
            "method": "POST",
            "endpoint": "/restaurants/save",
            "description": "Save a restaurant to the local database",
            "example": "POST http://localhost:8000/restaurants/save?place_id=ChIJ..."
        }
    ]
    
    for endpoint in endpoints:
        print(f"\n🔗 {endpoint['method']} {endpoint['endpoint']}")
        print(f"   📝 {endpoint['description']}")
        print(f"   🌐 {endpoint['example']}")

def main():
    print("🚀 Restaurant Descriptions Feature Demo")
    print("This demo shows the restaurant descriptions functionality.")
    print("For full functionality, you'll need a valid Google Places API key.\n")
    
    # Try the coordinate-based demo
    demo_with_coordinates()
    
    # Show expected data structure
    demo_api_structure()
    
    # Show available endpoints
    demo_api_endpoints()
    
    print("\n" + "="*60)
    print("🚀 Next Steps:")
    print("1. Add your Google Places API key")
    print("2. Start the FastAPI server: uvicorn main:app --reload")
    print("3. Open: http://localhost:8000/docs")
    print("4. Try the endpoints with real data!")
    print("\n💡 Check RESTAURANT_DESCRIPTIONS_README.md for full setup instructions.")

if __name__ == "__main__":
    main()
