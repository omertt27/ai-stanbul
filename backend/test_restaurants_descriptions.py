#!/usr/bin/env python3
"""
Test script to demonstrate Google Places API restaurant descriptions functionality.
Run this to see examples of restaurant data with descriptions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_clients.google_places import GooglePlacesClient, get_istanbul_restaurants_with_descriptions
import json
from typing import Dict, List

def print_restaurant_info(restaurant: Dict):
    """Pretty print restaurant information."""
    print(f"\n{'='*60}")
    print(f"🍽️  {restaurant['name']}")
    print(f"{'='*60}")
    print(f"📍 Address: {restaurant['address']}")
    print(f"⭐ Rating: {restaurant['rating']}/5 ({restaurant['user_ratings_total']} reviews)")
    print(f"💰 Price Level: {restaurant.get('price_level', 'Not specified')}")
    print(f"🍴 Cuisine: {restaurant['cuisine_types']}")
    
    if restaurant.get('phone'):
        print(f"📞 Phone: {restaurant['phone']}")
    
    if restaurant.get('website'):
        print(f"🌐 Website: {restaurant['website']}")
    
    print(f"\n📝 Description:")
    print(f"{restaurant['description']}")
    
    if restaurant.get('opening_hours', {}).get('weekday_text'):
        print(f"\n🕒 Hours:")
        for hours in restaurant['opening_hours']['weekday_text'][:3]:  # Show first 3 days
            print(f"   {hours}")
    
    if restaurant.get('reviews_summary', {}).get('recent_review_snippet'):
        print(f"\n💬 Recent Review:")
        print(f"   \"{restaurant['reviews_summary']['recent_review_snippet']}\"")
    
    if restaurant.get('photos'):
        print(f"\n📸 Photos: {len(restaurant['photos'])} available")

def test_basic_search():
    """Test basic restaurant search functionality."""
    print("🔍 Testing Basic Restaurant Search...")
    print("Searching for restaurants in Istanbul...")
    
    try:
        restaurants = get_istanbul_restaurants_with_descriptions(limit=3)
        
        if restaurants:
            print(f"\n✅ Found {len(restaurants)} restaurants with descriptions!")
            for restaurant in restaurants:
                print_restaurant_info(restaurant)
        else:
            print("❌ No restaurants found.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_district_search():
    """Test district-specific search."""
    print("\n" + "="*80)
    print("🏛️  Testing District-Specific Search...")
    print("Searching for restaurants in Beyoğlu...")
    
    try:
        restaurants = get_istanbul_restaurants_with_descriptions(
            district="Beyoğlu", 
            limit=2
        )
        
        if restaurants:
            print(f"\n✅ Found {len(restaurants)} restaurants in Beyoğlu!")
            for restaurant in restaurants:
                print_restaurant_info(restaurant)
        else:
            print("❌ No restaurants found in Beyoğlu.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_advanced_search():
    """Test advanced search with filters."""
    print("\n" + "="*80)
    print("🔍 Testing Advanced Search...")
    print("Searching for cafes in Kadıköy...")
    
    try:
        client = GooglePlacesClient()
        restaurants = client.get_restaurants_with_descriptions(
            location="Kadıköy, Istanbul, Turkey",
            keyword="cafe",
            limit=2,
            radius=1000
        )
        
        if restaurants:
            print(f"\n✅ Found {len(restaurants)} cafes in Kadıköy!")
            for restaurant in restaurants:
                print_restaurant_info(restaurant)
        else:
            print("❌ No cafes found in Kadıköy.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_place_details():
    """Test getting detailed place information."""
    print("\n" + "="*80)
    print("🔍 Testing Place Details Lookup...")
    
    try:
        # First get a place ID from search
        client = GooglePlacesClient()
        search_results = client.search_restaurants(
            lat_lng="41.0082,28.9784",  # Istanbul center
            radius=1000
        )
        
        if search_results.get("results"):
            place_id = search_results["results"][0]["place_id"]
            place_name = search_results["results"][0]["name"]
            
            print(f"Getting details for: {place_name}")
            
            details = client.get_place_details(place_id)
            
            if details.get("status") == "OK":
                result = details["result"]
                print(f"\n✅ Successfully retrieved details!")
                print(f"Name: {result.get('name')}")
                print(f"Rating: {result.get('rating')}/5")
                print(f"Address: {result.get('formatted_address')}")
                print(f"Types: {', '.join(result.get('types', [])[:5])}")
                
                if result.get('editorial_summary'):
                    print(f"Summary: {result['editorial_summary'].get('overview', 'N/A')}")
            else:
                print(f"❌ Failed to get details: {details.get('status')}")
        else:
            print("❌ No places found to get details for.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all tests."""
    print("🚀 Google Places API Restaurant Descriptions Test")
    print("=" * 80)
    
    # Test basic functionality
    test_basic_search()
    
    # Test district search
    test_district_search()
    
    # Test advanced search
    test_advanced_search()
    
    # Test place details
    test_place_details()
    
    print("\n" + "="*80)
    print("✨ Testing Complete!")
    print("\nTo use in your application:")
    print("1. Start the FastAPI server: uvicorn main:app --reload")
    print("2. Visit: http://localhost:8000/restaurants/search")
    print("3. Try: http://localhost:8000/restaurants/istanbul/Beyoğlu")
    print("4. API docs: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
