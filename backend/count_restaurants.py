#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from api_clients.google_places import GooglePlacesClient

def count_restaurants():
    """Count total restaurants in the mock dataset."""
    client = GooglePlacesClient("fake_key")  # Use fake key to force mock data
    
    # Get all restaurants (no filters)
    result = client._get_mock_restaurant_data()
    total_restaurants = len(result["results"])
    
    print(f"\n=== RESTAURANT COUNT ANALYSIS ===")
    print(f"Total restaurants in mock dataset: {total_restaurants}")
    
    # Count by district
    districts = {}
    for restaurant in result["results"]:
        vicinity = restaurant.get("vicinity", "Unknown")
        district = vicinity.split(",")[0].strip()
        districts[district] = districts.get(district, 0) + 1
    
    print(f"\nRestaurants by district:")
    for district, count in sorted(districts.items()):
        print(f"  {district}: {count}")
    
    # Count by cuisine
    cuisines = {}
    for restaurant in result["results"]:
        cuisine = restaurant.get("cuisine", "Unknown")
        cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
    
    print(f"\nRestaurants by cuisine:")
    for cuisine, count in sorted(cuisines.items()):
        print(f"  {cuisine}: {count}")
    
    # Count by budget
    budgets = {}
    for restaurant in result["results"]:
        budget = restaurant.get("budget", "Unknown")
        budgets[budget] = budgets.get(budget, 0) + 1
    
    print(f"\nRestaurants by budget:")
    for budget, count in sorted(budgets.items()):
        print(f"  {budget}: {count}")
    
    return total_restaurants

if __name__ == "__main__":
    count = count_restaurants()
    if count >= 100:
        print(f"\n✅ SUCCESS: Dataset has {count} restaurants (target: 100+)")
    else:
        print(f"\n❌ NEEDS MORE: Dataset has {count} restaurants (target: 100+)")
        print(f"Need to add {100 - count} more restaurants")
