#!/usr/bin/env python3

import sys
import os
import requests
import json
sys.path.append(os.path.join(os.path.dirname(__file__)))

from api_clients.google_places import GooglePlacesClient

def comprehensive_validation():
    """Comprehensive validation of the expanded restaurant dataset."""
    
    print("🍽️  ISTANBUL RESTAURANT DATASET FINAL VALIDATION")
    print("=" * 60)
    
    # Test 1: Count total restaurants
    client = GooglePlacesClient("fake_key")
    result = client._get_mock_restaurant_data()
    total_restaurants = len(result["results"])
    
    print(f"📊 DATASET STATISTICS:")
    print(f"   • Total Restaurants: {total_restaurants}")
    
    # Analyze by district
    districts = {}
    cuisines = {}
    budgets = {}
    
    for restaurant in result["results"]:
        # District analysis
        vicinity = restaurant.get("vicinity", "Unknown")
        district = vicinity.split(",")[0].strip()
        districts[district] = districts.get(district, 0) + 1
        
        # Cuisine analysis
        cuisine = restaurant.get("cuisine", "Unknown")
        cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
        
        # Budget analysis
        budget = restaurant.get("budget", "Unknown")
        budgets[budget] = budgets.get(budget, 0) + 1
    
    print(f"   • Districts Covered: {len(districts)}")
    print(f"   • Cuisine Types: {len(cuisines)}")
    print(f"   • Budget Levels: {len(budgets)}")
    
    # Test 2: API Integration
    try:
        response = requests.get("http://localhost:8001/api/restaurants/all", timeout=5)
        api_working = response.status_code == 200
        if api_working:
            api_data = response.json()
            api_total = api_data.get("total_restaurants", 0)
            print(f"   • API Integration: ✅ Working ({api_total} restaurants via API)")
        else:
            print(f"   • API Integration: ❌ Error {response.status_code}")
    except:
        print(f"   • API Integration: ❌ Server not accessible")
        api_working = False
    
    # Test 3: Target Achievement
    target_met = total_restaurants >= 100
    print(f"   • Target (100+ restaurants): {'✅ ACHIEVED' if target_met else '❌ NOT MET'}")
    
    print(f"\n📍 DISTRICT DISTRIBUTION:")
    for district, count in sorted(districts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {district}: {count} restaurants")
    
    print(f"\n🍴 CUISINE DIVERSITY:")
    for cuisine, count in sorted(cuisines.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {cuisine}: {count} restaurants")
    
    print(f"\n💰 BUDGET LEVELS:")
    for budget, count in sorted(budgets.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {budget}: {count} restaurants")
    
    # Test 4: Filtering functionality
    print(f"\n🔍 FILTERING TESTS:")
    
    # Test district filtering
    beyoglu_result = client._get_mock_restaurant_data(location="Beyoğlu")
    beyoglu_count = len(beyoglu_result["results"])
    print(f"   • Beyoğlu filter: {beyoglu_count} restaurants")
    
    # Test cuisine filtering
    seafood_result = client._get_mock_restaurant_data(keyword="seafood")
    seafood_count = len(seafood_result["results"])
    print(f"   • Seafood filter: {seafood_count} restaurants")
    
    # Test budget filtering
    luxury_result = client._get_mock_restaurant_data(keyword="luxury")
    luxury_count = len(luxury_result["results"])
    print(f"   • Luxury filter: {luxury_count} restaurants")
    
    # Test API filtering if available
    if api_working:
        print(f"\n🌐 API FILTERING TESTS:")
        try:
            # Test location filtering
            response = requests.get("http://localhost:8001/api/restaurants/search?location=Kadikoy&limit=50", timeout=5)
            if response.status_code == 200:
                api_kadikoy = len(response.json().get("restaurants", []))
                print(f"   • Kadıköy (API): {api_kadikoy} restaurants")
            
            # Test keyword filtering
            response = requests.get("http://localhost:8001/api/restaurants/search?keyword=turkish&limit=50", timeout=5)
            if response.status_code == 200:
                api_turkish = len(response.json().get("restaurants", []))
                print(f"   • Turkish cuisine (API): {api_turkish} restaurants")
                
        except Exception as e:
            print(f"   • API filtering tests failed: {e}")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    
    success_criteria = [
        (total_restaurants >= 100, f"Dataset size: {total_restaurants}/100+ restaurants"),
        (len(districts) >= 8, f"District coverage: {len(districts)}/8+ districts"),
        (len(cuisines) >= 10, f"Cuisine diversity: {len(cuisines)}/10+ cuisines"),
        (len([d for d in districts.keys() if d in ['Beyoğlu', 'Fatih', 'Sultanahmet', 'Kadıköy', 'Sarıyer']]) >= 4, 
         "Major tourist districts covered"),
        (api_working, "API integration functional"),
        (beyoglu_count > 0 and seafood_count > 0 and luxury_count > 0, "Filtering logic working")
    ]
    
    passed = sum(1 for success, _ in success_criteria if success)
    total_criteria = len(success_criteria)
    
    for success, description in success_criteria:
        status = "✅" if success else "❌"
        print(f"   {status} {description}")
    
    print(f"\n{'='*60}")
    if passed == total_criteria:
        print(f"🎉 SUCCESS! All {total_criteria} criteria met!")
        print(f"📦 Istanbul restaurant dataset expansion COMPLETE")
        print(f"🔥 Ready for production use with {total_restaurants} restaurants")
    else:
        print(f"⚠️  Progress: {passed}/{total_criteria} criteria met")
        print(f"📝 Review items marked with ❌ above")
    
    return passed == total_criteria

if __name__ == "__main__":
    success = comprehensive_validation()
    sys.exit(0 if success else 1)
