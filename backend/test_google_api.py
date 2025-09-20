#!/usr/bin/env python3
"""
Quick test script to check Google Places API functionality
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_google_places_api():
    """Test if Google Places API is working"""
    
    # Check if API key is loaded
    api_key = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
    
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API Key starts with: {api_key[:10]}..." if len(api_key) > 10 else api_key)
    
    if not api_key:
        print("❌ No API key found in environment variables")
        return False
    
    # Test a simple Places API call
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "key": api_key,
        "query": "restaurant in Fatih, Istanbul",
        "type": "restaurant"
    }
    
    try:
        print("\n🔍 Testing Google Places API...")
        response = requests.get(url, params=params, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "UNKNOWN")
            results_count = len(data.get("results", []))
            
            print(f"API Status: {status}")
            print(f"Results found: {results_count}")
            
            if status == "OK" and results_count > 0:
                print("✅ Google Places API is working!")
                
                # Show first result as example
                first_result = data["results"][0]
                print(f"Example restaurant: {first_result.get('name', 'Unknown')}")
                print(f"Rating: {first_result.get('rating', 'N/A')}")
                print(f"Address: {first_result.get('formatted_address', 'N/A')}")
                
                return True
            elif status == "ZERO_RESULTS":
                print("⚠️  API working but no results found for this query")
                return True
            else:
                print(f"❌ API returned status: {status}")
                if "error_message" in data:
                    print(f"Error message: {data['error_message']}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Google Places API Test")
    print("=" * 40)
    
    success = test_google_places_api()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed - API may not be working properly")
        print("\nPossible issues:")
        print("1. API key is invalid or expired")
        print("2. API key doesn't have Places API enabled")
        print("3. Network connectivity issues")
        print("4. API quota exceeded")
