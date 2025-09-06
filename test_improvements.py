#!/usr/bin/env python3
"""Test script for the improvements to tips diversity and follow-up restaurant queries."""

import requests
import time
import json

# Test configuration
BASE_URL = "http://localhost:8001"
TEST_SESSION = f"test_session_{int(time.time())}"

def test_query(query: str, description: str = ""):
    """Send a test query and print the response"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {description}")
    print(f"Query: '{query}'")
    print("-" * 60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/ai",
            json={"query": query, "session_id": TEST_SESSION},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", "No message")
            print(f"✅ Response:\n{message}")
            
            # Check for additional data
            if data.get("personalized"):
                print(f"🎯 Personalized: {data.get('personalized')}")
            if data.get("user_context"):
                print(f"👤 User Context: {data.get('user_context')}")
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    print("🚀 Testing Improvements: Diverse Tips & Follow-up Restaurant Queries")
    print(f"Using session ID: {TEST_SESSION}")
    
    # Test 1: Basic places query to see diverse tips
    test_query(
        "places to visit in sultanahmet",
        "Testing diverse tips generation for Sultanahmet"
    )
    
    # Wait a bit to allow context to be saved
    time.sleep(2)
    
    # Test 2: Follow-up restaurant query using context
    test_query(
        "give me also restaurants in sultanahmet",
        "Testing follow-up restaurant query with context"
    )
    
    # Test 3: Another places query in different location for tip diversity
    test_query(
        "places to visit in beyoglu", 
        "Testing diverse tips for Beyoğlu (different location)"
    )
    
    # Wait a bit
    time.sleep(2)
    
    # Test 4: Different follow-up restaurant pattern
    test_query(
        "show me more restaurants in beyoglu",
        "Testing different follow-up restaurant pattern"
    )
    
    # Test 5: Context-aware follow-up without explicit location
    test_query(
        "kadikoy",
        "Setting up context with Kadıköy"
    )
    
    time.sleep(2)
    
    # Test 6: Follow-up using context
    test_query(
        "restaurants also",
        "Testing context-aware restaurant follow-up without explicit location"
    )
    
    # Test 7: Test another area for tip diversity
    test_query(
        "attractions in galata",
        "Testing diverse tips for Galata area"
    )
    
    print(f"\n{'='*60}")
    print("🎉 Test completed! Check the responses above for:")
    print("✅ Diverse tips (different tips for different locations)")
    print("✅ Context-aware follow-up restaurant queries")
    print("✅ Proper conversational context handling")

if __name__ == "__main__":
    main()
